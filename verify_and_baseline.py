"""Check mapping geometry, constant-image transport, and optional model skill over zero."""
import argparse
import torch
import torch.nn.functional as F
import data
from differentiable_lensing import DifferentiableLensing
from mapping_io import load_mapping_bundle
from psf import apply_psf, build_psf_kernel
from sisr import SISR

p = argparse.ArgumentParser()
p.add_argument("--mapping-dir", required=True)
p.add_argument("--resolution", type=float, required=True)
p.add_argument("--theta-e", type=float, required=True)
p.add_argument("--image-shape", type=int, default=64)
p.add_argument("--magnification", type=int, default=2)
p.add_argument("--checkpoint", default=None)
p.add_argument("--psf-path", required=True)
p.add_argument("--psf-source-pixscale-arcsec", type=float, required=True)
p.add_argument("--val-class", default="no_sub", choices=["no_sub", "axion", "cdm"])
p.add_argument("--val-index", type=int, default=0)
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_shape = args.image_shape * args.magnification
target_resolution = args.resolution / args.magnification
expected = {
    "lens_model": "SIS", "theta_e_arcsec": args.theta_e,
    "lr_pixel_scale_arcsec": args.resolution,
    "hr_pixel_scale_arcsec": target_resolution,
    "image_shape": args.image_shape, "target_shape": target_shape,
}
paths = {
    "backward": "sparse_grid_fracs_euclid_backward_bundle.pt",
    "to_log": "scatter_to_log_128_bundle.pt",
    "forward_from_log": "forward_from_log_128_bundle.pt",
    "from_log": "scatter_from_log_128_bundle.pt",
}
matrices = {}
for role, filename in paths.items():
    matrix, meta = load_mapping_bundle(f"{args.mapping_dir}/{filename}", device=device, expected={**expected, "mapping_role": role})
    matrices[role] = matrix
    print(role, tuple(matrix.shape), meta)

lens = DifferentiableLensing(device=device, alpha=None, target_resolution=target_resolution, target_shape=target_shape).to(device)
forward = [matrices["to_log"], matrices["forward_from_log"], matrices["from_log"]]

def stats(name, x):
    x = x.detach().float().cpu()
    print(f"{name}: shape={tuple(x.shape)} min={x.min():.6g} mean={x.mean():.6g} max={x.max():.6g} sum={x.sum():.6g}")

ones_lr = torch.ones(1, 1, args.image_shape, args.image_shape, device=device)
ones_hr = torch.ones(1, 1, target_shape, target_shape, device=device)
stats("backward(ones_LR)", lens.cross_grid_fill(ones_lr, [matrices["backward"]]))
stats("forward(ones_HR)", lens.cross_grid_fill(ones_hr, forward))
print("critical_radius_lr_pixels", args.theta_e / args.resolution)

dataset = data.LensingDataset("val/", [args.val_class], args.val_index + 1)
lr = dataset[args.val_index].unsqueeze(0).float().to(device)
if lr.ndim == 5:
    lr = lr.squeeze(1)
zero_mse = F.mse_loss(torch.zeros_like(lr), lr)
print("zero_baseline_mse", zero_mse.item())

if args.checkpoint:
    model = SISR(2, 1, 3, in_channels=2, latent_channel_count=64).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state)
    model.eval()
    psf = build_psf_kernel("fits", 0.16, target_resolution, path=args.psf_path, source_pixscale_arcsec=args.psf_source_pixscale_arcsec, device=device)
    with torch.no_grad():
        source_lr = lens.cross_grid_fill(lr, [matrices["backward"]])
        source_hr = model(torch.cat([source_lr, lr], dim=1))
        pred_hr = apply_psf(lens.cross_grid_fill(source_hr, forward), psf)
        pred_lr = F.interpolate(pred_hr, size=lr.shape[-2:], mode="area")
    model_mse = F.mse_loss(pred_lr, lr)
    print("model_mse", model_mse.item(), "skill_over_zero", (1.0 - model_mse / zero_mse.clamp_min(1e-12)).item())
