"""Check legacy sparse mappings and compare an optional model with zero output."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import data
from differentiable_lensing import DifferentiableLensing
from psf import apply_psf, build_psf_kernel
from sisr import SISR


def row_normalize(mapping):
    mapping = mapping.coalesce()
    idx, values = mapping.indices(), mapping.values().clamp_min(0)
    sums = torch.zeros(mapping.shape[0], dtype=values.dtype, device=values.device)
    sums.scatter_add_(0, idx[0], values)
    values = values / sums[idx[0]].clamp_min(1e-8)
    return torch.sparse_coo_tensor(idx, values, mapping.shape, device=mapping.device).coalesce()


def sparse_apply(image, mapping, side):
    b, c, h, w = image.shape
    out = torch.sparse.mm(mapping, image.reshape(b * c, h * w).T)
    return out.T.reshape(b, c, side, side)


p = argparse.ArgumentParser()
p.add_argument("--mapping-dir", default=".")
p.add_argument("--resolution", type=float, required=True)
p.add_argument("--theta-e", type=float, default=0.75)
p.add_argument("--psf-path", required=True)
p.add_argument("--psf-source-pixscale-arcsec", type=float, required=True)
p.add_argument("--checkpoint", default=None)
p.add_argument("--val-index", type=int, default=0)
a = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = Path(a.mapping_dir)
backward = torch.load(root / "sparse_grid_fracs_euclid_backward.pt", map_location=device).to(device).coalesce()
forward = [
    torch.load(root / "scatter_to_log_128.pt", map_location=device).to(device).coalesce(),
    torch.load(root / "forward_from_log_128.pt", map_location=device).to(device).coalesce(),
    torch.load(root / "scatter_from_log_128.pt", map_location=device).to(device).coalesce(),
]
print("mapping shapes:", backward.shape, [m.shape for m in forward])
print("declared critical radius [LR pixels]:", a.theta_e / a.resolution)
print("WARNING: raw legacy mappings contain no metadata, so geometry agreement cannot be proven.")

backward_avg = row_normalize(backward)
ones_lr = torch.ones(1, 1, 64, 64, device=device)
source_ones = sparse_apply(ones_lr, backward_avg, 64)
print("constant backward min/mean/max:", float(source_ones.min()), float(source_ones.mean()), float(source_ones.max()))

if a.checkpoint:
    lens = DifferentiableLensing(device=device, alpha=None, target_resolution=a.resolution / 2, target_shape=128).to(device)
    psf = build_psf_kernel("fits", 0.16, a.resolution / 2, path=a.psf_path, source_pixscale_arcsec=a.psf_source_pixscale_arcsec, device=device)
    checkpoint = torch.load(a.checkpoint, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model = SISR(2, 1, 3, in_channels=2, latent_channel_count=64).to(device)
    model.load_state_dict(state)
    model.eval()
    lr = data.LensingDataset("val/", ["no_sub"], 2000)[a.val_index].unsqueeze(0).float().to(device)
    if lr.ndim == 3:
        lr = lr.unsqueeze(1)
    with torch.inference_mode():
        source_lr = sparse_apply(lr, backward_avg, 64)
        source_hr = model(torch.cat([source_lr, lr], dim=1))
        intrinsic = lens.cross_grid_fill(source_hr, forward)
        pred = F.interpolate(apply_psf(intrinsic, psf), size=(64, 64), mode="area")
    model_mse = F.mse_loss(pred, lr)
    zero_mse = lr.square().mean()
    print("model MSE:", float(model_mse))
    print("zero MSE:", float(zero_mse))
    print("skill over zero:", float(1.0 - model_mse / zero_mse.clamp_min(1e-8)))
