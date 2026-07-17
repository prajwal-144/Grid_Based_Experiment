"""Verify fixed sparse SIS geometry and establish the zero-output baseline.

This is diagnostic code: it does not train or modify a model.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

import data
from differentiable_lensing import DifferentiableLensing
from psf import apply_psf, build_psf_kernel
from sisr import SISR


def as_matrix(obj):
    return obj["matrix"] if isinstance(obj, dict) and "matrix" in obj else obj


def stats(name, x):
    x = x.detach().float().cpu()
    print(f"{name}: shape={tuple(x.shape)} min={x.min():.6g} mean={x.mean():.6g} max={x.max():.6g} sum={x.sum():.6g}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", type=float, required=True)
    p.add_argument("--theta-e", type=float, required=True)
    p.add_argument("--image-shape", type=int, default=64)
    p.add_argument("--magnification", type=int, default=2)
    p.add_argument("--checkpoint", default=None, help="optional Stage 1-2 raw state_dict")
    p.add_argument("--psf-path", required=True)
    p.add_argument("--psf-source-pixscale-arcsec", type=float, required=True)
    p.add_argument("--val-class", default="no_sub", choices=["no_sub", "axion", "cdm"])
    p.add_argument("--val-index", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_shape = args.image_shape * args.magnification
    target_resolution = args.resolution / args.magnification
    lens = DifferentiableLensing(device=device, alpha=None, target_resolution=target_resolution, target_shape=target_shape).to(device)

    backward = as_matrix(torch.load("sparse_grid_fracs_euclid_backward.pt", map_location=device)).to(device)
    forward = [
        as_matrix(torch.load("scatter_to_log_128.pt", map_location=device)).to(device),
        as_matrix(torch.load("forward_from_log_128.pt", map_location=device)).to(device),
        as_matrix(torch.load("scatter_from_log_128.pt", map_location=device)).to(device),
    ]
    print("Matrix shapes:", tuple(backward.shape), [tuple(m.shape) for m in forward])
    print("Declared critical radius [LR pixels]:", args.theta_e / args.resolution)

    ones_lr = torch.ones(1, 1, args.image_shape, args.image_shape, device=device)
    source_from_ones = lens.cross_grid_fill(ones_lr, [backward])
    stats("backward(ones_LR)", source_from_ones)

    ones_hr = torch.ones(1, 1, target_shape, target_shape, device=device)
    image_from_ones = lens.cross_grid_fill(ones_hr, forward)
    stats("forward(ones_HR_source)", image_from_ones)
    roundtrip = lens.cross_grid_fill(
        F.interpolate(image_from_ones, size=(args.image_shape, args.image_shape), mode="area"),
        [backward],
    )
    stats("backward(downsample(forward(ones)))", roundtrip)

    dataset = data.LensingDataset("val/", [args.val_class], max(args.val_index + 1, 1))
    lr = dataset[args.val_index].unsqueeze(0).float().to(device)
    if lr.ndim == 5:
        lr = lr.squeeze(1)
    zero_mse = F.mse_loss(torch.zeros_like(lr), lr)
    print(f"zero_baseline_mse={zero_mse.item():.8g}")

    if args.checkpoint:
        model = SISR(2, 1, 3, in_channels=2, latent_channel_count=64).to(device)
        state = torch.load(args.checkpoint, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        psf = build_psf_kernel(
            psf_type="fits",
            fwhm_arcsec=0.16,
            pixscale_arcsec=target_resolution,
            path=args.psf_path,
            source_pixscale_arcsec=args.psf_source_pixscale_arcsec,
            device=device,
        )
        with torch.no_grad():
            source_lr = lens.cross_grid_fill(lr, [backward])
            source_hr = model(torch.cat([source_lr, lr], dim=1))
            pred_hr = apply_psf(lens.cross_grid_fill(source_hr, forward), psf)
            pred_lr = F.interpolate(pred_hr, size=lr.shape[-2:], mode="area")
        model_mse = F.mse_loss(pred_lr, lr)
        skill = 1.0 - model_mse / zero_mse.clamp_min(1e-12)
        print(f"model_mse={model_mse.item():.8g} skill_over_zero={skill.item():.8g}")
        stats("LR observation", lr)
        stats("LR prediction", pred_lr)


if __name__ == "__main__":
    main()
