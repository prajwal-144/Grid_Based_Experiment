"""Corrected fixed-SIS + FITS-PSF training using legacy raw mapping tensors.

This keeps the current mappings unchanged while applying the other diagnosed fixes:
row-normalized backward reconstruction, arc-balanced normalized losses, explicit
zero-baseline skill, delayed magnification regularization, and no circular
source-cycle loss.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import data
from differentiable_lensing import DifferentiableLensing
from physics_losses import build_fixed_sis_information_maps
from psf import apply_psf, build_psf_kernel
from sisr import SISR

EPS = 1e-8


def ensure_bchw(x):
    while x.ndim > 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    return x.unsqueeze(1) if x.ndim == 3 else x


def normalize_sparse_rows(mapping):
    """Average image contributors instead of summing their brightness."""
    mapping = mapping.coalesce()
    idx, values = mapping.indices(), mapping.values().clamp_min(0)
    sums = torch.zeros(mapping.shape[0], device=values.device, dtype=values.dtype)
    sums.scatter_add_(0, idx[0], values)
    normalized = values / sums[idx[0]].clamp_min(EPS)
    return torch.sparse_coo_tensor(idx, normalized, mapping.shape, device=mapping.device).coalesce()


def sparse_apply(image, mapping, output_side):
    b, c, h, w = image.shape
    flat = image.reshape(b * c, h * w).T
    out = torch.sparse.mm(mapping, flat)
    return out.T.contiguous().reshape(b, c, output_side, output_side)


def arc_balanced_weights(target, threshold_fraction, arc_boost, background_weight):
    peak = target.amax(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    arc = (target >= threshold_fraction * peak).to(target.dtype)
    return background_weight + arc_boost * arc


def normalized_wmse(pred, target, weight):
    weight = weight.expand_as(pred)
    return (weight * (pred - target).square()).sum() / weight.sum().clamp_min(EPS)


def laplacian(x):
    kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=x.device, dtype=x.dtype)
    weight = kernel.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, weight, padding=1, groups=x.shape[1])


def magnification_regularizer(source, information):
    """Suppress unsupported curvature only where SIS information is low."""
    info = F.interpolate(information, size=source.shape[-2:], mode="bilinear", align_corners=False)
    info = info.expand_as(source).clamp(0, 1)
    weight = (1.0 - info).square()
    return (weight * laplacian(source).square()).sum() / weight.sum().clamp_min(EPS)


def physics_scale(epoch, delay, ramp):
    if epoch < delay:
        return 0.0
    return 1.0 if ramp <= 0 else min(1.0, (epoch - delay + 1) / ramp)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", default="fixed_sis_corrected_legacy_mappings")
    p.add_argument("--output-dir", default="outputs_corrected")
    p.add_argument("--mapping-dir", default=".", help="directory containing the existing raw .pt mappings")
    p.add_argument("--resolution", type=float, required=True)
    p.add_argument("--theta-e", type=float, default=0.75)
    p.add_argument("--psf-path", required=True)
    p.add_argument("--psf-source-pixscale-arcsec", type=float, required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-shape", type=int, default=64)
    p.add_argument("--magnification", type=int, default=2)
    p.add_argument("--n-mag", type=int, default=1)
    p.add_argument("--residual-depth", type=int, default=3)
    p.add_argument("--latent-space-size", type=int, default=64)
    p.add_argument("--classes", nargs="+", default=["no_sub"], choices=["no_sub", "axion", "cdm"])
    p.add_argument("--train-samples-per-class", type=int, default=5000)
    p.add_argument("--val-samples-per-class", type=int, default=2000)
    p.add_argument("--dataset-fraction", type=float, default=1.0)
    p.add_argument("--arc-threshold-fraction", type=float, default=0.08)
    p.add_argument("--arc-boost", type=float, default=20.0)
    p.add_argument("--background-weight", type=float, default=0.05)
    p.add_argument("--source-loss-weight", type=float, default=0.2)
    p.add_argument("--mu-weight", type=float, default=0.001)
    p.add_argument("--physics-delay-epochs", type=int, default=20)
    p.add_argument("--physics-ramp-epochs", type=int, default=20)
    p.add_argument("--src-cons-weight", type=float, default=0.0)
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.src_cons_weight != 0:
        p.error("The old self-referential source consistency is disabled; use --src-cons-weight 0")
    args.effective_magnification = args.magnification ** args.n_mag
    args.target_shape = args.image_shape * args.effective_magnification
    args.target_resolution = args.resolution / args.effective_magnification
    args.device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    return args


def load_raw_mappings(args, device):
    root = Path(args.mapping_dir)
    names = {
        "backward": "sparse_grid_fracs_euclid_backward.pt",
        "to_log": "scatter_to_log_128.pt",
        "forward_from_log": "forward_from_log_128.pt",
        "from_log": "scatter_from_log_128.pt",
    }
    mappings = {key: torch.load(root / name, map_location=device).to(device).coalesce() for key, name in names.items()}
    expected_lr = args.image_shape ** 2
    expected_hr = args.target_shape ** 2
    if mappings["backward"].shape != (expected_lr, expected_lr):
        raise ValueError(f"Unexpected backward shape {mappings['backward'].shape}; expected {(expected_lr, expected_lr)}")
    if mappings["to_log"].shape[1] != expected_hr or mappings["from_log"].shape[0] != expected_hr:
        raise ValueError("Forward mapping dimensions do not match the requested HR shape")
    print("WARNING: legacy mappings have no geometry metadata; theta_E and pixel-scale consistency cannot be proven.")
    return mappings


def make_loader(args, split, shuffle):
    count = args.train_samples_per_class if split == "train" else args.val_samples_per_class
    sets = [data.LensingDataset(f"{split}/", [name], count) for name in args.classes]
    dataset = torch.utils.data.ConcatDataset(sets)
    keep = max(1, int(len(dataset) * args.dataset_fraction))
    dataset, _ = torch.utils.data.random_split(dataset, [keep, len(dataset) - keep], generator=torch.Generator().manual_seed(args.seed))
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=args.device == "cuda")


def run_epoch(model, loader, optimizer, training, epoch, args, lens, mappings, backward_avg, psf, info_hr):
    model.train(training)
    totals = {k: 0.0 for k in ["total", "image", "source", "mu", "raw_mse", "zero_mse", "skill"]}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for lr_image in tqdm(loader, desc=("train" if training else "val") + f" {epoch + 1}"):
            lr_image = ensure_bchw(lr_image.float()).to(args.device)
            source_lr = sparse_apply(lr_image, backward_avg, args.image_shape)
            source_hr = model(torch.cat([source_lr, lr_image], dim=1))
            intrinsic_hr = lens.cross_grid_fill(source_hr, [mappings["to_log"], mappings["forward_from_log"], mappings["from_log"]])
            pred_lr = F.interpolate(apply_psf(intrinsic_hr, psf), size=lr_image.shape[-2:], mode="area")
            source_down = F.interpolate(source_hr, size=source_lr.shape[-2:], mode="area")

            obs_weight = arc_balanced_weights(lr_image, args.arc_threshold_fraction, args.arc_boost, args.background_weight)
            image_loss = normalized_wmse(pred_lr, lr_image, obs_weight)
            source_weight = (source_lr > 0).to(source_lr.dtype) + 0.05
            source_loss = normalized_wmse(source_down, source_lr, source_weight)
            mu_loss = magnification_regularizer(source_hr, info_hr)
            ramp = physics_scale(epoch, args.physics_delay_epochs, args.physics_ramp_epochs)
            total = image_loss + args.source_loss_weight * source_loss + ramp * args.mu_weight * mu_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            raw_mse = F.mse_loss(pred_lr, lr_image)
            zero_mse = lr_image.square().mean()
            skill = 1.0 - raw_mse / zero_mse.clamp_min(EPS)
            for key, value in {"total": total, "image": image_loss, "source": source_loss, "mu": mu_loss, "raw_mse": raw_mse, "zero_mse": zero_mse, "skill": skill}.items():
                totals[key] += float(value.detach())
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_dir = Path(args.output_dir) / args.exp_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    mappings = load_raw_mappings(args, device)
    backward_avg = normalize_sparse_rows(mappings["backward"])
    lens = DifferentiableLensing(device=device, alpha=None, target_resolution=args.target_resolution, target_shape=args.target_shape).to(device)
    psf = build_psf_kernel("fits", 0.16, args.target_resolution, path=args.psf_path, source_pixscale_arcsec=args.psf_source_pixscale_arcsec, device=device)
    info = build_fixed_sis_information_maps(mappings["backward"], args.image_shape, args.resolution, args.theta_e)
    info_hr = F.interpolate(info.source_information_lr, size=(args.target_shape, args.target_shape), mode="bilinear", align_corners=False).to(device)
    print("source_information range", float(info_hr.min()), float(info_hr.max()), "critical_radius_lr_pixels", args.theta_e / args.resolution)

    model = SISR(args.magnification, args.n_mag, args.residual_depth, in_channels=2, latent_channel_count=args.latent_space_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = make_loader(args, "train", True)
    val_loader = make_loader(args, "val", False)
    history = {"train": [], "val": []}
    best = float("inf")

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, True, epoch, args, lens, mappings, backward_avg, psf, info_hr)
        val_metrics = run_epoch(model, val_loader, optimizer, False, epoch, args, lens, mappings, backward_avg, psf, info_hr)
        history["train"].append({"epoch": epoch + 1, **train_metrics})
        history["val"].append({"epoch": epoch + 1, **val_metrics})
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print("train", train_metrics); print("val", val_metrics)
        payload = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": vars(args), "history": history}
        torch.save(payload, run_dir / "checkpoints" / "latest.pt")
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            torch.save(payload, run_dir / "checkpoints" / "best.pt")


if __name__ == "__main__":
    main()
