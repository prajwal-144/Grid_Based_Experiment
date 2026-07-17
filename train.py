"""Stage D/E training with a fixed SIS, FITS PSF, and two physics losses.

This script intentionally does not replace ``train.py``. It keeps the current
fixed sparse SIS forward/backward operators and adds a magnification-adaptive
soft source grid, L_mu, and source-plane surface-brightness consistency.

The fixed SIS Einstein radius passed with ``--theta-e`` must match the value used
to precompute the sparse mapping tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import data
from differentiable_lensing import DifferentiableLensing
from metrics import (
    MetricTracker,
    flux_conservation_error,
    lr_redegradation_error,
    source_plane_reconstruction_consistency,
)
from physics_losses import (
    SourcePlaneIntensityConsistency,
    build_fixed_sis_information_maps,
    magnification_adaptive_loss,
    magnification_adaptive_source_grid,
)
from psf import apply_psf, build_psf_kernel
from sisr import SISR


def strtobool(x: str) -> bool:
    return x.lower().strip() in {"true", "1", "yes", "y"}


def ensure_bchw(image: torch.Tensor) -> torch.Tensor:
    """Normalize dataset output to [B,C,H,W]."""
    while image.ndim > 4 and image.shape[1] == 1:
        image = image.squeeze(1)
    if image.ndim == 3:
        image = image.unsqueeze(1)
    if image.ndim != 4:
        raise ValueError(f"Expected dataset batch convertible to BCHW; got {tuple(image.shape)}")
    return image


def wmse_loss(y1: torch.Tensor, y2: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mean((y1 - y2).square() * weight)


def interpolate_image(
    x: torch.Tensor,
    *,
    scale_factor: Optional[float] = None,
    size: Optional[tuple[int, int]] = None,
    mode: str,
) -> torch.Tensor:
    kwargs = {"mode": mode}
    if scale_factor is not None:
        kwargs["scale_factor"] = scale_factor
    elif size is not None:
        kwargs["size"] = size
    else:
        raise ValueError("Provide scale_factor or size")
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return F.interpolate(x, **kwargs)


def physics_ramp(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch + 1) / float(warmup_epochs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-SIS magnification-adaptive and source-consistent SR"
    )
    parser.add_argument("--exp-name", type=str, default="fixed_sis_mag_srccons")
    parser.add_argument("--output-dir", type=str, default="outputs_mag")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--log-train", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)

    parser.add_argument("--train-samples-per-class", type=int, default=5000)
    parser.add_argument("--val-samples-per-class", type=int, default=2000)
    parser.add_argument("--dataset-fraction", type=float, default=0.34)

    parser.add_argument("--resolution", type=float, required=True, help="LR arcsec/pixel")
    parser.add_argument("--magnification", type=int, default=2)
    parser.add_argument("--n-mag", type=int, default=1)
    parser.add_argument("--residual-depth", type=int, default=3)
    parser.add_argument("--in-channels", type=int, default=2)
    parser.add_argument("--latent-space-size", type=int, default=64)
    parser.add_argument("--image-shape", type=int, default=64)
    parser.add_argument(
        "--theta-e",
        type=float,
        default=0.75,
        help="fixed SIS Einstein radius used to create the sparse mappings",
    )
    parser.add_argument("--vdl-weight", type=float, default=0.0)

    parser.add_argument("--mu-weight", type=float, default=0.02)
    parser.add_argument("--mu-clip", type=float, default=20.0)
    parser.add_argument("--adaptive-medium-factor", type=int, default=2)
    parser.add_argument("--adaptive-coarse-factor", type=int, default=4)

    parser.add_argument("--src-cons-weight", type=float, default=0.05)
    parser.add_argument("--src-cons-variance-weight", type=float, default=1.0)
    parser.add_argument("--src-cons-min-neff", type=float, default=1.5)
    parser.add_argument("--physics-warmup-epochs", type=int, default=10)

    parser.add_argument(
        "--psf-type",
        type=str,
        default="fits",
        choices=["none", "gaussian", "moffat", "empirical", "fits"],
    )
    parser.add_argument("--psf-fwhm-arcsec", type=float, default=0.16)
    parser.add_argument("--psf-beta", type=float, default=4.765)
    parser.add_argument("--psf-kernel-size", type=int, default=None)
    parser.add_argument("--psf-path", type=str, default=None)
    parser.add_argument("--psf-ellipticity-q", type=float, default=1.0)
    parser.add_argument("--psf-angle-deg", type=float, default=0.0)
    parser.add_argument("--psf-fits-hdu", type=int, default=0)
    parser.add_argument("--psf-fits-extname", type=str, default=None)
    parser.add_argument("--psf-fits-crop-size", type=int, default=None)
    parser.add_argument("--psf-source-pixscale-arcsec", type=float, default=None)
    parser.add_argument(
        "--downsample-mode",
        type=str,
        default="area",
        choices=["nearest", "bilinear", "bicubic", "area"],
    )
    parser.add_argument(
        "--upsample-mode",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic"],
    )

    args = parser.parse_args()
    if not (0.0 < args.dataset_fraction <= 1.0):
        parser.error("--dataset-fraction must be in (0,1]")
    if args.adaptive_medium_factor < 1 or args.adaptive_coarse_factor < args.adaptive_medium_factor:
        parser.error("Require 1 <= medium factor <= coarse factor")
    if args.psf_type in {"fits", "empirical"} and not args.psf_path:
        parser.error("--psf-path is required for FITS/empirical PSF training")

    args.effective_magnification = int(args.magnification ** args.n_mag)
    args.target_shape = args.image_shape * args.effective_magnification
    args.target_resolution = args.resolution / args.effective_magnification
    args.device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    return args


def make_dataloaders(args: argparse.Namespace):
    train_sets = [
        data.LensingDataset("train/", [name], args.train_samples_per_class)
        for name in ("no_sub", "axion", "cdm")
    ]
    val_sets = [
        data.LensingDataset("val/", [name], args.val_samples_per_class)
        for name in ("no_sub", "axion", "cdm")
    ]
    train_all = torch.utils.data.ConcatDataset(train_sets)
    val_all = torch.utils.data.ConcatDataset(val_sets)

    generator = torch.Generator().manual_seed(args.seed)
    train_size = max(1, int(round(args.dataset_fraction * len(train_all))))
    val_size = max(1, int(round(args.dataset_fraction * len(val_all))))
    train_subset, _ = torch.utils.data.random_split(
        train_all,
        [train_size, len(train_all) - train_size],
        generator=generator,
    )
    val_subset, _ = torch.utils.data.random_split(
        val_all,
        [val_size, len(val_all) - val_size],
        generator=generator,
    )

    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    return (
        torch.utils.data.DataLoader(train_subset, shuffle=True, **common),
        torch.utils.data.DataLoader(val_subset, shuffle=False, **common),
    )


def build_base_metrics(
    *,
    total_loss: torch.Tensor,
    image_loss: torch.Tensor,
    source_loss: torch.Tensor,
    mu_loss: torch.Tensor,
    src_cons_loss: torch.Tensor,
    src_diag: Dict[str, torch.Tensor],
    ramp: float,
    downsampled_image: torch.Tensor,
    lr_image: torch.Tensor,
    downsampled_source: torch.Tensor,
    reconstructed_source: torch.Tensor,
    source_convergence_map: torch.Tensor,
    image_convergence_map: torch.Tensor,
) -> Dict[str, torch.Tensor | float]:
    return {
        "loss_total": total_loss,
        "loss_image": image_loss,
        "loss_source": source_loss,
        "loss_mu": mu_loss,
        "loss_src_cons": src_cons_loss,
        "loss_src_cons_cycle": src_diag["cycle_loss"],
        "loss_src_cons_variance": src_diag["variance_loss"],
        "physics_ramp": ramp,
        "metric_multi_image_fraction": src_diag["multi_image_fraction"],
        "metric_lr_redegradation": lr_redegradation_error(
            downsampled_image,
            lr_image,
            weight=source_convergence_map,
        ),
        "metric_source_consistency": source_plane_reconstruction_consistency(
            downsampled_source,
            reconstructed_source,
            weight=image_convergence_map,
        ),
        "metric_flux_lr": flux_conservation_error(lr_image, downsampled_image),
        "metric_flux_source": flux_conservation_error(reconstructed_source, downsampled_source),
    }


def run_epoch(
    *,
    epoch: int,
    training: bool,
    dataloader,
    model: SISR,
    optimizer: torch.optim.Optimizer,
    lensing_module: DifferentiableLensing,
    forward_mappings,
    backward_mapping: torch.Tensor,
    psf_kernel: Optional[torch.Tensor],
    source_information_hr: torch.Tensor,
    source_consistency: SourcePlaneIntensityConsistency,
    source_convergence_map: torch.Tensor,
    image_convergence_map: torch.Tensor,
    args: argparse.Namespace,
) -> MetricTracker:
    model.train(training)
    tracker = MetricTracker()
    ramp = physics_ramp(epoch, args.physics_warmup_epochs)
    description = ("Training" if training else "Validation") + f" epoch {epoch + 1}/{args.epochs}"

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for lr_image in tqdm(dataloader, desc=description):
            lr_image = ensure_bchw(lr_image.float()).to(args.device, non_blocking=True)

            reconstructed_source = lensing_module.cross_grid_fill(lr_image, [backward_mapping])
            model_feed = torch.cat([reconstructed_source, lr_image], dim=1)
            fine_source = model(model_feed)

            adaptive_source_full, _ = magnification_adaptive_source_grid(
                fine_source,
                source_information_hr,
                medium_factor=args.adaptive_medium_factor,
                coarse_factor=args.adaptive_coarse_factor,
            )
            adaptive_source = torch.lerp(fine_source, adaptive_source_full, ramp)
            mu_loss = magnification_adaptive_loss(
                fine_source,
                adaptive_source_full,
                source_information_hr,
            )

            intrinsic_hr_image = lensing_module.cross_grid_fill(
                adaptive_source,
                forward_mappings,
            )
            convolved_hr_image = apply_psf(intrinsic_hr_image, psf_kernel)

            downsampled_image = interpolate_image(
                convolved_hr_image,
                scale_factor=1.0 / args.effective_magnification,
                mode=args.downsample_mode,
            )
            downsampled_intrinsic_image = interpolate_image(
                intrinsic_hr_image,
                scale_factor=1.0 / args.effective_magnification,
                mode=args.downsample_mode,
            )
            downsampled_source = interpolate_image(
                adaptive_source,
                scale_factor=1.0 / args.effective_magnification,
                mode=args.downsample_mode,
            )

            image_loss = wmse_loss(downsampled_image, lr_image, source_convergence_map)
            source_loss = wmse_loss(downsampled_source, reconstructed_source, image_convergence_map)

            src_cons_loss, src_diag = source_consistency(
                downsampled_intrinsic_image,
                downsampled_source,
            )

            interpolated_image = interpolate_image(
                lr_image,
                scale_factor=args.effective_magnification,
                mode=args.upsample_mode,
            )
            interpolated_vd = lensing_module.compute_variation_density(interpolated_image)
            predicted_vd = lensing_module.compute_variation_density(convolved_hr_image)
            vdl_loss = F.mse_loss(interpolated_vd, predicted_vd)

            total_loss = (
                image_loss
                + source_loss
                + args.vdl_weight * vdl_loss
                + ramp * args.mu_weight * mu_loss
                + ramp * args.src_cons_weight * src_cons_loss
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

            tracker.update(
                **build_base_metrics(
                    total_loss=total_loss,
                    image_loss=image_loss,
                    source_loss=source_loss,
                    mu_loss=mu_loss,
                    src_cons_loss=src_cons_loss,
                    src_diag=src_diag,
                    ramp=ramp,
                    downsampled_image=downsampled_image,
                    lr_image=lr_image,
                    downsampled_source=downsampled_source,
                    reconstructed_source=reconstructed_source,
                    source_convergence_map=source_convergence_map,
                    image_convergence_map=image_convergence_map,
                ),
                loss_vdl=vdl_loss,
            )
    return tracker


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: SISR,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    history: Dict,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "history": history,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    print(f"[SYS] Device: {args.device}")
    print(
        "[PHYS] Fixed SIS: theta_E=%.6f arcsec; this must match the precomputed sparse grids."
        % args.theta_e
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = not args.torch_deterministic

    run_dir = Path(args.output_dir) / args.exp_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "args.json", vars(args))

    train_loader, val_loader = make_dataloaders(args)

    model = SISR(
        magnification=args.magnification,
        n_mag=args.n_mag,
        residual_depth=args.residual_depth,
        in_channels=args.in_channels,
        latent_channel_count=args.latent_space_size,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lensing_module = DifferentiableLensing(
        device=torch.device(args.device),
        alpha=None,
        target_resolution=args.target_resolution,
        target_shape=args.target_shape,
    ).to(args.device)

    forward_mappings = [
        torch.load("scatter_to_log_128.pt", map_location=args.device).to(args.device),
        torch.load("forward_from_log_128.pt", map_location=args.device).to(args.device),
        torch.load("scatter_from_log_128.pt", map_location=args.device).to(args.device),
    ]
    backward_mapping = torch.load(
        "sparse_grid_fracs_euclid_backward.pt",
        map_location=args.device,
    ).to(args.device)
    source_convergence_map = torch.load(
        "source_convergence_map.pt",
        map_location=args.device,
    ).to(args.device)
    image_convergence_map = torch.load(
        "image_convergence_map.pt",
        map_location=args.device,
    ).to(args.device)

    psf_kernel = build_psf_kernel(
        psf_type=args.psf_type,
        fwhm_arcsec=args.psf_fwhm_arcsec,
        pixscale_arcsec=args.target_resolution,
        beta=args.psf_beta,
        kernel_size=args.psf_kernel_size,
        path=args.psf_path,
        ellipticity_q=args.psf_ellipticity_q,
        angle_deg=args.psf_angle_deg,
        fits_hdu=args.psf_fits_hdu,
        fits_extname=args.psf_fits_extname,
        fits_crop_size=args.psf_fits_crop_size,
        source_pixscale_arcsec=args.psf_source_pixscale_arcsec,
        device=args.device,
    )
    if psf_kernel is not None:
        print(
            "[PSF] type=%s shape=%s sum=%.6f HR_pixel_scale=%.6f arcsec/pixel"
            % (
                args.psf_type,
                tuple(psf_kernel.shape),
                psf_kernel.sum().item(),
                args.target_resolution,
            )
        )

    info_maps = build_fixed_sis_information_maps(
        backward_mapping=backward_mapping,
        image_shape=args.image_shape,
        pixel_scale_arcsec=args.resolution,
        theta_e_arcsec=args.theta_e,
        mu_clip=args.mu_clip,
    )
    source_information_hr = interpolate_image(
        info_maps.source_information_lr,
        size=(args.target_shape, args.target_shape),
        mode="bilinear",
    ).clamp(0.0, 1.0)

    source_consistency = SourcePlaneIntensityConsistency(
        backward_mapping,
        source_shape=(args.image_shape, args.image_shape),
        min_effective_contributors=args.src_cons_min_neff,
        variance_weight=args.src_cons_variance_weight,
        normalize_by_source_power=True,
    ).to(args.device)

    maps_path = run_dir / "physics_maps.pt"
    torch.save(
        {
            "theta_e_arcsec": args.theta_e,
            "mu_clip": args.mu_clip,
            "signed_magnification_image": info_maps.signed_magnification_image.detach().cpu(),
            "absolute_magnification_image": info_maps.absolute_magnification_image.detach().cpu(),
            "image_information": info_maps.image_information.detach().cpu(),
            "source_information_lr": info_maps.source_information_lr.detach().cpu(),
            "source_information_hr": source_information_hr.detach().cpu(),
            "source_coverage_lr": info_maps.source_coverage_lr.detach().cpu(),
            "source_effective_contributors_lr": info_maps.source_effective_contributors_lr.detach().cpu(),
        },
        maps_path,
    )
    print(
        "[PHYS] Source information map: min=%.4f mean=%.4f max=%.4f"
        % (
            source_information_hr.min().item(),
            source_information_hr.mean().item(),
            source_information_hr.max().item(),
        )
    )

    writer = SummaryWriter(f"runs/{args.exp_name}") if args.log_train else None
    if writer is not None:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
        )
        writer.add_image("physics/image_information", info_maps.image_information[0], 0)
        writer.add_image("physics/source_information_hr", source_information_hr[0], 0)

    history: Dict[str, list] = {"train": [], "val": []}
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = checkpoint.get("history", history)
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        print(f"[SYS] Resumed from {args.resume} at epoch {start_epoch + 1}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            epoch=epoch,
            training=True,
            dataloader=train_loader,
            model=model,
            optimizer=optimizer,
            lensing_module=lensing_module,
            forward_mappings=forward_mappings,
            backward_mapping=backward_mapping,
            psf_kernel=psf_kernel,
            source_information_hr=source_information_hr,
            source_consistency=source_consistency,
            source_convergence_map=source_convergence_map,
            image_convergence_map=image_convergence_map,
            args=args,
        )
        val_metrics = run_epoch(
            epoch=epoch,
            training=False,
            dataloader=val_loader,
            model=model,
            optimizer=optimizer,
            lensing_module=lensing_module,
            forward_mappings=forward_mappings,
            backward_mapping=backward_mapping,
            psf_kernel=psf_kernel,
            source_information_hr=source_information_hr,
            source_consistency=source_consistency,
            source_convergence_map=source_convergence_map,
            image_convergence_map=image_convergence_map,
            args=args,
        )

        train_values = train_metrics.as_dict()
        val_values = val_metrics.as_dict()
        history["train"].append({"epoch": epoch + 1, **train_values})
        history["val"].append({"epoch": epoch + 1, **val_values})
        save_json(run_dir / "history.json", history)

        print(f"[SYS] Train epoch {epoch + 1}: {train_metrics.summary()}")
        print(f"[SYS] Validation epoch {epoch + 1}: {val_metrics.summary()}")

        if writer is not None:
            for key, value in train_values.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_values.items():
                writer.add_scalar(f"val/{key}", value, epoch)

        current_val = val_values["loss_total"]
        if current_val < best_val_loss:
            best_val_loss = current_val
            save_checkpoint(
                checkpoint_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                args=args,
                history=history,
                best_val_loss=best_val_loss,
            )
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                args=args,
                history=history,
                best_val_loss=best_val_loss,
            )

    save_checkpoint(
        checkpoint_dir / "final.pt",
        epoch=args.epochs - 1,
        model=model,
        optimizer=optimizer,
        args=args,
        history=history,
        best_val_loss=best_val_loss,
    )
    torch.save(model.state_dict(), run_dir / f"{args.exp_name}_weights.pt")
    if writer is not None:
        writer.close()
    print(f"[SYS] Outputs written to {run_dir}")


if __name__ == "__main__":
    main()
