"""Evaluate a controlled-dataset checkpoint from train_analytic_sis.py.

This script is intentionally separate from test_HSC_FITS.ipynb. It rebuilds the
same analytic SIS renderer, analytic backprojector, PSF, and detector
integration used during training. It evaluates best.pt on the controlled
validation split, compares against zero-output and source-baseline references,
and saves a JSON report plus representative diagnostic figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from analytic_sis import AnalyticSISRenderer
from psf import apply_psf, build_psf_kernel
from sisr import SISR
from train_analytic_sis import ControlledDataset

EPS = 1e-8


def normalize_per_sample(image: torch.Tensor) -> torch.Tensor:
    """Normalize BCHW tensors to [0, 1] per sample for morphology comparison."""
    minimum = image.amin(dim=(-2, -1), keepdim=True)
    maximum = image.amax(dim=(-2, -1), keepdim=True)
    return (image - minimum) / (maximum - minimum).clamp_min(EPS)


def scalar_stats(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoints/best.pt")
    parser.add_argument("--controlled-root", default=None, help="Override controlled dataset root")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-visuals", type=int, default=6)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Conservative controlled-baseline acceptance thresholds. They can be
    # changed explicitly, but the defaults are printed and stored in the report.
    parser.add_argument("--min-mean-lr-skill", type=float, default=0.50)
    parser.add_argument("--max-negative-skill-fraction", type=float, default=0.10)
    parser.add_argument("--min-median-flux-ratio", type=float, default=0.80)
    parser.add_argument("--max-median-flux-ratio", type=float, default=1.20)
    parser.add_argument("--min-mean-source-skill", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device)
    checkpoint_path = Path(cli.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved = checkpoint.get("args", {})

    if saved.get("dataset_mode") != "controlled":
        raise ValueError(
            "evaluate_analytic_sis.py currently validates the Stage-B controlled "
            "baseline only. The checkpoint was not trained with --dataset-mode controlled."
        )

    controlled_root = cli.controlled_root or saved.get("controlled_root", "controlled_sis")
    resolution = float(saved["resolution"])
    image_shape = int(saved.get("image_shape", 64))
    scale_factor = int(saved.get("scale_factor", 2))
    hr_shape = image_shape * scale_factor
    hr_scale = resolution / scale_factor

    dataset = ControlledDataset(controlled_root, cli.split)
    if cli.max_samples is not None:
        count = min(int(cli.max_samples), len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(count))
    loader = DataLoader(
        dataset,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
    )

    model = SISR(
        magnification=scale_factor,
        n_mag=1,
        residual_depth=int(saved.get("residual_depth", 3)),
        in_channels=2,
        latent_channel_count=int(saved.get("latent_space_size", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    forward_renderer = AnalyticSISRenderer(
        hr_shape,
        hr_scale,
        hr_shape,
        hr_scale,
    ).to(device)
    backward_renderer = AnalyticSISRenderer(
        image_shape,
        resolution,
        image_shape,
        resolution,
    ).to(device)
    psf = build_psf_kernel(
        psf_type=saved.get("psf_type", "none"),
        fwhm_arcsec=float(saved.get("psf_fwhm", 0.16)),
        pixscale_arcsec=hr_scale,
        path=saved.get("psf_path"),
        source_pixscale_arcsec=saved.get("psf_source_pixscale"),
        device=device,
    )

    lr_skill: List[float] = []
    lr_mse: List[float] = []
    zero_mse: List[float] = []
    flux_ratio: List[float] = []
    source_shape_mse: List[float] = []
    source_baseline_shape_mse: List[float] = []
    source_skill: List[float] = []
    visuals = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate {cli.split}"):
            lr_image = batch["image"].float().to(device)
            theta_e = batch["theta_e"].float().to(device)
            source_truth = batch["source_truth"].float().to(device)
            if lr_image.ndim == 3:
                lr_image = lr_image.unsqueeze(1)
            if source_truth.ndim == 3:
                source_truth = source_truth.unsqueeze(1)

            source_lr, coverage = backward_renderer.backproject(lr_image, theta_e)
            source_hr = model(torch.cat([source_lr, lr_image], dim=1))
            intrinsic_hr = forward_renderer(source_hr, theta_e)
            convolved_hr = apply_psf(intrinsic_hr, psf)
            pred_lr = F.interpolate(
                convolved_hr,
                size=lr_image.shape[-2:],
                mode="area",
            )

            per_sample_lr_mse = (pred_lr - lr_image).square().mean(dim=(1, 2, 3))
            per_sample_zero_mse = lr_image.square().mean(dim=(1, 2, 3)).clamp_min(EPS)
            per_sample_skill = 1.0 - per_sample_lr_mse / per_sample_zero_mse
            per_sample_flux = pred_lr.sum(dim=(1, 2, 3)) / lr_image.sum(dim=(1, 2, 3)).clamp_min(EPS)

            truth_shape = normalize_per_sample(source_truth)
            pred_shape = normalize_per_sample(source_hr)
            baseline_hr = F.interpolate(
                source_lr,
                size=source_hr.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            baseline_shape = normalize_per_sample(baseline_hr)
            per_sample_source_mse = (pred_shape - truth_shape).square().mean(dim=(1, 2, 3))
            per_sample_baseline_mse = (baseline_shape - truth_shape).square().mean(dim=(1, 2, 3)).clamp_min(EPS)
            per_sample_source_skill = 1.0 - per_sample_source_mse / per_sample_baseline_mse

            lr_mse.extend(per_sample_lr_mse.cpu().tolist())
            zero_mse.extend(per_sample_zero_mse.cpu().tolist())
            lr_skill.extend(per_sample_skill.cpu().tolist())
            flux_ratio.extend(per_sample_flux.cpu().tolist())
            source_shape_mse.extend(per_sample_source_mse.cpu().tolist())
            source_baseline_shape_mse.extend(per_sample_baseline_mse.cpu().tolist())
            source_skill.extend(per_sample_source_skill.cpu().tolist())

            needed = cli.num_visuals - len(visuals)
            if needed > 0:
                for index in range(min(needed, lr_image.shape[0])):
                    visuals.append({
                        "lr": lr_image[index, 0].cpu(),
                        "source_lr": source_lr[index, 0].cpu(),
                        "source_pred": source_hr[index, 0].cpu(),
                        "source_truth": source_truth[index, 0].cpu(),
                        "intrinsic": intrinsic_hr[index, 0].cpu(),
                        "pred_lr": pred_lr[index, 0].cpu(),
                        "residual": (pred_lr[index, 0] - lr_image[index, 0]).cpu(),
                        "skill": float(per_sample_skill[index].cpu()),
                    })

    negative_fraction = float(np.mean(np.asarray(lr_skill) < 0.0))
    metrics = {
        "sample_count": len(lr_skill),
        "lr_mse": scalar_stats(lr_mse),
        "zero_mse": scalar_stats(zero_mse),
        "lr_skill_over_zero": scalar_stats(lr_skill),
        "negative_lr_skill_fraction": negative_fraction,
        "flux_ratio_prediction_over_target": scalar_stats(flux_ratio),
        "source_shape_mse": scalar_stats(source_shape_mse),
        "source_baseline_shape_mse": scalar_stats(source_baseline_shape_mse),
        "source_shape_skill_over_backprojection_baseline": scalar_stats(source_skill),
    }
    thresholds = {
        "min_mean_lr_skill": cli.min_mean_lr_skill,
        "max_negative_skill_fraction": cli.max_negative_skill_fraction,
        "median_flux_ratio_range": [
            cli.min_median_flux_ratio,
            cli.max_median_flux_ratio,
        ],
        "min_mean_source_skill": cli.min_mean_source_skill,
    }
    checks = {
        "lr_skill": metrics["lr_skill_over_zero"]["mean"] >= cli.min_mean_lr_skill,
        "negative_skill_fraction": negative_fraction <= cli.max_negative_skill_fraction,
        "flux_ratio": (
            cli.min_median_flux_ratio
            <= metrics["flux_ratio_prediction_over_target"]["median"]
            <= cli.max_median_flux_ratio
        ),
        "source_improves_over_baseline": (
            metrics["source_shape_skill_over_backprojection_baseline"]["mean"]
            >= cli.min_mean_source_skill
        ),
    }
    passed = all(checks.values())

    output_dir = Path(cli.output_dir) if cli.output_dir else checkpoint_path.parent.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset_root": str(controlled_root),
        "split": cli.split,
        "saved_training_args": saved,
        "metrics": metrics,
        "thresholds": thresholds,
        "checks": checks,
        "passed": passed,
        "interpretation": (
            "PASS: controlled analytic-SIS checkpoint beats the zero-image and "
            "backprojection source baselines with acceptable flux calibration."
            if passed
            else "FAIL: inspect failed checks and diagnostic panels before moving to PSF, magnification, or non-SIS data."
        ),
    }
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2))

    if visuals:
        rows = len(visuals)
        figure, axes = plt.subplots(rows, 7, figsize=(21, 3.2 * rows), squeeze=False)
        titles = [
            "LR observation",
            "LR backprojection",
            "Predicted HR source",
            "True HR source",
            "Intrinsic HR lens",
            "Re-degraded LR",
            "LR residual",
        ]
        for row, item in enumerate(visuals):
            panels = [
                item["lr"],
                item["source_lr"],
                item["source_pred"],
                item["source_truth"],
                item["intrinsic"],
                item["pred_lr"],
                item["residual"],
            ]
            for column, panel in enumerate(panels):
                if column == 6:
                    limit = max(abs(float(panel.min())), abs(float(panel.max())), EPS)
                    axes[row, column].imshow(panel, cmap="gray", vmin=-limit, vmax=limit)
                else:
                    axes[row, column].imshow(panel, cmap="gray", vmin=0.0)
                axes[row, column].set_title(titles[column] if row == 0 else "")
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
            axes[row, 0].set_ylabel(f"skill={item['skill']:.3f}")
        figure.tight_layout()
        figure.savefig(output_dir / "evaluation_samples.png", dpi=160)
        plt.close(figure)

    print(json.dumps(report, indent=2))
    print(f"Saved evaluation to {output_dir}")
    print("RESULT:", "PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
