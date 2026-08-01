"""Diagnostic suite for fixed-SIS magnification experiments.

Run this before any new training. It checks whether the sparse mappings, SIS
magnification map, PSF sampling, and optional trained checkpoint are mutually
consistent. The script is intentionally read-only: it never modifies matrices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data
from differentiable_lensing import DifferentiableLensing
from physics_losses import build_fixed_sis_information_maps
from psf import apply_psf, build_psf_kernel
from sisr import SISR

EPS = 1e-8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mapping-dir", required=True)
    p.add_argument("--resolution", type=float, required=True, help="LR image pixel scale in arcsec/pixel")
    p.add_argument("--theta-e", type=float, required=True, help="SIS Einstein radius in arcsec")
    p.add_argument("--image-shape", type=int, default=64)
    p.add_argument("--magnification", type=int, default=2)
    p.add_argument("--n-mag", type=int, default=1)
    p.add_argument("--log-c", type=float, default=None, help="Optional log-grid parameter used to generate mappings")
    p.add_argument("--coordinate-units", choices=["arcsec", "pixels", "unknown"], default="unknown")
    p.add_argument("--psf-path", required=True)
    p.add_argument("--psf-source-pixscale-arcsec", type=float, required=True,
                   help="Native sampling of the FITS PSF, not the LR image scale")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--val-dir", default="val/")
    p.add_argument("--val-class", default="no_sub")
    p.add_argument("--val-index", type=int, default=5)
    p.add_argument("--val-samples", type=int, default=100)
    p.add_argument("--output-dir", default="diagnostics_output")
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def sha256_sparse(matrix: torch.Tensor) -> str:
    m = matrix.coalesce().cpu()
    h = hashlib.sha256()
    h.update(m.indices().numpy().tobytes())
    h.update(m.values().numpy().tobytes())
    h.update(np.asarray(m.shape, dtype=np.int64).tobytes())
    return h.hexdigest()


def sparse_stats(matrix: torch.Tensor) -> dict:
    m = matrix.coalesce()
    idx, val = m.indices(), m.values()
    row = torch.zeros(m.shape[0], dtype=val.dtype, device=val.device)
    col = torch.zeros(m.shape[1], dtype=val.dtype, device=val.device)
    row.scatter_add_(0, idx[0], val)
    col.scatter_add_(0, idx[1], val)
    q = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], device=val.device)
    return {
        "shape": list(m.shape),
        "nnz": int(m._nnz()),
        "sha256": sha256_sparse(m),
        "value_quantiles": torch.quantile(val.float(), q).cpu().tolist(),
        "row_sum_quantiles": torch.quantile(row.float(), q).cpu().tolist(),
        "column_sum_quantiles": torch.quantile(col.float(), q).cpu().tolist(),
        "negative_values": int((val < 0).sum()),
        "nonfinite_values": int((~torch.isfinite(val)).sum()),
        "zero_rows": int((row.abs() < EPS).sum()),
        "zero_columns": int((col.abs() < EPS).sum()),
    }


def normalize_sparse_rows(mapping: torch.Tensor) -> torch.Tensor:
    m = mapping.coalesce()
    idx, val = m.indices(), m.values().clamp_min(0)
    sums = torch.zeros(m.shape[0], device=val.device, dtype=val.dtype)
    sums.scatter_add_(0, idx[0], val)
    return torch.sparse_coo_tensor(
        idx, val / sums[idx[0]].clamp_min(EPS), m.shape, device=m.device
    ).coalesce()


def apply_sparse(image: torch.Tensor, mapping: torch.Tensor, side: int) -> torch.Tensor:
    b, c, h, w = image.shape
    flat = image.reshape(b * c, h * w).T
    out = torch.sparse.mm(mapping, flat)
    return out.T.contiguous().reshape(b, c, side, side)


def radial_profile(image: torch.Tensor) -> torch.Tensor:
    x = image.detach().float().cpu()
    if x.ndim == 4:
        x = x[0, 0]
    h, w = x.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    cy, cx = (h - 1) / 2, (w - 1) / 2
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).long()
    max_r = int(rr.max())
    sums = torch.zeros(max_r + 1)
    counts = torch.zeros(max_r + 1)
    sums.scatter_add_(0, rr.reshape(-1), x.reshape(-1))
    counts.scatter_add_(0, rr.reshape(-1), torch.ones_like(x).reshape(-1))
    return sums / counts.clamp_min(1)


def peak_radii(profile: torch.Tensor, min_separation: int = 2) -> list[int]:
    if profile.numel() < 3:
        return []
    peaks = []
    for i in range(1, profile.numel() - 1):
        if profile[i] > profile[i - 1] and profile[i] >= profile[i + 1]:
            if not peaks or i - peaks[-1] >= min_separation:
                peaks.append(i)
    return sorted(peaks, key=lambda i: float(profile[i]), reverse=True)[:3]


def synthetic_source(side: int, sigma: float, ring_radius: float | None = None) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    cy = cx = (side - 1) / 2
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if ring_radius is None:
        image = torch.exp(-0.5 * (rr / sigma) ** 2)
    else:
        image = torch.exp(-0.5 * ((rr - ring_radius) / sigma) ** 2)
    return image[None, None]


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).square().mean() / b.square().mean().clamp_min(EPS))


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_shape = args.image_shape * (args.magnification ** args.n_mag)
    target_resolution = args.resolution / (args.magnification ** args.n_mag)
    root = Path(args.mapping_dir)
    files = {
        "backward": "sparse_grid_fracs_euclid_backward.pt",
        "to_log": "scatter_to_log_128.pt",
        "forward_from_log": "forward_from_log_128.pt",
        "from_log": "scatter_from_log_128.pt",
    }
    mappings = {k: torch.load(root / v, map_location=device).to(device).coalesce() for k, v in files.items()}

    report: dict = {
        "configuration": {
            "resolution_lr_arcsec": args.resolution,
            "resolution_hr_arcsec": target_resolution,
            "theta_e_arcsec": args.theta_e,
            "critical_radius_lr_pixels": args.theta_e / args.resolution,
            "critical_radius_hr_pixels": args.theta_e / target_resolution,
            "psf_native_pixel_scale_arcsec": args.psf_source_pixscale_arcsec,
            "psf_target_pixel_scale_arcsec": target_resolution,
            "psf_resampling_factor": args.psf_source_pixscale_arcsec / target_resolution,
            "coordinate_units": args.coordinate_units,
            "log_c": args.log_c,
        },
        "matrix_stats": {k: sparse_stats(v) for k, v in mappings.items()},
        "tests": {},
        "warnings": [],
    }

    # Test 1: shape and finite-value checks.
    expected_lr = args.image_shape ** 2
    expected_hr = target_shape ** 2
    report["tests"]["shape_check"] = {
        "passed": mappings["backward"].shape == (expected_lr, expected_lr)
        and all(mappings[k].shape == (expected_hr, expected_hr) for k in ["to_log", "forward_from_log", "from_log"])
    }

    # Test 2: constant-surface-brightness preservation.
    ones_hr = torch.ones(1, 1, target_shape, target_shape, device=device)
    lens = DifferentiableLensing(device=device, alpha=None, target_resolution=target_resolution, target_shape=target_shape).to(device)
    with torch.inference_mode():
        const_intrinsic = lens.cross_grid_fill(ones_hr, [mappings["to_log"], mappings["forward_from_log"], mappings["from_log"]])
    interior = const_intrinsic[..., 4:-4, 4:-4]
    report["tests"]["constant_brightness"] = {
        "mean": float(interior.mean()),
        "std": float(interior.std()),
        "min": float(interior.min()),
        "max": float(interior.max()),
        "passed": abs(float(interior.mean()) - 1.0) < 0.05 and float(interior.std()) < 0.05,
    }

    # Test 3: regular -> log -> regular identity, excluding SIS.
    gaussian = synthetic_source(target_shape, sigma=4.0).to(device)
    with torch.inference_mode():
        identity_out = apply_sparse(apply_sparse(gaussian, mappings["to_log"], target_shape), mappings["from_log"], target_shape)
    report["tests"]["log_roundtrip"] = {
        "relative_mse": relative_error(identity_out, gaussian),
        "passed": relative_error(identity_out, gaussian) < 0.05,
    }

    # Test 4: compact centred Gaussian through the SIS chain.
    with torch.inference_mode():
        gaussian_lensed = lens.cross_grid_fill(gaussian, [mappings["to_log"], mappings["forward_from_log"], mappings["from_log"]])
    gp = radial_profile(gaussian_lensed)
    gpeaks = peak_radii(gp)
    report["tests"]["centred_gaussian_sis"] = {
        "strongest_radial_peaks_pixels": gpeaks,
        "expected_primary_radius_pixels": args.theta_e / target_resolution,
        "double_peak_warning": len(gpeaks) >= 2 and abs(gpeaks[0] - gpeaks[1]) >= 2,
    }

    # Test 5: deliberately ring-shaped source. Two SIS image radii are physical here.
    source_ring_radius = 3.0
    ring_source = synthetic_source(target_shape, sigma=1.0, ring_radius=source_ring_radius).to(device)
    with torch.inference_mode():
        ring_lensed = lens.cross_grid_fill(ring_source, [mappings["to_log"], mappings["forward_from_log"], mappings["from_log"]])
    rp = radial_profile(ring_lensed)
    report["tests"]["ring_source_sis"] = {
        "strongest_radial_peaks_pixels": peak_radii(rp),
        "expected_inner_outer_pixels": [
            args.theta_e / target_resolution - source_ring_radius,
            args.theta_e / target_resolution + source_ring_radius,
        ],
    }

    # Test 6: backward reconstruction of a constant LR image.
    backward_avg = normalize_sparse_rows(mappings["backward"])
    ones_lr = torch.ones(1, 1, args.image_shape, args.image_shape, device=device)
    back_const = apply_sparse(ones_lr, backward_avg, args.image_shape)
    covered = back_const > 0
    values = back_const[covered]
    report["tests"]["backward_constant"] = {
        "mean": float(values.mean()) if values.numel() else 0.0,
        "std": float(values.std()) if values.numel() > 1 else 0.0,
        "passed": values.numel() > 0 and abs(float(values.mean()) - 1.0) < 0.01,
    }

    # Test 7: magnification-information geometry.
    info = build_fixed_sis_information_maps(
        mappings["backward"], args.image_shape, args.resolution, args.theta_e
    )
    info_profile = radial_profile(info.image_information)
    report["tests"]["magnification_information"] = {
        "image_information_min": float(info.image_information.min()),
        "image_information_max": float(info.image_information.max()),
        "source_information_min": float(info.source_information_lr.min()),
        "source_information_max": float(info.source_information_lr.max()),
        "strongest_image_information_peaks_pixels": peak_radii(info_profile),
        "expected_critical_radius_lr_pixels": args.theta_e / args.resolution,
    }

    # Test 8: PSF construction and flux conservation.
    psf = build_psf_kernel(
        "fits", 0.16, target_resolution,
        path=args.psf_path,
        source_pixscale_arcsec=args.psf_source_pixscale_arcsec,
        device=device,
    )
    delta = torch.zeros(1, 1, target_shape, target_shape, device=device)
    delta[..., target_shape // 2, target_shape // 2] = 1.0
    psf_delta = apply_psf(delta, psf)
    report["tests"]["psf"] = {
        "kernel_shape": list(psf.shape),
        "kernel_sum": float(psf.sum()),
        "delta_output_sum": float(psf_delta.sum()),
        "passed": abs(float(psf.sum()) - 1.0) < 1e-5 and abs(float(psf_delta.sum()) - 1.0) < 1e-3,
    }

    if args.coordinate_units == "pixels" and args.log_c is not None:
        equivalent_arcsec_c = args.log_c / target_resolution
        report["warnings"].append(
            f"Pixel-coordinate log grid: log_c={args.log_c} corresponds to {equivalent_arcsec_c:.6g} arcsec^-1. "
            "Confirm this was intentional; log_c is not invariant under coordinate-unit changes."
        )

    # Optional trained checkpoint test. It must use the same mapping files.
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        ckpt_args = checkpoint["args"]
        model = SISR(
            ckpt_args["magnification"], ckpt_args["n_mag"], ckpt_args["residual_depth"],
            in_channels=2, latent_channel_count=ckpt_args["latent_space_size"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        ds = data.LensingDataset(args.val_dir, [args.val_class], args.val_samples)
        lr = ds[args.val_index].unsqueeze(0).float().to(device)
        if lr.ndim == 3:
            lr = lr.unsqueeze(1)
        with torch.inference_mode():
            source_lr = apply_sparse(lr, backward_avg, args.image_shape)
            source_hr = model(torch.cat([source_lr, lr], dim=1))
            intrinsic = lens.cross_grid_fill(source_hr, [mappings["to_log"], mappings["forward_from_log"], mappings["from_log"]])
            pred = F.interpolate(apply_psf(intrinsic, psf), size=lr.shape[-2:], mode="area")
        zero_mse = lr.square().mean()
        model_mse = F.mse_loss(pred, lr)
        report["tests"]["checkpoint"] = {
            "zero_mse": float(zero_mse),
            "model_mse": float(model_mse),
            "skill_over_zero": float(1 - model_mse / zero_mse.clamp_min(EPS)),
            "prediction_radial_peaks_pixels": peak_radii(radial_profile(pred)),
            "observation_radial_peaks_pixels": peak_radii(radial_profile(lr)),
        }

    report_path = out_dir / "magnification_diagnostics_checkpoint_old_matrices.json"
    report_path.write_text(json.dumps(report, indent=2))
    torch.save(
        {
            "constant_intrinsic": const_intrinsic.cpu(),
            "identity_roundtrip": identity_out.cpu(),
            "gaussian_source": gaussian.cpu(),
            "gaussian_lensed": gaussian_lensed.cpu(),
            "ring_source": ring_source.cpu(),
            "ring_lensed": ring_lensed.cpu(),
            "image_information": info.image_information.cpu(),
            "source_information": info.source_information_lr.cpu(),
            "psf": psf.cpu(),
        },
        out_dir / "magnification_diagnostic_tensors.pt",
    )
    print(json.dumps(report, indent=2))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()