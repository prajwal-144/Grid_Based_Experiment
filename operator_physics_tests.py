"""Stagewise physics tests for sparse SIS mappings and empirical FITS PSFs.

This script is read-only. It identifies where surface-brightness modulation,
radial duplication, support loss, or PSF resampling enters the pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from differentiable_lensing import DifferentiableLensing
from psf import apply_psf, build_psf_kernel, load_fits_psf_kernel

EPS = 1e-8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mapping-dir", required=True)
    p.add_argument("--resolution", type=float, required=True, help="LR image scale in arcsec/pixel")
    p.add_argument("--theta-e", type=float, required=True)
    p.add_argument("--psf-path", required=True)
    p.add_argument("--psf-source-pixscale-arcsec", type=float, required=True)
    p.add_argument("--image-shape", type=int, default=64)
    p.add_argument("--magnification", type=int, default=2)
    p.add_argument("--output-dir", default="operator_physics_tests")
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def apply_sparse(image, matrix, side):
    b, c, h, w = image.shape
    flat = image.reshape(b * c, h * w).T
    out = torch.sparse.mm(matrix, flat)
    return out.T.contiguous().reshape(b, c, side, side)


def sparse_sums(matrix):
    m = matrix.coalesce()
    idx, val = m.indices(), m.values()
    rows = torch.zeros(m.shape[0], dtype=val.dtype, device=val.device)
    cols = torch.zeros(m.shape[1], dtype=val.dtype, device=val.device)
    rows.scatter_add_(0, idx[0], val)
    cols.scatter_add_(0, idx[1], val)
    return rows, cols


def row_normalize(matrix):
    m = matrix.coalesce()
    idx, val = m.indices(), m.values().clamp_min(0)
    rows, _ = sparse_sums(m)
    return torch.sparse_coo_tensor(idx, val / rows[idx[0]].clamp_min(EPS), m.shape, device=m.device).coalesce()


def radial_profile(image):
    x = image.detach().float().cpu()
    if x.ndim == 4:
        x = x[0, 0]
    h, w = x.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    cy, cx = (h - 1) / 2, (w - 1) / 2
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).long()
    sums = torch.zeros(int(rr.max()) + 1)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, rr.reshape(-1), x.reshape(-1))
    counts.scatter_add_(0, rr.reshape(-1), torch.ones_like(x).reshape(-1))
    return sums / counts.clamp_min(1)


def peaks(profile):
    out = []
    for i in range(1, len(profile) - 1):
        if profile[i] > profile[i - 1] and profile[i] >= profile[i + 1]:
            out.append(i)
    return sorted(out, key=lambda i: float(profile[i]), reverse=True)[:5]


def gaussian(side, sigma=4.0, offset_x=0.0, offset_y=0.0):
    yy, xx = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    cy, cx = (side - 1) / 2 + offset_y, (side - 1) / 2 + offset_x
    rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return torch.exp(-0.5 * rr2 / sigma**2)[None, None]


def stats(x):
    return {
        "mean": float(x.mean()), "std": float(x.std()),
        "min": float(x.min()), "max": float(x.max()), "sum": float(x.sum())
    }


def analytic_sis(source, theta_e_arcsec, pixel_scale_arcsec):
    """Inverse-ray-shooting reference using bilinear grid_sample."""
    b, c, h, w = source.shape
    half = pixel_scale_arcsec * h / 2
    coords = torch.linspace(-half, half, h, device=source.device, dtype=source.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    r = torch.sqrt(xx.square() + yy.square()).clamp_min(EPS)
    beta_x = xx - theta_e_arcsec * xx / r
    beta_y = yy - theta_e_arcsec * yy / r
    gx = beta_x / half
    gy = beta_y / half
    grid = torch.stack([gx, gy], dim=-1)[None]
    return F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    hr_shape = args.image_shape * args.magnification
    hr_scale = args.resolution / args.magnification

    root = Path(args.mapping_dir)
    names = {
        "backward": "sparse_grid_fracs_euclid_backward.pt",
        "to_log": "scatter_to_log_128.pt",
        "forward_from_log": "forward_from_log_128.pt",
        "from_log": "scatter_from_log_128.pt",
    }
    M = {k: torch.load(root / v, map_location=device).to(device).coalesce() for k, v in names.items()}

    report = {"configuration": vars(args) | {"hr_scale": hr_scale, "hr_shape": hr_shape}, "tests": {}}

    # 1. Stagewise constant response. Correct surface-brightness mappings keep ones near one.
    ones = torch.ones(1, 1, hr_shape, hr_shape, device=device)
    stage = {"input": ones}
    stage["to_log"] = apply_sparse(stage["input"], M["to_log"], hr_shape)
    stage["sis"] = apply_sparse(stage["to_log"], M["forward_from_log"], hr_shape)
    stage["from_log"] = apply_sparse(stage["sis"], M["from_log"], hr_shape)
    report["tests"]["stagewise_constant"] = {k: stats(v[..., 4:-4, 4:-4]) for k, v in stage.items()}

    # 2. Row-normalized diagnostic. If this fixes ones and removes extra rings, normalization is causal.
    Mr = {k: row_normalize(v) for k, v in M.items() if k != "backward"}
    stage_r = {"input": ones}
    stage_r["to_log"] = apply_sparse(stage_r["input"], Mr["to_log"], hr_shape)
    stage_r["sis"] = apply_sparse(stage_r["to_log"], Mr["forward_from_log"], hr_shape)
    stage_r["from_log"] = apply_sparse(stage_r["sis"], Mr["from_log"], hr_shape)
    report["tests"]["stagewise_constant_row_normalized"] = {k: stats(v[..., 4:-4, 4:-4]) for k, v in stage_r.items()}

    # 3. Support masks and sum maps reveal whether missing support is on inputs or outputs.
    support = {}
    for name, matrix in M.items():
        rs, cs = sparse_sums(matrix)
        side_r, side_c = int(matrix.shape[0] ** 0.5), int(matrix.shape[1] ** 0.5)
        support[name] = {
            "zero_rows": int((rs.abs() < EPS).sum()),
            "zero_columns": int((cs.abs() < EPS).sum()),
            "row_sum_minmax": [float(rs.min()), float(rs.max())],
            "column_sum_minmax": [float(cs.min()), float(cs.max())],
        }
        torch.save({"row_sums": rs.reshape(side_r, side_r).cpu(), "column_sums": cs.reshape(side_c, side_c).cpu()}, out / f"{name}_support.pt")
    report["tests"]["support"] = support

    # 4. Sparse SIS versus an independent analytic SIS reference.
    src = gaussian(hr_shape, sigma=4.0).to(device)
    sparse_lensed = apply_sparse(apply_sparse(apply_sparse(src, M["to_log"], hr_shape), M["forward_from_log"], hr_shape), M["from_log"], hr_shape)
    sparse_row_lensed = apply_sparse(apply_sparse(apply_sparse(src, Mr["to_log"], hr_shape), Mr["forward_from_log"], hr_shape), Mr["from_log"], hr_shape)
    analytic = analytic_sis(src, args.theta_e, hr_scale)
    rp_sparse, rp_row, rp_analytic = radial_profile(sparse_lensed), radial_profile(sparse_row_lensed), radial_profile(analytic)
    report["tests"]["sis_reference"] = {
        "expected_radius_pixels": args.theta_e / hr_scale,
        "sparse_peaks": peaks(rp_sparse),
        "row_normalized_sparse_peaks": peaks(rp_row),
        "analytic_peaks": peaks(rp_analytic),
        "sparse_vs_analytic_relative_mse": float((sparse_lensed - analytic).square().mean() / analytic.square().mean().clamp_min(EPS)),
        "row_normalized_vs_analytic_relative_mse": float((sparse_row_lensed - analytic).square().mean() / analytic.square().mean().clamp_min(EPS)),
    }

    # 5. PSF input/output sizes and edge flux. Upsampling is expected when target pixels are smaller.
    raw_psf = load_fits_psf_kernel(args.psf_path, source_pixscale_arcsec=None, target_pixscale_arcsec=None, device=device)
    psf = build_psf_kernel("fits", 0.16, hr_scale, path=args.psf_path, source_pixscale_arcsec=args.psf_source_pixscale_arcsec, device=device)
    delta_center = torch.zeros(1, 1, hr_shape, hr_shape, device=device); delta_center[..., hr_shape//2, hr_shape//2] = 1
    delta_edge = torch.zeros_like(delta_center); delta_edge[..., 8, 8] = 1
    report["tests"]["psf_sampling"] = {
        "raw_shape": list(raw_psf.shape),
        "resampled_shape": list(psf.shape),
        "expected_linear_scale_factor": args.psf_source_pixscale_arcsec / hr_scale,
        "raw_sum": float(raw_psf.sum()),
        "resampled_sum": float(psf.sum()),
        "center_delta_output_sum": float(apply_psf(delta_center, psf).sum()),
        "edge_delta_output_sum": float(apply_psf(delta_edge, psf).sum()),
        "kernel_larger_than_hr_image": bool(psf.shape[-1] > hr_shape),
    }

    # Save images/profiles for direct inspection.
    torch.save({
        "stagewise_constant": {k: v.cpu() for k, v in stage.items()},
        "stagewise_constant_row_normalized": {k: v.cpu() for k, v in stage_r.items()},
        "source": src.cpu(), "sparse_lensed": sparse_lensed.cpu(),
        "row_normalized_sparse_lensed": sparse_row_lensed.cpu(), "analytic_lensed": analytic.cpu(),
        "radial_sparse": rp_sparse, "radial_row_normalized": rp_row, "radial_analytic": rp_analytic,
        "raw_psf": raw_psf.cpu(), "resampled_psf": psf.cpu(),
    }, out / "operator_physics_tensors.pt")

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    for a, x, title in zip(ax.flat,
        [stage["to_log"], stage["sis"], stage["from_log"], analytic, sparse_lensed, sparse_row_lensed],
        ["ones after to_log", "ones after SIS", "ones after from_log", "analytic SIS", "sparse SIS", "row-normalized sparse SIS"]):
        a.imshow(x[0,0].detach().cpu(), cmap="gray", origin="lower"); a.set_title(title)
    plt.tight_layout(); plt.savefig(out / "operator_physics_summary_168.png", dpi=160); plt.close()

    (out / "operator_physics_report_168.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved diagnostics to {out}")


if __name__ == "__main__":
    main()
