"""Flux and observation-consistency diagnostics for Stage 1-2 FITS-PSF SR.

This module intentionally uses the original fixed-SIS sparse mappings. It does
not use the Stage-3 learnable SIS implementation.

The decisive comparison is not raw HR SR versus raw LR. It is:

    SR source -> fixed SIS -> HR lensed image -> FITS PSF -> downsample -> LR

The re-degraded output should reproduce the observed LR image. Raw HR peaks may
be larger than LR peaks because deconvolution concentrates brightness into more,
smaller pixels. Integrated flux must include pixel area when grids differ.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8


def ensure_bchw(image: torch.Tensor) -> torch.Tensor:
    """Convert common image shapes to BCHW."""
    while image.ndim > 4 and image.shape[1] == 1:
        image = image.squeeze(1)
    if image.ndim == 2:
        image = image[None, None]
    elif image.ndim == 3:
        image = image[None]
    if image.ndim != 4:
        raise ValueError(f"Expected image convertible to BCHW; got {tuple(image.shape)}")
    return image


def interpolate_image(image: torch.Tensor, scale_factor: float, mode: str = "area") -> torch.Tensor:
    """Interpolate with safe align_corners handling."""
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        return F.interpolate(image, scale_factor=scale_factor, mode=mode, align_corners=False)
    return F.interpolate(image, scale_factor=scale_factor, mode=mode)


def load_sr_weights(model: torch.nn.Module, path: str, device: torch.device) -> None:
    """Load either a Stage 1-2 model state dict or a checkpoint containing one."""
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "sr_model_state_dict" in obj:
        obj = obj["sr_model_state_dict"]
    model.load_state_dict(obj)


def fixed_sis_forward(
    source_hr: torch.Tensor,
    lensing_module,
    cross_grid_to_log: torch.Tensor,
    cross_grid_forward_from_log: torch.Tensor,
    cross_grid_from_log: torch.Tensor,
) -> torch.Tensor:
    """Forward-lens an HR source using the original fixed-SIS sparse matrices."""
    return lensing_module.cross_grid_fill(
        source_hr,
        [cross_grid_to_log, cross_grid_forward_from_log, cross_grid_from_log],
    )


def run_stage1_2_sample(
    lr_image: torch.Tensor,
    model: torch.nn.Module,
    lensing_module,
    cross_grid_backward: torch.Tensor,
    cross_grid_to_log: torch.Tensor,
    cross_grid_forward_from_log: torch.Tensor,
    cross_grid_from_log: torch.Tensor,
    psf_kernel: Optional[torch.Tensor],
    effective_magnification: int,
    downsample_mode: str = "area",
) -> Dict[str, torch.Tensor]:
    """Run the complete fixed-SIS Stage 1-2 inference and degradation cycle."""
    from psf import apply_psf

    lr_image = ensure_bchw(lr_image)
    reconstructed_source = lensing_module.cross_grid_fill(lr_image, [cross_grid_backward])
    model_feed = torch.cat([reconstructed_source, lr_image], dim=1)
    source_hr = model(model_feed)
    image_hr = fixed_sis_forward(
        source_hr,
        lensing_module,
        cross_grid_to_log,
        cross_grid_forward_from_log,
        cross_grid_from_log,
    )
    image_hr_psf = apply_psf(image_hr, psf_kernel)
    lr_redegraded = interpolate_image(
        image_hr_psf,
        scale_factor=1.0 / float(effective_magnification),
        mode=downsample_mode,
    )
    source_lr_cycle = interpolate_image(
        source_hr,
        scale_factor=1.0 / float(effective_magnification),
        mode=downsample_mode,
    )
    return {
        "lr": lr_image,
        "source_lr": reconstructed_source,
        "source_hr": source_hr,
        "image_hr": image_hr,
        "image_hr_psf": image_hr_psf,
        "lr_redegraded": lr_redegraded,
        "source_lr_cycle": source_lr_cycle,
        "residual_lr": lr_redegraded - lr_image,
    }


def area_weighted_flux(image: torch.Tensor, pixel_scale_arcsec: float) -> torch.Tensor:
    """Return per-sample integral sum(I * pixel_area)."""
    image = ensure_bchw(image)
    pixel_area = float(pixel_scale_arcsec) ** 2
    return image.sum(dim=(1, 2, 3)) * pixel_area


def weighted_centroid(image_2d: np.ndarray) -> Tuple[float, float]:
    """Intensity-weighted centroid as (y, x), falling back to image center."""
    image = np.nan_to_num(np.asarray(image_2d, dtype=float), nan=0.0)
    image = np.clip(image, 0.0, None)
    total = image.sum()
    h, w = image.shape
    if total <= EPS:
        return (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.indices(image.shape)
    return float((yy * image).sum() / total), float((xx * image).sum() / total)


def radial_profile(image_2d: np.ndarray, center: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged radial profile in pixel units."""
    image = np.asarray(image_2d, dtype=float)
    if center is None:
        center = weighted_centroid(image)
    yy, xx = np.indices(image.shape)
    radius = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    rbin = np.floor(radius).astype(int)
    sums = np.bincount(rbin.ravel(), weights=image.ravel())
    counts = np.bincount(rbin.ravel())
    profile = sums / np.maximum(counts, 1)
    return np.arange(len(profile)), profile


def psf_diagnostics(kernel: torch.Tensor) -> Dict[str, float]:
    """Flux normalization and centering diagnostics for a 2D PSF kernel."""
    k = kernel.detach().float().cpu().numpy()
    k = np.nan_to_num(k, nan=0.0)
    h, w = k.shape
    yy, xx = np.indices(k.shape)
    total = k.sum()
    cy = float((yy * k).sum() / max(total, EPS))
    cx = float((xx * k).sum() / max(total, EPS))
    geometric_y, geometric_x = (h - 1) / 2.0, (w - 1) / 2.0
    peak_y, peak_x = np.unravel_index(np.argmax(k), k.shape)
    return {
        "psf_sum": float(total),
        "psf_min": float(k.min()),
        "psf_max": float(k.max()),
        "psf_centroid_offset_pix": float(np.hypot(cy - geometric_y, cx - geometric_x)),
        "psf_peak_offset_pix": float(np.hypot(peak_y - geometric_y, peak_x - geometric_x)),
        "psf_height": int(h),
        "psf_width": int(w),
    }


def image_metrics(
    outputs: Dict[str, torch.Tensor],
    lr_pixel_scale_arcsec: float,
    hr_pixel_scale_arcsec: float,
) -> Dict[str, float]:
    """Compute brightness, flux, and LR observation-consistency metrics."""
    lr = outputs["lr"].detach().float()
    hr = outputs["image_hr"].detach().float()
    hr_psf = outputs["image_hr_psf"].detach().float()
    redegraded = outputs["lr_redegraded"].detach().float()
    residual = redegraded - lr

    lr_flux = area_weighted_flux(lr, lr_pixel_scale_arcsec).mean()
    hr_flux = area_weighted_flux(hr, hr_pixel_scale_arcsec).mean()
    hr_psf_flux = area_weighted_flux(hr_psf, hr_pixel_scale_arcsec).mean()
    redegraded_flux = area_weighted_flux(redegraded, lr_pixel_scale_arcsec).mean()

    mse = residual.square().mean()
    mae = residual.abs().mean()
    target_power = lr.square().mean().clamp_min(EPS)
    normalized_mse = mse / target_power
    data_range = (lr.max() - lr.min()).clamp_min(EPS)
    psnr = 20.0 * torch.log10(data_range) - 10.0 * torch.log10(mse.clamp_min(EPS))

    lr_flat = lr.flatten()
    re_flat = redegraded.flatten()
    lr_centered = lr_flat - lr_flat.mean()
    re_centered = re_flat - re_flat.mean()
    correlation = (
        (lr_centered * re_centered).sum()
        / (torch.sqrt((lr_centered.square()).sum() * (re_centered.square()).sum()) + EPS)
    )

    q99_lr = torch.quantile(lr, 0.99)
    q99_hr = torch.quantile(hr, 0.99)
    q99_re = torch.quantile(redegraded, 0.99)

    return {
        "lr_peak": float(lr.max().cpu()),
        "hr_peak": float(hr.max().cpu()),
        "hr_psf_peak": float(hr_psf.max().cpu()),
        "redegraded_peak": float(redegraded.max().cpu()),
        "hr_to_lr_peak_ratio": float((hr.max() / lr.max().clamp_min(EPS)).cpu()),
        "redegraded_to_lr_peak_ratio": float((redegraded.max() / lr.max().clamp_min(EPS)).cpu()),
        "lr_q99": float(q99_lr.cpu()),
        "hr_q99": float(q99_hr.cpu()),
        "redegraded_q99": float(q99_re.cpu()),
        "lr_flux_area_weighted": float(lr_flux.cpu()),
        "hr_flux_area_weighted": float(hr_flux.cpu()),
        "hr_psf_flux_area_weighted": float(hr_psf_flux.cpu()),
        "redegraded_flux_area_weighted": float(redegraded_flux.cpu()),
        "hr_to_lr_flux_ratio": float((hr_flux / lr_flux.clamp_min(EPS)).cpu()),
        "redegraded_to_lr_flux_ratio": float((redegraded_flux / lr_flux.clamp_min(EPS)).cpu()),
        "lr_redegradation_mae": float(mae.cpu()),
        "lr_redegradation_nmse": float(normalized_mse.cpu()),
        "lr_redegradation_psnr": float(psnr.cpu()),
        "lr_redegradation_correlation": float(correlation.cpu()),
        "negative_fraction_hr": float((hr < 0).float().mean().cpu()),
        "negative_fraction_redegraded": float((redegraded < 0).float().mean().cpu()),
    }
