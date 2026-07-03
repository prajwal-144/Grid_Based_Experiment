"""PSF utilities for the grid-based unsupervised SR pipeline.

The observation model used during training is

    y_lr ~= downsample(PSF * x_hr)

This module keeps PSF construction separate from the training loop so that
simulated, survey-provided, and analytic PSFs can be swapped without touching
model code.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def _odd_kernel_size(size: int) -> int:
    """Return a positive odd kernel size."""
    size = int(max(3, size))
    return size if size % 2 == 1 else size + 1


def _default_kernel_size(fwhm_arcsec: float, pixscale_arcsec: float, support_radius_fwhm: float = 5.0) -> int:
    """Choose a conservative odd kernel size from FWHM and pixel scale.

    The returned width covers roughly +/- support_radius_fwhm * FWHM.
    """
    if pixscale_arcsec <= 0:
        raise ValueError("pixscale_arcsec must be positive")
    fwhm_pix = max(float(fwhm_arcsec) / float(pixscale_arcsec), 1e-6)
    size = int(math.ceil(2.0 * support_radius_fwhm * fwhm_pix))
    return _odd_kernel_size(size)


def _coordinate_grid(kernel_size: int, device=None, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Centered coordinate grid in pixel units."""
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def normalize_kernel(kernel: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Clamp small negative numerical values and normalize kernel sum to one."""
    kernel = torch.nan_to_num(kernel.float(), nan=0.0, posinf=0.0, neginf=0.0)
    kernel = torch.clamp(kernel, min=0.0)
    total = kernel.sum()
    if total <= eps:
        raise ValueError("PSF kernel has non-positive total flux")
    return kernel / total


def gaussian_psf_kernel(
    fwhm_arcsec: float,
    pixscale_arcsec: float,
    kernel_size: Optional[int] = None,
    ellipticity_q: float = 1.0,
    angle_deg: float = 0.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Create a normalized elliptical Gaussian PSF kernel.

    Parameters
    ----------
    fwhm_arcsec:
        FWHM of the major axis in arcseconds.
    pixscale_arcsec:
        Pixel scale of the image being convolved, in arcsec/pixel.
    kernel_size:
        Optional odd kernel width. If omitted, a support size is inferred.
    ellipticity_q:
        Minor-to-major axis ratio. q=1 gives a circular Gaussian.
    angle_deg:
        Counter-clockwise major-axis angle in degrees.
    """
    if fwhm_arcsec <= 0:
        raise ValueError("fwhm_arcsec must be positive")
    if pixscale_arcsec <= 0:
        raise ValueError("pixscale_arcsec must be positive")
    if not (0 < ellipticity_q <= 1.0):
        raise ValueError("ellipticity_q must be in (0, 1]")

    kernel_size = _odd_kernel_size(kernel_size or _default_kernel_size(fwhm_arcsec, pixscale_arcsec))
    xx, yy = _coordinate_grid(kernel_size, device=device, dtype=dtype)

    theta = math.radians(float(angle_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x_rot = cos_t * xx + sin_t * yy
    y_rot = -sin_t * xx + cos_t * yy

    sigma_major_pix = (float(fwhm_arcsec) / 2.354820045) / float(pixscale_arcsec)
    sigma_minor_pix = sigma_major_pix * float(ellipticity_q)
    kernel = torch.exp(-0.5 * ((x_rot / sigma_major_pix) ** 2 + (y_rot / sigma_minor_pix) ** 2))
    return normalize_kernel(kernel)


def moffat_psf_kernel(
    fwhm_arcsec: float,
    pixscale_arcsec: float,
    beta: float = 4.765,
    kernel_size: Optional[int] = None,
    ellipticity_q: float = 1.0,
    angle_deg: float = 0.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Create a normalized elliptical Moffat PSF kernel.

    Moffat profiles have heavier wings than Gaussian PSFs and are often a
    better analytic fallback for ground-based seeing.
    """
    if fwhm_arcsec <= 0:
        raise ValueError("fwhm_arcsec must be positive")
    if pixscale_arcsec <= 0:
        raise ValueError("pixscale_arcsec must be positive")
    if beta <= 1:
        raise ValueError("Moffat beta must be > 1 for finite total flux")
    if not (0 < ellipticity_q <= 1.0):
        raise ValueError("ellipticity_q must be in (0, 1]")

    kernel_size = _odd_kernel_size(kernel_size or _default_kernel_size(fwhm_arcsec, pixscale_arcsec))
    xx, yy = _coordinate_grid(kernel_size, device=device, dtype=dtype)

    theta = math.radians(float(angle_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x_rot = cos_t * xx + sin_t * yy
    y_rot = -sin_t * xx + cos_t * yy

    fwhm_pix = float(fwhm_arcsec) / float(pixscale_arcsec)
    alpha = fwhm_pix / (2.0 * math.sqrt(2.0 ** (1.0 / float(beta)) - 1.0))
    r2 = (x_rot / alpha) ** 2 + (y_rot / (alpha * float(ellipticity_q))) ** 2
    kernel = (1.0 + r2) ** (-float(beta))
    return normalize_kernel(kernel)


def load_empirical_psf_kernel(path: Union[str, Path], device=None, dtype=torch.float32) -> torch.Tensor:
    """Load a PSF kernel from .npy, .npz, or .pt/.pth and normalize it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PSF kernel not found: {path}")

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        kernel = torch.as_tensor(arr, device=device, dtype=dtype)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        if "psf" in data:
            arr = data["psf"]
        else:
            first_key = list(data.keys())[0]
            arr = data[first_key]
        kernel = torch.as_tensor(arr, device=device, dtype=dtype)
    elif path.suffix.lower() in {".pt", ".pth"}:
        obj = torch.load(path, map_location=device)
        if isinstance(obj, dict):
            if "psf" in obj:
                obj = obj["psf"]
            elif "kernel" in obj:
                obj = obj["kernel"]
            else:
                first_key = next(iter(obj))
                obj = obj[first_key]
        kernel = torch.as_tensor(obj, device=device, dtype=dtype)
    else:
        raise ValueError("Empirical PSF path must be .npy, .npz, .pt, or .pth")

    kernel = kernel.squeeze()
    if kernel.ndim != 2:
        raise ValueError(f"Expected a 2D PSF kernel after squeeze; got shape {tuple(kernel.shape)}")
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("PSF kernel must be square")
    if kernel.shape[0] % 2 == 0:
        raise ValueError("PSF kernel size must be odd for centered same-padding convolution")
    return normalize_kernel(kernel)


def build_psf_kernel(
    psf_type: str,
    fwhm_arcsec: float,
    pixscale_arcsec: float,
    beta: float = 4.765,
    kernel_size: Optional[int] = None,
    path: Optional[str] = None,
    ellipticity_q: float = 1.0,
    angle_deg: float = 0.0,
    device=None,
    dtype=torch.float32,
) -> Optional[torch.Tensor]:
    """Build or load a 2D normalized PSF kernel.

    Returns None when psf_type == "none".
    """
    psf_type = psf_type.lower().strip()
    if psf_type == "none":
        return None
    if psf_type == "gaussian":
        return gaussian_psf_kernel(
            fwhm_arcsec=fwhm_arcsec,
            pixscale_arcsec=pixscale_arcsec,
            kernel_size=kernel_size,
            ellipticity_q=ellipticity_q,
            angle_deg=angle_deg,
            device=device,
            dtype=dtype,
        )
    if psf_type == "moffat":
        return moffat_psf_kernel(
            fwhm_arcsec=fwhm_arcsec,
            pixscale_arcsec=pixscale_arcsec,
            beta=beta,
            kernel_size=kernel_size,
            ellipticity_q=ellipticity_q,
            angle_deg=angle_deg,
            device=device,
            dtype=dtype,
        )
    if psf_type == "empirical":
        if not path:
            raise ValueError("--psf-path is required when --psf-type empirical")
        return load_empirical_psf_kernel(path, device=device, dtype=dtype)
    raise ValueError(f"Unsupported psf_type: {psf_type}")


def apply_psf(image: torch.Tensor, kernel: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply a 2D PSF kernel to a BCHW image tensor.

    The same PSF is applied independently to each channel. If kernel is None,
    the input image is returned unchanged.
    """
    if kernel is None:
        return image
    if image.ndim != 4:
        raise ValueError(f"Expected image with shape [B, C, H, W], got {tuple(image.shape)}")
    if kernel.ndim != 2:
        raise ValueError(f"Expected 2D PSF kernel, got {tuple(kernel.shape)}")

    channels = image.shape[1]
    kernel = kernel.to(device=image.device, dtype=image.dtype)
    weight = kernel.view(1, 1, *kernel.shape).repeat(channels, 1, 1, 1)
    padding = kernel.shape[-1] // 2
    return F.conv2d(image, weight, padding=padding, groups=channels)