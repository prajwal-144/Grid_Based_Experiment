"""PSF utilities for the grid-based unsupervised SR pipeline.

The observation model used during training is

    y_lr ~= downsample(PSF * x_hr)

For Stage 1-2 we keep the original fixed-SIS sparse lensing matrices and only
make the telescope degradation configurable. This module supports analytic
Gaussian/Moffat PSFs and empirical PSF kernels, including FITS files from HSC or
HST.

Important convention
--------------------
``apply_psf`` convolves the high-resolution lensed image before downsampling.
Therefore an empirical PSF should be sampled on the same pixel scale as the HR
image grid. If your FITS PSF is sampled on the native observed pixel grid, pass
``source_pixscale_arcsec`` so the kernel is resampled to ``target_pixscale``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def _odd_kernel_size(size: int) -> int:
    """Return a positive odd kernel size."""
    size = int(max(3, size))
    return size if size % 2 == 1 else size + 1


def _default_kernel_size(
    fwhm_arcsec: float,
    pixscale_arcsec: float,
    support_radius_fwhm: float = 5.0,
) -> int:
    """Choose a conservative odd kernel size from FWHM and pixel scale.

    The returned width covers roughly +/- support_radius_fwhm * FWHM.
    """
    if pixscale_arcsec <= 0:
        raise ValueError("pixscale_arcsec must be positive")
    fwhm_pix = max(float(fwhm_arcsec) / float(pixscale_arcsec), 1e-6)
    size = int(math.ceil(2.0 * support_radius_fwhm * fwhm_pix))
    return _odd_kernel_size(size)


def _coordinate_grid(
    kernel_size: int,
    device=None,
    dtype=torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Centered coordinate grid in pixel units."""
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return xx, yy


def _center_crop_2d(kernel: torch.Tensor, crop_size: Optional[int]) -> torch.Tensor:
    """Center-crop a 2D kernel to an odd square size.

    If crop_size is None, no crop is applied. If the requested crop is larger
    than the kernel, the original kernel is returned.
    """
    if crop_size is None:
        return kernel
    crop_size = _odd_kernel_size(crop_size)
    if kernel.ndim != 2:
        raise ValueError("_center_crop_2d expects a 2D tensor")
    h, w = kernel.shape
    if crop_size >= h or crop_size >= w:
        return kernel
    cy, cx = h // 2, w // 2
    radius = crop_size // 2
    return kernel[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]


def _force_odd_shape(kernel: torch.Tensor) -> torch.Tensor:
    """Trim one row/column if interpolation produced an even-sized kernel."""
    if kernel.ndim != 2:
        raise ValueError("_force_odd_shape expects a 2D tensor")
    h, w = kernel.shape
    if h % 2 == 0:
        kernel = kernel[1:, :]
    if w % 2 == 0:
        kernel = kernel[:, 1:]
    return kernel


def _extract_first_2d_plane(data: np.ndarray) -> np.ndarray:
    """Extract a 2D PSF image from common FITS data shapes.

    FITS PSF products are sometimes stored as [Y, X], [1, Y, X], or
    [N, Y, X]. This function squeezes singleton dimensions and, if needed,
    takes the first plane until a 2D image remains.
    """
    arr = np.asarray(data)
    arr = np.squeeze(arr)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Could not extract a 2D PSF plane from shape {data.shape}")
    return arr


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
    """Create a normalized elliptical Gaussian PSF kernel."""
    if fwhm_arcsec <= 0:
        raise ValueError("fwhm_arcsec must be positive")
    if pixscale_arcsec <= 0:
        raise ValueError("pixscale_arcsec must be positive")
    if not (0 < ellipticity_q <= 1.0):
        raise ValueError("ellipticity_q must be in (0, 1]")

    kernel_size = _odd_kernel_size(
        kernel_size or _default_kernel_size(fwhm_arcsec, pixscale_arcsec)
    )
    xx, yy = _coordinate_grid(kernel_size, device=device, dtype=dtype)

    theta = math.radians(float(angle_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x_rot = cos_t * xx + sin_t * yy
    y_rot = -sin_t * xx + cos_t * yy

    sigma_major_pix = (float(fwhm_arcsec) / 2.354820045) / float(pixscale_arcsec)
    sigma_minor_pix = sigma_major_pix * float(ellipticity_q)
    kernel = torch.exp(
        -0.5 * ((x_rot / sigma_major_pix) ** 2 + (y_rot / sigma_minor_pix) ** 2)
    )
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

    kernel_size = _odd_kernel_size(
        kernel_size or _default_kernel_size(fwhm_arcsec, pixscale_arcsec)
    )
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


def resample_kernel_to_pixscale(
    kernel: torch.Tensor,
    source_pixscale_arcsec: Optional[float],
    target_pixscale_arcsec: float,
    mode: str = "bicubic",
) -> torch.Tensor:
    """Resample a kernel from its native pixel scale to the HR target grid.

    Example: if the PSF FITS kernel is sampled at 0.168 arcsec/pixel and the HR
    image grid is 0.084 arcsec/pixel, scale_factor = 0.168 / 0.084 = 2, so the
    kernel is upsampled by 2 before convolution.
    """
    if source_pixscale_arcsec is None:
        return normalize_kernel(_force_odd_shape(kernel))
    if source_pixscale_arcsec <= 0 or target_pixscale_arcsec <= 0:
        raise ValueError("source and target PSF pixel scales must be positive")

    scale_factor = float(source_pixscale_arcsec) / float(target_pixscale_arcsec)
    if abs(scale_factor - 1.0) < 1e-6:
        return normalize_kernel(_force_odd_shape(kernel))

    kernel_4d = kernel[None, None].float()
    align_corners = False if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None
    if align_corners is None:
        resized = F.interpolate(kernel_4d, scale_factor=scale_factor, mode=mode)
    else:
        resized = F.interpolate(
            kernel_4d,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    return normalize_kernel(_force_odd_shape(resized[0, 0]))


def load_fits_psf_kernel(
    path: Union[str, Path],
    hdu: int = 0,
    extname: Optional[str] = None,
    crop_size: Optional[int] = None,
    source_pixscale_arcsec: Optional[float] = None,
    target_pixscale_arcsec: Optional[float] = None,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Load and normalize a PSF kernel from a FITS file.

    Parameters
    ----------
    path:
        FITS file containing a 2D PSF image or an image cube with a 2D plane.
    hdu/extname:
        Select the FITS HDU. If extname is given it overrides hdu.
    crop_size:
        Optional center crop after loading. Use this to keep very large PSF
        stamps computationally manageable. The crop size is forced odd.
    source_pixscale_arcsec:
        Native pixel scale of the FITS PSF kernel. If supplied, the kernel is
        resampled to target_pixscale_arcsec before use.
    target_pixscale_arcsec:
        Pixel scale of the HR image grid being convolved.
    """
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError(
            "Reading FITS PSF files requires astropy. Install it with: pip install astropy"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FITS PSF kernel not found: {path}")

    with fits.open(path, memmap=False) as hdul:
        selected_hdu = hdul[extname] if extname else hdul[hdu]
        data = selected_hdu.data
        if data is None:
            # Some FITS files have an empty primary HDU. Fall back to the first
            # image-like HDU with data.
            for candidate in hdul:
                if candidate.data is not None:
                    data = candidate.data
                    break
        if data is None:
            raise ValueError(f"No image data found in FITS file: {path}")

    arr = _extract_first_2d_plane(data).astype(np.float32)
    kernel = torch.as_tensor(arr, device=device, dtype=dtype)
    kernel = _center_crop_2d(kernel, crop_size)

    if target_pixscale_arcsec is None:
        target_pixscale_arcsec = source_pixscale_arcsec
    if source_pixscale_arcsec is not None and target_pixscale_arcsec is None:
        raise ValueError("target_pixscale_arcsec is required when source_pixscale_arcsec is set")

    kernel = resample_kernel_to_pixscale(
        normalize_kernel(kernel),
        source_pixscale_arcsec=source_pixscale_arcsec,
        target_pixscale_arcsec=target_pixscale_arcsec or 1.0,
    )
    return kernel.to(device=device, dtype=dtype)


def summarize_fits_hdus(path: Union[str, Path]) -> List[Dict[str, object]]:
    """Return a compact HDU summary for notebook inspection."""
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError(
            "FITS inspection requires astropy. Install it with: pip install astropy"
        ) from exc

    rows: List[Dict[str, object]] = []
    with fits.open(path, memmap=False) as hdul:
        for idx, h in enumerate(hdul):
            data = h.data
            rows.append(
                {
                    "index": idx,
                    "name": h.name,
                    "type": type(h).__name__,
                    "shape": None if data is None else tuple(data.shape),
                    "dtype": None if data is None else str(data.dtype),
                    "bunit": h.header.get("BUNIT"),
                    "cdelt1": h.header.get("CDELT1"),
                    "cdelt2": h.header.get("CDELT2"),
                    "pixscale": h.header.get("PIXSCALE") or h.header.get("PIXSCAL1"),
                }
            )
    return rows


def load_empirical_psf_kernel(
    path: Union[str, Path],
    device=None,
    dtype=torch.float32,
    fits_hdu: int = 0,
    fits_extname: Optional[str] = None,
    fits_crop_size: Optional[int] = None,
    source_pixscale_arcsec: Optional[float] = None,
    target_pixscale_arcsec: Optional[float] = None,
) -> torch.Tensor:
    """Load a PSF kernel from .npy, .npz, .pt/.pth, or FITS and normalize it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PSF kernel not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fts"}:
        return load_fits_psf_kernel(
            path,
            hdu=fits_hdu,
            extname=fits_extname,
            crop_size=fits_crop_size,
            source_pixscale_arcsec=source_pixscale_arcsec,
            target_pixscale_arcsec=target_pixscale_arcsec,
            device=device,
            dtype=dtype,
        )

    if suffix == ".npy":
        arr = np.load(path)
        kernel = torch.as_tensor(arr, device=device, dtype=dtype)
    elif suffix == ".npz":
        data = np.load(path)
        arr = data["psf"] if "psf" in data else data[list(data.keys())[0]]
        kernel = torch.as_tensor(arr, device=device, dtype=dtype)
    elif suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location=device)
        if isinstance(obj, dict):
            if "psf" in obj:
                obj = obj["psf"]
            elif "kernel" in obj:
                obj = obj["kernel"]
            else:
                obj = obj[next(iter(obj))]
        kernel = torch.as_tensor(obj, device=device, dtype=dtype)
    else:
        raise ValueError("Empirical PSF path must be .npy, .npz, .pt, .pth, .fits, .fit, or .fts")

    kernel = kernel.squeeze()
    if kernel.ndim != 2:
        raise ValueError(f"Expected a 2D PSF kernel after squeeze; got shape {tuple(kernel.shape)}")
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("PSF kernel must be square")
    kernel = _center_crop_2d(kernel, fits_crop_size)
    kernel = resample_kernel_to_pixscale(
        normalize_kernel(kernel),
        source_pixscale_arcsec=source_pixscale_arcsec,
        target_pixscale_arcsec=target_pixscale_arcsec or 1.0,
    )
    return kernel.to(device=device, dtype=dtype)


def build_psf_kernel(
    psf_type: str,
    fwhm_arcsec: float,
    pixscale_arcsec: float,
    beta: float = 4.765,
    kernel_size: Optional[int] = None,
    path: Optional[str] = None,
    ellipticity_q: float = 1.0,
    angle_deg: float = 0.0,
    fits_hdu: int = 0,
    fits_extname: Optional[str] = None,
    fits_crop_size: Optional[int] = None,
    source_pixscale_arcsec: Optional[float] = None,
    device=None,
    dtype=torch.float32,
) -> Optional[torch.Tensor]:
    """Build or load a 2D normalized PSF kernel.

    Returns None when psf_type == "none". Use ``psf_type='fits'`` for FITS PSF
    files or ``psf_type='empirical'`` for any supported empirical kernel format.
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
    if psf_type in {"empirical", "fits"}:
        if not path:
            raise ValueError("--psf-path is required when --psf-type empirical/fits")
        return load_empirical_psf_kernel(
            path,
            device=device,
            dtype=dtype,
            fits_hdu=fits_hdu,
            fits_extname=fits_extname,
            fits_crop_size=fits_crop_size,
            source_pixscale_arcsec=source_pixscale_arcsec,
            target_pixscale_arcsec=pixscale_arcsec,
        )
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
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("PSF kernel must be square")
    if kernel.shape[0] % 2 == 0:
        raise ValueError("PSF kernel size must be odd for same-padding convolution")

    channels = image.shape[1]
    kernel = kernel.to(device=image.device, dtype=image.dtype)
    weight = kernel.view(1, 1, *kernel.shape).repeat(channels, 1, 1, 1)
    padding = kernel.shape[-1] // 2
    return F.conv2d(image, weight, padding=padding, groups=channels)
