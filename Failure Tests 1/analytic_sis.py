"""Direct, differentiable SIS lensing without precomputed overlap matrices.

The renderer implements inverse ray shooting:
    beta(theta) = theta - alpha(theta)
    I_image(theta) = I_source(beta(theta))

This preserves surface brightness by construction.  The optional backprojector
uses a normalized bilinear splat and is intended for fixed-geometry source
initialization; the forward renderer remains fully differentiable with respect
to source pixels and SIS parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

TensorLike = Union[float, int, torch.Tensor]


@dataclass(frozen=True)
class SISGridInfo:
    image_shape: Tuple[int, int]
    source_shape: Tuple[int, int]
    image_pixel_scale_arcsec: float
    source_pixel_scale_arcsec: float
    align_corners: bool


def _shape2(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"Expected an int or (height, width), received {value!r}")
    return int(value[0]), int(value[1])


def _batch_parameter(
    value: TensorLike,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.repeat(batch_size)
    elif tensor.ndim == 1 and tensor.numel() == 1:
        tensor = tensor.repeat(batch_size)
    elif tensor.ndim != 1 or tensor.numel() != batch_size:
        raise ValueError(
            f"{name} must be a scalar or a length-{batch_size} vector; "
            f"received shape {tuple(tensor.shape)}"
        )
    return tensor.view(batch_size, 1, 1)


def _pixel_centres(
    height: int,
    width: int,
    pixel_scale_arcsec: float,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return image-plane pixel-centre coordinates in arcseconds."""
    y = (torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2.0)
    x = (torch.arange(width, device=device, dtype=dtype) - (width - 1) / 2.0)
    y = y * float(pixel_scale_arcsec)
    x = x * float(pixel_scale_arcsec)
    theta_y, theta_x = torch.meshgrid(y, x, indexing="ij")
    return theta_x.unsqueeze(0), theta_y.unsqueeze(0)


class AnalyticSISRenderer(nn.Module):
    """Batched SIS inverse-ray renderer with optional fixed-geometry backprojection."""

    def __init__(
        self,
        image_shape: Union[int, Tuple[int, int]],
        image_pixel_scale_arcsec: float,
        source_shape: Optional[Union[int, Tuple[int, int]]] = None,
        source_pixel_scale_arcsec: Optional[float] = None,
        *,
        align_corners: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        image_shape_2d = _shape2(image_shape)
        source_shape_2d = _shape2(source_shape if source_shape is not None else image_shape)
        source_scale = (
            float(source_pixel_scale_arcsec)
            if source_pixel_scale_arcsec is not None
            else float(image_pixel_scale_arcsec)
        )
        if image_pixel_scale_arcsec <= 0 or source_scale <= 0:
            raise ValueError("Pixel scales must be positive.")

        theta_x, theta_y = _pixel_centres(
            image_shape_2d[0],
            image_shape_2d[1],
            float(image_pixel_scale_arcsec),
        )
        self.register_buffer("theta_x", theta_x)
        self.register_buffer("theta_y", theta_y)

        self.info = SISGridInfo(
            image_shape=image_shape_2d,
            source_shape=source_shape_2d,
            image_pixel_scale_arcsec=float(image_pixel_scale_arcsec),
            source_pixel_scale_arcsec=source_scale,
            align_corners=bool(align_corners),
        )
        self.eps = float(eps)

    def lens_equation(
        self,
        batch_size: int,
        theta_e_arcsec: TensorLike,
        center_x_arcsec: TensorLike = 0.0,
        center_y_arcsec: TensorLike = 0.0,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate beta = theta - alpha for a centered or offset SIS."""
        device = device if device is not None else self.theta_x.device
        dtype = dtype if dtype is not None else self.theta_x.dtype

        theta_x = self.theta_x.to(device=device, dtype=dtype)
        theta_y = self.theta_y.to(device=device, dtype=dtype)
        theta_e = _batch_parameter(theta_e_arcsec, batch_size, device, dtype, "theta_e_arcsec")
        center_x = _batch_parameter(center_x_arcsec, batch_size, device, dtype, "center_x_arcsec")
        center_y = _batch_parameter(center_y_arcsec, batch_size, device, dtype, "center_y_arcsec")

        dx = theta_x - center_x
        dy = theta_y - center_y
        radius = torch.sqrt(dx.square() + dy.square()).clamp_min(self.eps)
        alpha_x = theta_e * dx / radius
        alpha_y = theta_e * dy / radius

        beta_x = theta_x - alpha_x
        beta_y = theta_y - alpha_y
        return beta_x, beta_y

    def sampling_grid(
        self,
        batch_size: int,
        theta_e_arcsec: TensorLike,
        center_x_arcsec: TensorLike = 0.0,
        center_y_arcsec: TensorLike = 0.0,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a grid_sample grid and an in-bounds mask."""
        beta_x, beta_y = self.lens_equation(
            batch_size,
            theta_e_arcsec,
            center_x_arcsec,
            center_y_arcsec,
            dtype=dtype,
            device=device,
        )
        source_h, source_w = self.info.source_shape
        source_scale = self.info.source_pixel_scale_arcsec

        if self.info.align_corners:
            x_extent = max((source_w - 1) * source_scale / 2.0, self.eps)
            y_extent = max((source_h - 1) * source_scale / 2.0, self.eps)
            grid_x = beta_x / x_extent
            grid_y = beta_y / y_extent
        else:
            x_extent = max(source_w * source_scale / 2.0, self.eps)
            y_extent = max(source_h * source_scale / 2.0, self.eps)
            grid_x = beta_x / x_extent
            grid_y = beta_y / y_extent

        grid = torch.stack((grid_x, grid_y), dim=-1)
        in_bounds = (grid_x.abs() <= 1.0) & (grid_y.abs() <= 1.0)
        return grid, in_bounds.unsqueeze(1)

    def forward(
        self,
        source: torch.Tensor,
        theta_e_arcsec: TensorLike,
        center_x_arcsec: TensorLike = 0.0,
        center_y_arcsec: TensorLike = 0.0,
        *,
        return_grid: bool = False,
    ):
        """Render source-plane surface brightness onto the image plane."""
        if source.ndim != 4:
            raise ValueError(f"source must be BCHW, received shape {tuple(source.shape)}")
        if tuple(source.shape[-2:]) != self.info.source_shape:
            raise ValueError(
                f"Expected source shape {self.info.source_shape}, "
                f"received {tuple(source.shape[-2:])}"
            )
        grid, in_bounds = self.sampling_grid(
            source.shape[0],
            theta_e_arcsec,
            center_x_arcsec,
            center_y_arcsec,
            dtype=source.dtype,
            device=source.device,
        )
        image = F.grid_sample(
            source,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.info.align_corners,
        )
        if return_grid:
            return image, grid, in_bounds
        return image

    @torch.no_grad()
    def backproject(
        self,
        image: torch.Tensor,
        theta_e_arcsec: TensorLike,
        center_x_arcsec: TensorLike = 0.0,
        center_y_arcsec: TensorLike = 0.0,
        *,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backproject image rays by normalized bilinear splatting.

        This is intended for fixed-SIS source initialization. It averages all
        image rays landing in a source cell and therefore avoids brightness
        inflation from multiple images. Learnable lens parameters should be
        optimized through the differentiable forward renderer.
        """
        if image.ndim != 4:
            raise ValueError(f"image must be BCHW, received shape {tuple(image.shape)}")
        if tuple(image.shape[-2:]) != self.info.image_shape:
            raise ValueError(
                f"Expected image shape {self.info.image_shape}, "
                f"received {tuple(image.shape[-2:])}"
            )
        batch, channels, _, _ = image.shape
        beta_x, beta_y = self.lens_equation(
            batch,
            theta_e_arcsec,
            center_x_arcsec,
            center_y_arcsec,
            dtype=image.dtype,
            device=image.device,
        )
        source_h, source_w = self.info.source_shape
        scale = self.info.source_pixel_scale_arcsec

        u = beta_x / scale + (source_w - 1) / 2.0
        v = beta_y / scale + (source_h - 1) / 2.0
        x0 = torch.floor(u).to(torch.long)
        y0 = torch.floor(v).to(torch.long)
        du = u - x0.to(u.dtype)
        dv = v - y0.to(v.dtype)

        if mask is None:
            ray_mask = torch.ones(
                (batch, 1, *self.info.image_shape),
                device=image.device,
                dtype=image.dtype,
            )
        else:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[0] != batch or tuple(mask.shape[-2:]) != self.info.image_shape:
                raise ValueError("mask must have shape Bx1xH_imagexW_image")
            ray_mask = mask.to(device=image.device, dtype=image.dtype)
            if ray_mask.shape[1] != 1:
                ray_mask = ray_mask.amax(dim=1, keepdim=True)

        pixels = source_h * source_w
        rays = self.info.image_shape[0] * self.info.image_shape[1]
        output = image.new_zeros((batch, channels, pixels))
        coverage = image.new_zeros((batch, 1, pixels))
        image_flat = image.reshape(batch, channels, rays)
        mask_flat = ray_mask.reshape(batch, 1, rays)

        neighbours = (
            (x0, y0, (1.0 - du) * (1.0 - dv)),
            (x0 + 1, y0, du * (1.0 - dv)),
            (x0, y0 + 1, (1.0 - du) * dv),
            (x0 + 1, y0 + 1, du * dv),
        )
        for x_index, y_index, weight in neighbours:
            valid = (
                (x_index >= 0)
                & (x_index < source_w)
                & (y_index >= 0)
                & (y_index < source_h)
            )
            safe_x = x_index.clamp(0, source_w - 1)
            safe_y = y_index.clamp(0, source_h - 1)
            flat_index = (safe_y * source_w + safe_x).reshape(batch, 1, rays)
            weight_flat = (
                weight.reshape(batch, 1, rays)
                * valid.reshape(batch, 1, rays).to(image.dtype)
                * mask_flat
            )
            output.scatter_add_(
                2,
                flat_index.expand(-1, channels, -1),
                image_flat * weight_flat,
            )
            coverage.scatter_add_(2, flat_index, weight_flat)

        source = output / coverage.clamp_min(self.eps)
        source = torch.where(coverage > 0, source, torch.zeros_like(source))
        return (
            source.reshape(batch, channels, source_h, source_w),
            coverage.reshape(batch, 1, source_h, source_w),
        )


def radial_profile(image: torch.Tensor) -> torch.Tensor:
    """Return the azimuthal mean of a 2D image around its geometric centre."""
    if image.ndim != 2:
        raise ValueError("radial_profile expects a 2D image")
    height, width = image.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=image.device),
        torch.arange(width, device=image.device),
        indexing="ij",
    )
    radius = torch.sqrt(
        (x - (width - 1) / 2.0).square()
        + (y - (height - 1) / 2.0).square()
    ).to(torch.long)
    sums = torch.zeros(int(radius.max().item()) + 1, device=image.device, dtype=image.dtype)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, radius.reshape(-1), image.reshape(-1))
    counts.scatter_add_(0, radius.reshape(-1), torch.ones_like(image).reshape(-1))
    return sums / counts.clamp_min(1)
