"""Fixed-SIS physics utilities for magnification-aware source SR.

This module deliberately keeps the lens model fixed.  It adds two constraints on
an existing source-plane SR network:

1. A magnification-adaptive *soft source grid*.  A fixed SIS magnification map is
   back-projected to the source plane and used to blend fine, medium, and coarse
   source representations.  High-magnification regions retain fine resolution;
   low-magnification regions are represented more conservatively.
2. Source-plane surface-brightness consistency.  Image pixels that the fixed
   lens mapping sends to the same source cell must agree in surface brightness.

No intensity is multiplied by magnification.  Lensing conserves surface
brightness; magnification is used only as an information/resolution map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def _as_bchw(x: torch.Tensor) -> torch.Tensor:
    """Convert a 2D/3D/4D image tensor to BCHW."""
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected 2D, 3D, or 4D tensor; got {tuple(x.shape)}")


def _square_shape(pixel_count: int, name: str) -> int:
    side = int(round(math.sqrt(int(pixel_count))))
    if side * side != int(pixel_count):
        raise ValueError(f"{name} pixel count must be a perfect square; got {pixel_count}")
    return side


def _normalized_sparse_rows(mapping: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a row-normalized COO mapping, row sums, and effective counts.

    The effective number of contributors is

        n_eff = (sum_p w_qp)^2 / sum_p w_qp^2.

    It equals the actual number of contributors for equal non-zero weights and
    remains useful when the conservative overlap weights are unequal.
    """
    if not mapping.is_sparse:
        raise TypeError("mapping must be a sparse COO tensor")
    mapping = mapping.coalesce()
    indices = mapping.indices()
    values = mapping.values()
    if values.numel() == 0:
        raise ValueError("mapping contains no non-zero entries")
    if torch.any(values < -1e-8):
        raise ValueError("mapping contains negative overlap weights")
    values = values.clamp_min(0.0)

    rows = indices[0]
    q = mapping.shape[0]
    row_sum = torch.zeros(q, device=values.device, dtype=values.dtype)
    row_sq_sum = torch.zeros_like(row_sum)
    row_sum.scatter_add_(0, rows, values)
    row_sq_sum.scatter_add_(0, rows, values.square())
    normalized_values = values / row_sum[rows].clamp_min(EPS)
    normalized = torch.sparse_coo_tensor(
        indices,
        normalized_values,
        size=mapping.shape,
        device=mapping.device,
        dtype=mapping.dtype,
    ).coalesce()
    n_eff = row_sum.square() / row_sq_sum.clamp_min(EPS)
    return normalized, row_sum, n_eff


def apply_sparse_mapping(image: torch.Tensor, mapping: torch.Tensor, output_shape: Tuple[int, int]) -> torch.Tensor:
    """Apply a sparse QxP mapping to a BCHW image and reshape to output_shape."""
    image = _as_bchw(image)
    mapping = mapping.coalesce()
    batch, channels, height, width = image.shape
    if height * width != mapping.shape[1]:
        raise ValueError(
            "Input image pixels do not match mapping columns: "
            f"{height * width} vs {mapping.shape[1]}"
        )
    if output_shape[0] * output_shape[1] != mapping.shape[0]:
        raise ValueError(
            "output_shape does not match mapping rows: "
            f"{output_shape} vs {mapping.shape[0]}"
        )
    flat = image.reshape(batch * channels, height * width).T
    mapped = torch.sparse.mm(mapping, flat)
    return mapped.T.contiguous().reshape(batch, channels, *output_shape)


@dataclass
class SISInformationMaps:
    """Static maps used by the fixed-SIS magnification stage."""

    signed_magnification_image: torch.Tensor
    absolute_magnification_image: torch.Tensor
    image_information: torch.Tensor
    source_information_lr: torch.Tensor
    source_coverage_lr: torch.Tensor
    source_effective_contributors_lr: torch.Tensor


def fixed_sis_magnification(
    image_shape: int,
    pixel_scale_arcsec: float,
    theta_e_arcsec: float,
    mu_clip: float = 20.0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute signed and clipped absolute magnification for a centered SIS.

    For an SIS, the radial and tangential Jacobian eigenvalues are

        lambda_r = 1,
        lambda_t = 1 - theta_E / r,

    hence mu = 1 / (1 - theta_E / r).  The critical curve at r=theta_E is
    singular, so the absolute map is clipped for numerical use.

    The coordinate convention intentionally matches ``DifferentiableLensing``:
    a linspace spanning +/- (pixel_scale * image_shape / 2).
    """
    if image_shape < 2:
        raise ValueError("image_shape must be at least 2")
    if pixel_scale_arcsec <= 0 or theta_e_arcsec <= 0:
        raise ValueError("pixel scale and Einstein radius must be positive")
    if mu_clip <= 1.0:
        raise ValueError("mu_clip must be greater than 1")

    half_extent = float(pixel_scale_arcsec) * int(image_shape) / 2.0
    coords = torch.linspace(-half_extent, half_extent, image_shape, device=device, dtype=dtype)
    theta_y, theta_x = torch.meshgrid(coords, coords, indexing="ij")
    radius = torch.sqrt(theta_x.square() + theta_y.square()).clamp_min(EPS)
    det_a = 1.0 - float(theta_e_arcsec) / radius

    sign = torch.where(det_a >= 0, torch.ones_like(det_a), -torch.ones_like(det_a))
    det_safe = sign * det_a.abs().clamp_min(1.0 / float(mu_clip))
    signed_mu = 1.0 / det_safe
    absolute_mu = signed_mu.abs().clamp_max(float(mu_clip))
    return signed_mu[None, None], absolute_mu[None, None]


def magnification_to_information(
    absolute_mu: torch.Tensor,
    mu_clip: float,
    unlensed_floor: float = 1.0,
) -> torch.Tensor:
    """Compress |mu| to a [0,1] lens-assisted resolution score.

    Values at or below the unlensed reference |mu|=1 receive zero *additional*
    resolution credit.  A logarithm prevents a few near-critical pixels from
    dominating the entire map.
    """
    if mu_clip <= unlensed_floor:
        raise ValueError("mu_clip must be greater than unlensed_floor")
    absolute_mu = _as_bchw(absolute_mu).clamp_min(0.0).clamp_max(float(mu_clip))
    low = math.log1p(float(unlensed_floor))
    high = math.log1p(float(mu_clip))
    score = (torch.log1p(absolute_mu) - low) / max(high - low, EPS)
    return score.clamp(0.0, 1.0)


def build_fixed_sis_information_maps(
    backward_mapping: torch.Tensor,
    image_shape: int,
    pixel_scale_arcsec: float,
    theta_e_arcsec: float,
    mu_clip: float = 20.0,
) -> SISInformationMaps:
    """Build image- and source-plane information maps for the fixed SIS.

    The existing conservative backward sparse operator is reused.  The
    magnification-derived *information score* is row-averaged into source cells;
    it is never applied as an intensity multiplier.
    """
    mapping, row_sum, n_eff = _normalized_sparse_rows(backward_mapping)
    expected_image_side = _square_shape(mapping.shape[1], "image")
    source_side = _square_shape(mapping.shape[0], "source")
    if expected_image_side != int(image_shape):
        raise ValueError(
            f"image_shape={image_shape} does not match backward mapping ({expected_image_side})"
        )

    signed_mu, absolute_mu = fixed_sis_magnification(
        image_shape=image_shape,
        pixel_scale_arcsec=pixel_scale_arcsec,
        theta_e_arcsec=theta_e_arcsec,
        mu_clip=mu_clip,
        device=backward_mapping.device,
        dtype=backward_mapping.dtype,
    )
    image_information = magnification_to_information(absolute_mu, mu_clip=mu_clip)
    source_information = apply_sparse_mapping(
        image_information,
        mapping,
        output_shape=(source_side, source_side),
    )
    return SISInformationMaps(
        signed_magnification_image=signed_mu,
        absolute_magnification_image=absolute_mu,
        image_information=image_information,
        source_information_lr=source_information.clamp(0.0, 1.0),
        source_coverage_lr=row_sum.reshape(1, 1, source_side, source_side),
        source_effective_contributors_lr=n_eff.reshape(1, 1, source_side, source_side),
    )


def _coarsen_and_restore(source: torch.Tensor, factor: int) -> torch.Tensor:
    """Average-pool by factor and interpolate back to the original grid."""
    if factor <= 1:
        return source
    height, width = source.shape[-2:]
    if factor > min(height, width):
        raise ValueError(f"coarsening factor {factor} exceeds source size {(height, width)}")
    pooled = F.avg_pool2d(source, kernel_size=factor, stride=factor)
    return F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)


def magnification_adaptive_source_grid(
    fine_source: torch.Tensor,
    source_information_hr: torch.Tensor,
    medium_factor: int = 2,
    coarse_factor: int = 4,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Construct a differentiable soft multi-resolution source grid."""
    fine_source = _as_bchw(fine_source)
    info = _as_bchw(source_information_hr).to(
        device=fine_source.device,
        dtype=fine_source.dtype,
    )
    if info.shape[-2:] != fine_source.shape[-2:]:
        info = F.interpolate(info, size=fine_source.shape[-2:], mode="bilinear", align_corners=False)
    if info.shape[0] == 1 and fine_source.shape[0] > 1:
        info = info.expand(fine_source.shape[0], -1, -1, -1)
    if info.shape[1] == 1 and fine_source.shape[1] > 1:
        info = info.expand(-1, fine_source.shape[1], -1, -1)
    if info.shape != fine_source.shape:
        raise ValueError(f"information map cannot broadcast to source: {info.shape} vs {fine_source.shape}")

    info = info.clamp(0.0, 1.0)
    medium = _coarsen_and_restore(fine_source, int(medium_factor))
    coarse = _coarsen_and_restore(fine_source, int(coarse_factor))
    w_fine = info.square()
    w_medium = 2.0 * info * (1.0 - info)
    w_coarse = (1.0 - info).square()
    adaptive = w_fine * fine_source + w_medium * medium + w_coarse * coarse
    return adaptive, {
        "fine_weight": w_fine,
        "medium_weight": w_medium,
        "coarse_weight": w_coarse,
        "medium_source": medium,
        "coarse_source": coarse,
    }


def magnification_adaptive_loss(
    fine_source: torch.Tensor,
    adaptive_source: torch.Tensor,
    source_information_hr: torch.Tensor,
    normalize_by_source_power: bool = True,
) -> torch.Tensor:
    """L_mu: suppress unsupported fine structure in low-information regions."""
    fine_source = _as_bchw(fine_source)
    adaptive_source = _as_bchw(adaptive_source).to(fine_source)
    info = _as_bchw(source_information_hr).to(fine_source)
    if info.shape[-2:] != fine_source.shape[-2:]:
        info = F.interpolate(info, size=fine_source.shape[-2:], mode="bilinear", align_corners=False)
    info = info.expand_as(fine_source).clamp(0.0, 1.0)
    weight = (1.0 - info).square()
    numerator = (weight * (fine_source - adaptive_source).square()).sum()
    denominator = weight.sum().clamp_min(EPS)
    loss = numerator / denominator
    if normalize_by_source_power:
        source_power = (weight * fine_source.square()).sum() / denominator
        loss = loss / source_power.clamp_min(EPS)
    return loss


class SourcePlaneIntensityConsistency(nn.Module):
    """Surface-brightness consistency induced by a fixed backward mapping."""

    def __init__(
        self,
        backward_mapping: torch.Tensor,
        source_shape: Tuple[int, int] | None = None,
        min_effective_contributors: float = 1.5,
        variance_weight: float = 1.0,
        normalize_by_source_power: bool = True,
    ) -> None:
        super().__init__()
        normalized, row_sum, n_eff = _normalized_sparse_rows(backward_mapping)
        source_side = _square_shape(normalized.shape[0], "source")
        image_side = _square_shape(normalized.shape[1], "image")
        source_shape = source_shape or (source_side, source_side)
        if source_shape[0] * source_shape[1] != normalized.shape[0]:
            raise ValueError("source_shape does not match backward mapping rows")
        self.source_shape = tuple(int(v) for v in source_shape)
        self.image_shape = (image_side, image_side)
        self.min_effective_contributors = float(min_effective_contributors)
        self.variance_weight = float(variance_weight)
        self.normalize_by_source_power = bool(normalize_by_source_power)
        self.register_buffer("normalized_mapping", normalized)
        self.register_buffer("row_sum", row_sum)
        self.register_buffer("effective_contributors", n_eff)

    def forward(
        self,
        intrinsic_image_lr: torch.Tensor,
        source_lr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        intrinsic_image_lr = _as_bchw(intrinsic_image_lr)
        source_lr = _as_bchw(source_lr).to(intrinsic_image_lr)
        if intrinsic_image_lr.shape[-2:] != self.image_shape:
            raise ValueError(
                f"Expected intrinsic image shape {self.image_shape}, got {intrinsic_image_lr.shape[-2:]}"
            )
        if source_lr.shape[-2:] != self.source_shape:
            raise ValueError(f"Expected source shape {self.source_shape}, got {source_lr.shape[-2:]}")
        if source_lr.shape[:2] != intrinsic_image_lr.shape[:2]:
            raise ValueError("intrinsic image and source batch/channel dimensions must match")

        mean_source = apply_sparse_mapping(
            intrinsic_image_lr,
            self.normalized_mapping,
            output_shape=self.source_shape,
        )
        second_moment = apply_sparse_mapping(
            intrinsic_image_lr.square(),
            self.normalized_mapping,
            output_shape=self.source_shape,
        )
        variance_map = (second_moment - mean_source.square()).clamp_min(0.0)

        coverage_mask = (self.row_sum > EPS).to(source_lr.dtype).reshape(1, 1, *self.source_shape)
        multi_mask = (
            self.effective_contributors >= self.min_effective_contributors
        ).to(source_lr.dtype).reshape(1, 1, *self.source_shape)
        coverage_mask = coverage_mask.expand_as(source_lr)
        multi_mask = multi_mask.expand_as(source_lr)

        cycle_num = (coverage_mask * (mean_source - source_lr).square()).sum()
        cycle_den = coverage_mask.sum().clamp_min(EPS)
        cycle_loss = cycle_num / cycle_den

        variance_num = (multi_mask * variance_map).sum()
        variance_den = multi_mask.sum().clamp_min(EPS)
        variance_loss = variance_num / variance_den

        if self.normalize_by_source_power:
            source_power = (coverage_mask * source_lr.square()).sum() / cycle_den
            scale = source_power.clamp_min(EPS)
            cycle_loss = cycle_loss / scale
            variance_loss = variance_loss / scale

        total = cycle_loss + self.variance_weight * variance_loss
        return total, {
            "cycle_loss": cycle_loss,
            "variance_loss": variance_loss,
            "backprojected_mean": mean_source,
            "variance_map": variance_map,
            "coverage_mask": coverage_mask,
            "multi_image_mask": multi_mask,
            "effective_contributors": self.effective_contributors.reshape(
                1, 1, *self.source_shape
            ),
            "multi_image_fraction": multi_mask.mean(),
        }
