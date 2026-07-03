"""Differentiable parametric lens models used by the SR training pipeline.

Stage 3 introduces a learnable Singular Isothermal Sphere (SIS) Einstein
radius. The original project encodes a fixed SIS into precomputed sparse grid
mappings. Those mappings do not change when a parameter changes and therefore
cannot provide gradients with respect to the Einstein radius.

This module supplies a differentiable forward renderer based on the lens
equation and ``torch.nn.functional.grid_sample``. The original conservative
sparse grid is still used to initialize/reconstruct the LR source; the
learnable SIS renderer is used for the forward relensing step so that the LR
re-degradation loss can optimize the Einstein radius.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSIS(nn.Module):
    """Singular Isothermal Sphere with a bounded learnable Einstein radius.

    Parameters
    ----------
    theta_e_init:
        Initial Einstein radius in arcseconds. This should match the SIS used
        to generate the fixed backward sparse mapping as closely as possible.
    target_resolution:
        Pixel scale of the HR source/image grid in arcsec/pixel.
    target_shape:
        Height and width of the square HR source/image grid.
    theta_e_min, theta_e_max:
        Physical optimization bounds in arcseconds.
    learnable:
        If False, the renderer behaves as a fixed differentiable SIS.
    interpolation_mode:
        Sampling mode used by ``grid_sample``. ``bilinear`` gives useful
        gradients with respect to the Einstein radius.
    """

    def __init__(
        self,
        theta_e_init: float,
        target_resolution: float,
        target_shape: int,
        theta_e_min: float = 0.05,
        theta_e_max: float = 4.0,
        learnable: bool = True,
        interpolation_mode: str = "bilinear",
    ) -> None:
        super().__init__()

        if target_resolution <= 0:
            raise ValueError("target_resolution must be positive")
        if target_shape < 2:
            raise ValueError("target_shape must be at least 2")
        if theta_e_min <= 0 or theta_e_max <= theta_e_min:
            raise ValueError("Require 0 < theta_e_min < theta_e_max")
        if not (theta_e_min < theta_e_init < theta_e_max):
            raise ValueError(
                "theta_e_init must lie strictly inside "
                f"({theta_e_min}, {theta_e_max}); got {theta_e_init}"
            )
        if interpolation_mode not in {"bilinear", "nearest", "bicubic"}:
            raise ValueError("interpolation_mode must be bilinear, nearest, or bicubic")

        self.target_resolution = float(target_resolution)
        self.target_shape = int(target_shape)
        self.theta_e_min = float(theta_e_min)
        self.theta_e_max = float(theta_e_max)
        self.interpolation_mode = interpolation_mode

        # Bounded parameterization:
        # theta_E = min + (max-min) * sigmoid(raw_theta_e)
        fraction = (float(theta_e_init) - self.theta_e_min) / (
            self.theta_e_max - self.theta_e_min
        )
        fraction = min(max(fraction, 1e-6), 1.0 - 1e-6)
        raw_init = math.log(fraction / (1.0 - fraction))
        self.raw_theta_e = nn.Parameter(
            torch.tensor(raw_init, dtype=torch.float32),
            requires_grad=bool(learnable),
        )

        # Match the angular convention already used by DifferentiableLensing.
        half_arcsec_bound = self.target_resolution * self.target_shape / 2.0
        self.half_arcsec_bound = float(half_arcsec_bound)
        coordinates = torch.linspace(
            -half_arcsec_bound,
            half_arcsec_bound,
            self.target_shape,
            dtype=torch.float32,
        )
        theta_y, theta_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("theta_x", theta_x)
        self.register_buffer("theta_y", theta_y)

    def einstein_radius(self) -> torch.Tensor:
        """Return the bounded Einstein radius in arcseconds."""
        scale = self.theta_e_max - self.theta_e_min
        return self.theta_e_min + scale * torch.sigmoid(self.raw_theta_e)

    def deflection(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return SIS deflection components on the HR image-plane grid.

        For an SIS,

            alpha(theta) = theta_E * theta / |theta|.
        """
        radius = torch.sqrt(self.theta_x.square() + self.theta_y.square() + 1e-12)
        theta_e = self.einstein_radius()
        alpha_x = theta_e * self.theta_x / radius
        alpha_y = theta_e * self.theta_y / radius
        return alpha_x, alpha_y

    def source_coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map image-plane coordinates theta to source-plane coordinates beta."""
        alpha_x, alpha_y = self.deflection()
        beta_x = self.theta_x - alpha_x
        beta_y = self.theta_y - alpha_y
        return beta_x, beta_y

    def sampling_grid(self) -> torch.Tensor:
        """Return normalized source coordinates for ``grid_sample``.

        ``grid_sample`` expects the last dimension in (x, y) order and values
        in [-1, 1]. Coordinates outside this range are zero padded.
        """
        beta_x, beta_y = self.source_coordinates()
        grid_x = beta_x / self.half_arcsec_bound
        grid_y = beta_y / self.half_arcsec_bound
        return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    def forward(self, source_image: torch.Tensor) -> torch.Tensor:
        """Forward-lens a source-plane BCHW image through the SIS model."""
        if source_image.ndim != 4:
            raise ValueError(
                f"source_image must have shape [B, C, H, W]; got {tuple(source_image.shape)}"
            )
        if source_image.shape[-2:] != (self.target_shape, self.target_shape):
            raise ValueError(
                "LearnableSIS grid and source image size differ: "
                f"expected {(self.target_shape, self.target_shape)}, "
                f"got {tuple(source_image.shape[-2:])}"
            )

        grid = self.sampling_grid().to(
            device=source_image.device,
            dtype=source_image.dtype,
        )
        grid = grid.expand(source_image.shape[0], -1, -1, -1)
        return F.grid_sample(
            source_image,
            grid,
            mode=self.interpolation_mode,
            padding_mode="zeros",
            align_corners=True,
        )

    def prior_loss(self, reference_theta_e: float) -> torch.Tensor:
        """Quadratic prior around a reference Einstein radius."""
        reference = torch.as_tensor(
            float(reference_theta_e),
            device=self.raw_theta_e.device,
            dtype=self.raw_theta_e.dtype,
        )
        return (self.einstein_radius() - reference).square()