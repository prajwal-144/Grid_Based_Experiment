"""Small helpers for saving/loading sparse lens mappings with geometry metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

REQUIRED_METADATA = {
    "lens_model",
    "theta_e_arcsec",
    "lr_pixel_scale_arcsec",
    "hr_pixel_scale_arcsec",
    "image_shape",
    "target_shape",
    "center_x_arcsec",
    "center_y_arcsec",
    "mapping_role",
}


def save_mapping_bundle(path: str | Path, matrix: torch.Tensor, metadata: Dict[str, Any]) -> None:
    """Save a sparse matrix together with the geometry that generated it."""
    missing = REQUIRED_METADATA.difference(metadata)
    if missing:
        raise ValueError(f"Missing mapping metadata: {sorted(missing)}")
    if not matrix.is_sparse:
        raise TypeError("matrix must be a sparse COO tensor")
    bundle = {"matrix": matrix.coalesce().cpu(), "metadata": dict(metadata)}
    torch.save(bundle, Path(path))


def load_mapping_bundle(path: str | Path, *, device=None, expected: Dict[str, Any] | None = None):
    """Load a bundle and optionally reject geometry mismatches."""
    bundle = torch.load(Path(path), map_location=device)
    if not isinstance(bundle, dict) or "matrix" not in bundle or "metadata" not in bundle:
        raise ValueError(f"{path} is a legacy raw tensor, not a metadata bundle")
    metadata = bundle["metadata"]
    missing = REQUIRED_METADATA.difference(metadata)
    if missing:
        raise ValueError(f"Bundle metadata incomplete: {sorted(missing)}")
    for key, value in (expected or {}).items():
        if key in metadata and metadata[key] != value:
            raise ValueError(f"Mapping mismatch for {key}: stored={metadata[key]!r}, requested={value!r}")
    return bundle["matrix"].to(device) if device is not None else bundle["matrix"], metadata
