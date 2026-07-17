"""Save and load sparse lens mappings together with their generating geometry."""
from pathlib import Path
import torch

REQUIRED_METADATA = {"lens_model", "theta_e_arcsec", "lr_pixel_scale_arcsec", "hr_pixel_scale_arcsec", "image_shape", "target_shape", "center_x_arcsec", "center_y_arcsec", "mapping_role"}


def save_mapping_bundle(path, matrix, metadata):
    missing = REQUIRED_METADATA.difference(metadata)
    if missing:
        raise ValueError(f"Missing mapping metadata: {sorted(missing)}")
    if not matrix.is_sparse:
        raise TypeError("matrix must be sparse COO")
    torch.save({"matrix": matrix.coalesce().cpu(), "metadata": dict(metadata)}, Path(path))


def load_mapping_bundle(path, device=None, expected=None):
    bundle = torch.load(Path(path), map_location=device)
    if not isinstance(bundle, dict) or "matrix" not in bundle or "metadata" not in bundle:
        raise ValueError(f"{path} is a legacy raw tensor; regenerate/package it first")
    metadata = bundle["metadata"]
    missing = REQUIRED_METADATA.difference(metadata)
    if missing:
        raise ValueError(f"Incomplete metadata: {sorted(missing)}")
    for key, value in (expected or {}).items():
        if metadata.get(key) != value:
            raise ValueError(f"Mapping mismatch for {key}: stored={metadata.get(key)!r}, requested={value!r}")
    return bundle["matrix"].to(device) if device is not None else bundle["matrix"], metadata
