"""Package freshly regenerated raw sparse mappings with explicit SIS metadata.

Run the existing grid-generation notebooks first, then use this script on each raw
`.pt` tensor. It deliberately requires all geometry values on the command line so
an old matrix cannot silently masquerade as a new one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mapping_io import save_mapping_bundle


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="freshly regenerated raw sparse .pt tensor")
    p.add_argument("--output", required=True, help="metadata bundle output path")
    p.add_argument("--mapping-role", required=True, choices=["backward", "to_log", "forward_from_log", "from_log"])
    p.add_argument("--theta-e-arcsec", type=float, required=True)
    p.add_argument("--lr-pixel-scale-arcsec", type=float, required=True)
    p.add_argument("--hr-pixel-scale-arcsec", type=float, required=True)
    p.add_argument("--image-shape", type=int, required=True)
    p.add_argument("--target-shape", type=int, required=True)
    p.add_argument("--center-x-arcsec", type=float, default=0.0)
    p.add_argument("--center-y-arcsec", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    matrix = torch.load(args.input, map_location="cpu")
    if isinstance(matrix, dict):
        matrix = matrix.get("matrix")
    if matrix is None or not matrix.is_sparse:
        raise TypeError("Input must contain a sparse COO tensor")

    # Basic dimensional checks catch accidentally mixed 64/128-grid products.
    rows, cols = matrix.shape
    valid_pixels = {args.image_shape ** 2, args.target_shape ** 2}
    if rows not in valid_pixels or cols not in valid_pixels:
        raise ValueError(
            f"Unexpected matrix shape {tuple(matrix.shape)} for image={args.image_shape}, target={args.target_shape}"
        )

    metadata = {
        "lens_model": "SIS",
        "theta_e_arcsec": args.theta_e_arcsec,
        "lr_pixel_scale_arcsec": args.lr_pixel_scale_arcsec,
        "hr_pixel_scale_arcsec": args.hr_pixel_scale_arcsec,
        "image_shape": args.image_shape,
        "target_shape": args.target_shape,
        "center_x_arcsec": args.center_x_arcsec,
        "center_y_arcsec": args.center_y_arcsec,
        "mapping_role": args.mapping_role,
        "source_file": str(Path(args.input)),
    }
    save_mapping_bundle(args.output, matrix, metadata)
    print(f"Saved {args.output} with shape={tuple(matrix.shape)} and metadata={metadata}")


if __name__ == "__main__":
    main()
