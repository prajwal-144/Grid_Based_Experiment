"""Package freshly regenerated raw sparse mappings with explicit SIS metadata."""
import argparse
from pathlib import Path
import torch
from mapping_io import save_mapping_bundle

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--mapping-role", required=True, choices=["backward", "to_log", "forward_from_log", "from_log"])
p.add_argument("--theta-e-arcsec", type=float, required=True)
p.add_argument("--lr-pixel-scale-arcsec", type=float, required=True)
p.add_argument("--hr-pixel-scale-arcsec", type=float, required=True)
p.add_argument("--image-shape", type=int, required=True)
p.add_argument("--target-shape", type=int, required=True)
p.add_argument("--center-x-arcsec", type=float, default=0.0)
p.add_argument("--center-y-arcsec", type=float, default=0.0)
args = p.parse_args()
obj = torch.load(args.input, map_location="cpu")
matrix = obj.get("matrix") if isinstance(obj, dict) else obj
if matrix is None or not matrix.is_sparse:
    raise TypeError("Input must contain a sparse COO tensor")
valid = {args.image_shape ** 2, args.target_shape ** 2}
if matrix.shape[0] not in valid or matrix.shape[1] not in valid:
    raise ValueError(f"Unexpected matrix shape {tuple(matrix.shape)} for declared grids")
metadata = {
    "lens_model": "SIS", "theta_e_arcsec": args.theta_e_arcsec,
    "lr_pixel_scale_arcsec": args.lr_pixel_scale_arcsec,
    "hr_pixel_scale_arcsec": args.hr_pixel_scale_arcsec,
    "image_shape": args.image_shape, "target_shape": args.target_shape,
    "center_x_arcsec": args.center_x_arcsec, "center_y_arcsec": args.center_y_arcsec,
    "mapping_role": args.mapping_role, "source_file": str(Path(args.input)),
}
save_mapping_bundle(args.output, matrix, metadata)
print(f"Saved {args.output}: shape={tuple(matrix.shape)} metadata={metadata}")
