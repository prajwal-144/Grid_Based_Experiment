"""Stage A: deterministic physics checks for AnalyticSISRenderer."""

import argparse
import json
from pathlib import Path

import torch

from analytic_sis import AnalyticSISRenderer, radial_profile


def gaussian(shape, sigma, device):
    h, w = shape
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    r2 = (x - (w - 1) / 2.0).square() + (y - (h - 1) / 2.0).square()
    return torch.exp(-0.5 * r2 / float(sigma) ** 2)[None, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, default=128)
    parser.add_argument("--pixel-scale", type=float, default=0.0505)
    parser.add_argument("--theta-e", type=float, default=0.75)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--output", default="analytic_sis_validation.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    renderer = AnalyticSISRenderer(
        image_shape=args.shape,
        image_pixel_scale_arcsec=args.pixel_scale,
        source_shape=args.shape,
        source_pixel_scale_arcsec=args.pixel_scale,
    ).to(device)

    ones = torch.ones(1, 1, args.shape, args.shape, device=device)
    rendered_ones, _, mask = renderer(ones, args.theta_e, return_grid=True)
    covered = rendered_ones[mask]
    constant_mean = float(covered.mean())
    constant_std = float(covered.std())

    expected_radius = args.theta_e / args.pixel_scale
    width_results = []
    for sigma in args.sigmas:
        source = gaussian((args.shape, args.shape), sigma, device)
        image = renderer(source, args.theta_e)
        profile = radial_profile(image[0, 0])
        peak = int(torch.argmax(profile).item())
        width_results.append({"sigma": sigma, "peak_radius": peak})

    peak_values = [item["peak_radius"] for item in width_results]
    report = {
        "expected_radius_pixels": expected_radius,
        "constant_mean": constant_mean,
        "constant_std": constant_std,
        "width_results": width_results,
        "peak_drift_pixels": max(peak_values) - min(peak_values),
    }
    report["passed"] = (
        abs(constant_mean - 1.0) < 0.01
        and constant_std < 0.01
        and report["peak_drift_pixels"] <= 1
        and all(abs(value - expected_radius) <= 1.5 for value in peak_values)
    )
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()