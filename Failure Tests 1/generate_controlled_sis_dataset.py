"""Stage B: generate a small controlled fixed-SIS dataset.

The script creates source-plane Gaussian mixtures, lenses them with the same
AnalyticSISRenderer used in training, optionally applies a PSF, and area
integrates to the LR detector grid.  Source truth and exact lens metadata are
saved with every sample.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analytic_sis import AnalyticSISRenderer
from psf import apply_psf, build_psf_kernel


def random_source(shape, generator, device):
    y, x = torch.meshgrid(
        torch.arange(shape, device=device),
        torch.arange(shape, device=device),
        indexing="ij",
    )
    image = torch.zeros(shape, shape, device=device)
    components = int(torch.randint(1, 4, (1,), generator=generator).item())
    for _ in range(components):
        cx = (shape - 1) / 2 + float(torch.empty(1).uniform_(-0.12, 0.12, generator=generator)) * shape
        cy = (shape - 1) / 2 + float(torch.empty(1).uniform_(-0.12, 0.12, generator=generator)) * shape
        sx = float(torch.empty(1).uniform_(1.5, 6.0, generator=generator))
        sy = float(torch.empty(1).uniform_(1.5, 6.0, generator=generator))
        amp = float(torch.empty(1).uniform_(0.4, 1.0, generator=generator))
        image += amp * torch.exp(-0.5 * (((x - cx) / sx).square() + ((y - cy) / sy).square()))
    return image / image.max().clamp_min(1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="controlled_sis")
    parser.add_argument("--train-count", type=int, default=800)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--lr-shape", type=int, default=64)
    parser.add_argument("--scale-factor", type=int, default=2)
    parser.add_argument("--lr-pixel-scale", type=float, default=0.101)
    parser.add_argument("--theta-e", type=float, default=0.75)
    parser.add_argument("--psf-type", choices=["none", "gaussian", "fits"], default="none")
    parser.add_argument("--psf-path", default=None)
    parser.add_argument("--psf-source-pixscale", type=float, default=None)
    parser.add_argument("--psf-fwhm", type=float, default=0.16)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    hr_shape = args.lr_shape * args.scale_factor
    hr_scale = args.lr_pixel_scale / args.scale_factor
    renderer = AnalyticSISRenderer(hr_shape, hr_scale, hr_shape, hr_scale).to(device)
    psf = build_psf_kernel(
        psf_type=args.psf_type,
        fwhm_arcsec=args.psf_fwhm,
        pixscale_arcsec=hr_scale,
        path=args.psf_path,
        source_pixscale_arcsec=args.psf_source_pixscale,
        device=device,
    )

    root = Path(args.output_dir)
    metadata = {
        "lens_model": "SIS",
        "theta_e_arcsec": args.theta_e,
        "lr_shape": args.lr_shape,
        "hr_shape": hr_shape,
        "lr_pixel_scale_arcsec": args.lr_pixel_scale,
        "hr_pixel_scale_arcsec": hr_scale,
        "psf_type": args.psf_type,
        "psf_path": args.psf_path,
        "psf_source_pixscale_arcsec": args.psf_source_pixscale,
        "noise_std": args.noise_std,
        "seed": args.seed,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    for split, count in (("train", args.train_count), ("val", args.val_count)):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            source = random_source(hr_shape, generator, device)[None, None]
            intrinsic = renderer(source, args.theta_e)
            convolved = apply_psf(intrinsic, psf)
            lr = F.interpolate(convolved, size=(args.lr_shape, args.lr_shape), mode="area")
            if args.noise_std > 0:
                lr = lr + args.noise_std * torch.randn_like(lr)
            lr = lr.clamp_min(0)
            np.savez_compressed(
                directory / f"sample_{index:06d}.npz",
                lr_image=lr[0, 0].cpu().numpy().astype(np.float32),
                hr_source=source[0, 0].cpu().numpy().astype(np.float32),
                intrinsic_hr=intrinsic[0, 0].cpu().numpy().astype(np.float32),
                theta_E=np.float32(args.theta_e),
            )
    print(f"Wrote controlled SIS dataset to {root}")


if __name__ == "__main__":
    main()