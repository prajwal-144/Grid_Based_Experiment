# External imports
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Internal imports
import data
from differentiable_lensing import DifferentiableLensing
from metrics import (
    MetricTracker,
    flux_conservation_error,
    lr_redegradation_error,
    source_plane_reconstruction_consistency,
)
from psf import apply_psf, build_psf_kernel
from sisr import SISR


# === small helpers ===========================================================
def strtobool(x):
    """Convert a string-like boolean into a Python bool."""
    return x.lower().strip() in {"true", "1", "yes", "y"}


def wmse_loss(y1, y2, w):
    """Weighted MSE wrapper: returns mean((y1-y2)^2 * w)."""
    return torch.mean((y1 - y2) ** 2 * w)


def interpolate_image(x, scale_factor, mode):
    """F.interpolate wrapper with correct align_corners handling."""
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)


# === argument parsing =======================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--lr", type=float, default=2.5e-4,
                        help="the LR of the optimizer")
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed of the experiment")
    parser.add_argument("--epochs", type=int, default=100,
                        help="total timesteps of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if False, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if True, cuda will be enabled when possible")
    parser.add_argument("--log-train", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if True, training will be logged with Tensorboard")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="the number of images loaded at any moment")
    parser.add_argument("--vdl-weight", type=float, default=0.0,
                        help="weight of the VDL loss")

    # Performance / architecture options
    parser.add_argument("--resolution", type=float, required=True,
                        help="LR arcsec/pixel resolution of the input images")
    parser.add_argument("--magnification", type=int, default=2,
                        help="magnification value achieved by the SR network")
    parser.add_argument("--n-mag", type=int, default=1,
                        help="number of times the magnification value is applied by the SR network")
    parser.add_argument("--residual-depth", type=int, default=3,
                        help="the number of residual layers in the SR network")
    parser.add_argument("--in-channels", type=int, default=2,
                        help="the number of input channels: reconstructed source + LR image")
    parser.add_argument("--latent-space-size", type=int, default=64,
                        help="the number of neurons in the latent space(s)")
    parser.add_argument("--image-shape", type=int, default=64,
                        help="the shape of the square LR image in one axis")
    parser.add_argument("--theta-e", type=float, default=0.75,
                        help="kept for compatibility; Stage 1-2 uses fixed precomputed SIS matrices")

    # Stage 2: modular PSF / observation operator options.
    # `fits` is the intended option for actual HSC/HST PSF kernels.
    parser.add_argument("--psf-type", type=str, default="fits",
                        choices=["none", "gaussian", "moffat", "empirical", "fits"],
                        help="PSF model applied before LR re-degradation")
    parser.add_argument("--psf-fwhm-arcsec", type=float, default=0.16,
                        help="PSF FWHM in arcseconds for analytic Gaussian/Moffat kernels")
    parser.add_argument("--psf-beta", type=float, default=4.765,
                        help="Moffat beta parameter when --psf-type=moffat")
    parser.add_argument("--psf-kernel-size", type=int, default=None,
                        help="optional odd analytic PSF kernel width in pixels")
    parser.add_argument("--psf-path", type=str, default=None,
                        help="path to empirical PSF kernel: .fits, .fit, .fts, .npy, .npz, .pt, or .pth")
    parser.add_argument("--psf-ellipticity-q", type=float, default=1.0,
                        help="minor/major axis ratio for analytic PSFs")
    parser.add_argument("--psf-angle-deg", type=float, default=0.0,
                        help="counter-clockwise analytic PSF major-axis angle in degrees")
    parser.add_argument("--psf-fits-hdu", type=int, default=0,
                        help="FITS HDU index containing the PSF image; ignored if --psf-fits-extname is set")
    parser.add_argument("--psf-fits-extname", type=str, default=None,
                        help="optional FITS extension name containing the PSF image")
    parser.add_argument("--psf-fits-crop-size", type=int, default=None,
                        help="optional odd center-crop size for large empirical/FITS PSF stamps")
    parser.add_argument("--psf-source-pixscale-arcsec", type=float, default=None,
                        help=(
                            "native arcsec/pixel scale of the FITS/empirical PSF kernel. "
                            "If supplied, the PSF is resampled to the HR target grid before convolution. "
                            "Example: HSC native PSF at 0.168 with target_resolution 0.084 uses scale factor 2."
                        ))
    parser.add_argument("--downsample-mode", type=str, default="area", choices=["nearest", "bilinear", "bicubic", "area"],
                        help="downsampling mode for observation consistency")
    parser.add_argument("--upsample-mode", type=str, default="bilinear", choices=["nearest", "bilinear", "bicubic"],
                        help="upsampling mode used for interpolation reference in VDL")
    args = parser.parse_args()

    # derived args
    args.effective_magnification = int(args.magnification ** args.n_mag)
    args.target_shape = args.image_shape * args.effective_magnification
    args.target_resolution = args.resolution / args.effective_magnification
    args.device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print("[SYS] Device is set to %s" % args.device)
    return args


def build_epoch_metrics(
    total_loss,
    image_reconstruction_loss,
    source_reconstruction_loss,
    downsampled_image,
    lr_image,
    downsampled_source,
    reconstructed_source,
    source_convergence_map,
    image_convergence_map,
):
    """Compute Stage-1 physics-aware metrics for one batch."""
    return {
        "loss_total": total_loss,
        "loss_image": image_reconstruction_loss,
        "loss_source": source_reconstruction_loss,
        "metric_lr_redegradation": lr_redegradation_error(
            downsampled_image,
            lr_image,
            weight=source_convergence_map,
        ),
        "metric_source_consistency": source_plane_reconstruction_consistency(
            downsampled_source,
            reconstructed_source,
            weight=image_convergence_map,
        ),
        "metric_flux_lr": flux_conservation_error(lr_image, downsampled_image),
        "metric_flux_source": flux_conservation_error(reconstructed_source, downsampled_source),
    }


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}"

    # seeds for repeatability
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device)
    BATCH_SIZE = args.batch_size

    # --- dataset loading ---------------------------------------------------
    train_dataset_no_sub = data.LensingDataset("train/", ["no_sub"], 5000)
    val_dataset_no_sub = data.LensingDataset("val/", ["no_sub"], 2000)

    train_dataset_axion = data.LensingDataset("train/", ["axion"], 5000)
    val_dataset_axion = data.LensingDataset("val/", ["axion"], 2000)

    train_dataset_cdm = data.LensingDataset("train/", ["cdm"], 5000)
    val_dataset_cdm = data.LensingDataset("val/", ["cdm"], 2000)

    train_dataset = torch.utils.data.ConcatDataset([train_dataset_no_sub, train_dataset_axion, train_dataset_cdm])
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_no_sub, val_dataset_axion, val_dataset_cdm])

    train_dataset, train_rest = torch.utils.data.random_split(train_dataset, [0.34, 0.66])
    val_dataset, val_rest = torch.utils.data.random_split(val_dataset, [0.34, 0.66])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=min(8, os.cpu_count()))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=min(8, os.cpu_count()))

    # --- model / modules ----------------------------------------------------
    model = SISR(
        magnification=args.magnification,
        n_mag=args.n_mag,
        residual_depth=args.residual_depth,
        in_channels=args.in_channels,
        latent_channel_count=args.latent_space_size,
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    lensing_module = DifferentiableLensing(
        device=device,
        alpha=None,
        target_resolution=args.target_resolution,
        target_shape=args.target_shape,
    ).to(args.device)

    # TensorBoard logging
    writer = None
    if args.log_train:
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])),
        )

    # --- load precomputed sparse mappings and maps --------------------------
    # Stage 1-2 deliberately uses the original fixed SIS lensing pipeline.
    # The SIS parameters are already encoded in these sparse matrices, so this
    # script does not instantiate the learnable Stage-3 SIS model.
    cross_grid_to_log = torch.load("scatter_to_log_128.pt").to(args.device)
    cross_grid_forward_from_log = torch.load("forward_from_log_128.pt").to(args.device)
    cross_grid_from_log = torch.load("scatter_from_log_128.pt").to(args.device)
    cross_grid_backward = torch.load("sparse_grid_fracs_euclid_backward.pt").to(args.device)

    # convergence maps
    source_convergence_map = torch.load("source_convergence_map.pt").to(args.device)
    image_convergence_map = torch.load("image_convergence_map.pt").to(args.device)

    # --- PSF kernel setup ---------------------------------------------------
    # If --psf-type=fits, the FITS PSF is loaded, optionally center-cropped,
    # normalized, and resampled from --psf-source-pixscale-arcsec to the HR grid
    # pixel scale args.target_resolution before it is applied to the HR image.
    psf_kernel = build_psf_kernel(
        psf_type=args.psf_type,
        fwhm_arcsec=args.psf_fwhm_arcsec,
        pixscale_arcsec=args.target_resolution,
        beta=args.psf_beta,
        kernel_size=args.psf_kernel_size,
        path=args.psf_path,
        ellipticity_q=args.psf_ellipticity_q,
        angle_deg=args.psf_angle_deg,
        fits_hdu=args.psf_fits_hdu,
        fits_extname=args.psf_fits_extname,
        fits_crop_size=args.psf_fits_crop_size,
        source_pixscale_arcsec=args.psf_source_pixscale_arcsec,
        device=args.device,
    )
    if psf_kernel is None:
        print("[SYS] PSF disabled (--psf-type=none).")
    else:
        print("[SYS] Using %s PSF: shape=%s sum=%.6f" % (args.psf_type, tuple(psf_kernel.shape), psf_kernel.sum().item()))
        print("[SYS] HR target pixel scale for PSF convolution: %.6f arcsec/pixel" % args.target_resolution)
        if args.psf_source_pixscale_arcsec is not None:
            print("[SYS] Native PSF pixel scale: %.6f arcsec/pixel" % args.psf_source_pixscale_arcsec)

    # --- training loop ------------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        train_metrics = MetricTracker()

        for i, lr_image in enumerate(tqdm(train_dataloader, desc=f"Training epoch {epoch+1} of {args.epochs}")):
            lr_image = lr_image.float().to(device)

            # Source reconstruction through backward lensing using the fixed SIS sparse mapping.
            reconstructed_source = lensing_module.cross_grid_fill(lr_image, [cross_grid_backward])

            # Upscaling using a neural network: concatenate source and LR image along channels.
            model_feed = torch.cat([reconstructed_source, lr_image], dim=1)
            upscaled_source_ = model(model_feed)

            # Forward lens through the original fixed SIS sparse-grid chain.
            upscaled_image_ = lensing_module.cross_grid_fill(
                upscaled_source_,
                [cross_grid_to_log, cross_grid_forward_from_log, cross_grid_from_log],
            )

            # Telescope degradation: empirical/FITS PSF convolution before downsampling.
            convolved_upscaled_image_ = apply_psf(upscaled_image_, psf_kernel)

            # Downsampling and source-cycle consistency.
            downsampled_image = interpolate_image(convolved_upscaled_image_, scale_factor=1 / args.effective_magnification, mode=args.downsample_mode)
            interpolated_image = interpolate_image(lr_image, scale_factor=args.effective_magnification, mode=args.upsample_mode)
            downsampled_source = interpolate_image(upscaled_source_, scale_factor=1 / args.effective_magnification, mode=args.downsample_mode)

            # Losses: weighted MSE.
            image_reconstruction_loss = wmse_loss(downsampled_image, lr_image, source_convergence_map)
            source_reconstruction_loss = wmse_loss(downsampled_source, reconstructed_source, image_convergence_map)

            # Variation density optional regularizer.
            interpolated_image_vd = lensing_module.compute_variation_density(interpolated_image)
            convolved_upscaled_image_vd = lensing_module.compute_variation_density(convolved_upscaled_image_)

            total_loss = image_reconstruction_loss + source_reconstruction_loss + args.vdl_weight * F.mse_loss(interpolated_image_vd, convolved_upscaled_image_vd)
            opt.zero_grad()
            total_loss.backward()
            opt.step()

            train_metrics.update(**build_epoch_metrics(
                total_loss=total_loss,
                image_reconstruction_loss=image_reconstruction_loss,
                source_reconstruction_loss=source_reconstruction_loss,
                downsampled_image=downsampled_image,
                lr_image=lr_image,
                downsampled_source=downsampled_source,
                reconstructed_source=reconstructed_source,
                source_convergence_map=source_convergence_map,
                image_convergence_map=image_convergence_map,
            ))

        if args.log_train:
            for key, value in train_metrics.as_dict().items():
                writer.add_scalar(f"train/{key}", value, global_step=epoch)
        print("[SYS] Train epoch %d: %s" % (epoch + 1, train_metrics.summary()))

        # --- validation loop -------------------------------------------------
        model.eval()
        val_metrics = MetricTracker()
        with torch.no_grad():
            for i, lr_image in enumerate(tqdm(val_dataloader, desc=f"Validation epoch {epoch+1} of {args.epochs}")):
                lr_image = lr_image.float().to(device)

                # Source reconstruction through fixed SIS sparse mapping.
                reconstructed_source = lensing_module.cross_grid_fill(lr_image, [cross_grid_backward])

                # Upscaling.
                model_feed = torch.cat([reconstructed_source, lr_image], dim=1)
                upscaled_source_ = model(model_feed)

                # Original fixed SIS forward-lensing chain.
                upscaled_image_ = lensing_module.cross_grid_fill(
                    upscaled_source_,
                    [cross_grid_to_log, cross_grid_forward_from_log, cross_grid_from_log],
                )
                convolved_upscaled_image_ = apply_psf(upscaled_image_, psf_kernel)

                # Downsampling.
                downsampled_image = interpolate_image(convolved_upscaled_image_, scale_factor=1 / args.effective_magnification, mode=args.downsample_mode)
                interpolated_image = interpolate_image(lr_image, scale_factor=args.effective_magnification, mode=args.upsample_mode)
                downsampled_source = interpolate_image(upscaled_source_, scale_factor=1 / args.effective_magnification, mode=args.downsample_mode)

                # Losses.
                image_reconstruction_loss = wmse_loss(downsampled_image, lr_image, source_convergence_map)
                source_reconstruction_loss = wmse_loss(downsampled_source, reconstructed_source, image_convergence_map)

                interpolated_image_vd = lensing_module.compute_variation_density(interpolated_image)
                convolved_upscaled_image_vd = lensing_module.compute_variation_density(convolved_upscaled_image_)

                total_loss = image_reconstruction_loss + source_reconstruction_loss + args.vdl_weight * F.mse_loss(interpolated_image_vd, convolved_upscaled_image_vd)

                val_metrics.update(**build_epoch_metrics(
                    total_loss=total_loss,
                    image_reconstruction_loss=image_reconstruction_loss,
                    source_reconstruction_loss=source_reconstruction_loss,
                    downsampled_image=downsampled_image,
                    lr_image=lr_image,
                    downsampled_source=downsampled_source,
                    reconstructed_source=reconstructed_source,
                    source_convergence_map=source_convergence_map,
                    image_convergence_map=image_convergence_map,
                ))

        if args.log_train:
            for key, value in val_metrics.as_dict().items():
                writer.add_scalar(f"val/{key}", value, global_step=epoch)
        print("[SYS] Validation epoch %d: %s" % (epoch + 1, val_metrics.summary()))

    # save model weights at the end
    torch.save(model.state_dict(), "%s_weights.pt" % args.exp_name)
    if writer is not None:
        writer.close()
