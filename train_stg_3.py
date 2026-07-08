# External imports
import argparse
import os
import test_for_SIS

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Internal imports
import data
from differentiable_lensing import DifferentiableLensing
from lens_models import LearnableSIS
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
    """Convert common string-like booleans to Python bool."""
    return x.lower().strip() in {"true", "1", "yes", "y"}


def wmse_loss(y1, y2, w):
    """Weighted mean-squared error."""
    return torch.mean((y1 - y2) ** 2 * w)


def interpolate_image(x, scale_factor, mode):
    """F.interpolate wrapper with correct align_corners handling."""
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        return F.interpolate(
            x,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False,
        )
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)


def ensure_bchw(image):
    """Normalize dataset output to [B, C, H, W].

    Some stored arrays contain an additional singleton channel dimension,
    producing [B, 1, 1, H, W]. The original training script used a fixed
    squeeze, which was fragile across the two possible dataset shapes.
    """
    while image.ndim > 4 and image.shape[1] == 1:
        image = image.squeeze(1)
    if image.ndim == 3:
        image = image.unsqueeze(1)
    if image.ndim != 4:
        raise ValueError(
            f"Expected dataset batch convertible to BCHW; got {tuple(image.shape)}"
        )
    return image


# === argument parsing ========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument("--lr", type=float, default=2.5e-4, help="SR model learning rate")
    parser.add_argument("--seed", type=int, default=0, help="experiment seed")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--log-train",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--vdl-weight", type=float, default=0.0)

    # Performance / architecture options
    parser.add_argument(
        "--resolution",
        type=float,
        required=True,
        help="LR pixel scale in arcseconds per pixel",
    )
    parser.add_argument("--magnification", type=int, default=2)
    parser.add_argument("--n-mag", type=int, default=1)
    parser.add_argument("--residual-depth", type=int, default=3)
    parser.add_argument("--in-channels", type=int, default=2)
    parser.add_argument("--latent-space-size", type=int, default=64)
    parser.add_argument("--image-shape", type=int, default=64)

    # Stage 3: learnable SIS
    parser.add_argument(
        "--theta-e",
        type=float,
        default=0.75,
        help=(
            "initial/reference SIS Einstein radius in arcseconds; this should "
            "match the SIS used to create the fixed backward mapping"
        ),
    )
    parser.add_argument(
        "--learn-theta-e",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help=(
            "jointly optimize the SIS Einstein radius. When false, the old "
            "fixed sparse forward mappings are used"
        ),
    )
    parser.add_argument(
        "--theta-e-lr",
        type=float,
        default=1e-4,
        help="learning rate for the global SIS Einstein radius",
    )
    parser.add_argument("--theta-e-min", type=float, default=0.05)
    parser.add_argument("--theta-e-max", type=float, default=2.0)
    parser.add_argument(
        "--theta-e-prior-weight",
        type=float,
        default=1e-3,
        help="quadratic prior weight around the initial Einstein radius",
    )
    parser.add_argument(
        "--sis-interpolation-mode",
        type=str,
        default="bilinear",
        choices=["bilinear", "nearest", "bicubic"],
        help="grid_sample interpolation for differentiable SIS rendering",
    )

    # Stage 2: modular PSF / observation operator options
    parser.add_argument(
        "--psf-type",
        type=str,
        default="gaussian",
        choices=["none", "gaussian", "moffat", "empirical"],
    )
    parser.add_argument("--psf-fwhm-arcsec", type=float, default=0.16)
    parser.add_argument("--psf-beta", type=float, default=4.765)
    parser.add_argument("--psf-kernel-size", type=int, default=None)
    parser.add_argument("--psf-path", type=str, default=None)
    parser.add_argument("--psf-ellipticity-q", type=float, default=1.0)
    parser.add_argument("--psf-angle-deg", type=float, default=0.0)
    parser.add_argument(
        "--downsample-mode",
        type=str,
        default="area",
        choices=["nearest", "bilinear", "bicubic", "area"],
    )
    parser.add_argument(
        "--upsample-mode",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic"],
    )

    args = parser.parse_args()
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
    theta_e_prior_loss,
    theta_e,
    downsampled_image,
    lr_image,
    downsampled_source,
    reconstructed_source,
    source_convergence_map,
    image_convergence_map,
):
    """Compute physics-aware metrics for one batch."""
    return {
        "loss_total": total_loss,
        "loss_image": image_reconstruction_loss,
        "loss_source": source_reconstruction_loss,
        "loss_theta_e_prior": theta_e_prior_loss,
        "theta_e_arcsec": theta_e,
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
        "metric_flux_source": flux_conservation_error(
            reconstructed_source,
            downsampled_source,
        ),
    }


def render_forward(
    upscaled_source,
    learn_theta_e,
    sis_lens,
    lensing_module,
    fixed_forward_mappings,
):
    """Render HR lensed image with learnable SIS or legacy fixed grid."""
    if learn_theta_e:
        return sis_lens(upscaled_source)
    return lensing_module.cross_grid_fill(upscaled_source, fixed_forward_mappings)


if __name__ == "__main__":
    args = parse_args()
    run_name = args.exp_name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device)
    batch_size = args.batch_size

    # --- dataset loading -----------------------------------------------------
    train_dataset_no_sub = data.LensingDataset("train/", ["no_sub"], 5000)
    val_dataset_no_sub = data.LensingDataset("val/", ["no_sub"], 2000)
    train_dataset_axion = data.LensingDataset("train/", ["axion"], 5000)
    val_dataset_axion = data.LensingDataset("val/", ["axion"], 2000)
    train_dataset_cdm = data.LensingDataset("train/", ["cdm"], 5000)
    val_dataset_cdm = data.LensingDataset("val/", ["cdm"], 2000)

    train_dataset = torch.utils.data.ConcatDataset(
        [train_dataset_no_sub, train_dataset_axion, train_dataset_cdm]
    )
    val_dataset = torch.utils.data.ConcatDataset(
        [val_dataset_no_sub, val_dataset_axion, val_dataset_cdm]
    )

    train_dataset, _ = torch.utils.data.random_split(train_dataset, [0.34, 0.66])
    val_dataset, _ = torch.utils.data.random_split(val_dataset, [0.34, 0.66])

    worker_count = min(8, os.cpu_count() or 1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=worker_count,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=worker_count,
    )

    # --- models --------------------------------------------------------------
    model = SISR(
        magnification=args.magnification,
        n_mag=args.n_mag,
        residual_depth=args.residual_depth,
        in_channels=args.in_channels,
        latent_channel_count=args.latent_space_size,
    ).to(device)

    lensing_module = DifferentiableLensing(
        device=device,
        alpha=None,
        target_resolution=args.target_resolution,
        target_shape=args.target_shape,
    ).to(device)

    sis_lens = LearnableSIS(
        theta_e_init=args.theta_e,
        target_resolution=args.target_resolution,
        target_shape=args.target_shape,
        theta_e_min=args.theta_e_min,
        theta_e_max=4.0,#args.theta_e_max,
        learnable=args.learn_theta_e,
        interpolation_mode=args.sis_interpolation_mode,
    ).to(device)

    optimizer_groups = [{"params": model.parameters(), "lr": args.lr}]
    if args.learn_theta_e:
        optimizer_groups.append(
            {"params": sis_lens.parameters(), "lr": args.theta_e_lr}
        )
    opt = torch.optim.Adam(optimizer_groups)

    # TensorBoard logging
    writer = None
    if args.log_train:
        writer = SummaryWriter("runs/%s" % run_name)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join(["|%s|%s" % (key, value) for key, value in vars(args).items()]),
        )

    # --- fixed grid mappings -------------------------------------------------
    # The backward matrix still supplies the initial LR source reconstruction.
    # It encodes the fixed/reference SIS used during grid precomputation.
    cross_grid_backward = torch.load(
        "sparse_grid_fracs_euclid_backward.pt",
        map_location=device,
    ).to(device)

    fixed_forward_mappings = None
    if not args.learn_theta_e:
        cross_grid_to_log = torch.load(
            "scatter_to_log_128.pt",
            map_location=device,
        ).to(device)
        cross_grid_forward_from_log = torch.load(
            "forward_from_log_128.pt",
            map_location=device,
        ).to(device)
        cross_grid_from_log = torch.load(
            "scatter_from_log_128.pt",
            map_location=device,
        ).to(device)
        fixed_forward_mappings = [
            cross_grid_to_log,
            cross_grid_forward_from_log,
            cross_grid_from_log,
        ]

    source_convergence_map = torch.load(
        "source_convergence_map.pt",
        map_location=device,
    ).to(device)
    image_convergence_map = torch.load(
        "image_convergence_map.pt",
        map_location=device,
    ).to(device)

    # --- PSF -----------------------------------------------------------------
    psf_kernel = build_psf_kernel(
        psf_type=args.psf_type,
        fwhm_arcsec=args.psf_fwhm_arcsec,
        pixscale_arcsec=args.target_resolution,
        beta=args.psf_beta,
        kernel_size=args.psf_kernel_size,
        path=args.psf_path,
        ellipticity_q=args.psf_ellipticity_q,
        angle_deg=args.psf_angle_deg,
        device=device,
    )

    if psf_kernel is None:
        print("[SYS] PSF disabled.")
    else:
        print(
            "[SYS] Using %s PSF: shape=%s sum=%.6f"
            % (args.psf_type, tuple(psf_kernel.shape), psf_kernel.sum().item())
        )

    if args.learn_theta_e:
        print(
            "[SYS] Learnable SIS enabled: theta_E init=%.6f arcsec, LR=%.2e"
            % (sis_lens.einstein_radius().item(), args.theta_e_lr)
        )
        print(
            "[SYS] The fixed backward mapping should have been generated near "
            "theta_E=%.6f arcsec." % args.theta_e
        )
    else:
        print("[SYS] Learnable SIS disabled; using legacy fixed forward mappings.")

    # --- training ------------------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        sis_lens.train()
        train_metrics = MetricTracker()

        for lr_image in tqdm(
            train_dataloader,
            desc="Training epoch %d of %d" % (epoch + 1, args.epochs),
        ):
            lr_image = ensure_bchw(lr_image.float().to(device))

            reconstructed_source = lensing_module.cross_grid_fill(
                lr_image,
                [cross_grid_backward],
            )
            model_feed = torch.cat([reconstructed_source, lr_image], dim=1)
            upscaled_source = model(model_feed)

            upscaled_image = render_forward(
                upscaled_source=upscaled_source,
                learn_theta_e=args.learn_theta_e,
                sis_lens=sis_lens,
                lensing_module=lensing_module,
                fixed_forward_mappings=fixed_forward_mappings,
            )
            convolved_upscaled_image = apply_psf(upscaled_image, psf_kernel)

            downsampled_image = interpolate_image(
                convolved_upscaled_image,
                scale_factor=1 / args.effective_magnification,
                mode=args.downsample_mode,
            )
            interpolated_image = interpolate_image(
                lr_image,
                scale_factor=args.effective_magnification,
                mode=args.upsample_mode,
            )
            downsampled_source = interpolate_image(
                upscaled_source,
                scale_factor=1 / args.effective_magnification,
                mode=args.downsample_mode,
            )

            image_reconstruction_loss = wmse_loss(
                downsampled_image,
                lr_image,
                source_convergence_map,
            )
            source_reconstruction_loss = wmse_loss(
                downsampled_source,
                reconstructed_source,
                image_convergence_map,
            )

            interpolated_image_vd = lensing_module.compute_variation_density(
                interpolated_image
            )
            convolved_upscaled_image_vd = lensing_module.compute_variation_density(
                convolved_upscaled_image
            )
            variation_loss = F.mse_loss(
                interpolated_image_vd,
                convolved_upscaled_image_vd,
            )

            if args.learn_theta_e:
                theta_e_prior_loss = sis_lens.prior_loss(args.theta_e)
            else:
                theta_e_prior_loss = torch.zeros((), device=device)

            total_loss = (
                image_reconstruction_loss
                + source_reconstruction_loss
                + args.vdl_weight * variation_loss
                + args.theta_e_prior_weight * theta_e_prior_loss
            )

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            train_metrics.update(
                **build_epoch_metrics(
                    total_loss=total_loss,
                    image_reconstruction_loss=image_reconstruction_loss,
                    source_reconstruction_loss=source_reconstruction_loss,
                    theta_e_prior_loss=theta_e_prior_loss,
                    theta_e=sis_lens.einstein_radius(),
                    downsampled_image=downsampled_image,
                    lr_image=lr_image,
                    downsampled_source=downsampled_source,
                    reconstructed_source=reconstructed_source,
                    source_convergence_map=source_convergence_map,
                    image_convergence_map=image_convergence_map,
                )
            )

        if writer is not None:
            for key, value in train_metrics.as_dict().items():
                writer.add_scalar("train/%s" % key, value, global_step=epoch)
        print("[SYS] Train epoch %d: %s" % (epoch + 1, train_metrics.summary()))

        # --- validation ------------------------------------------------------
        model.eval()
        sis_lens.eval()
        val_metrics = MetricTracker()

        with torch.no_grad():
            for lr_image in tqdm(
                val_dataloader,
                desc="Validation epoch %d of %d" % (epoch + 1, args.epochs),
            ):
                lr_image = ensure_bchw(lr_image.float().to(device))

                reconstructed_source = lensing_module.cross_grid_fill(
                    lr_image,
                    [cross_grid_backward],
                )
                model_feed = torch.cat([reconstructed_source, lr_image], dim=1)
                upscaled_source = model(model_feed)

                upscaled_image = render_forward(
                    upscaled_source=upscaled_source,
                    learn_theta_e=args.learn_theta_e,
                    sis_lens=sis_lens,
                    lensing_module=lensing_module,
                    fixed_forward_mappings=fixed_forward_mappings,
                )
                convolved_upscaled_image = apply_psf(upscaled_image, psf_kernel)

                downsampled_image = interpolate_image(
                    convolved_upscaled_image,
                    scale_factor=1 / args.effective_magnification,
                    mode=args.downsample_mode,
                )
                interpolated_image = interpolate_image(
                    lr_image,
                    scale_factor=args.effective_magnification,
                    mode=args.upsample_mode,
                )
                downsampled_source = interpolate_image(
                    upscaled_source,
                    scale_factor=1 / args.effective_magnification,
                    mode=args.downsample_mode,
                )

                image_reconstruction_loss = wmse_loss(
                    downsampled_image,
                    lr_image,
                    source_convergence_map,
                )
                source_reconstruction_loss = wmse_loss(
                    downsampled_source,
                    reconstructed_source,
                    image_convergence_map,
                )

                interpolated_image_vd = lensing_module.compute_variation_density(
                    interpolated_image
                )
                convolved_upscaled_image_vd = lensing_module.compute_variation_density(
                    convolved_upscaled_image
                )
                variation_loss = F.mse_loss(
                    interpolated_image_vd,
                    convolved_upscaled_image_vd,
                )

                if args.learn_theta_e:
                    theta_e_prior_loss = sis_lens.prior_loss(args.theta_e)
                else:
                    theta_e_prior_loss = torch.zeros((), device=device)

                total_loss = (
                    image_reconstruction_loss
                    + source_reconstruction_loss
                    + args.vdl_weight * variation_loss
                    + args.theta_e_prior_weight * theta_e_prior_loss
                )

                val_metrics.update(
                    **build_epoch_metrics(
                        total_loss=total_loss,
                        image_reconstruction_loss=image_reconstruction_loss,
                        source_reconstruction_loss=source_reconstruction_loss,
                        theta_e_prior_loss=theta_e_prior_loss,
                        theta_e=sis_lens.einstein_radius(),
                        downsampled_image=downsampled_image,
                        lr_image=lr_image,
                        downsampled_source=downsampled_source,
                        reconstructed_source=reconstructed_source,
                        source_convergence_map=source_convergence_map,
                        image_convergence_map=image_convergence_map,
                    )
                )

        if writer is not None:
            for key, value in val_metrics.as_dict().items():
                writer.add_scalar("val/%s" % key, value, global_step=epoch)
        print("[SYS] Validation epoch %d: %s" % (epoch + 1, val_metrics.summary()))

    # Preserve the old model-only file and add a complete Stage-3 checkpoint.
    torch.save(model.state_dict(), "%s_weights.pt" % args.exp_name)
    torch.save(
        {
            "sr_model_state_dict": model.state_dict(),
            "sis_lens_state_dict": sis_lens.state_dict(),
            "theta_e_arcsec": float(sis_lens.einstein_radius().detach().cpu().item()),
            "optimizer_state_dict": opt.state_dict(),
            "args": vars(args),
        },
        "%s_checkpoint.pt" % args.exp_name,
    )

    if writer is not None:
        writer.close()