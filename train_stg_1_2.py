# External imports
import torch
import numpy as np
import argparse
import os
import test_for_SIS
from tqdm import tqdm
import torch.nn.functional as F
import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Internal imports
from differentiable_lensing import DifferentiableLensing
import data
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
    """
    Convert a string-like boolean into a Python bool.
    - Accepts things like 'true' (case-insensitive) -> True, everything else -> False.
    NOTE: this is very strict. Consider using distutils.util.strtobool for more options.
    """
    if x.lower().strip() == 'true': return True
    else: return False

def wmse_loss(y1, y2, w):
    """
    Weighted MSE wrapper: returns mean((y1-y2)^2 * w).
    - y1, y2, w: tensors broadcastable to same shape.
    - Note: no epsilon / stability tweak; if w contains zeros everywhere the mean will be 0 which might be OK.
    """
    return torch.mean((y1-y2)**2*w)


def interpolate_image(x, scale_factor, mode):
    """F.interpolate wrapper with correct align_corners handling."""
    if mode in {'linear', 'bilinear', 'bicubic', 'trilinear'}:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)


# === argument parsing =======================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='the LR of the optimizer')
    parser.add_argument('--seed', type=int, default=0,
                        help='the seed of the experiment')
    parser.add_argument('--epochs', type=int, default=100,
                        help='total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if False, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if True, cuda will be enabled when possible')
    parser.add_argument('--log-train', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if True, training will be logged with Tensorboard')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='the number of images loaded to at any moment')
    parser.add_argument('--vdl-weight', type=float, default=0.0,
                        help='weight of the vdl loss')

    # Performance / architecture options
    parser.add_argument('--resolution', type=float,
                        help='arcsecond per pixel resolution the images are captured in')
    parser.add_argument('--magnification', type=int, default=2,
                        help='magnification value achieved by the SR network')
    parser.add_argument('--n-mag', type=int, default=1,
                        help='number of times the magnification value is applied by the SR network')
    parser.add_argument('--residual-depth', type=int, default=3,
                        help='the number of residual layers in the SR network')
    parser.add_argument('--in-channels', type=int, default=2,
                        help='the number of channels in the images')
    parser.add_argument('--latent-space-size', type=int, default=64,
                        help='the number of neurons in the latent space(s)')
    parser.add_argument('--image-shape', type=int, default=64,
                        help='the shape of the (square) image in one axis')
    parser.add_argument('--theta-e', type=float, default=0.75,
                        help='the value of the einstein radius used to compute the deflection field')

    # Stage 2: modular PSF / observation operator options
    parser.add_argument('--psf-type', type=str, default='gaussian', choices=['none', 'gaussian', 'moffat', 'empirical'],
                        help='PSF model applied before LR re-degradation')
    parser.add_argument('--psf-fwhm-arcsec', type=float, default=0.16,
                        help='PSF FWHM in arcseconds for analytic Gaussian/Moffat kernels')
    parser.add_argument('--psf-beta', type=float, default=4.765,
                        help='Moffat beta parameter when --psf-type=moffat')
    parser.add_argument('--psf-kernel-size', type=int, default=None,
                        help='optional odd PSF kernel width in pixels')
    parser.add_argument('--psf-path', type=str, default=None,
                        help='path to .npy/.npz/.pt/.pth empirical PSF kernel when --psf-type=empirical')
    parser.add_argument('--psf-ellipticity-q', type=float, default=1.0,
                        help='minor/major axis ratio for analytic PSFs')
    parser.add_argument('--psf-angle-deg', type=float, default=0.0,
                        help='counter-clockwise analytic PSF major-axis angle in degrees')
    parser.add_argument('--downsample-mode', type=str, default='area', choices=['nearest', 'bilinear', 'bicubic', 'area'],
                        help='downsampling mode for observation consistency')
    parser.add_argument('--upsample-mode', type=str, default='bilinear', choices=['nearest', 'bilinear', 'bicubic'],
                        help='upsampling mode used for interpolation reference in VDL')
    args = parser.parse_args()

    # derived args
    args.effective_magnification = int(args.magnification ** args.n_mag)
    args.target_shape = args.image_shape * args.effective_magnification
    args.target_resolution = args.resolution / args.effective_magnification
    args.device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    print('[SYS] Device is set to %s'%args.device)
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
        'loss_total': total_loss,
        'loss_image': image_reconstruction_loss,
        'loss_source': source_reconstruction_loss,
        'metric_lr_redegradation': lr_redegradation_error(
            downsampled_image,
            lr_image,
            weight=source_convergence_map,
        ),
        'metric_source_consistency': source_plane_reconstruction_consistency(
            downsampled_source,
            reconstructed_source,
            weight=image_convergence_map,
        ),
        'metric_flux_lr': flux_conservation_error(lr_image, downsampled_image),
        'metric_flux_source': flux_conservation_error(reconstructed_source, downsampled_source),
    }


if __name__ == '__main__':
    args = parse_args()
    run_name = f'{args.exp_name}'

    # seeds for repeatability
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = args.batch_size

    # --- dataset loading ---------------------------------------------------
    train_dataset_no_sub = data.LensingDataset('train/',['no_sub'],5000)
    val_dataset_no_sub = data.LensingDataset('val/',['no_sub'],2000)

    train_dataset_axion = data.LensingDataset('train/',['axion'],5000)
    val_dataset_axion = data.LensingDataset('val/',['axion'],2000)

    train_dataset_cdm = data.LensingDataset('train/',['cdm'],5000)
    val_dataset_cdm = data.LensingDataset('val/',['cdm'],2000)

    train_dataset = torch.utils.data.ConcatDataset([train_dataset_no_sub, train_dataset_axion, train_dataset_cdm])
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_no_sub, val_dataset_axion, val_dataset_cdm])

    train_dataset, train_rest = torch.utils.data.random_split(train_dataset, [0.34, 0.66])
    val_dataset, val_rest = torch.utils.data.random_split(val_dataset, [0.34, 0.66])

    train_dataloader = torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=BATCH_SIZE,num_workers=min(8, os.cpu_count()))
    val_dataloader = torch.utils.data.DataLoader(val_dataset,shuffle=True,batch_size=BATCH_SIZE,num_workers=min(8, os.cpu_count()))

    # --- model / modules ----------------------------------------------------
    model = SISR(magnification=args.magnification, n_mag=args.n_mag, residual_depth=args.residual_depth, in_channels=args.in_channels, latent_channel_count=args.latent_space_size).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    lensing_module = DifferentiableLensing(device=device, alpha=None, target_resolution=args.target_resolution, target_shape=args.target_shape).to(args.device)

    # TensorBoard logging
    if args.log_train:
        writer = SummaryWriter(f'runs/{run_name}')
        writer.add_text(
            'hyperparameters',
            '|param|value|\n|-|-|\n%s'%('\n'.join([f'|{key}|{value}' for key, value in vars(args).items()])),
        )

    # --- load precomputed sparse mappings and maps --------------------------
    cross_grid_to_log = torch.load('scatter_to_log_128.pt').to(args.device)
    cross_grid_forward_from_log = torch.load('forward_from_log_128.pt').to(args.device)
    cross_grid_from_log = torch.load('scatter_from_log_128.pt').to(args.device)
    cross_grid_backward = torch.load('sparse_grid_fracs_euclid_backward.pt').to(args.device)

    # convergence maps
    source_convergence_map = torch.load('source_convergence_map.pt').to(args.device)
    image_convergence_map = torch.load('image_convergence_map.pt').to(args.device)

    # --- PSF kernel setup ---------------------------------------------------
    psf_kernel = build_psf_kernel(
        psf_type=args.psf_type,
        fwhm_arcsec=args.psf_fwhm_arcsec,
        pixscale_arcsec=args.target_resolution,
        beta=args.psf_beta,
        kernel_size=args.psf_kernel_size,
        path=args.psf_path,
        ellipticity_q=args.psf_ellipticity_q,
        angle_deg=args.psf_angle_deg,
        device=args.device,
    )
    if psf_kernel is None:
        print('[SYS] PSF disabled (--psf-type=none).')
    else:
        print('[SYS] Using %s PSF: shape=%s sum=%.6f' % (args.psf_type, tuple(psf_kernel.shape), psf_kernel.sum().item()))

    # --- training loop ------------------------------------------------------
    for epoch in range(args.epochs):
        model.train()
        train_metrics = MetricTracker()

        for i,lr_image in enumerate(tqdm(train_dataloader, desc=f"Training epoch {epoch+1} of {args.epochs}")):
            lr_image = lr_image.float().to(device) #.squeeze(1)

            # Source reconstruction through backward lensing (using precomputed sparse mapping)
            reconstructed_source = lensing_module.cross_grid_fill(lr_image, [cross_grid_backward])

            # Upscaling using a neural network: concatenate source and lr image along channels
            model_feed = torch.cat([reconstructed_source, lr_image], dim=1)
            upscaled_source_ = model(model_feed)

            # Image construction through forward lensing: apply a chain of sparse mappings
            upscaled_image_ = lensing_module.cross_grid_fill(upscaled_source_, [cross_grid_to_log, cross_grid_forward_from_log, cross_grid_from_log])
            convolved_upscaled_image_ = apply_psf(upscaled_image_, psf_kernel)

            # Downsampling and upsampling
            downsampled_image = interpolate_image(convolved_upscaled_image_, scale_factor=1/args.effective_magnification, mode=args.downsample_mode)
            interpolated_image = interpolate_image(lr_image, scale_factor=args.effective_magnification, mode=args.upsample_mode)
            downsampled_source = interpolate_image(upscaled_source_, scale_factor=1/args.effective_magnification, mode=args.downsample_mode)

            # Losses: weighted MSE
            image_reconstruction_loss = wmse_loss(downsampled_image, lr_image, source_convergence_map)
            source_reconstruction_loss = wmse_loss(downsampled_source, reconstructed_source, image_convergence_map)

            # Variation density (an optional regularizer)
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

        # Logging/tracking after training loop
        if args.log_train:
            for key, value in train_metrics.as_dict().items():
                writer.add_scalar(f"train/{key}", value, global_step=epoch)
        print('[SYS] Train epoch %d: %s' % (epoch+1, train_metrics.summary()))

        # --- validation loop -------------------------------------------------
        model.eval()
        val_metrics = MetricTracker()
        with torch.no_grad():
            for i,lr_image in enumerate(tqdm(val_dataloader, desc=f"Validation epoch {epoch+1} of {args.epochs}")):
                lr_image = lr_image.float().to(device) #.squeeze(1)

                # Source reconstruction
                reconstructed_source = lensing_module.cross_grid_fill(lr_image, [cross_grid_backward])

                # Upscaling
                model_feed = torch.cat([reconstructed_source, lr_image], dim=1)
                upscaled_source_ = model(model_feed)

                # Image construction
                upscaled_image_ = lensing_module.cross_grid_fill(upscaled_source_, [cross_grid_to_log, cross_grid_forward_from_log, cross_grid_from_log])
                convolved_upscaled_image_ = apply_psf(upscaled_image_, psf_kernel)

                # Downsampling
                downsampled_image = interpolate_image(convolved_upscaled_image_, scale_factor=1/args.effective_magnification, mode=args.downsample_mode)
                interpolated_image = interpolate_image(lr_image, scale_factor=args.effective_magnification, mode=args.upsample_mode)
                downsampled_source = interpolate_image(upscaled_source_, scale_factor=1/args.effective_magnification, mode=args.downsample_mode)

                # Losses
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
        print('[SYS] Validation epoch %d: %s' % (epoch+1, val_metrics.summary()))

    # save model weights at the end
    torch.save(model.state_dict(), '%s_weights.pt'%args.exp_name)