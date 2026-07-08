# FITS PSF usage for Stage 1-2

This branch keeps the original Stage 1-2 fixed-SIS sparse-grid lensing pipeline and replaces the analytic PSF degradation with an optional empirical FITS PSF kernel.

## 1. Inspect a FITS PSF

Open `inspect_psf_fits.ipynb`, set:

```python
fits_path = Path("path/to/your_psf.fits")
```

Then run the cells. Check:

- which HDU contains image data;
- whether the PSF is 2D or stored inside a cube;
- the array shape;
- whether the header records a pixel scale;
- whether the PSF peak is near the center;
- whether the kernel has negative or NaN values.

The training loader clips negative numerical values to zero and normalizes the kernel to unit sum.

## 2. Which PSF to use

For HSC low-resolution training, use the HSC PSF as the degradation kernel, because the training objective compares the degraded prediction against the HSC-like LR input.

The HST PSF is useful when generating or comparing an HST-like observed image. It is not the right degradation kernel for an HSC LR likelihood unless the LR observation is HST-like.

## 3. Pixel-scale convention

`train_stg_1_2.py` applies the PSF to the high-resolution lensed image before downsampling:

```text
SR source -> fixed SIS forward lens -> HR lensed image -> PSF convolution -> downsample -> LR comparison
```

Therefore the PSF kernel must be sampled on the HR grid pixel scale:

```text
target_resolution = resolution / magnification**n_mag
```

If your FITS PSF is sampled at the native observed image scale, pass its native pixel scale:

```bash
--psf-source-pixscale-arcsec <native_psf_arcsec_per_pixel>
```

The loader resamples the FITS PSF to `target_resolution` before convolution.

## 4. HSC FITS PSF training example

```bash
python train_stg_1_2.py \
  --exp-name hsc_fits_psf \
  --resolution 0.168 \
  --epochs 100 \
  --batch-size 200 \
  --magnification 2 \
  --n-mag 1 \
  --residual-depth 3 \
  --latent-space-size 64 \
  --in-channels 2 \
  --psf-type fits \
  --psf-path path/to/hsc_psf.fits \
  --psf-fits-hdu 0 \
  --psf-source-pixscale-arcsec 0.168 \
  --psf-fits-crop-size 65 \
  --downsample-mode area \
  --upsample-mode bilinear \
  --vdl-weight 0.0 \
  --log-train True
```

If your PSF is in a named FITS extension, use:

```bash
--psf-fits-extname PSF
```

instead of `--psf-fits-hdu`.

## 5. Smoke test

Run one epoch first:

```bash
python train_stg_1_2.py \
  --exp-name smoke_hsc_fits_psf \
  --resolution 0.168 \
  --epochs 1 \
  --batch-size 8 \
  --magnification 2 \
  --n-mag 1 \
  --psf-type fits \
  --psf-path path/to/hsc_psf.fits \
  --psf-source-pixscale-arcsec 0.168 \
  --psf-fits-crop-size 65 \
  --downsample-mode area \
  --log-train True
```

Expected startup printout:

```text
[SYS] Using fits PSF: shape=(..., ...) sum=1.000000
[SYS] HR target pixel scale for PSF convolution: ... arcsec/pixel
[SYS] Native PSF pixel scale: ... arcsec/pixel
```

## 6. Dependency

FITS support requires Astropy:

```bash
pip install astropy
```
