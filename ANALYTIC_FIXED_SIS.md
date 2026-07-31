# Analytic fixed-SIS branch

This branch replaces the three-matrix forward lensing chain with direct inverse
ray shooting.  The physical operation is

\[
\boldsymbol\beta(\boldsymbol\theta)
=\boldsymbol\theta
-\theta_E\frac{\boldsymbol\theta-\boldsymbol\theta_0}
{|\boldsymbol\theta-\boldsymbol\theta_0|},
\qquad
I_{\rm image}(\boldsymbol\theta)
=S_{\rm source}(\boldsymbol\beta).
\]

`torch.nn.functional.grid_sample` performs bilinear source sampling.  This is
still grid-based lensing: the source and image are represented on discrete
regular grids and the lens equation maps image-grid coordinates into the
source grid.  What is removed is the faulty polygon-overlap/log-grid sparse
operator, not the grids themselves.

## Files

- `analytic_sis.py`: shared batched fixed/per-sample SIS renderer and normalized
  source-plane backprojection.
- `validate_analytic_sis.py`: Stage A deterministic tests; no dataset required.
- `generate_controlled_sis_dataset.py`: Stage B controlled simulation generator.
- `train_analytic_sis.py`: one training entrypoint with `controlled`, `legacy`,
  and `manifest` data adapters.

## The same renderer is used for all stages

The renderer does not change between Stage A, B and C.  Only the inputs change:

- Stage A supplies deterministic tensors such as constants and Gaussians.
- Stage B supplies generated observations with known fixed SIS parameters.
- Stage C supplies real dataset images and either a fixed theta_E (`legacy`) or
  per-sample theta_E (`manifest`).

The manifest mode is only an SIS diagnostic.  The supplied manifest also has
ellipticity, radial slope and external shear, so the final manifest renderer
must become SIE/EPL plus shear.  The direct ray-shooting interface is designed
for that extension.

## Stage A

```bash
python validate_analytic_sis.py \
  --shape 128 \
  --pixel-scale 0.0505 \
  --theta-e 0.75
```

Pass criteria:

- covered constant-source mean within 1% of one;
- covered constant-source standard deviation below 0.01;
- Gaussian peak radius within about one pixel of theta_E / pixel_scale;
- changing Gaussian width does not move the peak by more than one pixel.

## Stage B: generate controlled data

Start without a PSF or noise:

```bash
python generate_controlled_sis_dataset.py \
  --output-dir controlled_sis \
  --train-count 800 \
  --val-count 200 \
  --lr-pixel-scale 0.101 \
  --theta-e 0.75 \
  --psf-type none
```

Then train:

```bash
python train_analytic_sis.py \
  --dataset-mode controlled \
  --controlled-root controlled_sis \
  --resolution 0.101 \
  --epochs 40 \
  --batch-size 32 \
  --truth-loss-weight 0.1 \
  --psf-type none \
  --exp-name controlled_no_psf
```

After the no-PSF baseline works, regenerate a matched-PSF dataset and train with
the same PSF settings.  Do not generate with one PSF and train with another.

## Existing legacy dataset

This mode uses one fixed theta_E for every image and is only valid if that
assumption matches the dataset:

```bash
python train_analytic_sis.py \
  --dataset-mode legacy \
  --classes no_sub \
  --train-count 5000 \
  --val-count 2000 \
  --resolution 0.101 \
  --theta-e 0.75 \
  --epochs 40 \
  --batch-size 32 \
  --psf-type fits \
  --psf-path path/to/psf.fits \
  --psf-source-pixscale 0.168 \
  --exp-name legacy_analytic_sis
```

## Manifest dataset

Manifest mode reads per-sample theta_E:

```bash
python train_analytic_sis.py \
  --dataset-mode manifest \
  --manifest-csv path/to/manifest.csv \
  --manifest-data-root path/to/dataset/root \
  --resolution <verified-image-pixel-scale> \
  --train-count 800 \
  --val-count 200 \
  --epochs 20 \
  --batch-size 16 \
  --psf-type none \
  --exp-name manifest_sis_diagnostic
```

If an NPZ contains several image-like arrays, pass `--npz-image-key`.  This mode
must not be interpreted as a final physical fit because it ignores host_e1,
host_e2, host_slope and external shear.

## Backprojection

`AnalyticSISRenderer.backproject` maps every image ray to source coordinates and
uses normalized bilinear splatting:

\[
\hat S_q = \frac{\sum_p w_{qp} I_p}{\sum_p w_{qp}+\epsilon}.
\]

It is appropriate as a fixed-lens source initialization.  Lens-parameter
learning should use gradients through the forward `grid_sample` renderer.

## Long-term architecture

Direct ray shooting remains grid based.  Recommended final architecture:

1. regular or adaptive source grid;
2. differentiable SIS/SIE/EPL lens equation;
3. `grid_sample` image formation;
4. PSF convolution on a padded working canvas;
5. detector pixel integration;
6. LR likelihood and source/lens regularization.

A sparse equivalent is optional for a fixed lens, but direct sampling is more
suitable for learnable lens parameters because interpolation coordinates remain
differentiable.
