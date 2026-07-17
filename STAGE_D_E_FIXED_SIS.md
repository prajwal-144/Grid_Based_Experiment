# Stage D/E: fixed-SIS magnification-aware and source-consistent SR

This stage leaves `train.py` unchanged. It adds `train_mag.py`, which uses the
same fixed SIS sparse mappings and the same FITS/empirical PSF observation
operator as the current pipeline.

## Physics implemented

### 1. Fixed SIS magnification map

For a centered singular isothermal sphere (SIS),

\[
\boldsymbol{\alpha}(\boldsymbol{\theta})
= \theta_E \frac{\boldsymbol{\theta}}{|\boldsymbol{\theta}|}.
\]

The Jacobian eigenvalues are

\[
\lambda_r = 1, \qquad
\lambda_t = 1 - \frac{\theta_E}{r},
\]

so the signed magnification is

\[
\mu(\boldsymbol{\theta})
= \frac{1}{\lambda_r\lambda_t}
= \frac{1}{1-\theta_E/r}.
\]

The critical curve at `r = theta_E` is singular. The code retains a signed
magnification map for diagnostics, clips `|mu|` for numerical use, and converts
it to a logarithmically compressed information score `m(theta) in [0,1]`.
Values `|mu| <= 1` receive no *additional* lens-assisted resolution credit.

Magnification is **not** multiplied into intensity. Strong lensing conserves
surface brightness. The magnification map is used only to determine where the
source reconstruction can retain finer spatial structure.

### 2. Source-plane information map

The existing fixed backward sparse operator is row-normalized and used to map
the image-plane information score to the LR source grid. The resulting map is
then bilinearly upsampled to the HR source grid.

This ensures that the magnification information is expressed on the same source
coordinates as the network output without rebuilding the existing fixed sparse
lensing matrices.

### 3. Magnification-adaptive soft source grid

The present SISR architecture outputs a regular HR tensor. Replacing it with an
irregular adaptive mesh would require a new network representation and new
forward/backward sparse operators. For this validation stage, the code uses a
differentiable soft multi-resolution grid on the regular tensor.

For source information `m(beta)`, it creates fine, medium, and coarse source
representations and blends them using

\[
w_f=m^2, \qquad
w_m=2m(1-m), \qquad
w_c=(1-m)^2.
\]

These weights sum to one. High-information regions use the raw fine source;
low-information regions use the conservative coarse source; intermediate
regions transition smoothly.

The magnification loss is

\[
\mathcal L_\mu =
\frac{\sum_\beta (1-m)^2
  \left[s_{\rm fine}(\beta)-s_{\rm adaptive}(\beta)\right]^2}
 {\sum_\beta (1-m)^2},
\]

normalized by weighted source power. It penalizes unsupported fine structure
where the lens geometry does not justify it.

### 4. Source-plane intensity consistency

Surface brightness conservation implies

\[
I(\boldsymbol\theta_a)=I(\boldsymbol\theta_b)=S(\boldsymbol\beta)
\]

when multiple image-plane positions map to the same source location.

For every source cell `q`, the fixed backward mapping supplies image-plane
contributors `p` with non-negative weights `w_qp`. The code computes

\[
\bar I_q =
\frac{\sum_p w_{qp} I_p}{\sum_p w_{qp}},
\qquad
V_q =
\frac{\sum_p w_{qp} I_p^2}{\sum_p w_{qp}}-\bar I_q^2.
\]

The implemented loss is

\[
\mathcal L_{\rm src-cons}
=
\operatorname{MSE}(\bar I, S)
+
\lambda_{\rm var}\,\langle V\rangle_{n_{\rm eff}\ge n_{\rm min}}.
\]

The first term closes the source-to-image-to-source cycle. The second penalizes
multiple image-plane contributors that disagree about the same source cell.
The loss is evaluated on the predicted **pre-PSF** image because the telescope
PSF mixes neighboring surface brightness values after lensing.

The effective number of contributors is

\[
n_{\rm eff,q}=
\frac{(\sum_p w_{qp})^2}{\sum_p w_{qp}^2}.
\]

### 5. Full objective

`train_mag.py` minimizes

\[
\mathcal L =
\mathcal L_{\rm image}
+\mathcal L_{\rm source}
+\lambda_{\rm VDL}\mathcal L_{\rm VDL}
+r(t)\lambda_\mu\mathcal L_\mu
+r(t)\lambda_{\rm src}\mathcal L_{\rm src-cons},
\]

where `r(t)` is a linear physics warm-up. The same ramp also transitions the
forward renderer from the raw fine source to the fully adaptive source. This
prevents the new constraints from dominating a randomly initialized SISR model.

## Files added

- `physics_losses.py`: fixed-SIS magnification maps, adaptive soft grid,
  `L_mu`, and source-plane intensity consistency.
- `train_mag.py`: new training entry point; `train.py` is untouched.
- `test_physics_losses.py`: small synthetic unit tests for the new physics.
- `evaluate_mag.ipynb`: training-history, map, reconstruction, residual, and
  source-consistency evaluation.

## Run order

### 1. Install the existing project dependencies

FITS PSFs additionally require `astropy`.

```bash
pip install astropy pytest jupyter matplotlib tensorboard
```

### 2. Run the physics unit tests

```bash
python -m pytest -q test_physics_losses.py
```

Expected result: all tests pass.

### 3. Verify the fixed SIS parameter

`--theta-e` must be the Einstein radius used when creating:

- `sparse_grid_fracs_euclid_backward.pt`
- `scatter_to_log_128.pt`
- `forward_from_log_128.pt`
- `scatter_from_log_128.pt`

The default is `0.75` arcsec because that is the compatibility value in the
current `train.py`. Do not change it unless the sparse mappings were generated
with another value.

### 4. Train Stage D/E

Example for an LR pixel scale of `0.168` arcsec/pixel and a FITS PSF sampled at
the same native scale:

```bash
python train_mag.py \
  --exp-name fixed_sis_mag_srccons \
  --resolution 0.168 \
  --image-shape 64 \
  --magnification 2 \
  --n-mag 1 \
  --theta-e 0.75 \
  --psf-type fits \
  --psf-path /absolute/path/to/psf.fits \
  --psf-source-pixscale-arcsec 0.168 \
  --mu-weight 0.02 \
  --src-cons-weight 0.05 \
  --src-cons-variance-weight 1.0 \
  --physics-warmup-epochs 10 \
  --epochs 100 \
  --log-train true
```

The default physics coefficients are deliberately conservative starting
values. Inspect the logged raw loss magnitudes. The weighted physics terms
should influence training without exceeding the image re-degradation term by
orders of magnitude.

Outputs are written to:

```text
outputs_mag/fixed_sis_mag_srccons/
  args.json
  history.json
  physics_maps.pt
  fixed_sis_mag_srccons_weights.pt
  checkpoints/
    best.pt
    final.pt
    epoch_0010.pt ...
```

### 5. Inspect TensorBoard

```bash
tensorboard --logdir runs
```

Track at least:

- `loss_image`
- `loss_source`
- `loss_mu`
- `loss_src_cons_cycle`
- `loss_src_cons_variance`
- `metric_lr_redegradation`
- `metric_source_consistency`
- `metric_flux_lr`

### 6. Evaluate

```bash
jupyter notebook evaluate_mag.ipynb
```

Set `RUN_DIR`, select a validation class/sample, then run all cells.

## Validation criteria before moving to learnable SIS/SIE/shear

1. The LR re-degradation metric must not materially regress relative to the
   unchanged `train.py` baseline with the same seed, data fraction, and PSF.
2. Fine-minus-adaptive residual power should be lower in low-information source
   regions than in high-information regions.
3. `L_src-cons` and its variance component should decrease on validation data.
4. Whitened or normalized LR residuals should not acquire ring-shaped artifacts
   at the SIS critical curve.
5. Results should be stable across at least three random seeds.
6. Repeat with `--mu-weight 0` and `--src-cons-weight 0` separately. These
   ablations are needed to establish which term causes each change.
