# Magnification diagnostics before further training

Run `diagnose_magnification_pipeline.py` before another fixed-SIS, SIE, or real-image experiment. The tests are ordered so that simple operator failures are found before any neural-network interpretation.

## Why these tests are necessary

A double ring can arise from several distinct causes:

1. a physically ring-shaped source under an SIS;
2. a forward sparse operator that duplicates radial structure;
3. an inconsistent analytic magnification map and sparse lens geometry;
4. a log-grid remapping error;
5. an incorrectly sampled PSF;
6. a trained network compensating for one of the above.

The tests isolate these possibilities without retraining.

## Test order and interpretation

### 1. Matrix identity and statistics

The script records SHA-256 hashes, shapes, nonzero counts, value quantiles, row sums, and column sums.

**Why:** two files with the same name and shape may contain different mappings. A checkpoint must only be evaluated with the exact matrices used during training.

**Pass condition:** expected dimensions, finite values, and no unexplained negative weights.

### 2. Constant surface brightness

A constant HR source is passed through the complete forward sparse chain.

Gravitational lensing conserves surface brightness, so away from uncovered boundaries:

`constant source -> approximately constant intrinsic lensed image`

**Why:** a ring in this test proves that the sparse operator or its normalisation is wrong. No neural network or magnification regulariser is involved.

### 3. Regular-to-log-to-regular identity

A centred Gaussian is mapped through:

`scatter_to_log_128 -> scatter_from_log_128`

with the SIS matrix omitted.

**Why:** this isolates the log-grid remapping. If a single Gaussian becomes a ring or double structure here, the issue is not lensing physics.

### 4. Centred Gaussian through the SIS

A compact centred Gaussian source is passed through the full SIS chain.

For a compact centred source, the dominant image should be one broadened Einstein annulus near `theta_E / HR_pixel_scale`.

**Why:** two separated strong radial peaks indicate either an unusual source profile or radial duplication by the sparse chain.

### 5. Ring source through the SIS

A deliberately ring-shaped source is also tested.

For a source ring at radius `beta`, SIS physics can produce image radii approximately:

`theta_E - beta` and `theta_E + beta`.

**Why:** this control demonstrates when a double ring is physically expected. It should not be confused with duplication of a compact source.

### 6. Backward constant-image test

A constant LR image is mapped through the row-normalised backward operator.

**Why:** source reconstruction should preserve surface-brightness scale. If a constant image becomes structured, the network receives an artificial source prior.

### 7. Magnification-information geometry

The script computes the analytic SIS information map and compares its radial peak with:

`theta_E / LR_pixel_scale`.

**Why:** the magnification regulariser is only valid when its critical radius matches the SIS encoded in the sparse matrices.

### 8. PSF flux and sampling

The FITS PSF is rebuilt using its native sampling and the HR target sampling. The script checks kernel sum and a centred delta-function response.

**Why:** `--psf-source-pixscale-arcsec` is the native pixel scale of the FITS stamp, not the LR image scale. A wrongly declared native scale changes the physical PSF width.

### 9. Optional checkpoint test

When `--checkpoint` is supplied, the script reconstructs one validation example using the same mappings and PSF, then reports:

- zero-output MSE;
- model MSE;
- skill over zero;
- observation and prediction radial peaks.

**Why:** this test is meaningful only after the operator-only tests pass.

## Example for Euclid-like LR data and an HSC-native PSF

Assuming:

- LR data sampling: `0.101 arcsec/pixel`;
- HR factor: 2, hence `0.0505 arcsec/pixel`;
- SIS Einstein radius: `0.75 arcsec`;
- FITS PSF native sampling: `0.168 arcsec/pixel`;

run:

```bash
python diagnose_magnification_pipeline.py \
  --mapping-dir path/to/euclid_101_matrices \
  --resolution 0.101 \
  --theta-e 0.75 \
  --psf-path path/to/hsc.fits \
  --psf-source-pixscale-arcsec 0.168 \
  --coordinate-units arcsec \
  --log-c 4.5 \
  --output-dir diagnostics_101
```

With a checkpoint:

```bash
python diagnose_magnification_pipeline.py \
  --mapping-dir path/to/euclid_101_matrices \
  --resolution 0.101 \
  --theta-e 0.75 \
  --psf-path path/to/hsc.fits \
  --psf-source-pixscale-arcsec 0.168 \
  --coordinate-units arcsec \
  --log-c 4.5 \
  --checkpoint outputs_corrected/EXPERIMENT/checkpoints/best.pt \
  --output-dir diagnostics_101_checkpoint
```

## Decision rule before proceeding

Do not move to SIE if any of the following occurs:

- constant-source forward image contains a ring;
- regular/log round trip duplicates a Gaussian;
- centred Gaussian produces two strong unexplained peaks;
- analytic information peak disagrees with the matrix Einstein radius;
- PSF sum is not approximately one;
- checkpoint is evaluated with matrix hashes different from training.

SIE should be introduced only after the fixed-SIS operator passes these tests. Otherwise SIE adds ellipticity and orientation and makes the same failure harder to identify.
