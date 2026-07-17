# Corrected fixed-SIS workflow

## What changed

- Mapping files are loaded as metadata bundles and rejected when SIS geometry, pixel scale, role, or dimensions disagree.
- Backward source reconstruction uses row-normalized sparse weights so a constant image remains on the same surface-brightness scale.
- Observation and source losses divide by the active weight sum.
- Observation weights emphasize arc pixels while retaining a small background term.
- Training defaults to `no_sub` only for controlled validation.
- Magnification is a low-information Laplacian regularizer; it does not replace the rendered source with a pooled source.
- Physics starts after a delay and ramps gradually.
- The old source-cycle/effective-pixel consistency is disabled because it does not identify distinct lensed image branches.
- Every epoch reports raw MSE, zero-output MSE, and skill over zero.
- `test.ipynb` and `test_stage1_2_fits.ipynb` use `model.eval()` and the FITS PSF.

## Mapping preparation

Regenerate the raw sparse matrices with one consistent SIS configuration, then package each matrix. Example for a 64-to-128 run:

```bash
python regenerate_mappings.py --input sparse_grid_fracs_euclid_backward.pt --output mappings/sparse_grid_fracs_euclid_backward_bundle.pt --mapping-role backward --theta-e-arcsec 0.75 --lr-pixel-scale-arcsec 0.168 --hr-pixel-scale-arcsec 0.084 --image-shape 64 --target-shape 128
python regenerate_mappings.py --input scatter_to_log_128.pt --output mappings/scatter_to_log_128_bundle.pt --mapping-role to_log --theta-e-arcsec 0.75 --lr-pixel-scale-arcsec 0.168 --hr-pixel-scale-arcsec 0.084 --image-shape 64 --target-shape 128
python regenerate_mappings.py --input forward_from_log_128.pt --output mappings/forward_from_log_128_bundle.pt --mapping-role forward_from_log --theta-e-arcsec 0.75 --lr-pixel-scale-arcsec 0.168 --hr-pixel-scale-arcsec 0.084 --image-shape 64 --target-shape 128
python regenerate_mappings.py --input scatter_from_log_128.pt --output mappings/scatter_from_log_128_bundle.pt --mapping-role from_log --theta-e-arcsec 0.75 --lr-pixel-scale-arcsec 0.168 --hr-pixel-scale-arcsec 0.084 --image-shape 64 --target-shape 128
```

Do not use these example geometry values unless they are the values used to regenerate the matrices.

## Verify first

```bash
python verify_and_baseline.py --mapping-dir mappings --resolution 0.168 --theta-e 0.75 --psf-path path/hsc.fits --psf-source-pixscale-arcsec 0.168
```

## Smoke test

```bash
python train_all_modifications.py --exp-name smoke_corrected --mapping-dir mappings --resolution 0.168 --theta-e 0.75 --psf-path path/hsc.fits --psf-source-pixscale-arcsec 0.168 --epochs 2 --batch-size 8 --num-workers 0 --train-samples-per-class 32 --val-samples-per-class 16 --dataset-fraction 1.0 --classes no_sub --physics-delay-epochs 10 --physics-ramp-epochs 10 --mu-weight 0.001 --src-cons-weight 0
```

## Main experiment

```bash
python train_all_modifications.py --exp-name fixed_sis_corrected --mapping-dir mappings --resolution 0.168 --theta-e 0.75 --psf-path path/hsc.fits --psf-source-pixscale-arcsec 0.168 --epochs 60 --batch-size 64 --num-workers 8 --train-samples-per-class 5000 --val-samples-per-class 2000 --dataset-fraction 1.0 --classes no_sub --physics-delay-epochs 20 --physics-ramp-epochs 20 --mu-weight 0.001 --source-loss-weight 0.2 --arc-threshold-fraction 0.08 --arc-boost 20 --background-weight 0.05 --src-cons-weight 0
```

Move to `axion`/`cdm` only after validation skill over zero is consistently positive and the LR arcs are reproduced visually.
