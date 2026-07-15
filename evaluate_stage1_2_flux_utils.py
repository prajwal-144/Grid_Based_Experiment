"""Flux and observation-consistency diagnostics for Stage 1-2 FITS-PSF SR.

This module intentionally uses the original fixed-SIS sparse mappings. It does
not use the Stage-3 learnable SIS implementation.

The decisive check is not raw HR SR versus raw LR. It is:

    SR source -> fixed SIS -> HR lensed image -> FITS PSF -> downsample -> LR

The re-degraded output should reproduce the observed LR