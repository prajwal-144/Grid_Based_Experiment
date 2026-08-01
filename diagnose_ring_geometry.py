"""
diagnose_ring_geometry.py

Measures the three radii needed to diagnose donut-shaped source reconstructions
in the grid-based lensing SR experiment.

The physics being tested
------------------------
For an SIS lens the backward map is  beta = theta - theta_E * theta_hat.
A data ring at angular radius R maps to a source-plane annulus of radius

    beta_0 = | R - theta_E_model |

which collapses to a point ONLY if theta_E_model == R. A non-zero beta_0
(the "donut") then forward-lenses into TWO image rings at theta_E +/- beta_0
(the "double ring"). So all three observables are linked:

    r_data          -- ring radius in the input LR image
    r_source        -- donut radius in the reconstructed source
    r_out, r_in     -- the two ring radii in the LR prediction

Consistency checks this script performs:
    (1) r_source ~= |r_data - theta_E_model|        -> theta_E mismatch
    (2) (r_out - r_in) / 2 ~= r_source              -> double ring is the donut
    (3) (r_out + r_in) / 2 ~= theta_E_model         -> ring centre is theta_E
    (4) r_source ~= r_data                          -> deflection ~ 0 (units bug)

Usage
-----
    import numpy as np
    from diagnose_ring_geometry import diagnose

    diagnose(
        lr_data=np.load("some_lr_image.npy"),
        source_recon=recon_array,
        lr_pred=prediction_array,
        theta_e=0.75,
        lr_pixel_scale=0.168,
        source_pixel_scale=0.084,
    )

Only numpy is required; matplotlib is optional (plots are skipped if absent).
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# radial profile machinery
# ----------------------------------------------------------------------------

def _as_2d(arr):
    """Squeeze a (1,1,H,W) / (1,H,W) / (H,W) array down to (H, W)."""
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"expected a 2D image after squeeze, got shape {a.shape}")
    return a.astype(np.float64)


def _geometric_center(shape):
    """Centre of an (H, W) grid in pixel coordinates.

    NOTE the (N-1)/2 convention: for an even-sized grid the centre falls
    BETWEEN pixels. If your matrix-generation code uses N/2 instead, you have a
    half-pixel registration offset -- worth checking independently.
    """
    h, w = shape
    return (h - 1) / 2.0, (w - 1) / 2.0


def radial_profile(img, center=None, n_bins=None):
    """Azimuthally averaged radial profile.

    Returns
    -------
    r_bin   : (n_bins,) bin centres, in PIXELS
    prof    : (n_bins,) mean pixel value in each annulus (NaN where empty)
    """
    img = _as_2d(img)
    if center is None:
        center = _geometric_center(img.shape)
    cy, cx = center

    yy, xx = np.indices(img.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    r_max = r.max()
    if n_bins is None:
        # ~1 pixel per bin out to the corner
        n_bins = max(8, int(np.ceil(r_max)))

    edges = np.linspace(0.0, r_max, n_bins + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sums = np.bincount(idx, weights=img.ravel(), minlength=n_bins)

    prof = np.full(n_bins, np.nan)
    nonempty = counts > 0
    prof[nonempty] = sums[nonempty] / counts[nonempty]

    r_bin = 0.5 * (edges[:-1] + edges[1:])
    return r_bin, prof


def _parabolic_refine(r_bin, prof, i):
    """Sub-bin peak position via a 3-point parabolic fit around index i."""
    if i <= 0 or i >= len(prof) - 1:
        return r_bin[i]
    y0, y1, y2 = prof[i - 1], prof[i], prof[i + 1]
    if not np.all(np.isfinite([y0, y1, y2])):
        return r_bin[i]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return r_bin[i]
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    step = r_bin[1] - r_bin[0]
    return r_bin[i] + delta * step


def find_ring_radii(img, center=None, n_peaks=2, min_separation_px=2.0,
                    exclude_center_px=0.0):
    """Locate up to `n_peaks` annular brightness peaks in a radial profile.

    Returns a list of (radius_px, peak_value), brightest first.

    `exclude_center_px` suppresses the r~0 peak, which is useful when the
    source really is a centred blob and you only want genuine annuli.
    """
    r_bin, prof = radial_profile(img, center=center)

    work = prof.copy()
    work[~np.isfinite(work)] = -np.inf
    if exclude_center_px > 0:
        work[r_bin < exclude_center_px] = -np.inf

    peaks = []
    for _ in range(n_peaks):
        i = int(np.argmax(work))
        if not np.isfinite(work[i]) or work[i] == -np.inf:
            break
        peaks.append((_parabolic_refine(r_bin, prof, i), float(prof[i])))
        # suppress a neighbourhood so the next peak is genuinely distinct
        work[np.abs(r_bin - r_bin[i]) < min_separation_px] = -np.inf

    return peaks, (r_bin, prof)


def is_donut(img, center=None, contrast=1.15):
    """Heuristic: True if the profile peaks off-centre by `contrast`x.

    A centrally-peaked source gives False; an annulus gives True.
    """
    r_bin, prof = radial_profile(img, center=center)
    finite = np.isfinite(prof)
    if finite.sum() < 4:
        return False, 0.0, 0.0
    r_bin, prof = r_bin[finite], prof[finite]
    i_peak = int(np.argmax(prof))
    r_peak = _parabolic_refine(r_bin, prof, i_peak)
    center_val = prof[0]
    peak_val = prof[i_peak]
    if center_val <= 0:
        ratio = np.inf if peak_val > 0 else 0.0
    else:
        ratio = peak_val / center_val
    return bool(r_peak > 1.0 and ratio > contrast), float(r_peak), float(ratio)


# ----------------------------------------------------------------------------
# the diagnostic itself
# ----------------------------------------------------------------------------

def diagnose(lr_data,
             source_recon=None,
             lr_pred=None,
             theta_e=None,
             lr_pixel_scale=None,
             source_pixel_scale=None,
             tol_frac=0.20,
             plot=False):
    """Run the full ring-geometry diagnostic.

    Parameters
    ----------
    lr_data            : 2D array, the INPUT low-resolution image (ground truth
                         observation). This is what sets the true ring radius.
    source_recon       : 2D array, the reconstructed source plane. Optional.
    lr_pred            : 2D array, the forward-lensed LR prediction. Optional.
    theta_e            : float, arcsec. The Einstein radius your matrices assume.
    lr_pixel_scale     : float, arcsec/pixel for lr_data and lr_pred.
    source_pixel_scale : float, arcsec/pixel for source_recon. Defaults to
                         lr_pixel_scale / 2 if not given.
    tol_frac           : fractional tolerance for the consistency checks.
    plot               : if True and matplotlib is available, show profiles.

    Returns
    -------
    dict of measured quantities (all radii reported in BOTH px and arcsec).
    """
    if lr_pixel_scale is None:
        raise ValueError("lr_pixel_scale is required -- radii are meaningless without it")
    if source_pixel_scale is None:
        source_pixel_scale = lr_pixel_scale / 2.0

    out = {}
    line = "-" * 74

    print(line)
    print("RING GEOMETRY DIAGNOSTIC")
    print(line)
    print(f"  assumed theta_E          : {theta_e} arcsec"
          if theta_e is not None else "  assumed theta_E          : (not supplied)")
    print(f"  LR pixel scale           : {lr_pixel_scale} arcsec/px")
    print(f"  source pixel scale       : {source_pixel_scale} arcsec/px")
    if theta_e is not None:
        print(f"  => theta_E in LR pixels   : {theta_e / lr_pixel_scale:.3f} px")
        print("     (this ratio is the ONLY thing the operator sees)")
    print()

    # --- 1. the data ring -------------------------------------------------
    peaks, prof_data = find_ring_radii(lr_data, n_peaks=1, exclude_center_px=1.0)
    if not peaks:
        raise RuntimeError("could not locate a ring in lr_data")
    r_data_px = peaks[0][0]
    r_data_as = r_data_px * lr_pixel_scale
    out["r_data_px"] = r_data_px
    out["r_data_arcsec"] = r_data_as

    print("[1] INPUT LR DATA")
    print(f"    ring radius            : {r_data_px:.3f} px  =  {r_data_as:.4f} arcsec")
    print(f"    -> theta_E consistent with this data = {r_data_as:.4f} arcsec")
    if theta_e is not None:
        mismatch = abs(r_data_as - theta_e)
        out["theta_e_required"] = r_data_as
        out["theta_e_mismatch_arcsec"] = mismatch
        print(f"    -> mismatch vs your theta_E  = {mismatch:.4f} arcsec"
              f"  ({mismatch / lr_pixel_scale:.2f} LR px)")
        print(f"    -> PREDICTED donut radius    = {mismatch:.4f} arcsec")
    print()

    # --- 2. the reconstructed source -------------------------------------
    if source_recon is not None:
        donut, r_src_px, contrast = is_donut(source_recon)
        r_src_as = r_src_px * source_pixel_scale
        out["r_source_px"] = r_src_px
        out["r_source_arcsec"] = r_src_as
        out["source_is_donut"] = donut
        out["source_center_contrast"] = contrast

        print("[2] RECONSTRUCTED SOURCE")
        print(f"    profile peak radius    : {r_src_px:.3f} px  =  {r_src_as:.4f} arcsec")
        print(f"    peak / centre ratio    : {contrast:.3f}")
        print(f"    donut?                 : {'YES' if donut else 'no'}")
        print()

    # --- 3. the LR prediction --------------------------------------------
    if lr_pred is not None:
        pk, prof_pred = find_ring_radii(lr_pred, n_peaks=2,
                                        min_separation_px=2.0,
                                        exclude_center_px=1.0)
        radii = sorted(p[0] for p in pk)
        out["pred_ring_radii_px"] = radii
        out["pred_ring_radii_arcsec"] = [r * lr_pixel_scale for r in radii]

        print("[3] LR PREDICTION")
        for k, r in enumerate(radii):
            print(f"    ring {k + 1}                 : {r:.3f} px  "
                  f"=  {r * lr_pixel_scale:.4f} arcsec")
        if len(radii) == 2:
            r_in, r_out_ = radii
            half_sep = (r_out_ - r_in) / 2.0 * lr_pixel_scale
            mid = (r_out_ + r_in) / 2.0 * lr_pixel_scale
            out["pred_half_separation_arcsec"] = half_sep
            out["pred_mid_radius_arcsec"] = mid
            print(f"    half-separation        : {half_sep:.4f} arcsec"
                  "   (should equal the donut radius)")
            print(f"    mid radius             : {mid:.4f} arcsec"
                  "   (should equal theta_E)")
        print()

    # --- 4. verdict -------------------------------------------------------
    print(line)
    print("VERDICT")
    print(line)

    def _agree(a, b):
        if a is None or b is None:
            return False
        scale = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / scale < tol_frac

    verdicts = []

    r_src_as = out.get("r_source_arcsec")
    predicted = out.get("theta_e_mismatch_arcsec")

    # check (4): deflection ~ zero -> units bug
    if r_src_as is not None and _agree(r_src_as, r_data_as):
        verdicts.append(
            "UNITS BUG (sub-case A2). The source donut radius matches the data\n"
            "  ring radius, meaning the deflection applied is ~zero. Your theta_E is\n"
            "  almost certainly being consumed as PIXELS while you supply ARCSEC.\n"
            "  Check the unit conversion in the matrix-generation code."
        )
    # check (1): donut radius matches |R - theta_E|
    elif r_src_as is not None and predicted is not None and _agree(r_src_as, predicted):
        verdicts.append(
            f"THETA_E MISMATCH (hypothesis A) -- CONFIRMED QUANTITATIVELY.\n"
            f"  Measured donut radius {r_src_as:.4f}\" matches the predicted\n"
            f"  |r_data - theta_E| = {predicted:.4f}\" to within {tol_frac:.0%}.\n"
            f"  FIX: set theta_E = {r_data_as:.4f} arcsec "
            f"(= {r_data_px:.3f} px x {lr_pixel_scale} arcsec/px)."
        )
    elif r_src_as is not None and out.get("source_is_donut"):
        verdicts.append(
            "Source IS a donut, but its radius does not match |r_data - theta_E|.\n"
            "  This is evidence AGAINST a simple theta_E mismatch. Suspect hypothesis B:\n"
            "  magnification weighting inside the operator, so you are rendering the\n"
            "  critical curves rather than the lensed source.\n"
            "  TEST: push a synthetic OFF-CENTRE point source through the forward\n"
            "  operator. If the rings do not move, it is hypothesis B."
        )

    # check (2) and (3)
    if "pred_half_separation_arcsec" in out and r_src_as is not None:
        if _agree(out["pred_half_separation_arcsec"], r_src_as):
            verdicts.append(
                "Double ring is CONFIRMED to be the donut forward-lensed\n"
                "  (half-separation == donut radius). These are one defect, not two.\n"
                "  Do not debug them separately."
            )
    if "pred_mid_radius_arcsec" in out and theta_e is not None:
        if _agree(out["pred_mid_radius_arcsec"], theta_e):
            verdicts.append(
                "Ring mid-radius == assumed theta_E, as expected. The lens model is\n"
                "  being applied at the radius you asked for; the problem is the VALUE\n"
                "  of theta_E, not its application."
            )

    if not verdicts:
        verdicts.append(
            "No single hypothesis matched cleanly. Re-run with the correct pixel\n"
            "  scales, and verify the grid-centre convention ((N-1)/2 vs N/2) used in\n"
            "  your matrix-generation code -- a half-pixel offset will blur these tests."
        )

    for v in verdicts:
        print("* " + v)
    print(line)

    if plot:
        _plot(prof_data, out, lr_data, source_recon, lr_pred)

    return out


def _plot(prof_data, out, lr_data, source_recon, lr_pred):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib unavailable -- skipping plots)")
        return

    panels = [("LR data", lr_data)]
    if source_recon is not None:
        panels.append(("source recon", source_recon))
    if lr_pred is not None:
        panels.append(("LR prediction", lr_pred))

    fig, axes = plt.subplots(2, len(panels), figsize=(4 * len(panels), 7))
    if len(panels) == 1:
        axes = axes.reshape(2, 1)

    for k, (name, img) in enumerate(panels):
        img2d = _as_2d(img)
        axes[0, k].imshow(img2d, origin="lower")
        axes[0, k].set_title(name)
        r_bin, prof = radial_profile(img2d)
        axes[1, k].plot(r_bin, prof)
        axes[1, k].set_xlabel("radius [px]")
        axes[1, k].set_ylabel("mean flux")
        axes[1, k].grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Self-test on a synthetic case with a KNOWN theta_E mismatch, so you can
    # confirm the diagnostic reports what it should before trusting it on real data.
    N, scale = 64, 0.168
    true_theta_e = 1.2       # arcsec -- what the "data" really has
    model_theta_e = 0.75     # arcsec -- what the "matrices" wrongly assume

    yy, xx = np.indices((N, N))
    cy = cx = (N - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) * scale

    def ring(radius, width=0.12):
        return np.exp(-0.5 * ((r - radius) / width) ** 2)

    fake_data = ring(true_theta_e)
    beta0 = abs(true_theta_e - model_theta_e)          # 0.45"
    fake_source = ring(beta0)                           # the donut
    fake_pred = ring(model_theta_e + beta0) + ring(abs(model_theta_e - beta0))

    print("SELF-TEST: true theta_E = 1.2\", model theta_E = 0.75\","
          " expected donut radius = 0.45\"\n")
    diagnose(fake_data, fake_source, fake_pred,
             theta_e=model_theta_e,
             lr_pixel_scale=scale,
             source_pixel_scale=scale)
