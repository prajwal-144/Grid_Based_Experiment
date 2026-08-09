"""
test_lens_symmetry.py -- is the lens in this dataset circular, or not?

THE QUESTION
------------
eval_sis_v3 showed skill RISING with |r_data - alpha| (0.505 -> 0.815) and a
negative bank headroom (-0.104). So a per-image Einstein radius is not the
missing ingredient. The proposed explanation is that the lens is elliptical
(SIE) rather than circular (SIS). This script tests that directly, on the data,
with no model and no training.

WHY THE ARGUMENT IS TESTABLE
----------------------------
For a CIRCULAR lens with a single source at offset beta:

  * there are at most TWO images
  * they lie on exactly OPPOSITE sides of the lens centre (180 degrees apart)
  * they sit at radii alpha+beta and alpha-beta, so tracing the ring's radius
    around azimuth, r(phi), gives a pure m=1 modulation -- one side further
    out, the opposite side further in
  * azimuthal asymmetry REQUIRES beta > 0, which necessarily moves the image
    radii away from alpha

For an ELLIPTICAL lens (or a circular lens plus external shear):

  * up to FOUR images, and fold pairs sit CLOSE together, not 180 apart
  * the tangential critical curve is an ellipse, so r(phi) carries m=2
  * strong azimuthal structure coexists with mean radius ~ theta_E

So four independent discriminators, in decreasing order of how conclusive they
are:

  [A] number of distinct arcs. More than two is impossible for a circular lens.
      This one is a proof, not a hint.
  [B] angular separation of the two brightest arcs. Circular => 180 exactly.
  [C] m=2 / m=1 amplitude ratio of r(phi). Circular => m1 dominates.
  [D] axis ratio q from the ring's flux-weighted second moments. Circular => 1.

The script reports all four over the whole set and, separately, over the
LOW-MISMATCH subset (|r_data - alpha| < 1). That subset is the crux: those are
the images a circular SIS should fit best and instead fits worst. If they turn
out to be strongly asymmetric, a circular lens cannot describe them, and that
is the whole argument in one number.

USAGE
-----
    python test_lens_symmetry.py --split val/ --classes no_sub --n 200
    python test_lens_symmetry.py --n 500 --alpha 8.139 --out-dir lens_symmetry
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EPS = 1e-8
LINE = "=" * 78


# ---------------------------------------------------------------------------
# ring extraction
# ---------------------------------------------------------------------------

def polar_ridge(img, n_ang=72, r_min=1.5, r_max=None):
    """Trace the ring: for each azimuth, the radius of peak brightness.

    Returns (phi, r_of_phi, intensity_of_phi). Azimuths where the image is dark
    carry a meaningless radius, so intensity is returned as a weight and every
    downstream fit is weighted by it rather than treating all azimuths equally.
    """
    a = np.clip(np.squeeze(np.asarray(img, dtype=np.float64)), 0, None)
    h, w = a.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r_max = r_max or (min(h, w) / 2.0 - 1.5)

    phi = np.linspace(-np.pi, np.pi, n_ang, endpoint=False)
    radii = np.arange(r_min, r_max, 0.5)
    yy = cy + radii[None, :] * np.sin(phi[:, None])
    xx = cx + radii[None, :] * np.cos(phi[:, None])

    # bilinear sample
    y0 = np.floor(yy).astype(int); x0 = np.floor(xx).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1); x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1); x0 = np.clip(x0, 0, w - 1)
    fy = yy - y0; fx = xx - x0
    vals = (a[y0, x0] * (1 - fy) * (1 - fx) + a[y1, x0] * fy * (1 - fx)
            + a[y0, x1] * (1 - fy) * fx + a[y1, x1] * fy * fx)

    j = np.argmax(vals, axis=1)
    return phi, radii[j], vals[np.arange(n_ang), j]


def count_arcs(intensity, phi, frac=0.35, min_sep_deg=25.0):
    """[A] Number of distinct azimuthal arcs above `frac` of the peak.

    A circular lens can produce at most two. Three or more is proof of a
    non-circular deflection field (or of substructure/multiple sources).
    """
    n = len(intensity)
    peak = intensity.max()
    if peak <= EPS:
        return 0, []
    thr = frac * peak
    min_sep = int(round(min_sep_deg / 360.0 * n))
    work = intensity.copy()
    found = []
    while True:
        i = int(np.argmax(work))
        if work[i] < thr:
            break
        found.append(i)
        lo = np.arange(i - min_sep, i + min_sep + 1) % n
        work[lo] = -np.inf
        if len(found) >= 6:
            break
    return len(found), [float(np.degrees(phi[i])) for i in found]


def top_two_separation(intensity, phi, frac=0.35):
    """[B] Angular separation of the two brightest arcs, in degrees (0-180)."""
    n_arcs, angles = count_arcs(intensity, phi, frac=frac)
    if n_arcs < 2:
        return float("nan")
    d = abs(angles[0] - angles[1]) % 360.0
    return float(min(d, 360.0 - d))


def fourier_rphi(phi, r, weight):
    """[C] Weighted least-squares fit r(phi) = r0 + m1 + m2. Returns (r0,a1,a2).

    m=1 is what a circular lens produces from a source offset (one side out,
    the opposite side in). m=2 is the signature of an elliptical critical
    curve, which a circular lens cannot generate at any source position.
    """
    w = np.clip(weight, 0, None)
    if w.sum() <= EPS:
        return float("nan"), float("nan"), float("nan")
    w = w / w.sum()
    A = np.stack([np.ones_like(phi), np.cos(phi), np.sin(phi),
                  np.cos(2 * phi), np.sin(2 * phi)], axis=1)
    W = np.sqrt(w)[:, None]
    try:
        coef, *_ = np.linalg.lstsq(A * W, r * W[:, 0], rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan")
    return (float(coef[0]), float(np.hypot(coef[1], coef[2])),
            float(np.hypot(coef[3], coef[4])))


def ring_axis_ratio(img, r0, width=2.0):
    """[D] Axis ratio q and PA from flux-weighted second moments of the ring.

    Only meaningful for a reasonably complete ring; a pair of opposed arcs
    yields a spuriously small q. `completeness` is reported alongside so those
    can be filtered out.
    """
    a = np.clip(np.squeeze(np.asarray(img, dtype=np.float64)), 0, None)
    h, w = a.shape
    yy, xx = np.indices(a.shape)
    dy = yy - (h - 1) / 2.0; dx = xx - (w - 1) / 2.0
    r = np.hypot(dy, dx)
    m = (np.abs(r - r0) < width) & (a > 0)
    if m.sum() < 12:
        return float("nan"), float("nan")
    f = a[m]; Y = dy[m]; X = dx[m]
    tot = f.sum()
    if tot <= EPS:
        return float("nan"), float("nan")
    qxx = float((f * X * X).sum() / tot)
    qyy = float((f * Y * Y).sum() / tot)
    qxy = float((f * X * Y).sum() / tot)
    tr = qxx + qyy
    det = qxx * qyy - qxy * qxy
    disc = max(tr * tr / 4.0 - det, 0.0)
    l1 = tr / 2.0 + np.sqrt(disc)
    l2 = tr / 2.0 - np.sqrt(disc)
    if l1 <= EPS or l2 < 0:
        return float("nan"), float("nan")
    q = float(np.sqrt(l2 / l1))
    pa = float(np.degrees(0.5 * np.arctan2(2 * qxy, qxx - qyy)))
    return q, pa


def completeness(intensity, frac=0.35):
    """Fraction of azimuths brighter than `frac` of the peak. 1 = full ring."""
    peak = intensity.max()
    if peak <= EPS:
        return 0.0
    return float((intensity >= frac * peak).mean())


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val/")
    ap.add_argument("--classes", nargs="+", default=["no_sub"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=8.139,
                    help="operator alpha in LR px, used to define the "
                         "low-mismatch subset")
    ap.add_argument("--n-ang", type=int, default=72)
    ap.add_argument("--peak-frac", type=float, default=0.35)
    ap.add_argument("--out-dir", default="lens_symmetry")
    a = ap.parse_args()

    import data as data_mod
    ds = data_mod.LensingDataset(a.split, a.classes, a.n)
    n = min(len(ds), a.n * len(a.classes))
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(LINE)
    print(f"LENS SYMMETRY TEST -- {n} images from {a.split}{a.classes}")
    print(LINE)

    recs = []
    for i in range(n):
        img = np.squeeze(ds[i].numpy())
        if img.ndim != 2:
            continue
        phi, rphi, ival = polar_ridge(img, n_ang=a.n_ang)
        n_arcs, angles = count_arcs(ival, phi, frac=a.peak_frac)
        sep = top_two_separation(ival, phi, frac=a.peak_frac)
        r0, m1, m2 = fourier_rphi(phi, rphi, ival)
        comp = completeness(ival, frac=a.peak_frac)
        q, pa = ring_axis_ratio(img, r0 if np.isfinite(r0) else np.median(rphi))
        recs.append({"index": i, "r0_px": r0, "m1_px": m1, "m2_px": m2,
                     "n_arcs": n_arcs, "sep_deg": sep, "q": q, "pa_deg": pa,
                     "completeness": comp, "arc_angles_deg": angles})
        if i % 200 == 0:
            print(f"  {i}/{n}", end="\r")

    def col(k):
        return np.array([r[k] for r in recs], dtype=float)

    r0 = col("r0_px"); m1 = col("m1_px"); m2 = col("m2_px")
    narc = col("n_arcs"); sep = col("sep_deg"); q = col("q"); comp = col("completeness")
    mism = np.abs(r0 - a.alpha)
    low = mism < 1.0

    def block(mask, title):
        if mask.sum() < 3:
            print(f"\n  {title}: too few images ({int(mask.sum())})")
            return {}
        s = {}
        print(f"\n  {title}  (n = {int(mask.sum())})")
        s["frac_gt2_arcs"] = float((narc[mask] > 2).mean())
        print(f"    [A] images with >2 arcs      : {s['frac_gt2_arcs']:.1%}"
              "     (circular lens: impossible)")
        v = sep[mask][np.isfinite(sep[mask])]
        if v.size:
            s["median_sep_deg"] = float(np.median(v))
            s["frac_sep_far_from_180"] = float((np.abs(v - 180) > 30).mean())
            print(f"    [B] median arc separation    : {s['median_sep_deg']:.1f} deg"
                  "        (circular lens: 180)")
            print(f"        more than 30 deg off 180 : "
                  f"{s['frac_sep_far_from_180']:.1%}")
        ok = np.isfinite(m1[mask]) & np.isfinite(m2[mask]) & (m1[mask] > 1e-3)
        if ok.sum():
            ratio = m2[mask][ok] / m1[mask][ok]
            s["median_m2_over_m1"] = float(np.median(ratio))
            s["median_m1_px"] = float(np.median(m1[mask][ok]))
            s["median_m2_px"] = float(np.median(m2[mask][ok]))
            print(f"    [C] median m1 amplitude      : {s['median_m1_px']:.2f} px")
            print(f"        median m2 amplitude      : {s['median_m2_px']:.2f} px")
            print(f"        median m2/m1             : {s['median_m2_over_m1']:.2f}"
                  "        (circular lens: << 1)")
        # q is only trustworthy on reasonably complete rings
        qm = mask & (comp > 0.5) & np.isfinite(q)
        if qm.sum() >= 3:
            s["median_q_complete_rings"] = float(np.median(q[qm]))
            s["n_complete_rings"] = int(qm.sum())
            print(f"    [D] median axis ratio q      : "
                  f"{s['median_q_complete_rings']:.3f}"
                  f"   over {int(qm.sum())} complete rings   (circular lens: 1.0)")
        else:
            print("    [D] too few complete rings for a reliable axis ratio")
        return s

    all_stats = block(np.ones(len(recs), bool), "ALL IMAGES")
    low_stats = block(low, f"LOW-MISMATCH SUBSET  |r0 - {a.alpha:.2f}| < 1 px")

    print()
    print(LINE); print("VERDICT"); print(LINE)
    votes = []
    if all_stats.get("frac_gt2_arcs", 0) > 0.10:
        votes.append(f"[A] {all_stats['frac_gt2_arcs']:.0%} of images show more than two "
                     "arcs -- a circular lens cannot produce these at all")
    if all_stats.get("frac_sep_far_from_180", 0) > 0.30:
        votes.append(f"[B] {all_stats['frac_sep_far_from_180']:.0%} of image pairs are "
                     "more than 30 deg from opposed -- circular lensing forces 180")
    if all_stats.get("median_m2_over_m1", 0) > 0.6:
        votes.append(f"[C] m2/m1 = {all_stats['median_m2_over_m1']:.2f}; the ring's "
                     "radius varies with 2*phi, which is an elliptical critical curve")
    if 0 < all_stats.get("median_q_complete_rings", 1.0) < 0.9:
        votes.append(f"[D] rings are elongated, q = "
                     f"{all_stats['median_q_complete_rings']:.2f}")

    if votes:
        print("  NOT a circular lens. Evidence:")
        for v in votes:
            print("    * " + v)
        print("\n  Move to SIE. Keep matrices_v3 as the q=1 regression test:")
        print("  an SIE operator at q=1.0 must reproduce alpha=8.139, the fold,")
        print("  and the 1e-4 px round trip exactly, or the deflection has a")
        print("  convention bug.")
    else:
        print("  Consistent with a CIRCULAR lens on every test.")
        print("  The SIE hypothesis is NOT supported, so do not restructure the")
        print("  operator layer. The eval_sis_v3 failure must come from something")
        print("  else -- most likely the unregularised source (mu_weight=0 leaves")
        print("  the inversion with no prior at all, and the HR source panels show")
        print("  isolated spikes), the PSF, or the per-image normalisation.")

    if low.sum() >= 3 and "median_m2_over_m1" in low_stats:
        print()
        print("  The low-mismatch subset is the crux. Those are the images a")
        print("  circular SIS should fit BEST, and eval_sis_v3 fits them WORST")
        print("  (skill 0.505). If they are asymmetric above, that is the whole")
        print("  argument: asymmetry with mean radius ~ alpha is a configuration")
        print("  a circular lens cannot make at any source position.")

    (out_dir / "symmetry.json").write_text(json.dumps(
        {"split": a.split, "classes": a.classes, "alpha": a.alpha,
         "all": all_stats, "low_mismatch": low_stats, "per_image": recs}, indent=2))
    print(f"\n  wrote {out_dir / 'symmetry.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].hist(narc, bins=np.arange(-0.5, 6.5, 1))
    ax[0, 0].axvline(2.5, color="r", ls="--", label="circular limit")
    ax[0, 0].set_xlabel("distinct arcs"); ax[0, 0].set_title("[A] arc count")
    ax[0, 0].legend()
    v = sep[np.isfinite(sep)]
    ax[0, 1].hist(v, bins=30)
    ax[0, 1].axvline(180, color="r", ls="--", label="circular = 180")
    ax[0, 1].set_xlabel("separation of two brightest arcs [deg]")
    ax[0, 1].set_title("[B] arc opposition"); ax[0, 1].legend()
    ok = np.isfinite(m1) & np.isfinite(m2)
    ax[1, 0].scatter(m1[ok], m2[ok], s=9, alpha=0.6)
    lim = [0, float(np.nanpercentile(np.concatenate([m1[ok], m2[ok]]), 98))]
    ax[1, 0].plot(lim, lim, "r--", label="m2 = m1")
    ax[1, 0].set_xlabel("m=1 amplitude of r(phi) [px]")
    ax[1, 0].set_ylabel("m=2 amplitude [px]")
    ax[1, 0].set_title("[C] source offset (m1) vs ellipticity (m2)")
    ax[1, 0].legend()
    qm = np.isfinite(q) & (comp > 0.5)
    if qm.sum():
        ax[1, 1].hist(q[qm], bins=30)
        ax[1, 1].axvline(1.0, color="r", ls="--", label="circular")
        ax[1, 1].legend()
    ax[1, 1].set_xlabel("ring axis ratio q (complete rings only)")
    ax[1, 1].set_title("[D] ring ellipticity")
    fig.tight_layout(); fig.savefig(out_dir / "symmetry.png", dpi=110)
    print(f"  wrote {out_dir / 'symmetry.png'}")


if __name__ == "__main__":
    main()
