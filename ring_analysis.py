"""
ring_analysis.py -- quantify the fixed-SIS ceiling, and index rings for a bank.

TWO SUBCOMMANDS
---------------

  analyze : read an evaluate_sis.py metrics.json and bin every image by its
            lens-model mismatch  |r_data - alpha|.  This answers, WITHOUT any
            retraining, the question "how much would a per-image Einstein
            radius actually buy me?"  If skill in the smallest-mismatch bin is
            already high, the operator is fine and the only remaining problem
            is that one alpha cannot describe every image.

  index   : measure the ring radius of every image in a split and write an
            index file.  Needed to route images to the right operator in a
            bank, and to choose the bin edges sensibly rather than guessing.

WHY THIS MATTERS RIGHT NOW
--------------------------
Moving the forward operator from scatter (primary image only) to gather (both
SIS images) made the physics correct and the median skill WORSE, 0.726 -> 0.600.
That is not a regression. A fixed-SIS inversion with the wrong theta_E puts the
source at beta = |r_data - alpha| instead of at beta ~ 0.  With a primary-only
forward map that error was partly absorbed.  With a correct two-image map the
same error re-emits a counter-image at radius |alpha - beta|, which the data
does not contain and the network cannot remove.

Worked example from eval_sis_v3/metrics.json, image #2:
    r_data = 15.28 px,  alpha = 8.14 px
    beta   = |15.28 - 8.14| = 7.14 px      (measured source radius: 7.21)
    counter-image radius = |alpha - beta| = 0.93 px
and a compact blob is visible at the centre of that prediction panel.

So the counter-image is behaving exactly as SIS geometry requires. It is a
misspecification detector: it only appears when alpha is wrong for that image.

USAGE
-----
    python ring_analysis.py analyze --metrics eval_sis_v3/metrics.json
    python ring_analysis.py analyze --metrics eval_sis_v3/metrics.json \
        --compare eval_sis_v2/metrics.json
    python ring_analysis.py index --split train/ --classes no_sub \
        --n 5000 --out ring_index_train.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EPS = 1e-8
LINE = "=" * 78


# ---------------------------------------------------------------------------

def radial_profile(img, n_bins=None):
    a = np.squeeze(np.asarray(img, dtype=np.float64))
    h, w = a.shape
    yy, xx = np.indices(a.shape)
    r = np.hypot(yy - (h - 1) / 2.0, xx - (w - 1) / 2.0)
    rmax = r.max()
    n_bins = n_bins or max(8, int(np.ceil(rmax)))
    edges = np.linspace(0.0, rmax, n_bins + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins).astype(float)
    tot = np.bincount(idx, weights=a.ravel(), minlength=n_bins)
    prof = np.full(n_bins, np.nan)
    prof[cnt > 0] = tot[cnt > 0] / cnt[cnt > 0]
    return 0.5 * (edges[:-1] + edges[1:]), prof


def peak_radius(img, exclude_center_px=1.0):
    r, prof = radial_profile(img)
    w = np.where(np.isfinite(prof), prof, -np.inf)
    if exclude_center_px > 0:
        w[r < exclude_center_px] = -np.inf
    i = int(np.argmax(w))
    if 0 < i < len(prof) - 1 and np.all(np.isfinite(prof[i - 1:i + 2])):
        y0, y1, y2 = prof[i - 1:i + 2]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            return float(r[i] + np.clip(0.5 * (y0 - y2) / den, -1, 1) * (r[1] - r[0]))
    return float(r[i])


def ring_quality(img, r_peak):
    """How ring-like is this image? Azimuthal uniformity of the peak annulus.

    Returns a value in [0,1]; 1 means a complete, uniform Einstein ring, low
    values mean a few disconnected arcs. A radius measured on a low-quality
    image is unreliable, so bank routing should treat those separately.
    """
    a = np.squeeze(np.asarray(img, dtype=np.float64))
    h, w = a.shape
    yy, xx = np.indices(a.shape)
    dy, dx = yy - (h - 1) / 2.0, xx - (w - 1) / 2.0
    r = np.hypot(dy, dx)
    sel = np.abs(r - r_peak) < 1.5
    if sel.sum() < 12:
        return 0.0
    ang = np.arctan2(dy[sel], dx[sel])
    vals = np.clip(a[sel], 0, None)
    nb = 12
    b = np.clip(((ang + np.pi) / (2 * np.pi) * nb).astype(int), 0, nb - 1)
    s = np.bincount(b, weights=vals, minlength=nb)
    c = np.bincount(b, minlength=nb).clip(1)
    m = s / c
    if m.max() <= EPS:
        return 0.0
    return float(m.min() / m.max())


# ---------------------------------------------------------------------------

BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 1e9)]


def cmd_analyze(a):
    d = json.loads(Path(a.metrics).read_text())
    alpha = a.alpha or abs(d["operator"]["backward_px"])
    rows = d["per_image"]
    print(LINE)
    print(f"MISMATCH ANALYSIS -- {a.metrics}")
    print(f"  operator alpha = {alpha:.3f} px   ({len(rows)} images)")
    print(LINE)

    mism = np.array([abs(r["ring_lr_px"] - alpha) for r in rows])
    skill = np.array([r["skill"] for r in rows])
    comp = np.array([r["source_lr_compactness"] for r in rows])
    srad = np.array([r["source_lr_radius_px"] for r in rows])
    rpred = np.array([r["ring_pred_px"] for r in rows])
    rdata = np.array([r["ring_lr_px"] for r in rows])

    w1 = np.array([r.get("radial_w1_px", np.nan) for r in rows], dtype=float)
    nring = np.array([r.get("ring_pred_n", np.nan) for r in rows], dtype=float)
    has_new = np.isfinite(w1).any()
    if not has_new:
        print("  NOTE: this metrics.json predates the ring-metric fix. 'ring err'")
        print("  below uses the BRIGHTEST predicted annulus, which for a two-image")
        print("  operator often selects the counter-image. Re-run evaluate_sis.py")
        print("  to get the matched ring, W1 and ring-count columns.\n")

    print(f"  {'|r-alpha| bin':<16}{'n':>5}{'med skill':>11}{'p10':>9}"
          f"{'compact':>9}{'src rad':>9}{'ring err':>10}{'W1 px':>8}{'#rings':>8}")
    for lo, hi in BINS:
        m = (mism >= lo) & (mism < hi)
        if m.sum() == 0:
            continue
        lbl = f"{lo:.1f}-{hi:.1f}" if hi < 1e8 else f">{lo:.1f}"
        w1s = f"{np.nanmedian(w1[m]):>8.2f}" if has_new else f"{'-':>8}"
        nrs = f"{np.nanmedian(nring[m]):>8.1f}" if has_new else f"{'-':>8}"
        print(f"  {lbl:<16}{m.sum():>5}{np.median(skill[m]):>11.3f}"
              f"{np.percentile(skill[m],10):>9.3f}{np.median(comp[m]):>9.3f}"
              f"{np.median(srad[m]):>9.2f}"
              f"{np.nanmedian(rpred[m]-rdata[m]):>10.2f}{w1s}{nrs}")

    near = mism < 1.0
    print()
    print(f"  images within 1 px of alpha : {near.sum()} / {len(rows)}"
          f"  ({near.mean():.0%})")
    if near.sum() >= 5:
        print(f"  median skill THERE          : {np.median(skill[near]):.3f}")
        print(f"  median skill everywhere     : {np.median(skill):.3f}")
        gain = np.median(skill[near]) - np.median(skill)
        print(f"  headroom from a perfect bank: {gain:+.3f} median skill")
        print()
        if np.median(skill[near]) > 0.85:
            print("  READ: where the lens model is right, the pipeline already works.")
            print("  The operator is not the problem any more -- theta_E variation is.")
            print("  A bank of operators should recover most of the headroom above.")
        else:
            print("  READ: skill is mediocre even where alpha is correct, so something")
            print("  BESIDES theta_E mismatch is still wrong. Do NOT build the bank yet;")
            print("  inspect those low-mismatch images individually first.")

    if a.compare:
        d2 = json.loads(Path(a.compare).read_text())
        alpha2 = abs(d2["operator"]["backward_px"])
        rows2 = d2["per_image"]
        n = min(len(rows), len(rows2))
        s1 = np.array([r["skill"] for r in rows[:n]])
        s2 = np.array([r["skill"] for r in rows2[:n]])
        m1 = np.array([abs(r["ring_lr_px"] - alpha) for r in rows[:n]])
        print()
        print(LINE)
        print(f"COMPARISON vs {a.compare}")
        print(LINE)
        print(f"  {'|r-alpha| bin':<16}{'n':>5}{'this':>9}{'other':>9}{'delta':>9}")
        for lo, hi in BINS:
            m = (m1 >= lo) & (m1 < hi)
            if m.sum() == 0:
                continue
            lbl = f"{lo:.1f}-{hi:.1f}" if hi < 1e8 else f">{lo:.1f}"
            print(f"  {lbl:<16}{m.sum():>5}{np.median(s1[m]):>9.3f}"
                  f"{np.median(s2[m]):>9.3f}{np.median(s1[m])-np.median(s2[m]):>+9.3f}")
        print()
        print("  If 'this' loses mainly in the LARGE-mismatch bins, the change is")
        print("  exposing lens misspecification rather than causing damage: a")
        print("  two-image operator re-emits a counter-image whenever alpha is")
        print("  wrong, and only then.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    out = Path(a.metrics).parent / "mismatch.png"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(mism, skill, s=9, alpha=0.6)
    ax[0].set_xlabel(r"$|r_{data}-\alpha|$ [LR px]"); ax[0].set_ylabel("skill")
    ax[0].set_title("skill vs lens-model mismatch"); ax[0].grid(alpha=0.3)
    # Faint grey = EVERY predicted annulus, blue = the one matched to the data
    # ring. Without the grey cloud the matched points look artificially clean;
    # without the matching the counter-image masquerades as a wrong prediction.
    xs, ys = [], []
    for r in rows:
        for v in r.get("ring_pred_all_px", []):
            xs.append(r["ring_lr_px"]); ys.append(v)
    if xs:
        ax[1].scatter(xs, ys, s=6, alpha=0.25, color="0.6", label="all predicted rings")
    ax[1].scatter(rdata, rpred, s=9, alpha=0.7, label="matched to data ring")
    lim = [float(np.nanmin(rdata)), float(np.nanmax(rdata))]
    ax[1].plot(lim, lim, "r-", label="perfect")
    if xs:
        ax[1].plot(lim, [2 * alpha - v for v in lim], "g--", lw=1,
                   label=r"$2\alpha-r$ (counter-image)")
    ax[1].set_xlabel("data ring radius [LR px]")
    ax[1].set_ylabel("predicted ring radius [LR px]")
    ax[1].set_title("ring radius fidelity"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"\n  wrote {out}")


def cmd_index(a):
    """Index per-image THETA_E, not per-image ring radius.

    THIS IS THE CORRECTION THAT MATTERS. The brightest annulus sits at
    r_data = theta_E + beta, so routing a bank by peak_radius conflates the
    Einstein radius with the source offset -- an image with the right theta_E
    but a large source offset would be sent to the wrong operator.

    Decompose r(phi) instead. For a circular lens the two images lie at
    theta_E + beta and theta_E - beta on opposite sides, so tracing the ring's
    radius around azimuth gives

        m=0 term  ->  theta_E     (what the operator needs)
        m=1 term  ->  beta        (source offset; NOT a lens property)

    Verified on the near-complete rings (beta ~ 0), where theta_E must equal
    the observed radius directly: that subset gives theta_E = 3.15-16.94 px,
    a factor of 5.4, confirming theta_E genuinely varies in this dataset.
    """
    import data as data_mod
    from test_lens_symmetry import polar_ridge, fourier_rphi, completeness
    ds = data_mod.LensingDataset(a.split, a.classes, a.n)
    n = min(len(ds), a.n * len(a.classes))
    print(f"measuring {n} images from {a.split}{a.classes} ...")
    print("  routing on theta_E (m=0 of r(phi)), not on the brightest annulus")
    recs = []
    for i in range(n):
        img = np.squeeze(ds[i].numpy())
        if img.ndim != 2:
            continue
        phi, rphi, ival = polar_ridge(img)
        theta_e, beta, m2 = fourier_rphi(phi, rphi, ival)
        r_peak = peak_radius(img)
        recs.append({"index": i,
                     "theta_e_px": float(theta_e),
                     "beta_px": float(beta),
                     "m2_px": float(m2),
                     "ring_px": float(r_peak),
                     "completeness": float(completeness(ival)),
                     "ring_quality": ring_quality(img, r_peak)})
        if i % 500 == 0:
            print(f"  {i}/{n}", end="\r")

    rr = np.array([r["theta_e_px"] for r in recs])
    bb = np.array([r["beta_px"] for r in recs])
    ok = np.isfinite(rr)
    rr, bb = rr[ok], bb[ok]
    recs = [r for r, k in zip(recs, ok) if k]
    qq = np.array([r["ring_quality"] for r in recs])
    print(f"\n  beta (source offset): median {np.median(bb):.2f} px, "
          f"IQR {np.percentile(bb,25):.2f}-{np.percentile(bb,75):.2f}")
    print(f"  theta_E: median {np.median(rr):.2f} px, "
          f"IQR {np.percentile(rr,25):.2f}-{np.percentile(rr,75):.2f}, "
          f"range {rr.min():.2f}-{rr.max():.2f}")
    print(f"  ring quality: median {np.median(qq):.3f}, "
          f"{(qq < 0.1).mean():.0%} of images are arcs rather than rings")

    # Equal-population bin edges give every operator a similar amount of data.
    k = a.n_bins
    edges = np.percentile(rr, np.linspace(0, 100, k + 1))
    centres = [float(np.median(rr[(rr >= edges[j]) & (rr <= edges[j + 1])]))
               for j in range(k)]
    print(f"\n  suggested {k} operators (equal-population bins):")
    for j, c in enumerate(centres):
        m = (rr >= edges[j]) & (rr <= edges[j + 1])
        print(f"    alpha = {c:6.2f} px   covers {edges[j]:5.2f}-{edges[j+1]:5.2f} px"
              f"   n = {m.sum()}")
    print("\n  build them with:")
    print("    python build_sis_mappings.py --alpha-lr-px "
          + " ".join(f"{c:.2f}" for c in centres) + " --out-dir bank")

    Path(a.out).write_text(json.dumps(
        {"split": a.split, "classes": a.classes, "routing_key": "theta_e_px",
         "bin_edges": [float(e) for e in edges],
         "alphas": centres, "per_image": recs}, indent=2))
    print(f"\n  wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("analyze")
    p1.add_argument("--metrics", required=True)
    p1.add_argument("--alpha", type=float, default=None)
    p1.add_argument("--compare", default=None)
    p1.set_defaults(func=cmd_analyze)

    p2 = sub.add_parser("index")
    p2.add_argument("--split", default="train/")
    p2.add_argument("--classes", nargs="+", default=["no_sub"])
    p2.add_argument("--n", type=int, default=5000)
    p2.add_argument("--n-bins", type=int, default=6)
    p2.add_argument("--out", default="ring_index.json")
    p2.set_defaults(func=cmd_index)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()