"""
evaluate_sis.py -- honest evaluation of a trained fixed-SIS SR checkpoint.

WHY NOT THE test_mag_* NOTEBOOKS
--------------------------------
  * test_mag_mu_0.001.ipynb / test_mag_mu_0.01.ipynb each contain one cell that
    hardcodes  root = Path('matrices_orig')  and another that uses
    args['mapping_dir']. Those are not "old vs new matrices" -- one of them
    swaps the operator under frozen weights. That comparison is what made the
    reconstructions look broken when the matrices were fine (and vice versa).
  * test_mag_new_matrices_bundled.ipynb validates mapping metadata, but
    regenerate_mappings.py writes that metadata from command-line assertions
    without ever inspecting the matrix, so a passing check proves nothing.
  * All three evaluate a single image (VAL_INDEX=5) and report MSE only.

This script instead:
  * defaults to the checkpoint's OWN mapping_dir and shouts if you override it
  * MEASURES the operator it is about to use, before evaluating anything
  * evaluates over many images and reports a distribution, not one number
  * shows the full chain: LR / source_lr / source_hr / intrinsic / PSF / pred /
    residual, so a failure can be localised to a stage
  * reports source compactness and ring radii, which is what actually tells you
    whether the source reconstruction is right -- MSE on the re-degraded
    prediction can look fine while the source is a donut

USAGE
-----
    python evaluate_sis.py --checkpoint outputs_corrected/sis_v2/checkpoints/best.pt
    python evaluate_sis.py --checkpoint ... --n-images 200 --n-panels 6
    python evaluate_sis.py --checkpoint ... --out-dir eval_sis_v2

Outputs <out-dir>/panels.png, <out-dir>/skill_hist.png and
<out-dir>/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8
LINE = "=" * 78

RAW_NAMES = {
    "backward": "sparse_grid_fracs_euclid_backward.pt",
    "to_log": "scatter_to_log_128.pt",
    "forward_from_log": "forward_from_log_128.pt",
    "from_log": "scatter_from_log_128.pt",
}
BUNDLE_NAMES = {
    "backward": "sparse_grid_fracs_euclid_backward_bundle.pt",
    "to_log": "scatter_to_log_128_bundle.pt",
    "forward_from_log": "forward_from_log_128_bundle.pt",
    "from_log": "scatter_from_log_128_bundle.pt",
}


# ---------------------------------------------------------------------------

def load_dir(root: Path):
    out = {}
    for role in RAW_NAMES:
        for names in (RAW_NAMES, BUNDLE_NAMES):
            p = root / names[role]
            if p.exists():
                obj = torch.load(p, map_location="cpu", weights_only=False)
                meta = None
                if isinstance(obj, dict):
                    meta = obj.get("metadata")
                    obj = obj.get("matrix", obj.get("mapping"))
                out[role] = (obj.coalesce().float(), meta)
                break
    missing = set(RAW_NAMES) - set(out)
    if missing:
        raise FileNotFoundError(f"{root} is missing {sorted(missing)}")
    return out


def row_normalize(M: torch.Tensor) -> torch.Tensor:
    M = M.coalesce()
    idx, val = M.indices(), M.values().clamp_min(0)
    s = torch.zeros(M.shape[0], device=val.device, dtype=val.dtype)
    s.scatter_add_(0, idx[0], val)
    return torch.sparse_coo_tensor(idx, val / s[idx[0]].clamp_min(EPS),
                                   M.shape, device=M.device).coalesce()


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


def peak_radius(img, exclude_center_px=0.0):
    """Radius of the brightest annulus. exclude_center_px=0 keeps r~0 eligible,
    so a source that collapsed to a point correctly reports ~0."""
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


def compactness(img, radius_px=4.0):
    """Fraction of positive flux inside radius_px of the centre.

    For a correctly inverted source this should be HIGH. A donut drives it down.
    This is the metric the test_mag_* notebooks lacked: image-plane MSE can look
    healthy while the source is annular.
    """
    a = np.squeeze(np.asarray(img, dtype=np.float64))
    a = np.clip(a, 0, None)
    h, w = a.shape
    yy, xx = np.indices(a.shape)
    r = np.hypot(yy - (h - 1) / 2.0, xx - (w - 1) / 2.0)
    tot = a.sum()
    return float(a[r <= radius_px].sum() / tot) if tot > EPS else 0.0


def psnr(pred, target):
    mse = float(np.mean((np.asarray(pred) - np.asarray(target)) ** 2))
    rng = float(np.max(target) - np.min(target))
    if mse <= 0 or rng <= 0:
        return float("inf")
    return 20 * math.log10(rng) - 10 * math.log10(mse)


# ---------------------------------------------------------------------------

def check_operator(mats, lr_shape, hr_shape):
    """Measure the operator before trusting any result computed with it."""
    try:
        from diagnose_ring_geometry import column_centroids
    except Exception as exc:
        print(f"  (diagnose_ring_geometry unavailable: {exc}; operator NOT verified)")
        return None

    def m(seq, side):
        r_in, r_out, mass = column_centroids(seq, side, side)
        ok = np.isfinite(r_out) & (mass > 1e-6)
        half = side / 2.0
        band = ok & (r_in > 0.35 * half) & (r_in < 0.85 * half)
        if band.sum() < 20:
            return None
        return float(np.median(r_out[band] - r_in[band]))

    db = m([mats["backward"][0]], lr_shape)
    df = m([mats["to_log"][0], mats["forward_from_log"][0], mats["from_log"][0]], hr_shape)
    if db is None or df is None:
        print("  operator measurement inconclusive")
        return None
    df_lr = df / (hr_shape / lr_shape)
    print(f"  backward {db:+.3f} px   forward {df:+.3f} HR px ({df_lr:+.3f} LR px)"
          f"   round trip {db + df_lr:+.3f} px")
    if db * df > 0:
        print("  *** WARNING: same sign -- deflection applied twice. "
              "Results below are meaningless. ***")
    elif abs(db + df_lr) > 1.0:
        print(f"  *** WARNING: round trip does not close ({db + df_lr:+.2f} px). ***")
    else:
        print("  operator OK: opposite signs, round trip closes")
    return {"backward_px": db, "forward_lr_px": df_lr, "roundtrip_px": db + df_lr}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mapping-dir", default=None,
                    help="OVERRIDE the checkpoint's mapping_dir. Doing this "
                         "evaluates the model against an operator it was not "
                         "trained on; the script will warn loudly.")
    ap.add_argument("--split", default="val/")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="default: the classes the checkpoint was trained on")
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--n-panels", type=int, default=5)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import data as data_mod
    from differentiable_lensing import DifferentiableLensing
    from psf import apply_psf, build_psf_kernel
    from sisr import SISR

    dev = torch.device(a.device)
    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    args = ck["args"]
    classes = a.classes or args["classes"]
    mdir = a.mapping_dir or args.get("mapping_dir", ".")
    out_dir = Path(a.out_dir or f"eval_{Path(a.checkpoint).parents[1].name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(LINE)
    print(f"CHECKPOINT {a.checkpoint}")
    print(LINE)
    print(f"  trained mapping_dir : {args.get('mapping_dir')}")
    print(f"  using mapping_dir   : {mdir}")
    if a.mapping_dir and a.mapping_dir != args.get("mapping_dir"):
        print("\n  *** OPERATOR OVERRIDE ***")
        print("  You are evaluating frozen weights against a different operator")
        print("  than they were trained with. This is the exact mistake the")
        print("  test_mag_* notebooks made. Expect a large, meaningless drop.\n")
    print(f"  resolution {args['resolution']}   theta_e {args.get('theta_e')}"
          f"   -> {args.get('theta_e', 0) / max(args['resolution'], EPS):.3f} px")
    print(f"  mu_weight {args['mu_weight']}   epochs {args['epochs']}"
          f"   classes {classes}\n")

    mats = load_dir(Path(mdir))
    for role, (M, meta) in mats.items():
        if meta:
            print(f"  {role:<18} metadata theta_E={meta.get('theta_e_arcsec')} "
                  f"alpha_lr_px={meta.get('alpha_lr_px')}")
    op = check_operator(mats, args["image_shape"], args["target_shape"])
    declared = args.get("theta_e", 0) / max(args["resolution"], EPS)
    if op and abs(abs(op["backward_px"]) - declared) > 0.5:
        print(f"\n  *** theta_e/resolution = {declared:.2f} px but the matrix "
              f"encodes {abs(op['backward_px']):.2f} px. The magnification/info "
              f"map disagrees with the operator. ***")
    print()

    model = SISR(args["magnification"], args["n_mag"], args["residual_depth"],
                 in_channels=2, latent_channel_count=args["latent_space_size"]).to(dev)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    lens = DifferentiableLensing(device=dev, alpha=None,
                                 target_resolution=args["target_resolution"],
                                 target_shape=args["target_shape"]).to(dev)
    bavg = row_normalize(mats["backward"][0]).to(dev)
    chain = [mats["to_log"][0].to(dev), mats["forward_from_log"][0].to(dev),
             mats["from_log"][0].to(dev)]

    psf = None
    try:
        psf = build_psf_kernel("fits", 0.16, args["target_resolution"],
                               path=args["psf_path"],
                               source_pixscale_arcsec=args["psf_source_pixscale_arcsec"],
                               device=dev)
    except Exception as exc:
        print(f"  PSF unavailable ({exc}); evaluating without convolution\n")

    ds = data_mod.LensingDataset(a.split, classes, a.n_images)
    n = min(len(ds), a.n_images * len(classes))
    print(f"  evaluating {n} images from {a.split}{classes}\n")

    rows, panels = [], []
    for i in range(n):
        lr = ds[i].unsqueeze(0).float().to(dev)
        if lr.ndim == 3:
            lr = lr.unsqueeze(1)
        if lr.ndim == 5:
            lr = lr.squeeze(1)
        with torch.no_grad():
            flat = lr.reshape(lr.shape[0] * lr.shape[1], -1).T
            src_lr = torch.sparse.mm(bavg, flat).T.reshape(
                1, 1, args["image_shape"], args["image_shape"])
            src_hr = model(torch.cat([src_lr, lr], dim=1))
            intrinsic = lens.cross_grid_fill(src_hr, chain)
            conv = apply_psf(intrinsic, psf) if psf is not None else intrinsic
            pred = F.interpolate(conv, size=lr.shape[-2:], mode="area")

        L = lr[0, 0].cpu().numpy(); P = pred[0, 0].cpu().numpy()
        SL = src_lr[0, 0].cpu().numpy(); SH = src_hr[0, 0].cpu().numpy()
        IN = intrinsic[0, 0].cpu().numpy(); CV = conv[0, 0].cpu().numpy()

        zero = float(np.mean(L ** 2)); mse = float(np.mean((P - L) ** 2))
        hr_over_lr = args["target_shape"] / args["image_shape"]
        rows.append({
            "index": i,
            "zero_mse": zero, "model_mse": mse,
            "skill": 1 - mse / max(zero, EPS),
            "psnr": psnr(P, L),
            "ring_lr_px": peak_radius(L, exclude_center_px=1.0),
            "ring_pred_px": peak_radius(P, exclude_center_px=1.0),
            "source_lr_radius_px": peak_radius(SL),
            "source_hr_radius_px": peak_radius(SH) / hr_over_lr,
            "source_lr_compactness": compactness(SL, 4.0),
            "source_hr_compactness": compactness(SH, 4.0 * hr_over_lr),
        })
        if len(panels) < a.n_panels:
            panels.append((i, L, SL, SH, IN, CV, P, P - L))

    # ---------------- summary ----------------
    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    print(LINE); print("SUMMARY"); print(LINE)
    print(f"  {'metric':<26}{'median':>10}{'mean':>10}{'p10':>10}{'p90':>10}")
    for k in ["skill", "psnr", "ring_lr_px", "ring_pred_px",
              "source_lr_radius_px", "source_hr_radius_px",
              "source_lr_compactness", "source_hr_compactness"]:
        v = col(k); v = v[np.isfinite(v)]
        print(f"  {k:<26}{np.median(v):>10.4f}{v.mean():>10.4f}"
              f"{np.percentile(v,10):>10.4f}{np.percentile(v,90):>10.4f}")

    resid = np.abs(col("ring_lr_px") - abs(op["backward_px"])) if op else None
    print()
    print(f"  ring radius spread (data)   : {np.percentile(col('ring_lr_px'),25):.2f}"
          f" - {np.percentile(col('ring_lr_px'),75):.2f} px  (IQR)")
    if resid is not None:
        print(f"  |r_data - alpha| predicted  : median {np.median(resid):.2f} px")
        print(f"  source radius measured      : median "
              f"{np.median(col('source_lr_radius_px')):.2f} px")
        print("  These two should agree. If they do, the residual annulus is the")
        print("  fixed-SIS ceiling (one alpha cannot fit a spread of Einstein")
        print("  radii) and an operator bank is the only way to reduce it.")
    print()

    (out_dir / "metrics.json").write_text(json.dumps(
        {"checkpoint": str(a.checkpoint), "mapping_dir": str(mdir),
         "operator": op, "per_image": rows}, indent=2))
    print(f"  wrote {out_dir / 'metrics.json'}")

    # ---------------- figures ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  matplotlib unavailable ({exc}); figures skipped")
        return

    titles = ["LR observation", "source (backward)", "source HR (network)",
              "intrinsic (forward)", "after PSF", "prediction", "residual"]
    fig, axes = plt.subplots(len(panels), 7, figsize=(21, 3 * len(panels)))
    if len(panels) == 1:
        axes = axes[None, :]
    for r, (idx, *imgs) in enumerate(panels):
        for c, (ax, im, t) in enumerate(zip(axes[r], imgs, titles)):
            cmap = "coolwarm" if c == 6 else "gray"
            if c == 6:
                m = np.abs(im).max()
                ax.imshow(im, origin="lower", cmap=cmap, vmin=-m, vmax=m)
            else:
                ax.imshow(im, origin="lower", cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(t, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"#{idx}\nskill {rows[idx]['skill']:.3f}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "panels.png", dpi=110)
    print(f"  wrote {out_dir / 'panels.png'}")

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
    ax2[0].hist(col("skill"), bins=30); ax2[0].set_xlabel("skill")
    ax2[0].set_title(f"skill (median {np.median(col('skill')):.3f})")
    ax2[1].scatter(col("ring_lr_px"), col("source_lr_radius_px"), s=8, alpha=0.6)
    if op:
        xs = np.linspace(col("ring_lr_px").min(), col("ring_lr_px").max(), 100)
        ax2[1].plot(xs, np.abs(xs - abs(op["backward_px"])), "r-",
                    label=r"$|r_{data}-\alpha|$")
        ax2[1].legend()
    ax2[1].set_xlabel("data ring radius [LR px]")
    ax2[1].set_ylabel("reconstructed source radius [LR px]")
    ax2[1].set_title("residual annulus vs prediction")
    fig2.tight_layout(); fig2.savefig(out_dir / "skill_hist.png", dpi=110)
    print(f"  wrote {out_dir / 'skill_hist.png'}")


if __name__ == "__main__":
    main()
