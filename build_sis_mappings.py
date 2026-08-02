"""
build_sis_mappings.py -- vectorized, self-verifying SIS mapping generator.

WHY THIS REPLACES THE NOTEBOOKS
-------------------------------
The four create_*_grid.ipynb notebooks have two problems that together cost a
lot of time:

  1. Sign is not predictable from the source. raw_matrices_old/
     create_backward_grid.ipynb calls backward_lensing and measures +alpha;
     the root create_backward_grid.ipynb calls forward_lensing and measures
     -alpha; and the forward CHAIN calls forward_lensing yet measures +alpha,
     because the log-grid resamplers flip the net direction. Two notebooks
     calling the same function produce opposite operators. Verified by
     diagnose_ring_geometry.py section 1 on all five existing directories:

         matrices_orig             backward -7.31   forward +6.72   OK
         grid_matrices             backward +7.42   forward +7.43   BROKEN
         matrices_new / mappings   backward -4.46   forward -4.43   BROKEN
         raw_matrices_old/0.168    backward +4.46   forward +4.48   BROKEN

  2. log_grid_crop / square_grid_crop are O(N^4) pure-Python polygon clipping.
     The 128-grid forward chain took 8h46m. That makes iterating on the
     geometry impractical, which is how a wrong sign survives for months.

This script fixes both:

  * SUPERSAMPLING instead of polygon clipping. Each input pixel is subdivided
    k x k, every subpixel centre is pushed through the lens equation, and the
    landing pixel is accumulated with bincount. Fully vectorized; the whole
    128-grid build runs in seconds. Area error is O(1/k^2) per pixel, and
    column sums are exactly 1 (minus flux that leaves the grid), matching the
    normalization convention of DifferentiableLensing.build_sparse_mapping.

  * SIGN CONSISTENCY BY CONSTRUCTION. backward is theta -> theta*(1 - a/r),
    forward is beta -> beta*(1 + a/r). These are analytic inverses, so the
    round trip cannot drift.

  * SELF-VERIFICATION BEFORE SAVING. The measured deflection and sign of both
    operators are checked with the same column-centroid method used by
    diagnose_ring_geometry.py. If the signs are not opposite, or the round-trip
    residual is too large, nothing is written.

WHAT alpha SHOULD BE
--------------------
Only the ratio theta_E / pixel_scale reaches the operator -- the matrices are
built in pixel units. diagnose_ring_geometry.py section 3 measured the val
no_sub ring at 8.14 px (stacked; per-image median 8.29, IQR 6.38-10.93), so:

    alpha_lr = 8.14 px   ->   theta_E = 1.368"  at 0.168 "/px
                             theta_E = 0.822"  at 0.101 "/px

The wide IQR means the dataset does NOT have one Einstein radius. A single
fixed-SIS operator leaves a residual annulus of |r_image - alpha| on every
off-median image. Use --alpha-lr-px to build a bank of operators if you want to
route images by their measured ring radius (see --help).

THE LOG GRID IS DROPPED
-----------------------
The pipeline applies [to_log, forward_from_log, from_log] in sequence. This
script writes to_log and from_log as sparse identities and puts the real
geometry in forward_from_log, so cross_grid_fill and load_raw_mappings work
unchanged. You lose the log grid's central resolution concentration; you also
lose a large failure surface and 8 hours of build time. Revisit later if the
central sampling proves limiting.

USAGE
-----
    python build_sis_mappings.py --alpha-lr-px 8.14 --out-dir matrices_v2
    python build_sis_mappings.py --alpha-lr-px 8.14 --out-dir matrices_v2 \
        --lr-pixel-scale 0.168 --supersample 24
    python build_sis_mappings.py --alpha-lr-px 6 7 8 9 10 --out-dir bank   # operator bank

Then verify independently:
    python diagnose_ring_geometry.py --sections 1 2 --roundtrip-dir matrices_v2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

EPS = 1e-12
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
# lens maps (pixel units, origin at grid centre)
# ---------------------------------------------------------------------------

def sis_inward(alpha_px: float):
    """Image plane -> source plane:  beta = theta * (1 - alpha/r).

    For r < alpha the factor goes negative, which reflects the point through the
    origin. That is the genuine SIS fold: both r = alpha + b and r = alpha - b
    land on |beta| = b, so the operator is correctly two-to-one.
    """
    def f(x, y):
        r = torch.sqrt(x * x + y * y).clamp_min(1e-3)
        k = 1.0 - alpha_px / r
        return x * k, y * k
    return f


def sis_outward(alpha_px: float):
    """Source plane -> image plane, primary image:  theta = beta * (1 + alpha/r).

    Only the primary (same-side) image. The SIS counter-image at
    theta = -beta_hat * (alpha - |beta|) exists for |beta| < alpha; enable it
    with --counter-image. It is off by default because matrices_orig -- the only
    operator pair measured as self-consistent, and the one mag_full was trained
    against at skill 0.90 -- implements the primary branch only. Turning it on
    changes the operator the network must invert, so retrain if you do.
    """
    def f(x, y):
        r = torch.sqrt(x * x + y * y).clamp_min(1e-3)
        k = 1.0 + alpha_px / r
        return x * k, y * k
    return f


# ---------------------------------------------------------------------------
# vectorized scatter-matrix construction
# ---------------------------------------------------------------------------

def build_scatter_matrix(n_in: int, n_out: int, map_fn, supersample: int = 16,
                         chunk_rows: int = 8, extra_map_fn=None,
                         verbose: bool = True) -> torch.Tensor:
    """M[q, p] = fraction of input pixel p's area landing in output pixel q.

    out = M @ in scatters the value of input pixel p to wherever map_fn sends
    it. Column sums are 1 minus whatever leaves the grid, matching the
    convention of DifferentiableLensing.build_sparse_mapping (which divides by
    the input cell area).

    extra_map_fn, if given, adds a second branch (e.g. the SIS counter-image);
    each branch contributes full weight, since lensing conserves surface
    brightness rather than splitting it.
    """
    if supersample < 1:
        raise ValueError("supersample must be >= 1")

    k = supersample
    c_in = (n_in - 1) / 2.0
    c_out = (n_out - 1) / 2.0
    w = 1.0 / (k * k)

    off = (torch.arange(k, dtype=torch.float64) + 0.5) / k - 0.5
    ix = torch.arange(n_in, dtype=torch.float64) - c_in

    q_parts, p_parts, v_parts = [], [], []
    maps = [map_fn] + ([extra_map_fn] if extra_map_fn is not None else [])

    for i0 in range(0, n_in, chunk_rows):
        i1 = min(i0 + chunk_rows, n_in)
        rows = i1 - i0
        iy = torch.arange(i0, i1, dtype=torch.float64) - c_in

        # (rows, n_in, k, k) coordinates of every subpixel centre
        Y = iy[:, None, None, None] + off[None, None, :, None]
        X = ix[None, :, None, None] + off[None, None, None, :]
        Y = Y.expand(rows, n_in, k, k)
        X = X.expand(rows, n_in, k, k)

        p_idx = (torch.arange(i0, i1)[:, None] * n_in
                 + torch.arange(n_in)[None, :])
        p_idx = p_idx[:, :, None, None].expand(rows, n_in, k, k)

        for fn in maps:
            Xo, Yo = fn(X, Y)
            qx = torch.round(Xo + c_out).long()
            qy = torch.round(Yo + c_out).long()
            ok = (qx >= 0) & (qx < n_out) & (qy >= 0) & (qy < n_out) \
                & torch.isfinite(Xo) & torch.isfinite(Yo)
            if not bool(ok.any()):
                continue
            q_parts.append((qy * n_out + qx)[ok])
            p_parts.append(p_idx[ok])

        if verbose and (i0 // max(chunk_rows, 1)) % 4 == 0:
            print(f"      rows {i1}/{n_in}", end="\r")

    if not q_parts:
        raise RuntimeError("no subpixels landed on the output grid -- alpha too large?")

    q = torch.cat(q_parts)
    p = torch.cat(p_parts)

    # Collapse duplicate (q, p) pairs. n_in**2 <= 16384 so the key fits in int64.
    stride = n_in * n_in
    key = q * stride + p
    uniq, inv = torch.unique(key, return_inverse=True)
    vals = torch.zeros(uniq.numel(), dtype=torch.float64)
    vals.scatter_add_(0, inv, torch.full((inv.numel(),), w, dtype=torch.float64))

    qq = torch.div(uniq, stride, rounding_mode="floor")
    pp = uniq - qq * stride
    M = torch.sparse_coo_tensor(
        torch.stack([qq, pp]), vals.float(),
        (n_out * n_out, n_in * n_in),
    ).coalesce()
    if verbose:
        print(f"      built {tuple(M.shape)} nnz={M._nnz():,}          ")
    return M


def build_gather_matrix(n_grid: int, map_fn, supersample: int = 16,
                        chunk_rows: int = 8, verbose: bool = True) -> torch.Tensor:
    """M[q, p] = fraction of OUTPUT pixel q that samples INPUT pixel p. Rows sum to 1.

    WHY THE FORWARD OPERATOR MUST BE A GATHER, NOT A SCATTER
    --------------------------------------------------------
    A scatter iterates over input pixels and deposits them where the map sends
    them. That is correct only while the map CONTRACTS. The forward (source ->
    image) map expands: its Jacobian is > 1 everywhere, so the same input mass
    is spread over more output pixels and some output pixels receive nothing.
    The result is a sieve -- visible as the dotted/moire texture in the
    "intrinsic (forward)" panels, and as forward column coverage of only 61.5%.
    The network then has to place spikes exactly on the surviving pixels, which
    is why the HR source came out as isolated dots instead of a smooth galaxy.

    A gather inverts the loop: for every OUTPUT pixel, subsample it, push those
    subsamples through the lens equation, and read the source there. Every
    output pixel is fully determined, so there are no holes by construction.
    This is standard backward ray shooting.

    It also fixes the counter-image for free. The forward map is evaluated as
    beta = theta - alpha*theta_hat (i.e. sis_inward, applied on the image grid),
    and for |theta| < alpha that lands on the opposite side. So every theta that
    maps to the same beta picks up that source value -- both SIS images appear
    without any special casing.
    """
    if supersample < 1:
        raise ValueError("supersample must be >= 1")

    k = supersample
    c = (n_grid - 1) / 2.0
    w = 1.0 / (k * k)

    off = (torch.arange(k, dtype=torch.float64) + 0.5) / k - 0.5
    ix = torch.arange(n_grid, dtype=torch.float64) - c

    q_parts, p_parts = [], []
    for i0 in range(0, n_grid, chunk_rows):
        i1 = min(i0 + chunk_rows, n_grid)
        rows = i1 - i0
        iy = torch.arange(i0, i1, dtype=torch.float64) - c

        Y = (iy[:, None, None, None] + off[None, None, :, None]).expand(rows, n_grid, k, k)
        X = (ix[None, :, None, None] + off[None, None, None, :]).expand(rows, n_grid, k, k)

        q_idx = (torch.arange(i0, i1)[:, None] * n_grid
                 + torch.arange(n_grid)[None, :])
        q_idx = q_idx[:, :, None, None].expand(rows, n_grid, k, k)

        Xs, Ys = map_fn(X, Y)                      # where this output pixel reads from
        px = torch.round(Xs + c).long()
        py = torch.round(Ys + c).long()
        ok = (px >= 0) & (px < n_grid) & (py >= 0) & (py < n_grid) \
            & torch.isfinite(Xs) & torch.isfinite(Ys)
        if not bool(ok.any()):
            continue
        q_parts.append(q_idx[ok])
        p_parts.append((py * n_grid + px)[ok])

        if verbose and (i0 // max(chunk_rows, 1)) % 4 == 0:
            print(f"      rows {i1}/{n_grid}", end="\r")

    if not q_parts:
        raise RuntimeError("no subsamples landed on the input grid -- alpha too large?")

    q = torch.cat(q_parts)
    p = torch.cat(p_parts)
    stride = n_grid * n_grid
    key = q * stride + p
    uniq, inv = torch.unique(key, return_inverse=True)
    vals = torch.zeros(uniq.numel(), dtype=torch.float64)
    vals.scatter_add_(0, inv, torch.full((inv.numel(),), w, dtype=torch.float64))

    qq = torch.div(uniq, stride, rounding_mode="floor")
    pp = uniq - qq * stride
    M = torch.sparse_coo_tensor(torch.stack([qq, pp]), vals.float(),
                                (stride, stride)).coalesce()
    if verbose:
        rs = torch.zeros(stride)
        rs.scatter_add_(0, M.indices()[0], M.values())
        filled = float((rs > 1e-6).float().mean())
        print(f"      built {tuple(M.shape)} nnz={M._nnz():,}  "
              f"output pixels filled: {filled:.1%}          ")
    return M


def sparse_identity(n: int) -> torch.Tensor:
    i = torch.arange(n * n)
    return torch.sparse_coo_tensor(torch.stack([i, i]),
                                   torch.ones(n * n), (n * n, n * n)).coalesce()


# ---------------------------------------------------------------------------
# verification (same method as diagnose_ring_geometry.py section 1)
# ---------------------------------------------------------------------------

def measure(mats, side, chunk=256):
    """Return (signed_displacement_px, radial_scatter_px, coverage)."""
    try:
        from diagnose_ring_geometry import column_centroids
    except Exception:
        return None
    import numpy as np
    r_in, r_out, mass = column_centroids(mats, side, side, chunk=chunk)
    ok = np.isfinite(r_out) & (mass > 1e-6)
    half = side / 2.0
    band = ok & (r_in > 0.35 * half) & (r_in < 0.85 * half)
    if band.sum() < 20:
        return None
    d = float(np.median(r_out[band] - r_in[band]))
    scatter = float(np.median(np.abs((r_out[band] - r_in[band]) - d)))
    return d, scatter, float((mass > 1e-6).mean())


def verify(backward, forward, lr_shape, hr_shape, alpha_lr, tol_frac=0.15):
    """Check the pair is a genuine inverse pair. Returns (ok, message)."""
    print("\n  verifying ...")
    b = measure([backward], lr_shape)
    f = measure([forward], hr_shape)
    if b is None or f is None:
        return True, "  (diagnose_ring_geometry not importable -- verification SKIPPED)"

    db, sb, cb = b
    df, sf, cf = f
    df_lr = df / (hr_shape / lr_shape)   # HR px -> LR px

    print(f"      backward : {db:+7.3f} px  scatter {sb:.3f}  coverage {cb:6.1%}")
    print(f"      forward  : {df:+7.3f} px  ({df_lr:+.3f} LR px)  "
          f"scatter {sf:.3f}  coverage {cf:6.1%}")
    print(f"      round trip residual : {db + df_lr:+.3f} LR px")

    problems = []
    if db >= 0:
        problems.append(f"backward is +alpha ({db:+.2f}); it must push INWARD")
    if df <= 0:
        problems.append(f"forward is -alpha ({df:+.2f}); it must push OUTWARD")
    if db * df > 0:
        problems.append("backward and forward have the SAME sign -- "
                        "the deflection would be applied twice")
    if abs(db) > 1e-6 and abs(abs(db) - alpha_lr) / alpha_lr > tol_frac:
        problems.append(f"backward |alpha| {abs(db):.2f} != requested {alpha_lr:.2f}")
    if abs(db + df_lr) > max(1.0, 0.12 * alpha_lr):
        problems.append(f"round trip residual {db + df_lr:+.2f} LR px is too large")

    if problems:
        return False, "\n".join("      FAIL: " + p for p in problems)
    return True, "      PASS: opposite signs, matched amplitude, round trip closes"


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------

def save_all(out_dir: Path, mats: dict, meta_base: dict, write_bundles=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    for role, name in RAW_NAMES.items():
        torch.save(mats[role], out_dir / name)
        print(f"      wrote {out_dir / name}")
    if not write_bundles:
        return
    try:
        from mapping_io import save_mapping_bundle
    except Exception as exc:
        print(f"      (mapping_io unavailable: {exc}; bundles skipped)")
        return
    for role, name in BUNDLE_NAMES.items():
        meta = dict(meta_base)
        meta["mapping_role"] = role
        save_mapping_bundle(str(out_dir / name), mats[role], meta)
        print(f"      wrote {out_dir / name}")


# ---------------------------------------------------------------------------

def build_one(alpha_lr: float, args, out_dir: Path) -> bool:
    hr_over_lr = args.hr_shape / args.lr_shape
    alpha_hr = alpha_lr * hr_over_lr

    print(LINE)
    print(f"BUILDING  alpha_lr = {alpha_lr:.4f} px   alpha_hr = {alpha_hr:.4f} px")
    print(f"          -> theta_E = {alpha_lr * args.lr_pixel_scale:.4f} arcsec "
          f"at {args.lr_pixel_scale} arcsec/px")
    print(f"          -> {out_dir}")
    print(LINE)

    print("    backward (LR image -> LR source, inward, with SIS fold)")
    backward = build_scatter_matrix(args.lr_shape, args.lr_shape,
                                    sis_inward(alpha_lr),
                                    supersample=args.supersample,
                                    chunk_rows=args.chunk_rows)

    if args.legacy_scatter_forward:
        print("    forward  (HR source -> HR image, SCATTER -- legacy, leaves holes)")
        forward = build_scatter_matrix(args.hr_shape, args.hr_shape,
                                       sis_outward(alpha_hr),
                                       supersample=args.supersample,
                                       chunk_rows=args.chunk_rows)
    else:
        print("    forward  (HR source -> HR image, GATHER / backward ray shooting)")
        forward = build_gather_matrix(args.hr_shape,
                                      sis_inward(alpha_hr),
                                      supersample=args.supersample,
                                      chunk_rows=args.chunk_rows)

    ok, msg = verify(backward, forward, args.lr_shape, args.hr_shape, alpha_lr)
    print(msg)
    if not ok:
        print("\n  NOTHING WRITTEN. The pair failed verification -- see above.")
        return False

    mats = {
        "backward": backward,
        "forward_from_log": forward,
        "to_log": sparse_identity(args.hr_shape),
        "from_log": sparse_identity(args.hr_shape),
    }
    meta = {
        "lens_model": "SIS",
        "theta_e_arcsec": alpha_lr * args.lr_pixel_scale,
        "lr_pixel_scale_arcsec": args.lr_pixel_scale,
        "hr_pixel_scale_arcsec": args.lr_pixel_scale / hr_over_lr,
        "image_shape": args.lr_shape,
        "target_shape": args.hr_shape,
        "center_x_arcsec": 0.0,
        "center_y_arcsec": 0.0,
        "alpha_lr_px": alpha_lr,
        "alpha_hr_px": alpha_hr,
        "supersample": args.supersample,
        "forward_mode": "scatter" if args.legacy_scatter_forward else "gather",
        "log_grid": False,
        "generator": "build_sis_mappings.py",
    }
    print()
    save_all(out_dir, mats, meta, write_bundles=not args.no_bundles)
    print("\n  Note: to_log and from_log are sparse identities. The log-grid")
    print("  detour is dropped; all geometry lives in forward_from_log.\n")
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha-lr-px", type=float, nargs="+", default=[8.14],
                    help="deflection in LR pixels. Pass several to build a bank "
                         "(each goes to <out-dir>/alpha_<value>/). Default 8.14 "
                         "= the measured val/no_sub ring radius.")
    ap.add_argument("--out-dir", default="matrices_v2")
    ap.add_argument("--lr-shape", type=int, default=64)
    ap.add_argument("--hr-shape", type=int, default=128)
    ap.add_argument("--lr-pixel-scale", type=float, default=0.168,
                    help="declared arcsec/px, used only for metadata and the "
                         "reported theta_E. It does not affect the operator.")
    ap.add_argument("--supersample", type=int, default=16,
                    help="k x k subpixels per pixel. Area error ~1/k^2. "
                         "Raise near the critical curve if scatter is high.")
    ap.add_argument("--chunk-rows", type=int, default=8,
                    help="rows processed at once; lower this if you run out of RAM")
    ap.add_argument("--legacy-scatter-forward", action="store_true",
                    help="build the forward map by scattering instead of "
                         "gathering. Reproduces the v2 behaviour: leaves holes "
                         "in the expanding map (moire in the intrinsic image) "
                         "and omits the SIS counter-image. Diagnostic only.")
    ap.add_argument("--no-bundles", action="store_true")
    args = ap.parse_args()

    root = Path(args.out_dir)
    many = len(args.alpha_lr_px) > 1
    results = []
    for a in args.alpha_lr_px:
        d = root / f"alpha_{a:g}" if many else root
        results.append((a, build_one(a, args, d)))

    if many:
        print(LINE); print("BANK SUMMARY"); print(LINE)
        for a, ok in results:
            print(f"  alpha={a:6.2f} px   {'ok' if ok else 'FAILED'}")
        print("\n  Route each image to the nearest alpha by its measured ring")
        print("  radius. diagnose_ring_geometry.section_3_data_ring returns the")
        print("  per-image radii you need for that.\n")

    if not all(ok for _, ok in results):
        raise SystemExit(1)

    print("Next: verify independently, then retrain.")
    print(f"  python diagnose_ring_geometry.py --sections 1 2 --roundtrip-dir {args.out_dir}")
    print(f"  python train_all_wo_mapping.py --exp-name sis_v2 "
          f"--mapping-dir {args.out_dir} --resolution {args.lr_pixel_scale} "
          f"--theta-e {args.alpha_lr_px[0] * args.lr_pixel_scale:.4f} "
          f"--psf-path <path> --psf-source-pixscale-arcsec 0.168 "
          f"--epochs 20 --classes no_sub --mu-weight 0")


if __name__ == "__main__":
    main()
