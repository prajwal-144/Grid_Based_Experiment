"""
diagnose_ring_geometry.py  --  repo-specific operator audit for Grid_Based_Experiment

WHY THIS EXISTS
---------------
The donut source / double-ring prediction seen in test_mag_mu_0.001.ipynb,
test_mag_mu_0.01.ipynb and test_mag_new_matrices_bundled.ipynb cannot be
diagnosed from the rendered figures, because the sparse .pt mapping files carry
no geometry metadata (train_all_wo_mapping.load_raw_mappings even prints
"WARNING: legacy mappings have no geometry metadata"). The mapping_io bundles
do not fix this either: regenerate_mappings.py takes theta_E and the pixel
scales as *command-line assertions* and stores them verbatim, so the metadata
records what someone typed, not what the matrix contains.

This script therefore MEASURES the operators instead of trusting labels.

Key idea: for out = M @ in, column p of M is the mass distribution of where
input pixel p is deposited. So the mass-weighted centroid of column p recovers
the geometric map  theta_p -> beta_p  exactly, with no test images and no
fitting. Doing that for every column recovers the whole deflection field,
including its SIGN and its amplitude in pixels.

WHAT IT CHECKS
--------------
  [0] Fingerprint every mapping directory; report which ones are byte-identical.
      (Needed because grid_matrices appears to have been overwritten in place
      between the mag_full and mag_full_latest_101 runs.)
  [1] Recover alpha_r (deflection amplitude, in pixels) and its SIGN for the
      backward matrix and for the composed forward chain, in each directory.
  [2] Round-trip test: does forward(backward(x)) preserve radius? The pipeline
      is only self-consistent if the two operators use OPPOSITE signs.
  [3] Measure the actual Einstein-ring radius of the val/no_sub data, which is
      the only ground truth for what alpha_r should be.
  [4] Profile the magnification regularizer's weight (1 - info)^2 that
      train_all_wo_mapping.magnification_regularizer applies, and report
      whether it is a donut prior (max at the source centre, zero on the ring).
  [5] Confirm whether the 0.168 -> 0.101 switch changed the PSF at all
      (scale_factor = source_pixscale / target_pixscale in psf.py).
  [6] End-to-end run of a checkpoint against its OWN mapping directory, with
      ring radii measured on every stage.

USAGE
-----
    python diagnose_ring_geometry.py                 # run everything it can find
    python diagnose_ring_geometry.py --sections 0 1 2
    python diagnose_ring_geometry.py --checkpoint outputs_corrected/mag_full/checkpoints/best.pt

Run from the repository root. torch + numpy required; matplotlib and astropy
optional (their sections are skipped if missing).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

EPS = 1e-8
LINE = "=" * 78
SUB = "-" * 78

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
CANDIDATE_DIRS = ["matrices_orig", "grid_matrices", "matrices_new", "mappings",
                  "raw_matrices_old/0.168", "matrices_v2"]


# ---------------------------------------------------------------------------
# loading helpers
# ---------------------------------------------------------------------------

def _load_sparse(path: Path):
    """Load a raw sparse .pt or a mapping_io bundle; return (matrix, metadata)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    meta = None
    if isinstance(obj, dict):
        meta = obj.get("metadata")
        obj = obj.get("matrix", obj.get("mapping"))
    if obj is None or not torch.is_tensor(obj) or not obj.is_sparse:
        raise TypeError(f"{path} does not contain a sparse COO tensor")
    return obj.coalesce().float(), meta


def load_dir(root: Path):
    """Load the four mappings from a directory, raw or bundled. Returns dict."""
    out = {}
    for role in RAW_NAMES:
        for names in (RAW_NAMES, BUNDLE_NAMES):
            p = root / names[role]
            if p.exists():
                out[role] = _load_sparse(p)
                break
    return out


def fingerprint(M: torch.Tensor) -> str:
    """Stable content hash of a coalesced sparse tensor."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(M.indices().numpy()).tobytes())
    h.update(np.ascontiguousarray(M.values().numpy().round(9)).tobytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# the core measurement: recover the geometric map from the operator columns
# ---------------------------------------------------------------------------

def column_centroids(mats, in_side: int, out_side: int, chunk: int = 256):
    """Mass-weighted output centroid of every input pixel, for out = Mn@...@M1@in.

    Returns (r_in, r_out, mass) as 1-D numpy arrays of length in_side**2, with
    radii measured in pixels from the grid centre ((N-1)/2 convention).
    NaN in r_out marks a column whose mass fell entirely off the output grid.
    """
    n_in = in_side * in_side
    n_out = out_side * out_side

    q = torch.arange(n_out, dtype=torch.float32)
    oy = (q // out_side) - (out_side - 1) / 2.0
    ox = (q % out_side) - (out_side - 1) / 2.0

    p = np.arange(n_in)
    r_in = np.hypot(p // in_side - (in_side - 1) / 2.0,
                    p % in_side - (in_side - 1) / 2.0)

    r_out = np.full(n_in, np.nan)
    mass_all = np.zeros(n_in)

    for start in range(0, n_in, chunk):
        stop = min(start + chunk, n_in)
        k = stop - start
        E = torch.zeros(n_in, k)
        E[torch.arange(start, stop), torch.arange(k)] = 1.0
        for M in mats:
            E = torch.sparse.mm(M, E)
        mass = E.sum(0)
        good = mass > 1e-9
        cy = torch.where(good, (E * oy[:, None]).sum(0) / mass.clamp_min(EPS), torch.nan)
        cx = torch.where(good, (E * ox[:, None]).sum(0) / mass.clamp_min(EPS), torch.nan)
        r_out[start:stop] = torch.hypot(cy, cx).numpy()
        mass_all[start:stop] = mass.numpy()

    return r_in, r_out, mass_all


def characterise_operator(mats, in_side, out_side, label, chunk=256):
    """Recover alpha_r (pixels) and the sign convention of a mapping."""
    r_in, r_out, mass = column_centroids(mats, in_side, out_side, chunk=chunk)

    ok = np.isfinite(r_out) & (mass > 1e-6)
    half = in_side / 2.0
    # Use an annulus that is well inside the grid so edge clipping does not bias us.
    outer = ok & (r_in > 0.35 * half) & (r_in < 0.85 * half)
    if outer.sum() < 20:
        print(f"  {label}: too few usable columns ({int(outer.sum())}) -- skipped")
        return None

    d = np.median(r_out[outer] - r_in[outer])
    alpha = abs(d)
    sign = "+alpha  (OUTWARD push, theta -> theta + alpha)" if d > 0 else \
           "-alpha  (INWARD push, theta -> theta - alpha)"

    # A true SIS fold satisfies r_out ~ |r_in - alpha| on the inside too.
    inner = ok & (r_in > 0.5) & (r_in < max(alpha - 0.5, 0.6))
    fold = np.nan
    if d < 0 and inner.sum() >= 8:
        fold = float(np.median(np.abs(r_out[inner] - (alpha - r_in[inner]))))

    resid = float(np.median(np.abs((r_out[outer] - r_in[outer]) - d)))
    coverage = float((mass > 1e-6).mean())

    print(f"  {label}")
    print(f"      alpha_r        : {alpha:8.3f} px      sign: {sign}")
    print(f"      radial scatter : {resid:8.3f} px      (small => clean radial map)")
    print(f"      column coverage: {coverage:8.1%}       (fraction of inputs with mass)")
    if np.isfinite(fold):
        verdict = "fold confirmed (genuine SIS two-image map)" if fold < 1.0 else \
                  "NO fold -- inner region does not behave like an SIS"
        print(f"      inner-fold err : {fold:8.3f} px      {verdict}")
    return {"alpha_px": alpha, "signed": float(d), "scatter": resid,
            "coverage": coverage, "r_in": r_in, "r_out": r_out, "mass": mass}


# ---------------------------------------------------------------------------
# radial profile utilities (for images)
# ---------------------------------------------------------------------------

def radial_profile(img, n_bins=None):
    a = np.squeeze(np.asarray(img, dtype=np.float64))
    if a.ndim != 2:
        raise ValueError(f"expected 2D, got {a.shape}")
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
    w[r < exclude_center_px] = -np.inf
    i = int(np.argmax(w))
    if 0 < i < len(prof) - 1 and np.all(np.isfinite(prof[i - 1:i + 2])):
        y0, y1, y2 = prof[i - 1:i + 2]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            return float(r[i] + np.clip(0.5 * (y0 - y2) / den, -1, 1) * (r[1] - r[0]))
    return float(r[i])


def ring_radii(img, n=2, min_sep=2.0, exclude_center_px=1.0):
    r, prof = radial_profile(img)
    w = np.where(np.isfinite(prof), prof, -np.inf)
    w[r < exclude_center_px] = -np.inf
    out = []
    for _ in range(n):
        i = int(np.argmax(w))
        if not np.isfinite(w[i]) or w[i] == -np.inf:
            break
        out.append(float(r[i]))
        w[np.abs(r - r[i]) < min_sep] = -np.inf
    return sorted(out)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def section_0_fingerprints(root: Path):
    print(LINE); print("[0] MAPPING DIRECTORY FINGERPRINTS"); print(LINE)
    print("Purpose: grid_matrices may have been overwritten in place between the")
    print("mag_full and mag_full_latest_101 runs. If so, the notebook cells that")
    print("load matrices_orig vs grid_matrices are NOT 'old vs new matrices' --")
    print("one of them is the operator the checkpoint was trained with and the")
    print("other is a foreign operator hot-swapped under frozen weights.\n")

    table = {}
    for d in CANDIDATE_DIRS:
        p = root / d
        if not p.is_dir():
            continue
        try:
            mats = load_dir(p)
        except Exception as exc:
            print(f"  {d:<24} load failed: {exc}")
            continue
        if not mats:
            continue
        row = {}
        for role, (M, meta) in mats.items():
            row[role] = fingerprint(M)
        table[d] = row
        mtime = max((q.stat().st_mtime for q in p.glob("*.pt")), default=0)
        import datetime
        stamp = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {d:<24} modified {stamp}")
        for role in RAW_NAMES:
            if role in row:
                M, meta = mats[role]
                extra = ""
                if meta:
                    extra = (f"  meta: theta_E={meta.get('theta_e_arcsec')} "
                             f"lr={meta.get('lr_pixel_scale_arcsec')} "
                             f"hr={meta.get('hr_pixel_scale_arcsec')}")
                print(f"      {role:<18} {row[role]}  shape={tuple(M.shape)} "
                      f"nnz={M._nnz()}{extra}")
        print()

    dirs = list(table)
    print("  Identity between directories (backward matrix):")
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            a, b = dirs[i], dirs[j]
            if "backward" in table[a] and "backward" in table[b]:
                same = table[a]["backward"] == table[b]["backward"]
                print(f"      {a:<22} vs {b:<22} {'IDENTICAL' if same else 'different'}")
    print()
    return table


def section_1_operators(root: Path, dirs=None, chunk=256):
    print(LINE); print("[1] MEASURED DEFLECTION AMPLITUDE AND SIGN"); print(LINE)
    print("Recovered directly from operator columns -- no labels trusted.")
    print("Reference: theta_E=0.75\" gives 0.75/0.168 = 4.46 px, 0.75/0.101 = 7.43 px.\n")
    results = {}
    for d in (dirs or CANDIDATE_DIRS):
        p = root / d
        if not p.is_dir():
            continue
        try:
            mats = load_dir(p)
        except Exception as exc:
            print(f"  {d}: load failed ({exc})\n"); continue
        if "backward" not in mats:
            continue
        print(f"  --- {d} ---")
        res = {}
        B = mats["backward"][0]
        side = int(round(math.sqrt(B.shape[0])))
        res["backward"] = characterise_operator([B], side, side, "backward (LR->source)", chunk)
        if all(k in mats for k in ("to_log", "forward_from_log", "from_log")):
            chain = [mats["to_log"][0], mats["forward_from_log"][0], mats["from_log"][0]]
            hr = int(round(math.sqrt(chain[0].shape[1])))
            res["forward"] = characterise_operator(chain, hr, hr,
                                                   "forward chain (source->image)", chunk)
        results[d] = res
        print()

    print(SUB)
    print("  INTERPRETATION")
    print("  The pipeline composes backward then forward. It is self-consistent")
    print("  ONLY if the two carry OPPOSITE signs (one +alpha, one -alpha), so the")
    print("  round trip is the identity. Same sign on both means the deflection is")
    print("  applied twice in the same direction.")
    print("  Note raw_matrices_old/create_backward_grid.ipynb calls backward_lensing")
    print("  (theta + alpha) while create_backward_grid.ipynb calls forward_lensing")
    print("  (theta - alpha). Both forward notebooks call forward_lensing. If the")
    print("  measurement above shows the same sign for backward and forward in a")
    print("  directory, that directory's backward matrix has the flipped convention.")
    print(SUB + "\n")
    return results


def section_2_roundtrip(root: Path, d: str, chunk=256):
    print(LINE); print(f"[2] ROUND-TRIP TEST -- {d}"); print(LINE)
    p = root / d
    mats = load_dir(p)
    need = ("backward", "to_log", "forward_from_log", "from_log")
    if not all(k in mats for k in need):
        print("  incomplete mapping set; skipped\n"); return None

    B = mats["backward"][0]
    lr_side = int(round(math.sqrt(B.shape[0])))
    hr_side = int(round(math.sqrt(mats["to_log"][0].shape[1])))

    # Row-normalise the backward map exactly as the notebooks and trainer do.
    b = B.coalesce(); idx, val = b.indices(), b.values().clamp_min(0)
    s = torch.zeros(b.shape[0]); s.scatter_add_(0, idx[0], val)
    Bavg = torch.sparse_coo_tensor(idx, val / s[idx[0]].clamp_min(EPS), b.shape).coalesce()

    # Thin test rings at a range of radii, pushed all the way through.
    yy, xx = np.indices((lr_side, lr_side))
    r = np.hypot(yy - (lr_side - 1) / 2.0, xx - (lr_side - 1) / 2.0)
    print(f"  {'ring in':>9} {'-> source':>10} {'-> image':>10}   comment")
    for r0 in [4.0, 6.0, 8.0, 10.0, 13.0, 16.0]:
        img = np.exp(-0.5 * ((r - r0) / 0.8) ** 2).astype(np.float32)
        t = torch.from_numpy(img).reshape(1, 1, lr_side, lr_side)
        flat = t.reshape(1, -1).T
        src = torch.sparse.mm(Bavg, flat).T.reshape(lr_side, lr_side).numpy()
        r_src = peak_radius(src)
        # upsample source to the HR grid the forward chain expects
        up = torch.nn.functional.interpolate(
            torch.from_numpy(src)[None, None], size=(hr_side, hr_side), mode="bilinear",
            align_corners=False)
        f = up.reshape(1, -1).T
        for M in (mats["to_log"][0], mats["forward_from_log"][0], mats["from_log"][0]):
            f = torch.sparse.mm(M, f)
        img_out = f.T.reshape(hr_side, hr_side).numpy()
        radii = ring_radii(img_out, n=2)
        # HR pixels -> LR pixels for comparison
        radii_lr = [x * lr_side / hr_side for x in radii]
        note = ""
        if radii_lr:
            err = abs(radii_lr[-1] - r0)
            note = "round trip OK" if err < 1.5 else f"radius moved by {radii_lr[-1]-r0:+.2f} px"
        if len(radii_lr) == 2 and abs(radii_lr[1] - radii_lr[0]) > 2.0:
            note += "  <-- DOUBLE RING produced by the operator alone"
        shown = ", ".join(f"{x:.2f}" for x in radii_lr) if radii_lr else "none"
        print(f"  {r0:9.2f} {r_src:10.2f} {shown:>10}   {note}")
    print("\n  A correct operator pair returns the ring to its input radius and")
    print("  produces exactly ONE ring. Two rings here means the geometry alone")
    print("  creates the artefact -- the network and the mu loss are not to blame.\n")


def section_3_data_ring(root: Path, cls="no_sub", n=60):
    print(LINE); print("[3] EINSTEIN-RING RADIUS OF THE ACTUAL DATA"); print(LINE)
    d = root / "val" / cls
    if not d.is_dir():
        print(f"  {d} not found; skipped\n"); return None
    files = sorted(d.glob("sim_*.npy"))[:n]
    if not files:
        print("  no sim_*.npy found; skipped\n"); return None

    radii, stack = [], None
    for f in files:
        a = np.load(f, allow_pickle=True)
        if isinstance(a, np.ndarray) and a.dtype == object:
            a = a.item() if a.size == 1 else a[0]
        a = np.squeeze(np.asarray(a, dtype=np.float64))
        if a.ndim != 2:
            continue
        a = (a - a.min()) / max(a.max() - a.min(), EPS)
        stack = a if stack is None else stack + a
        radii.append(peak_radius(a))

    stack /= len(radii)
    r_stack = peak_radius(stack)
    med = float(np.median(radii))
    print(f"  images used                 : {len(radii)}  ({stack.shape[0]}x{stack.shape[1]})")
    print(f"  per-image ring radius       : median {med:.2f} px, "
          f"IQR {np.percentile(radii,25):.2f}-{np.percentile(radii,75):.2f} px")
    print(f"  stacked-image ring radius   : {r_stack:.2f} px")
    print()
    print("  Required alpha_r for the operator to collapse this ring to a point:")
    print(f"      alpha_r  = {r_stack:.2f} px")
    for scale in (0.168, 0.101):
        print(f"      theta_E  = {r_stack*scale:.4f} arcsec   if you declare "
              f"{scale} arcsec/px")
    print()
    print("  Compare against what the matrices actually encode (section 1). Only")
    print("  the RATIO theta_E / pixel_scale reaches the operator -- the matrices")
    print("  are built in pixel units, so declaring a different pixel scale in")
    print("  --resolution does not move the ring.")
    print("  Caveat: a spread in per-image radii means the dataset does NOT have a")
    print("  single Einstein radius, in which case one fixed-SIS operator cannot")
    print("  be right for every image and the residual annulus is irreducible.\n")
    return {"median_px": med, "stack_px": r_stack, "per_image": radii}


def section_4_mu_prior(root: Path, image_shape=64, resolutions=(0.168, 0.101),
                       theta_e=0.75, mapping_dir="matrices_v2"):
    print(LINE); print("[4] THE MAGNIFICATION REGULARIZER IS A DONUT PRIOR"); print(LINE)
    print("train_all_wo_mapping.magnification_regularizer computes")
    print("    (weight * laplacian(source)^2).sum() / weight.sum(),  weight = (1-info)^2")
    print("info comes from |mu| = |1/(1 - theta_E/r)| compressed to [0,1].")
    print("For an SIS, |mu| -> 0 at r=0, peaks at r=theta_E, -> 1 far out. So info")
    print("is an ANNULUS, and (1-info)^2 is LARGE at the source centre and ZERO on")
    print("the ring. The penalty therefore forbids curvature exactly where a")
    print("compact source lives, and permits it exactly on a ring.\n")

    try:
        from physics_losses import build_fixed_sis_information_maps
    except Exception as exc:
        print(f"  could not import physics_losses ({exc}); skipped\n"); return

    p = root / mapping_dir
    try:
        B = load_dir(p)["backward"][0]
    except Exception as exc:
        print(f"  could not load backward from {mapping_dir} ({exc}); skipped\n"); return

    for res in resolutions:
        info = build_fixed_sis_information_maps(B, image_shape, res, theta_e)
        s = info.source_information_lr.clamp(0, 1)[0, 0].numpy()
        w = (1.0 - s) ** 2
        rb, wp = radial_profile(w)
        rb2, sp = radial_profile(s)
        crit = theta_e / res
        print(f"  --- resolution {res} arcsec/px   critical radius {crit:.2f} px ---")
        print(f"      info(source) range      : {s.min():.3f} .. {s.max():.3f}")
        print(f"      weight at r=0           : {wp[0]:.3f}")
        i_ring = int(np.nanargmax(sp))
        print(f"      info peaks at r         : {rb2[i_ring]:.2f} px")
        print(f"      weight there            : {wp[i_ring]:.3f}")
        print(f"      centre/ring weight ratio: {wp[0]/max(wp[i_ring],1e-6):.1f}x")
        print("      => curvature at the centre is penalised "
              f"{wp[0]/max(wp[i_ring],1e-6):.0f}x harder than on the ring\n")

    print("  This term is scaled by --mu-weight. Going 0.001 -> 0.01 multiplies")
    print("  the donut prior by ten. Its share of the total loss stays small, but")
    print("  the image-fidelity term barely constrains the source interior (that")
    print("  is the null space of a fixed-SIS inversion), so a small term can still")
    print("  dictate what the source looks like there.")
    print("  history.json corroborates this: in mag_full_latest_101 the mu term")
    print("  falls 0.775 -> 0.11 (7x) once the ramp engages, while the source loss")
    print("  RISES 0.033 -> 0.037.\n")


def section_5_psf(root: Path):
    print(LINE); print("[5] DID THE 0.168 -> 0.101 SWITCH CHANGE THE PSF?"); print(LINE)
    print("psf.resample_kernel_to_pixscale uses")
    print("    scale_factor = source_pixscale_arcsec / target_pixscale_arcsec")
    for src, tgt in ((0.168, 0.084), (0.101, 0.0505)):
        print(f"    {src} / {tgt} = {src/tgt:.6f}")
    print()
    print("  Both runs pass --psf-source-pixscale-arcsec equal to --resolution, and")
    print("  target_resolution = resolution / 2. The ratio is 2.0 in both cases, so")
    print("  the resampled kernel is IDENTICAL. Switching to Euclid pixel scales")
    print("  did not change the PSF by a single pixel.")
    print()
    print("  Separately: the FITS file is an HSC PDR2 PSF, natively 0.168 arcsec/px.")
    print("  Declaring --psf-source-pixscale-arcsec 0.101 asserts a pixel scale the")
    print("  stamp does not have. That is wrong on its own terms, but because only")
    print("  the ratio is used it happens to cancel here. Fix it anyway, after the")
    print("  geometry: a PSF cannot move flux from the centre out to an annulus, so")
    print("  it is not what makes the donut.\n")


def section_6_pipeline(root: Path, ckpt_path: Path, val_index=5, val_class="no_sub"):
    print(LINE); print(f"[6] END-TO-END -- {ckpt_path}"); print(LINE)
    try:
        import torch.nn.functional as F
        import data as data_mod
        from differentiable_lensing import DifferentiableLensing
        from psf import apply_psf, build_psf_kernel
        from sisr import SISR
    except Exception as exc:
        print(f"  imports failed ({exc}); skipped\n"); return

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck["args"]
    mdir = args.get("mapping_dir", ".")
    print(f"  checkpoint mapping_dir = {mdir!r}   resolution = {args['resolution']}"
          f"   mu_weight = {args['mu_weight']}")
    print("  Using the checkpoint's OWN mapping_dir. Loading a different directory")
    print("  here swaps the operator under frozen weights and is not a fair test.\n")

    mats = load_dir(root / mdir)
    if len(mats) < 4:
        print(f"  {mdir} incomplete; skipped\n"); return

    model = SISR(args["magnification"], args["n_mag"], args["residual_depth"],
                 in_channels=2, latent_channel_count=args["latent_space_size"])
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    lens = DifferentiableLensing(device="cpu", alpha=None,
                                 target_resolution=args["target_resolution"],
                                 target_shape=args["target_shape"])

    b = mats["backward"][0]; idx, val = b.indices(), b.values().clamp_min(0)
    s = torch.zeros(b.shape[0]); s.scatter_add_(0, idx[0], val)
    bavg = torch.sparse_coo_tensor(idx, val / s[idx[0]].clamp_min(EPS), b.shape).coalesce()

    psf = None
    try:
        psf = build_psf_kernel("fits", 0.16, args["target_resolution"],
                               path=args["psf_path"],
                               source_pixscale_arcsec=args["psf_source_pixscale_arcsec"],
                               device="cpu")
    except Exception as exc:
        print(f"  PSF unavailable ({exc}); continuing without convolution\n")

    lr = data_mod.LensingDataset("val/", [val_class], val_index + 1)[val_index]
    lr = lr.unsqueeze(0).float()
    if lr.ndim == 3:
        lr = lr.unsqueeze(1)
    if lr.ndim == 5:
        lr = lr.squeeze(1)

    with torch.no_grad():
        flat = lr.reshape(lr.shape[0] * lr.shape[1], -1).T
        src_lr = torch.sparse.mm(bavg, flat).T.reshape(1, 1, args["image_shape"],
                                                       args["image_shape"])
        src_hr = model(torch.cat([src_lr, lr], dim=1))
        intrinsic = lens.cross_grid_fill(src_hr, [mats["to_log"][0],
                                                  mats["forward_from_log"][0],
                                                  mats["from_log"][0]])
        conv = apply_psf(intrinsic, psf) if psf is not None else intrinsic
        pred = torch.nn.functional.interpolate(conv, size=lr.shape[-2:], mode="area")

    zero = float(lr.square().mean())
    mse = float(torch.nn.functional.mse_loss(pred, lr))
    print(f"  zero MSE {zero:.6f}   model MSE {mse:.6f}   skill {1-mse/max(zero,EPS):.4f}\n")

    lr_side = lr.shape[-1]
    hr_side = src_hr.shape[-1]
    stages = [("LR observation", lr[0, 0].numpy(), 1.0),
              ("LR source (backward)", src_lr[0, 0].numpy(), 1.0),
              ("HR source (network)", src_hr[0, 0].numpy(), lr_side / hr_side),
              ("intrinsic (forward)", intrinsic[0, 0].numpy(), lr_side / hr_side),
              ("prediction", pred[0, 0].numpy(), 1.0)]
    print(f"  {'stage':<24} {'rings (LR px)':<24} donut?")
    for name, arr, sc in stages:
        rr = [x * sc for x in ring_radii(arr, n=2)]
        rp = peak_radius(arr) * sc
        r, prof = radial_profile(arr)
        centre = prof[0] if np.isfinite(prof[0]) else 0.0
        pk = np.nanmax(prof)
        donut = "YES" if (rp > 1.0 and centre > 0 and pk / centre > 1.15) else "no"
        print(f"  {name:<24} {', '.join(f'{x:.2f}' for x in rr):<24} {donut}")

    print()
    print("  Read this as a chain. If 'LR source (backward)' is already a donut,")
    print("  the backward operator is the cause and nothing downstream can undo it.")
    print("  If it is compact and 'prediction' has two rings, the forward chain is.")
    print("  If both are clean but the HR source is hollow, the mu prior is.\n")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--sections", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--roundtrip-dir", default="grid_matrices")
    ap.add_argument("--mu-dir", default="grid_matrices",
                    help="mapping dir used for the section-4 info map. Point "
                         "this at the SAME directory your checkpoint trained "
                         "with, or section 4 reports a stale operator.")
    ap.add_argument("--mu-theta-e", type=float, default=0.75)
    ap.add_argument("--mu-resolutions", nargs="+", type=float, default=[0.168, 0.101])
    ap.add_argument("--checkpoint", default="outputs_corrected/mag_full/checkpoints/best.pt")
    ap.add_argument("--chunk", type=int, default=256,
                    help="columns probed per batch in section 1 (lower = less RAM)")
    ap.add_argument("--val-index", type=int, default=5)
    a = ap.parse_args()
    root = Path(a.root).resolve()

    print(f"\nrepository root: {root}\n")
    if 0 in a.sections:
        section_0_fingerprints(root)
    if 1 in a.sections:
        section_1_operators(root, chunk=a.chunk)
    if 2 in a.sections:
        section_2_roundtrip(root, a.roundtrip_dir, chunk=a.chunk)
    if 3 in a.sections:
        section_3_data_ring(root)
    if 4 in a.sections:
        section_4_mu_prior(root, resolutions=tuple(a.mu_resolutions),
                           theta_e=a.mu_theta_e, mapping_dir=a.mu_dir)
    if 5 in a.sections:
        section_5_psf(root)
    if 6 in a.sections:
        ck = root / a.checkpoint
        if ck.exists():
            section_6_pipeline(root, ck, val_index=a.val_index)
        else:
            print(f"[6] checkpoint {ck} not found; skipped\n")


if __name__ == "__main__":
    main()