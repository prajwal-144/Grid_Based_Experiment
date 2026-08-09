"""
train_sis_bank.py -- per-image Einstein radius via an operator bank.

WHY
---
ring_analysis.py index measured theta_E per image by decomposing r(phi):
the m=0 term is theta_E, the m=1 term is the source offset beta.

    train (n=5000):  theta_E median 9.22 px, IQR 6.59-11.49, range 1.50-21.64
                     beta    median 3.13 px, IQR 2.24- 3.87

theta_E varies by more than a factor of ten. train_all_wo_mapping.py applies
ONE operator (alpha = 8.139 px) to all of them, so most images are lensed
through the wrong lens. That is what produced the leftover double rings: an
image whose true theta_E is 15 must be explained with alpha = 8.14, which
forces a large fake source offset and therefore a bright counter-image the data
does not contain.

Validation of the estimator: on near-complete rings, where beta must be ~0 by
definition, the decomposition returns |theta_E - r_peak| = 0.22 px (median) and
beta = 1.10 px. It recovers the right answer where the right answer is known.

WHAT THIS SCRIPT DOES DIFFERENTLY
---------------------------------
1. BANK. Loads K operator sets built by build_sis_mappings.py, one per alpha.
   13 uniform 1-px bins over [3.5, 15.5] give median |theta_E - alpha| = 0.26 px
   and p90 = 0.47 px, versus 0.49 / 1.07 (max 7.97) for 6 equal-population bins.
   Uniform bins beat equal-population here because equal-population puts one
   operator across the whole 12.35-21.64 tail.

2. PER-BATCH GROUPING, not a bin-homogeneous sampler. Batches stay randomly
   shuffled; inside a batch the images are grouped by bin and each group gets
   its own sparse matmul. A homogeneous sampler would correlate batch
   composition with theta_E, which biases batch statistics and gradient noise.
   Cost is a handful of extra sparse matmuls per step.

3. ALPHA CONDITIONING CHANNEL. The network is given alpha as a constant extra
   input plane (in_channels 2 -> 3). Without it the model cannot tell which
   operator its output will be pushed through, and would have to average over
   bins -- which is the failure we are trying to remove.

USAGE
-----
    python build_sis_mappings.py --alpha-lr-px 3.5 4.5 5.5 6.5 7.5 8.5 9.5 \
        10.5 11.5 12.5 13.5 14.5 15.5 --out-dir bank

    python ring_analysis.py index --split train/ --classes no_sub --n 5000 \
        --out ring_index_train.json
    python ring_analysis.py index --split val/ --classes no_sub --n 2000 \
        --out ring_index_val.json

    python train_sis_bank.py --exp-name sis_bank --bank-dir bank \
        --index-train ring_index_train.json --index-val ring_index_val.json \
        --resolution 0.168 --psf-path <path> --psf-source-pixscale-arcsec 0.168 \
        --epochs 40 --classes no_sub --tv-weight 3 --mu-weight 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import data
from differentiable_lensing import DifferentiableLensing
from psf import apply_psf, build_psf_kernel
from sisr import SISR

EPS = 1e-8

RAW_NAMES = {
    "backward": "sparse_grid_fracs_euclid_backward.pt",
    "to_log": "scatter_to_log_128.pt",
    "forward_from_log": "forward_from_log_128.pt",
    "from_log": "scatter_from_log_128.pt",
}


# ---------------------------------------------------------------------------
# bank loading
# ---------------------------------------------------------------------------

def normalize_sparse_rows(M):
    M = M.coalesce()
    idx, val = M.indices(), M.values().clamp_min(0)
    s = torch.zeros(M.shape[0], device=val.device, dtype=val.dtype)
    s.scatter_add_(0, idx[0], val)
    return torch.sparse_coo_tensor(idx, val / s[idx[0]].clamp_min(EPS),
                                   M.shape, device=M.device).coalesce()


def load_bank(bank_dir: Path, device, image_shape, target_shape):
    """Load every alpha_* subdirectory. Returns (alphas, backward_avg, chains)."""
    subs = sorted(bank_dir.glob("alpha_*"),
                  key=lambda p: float(re.sub(r"^alpha_", "", p.name)))
    if not subs:
        raise FileNotFoundError(
            f"no alpha_* subdirectories in {bank_dir}. Build the bank first:\n"
            "  python build_sis_mappings.py --alpha-lr-px 3.5 4.5 ... --out-dir bank")

    alphas, bavg, chains = [], [], []
    for d in subs:
        a = float(re.sub(r"^alpha_", "", d.name))
        mats = {}
        for role, name in RAW_NAMES.items():
            p = d / name
            if not p.exists():
                raise FileNotFoundError(f"{p} missing")
            mats[role] = torch.load(p, map_location="cpu",
                                    weights_only=False).coalesce().float()
        if mats["backward"].shape != (image_shape ** 2, image_shape ** 2):
            raise ValueError(f"{d}: backward shape {tuple(mats['backward'].shape)}")
        if mats["to_log"].shape[1] != target_shape ** 2:
            raise ValueError(f"{d}: forward chain shape mismatch")
        alphas.append(a)
        bavg.append(normalize_sparse_rows(mats["backward"]).to(device))
        chains.append([mats["to_log"].to(device),
                       mats["forward_from_log"].to(device),
                       mats["from_log"].to(device)])
        print(f"    alpha = {a:6.2f} px   loaded from {d.name}")
    return torch.tensor(alphas, dtype=torch.float32), bavg, chains


def load_index(path: Path, n_expected, alphas):
    """Map dataset index -> nearest-alpha bin. Returns (bin_of_index, stats)."""
    D = json.loads(Path(path).read_text())
    if D.get("routing_key") != "theta_e_px":
        print(f"  WARNING: {path} has routing_key={D.get('routing_key')!r}. "
              "Re-run `ring_analysis.py index` -- routing on the brightest "
              "annulus conflates theta_E with the source offset beta.")
    a = alphas.numpy()
    med_bin = int(np.abs(a - np.median([r["theta_e_px"] for r in D["per_image"]
                                        if np.isfinite(r["theta_e_px"])])).argmin())
    bins = np.full(n_expected, med_bin, dtype=np.int64)
    seen = 0
    clamped = 0
    for r in D["per_image"]:
        i = int(r["index"])
        t = float(r["theta_e_px"])
        if i >= n_expected or not np.isfinite(t):
            continue
        bins[i] = int(np.abs(a - t).argmin())
        if t < a.min() - 0.5 or t > a.max() + 0.5:
            clamped += 1
        seen += 1
    stats = {"indexed": seen, "total": n_expected,
             "defaulted": int(n_expected - seen), "clamped": clamped}
    return torch.from_numpy(bins), stats


# ---------------------------------------------------------------------------
# dataset that also yields the bin
# ---------------------------------------------------------------------------

class BinnedLensing(torch.utils.data.Dataset):
    def __init__(self, split, classes, count, bins):
        self.ds = data.LensingDataset(split, classes, count)
        self.bins = bins

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i], int(self.bins[i])


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def ensure_bchw(x):
    while x.ndim > 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    return x.unsqueeze(1) if x.ndim == 3 else x


def arc_balanced_weights(target, threshold_fraction, arc_boost, background_weight):
    peak = target.amax(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    arc = (target >= threshold_fraction * peak).to(target.dtype)
    return background_weight + arc_boost * arc


def normalized_wmse(pred, target, weight):
    weight = weight.expand_as(pred)
    return (weight * (pred - target).square()).sum() / weight.sum().clamp_min(EPS)


def total_variation(x):
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs()
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs()
    return dx.mean() + dy.mean()


# ---------------------------------------------------------------------------
# grouped operator application
# ---------------------------------------------------------------------------

def grouped_backward(images, bins, bavg, out_side):
    """Apply each image's own backward operator. Differentiable via index_copy."""
    b, c, h, w = images.shape
    out = images.new_zeros(b, c, out_side, out_side)
    for k in bins.unique().tolist():
        idx = (bins == k).nonzero(as_tuple=True)[0]
        sub = images.index_select(0, idx)
        flat = sub.reshape(idx.numel() * c, h * w).T
        res = torch.sparse.mm(bavg[k], flat).T.contiguous()
        out = out.index_copy(0, idx, res.reshape(idx.numel(), c, out_side, out_side))
    return out


def grouped_forward(source, bins, chains, lens, out_side):
    b, c, h, w = source.shape
    out = source.new_zeros(b, c, out_side, out_side)
    for k in bins.unique().tolist():
        idx = (bins == k).nonzero(as_tuple=True)[0]
        sub = source.index_select(0, idx)
        out = out.index_copy(0, idx, lens.cross_grid_fill(sub, chains[k]))
    return out


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", default="sis_bank")
    p.add_argument("--output-dir", default="outputs_corrected")
    p.add_argument("--bank-dir", required=True)
    p.add_argument("--index-train", required=True)
    p.add_argument("--index-val", required=True)
    p.add_argument("--resolution", type=float, required=True)
    p.add_argument("--psf-path", required=True)
    p.add_argument("--psf-source-pixscale-arcsec", type=float, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-shape", type=int, default=64)
    p.add_argument("--magnification", type=int, default=2)
    p.add_argument("--n-mag", type=int, default=1)
    p.add_argument("--residual-depth", type=int, default=3)
    p.add_argument("--latent-space-size", type=int, default=64)
    p.add_argument("--classes", nargs="+", default=["no_sub"])
    p.add_argument("--train-samples-per-class", type=int, default=5000)
    p.add_argument("--val-samples-per-class", type=int, default=2000)
    p.add_argument("--arc-threshold-fraction", type=float, default=0.08)
    p.add_argument("--arc-boost", type=float, default=5.0)
    p.add_argument("--background-weight", type=float, default=0.2)
    p.add_argument("--source-loss-weight", type=float, default=0.2)
    p.add_argument("--tv-weight", type=float, default=3.0)
    p.add_argument("--mu-weight", type=float, default=0.0)
    p.add_argument("--no-alpha-channel", action="store_true",
                   help="ablation: withhold the alpha conditioning plane")
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    a.effective_magnification = a.magnification ** a.n_mag
    a.target_shape = a.image_shape * a.effective_magnification
    a.target_resolution = a.resolution / a.effective_magnification
    a.device = "cuda" if a.cuda and torch.cuda.is_available() else "cpu"
    a.in_channels = 2 if a.no_alpha_channel else 3
    return a


def run_epoch(model, loader, opt, training, args, lens, alphas, bavg, chains, psf):
    model.train(training)
    totals = {k: 0.0 for k in ["total", "image", "source", "tv", "raw_mse",
                               "zero_mse", "skill"]}
    n = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    half = args.image_shape / 2.0
    with ctx:
        for lr_image, bins in tqdm(loader, desc="train" if training else "val"):
            lr_image = ensure_bchw(lr_image.float()).to(args.device)
            bins = bins.to(args.device)
            b = lr_image.shape[0]

            source_lr = grouped_backward(lr_image, bins, bavg, args.image_shape)
            inputs = [source_lr, lr_image]
            if not args.no_alpha_channel:
                # normalised so the plane sits in roughly [0, 0.5]
                av = alphas.to(args.device)[bins].view(b, 1, 1, 1) / half
                inputs.append(av.expand(b, 1, args.image_shape, args.image_shape))
            source_hr = model(torch.cat(inputs, dim=1))

            intrinsic = grouped_forward(source_hr, bins, chains, lens, args.target_shape)
            pred_lr = F.interpolate(apply_psf(intrinsic, psf),
                                    size=lr_image.shape[-2:], mode="area")
            source_down = F.interpolate(source_hr, size=source_lr.shape[-2:], mode="area")

            w = arc_balanced_weights(lr_image, args.arc_threshold_fraction,
                                     args.arc_boost, args.background_weight)
            image_loss = normalized_wmse(pred_lr, lr_image, w)
            sw = (source_lr > 0).to(source_lr.dtype) + 0.05
            source_loss = normalized_wmse(source_down, source_lr, sw)
            tv_loss = total_variation(source_hr)
            total = (image_loss + args.source_loss_weight * source_loss
                     + args.tv_weight * tv_loss)

            if training:
                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            raw = F.mse_loss(pred_lr, lr_image)
            zero = lr_image.square().mean()
            for k, v in {"total": total, "image": image_loss, "source": source_loss,
                         "tv": tv_loss, "raw_mse": raw, "zero_mse": zero,
                         "skill": 1.0 - raw / zero.clamp_min(EPS)}.items():
                totals[k] += float(v.detach())
            n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_dir = Path(args.output_dir) / args.exp_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"OPERATOR BANK  {args.bank_dir}")
    print("=" * 74)
    alphas, bavg, chains = load_bank(Path(args.bank_dir), device,
                                     args.image_shape, args.target_shape)
    args.alphas = [float(x) for x in alphas]
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    n_tr = args.train_samples_per_class * len(args.classes)
    n_va = args.val_samples_per_class * len(args.classes)
    tr_bins, tr_stats = load_index(Path(args.index_train), n_tr, alphas)
    va_bins, va_stats = load_index(Path(args.index_val), n_va, alphas)
    print(f"\n  train routing: {tr_stats['indexed']}/{tr_stats['total']} indexed, "
          f"{tr_stats['defaulted']} defaulted to the median bin, "
          f"{tr_stats['clamped']} clamped outside the bank")
    print(f"  val   routing: {va_stats['indexed']}/{va_stats['total']} indexed, "
          f"{va_stats['defaulted']} defaulted, {va_stats['clamped']} clamped")
    counts = torch.bincount(tr_bins, minlength=len(alphas)).tolist()
    print("  train images per operator: " + " ".join(f"{c}" for c in counts))
    if min(counts) < 50:
        print("  WARNING: some bins hold very few images; consider fewer, wider bins.")

    lens = DifferentiableLensing(device=device, alpha=None,
                                 target_resolution=args.target_resolution,
                                 target_shape=args.target_shape).to(device)
    psf = build_psf_kernel("fits", 0.16, args.target_resolution, path=args.psf_path,
                           source_pixscale_arcsec=args.psf_source_pixscale_arcsec,
                           device=device)

    model = SISR(args.magnification, args.n_mag, args.residual_depth,
                 in_channels=args.in_channels,
                 latent_channel_count=args.latent_space_size).to(device)
    print(f"\n  model in_channels = {args.in_channels}"
          f"{'  (alpha plane WITHHELD -- ablation)' if args.no_alpha_channel else ''}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    tl = torch.utils.data.DataLoader(
        BinnedLensing("train/", args.classes, args.train_samples_per_class, tr_bins),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=args.device == "cuda")
    vl = torch.utils.data.DataLoader(
        BinnedLensing("val/", args.classes, args.val_samples_per_class, va_bins),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=args.device == "cuda")

    history = {"train": [], "val": []}
    best = float("inf")
    for epoch in range(args.epochs):
        tr = run_epoch(model, tl, opt, True, args, lens, alphas, bavg, chains, psf)
        va = run_epoch(model, vl, opt, False, args, lens, alphas, bavg, chains, psf)
        history["train"].append({"epoch": epoch + 1, **tr})
        history["val"].append({"epoch": epoch + 1, **va})
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print("train", tr); print("val", va)
        payload = {"epoch": epoch, "model_state_dict": model.state_dict(),
                   "optimizer_state_dict": opt.state_dict(), "args": vars(args),
                   "history": history}
        torch.save(payload, run_dir / "checkpoints" / "latest.pt")
        if va["total"] < best:
            best = va["total"]
            torch.save(payload, run_dir / "checkpoints" / "best.pt")


if __name__ == "__main__":
    main()
