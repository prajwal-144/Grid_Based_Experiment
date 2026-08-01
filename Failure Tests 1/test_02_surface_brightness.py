"""Test 2: locate where the sparse chain violates surface-brightness conservation.

A constant source must remain constant under lensing wherever output pixels are covered.
The script tests raw and row-normalised operators stage by stage.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

EPS = 1e-8
NAMES = ["scatter_to_log_128.pt", "forward_from_log_128.pt", "scatter_from_log_128.pt"]

def apply(x, m, side):
    b, c, h, w = x.shape
    y = torch.sparse.mm(m, x.reshape(b*c, h*w).T)
    return y.T.reshape(b, c, side, side)

def row_normalise(m):
    m = m.coalesce(); idx, val = m.indices(), m.values().clamp_min(0)
    sums = torch.zeros(m.shape[0], dtype=val.dtype); sums.scatter_add_(0, idx[0], val)
    return torch.sparse_coo_tensor(idx, val / sums[idx[0]].clamp_min(EPS), m.shape).coalesce()

def stats(x):
    z = x[..., 4:-4, 4:-4]
    return {"mean": float(z.mean()), "std": float(z.std()), "min": float(z.min()), "max": float(z.max())}

def run(mats, normalise=False):
    x = torch.ones(1, 1, 128, 128)
    out = {"input": stats(x)}
    for name, m in zip(["to_log", "sis", "from_log"], mats):
        if normalise: m = row_normalise(m)
        x = apply(x, m, 128)
        out[name] = stats(x)
    out["passed"] = abs(out["from_log"]["mean"] - 1) < 0.01 and out["from_log"]["std"] < 0.01
    return out

def main():
    p = argparse.ArgumentParser(); p.add_argument("--mapping-dir", required=True); p.add_argument("--output", default="surface_brightness_report.json")
    args = p.parse_args(); root = Path(args.mapping_dir)
    mats = [torch.load(root / n, map_location="cpu").coalesce() for n in NAMES]
    report = {"raw": run(mats, False), "row_normalised_diagnostic": run(mats, True)}
    Path(args.output).write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
