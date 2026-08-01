"""Test 1: verify that training, evaluation, and convergence maps use one exact matrix set.

This script hashes sparse tensors and reports shapes, non-zero counts, and support.
Run it on every directory used by training/evaluation. Matching filenames are not enough.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch

FILES = [
    "sparse_grid_fracs_euclid_backward.pt",
    "scatter_to_log_128.pt",
    "forward_from_log_128.pt",
    "scatter_from_log_128.pt",
    "source_convergence_map.pt",
    "image_convergence_map.pt",
]

def tensor_hash(x: torch.Tensor) -> str:
    h = hashlib.sha256()
    if x.is_sparse:
        x = x.coalesce().cpu()
        h.update(x.indices().numpy().tobytes())
        h.update(x.values().numpy().tobytes())
        h.update(np.asarray(x.shape, dtype=np.int64).tobytes())
    else:
        x = x.detach().cpu().contiguous()
        h.update(x.numpy().tobytes())
        h.update(np.asarray(x.shape, dtype=np.int64).tobytes())
    return h.hexdigest()

def describe(path: Path) -> dict:
    x = torch.load(path, map_location="cpu")
    out = {"path": str(path), "shape": list(x.shape), "sha256": tensor_hash(x)}
    if x.is_sparse:
        x = x.coalesce(); idx, val = x.indices(), x.values()
        row = torch.zeros(x.shape[0]); col = torch.zeros(x.shape[1])
        row.scatter_add_(0, idx[0], val.float()); col.scatter_add_(0, idx[1], val.float())
        out.update({"nnz": int(x._nnz()), "zero_rows": int((row == 0).sum()), "zero_columns": int((col == 0).sum())})
    return out

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("directories", nargs="+", help="matrix directories to compare")
    p.add_argument("--output", default="matrix_identity_report.json")
    args = p.parse_args()
    report = {}
    for d in map(Path, args.directories):
        report[str(d)] = {}
        for name in FILES:
            path = d / name
            report[str(d)][name] = describe(path) if path.exists() else {"missing": True}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
