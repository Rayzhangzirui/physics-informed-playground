"""
Generate ts_snapshot.json from the NumPy model (same format as TS buildSnapshotForVerification).
Use this to test verify_ts_gradients.py (NumPy vs PyTorch) without running TypeScript.

Usage:
  python generate_snapshot.py [output_path]
  Default output: ts_snapshot.json in this directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import BILOModel


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "ts_snapshot.json"
    n_hidden, depth, seed = 4, 2, 42
    t_colloc = [0.5]
    a_colloc = [1.0]
    w_res, w_grad = 1.0, 0.1

    model = BILOModel(n_hidden=n_hidden, depth=depth, rng=np.random.default_rng(seed))
    losses, grads = model.compute_losses_and_gradients(
        np.array(t_colloc), np.array(a_colloc), w_res=w_res, w_grad=w_grad
    )

    def to_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return float(x)

    snap = {
        "n_hidden": n_hidden,
        "depth": depth,
        "seed": seed,
        "W": [model._W[k].tolist() if np.ndim(model._W[k]) > 1 else model._W[k].tolist() for k in range(depth)],
        "b": [model._b[k].tolist() if np.ndim(model._b[k]) != 0 else float(model._b[k]) for k in range(depth)],
        "t_colloc": t_colloc,
        "a_colloc": a_colloc,
        "w_res": w_res,
        "w_grad": w_grad,
        "w_data": 0,
        "losses": {k: float(v) for k, v in losses.items()},
        "grads": {k: (v.tolist() if hasattr(v, "tolist") else float(v)) for k, v in grads.items() if k != "a"},
    }
    with open(out_path, "w") as f:
        json.dump(snap, f)
    print("Wrote", out_path)
    print("Run: python verify_ts_gradients.py", out_path)


if __name__ == "__main__":
    main()
