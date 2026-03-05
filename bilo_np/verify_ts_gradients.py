"""
Verify TypeScript BILO gradients against PyTorch.

Usage:
  1. From playground: build a snapshot and write JSON:
       node -e "
       const path = require('path');
       const fs = require('fs');
       const { buildSnapshotForVerification } = require('./dist/bilo_test.js');
       const snap = buildSnapshotForVerification({
         n_hidden: 4, depth: 2, seed: 42,
         t_colloc: [0.5], a_colloc: [1.0],
         w_res: 1, w_grad: 0.1
       });
       fs.writeFileSync(path.join(__dirname, 'bilo_np/ts_snapshot.json'), JSON.stringify(snap));
       "
     Or run the write_snapshot script (see README in bilo_np).
  2. Run this script (with conda activate math10 for PyTorch):
       python verify_ts_gradients.py [path/to/ts_snapshot.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Add parent so we can import from model
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import BILOModel, BILOModelTorch, PINNModel, PINNModelTorch


def load_snapshot(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def snapshot_to_numpy_weights(snap: dict) -> tuple[list[np.ndarray], list[np.ndarray | float]]:
    """Convert snapshot W, b to NumPy model layout (_W, _b)."""
    W = snap["W"]
    b = snap["b"]
    _W = []
    _b = []
    for k, w in enumerate(W):
        if isinstance(w[0], list):
            _W.append(np.array(w, dtype=np.float64))
        else:
            _W.append(np.array(w, dtype=np.float64))
    for k, bb in enumerate(b):
        if isinstance(bb, list):
            _b.append(np.array(bb, dtype=np.float64))
        else:
            _b.append(float(bb))
    return _W, _b


def _run_verification_bilo(snap: dict, _W: list, _b: list, np_model: BILOModel, messages: list) -> tuple[bool, bool]:
    n_hidden = snap["n_hidden"]
    depth = snap["depth"]
    t_colloc = np.array(snap["t_colloc"], dtype=np.float64)
    a_colloc = np.array(snap["a_colloc"], dtype=np.float64)
    w_res = snap["w_res"]
    w_grad = snap["w_grad"]
    w_data = snap.get("w_data", 0)
    t_data = np.array(snap["t_data"], dtype=np.float64) if snap.get("t_data") else None
    a_data = np.array(snap["a_data"], dtype=np.float64) if snap.get("a_data") else None
    u_data = np.array(snap["u_data"], dtype=np.float64) if snap.get("u_data") else None

    losses_np, grads_np = np_model.compute_losses_and_gradients(
        t_colloc, a_colloc,
        t_data=t_data, a_data=a_data, u_data=u_data,
        w_res=w_res, w_grad=w_grad, w_data=w_data,
    )
    losses_ts = snap["losses"]
    messages.append(f"Losses TS: L_res={losses_ts['L_res']:.8f} L_grad={losses_ts['L_grad']:.8f} L_data={losses_ts['L_data']:.8f}")
    messages.append(f"Losses NP: L_res={losses_np['L_res']:.8f} L_grad={losses_np['L_grad']:.8f} L_data={losses_np['L_data']:.8f}")

    rtol, atol = 1e-5, 1e-8
    loss_ok = (
        np.isclose(losses_np["L_res"], losses_ts["L_res"], rtol=rtol, atol=atol)
        and np.isclose(losses_np["L_grad"], losses_ts["L_grad"], rtol=rtol, atol=atol)
        and np.isclose(losses_np["L_data"], losses_ts["L_data"], rtol=rtol, atol=atol)
    )
    if not loss_ok:
        messages.append("FAIL: losses do not match TS vs NumPy")
    else:
        messages.append("PASS: losses match (TS vs NumPy)")

    grads_ts = snap["grads"]
    grad_ok = True
    for k in range(depth):
        gts = grads_ts.get(f"W{k+1}")
        gnp = grads_np[f"W{k+1}"]
        if gts is None:
            grad_ok = False
            messages.append(f"FAIL: TS grads missing W{k+1}")
            continue
        gts_arr = np.array(gts, dtype=np.float64)
        if not np.allclose(gnp, gts_arr, rtol=rtol, atol=atol):
            grad_ok = False
            messages.append(f"FAIL: W{k+1} gradient mismatch (NumPy vs TS)")
        else:
            messages.append(f"PASS: W{k+1} gradient match")
        bts = grads_ts.get(f"b{k+1}")
        bnp = grads_np[f"b{k+1}"]
        bts_arr = np.array(bts, dtype=np.float64) if np.ndim(bts) != 0 else float(bts)
        if not np.allclose(np.atleast_1d(bnp), np.atleast_1d(bts_arr), rtol=rtol, atol=atol):
            grad_ok = False
            messages.append(f"FAIL: b{k+1} gradient mismatch")
        else:
            messages.append(f"PASS: b{k+1} gradient match")
    if "a" in grads_ts and "a" in grads_np:
        if not np.isclose(grads_np["a"], grads_ts["a"], rtol=rtol, atol=atol):
            grad_ok = False
            messages.append("FAIL: a gradient mismatch")
        else:
            messages.append("PASS: a gradient match")

    if not HAS_TORCH:
        messages.append("SKIP: PyTorch not installed, only TS vs NumPy comparison done")
        return loss_ok, grad_ok

    ode_type = snap.get("ode_type", "exponential")
    u0 = snap.get("u0")
    torch_model = BILOModelTorch(
        n_hidden=n_hidden, depth=depth, ode_type=ode_type, u0=u0, seed=snap["seed"]
    )
    with torch.no_grad():
        for k in range(depth):
            torch_model._W[k].copy_(torch.from_numpy(np_model._W[k]))
            b = np_model._b[k]
            if np.ndim(b) == 0:
                torch_model._b[k].copy_(torch.tensor(b, dtype=torch.float64))
            else:
                torch_model._b[k].copy_(torch.from_numpy(b))

    L_torch = torch.tensor(0.0, dtype=torch.float64)
    for i in range(len(t_colloc)):
        t = torch.tensor(t_colloc[i], dtype=torch.float64, requires_grad=True)
        a = torch.tensor(a_colloc[i], dtype=torch.float64, requires_grad=True)
        N, u, u_t, u_a, u_ta, R, R_a = torch_model(t, a)
        L_torch = L_torch + 0.5 * w_res * R * R + 0.5 * w_grad * R_a * R_a
    if t_data is not None and u_data is not None and w_data != 0:
        for i in range(len(t_data)):
            t = torch.tensor(t_data[i], dtype=torch.float64)
            a = torch.tensor(a_data[i], dtype=torch.float64) if a_data is not None else torch.tensor(a_colloc[0], dtype=torch.float64)
            u_pred = torch_model.eval_u(t, a)
            L_torch = L_torch + 0.5 * w_data * (u_pred - u_data[i]) ** 2
    L_torch.backward()

    torch_ok = True
    for k in range(depth):
        gts = grads_ts.get(f"W{k+1}")
        gtorch = torch_model._W[k].grad.numpy()
        if not np.allclose(np.array(gts, dtype=np.float64), gtorch, rtol=rtol, atol=atol):
            torch_ok = False
            messages.append(f"FAIL: W{k+1} TS vs PyTorch gradient mismatch")
        else:
            messages.append(f"PASS: W{k+1} TS vs PyTorch match")
        bts = grads_ts.get(f"b{k+1}")
        gbt = torch_model._b[k].grad
        bts_arr = np.array(bts) if np.ndim(bts) != 0 else float(bts)
        gbt_np = gbt.numpy() if gbt.numel() > 1 else np.array(gbt.item())
        if not np.allclose(np.atleast_1d(bts_arr), np.atleast_1d(gbt_np), rtol=rtol, atol=atol):
            torch_ok = False
            messages.append(f"FAIL: b{k+1} TS vs PyTorch mismatch")
        else:
            messages.append(f"PASS: b{k+1} TS vs PyTorch match")

    if not torch_ok:
        messages.append("FAIL: some TS gradients did not match PyTorch")
    else:
        messages.append("PASS: all TS gradients match PyTorch")
    return loss_ok, grad_ok and torch_ok


def _run_verification_pinn(snap: dict, _W: list, _b: list, np_model: PINNModel, messages: list) -> tuple[bool, bool]:
    n_hidden = snap["n_hidden"]
    depth = snap["depth"]
    t_colloc = np.array(snap["t_colloc"], dtype=np.float64)
    a_colloc = np.array(snap["a_colloc"], dtype=np.float64)
    w_res = snap["w_res"]
    w_data = snap.get("w_data", 0)
    t_data = np.array(snap["t_data"], dtype=np.float64) if snap.get("t_data") else None
    u_data = np.array(snap["u_data"], dtype=np.float64) if snap.get("u_data") else None

    losses_np, grads_np = np_model.compute_losses_and_gradients_pinn(
        t_colloc, a_colloc,
        t_data=t_data, u_data=u_data,
        w_res=w_res, w_data=w_data,
    )
    losses_ts = snap["losses"]
    messages.append(f"Losses TS (PINN): L_res={losses_ts['L_res']:.8f} L_data={losses_ts['L_data']:.8f}")
    messages.append(f"Losses NP (PINN): L_res={losses_np['L_res']:.8f} L_data={losses_np['L_data']:.8f}")

    rtol, atol = 1e-5, 1e-8
    loss_ok = (
        np.isclose(losses_np["L_res"], losses_ts["L_res"], rtol=rtol, atol=atol)
        and np.isclose(losses_np["L_data"], losses_ts["L_data"], rtol=rtol, atol=atol)
    )
    if not loss_ok:
        messages.append("FAIL: PINN losses do not match TS vs NumPy")
    else:
        messages.append("PASS: PINN losses match (TS vs NumPy)")

    grads_ts = snap["grads"]
    grad_ok = True
    for k in range(depth):
        gts = grads_ts.get(f"W{k+1}")
        gnp = grads_np[f"W{k+1}"]
        if gts is None:
            grad_ok = False
            messages.append(f"FAIL: TS grads missing W{k+1}")
            continue
        gts_arr = np.array(gts, dtype=np.float64)
        if not np.allclose(gnp, gts_arr, rtol=rtol, atol=atol):
            grad_ok = False
            messages.append(f"FAIL: W{k+1} gradient mismatch (NumPy vs TS)")
        else:
            messages.append(f"PASS: W{k+1} gradient match")
        bts = grads_ts.get(f"b{k+1}")
        bnp = grads_np[f"b{k+1}"]
        bts_arr = np.array(bts, dtype=np.float64) if np.ndim(bts) != 0 else float(bts)
        if not np.allclose(np.atleast_1d(bnp), np.atleast_1d(bts_arr), rtol=rtol, atol=atol):
            grad_ok = False
            messages.append(f"FAIL: b{k+1} gradient mismatch")
        else:
            messages.append(f"PASS: b{k+1} gradient match")
    if "a" in grads_ts and "a" in grads_np:
        if not np.isclose(grads_np["a"], grads_ts["a"], rtol=rtol, atol=atol):
            grad_ok = False
            messages.append("FAIL: a gradient mismatch")
        else:
            messages.append("PASS: a gradient match")

    if not HAS_TORCH or PINNModelTorch is None:
        messages.append("SKIP: PyTorch not installed, only TS vs NumPy comparison done")
        return loss_ok, grad_ok

    ode_type = snap.get("ode_type", "exponential")
    u0 = snap.get("u0")
    a_val = float(a_colloc[0]) if len(a_colloc) > 0 else 1.0
    torch_model = PINNModelTorch(
        n_hidden=n_hidden,
        depth=depth,
        ode_type=ode_type,
        u0=u0,
        seed=snap["seed"],
        a_init=a_val,
    )
    with torch.no_grad():
        for k in range(depth):
            torch_model._W[k].copy_(torch.from_numpy(np_model._W[k]))
            b = np_model._b[k]
            if np.ndim(b) == 0:
                torch_model._b[k].copy_(torch.tensor(b, dtype=torch.float64))
            else:
                torch_model._b[k].copy_(torch.from_numpy(b))
        torch_model._W[0][:, 1] = 0.0
        torch_model.a.copy_(torch.tensor(a_val, dtype=torch.float64))

    L_torch = torch.tensor(0.0, dtype=torch.float64)
    for i in range(len(t_colloc)):
        t = torch.tensor(t_colloc[i], dtype=torch.float64)
        _, _, _, R = torch_model(t)
        L_torch = L_torch + 0.5 * w_res * R * R
    if t_data is not None and u_data is not None:
        for i in range(len(t_data)):
            t = torch.tensor(t_data[i], dtype=torch.float64)
            u_pred = torch_model.eval_u(t)
            L_torch = L_torch + 0.5 * w_data * (u_pred - u_data[i]) ** 2
    L_torch.backward()

    with torch.no_grad():
        torch_model._W[0].grad[:, 1] = 0.0

    torch_ok = True
    for k in range(depth):
        gts = grads_ts.get(f"W{k+1}")
        gtorch = torch_model._W[k].grad.numpy()
        if not np.allclose(np.array(gts, dtype=np.float64), gtorch, rtol=rtol, atol=atol):
            torch_ok = False
            messages.append(f"FAIL: W{k+1} TS vs PyTorch gradient mismatch")
        else:
            messages.append(f"PASS: W{k+1} TS vs PyTorch match")
        bts = grads_ts.get(f"b{k+1}")
        gbt = torch_model._b[k].grad
        bts_arr = np.array(bts) if np.ndim(bts) != 0 else float(bts)
        gbt_np = gbt.numpy() if gbt.numel() > 1 else np.array(gbt.item())
        if not np.allclose(np.atleast_1d(bts_arr), np.atleast_1d(gbt_np), rtol=rtol, atol=atol):
            torch_ok = False
            messages.append(f"FAIL: b{k+1} TS vs PyTorch mismatch")
        else:
            messages.append(f"PASS: b{k+1} TS vs PyTorch match")
    if "a" in grads_ts:
        if not np.isclose(grads_ts["a"], torch_model.a.grad.item(), rtol=rtol, atol=atol):
            torch_ok = False
            messages.append("FAIL: a gradient TS vs PyTorch mismatch")
        else:
            messages.append("PASS: a gradient TS vs PyTorch match")

    if not torch_ok:
        messages.append("FAIL: some PINN TS gradients did not match PyTorch")
    else:
        messages.append("PASS: all PINN TS gradients match PyTorch")
    return loss_ok, grad_ok and torch_ok


def run_verification(snapshot_path: str | Path) -> tuple[bool, list[str]]:
    messages = []
    snap = load_snapshot(snapshot_path)
    model_type = snap.get("model_type", "bilo")
    n_hidden = snap["n_hidden"]
    depth = snap["depth"]
    ode_type = snap.get("ode_type", "exponential")
    u0 = snap.get("u0")
    _W, _b = snapshot_to_numpy_weights(snap)

    rng = np.random.default_rng(snap["seed"])
    if model_type == "pinn":
        np_model = PINNModel(
            n_hidden=n_hidden, depth=depth, ode_type=ode_type, u0=u0, rng=rng
        )
    else:
        np_model = BILOModel(
            n_hidden=n_hidden, depth=depth, ode_type=ode_type, u0=u0, rng=rng
        )

    for k in range(depth):
        np_model._W[k] = _W[k].copy()
        np_model._b[k] = _b[k].copy() if np.ndim(_b[k]) != 0 else _b[k]
    if model_type == "pinn":
        np_model._W[0][:, 1] = 0.0

    if model_type == "pinn":
        loss_ok, grad_ok = _run_verification_pinn(snap, _W, _b, np_model, messages)
    else:
        loss_ok, grad_ok = _run_verification_bilo(snap, _W, _b, np_model, messages)

    return loss_ok and grad_ok, messages


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "ts_snapshot.json"
    if not Path(path).exists():
        print(f"Usage: python verify_ts_gradients.py [path/to/ts_snapshot.json]")
        print(f"Snapshot not found: {path}")
        print("Generate it by running the TypeScript snapshot writer (see docstring in this file).")
        sys.exit(2)
    ok, messages = run_verification(path)
    for m in messages:
        print(m)
    print("All checks passed." if ok else "Some checks failed.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
