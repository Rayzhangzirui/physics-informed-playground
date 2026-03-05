"""Unit tests: compare manual NumPy PINN gradients with PyTorch autograd.

PINN: u(t; W), a is trainable parameter. L = L_res + L_data. No L_grad.
- W1[:,1] is fixed to 0 (never updated).
- dL_res/da = -sum(R*u) over collocation; dL_data/da = 0.

Tests cover:
- Forward: u does not depend on a (W1[:,1]=0)
- Gradients w.r.t. W (with W1[:,1] zeroed): L_res, L_data, and combined
- Gradient w.r.t. a: dL_res/da from residual only
"""

import sys
from pathlib import Path

# Ensure this directory is on path so "from model import ..." works when run via pytest from any cwd
_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from model import PINNModel, PINNModelTorch


def sync_weights_np_to_torch(
    np_model: PINNModel, torch_model: "PINNModelTorch", a: float | None = None
) -> None:
    """Copy weights from NumPy PINN to PyTorch PINN. Optionally set a (default 1.0)."""
    if np_model.depth != torch_model.depth:
        raise ValueError("depth mismatch")
    with torch.no_grad():
        for k in range(np_model.depth):
            torch_model._W[k].copy_(torch.from_numpy(np_model._W[k]))
            b = np_model._b[k]
            if np.ndim(b) == 0:
                torch_model._b[k].copy_(torch.tensor(b, dtype=torch.float64))
            else:
                torch_model._b[k].copy_(torch.from_numpy(b))
        torch_model.a.copy_(torch.tensor(a if a is not None else 1.0, dtype=torch.float64))
        # Ensure W1[:,1]=0
        torch_model._W[0][:, 1] = 0.0


# ---- Forward: u independent of a ----

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("ode_type", ["exponential", "logistic"])
@pytest.mark.parametrize("depth", [2, 3])
def test_pinn_forward_u_independent_of_a(ode_type: str, depth: int):
    """PINN u(t; W) should not depend on a (W1[:,1]=0)."""
    np.random.seed(42)
    model = PINNModel(n_hidden=4, depth=depth, ode_type=ode_type)
    t_val = 0.5
    u_a0 = model.eval_u(t_val, 0.0)
    u_a1 = model.eval_u(t_val, 1.0)
    u_a2 = model.eval_u(t_val, 2.5)
    assert np.isclose(u_a0, u_a1), "u should not depend on a"
    assert np.isclose(u_a1, u_a2), "u should not depend on a"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("ode_type", ["exponential", "logistic"])
@pytest.mark.parametrize("depth", [2, 3])
def test_pinn_forward_match_torch(ode_type: str, depth: int):
    """PINN forward u, u_t should match PyTorch."""
    np.random.seed(42)
    np_model = PINNModel(n_hidden=4, depth=depth, ode_type=ode_type)
    torch_model = PINNModelTorch(n_hidden=4, depth=depth, ode_type=ode_type, a_init=1.0)
    sync_weights_np_to_torch(np_model, torch_model)

    t_val = 0.5
    u_np = np_model.eval_u(t_val)
    _, _, _, _, _, u_t_np, _, _, _, _ = np_model.forward(t_val, 1.0)  # u_t at index 5

    t = torch.tensor(t_val, dtype=torch.float64)
    N_torch, u_torch, u_t_torch, R_torch = torch_model(t)
    u_torch = u_torch.item()
    u_t_torch = u_t_torch.item()

    rtol, atol = 1e-6, 1e-9
    assert np.isclose(u_np, u_torch, rtol=rtol, atol=atol), f"u mismatch depth={depth}"
    assert np.isclose(u_t_np, u_t_torch, rtol=rtol, atol=atol), f"u_t mismatch depth={depth}"


# ---- Gradients w.r.t. W: L_res only ----

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("ode_type", ["exponential", "logistic"])
@pytest.mark.parametrize("depth", [2, 3])
def test_pinn_residual_gradients_W_match(ode_type: str, depth: int):
    """Manual gradients for L_res w.r.t. W (with W1[:,1]=0) should match PyTorch."""
    np.random.seed(42)
    np_model = PINNModel(n_hidden=4, depth=depth, ode_type=ode_type)
    torch_model = PINNModelTorch(n_hidden=4, depth=depth, ode_type=ode_type, a_init=1.0)
    sync_weights_np_to_torch(np_model, torch_model)

    t_colloc = np.array([0.25, 0.5, 0.75])
    a_colloc = np.array([1.0, 1.0, 1.0])

    losses_np, grads_np = np_model.compute_losses_and_gradients_pinn(
        t_colloc, a_colloc, w_res=1.0, w_data=0.0
    )

    L_res_torch = torch.tensor(0.0, dtype=torch.float64)
    for i in range(len(t_colloc)):
        t = torch.tensor(t_colloc[i], dtype=torch.float64)
        _, _, _, R = torch_model(t)
        L_res_torch = L_res_torch + 0.5 * R * R
    L_res_torch.backward()

    # Zero W1[:,1] grad in PyTorch (PINN doesn't update it)
    with torch.no_grad():
        torch_model._W[0].grad[:, 1] = 0.0

    rtol, atol = 1e-5, 1e-8
    for k in range(depth):
        assert np.allclose(
            grads_np[f"W{k+1}"], torch_model._W[k].grad.numpy(), rtol=rtol, atol=atol
        ), f"W{k+1} grad mismatch depth={depth}"
        bn = grads_np[f"b{k+1}"]
        gb = torch_model._b[k].grad
        if np.ndim(bn) == 0:
            assert np.isclose(bn, gb.item(), rtol=rtol, atol=atol)
        else:
            assert np.allclose(bn, gb.numpy(), rtol=rtol, atol=atol)


# ---- Gradients w.r.t. W: L_data only ----

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("ode_type", ["exponential", "logistic"])
@pytest.mark.parametrize("depth", [2, 3])
def test_pinn_data_loss_gradients_W_match(ode_type: str, depth: int):
    """Manual gradients for L_data w.r.t. W should match PyTorch."""
    np.random.seed(43)
    np_model = PINNModel(n_hidden=4, depth=depth, ode_type=ode_type)
    torch_model = PINNModelTorch(n_hidden=4, depth=depth, ode_type=ode_type, a_init=1.0)
    sync_weights_np_to_torch(np_model, torch_model)

    t_data = np.array([0.3, 0.6])
    u_data = np.array([1.5, 2.2])

    losses_np, grads_np = np_model.compute_losses_and_gradients_pinn(
        t_colloc=np.array([]),
        a_colloc=np.array([]),
        t_data=t_data,
        u_data=u_data,
        w_res=0.0,
        w_data=1.0,
    )

    L_data_torch = torch.tensor(0.0, dtype=torch.float64)
    for i in range(len(t_data)):
        t = torch.tensor(t_data[i], dtype=torch.float64)
        u_target = torch.tensor(u_data[i], dtype=torch.float64)
        u_pred = torch_model.eval_u(t)
        L_data_torch = L_data_torch + 0.5 * (u_pred - u_target) ** 2
    L_data_torch.backward()

    with torch.no_grad():
        torch_model._W[0].grad[:, 1] = 0.0

    rtol, atol = 1e-5, 1e-8
    for k in range(depth):
        assert np.allclose(
            grads_np[f"W{k+1}"], torch_model._W[k].grad.numpy(), rtol=rtol, atol=atol
        ), f"W{k+1} grad mismatch depth={depth}"
        bn = grads_np[f"b{k+1}"]
        gb = torch_model._b[k].grad
        if np.ndim(bn) == 0:
            assert np.isclose(bn, gb.item(), rtol=rtol, atol=atol)
        else:
            assert np.allclose(bn, gb.numpy(), rtol=rtol, atol=atol)


# ---- Gradient w.r.t. a: dL_res/da ----

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("ode_type", ["exponential", "logistic"])
@pytest.mark.parametrize("depth", [2, 3])
def test_pinn_gradient_wrt_a(ode_type: str, depth: int):
    """Manual dL_res/da (exponential: -R*u, logistic: -R*u*(1-u)) should match PyTorch a.grad."""
    np.random.seed(44)
    np_model = PINNModel(n_hidden=4, depth=depth, ode_type=ode_type)
    torch_model = PINNModelTorch(n_hidden=4, depth=depth, ode_type=ode_type, a_init=1.2)
    a_val = 1.2
    sync_weights_np_to_torch(np_model, torch_model, a=a_val)

    t_colloc = np.array([0.25, 0.5, 0.75])
    a_colloc = np.array([a_val, a_val, a_val])

    losses_np, grads_np = np_model.compute_losses_and_gradients_pinn(
        t_colloc, a_colloc, w_res=1.0, w_data=0.0
    )

    L_res_torch = torch.tensor(0.0, dtype=torch.float64)
    for i in range(len(t_colloc)):
        t = torch.tensor(t_colloc[i], dtype=torch.float64)
        _, _, _, R = torch_model(t)
        L_res_torch = L_res_torch + 0.5 * R * R
    L_res_torch.backward()

    rtol, atol = 1e-5, 1e-8
    assert "a" in grads_np
    assert np.isclose(
        grads_np["a"], torch_model.a.grad.item(), rtol=rtol, atol=atol
    ), f"dL/da mismatch depth={depth}"


# ---- Combined loss: L_res + L_data, gradients for W and a ----

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("ode_type", ["exponential", "logistic"])
@pytest.mark.parametrize("depth", [2, 3])
def test_pinn_combined_loss_gradients_match(ode_type: str, depth: int):
    """Manual gradients for L_res + L_data (W and a) should match PyTorch."""
    np.random.seed(45)
    np_model = PINNModel(n_hidden=4, depth=depth, ode_type=ode_type)
    torch_model = PINNModelTorch(n_hidden=4, depth=depth, ode_type=ode_type, a_init=1.0)
    sync_weights_np_to_torch(np_model, torch_model)

    t_colloc = np.array([0.25, 0.5, 0.75])
    a_colloc = np.array([1.0, 1.0, 1.0])
    t_data = np.array([0.5])
    u_data = np.array([1.65])

    losses_np, grads_np = np_model.compute_losses_and_gradients_pinn(
        t_colloc, a_colloc,
        t_data=t_data, u_data=u_data,
        w_res=1.0, w_data=0.5,
    )

    L_total_torch = torch.tensor(0.0, dtype=torch.float64)
    for i in range(len(t_colloc)):
        t = torch.tensor(t_colloc[i], dtype=torch.float64)
        _, _, _, R = torch_model(t)
        L_total_torch = L_total_torch + 0.5 * R * R

    t = torch.tensor(t_data[0], dtype=torch.float64)
    u_target = torch.tensor(u_data[0], dtype=torch.float64)
    u_pred = torch_model.eval_u(t)
    L_total_torch = L_total_torch + 0.5 * 0.5 * (u_pred - u_target) ** 2

    L_total_torch.backward()

    with torch.no_grad():
        torch_model._W[0].grad[:, 1] = 0.0

    rtol, atol = 1e-4, 1e-6
    for k in range(depth):
        assert np.allclose(
            grads_np[f"W{k+1}"], torch_model._W[k].grad.numpy(), rtol=rtol, atol=atol
        ), f"W{k+1} grad mismatch depth={depth}"
        bn = grads_np[f"b{k+1}"]
        gb = torch_model._b[k].grad
        if np.ndim(bn) == 0:
            assert np.isclose(bn, gb.item(), rtol=rtol, atol=atol)
        else:
            assert np.allclose(bn, gb.numpy(), rtol=rtol, atol=atol)

    assert np.isclose(
        grads_np["a"], torch_model.a.grad.item(), rtol=rtol, atol=atol
    ), "dL/da mismatch"


# ---- Edge cases: zero weights ----

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("ode_type", ["exponential", "logistic"])
def test_pinn_w_res_zero_grad_a_zero(ode_type: str):
    """When w_res=0, gradient of a must be 0 (L_data does not depend on a)."""
    np.random.seed(46)
    np_model = PINNModel(n_hidden=4, depth=2, ode_type=ode_type)

    t_colloc = np.array([0.25, 0.5, 0.75])
    a_colloc = np.array([1.0, 1.0, 1.0])
    t_data = np.array([0.5])
    u_data = np.array([1.65])

    _, grads_np = np_model.compute_losses_and_gradients_pinn(
        t_colloc, a_colloc,
        t_data=t_data, u_data=u_data,
        w_res=0.0, w_data=1.0,
    )
    assert "a" in grads_np
    assert np.isclose(grads_np["a"], 0.0, atol=1e-14), "grads['a'] must be 0 when w_res=0"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("ode_type", ["exponential", "logistic"])
def test_pinn_both_weights_zero_no_update(ode_type: str):
    """When w_res=0 and w_data=0, all gradients should be zero."""
    np.random.seed(47)
    np_model = PINNModel(n_hidden=4, depth=2, ode_type=ode_type)

    t_colloc = np.array([0.5])
    a_colloc = np.array([1.0])

    _, grads_np = np_model.compute_losses_and_gradients_pinn(
        t_colloc, a_colloc, w_res=0.0, w_data=0.0
    )
    for k in range(np_model.depth):
        assert np.allclose(grads_np[f"W{k+1}"], 0.0, atol=1e-14), f"W{k+1} grad must be 0"
        assert np.allclose(np.atleast_1d(grads_np[f"b{k+1}"]), 0.0, atol=1e-14), f"b{k+1} grad must be 0"
    assert np.isclose(grads_np["a"], 0.0, atol=1e-14), "grads['a'] must be 0"
