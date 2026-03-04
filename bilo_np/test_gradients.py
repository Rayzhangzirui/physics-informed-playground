"""Unit tests: compare manual NumPy gradients with PyTorch autograd.

Tests cover:
- Forward pass match (N, u, u_t, u_a, u_ta) for depth 2 and 3
- Gradients of loss w.r.t. all weights (W1..Wd, b1..bd) for residual, data, and combined loss
- Gradients w.r.t. inputs t and a (dN/dt, dN/da, d²N/dtda, and dL_data/da)
- Multiple depths: 1 hidden layer (depth=2) and 2 hidden layers (depth=3)
"""

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from model import BILOModel, BILOModelTorch


def sync_weights_np_to_torch(np_model: BILOModel, torch_model: "BILOModelTorch") -> None:
    """Copy weights from NumPy model to PyTorch model (any depth)."""
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


# ---- Forward match ----

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_forward_match(depth: int):
    """Forward pass outputs should match between NumPy and PyTorch for depth 2 and 3."""
    np.random.seed(42)
    n_hidden = 4
    np_model = BILOModel(n_hidden, depth=depth)
    torch_model = BILOModelTorch(n_hidden, depth=depth)
    sync_weights_np_to_torch(np_model, torch_model)

    t, a = 0.5, 1.0
    N_np, N_t_np, N_a_np, N_ta_np, u_np, u_t_np, u_a_np, u_ta_np, _, _ = np_model.forward(t, a)
    N_torch, u_torch, u_t_torch, u_a_torch, u_ta_torch, R_torch, Ra_torch = torch_model(
        torch.tensor(t, dtype=torch.float64, requires_grad=True),
        torch.tensor(a, dtype=torch.float64, requires_grad=True),
    )

    assert np.isclose(N_np, N_torch.item()), f"N mismatch depth={depth}"
    assert np.isclose(u_np, u_torch.item()), f"u mismatch depth={depth}"
    assert np.isclose(u_t_np, u_t_torch.item()), f"u_t mismatch depth={depth}"
    assert np.isclose(u_a_np, u_a_torch.item()), f"u_a mismatch depth={depth}"
    assert np.isclose(u_ta_np, u_ta_torch.item()), f"u_ta mismatch depth={depth}"
    R_np = u_t_np - a * u_np
    R_a_np = u_ta_np - (u_np + a * u_a_np)
    assert np.isclose(R_np, R_torch.item()), f"R mismatch depth={depth}"
    assert np.isclose(R_a_np, Ra_torch.item()), f"R_a mismatch depth={depth}"


# ---- Gradients w.r.t. weights: residual + gradient loss ----

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_residual_gradients_match(depth: int):
    """Manual gradients for L_res + L_grad should match PyTorch autograd."""
    np.random.seed(42)
    n_hidden = 4
    np_model = BILOModel(n_hidden, depth=depth)
    torch_model = BILOModelTorch(n_hidden, depth=depth)
    sync_weights_np_to_torch(np_model, torch_model)

    t_colloc = np.array([0.5])
    a_colloc = np.array([1.0])

    losses_np, grads_np = np_model.compute_losses_and_gradients(
        t_colloc, a_colloc, w_res=1.0, w_grad=1.0
    )

    t = torch.tensor(t_colloc[0], dtype=torch.float64, requires_grad=True)
    a = torch.tensor(a_colloc[0], dtype=torch.float64, requires_grad=True)
    N, u, u_t, u_a, u_ta, R, R_a = torch_model(t, a)
    L_total = 0.5 * R * R + 0.5 * R_a * R_a
    L_total.backward()

    rtol, atol = 1e-5, 1e-8
    for k in range(depth):
        assert np.allclose(
            grads_np[f"W{k+1}"], torch_model._W[k].grad.numpy(), rtol=rtol, atol=atol
        ), f"W{k+1} grad mismatch depth={depth}"
        gb = torch_model._b[k].grad
        bn = grads_np[f"b{k+1}"]
        if np.ndim(bn) == 0:
            assert np.isclose(bn, gb.item(), rtol=rtol, atol=atol), f"b{k+1} grad mismatch depth={depth}"
        else:
            assert np.allclose(bn, gb.numpy(), rtol=rtol, atol=atol), f"b{k+1} grad mismatch depth={depth}"


# ---- Gradients w.r.t. weights: data loss ----

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_data_loss_gradients_match(depth: int):
    """Manual gradients for L_data (w.r.t. all W and b) should match PyTorch autograd."""
    np.random.seed(43)
    n_hidden = 4
    np_model = BILOModel(n_hidden, depth=depth)
    torch_model = BILOModelTorch(n_hidden, depth=depth)
    sync_weights_np_to_torch(np_model, torch_model)

    t_data = np.array([0.3])
    a_data = np.array([1.2])
    u_data = np.array([1.5])

    losses_np, grads_np = np_model.compute_losses_and_gradients(
        t_colloc=np.array([]),
        a_colloc=np.array([]),
        t_data=t_data,
        a_data=a_data,
        u_data=u_data,
        w_res=0.0,
        w_grad=0.0,
        w_data=1.0,
    )

    t = torch.tensor(t_data[0], dtype=torch.float64)
    a = torch.tensor(a_data[0], dtype=torch.float64)
    u_target = torch.tensor(u_data[0], dtype=torch.float64)
    u_pred = torch_model.eval_u(t, a)
    L_data = 0.5 * (u_pred - u_target) ** 2
    L_data.backward()

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


# ---- Gradient w.r.t. input a (dL_data/da) ----

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_data_loss_gradient_wrt_a(depth: int):
    """Manual gradient dL_data/da should match PyTorch autograd."""
    np.random.seed(45)
    n_hidden = 4
    np_model = BILOModel(n_hidden, depth=depth)
    torch_model = BILOModelTorch(n_hidden, depth=depth)
    sync_weights_np_to_torch(np_model, torch_model)

    t_data = np.array([0.5])
    a_data = np.array([1.0])
    u_data = np.array([2.0])

    losses_np, grads_np = np_model.compute_losses_and_gradients(
        t_colloc=np.array([]),
        a_colloc=np.array([]),
        t_data=t_data,
        a_data=a_data,
        u_data=u_data,
        w_res=0.0,
        w_grad=0.0,
        w_data=1.0,
    )

    t = torch.tensor(t_data[0], dtype=torch.float64)
    a = torch.tensor(a_data[0], dtype=torch.float64, requires_grad=True)
    u_target = torch.tensor(u_data[0], dtype=torch.float64)
    u_pred = torch_model.eval_u(t, a)
    L_data = 0.5 * (u_pred - u_target) ** 2
    L_data.backward()

    rtol, atol = 1e-5, 1e-8
    assert "a" in grads_np
    assert np.isclose(grads_np["a"], a.grad.item(), rtol=rtol, atol=atol), f"dL/da mismatch depth={depth}"


# ---- Gradients w.r.t. inputs t and a (dN/dt, dN/da, d²N/dtda) ----

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_gradients_wrt_t_and_a(depth: int):
    """Compare NumPy N, N_t, N_a, N_ta and PyTorch autograd derivatives of N w.r.t. t and a."""
    np.random.seed(46)
    n_hidden = 4
    np_model = BILOModel(n_hidden, depth=depth)
    torch_model = BILOModelTorch(n_hidden, depth=depth)
    sync_weights_np_to_torch(np_model, torch_model)

    t_val, a_val = 0.5, 1.0
    N_np, N_t_np, N_a_np, N_ta_np, _, _, _, _, _, _ = np_model.forward(t_val, a_val)

    t = torch.tensor(t_val, dtype=torch.float64, requires_grad=True)
    a = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)
    N = torch_model.forward_N_only(t, a)
    N_torch = N.item()

    N_t_torch = torch.autograd.grad(N, t, retain_graph=True, create_graph=True)[0]
    N_a_torch = torch.autograd.grad(N, a, retain_graph=True, create_graph=True)[0]
    N_ta_torch = torch.autograd.grad(N_a_torch, t, retain_graph=False)[0]

    rtol, atol = 1e-6, 1e-9
    assert np.isclose(N_np, N_torch, rtol=rtol, atol=atol), f"N mismatch depth={depth}"
    assert np.isclose(N_t_np, N_t_torch.item(), rtol=rtol, atol=atol), f"N_t mismatch depth={depth}"
    assert np.isclose(N_a_np, N_a_torch.item(), rtol=rtol, atol=atol), f"N_a mismatch depth={depth}"
    assert np.isclose(N_ta_np, N_ta_torch.item(), rtol=rtol, atol=atol), f"N_ta mismatch depth={depth}"


# ---- Combined loss gradients ----

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_combined_loss_gradients_match(depth: int):
    """Manual gradients for L_res + L_grad + L_data should match PyTorch autograd."""
    np.random.seed(44)
    n_hidden = 4
    np_model = BILOModel(n_hidden, depth=depth)
    torch_model = BILOModelTorch(n_hidden, depth=depth)
    sync_weights_np_to_torch(np_model, torch_model)

    t_colloc = np.array([0.25, 0.5, 0.75])
    a_colloc = np.array([1.0, 1.0, 1.0])
    t_data = np.array([0.5])
    a_data = np.array([1.0])
    u_data = np.array([1.65])

    losses_np, grads_np = np_model.compute_losses_and_gradients(
        t_colloc, a_colloc,
        t_data=t_data, a_data=a_data, u_data=u_data,
        w_res=1.0, w_grad=0.1, w_data=0.5,
    )

    L_total_torch = torch.tensor(0.0, dtype=torch.float64)
    for i in range(len(t_colloc)):
        t = torch.tensor(t_colloc[i], dtype=torch.float64, requires_grad=True)
        a = torch.tensor(a_colloc[i], dtype=torch.float64, requires_grad=True)
        N, u, u_t, u_a, u_ta, R, R_a = torch_model(t, a)
        L_total_torch = L_total_torch + 0.5 * R * R + 0.1 * 0.5 * R_a * R_a

    t = torch.tensor(t_data[0], dtype=torch.float64)
    a = torch.tensor(a_data[0], dtype=torch.float64)
    u_target = torch.tensor(u_data[0], dtype=torch.float64)
    u_pred = torch_model.eval_u(t, a)
    L_total_torch = L_total_torch + 0.5 * 0.5 * (u_pred - u_target) ** 2

    L_total_torch.backward()

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
