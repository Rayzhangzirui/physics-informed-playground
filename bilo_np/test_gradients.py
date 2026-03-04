"""Unit tests: compare manual NumPy gradients with PyTorch autograd."""

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from model import BILOModel, BILOModelTorch


def sync_weights_np_to_torch(np_model: BILOModel, torch_model: "BILOModelTorch") -> None:
    """Copy weights from NumPy model to PyTorch model."""
    with torch.no_grad():
        torch_model.W1t.copy_(torch.from_numpy(np_model.W1t))
        torch_model.W1a.copy_(torch.from_numpy(np_model.W1a))
        torch_model.b1.copy_(torch.from_numpy(np_model.b1))
        torch_model.W2.copy_(torch.from_numpy(np_model.W2))
        torch_model.b2.copy_(torch.tensor(np_model.b2, dtype=torch.float64))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_forward_match():
    """Forward pass outputs should match between NumPy and PyTorch."""
    np.random.seed(42)
    n_hidden = 4
    np_model = BILOModel(n_hidden)
    torch_model = BILOModelTorch(n_hidden)
    sync_weights_np_to_torch(np_model, torch_model)

    t, a = 0.5, 1.0
    N_np, N_t_np, N_a_np, N_ta_np, u_np, u_t_np, u_a_np, u_ta_np, _, _, _, _, _ = np_model.forward(t, a)
    N_torch, u_torch, u_t_torch, u_a_torch, u_ta_torch, R_torch, Ra_torch = torch_model(
        torch.tensor(t, dtype=torch.float64, requires_grad=True),
        torch.tensor(a, dtype=torch.float64, requires_grad=True),
    )

    assert np.isclose(N_np, N_torch.item())
    assert np.isclose(u_np, u_torch.item())
    assert np.isclose(u_t_np, u_t_torch.item())
    assert np.isclose(u_a_np, u_a_torch.item())
    assert np.isclose(u_ta_np, u_ta_torch.item())
    R_a_np = u_ta_np - (u_np + a * u_a_np)
    assert np.isclose(u_t_np - a * u_np, R_torch.item())
    assert np.isclose(R_a_np, Ra_torch.item())


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_residual_gradients_match():
    """Manual gradients for L_res + L_grad should match PyTorch autograd."""
    np.random.seed(42)
    n_hidden = 4
    np_model = BILOModel(n_hidden)
    torch_model = BILOModelTorch(n_hidden)
    sync_weights_np_to_torch(np_model, torch_model)

    # Single collocation point
    t_colloc = np.array([0.5])
    a_colloc = np.array([1.0])

    # NumPy: manual gradients
    losses_np, grads_np = np_model.compute_losses_and_gradients(
        t_colloc, a_colloc, w_res=1.0, w_grad=1.0
    )

    # PyTorch: autograd
    t = torch.tensor(t_colloc[0], dtype=torch.float64, requires_grad=True)
    a = torch.tensor(a_colloc[0], dtype=torch.float64, requires_grad=True)
    N, u, u_t, u_a, u_ta, R, R_a = torch_model(t, a)
    L_res = 0.5 * R * R
    L_grad = 0.5 * R_a * R_a
    L_total = L_res + L_grad
    L_total.backward()

    # Compare gradients
    rtol, atol = 1e-5, 1e-8
    assert np.allclose(grads_np["W1t"], torch_model.W1t.grad.numpy(), rtol=rtol, atol=atol)
    assert np.allclose(grads_np["W1a"], torch_model.W1a.grad.numpy(), rtol=rtol, atol=atol)
    assert np.allclose(grads_np["b1"], torch_model.b1.grad.numpy(), rtol=rtol, atol=atol)
    assert np.allclose(grads_np["W2"], torch_model.W2.grad.numpy(), rtol=rtol, atol=atol)
    assert np.isclose(grads_np["b2"], torch_model.b2.grad.item(), rtol=rtol, atol=atol)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_data_loss_gradients_match():
    """Manual gradients for L_data (w.r.t. W) should match PyTorch autograd."""
    np.random.seed(43)
    n_hidden = 4
    np_model = BILOModel(n_hidden)
    torch_model = BILOModelTorch(n_hidden)
    sync_weights_np_to_torch(np_model, torch_model)

    t_data = np.array([0.3])
    a_data = np.array([1.2])
    u_data = np.array([1.5])  # target

    # NumPy: manual gradients (only L_data, no collocation)
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

    # PyTorch: autograd for L_data (W gradients)
    t = torch.tensor(t_data[0], dtype=torch.float64)
    a = torch.tensor(a_data[0], dtype=torch.float64)
    u_target = torch.tensor(u_data[0], dtype=torch.float64)
    u_pred = torch_model.eval_u(t, a)
    L_data = 0.5 * (u_pred - u_target) ** 2
    L_data.backward()

    rtol, atol = 1e-5, 1e-8
    assert np.allclose(grads_np["W1t"], torch_model.W1t.grad.numpy(), rtol=rtol, atol=atol)
    assert np.allclose(grads_np["W1a"], torch_model.W1a.grad.numpy(), rtol=rtol, atol=atol)
    assert np.allclose(grads_np["b1"], torch_model.b1.grad.numpy(), rtol=rtol, atol=atol)
    assert np.allclose(grads_np["W2"], torch_model.W2.grad.numpy(), rtol=rtol, atol=atol)
    assert np.isclose(grads_np["b2"], torch_model.b2.grad.item(), rtol=rtol, atol=atol)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_data_loss_gradient_wrt_a():
    """Manual gradient dL_data/da should match PyTorch autograd."""
    np.random.seed(45)
    n_hidden = 4
    np_model = BILOModel(n_hidden)
    torch_model = BILOModelTorch(n_hidden)
    sync_weights_np_to_torch(np_model, torch_model)

    t_data = np.array([0.5])
    a_data = np.array([1.0])
    u_data = np.array([2.0])  # target

    # NumPy: manual dL_data/da
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

    # PyTorch: autograd for dL_data/da (a must have requires_grad=True)
    t = torch.tensor(t_data[0], dtype=torch.float64)
    a = torch.tensor(a_data[0], dtype=torch.float64, requires_grad=True)
    u_target = torch.tensor(u_data[0], dtype=torch.float64)
    u_pred = torch_model.eval_u(t, a)
    L_data = 0.5 * (u_pred - u_target) ** 2
    L_data.backward()

    rtol, atol = 1e-5, 1e-8
    assert "a" in grads_np
    assert np.isclose(grads_np["a"], a.grad.item(), rtol=rtol, atol=atol)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_combined_loss_gradients_match():
    """Manual gradients for L_res + L_grad + L_data should match PyTorch autograd."""
    np.random.seed(44)
    n_hidden = 4
    np_model = BILOModel(n_hidden)
    torch_model = BILOModelTorch(n_hidden)
    sync_weights_np_to_torch(np_model, torch_model)

    t_colloc = np.array([0.25, 0.5, 0.75])
    a_colloc = np.array([1.0, 1.0, 1.0])
    t_data = np.array([0.5])
    a_data = np.array([1.0])
    u_data = np.array([1.65])  # exp(1*0.5) ≈ 1.65

    losses_np, grads_np = np_model.compute_losses_and_gradients(
        t_colloc, a_colloc,
        t_data=t_data, a_data=a_data, u_data=u_data,
        w_res=1.0, w_grad=0.1, w_data=0.5,
    )

    # PyTorch: build full loss
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
    assert np.allclose(grads_np["W1t"], torch_model.W1t.grad.numpy(), rtol=rtol, atol=atol)
    assert np.allclose(grads_np["W1a"], torch_model.W1a.grad.numpy(), rtol=rtol, atol=atol)
    assert np.allclose(grads_np["b1"], torch_model.b1.grad.numpy(), rtol=rtol, atol=atol)
    assert np.allclose(grads_np["W2"], torch_model.W2.grad.numpy(), rtol=rtol, atol=atol)
    assert np.isclose(grads_np["b2"], torch_model.b2.grad.item(), rtol=rtol, atol=atol)
