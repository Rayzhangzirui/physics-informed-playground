"""BILO and PINN training routines."""

from __future__ import annotations

import numpy as np

from model import BILOModel, PINNModel

# ADAM defaults (match PyTorch torch.optim.Adam)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


def _adam_step_model(
    model: BILOModel,
    grads: dict,
    state: dict,
    step: int,
    lr: float,
    beta1: float = ADAM_BETA1,
    beta2: float = ADAM_BETA2,
    eps: float = ADAM_EPS,
) -> None:
    """Update model._W and model._b in place with ADAM. step is 1-based. state has 'm_W', 'v_W', 'm_b', 'v_b' (lists)."""
    if "m_W" not in state:
        state["m_W"] = [np.zeros_like(w) for w in model._W]
        state["v_W"] = [np.zeros_like(w) for w in model._W]
        state["m_b"] = [np.zeros_like(b) if np.ndim(b) != 0 else np.float64(0.0) for b in model._b]
        state["v_b"] = [np.zeros_like(b) if np.ndim(b) != 0 else np.float64(0.0) for b in model._b]
    for k in range(model.depth):
        gW = np.asarray(grads[f"W{k+1}"], dtype=np.float64)
        gb = np.asarray(grads[f"b{k+1}"], dtype=np.float64)
        mW, vW = state["m_W"][k], state["v_W"][k]
        mb, vb = state["m_b"][k], state["v_b"][k]
        mW[:] = beta1 * mW + (1.0 - beta1) * gW
        vW[:] = beta2 * vW + (1.0 - beta2) * (gW ** 2)
        # Scalar b (output layer): mb/vb are numpy scalars, no in-place assignment
        if np.ndim(mb) == 0:
            state["m_b"][k] = np.float64(beta1 * mb + (1.0 - beta1) * gb)
            state["v_b"][k] = np.float64(beta2 * vb + (1.0 - beta2) * (np.float64(gb) ** 2))
            mb = state["m_b"][k]
            vb = state["v_b"][k]
        else:
            mb[:] = beta1 * mb + (1.0 - beta1) * gb
            vb[:] = beta2 * vb + (1.0 - beta2) * (gb ** 2)
        mW_hat = mW / (1.0 - beta1**step)
        vW_hat = vW / (1.0 - beta2**step)
        mb_hat = mb / (1.0 - beta1**step)
        vb_hat = vb / (1.0 - beta2**step)
        model._W[k] = np.asarray(model._W[k], dtype=np.float64) - lr * mW_hat / (np.sqrt(vW_hat) + eps)
        if np.ndim(model._b[k]) == 0:
            model._b[k] = np.float64(model._b[k]) - lr * np.float64(mb_hat) / (np.sqrt(np.float64(vb_hat)) + eps)
        else:
            model._b[k] = np.asarray(model._b[k], dtype=np.float64) - lr * mb_hat / (np.sqrt(vb_hat) + eps)


def _adam_step_a(
    a: float,
    grad_a: float,
    state: dict,
    step: int,
    lr_a: float,
    beta1: float = ADAM_BETA1,
    beta2: float = ADAM_BETA2,
    eps: float = ADAM_EPS,
) -> float:
    """ADAM update for scalar a. Returns new a."""
    if "m_a" not in state:
        state["m_a"] = 0.0
        state["v_a"] = 0.0
    m, v = state["m_a"], state["v_a"]
    m = beta1 * m + (1.0 - beta1) * grad_a
    v = beta2 * v + (1.0 - beta2) * (grad_a ** 2)
    state["m_a"], state["v_a"] = m, v
    m_hat = m / (1.0 - beta1**step)
    v_hat = v / (1.0 - beta2**step)
    return a - lr_a * m_hat / (np.sqrt(v_hat) + eps)


def train_finetune(
    model: BILOModel,
    t_colloc: np.ndarray,
    a_learned: float,
    t_data: np.ndarray,
    u_data: np.ndarray,
    n_iters: int = 1000,
    lr: float = 0.01,
    lr_a: float = 0.001,
    w_res: float = 1.0,
    w_grad: float = 0.1,
    w_data: float = 0.5,
    log_every: int = 200,
    optimizer: str = "sgd",
) -> tuple[list[dict], float]:
    """Fine-tune: update W via L_res + L_grad, update a via L_data.

    Collocation uses a_learned (updated each step). Data is (t_data, u_data).
    optimizer: "sgd" or "adam". Returns (history, final_a_learned).
    """
    history = []
    a = float(a_learned)
    adam_state: dict = {}
    for step in range(n_iters):
        a_colloc = np.full_like(t_colloc, a)
        a_data = np.full_like(t_data, a)
        losses, grads = model.compute_losses_and_gradients(
            t_colloc,
            a_colloc,
            t_data=t_data,
            a_data=a_data,
            u_data=u_data,
            w_res=w_res,
            w_grad=w_grad,
            w_data=w_data,
        )
        L_res = losses["L_res"]
        L_grad = losses["L_grad"]
        L_data = losses["L_data"]
        L_total = w_res * L_res + w_grad * L_grad + w_data * L_data

        step1 = step + 1
        if optimizer == "adam":
            _adam_step_model(model, grads, adam_state, step1, lr)
            if "a" in grads:
                a = _adam_step_a(a, float(grads["a"]), adam_state, step1, lr_a)
        else:
            for k in range(model.depth):
                model._W[k] -= lr * grads[f"W{k+1}"]
                model._b[k] -= lr * grads[f"b{k+1}"]
            if "a" in grads:
                a -= lr_a * grads["a"]

        rec = {
            "step": step,
            "L_res": L_res,
            "L_grad": L_grad,
            "L_data": L_data,
            "L_total": L_total,
            "a": a,
        }
        history.append(rec)

        if step % log_every == 0 or step == n_iters - 1:
            print(
                f"step {step:5d}  L_res={L_res:.6f}  L_grad={L_grad:.6f}  "
                f"L_data={L_data:.6f}  a={a:.4f}  L_total={L_total:.6f}"
            )

    return history, a


def train(
    model: BILOModel,
    t_colloc: np.ndarray,
    a_colloc: np.ndarray,
    n_iters: int = 2000,
    lr: float = 0.02,
    w_res: float = 1.0,
    w_grad: float = 0.1,
    w_data: float = 0.0,
    t_data: np.ndarray | None = None,
    a_data: np.ndarray | None = None,
    u_data: np.ndarray | None = None,
    log_every: int = 200,
    optimizer: str = "sgd",
) -> list[dict]:
    """Train the BILO model. Returns history of losses per step.

    Args:
        model: BILOModel instance
        t_colloc, a_colloc: collocation points for physics loss
        n_iters: number of gradient steps
        lr: learning rate
        w_res: weight for L_res
        w_grad: weight for L_grad
        w_data: weight for L_data
        t_data, a_data, u_data: optional data points for L_data
        log_every: print every N steps
        optimizer: "sgd" or "adam"

    Returns:
        List of {step, L_res, L_grad, L_data, L_total} dicts
    """
    history = []
    adam_state: dict = {}
    for step in range(n_iters):
        losses, grads = model.compute_losses_and_gradients(
            t_colloc,
            a_colloc,
            t_data=t_data,
            a_data=a_data,
            u_data=u_data,
            w_res=w_res,
            w_grad=w_grad,
            w_data=w_data,
        )
        L_res = losses["L_res"]
        L_grad = losses["L_grad"]
        L_data = losses["L_data"]
        L_total = w_res * L_res + w_grad * L_grad + w_data * L_data

        step1 = step + 1
        if optimizer == "adam":
            _adam_step_model(model, grads, adam_state, step1, lr)
        else:
            for k in range(model.depth):
                model._W[k] -= lr * grads[f"W{k+1}"]
                model._b[k] -= lr * grads[f"b{k+1}"]

        rec = {
            "step": step,
            "L_res": L_res,
            "L_grad": L_grad,
            "L_data": L_data,
            "L_total": L_total,
        }
        history.append(rec)

        if step % log_every == 0 or step == n_iters - 1:
            print(
                f"step {step:5d}  L_res={L_res:.6f}  L_grad={L_grad:.6f}  "
                f"L_data={L_data:.6f}  L_total={L_total:.6f}"
            )

    return history


# -----------------------------------------------------------------------------
# PINN training
# -----------------------------------------------------------------------------


def train_pinn(
    model: PINNModel,
    t_colloc: np.ndarray,
    a_colloc: np.ndarray,
    n_iters: int = 1000,
    lr: float = 0.01,
    w_res: float = 1.0,
    t_data: np.ndarray | None = None,
    u_data: np.ndarray | None = None,
    w_data: float = 0.0,
    update_a: bool = False,
    lr_a: float = 0.001,
    log_every: int = 200,
) -> tuple[list[dict], float]:
    """Train PINN: update W via L_res (+ L_data if t_data/u_data given).

    If update_a=True and (t_data, u_data) given: also update a via L_res + L_data.
    If update_a=False: fix a (pretrain, solve PDE forward).
    Returns (history, final_a).
    """
    history = []
    a = float(a_colloc[0])  # same a for all colloc
    for step in range(n_iters):
        a_colloc_step = np.full_like(t_colloc, a)
        losses, grads = model.compute_losses_and_gradients_pinn(
            t_colloc,
            a_colloc_step,
            t_data=t_data,
            u_data=u_data,
            w_res=w_res,
            w_data=w_data,
        )
        L_res = losses["L_res"]
        L_data = losses["L_data"]
        L_total = w_res * L_res + w_data * L_data

        for k in range(model.depth):
            model._W[k] -= lr * grads[f"W{k+1}"]
            model._b[k] -= lr * grads[f"b{k+1}"]

        if update_a:
            a -= lr_a * grads["a"]

        rec = {
            "step": step,
            "L_res": L_res,
            "L_grad": 0.0,
            "L_data": L_data,
            "L_total": L_total,
            "a": a,
        }
        history.append(rec)

        if step % log_every == 0 or step == n_iters - 1:
            log = f"step {step:5d}  L_res={L_res:.6f}  L_data={L_data:.6f}"
            if update_a:
                log += f"  a={a:.4f}"
            log += f"  L_total={L_total:.6f}"
            print(log)

    return history, a


def train_pinn_finetune(
    model: PINNModel,
    t_colloc: np.ndarray,
    a_learned: float,
    t_data: np.ndarray,
    u_data: np.ndarray,
    n_iters: int = 1000,
    lr: float = 0.01,
    lr_a: float = 0.001,
    w_res: float = 1.0,
    w_data: float = 0.5,
    log_every: int = 200,
) -> tuple[list[dict], float]:
    """Train PINN with data: update W and a. Wrapper around train_pinn."""
    return train_pinn(
        model, t_colloc, np.full_like(t_colloc, a_learned),
        n_iters=n_iters, lr=lr, w_res=w_res,
        t_data=t_data, u_data=u_data, w_data=w_data,
        update_a=True, lr_a=lr_a, log_every=log_every,
    )
