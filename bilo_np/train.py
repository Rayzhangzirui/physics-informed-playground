"""BILO and PINN training routines."""

import numpy as np

from model import BILOModel, PINNModel


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
) -> tuple[list[dict], float]:
    """Fine-tune: update W via L_res + L_grad, update a via L_data.

    Collocation uses a_learned (updated each step). Data is (t_data, u_data).
    Returns (history, final_a_learned).
    """
    history = []
    a = float(a_learned)
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
        L_total = L_res + L_grad + L_data

        # Update W and b for all layers
        for k in range(model.depth):
            model._W[k] -= lr * grads[f"W{k+1}"]
            model._b[k] -= lr * grads[f"b{k+1}"]

        # Update a
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

    Returns:
        List of {step, L_res, L_grad, L_data, L_total} dicts
    """
    history = []
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
        L_total = L_res + L_grad + L_data

        # Gradient descent
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
        L_total = L_res + L_data

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
