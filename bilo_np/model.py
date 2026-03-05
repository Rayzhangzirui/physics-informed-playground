"""BILO model: manual implementation with NumPy and PyTorch.

Supports two ODEs:

1) Exponential: u_t = a*u, u(0)=1. Trial u = 1 + t*N(t,a).
   R = u_t - a*u, R_a = u_ta - (u + a*u_a).

2) Logistic: u_t = a*u*(1-u), u(0)=u0 (default 0.1). Trial u = u0 + t*N(t,a).
   R = u_t - a*u*(1-u), R_a = u_ta - u(1-u) - a*u_a*(1-2u).

Residual loss L_res = MSE(R), residual gradient loss L_grad = MSE(R_a), data loss L_data = MSE(u - u_data).

Architecture (d-layer PINN):
- Input: x = [t, a] in R^2, h_0 = x
- Hidden layers k=1..d-1: z_k = W_k h_{k-1} + b_k, h_k = sigma(z_k)
  - W_1 in R^{n x 2}, W_k in R^{n x n} for k>1, b_k in R^n
- Output layer d: N = z_d = W_d h_{d-1} + b_d, W_d in R^{1 x n}, b_d scalar

Forward kinematics propagate h_{k,t}, h_{k,a}, h_{k,ta} forward.
Backward pass computes adjoints and extracts dL/dW_k, dL/db_k.

For tanh: sigma' = 1 - sigma^2, sigma'' = -2*sigma*sigma', sigma''' = 2*sigma'*(3*sigma^2 - 1)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ODE_TYPE = Literal["exponential", "logistic"]
U0_LOGISTIC = 0.1


def _tanh_derivatives(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute sigma, sigma', sigma'', sigma''' for tanh activation."""
    sigma = np.tanh(z)
    sigma_p = 1.0 - sigma * sigma
    sigma_pp = -2.0 * sigma * sigma_p
    sigma_ppp = 2.0 * sigma_p * (3.0 * sigma * sigma - 1.0)
    return sigma, sigma_p, sigma_pp, sigma_ppp


def logistic_solution(t: np.ndarray, a: float, u0: float = U0_LOGISTIC) -> np.ndarray:
    """Analytical solution for u_t = a*u*(1-u), u(0)=u0: u(t) = u0*e^{at} / (1 + u0*(e^{at}-1))."""
    eat = np.exp(a * t)
    return (u0 * eat) / (1.0 + u0 * (eat - 1.0))


class BILOModel:
    """BILO model with manual backprop in NumPy, generalized to d hidden layers.

    Parameters:
        n_hidden: width n of each hidden layer
        depth: number of layers d (depth=2 => one hidden layer, depth=3 => two hidden, etc.)
        ode_type: "exponential" (u'=au, u(0)=1) or "logistic" (u'=au(1-u), u(0)=u0)
        u0: initial condition for logistic (default 0.1)
        Weights: W1 (n, 2), b1 (n,); W_k (n, n), b_k (n,) for k=2..d-1; W_d (n,), b_d scalar
    """

    def __init__(
        self,
        n_hidden: int,
        depth: int = 2,
        ode_type: ODE_TYPE = "exponential",
        u0: float = U0_LOGISTIC,
        rng: np.random.Generator | None = None,
    ):
        if depth < 2:
            raise ValueError("depth must be >= 2 (input -> at least one hidden -> output)")
        self.n_hidden = n_hidden
        self.depth = depth
        self.ode_type = ode_type
        self.u0 = u0
        self.rng = rng or np.random.default_rng(42)
        n = n_hidden
        d = depth

        # Layer 1: W1 (n, 2), b1 (n,)
        self._W = [self.rng.uniform(-0.5, 0.5, size=(n, 2)).astype(np.float64)]
        self._b = [np.zeros(n, dtype=np.float64)]
        # Layers 2 .. d-1: W_k (n, n), b_k (n,)
        for _ in range(d - 2):
            self._W.append(self.rng.uniform(-0.5, 0.5, size=(n, n)).astype(np.float64))
            self._b.append(np.zeros(n, dtype=np.float64))
        # Output layer d: W_d (n,) row vector, b_d scalar
        self._W.append(self.rng.uniform(-0.5, 0.5, size=n).astype(np.float64))
        self._b.append(0.0)

    def parameters(self) -> dict[str, np.ndarray | float]:
        """Return all parameters as a dict (W1, b1, W2, b2, ..., Wd, bd)."""
        out = {}
        for k in range(self.depth):
            out[f"W{k+1}"] = self._W[k]
            out[f"b{k+1}"] = self._b[k]
        return out

    def _forward_and_kinematics(
        self, t: float, a: float
    ) -> tuple[
        float, float, float, float,
        list[np.ndarray], list[np.ndarray], list[np.ndarray],
        list[np.ndarray], list[np.ndarray], list[np.ndarray],
        list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ]:
        """Forward pass plus forward kinematics. Returns N, N_t, N_a, N_ta and all layer states.

        Returns:
            N, N_t, N_a, N_ta (scalars)
            h_list: h_0..h_{d-1} (h_0 is (2,), rest (n,))
            z_list: z_1..z_{d-1} (each (n,))
            h_t_list, h_a_list, h_ta_list: derivatives of h at each layer
            z_t_list, z_a_list, z_ta_list: derivatives of z at hidden layers
            sigma_list: for k=1..d-1, (sigma, sigma_p, sigma_pp, sigma_ppp) at z_k
        """
        x = np.array([t, a], dtype=np.float64)
        h_0_t = np.array([1.0, 0.0], dtype=np.float64)
        h_0_a = np.array([0.0, 1.0], dtype=np.float64)
        h_0_ta = np.array([0.0, 0.0], dtype=np.float64)

        h_list = [x]
        z_list = []
        h_t_list = [h_0_t]
        h_a_list = [h_0_a]
        h_ta_list = [h_0_ta]
        z_t_list = []
        z_a_list = []
        z_ta_list = []
        sigma_list = []

        # Hidden layers k = 1 .. d-1
        for k in range(self.depth - 1):
            W_k = self._W[k]
            b_k = self._b[k]
            h_prev = h_list[-1]
            h_prev_t = h_t_list[-1]
            h_prev_a = h_a_list[-1]
            h_prev_ta = h_ta_list[-1]

            z_k = W_k @ h_prev + b_k
            sigma, sigma_p, sigma_pp, sigma_ppp = _tanh_derivatives(z_k)

            z_k_t = W_k @ h_prev_t
            z_k_a = W_k @ h_prev_a
            z_k_ta = W_k @ h_prev_ta

            h_k_t = sigma_p * z_k_t
            h_k_a = sigma_p * z_k_a
            h_k_ta = sigma_pp * z_k_t * z_k_a + sigma_p * z_k_ta

            z_list.append(z_k)
            h_list.append(sigma)
            h_t_list.append(h_k_t)
            h_a_list.append(h_k_a)
            h_ta_list.append(h_k_ta)
            z_t_list.append(z_k_t)
            z_a_list.append(z_k_a)
            z_ta_list.append(z_k_ta)
            sigma_list.append((sigma, sigma_p, sigma_pp, sigma_ppp))

        # Output layer d: N = W_d @ h_{d-1} + b_d
        W_d = self._W[-1]
        b_d = self._b[-1]
        h_last = h_list[-1]
        h_last_t = h_t_list[-1]
        h_last_a = h_a_list[-1]
        h_last_ta = h_ta_list[-1]

        N = float(np.dot(W_d, h_last) + b_d)
        N_t = float(np.dot(W_d, h_last_t))
        N_a = float(np.dot(W_d, h_last_a))
        N_ta = float(np.dot(W_d, h_last_ta))

        return (
            N, N_t, N_a, N_ta,
            h_list, z_list, h_t_list, h_a_list, h_ta_list,
            z_t_list, z_a_list, z_ta_list,
            sigma_list,
        )

    def forward(
        self, t: float, a: float
    ) -> tuple[
        float, float, float, float,
        float, float, float, float,
        list[np.ndarray], list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ]:
        """Forward pass. Returns (N, N_t, N_a, N_ta, u, u_t, u_a, u_ta, h_list, sigma_list).

        sigma_list[k] = (sigma, sigma_p, sigma_pp, sigma_ppp) at layer k+1 for k=0..d-2.
        For exponential: u = 1 + t*N; for logistic: u = u0 + t*N.
        """
        (
            N, N_t, N_a, N_ta,
            h_list, z_list, h_t_list, h_a_list, h_ta_list,
            z_t_list, z_a_list, z_ta_list,
            sigma_list,
        ) = self._forward_and_kinematics(t, a)

        if self.ode_type == "exponential":
            u = 1.0 + t * N
        else:
            u = self.u0 + t * N
        u_t = N + t * N_t
        u_a = t * N_a
        u_ta = N_a + t * N_ta

        return N, N_t, N_a, N_ta, u, u_t, u_a, u_ta, h_list, sigma_list

    def residuals(self, t: float, a: float) -> tuple[float, float]:
        """Compute R and R_a at (t, a)."""
        _, _, _, _, u, u_t, u_a, u_ta, _, _ = self.forward(t, a)
        if self.ode_type == "exponential":
            R = u_t - a * u
            R_a = u_ta - (u + a * u_a)
        else:
            # logistic: u_t = a*u*(1-u) => R = u_t - a*u*(1-u), R_a = u_ta - u(1-u) - a*u_a*(1-2u)
            R = u_t - a * u * (1.0 - u)
            R_a = u_ta - u * (1.0 - u) - a * u_a * (1.0 - 2.0 * u)
        return R, R_a

    def _backward_one_point(
        self,
        t: float,
        a: float,
        R: float,
        R_a: float,
        delta_N_data: float,
        w_res: float,
        w_grad: float,
        w_data: float,
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray | float],
        float,
    ]:
        """Backprop for a single (t,a) point. Returns grad_W_list, grad_b_list, dL_da (for data)."""
        (
            N, N_t, N_a, N_ta,
            h_list, z_list, h_t_list, h_a_list, h_ta_list,
            z_t_list, z_a_list, z_ta_list,
            sigma_list,
        ) = self._forward_and_kinematics(t, a)

        if self.ode_type == "exponential":
            u = 1.0 + t * N
            # dR/dN = 1 - a*t, dR/dN_t = t; dR_a/dN = -t, dR_a/dN_a = 1 - a*t, dR_a/dN_ta = t
            delta_N = w_res * R * (1.0 - a * t) - w_grad * R_a * t + delta_N_data
            delta_N_t = w_res * R * t
            delta_N_a = w_grad * R_a * (1.0 - a * t)
            delta_N_ta = w_grad * R_a * t
        else:
            # logistic: u = u0 + t*N, u_a = t*N_a
            u = self.u0 + t * N
            u_a = t * N_a
            # R = u_t - a*u*(1-u): dR/dN = 1 - a*(1-2u)*t, dR/dN_t = t
            # R_a = u_ta - u(1-u) - a*u_a*(1-2u): dR_a/dN = -t*(1-2u) + 2*a*u_a*t;
            #   dR_a/dN_a = 1 - a*t*(1-2u) (u(1-u) has no N_a, u_a has coeff t), dR_a/dN_ta = t
            one_minus_2u = 1.0 - 2.0 * u
            delta_N = (
                w_res * R * (1.0 - a * t * one_minus_2u)
                + w_grad * R_a * t * (-one_minus_2u + 2.0 * a * u_a)
                + delta_N_data
            )
            delta_N_t = w_res * R * t
            delta_N_a = w_grad * R_a * (1.0 - a * t * one_minus_2u)
            delta_N_ta = w_grad * R_a * t

        # Output layer d: delta_h_{d-1} = W_d^T * delta_N (W_d is (n,), so W_d^T delta_N = delta_N * W_d)
        W_d = self._W[-1]
        delta_h = np.float64(delta_N) * W_d
        delta_h_t = np.float64(delta_N_t) * W_d
        delta_h_a = np.float64(delta_N_a) * W_d
        delta_h_ta = np.float64(delta_N_ta) * W_d

        grad_W = [np.empty_like(w) for w in self._W]
        grad_b = [np.empty_like(b) for b in self._b]

        # Gradient for output layer: dL/dW_d = delta_N * h_{d-1} + ... , dL/db_d = delta_N
        grad_W[-1] = (
            delta_N * h_list[-1]
            + delta_N_t * h_t_list[-1]
            + delta_N_a * h_a_list[-1]
            + delta_N_ta * h_ta_list[-1]
        )
        grad_b[-1] = delta_N

        # Backprop through hidden layers k = d-1, d-2, ..., 1
        for k in range(self.depth - 2, -1, -1):
            z_k = z_list[k]
            sigma, sigma_p, sigma_pp, sigma_ppp = sigma_list[k]
            z_k_t = z_t_list[k]
            z_k_a = z_a_list[k]
            z_k_ta = z_ta_list[k]
            h_prev = h_list[k]
            h_prev_t = h_t_list[k]
            h_prev_a = h_a_list[k]
            h_prev_ta = h_ta_list[k]
            W_k = self._W[k]

            # Adjoints at this layer (from general formulas)
            delta_z_ta = delta_h_ta * sigma_p
            delta_z_t = delta_h_t * sigma_p + delta_h_ta * sigma_pp * z_k_a
            delta_z_a = delta_h_a * sigma_p + delta_h_ta * sigma_pp * z_k_t
            delta_z = (
                delta_h * sigma_p
                + delta_h_t * (sigma_pp * z_k_t)
                + delta_h_a * (sigma_pp * z_k_a)
                + delta_h_ta * (sigma_ppp * z_k_t * z_k_a + sigma_pp * z_k_ta)
            )

            # Weight gradient: dL/dW_k = delta_z_k h_{k-1}^T + ...
            grad_W[k] = (
                np.outer(delta_z, h_prev)
                + np.outer(delta_z_t, h_prev_t)
                + np.outer(delta_z_a, h_prev_a)
                + np.outer(delta_z_ta, h_prev_ta)
            )
            grad_b[k] = delta_z

            # Pass to layer below
            delta_h = W_k.T @ delta_z
            delta_h_t = W_k.T @ delta_z_t
            delta_h_a = W_k.T @ delta_z_a
            delta_h_ta = W_k.T @ delta_z_ta

        # dL_data/da = w_data * err * u_a, with u_a = t * N_a (no a in collocation backward)
        dL_da = 0.0  # set by caller when data point
        return grad_W, grad_b, dL_da

    def compute_losses_and_gradients(
        self,
        t_colloc: np.ndarray,
        a_colloc: np.ndarray,
        t_data: np.ndarray | None = None,
        a_data: np.ndarray | None = None,
        u_data: np.ndarray | None = None,
        w_res: float = 1.0,
        w_grad: float = 1.0,
        w_data: float = 1.0,
    ) -> tuple[dict[str, float], dict[str, np.ndarray | float]]:
        """Compute L_res, L_grad, L_data and gradients w.r.t. all parameters."""
        L_res = 0.0
        L_grad = 0.0
        L_data = 0.0

        grad_W_acc = [np.zeros_like(w) for w in self._W]
        grad_b_acc = [np.zeros_like(b) for b in self._b]
        dL_da_acc = 0.0

        # Collocation points
        for i in range(len(t_colloc)):
            t, a = float(t_colloc[i]), float(a_colloc[i])
            R, R_a = self.residuals(t, a)
            L_res += 0.5 * R * R
            L_grad += 0.5 * R_a * R_a

            gW, gb, _ = self._backward_one_point(
                t, a, R, R_a, delta_N_data=0.0,
                w_res=w_res, w_grad=w_grad, w_data=0.0,
            )
            for j in range(len(grad_W_acc)):
                grad_W_acc[j] += gW[j]
                grad_b_acc[j] += gb[j]

        # Data points
        if t_data is not None and a_data is not None and u_data is not None:
            for i in range(len(t_data)):
                t, a, u_target = float(t_data[i]), float(a_data[i]), float(u_data[i])
                _, _, _, _, u, _, u_a, _, _, _ = self.forward(t, a)
                err = u - u_target
                L_data += 0.5 * err * err
                # dL_data/dN = err * du/dN = err * t (same for exponential and logistic)
                delta_N_data = w_data * err * t
                R, R_a = 0.0, 0.0
                gW, gb, _ = self._backward_one_point(
                    t, a, R, R_a, delta_N_data=delta_N_data,
                    w_res=0.0, w_grad=0.0, w_data=w_data,
                )
                for j in range(len(grad_W_acc)):
                    grad_W_acc[j] += gW[j]
                    grad_b_acc[j] += gb[j]
                dL_da_acc += w_data * err * u_a

        losses = {"L_res": L_res, "L_grad": L_grad, "L_data": L_data}
        grads = {}
        for k in range(self.depth):
            grads[f"W{k+1}"] = grad_W_acc[k]
            grads[f"b{k+1}"] = grad_b_acc[k]
        if t_data is not None and a_data is not None and u_data is not None:
            grads["a"] = dL_da_acc
        return losses, grads

    def eval_u(self, t: np.ndarray | float, a: np.ndarray | float) -> np.ndarray | float:
        """Evaluate u(t,a) on scalar or arrays. Broadcasts if needed."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        a = np.atleast_1d(np.asarray(a, dtype=np.float64))
        t, a = np.broadcast_arrays(t, a)
        out = np.empty(t.size)
        for i in range(t.size):
            _, _, _, _, u, _, _, _, _, _ = self.forward(float(t.flat[i]), float(a.flat[i]))
            out.flat[i] = u
        return out[0] if out.size == 1 else out.reshape(t.shape)


# -----------------------------------------------------------------------------
# PINN (Physics-Informed Neural Network)
# -----------------------------------------------------------------------------
# PINN differs from BILO:
# - u(t; W): a is NOT input to the network. Implemented by W1[:,1]=0, never updated.
# - Loss: L_res + L_data only (no residual gradient loss)
# - a is a trainable parameter. R = u_t - a*u, so a participates in L_res only.
# - dL_res/da = -sum(R*u) over collocation; dL_data/da = 0.
# -----------------------------------------------------------------------------


class PINNModel(BILOModel):
    """PINN model: u(t; W) with a as trainable parameter. Reuses BILO architecture with W1[:,1]=0."""

    def __init__(
        self,
        n_hidden: int,
        depth: int = 2,
        ode_type: ODE_TYPE = "exponential",
        u0: float = U0_LOGISTIC,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(n_hidden=n_hidden, depth=depth, ode_type=ode_type, u0=u0, rng=rng)
        # Zero weights for input a so a has no influence on the network
        self._W[0][:, 1] = 0.0

    def compute_losses_and_gradients_pinn(
        self,
        t_colloc: np.ndarray,
        a_colloc: np.ndarray,
        t_data: np.ndarray | None = None,
        u_data: np.ndarray | None = None,
        w_res: float = 1.0,
        w_data: float = 1.0,
    ) -> tuple[dict[str, float], dict[str, np.ndarray | float]]:
        """Compute L_res + L_data and gradients w.r.t. W and a.

        L_res = 0.5 * MSE(R) over collocation; a participates (dL_res/da: exponential -R*u, logistic -R*u*(1-u)).
        L_data = 0.5 * MSE(u - u_data) over data; a does not participate.
        W1[:,1] gradients are zeroed (never update a-input weights).
        """
        L_res = 0.0
        L_data = 0.0
        grad_W_acc = [np.zeros_like(w) for w in self._W]
        grad_b_acc = [np.zeros_like(b) for b in self._b]
        dL_res_da = 0.0

        # Collocation points: L_res only (no L_grad), add dL_res/da
        for i in range(len(t_colloc)):
            t, a = float(t_colloc[i]), float(a_colloc[i])
            _, _, _, _, u, u_t, _, _, _, _ = self.forward(t, a)
            R, _ = self.residuals(t, a)
            L_res += 0.5 * R * R
            if self.ode_type == "exponential":
                dL_res_da += -R * u  # dR/da = -u
            else:
                dL_res_da += -R * u * (1.0 - u)  # dR/da = -u*(1-u) when u_a=0

            gW, gb, _ = self._backward_one_point(
                t, a, R, R_a=0.0, delta_N_data=0.0,
                w_res=w_res, w_grad=0.0, w_data=0.0,
            )
            for j in range(len(grad_W_acc)):
                grad_W_acc[j] += gW[j]
                grad_b_acc[j] += gb[j]

        # Data points: L_data only; a does not participate
        if t_data is not None and u_data is not None:
            for i in range(len(t_data)):
                t, u_target = float(t_data[i]), float(u_data[i])
                a_dummy = 0.0  # any value; u does not depend on a
                _, _, _, _, u, _, _, _, _, _ = self.forward(t, a_dummy)
                err = u - u_target
                L_data += 0.5 * err * err
                delta_N_data = w_data * err * t
                gW, gb, _ = self._backward_one_point(
                    t, a_dummy, R=0.0, R_a=0.0, delta_N_data=delta_N_data,
                    w_res=0.0, w_grad=0.0, w_data=w_data,
                )
                for j in range(len(grad_W_acc)):
                    grad_W_acc[j] += gW[j]
                    grad_b_acc[j] += gb[j]

        # Never update W1[:,1]
        grad_W_acc[0][:, 1] = 0.0

        losses = {"L_res": L_res, "L_grad": 0.0, "L_data": L_data}
        grads = {}
        for k in range(self.depth):
            grads[f"W{k+1}"] = grad_W_acc[k]
            grads[f"b{k+1}"] = grad_b_acc[k]
        grads["a"] = dL_res_da
        return losses, grads

    def eval_u(self, t: np.ndarray | float, a: np.ndarray | float = 0.0) -> np.ndarray | float:
        """Evaluate u(t; W). a is ignored (network does not use it); kept for API compatibility."""
        return super().eval_u(t, a)


try:
    import torch
    import torch.nn as nn

    class BILOModelTorch(nn.Module):
        """PyTorch d-layer BILO model for gradient verification.

        Implements the same forward and forward-kinematics recurrence as BILOModel
        (N, N_t, N_a, N_ta from explicit recurrence) so autograd matches manual backprop.
        Supports ode_type "exponential" and "logistic".
        """

        def __init__(
            self,
            n_hidden: int,
            depth: int = 2,
            ode_type: ODE_TYPE = "exponential",
            u0: float = U0_LOGISTIC,
            seed: int = 42,
        ):
            super().__init__()
            torch.manual_seed(seed)
            self.n_hidden = n_hidden
            self.depth = depth
            self.ode_type = ode_type
            self.u0 = u0
            dtype = torch.float64
            n, d = n_hidden, depth
            self._W = nn.ParameterList([
                nn.Parameter(torch.randn(n, 2, dtype=dtype) * 0.5),
            ])
            self._b = nn.ParameterList([
                nn.Parameter(torch.zeros(n, dtype=dtype)),
            ])
            for _ in range(d - 2):
                self._W.append(nn.Parameter(torch.randn(n, n, dtype=dtype) * 0.5))
                self._b.append(nn.Parameter(torch.zeros(n, dtype=dtype)))
            self._W.append(nn.Parameter(torch.randn(n, dtype=dtype) * 0.5))
            self._b.append(nn.Parameter(torch.tensor(0.0, dtype=dtype)))

        def _forward_and_kinematics(
            self, t: torch.Tensor, a: torch.Tensor
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            list[torch.Tensor], list[torch.Tensor],
        ]:
            """Same recurrence as NumPy: h_0=[t,a], z_k = W_k h_{k-1}+b_k, h_k=tanh(z_k), and kinematics."""
            x = torch.stack([t, a])
            h_0_t = torch.tensor([1.0, 0.0], dtype=x.dtype, device=x.device)
            h_0_a = torch.tensor([0.0, 1.0], dtype=x.dtype, device=x.device)
            h_0_ta = torch.tensor([0.0, 0.0], dtype=x.dtype, device=x.device)

            h_list = [x]
            h_t_list = [h_0_t]
            h_a_list = [h_0_a]
            h_ta_list = [h_0_ta]

            for k in range(self.depth - 1):
                h = h_list[-1]
                h_t = h_t_list[-1]
                h_a = h_a_list[-1]
                h_ta = h_ta_list[-1]
                z = torch.mv(self._W[k], h) + self._b[k]
                sigma = torch.tanh(z)
                sigma_p = 1.0 - sigma * sigma
                sigma_pp = -2.0 * sigma * sigma_p

                z_t = torch.mv(self._W[k], h_t)
                z_a = torch.mv(self._W[k], h_a)
                z_ta = torch.mv(self._W[k], h_ta)

                h_t_next = sigma_p * z_t
                h_a_next = sigma_p * z_a
                h_ta_next = sigma_pp * z_t * z_a + sigma_p * z_ta

                h_list.append(sigma)
                h_t_list.append(h_t_next)
                h_a_list.append(h_a_next)
                h_ta_list.append(h_ta_next)

            h_last = h_list[-1]
            h_last_t = h_t_list[-1]
            h_last_a = h_a_list[-1]
            h_last_ta = h_ta_list[-1]
            N = (self._W[-1] * h_last).sum() + self._b[-1]
            N_t = (self._W[-1] * h_last_t).sum()
            N_a = (self._W[-1] * h_last_a).sum()
            N_ta = (self._W[-1] * h_last_ta).sum()

            return N, N_t, N_a, N_ta, h_list, h_t_list, h_a_list, h_ta_list

        def forward(self, t: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, ...]:
            """Returns N, u, u_t, u_a, u_ta, R, R_a."""
            N, N_t, N_a, N_ta, _, _, _, _ = self._forward_and_kinematics(t, a)
            if self.ode_type == "exponential":
                u = 1.0 + t * N
            else:
                u = self.u0 + t * N
            u_t = N + t * N_t
            u_a = t * N_a
            u_ta = N_a + t * N_ta
            if self.ode_type == "exponential":
                R = u_t - a * u
                R_a = u_ta - (u + a * u_a)
            else:
                R = u_t - a * u * (1.0 - u)
                R_a = u_ta - u * (1.0 - u) - a * u_a * (1.0 - 2.0 * u)
            return N, u, u_t, u_a, u_ta, R, R_a

        def forward_N_only(self, t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            """Forward that returns only N (for grad checks w.r.t. inputs)."""
            N, _, _, _, _, _, _, _ = self._forward_and_kinematics(t, a)
            return N

        def eval_u(self, t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            """Evaluate u(t,a)."""
            N = self.forward_N_only(t, a)
            if self.ode_type == "exponential":
                return 1.0 + t * N
            return self.u0 + t * N

    class PINNModelTorch(BILOModelTorch):
        """PINN model for PyTorch gradient verification.

        Reuses BILOModelTorch with W1[:,1]=0 (fixed, non-trainable) and a as nn.Parameter.
        Forward uses self.a for residual. Network output u does not depend on a.
        """

        def __init__(
            self,
            n_hidden: int,
            depth: int = 2,
            ode_type: ODE_TYPE = "exponential",
            u0: float = U0_LOGISTIC,
            seed: int = 42,
            a_init: float = 1.0,
        ):
            super().__init__(n_hidden=n_hidden, depth=depth, ode_type=ode_type, u0=u0, seed=seed)
            self.a = nn.Parameter(torch.tensor(a_init, dtype=torch.float64))
            # W1[:,1]=0 so a has no influence on network output (non-trainable)
            with torch.no_grad():
                self._W[0][:, 1] = 0.0

        def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Returns N, u, u_t, R. Uses self.a for residual (network ignores a via W1[:,1]=0)."""
            N, u, u_t, u_a, u_ta, R, R_a = super().forward(t, self.a)
            return N, u, u_t, R

        def eval_u(self, t: torch.Tensor) -> torch.Tensor:
            """Evaluate u(t; W). a is ignored; uses self.a for API consistency."""
            return super().eval_u(t, self.a)

except ImportError:
    BILOModelTorch = None  # type: ignore
    PINNModelTorch = None  # type: ignore
