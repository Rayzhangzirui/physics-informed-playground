"""BILO model: manual implementation with NumPy and PyTorch.

Architecture:
- Input: [t, a]
- Hidden: z1 = W1t*t + W1a*a + b1, sigma = tanh(z1)
- Output: N = W2 @ sigma + b2
- Solution: u = 1 + t*N

For tanh: sigma' = 1 - sigma^2, sigma'' = -2*sigma*sigma', sigma''' = 2*sigma'*(3*sigma^2 - 1)
"""

import numpy as np


def _tanh_derivatives(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute sigma, sigma', sigma'', sigma''' for tanh activation."""
    sigma = np.tanh(z)
    sigma_p = 1.0 - sigma * sigma
    sigma_pp = -2.0 * sigma * sigma_p
    sigma_ppp = 2.0 * sigma_p * (3.0 * sigma * sigma - 1.0)
    return sigma, sigma_p, sigma_pp, sigma_ppp


class BILOModel:
    """BILO model with manual backprop in NumPy.

    Weights: W1t (n,), W1a (n,), b1 (n,), W2 (n,), b2 (scalar)
    """

    def __init__(self, n_hidden: int, rng: np.random.Generator | None = None):
        self.n_hidden = n_hidden
        self.rng = rng or np.random.default_rng(42)
        # Xavier-like init
        self.W1t = self.rng.uniform(-1, 1, size=n_hidden).astype(np.float64)
        self.W1a = self.rng.uniform(-1, 1, size=n_hidden).astype(np.float64)
        self.b1 = np.zeros(n_hidden, dtype=np.float64)
        self.W2 = self.rng.uniform(-1, 1, size=n_hidden).astype(np.float64)
        self.b2 = 0.0

    def parameters(self) -> dict[str, np.ndarray | float]:
        return {"W1t": self.W1t, "W1a": self.W1a, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(
        self, t: float, a: float
    ) -> tuple[
        float,
        float,
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Forward pass. Returns (N, N_t, N_a, N_ta, u, u_t, u_a, u_ta, z1, sigma, sigma_p, sigma_pp, sigma_ppp).

        Actually returns fewer - let me return what we need for losses and gradients.
        Returns: N, N_t, N_a, N_ta, u, u_t, u_a, u_ta, z1, sigma, sigma_p, sigma_pp, sigma_ppp
        """
        z1 = self.W1t * t + self.W1a * a + self.b1
        sigma, sigma_p, sigma_pp, sigma_ppp = _tanh_derivatives(z1)

        N = float(np.dot(self.W2, sigma)) + self.b2
        N_t = float(np.dot(self.W2, sigma_p * self.W1t))
        N_a = float(np.dot(self.W2, sigma_p * self.W1a))
        N_ta = float(np.dot(self.W2, sigma_pp * self.W1t * self.W1a))

        u = 1.0 + t * N
        u_t = N + t * N_t
        u_a = t * N_a
        u_ta = N_a + t * N_ta

        return N, N_t, N_a, N_ta, u, u_t, u_a, u_ta, z1, sigma, sigma_p, sigma_pp, sigma_ppp

    def residuals(self, t: float, a: float) -> tuple[float, float]:
        """Compute R and R_a at (t, a)."""
        _, _, _, _, u, u_t, u_a, u_ta, _, _, _, _, _ = self.forward(t, a)
        R = u_t - a * u
        R_a = u_ta - (u + a * u_a)
        return R, R_a

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
        """Compute L_res, L_grad, L_data and gradients w.r.t. all parameters.

        Collocation: (t_colloc[i], a_colloc[i]) for physics loss.
        Data: (t_data[i], a_data[i], u_data[i]) for data loss (optional).
        """
        L_res = 0.0
        L_grad = 0.0
        L_data = 0.0

        dL_dW1t = np.zeros_like(self.W1t)
        dL_dW1a = np.zeros_like(self.W1a)
        dL_db1 = np.zeros_like(self.b1)
        dL_dW2 = np.zeros_like(self.W2)
        dL_db2 = 0.0
        dL_da = 0.0

        n = self.n_hidden

        # Collocation points
        for i in range(len(t_colloc)):
            t, a = float(t_colloc[i]), float(a_colloc[i])
            N, N_t, N_a, N_ta, u, u_t, u_a, u_ta, z1, sigma, sigma_p, sigma_pp, sigma_ppp = (
                self.forward(t, a)
            )

            R = u_t - a * u
            R_a = u_ta - (u + a * u_a)

            L_res += 0.5 * R * R
            L_grad += 0.5 * R_a * R_a

            # ∂R/∂θ and ∂R_a/∂θ from blueprint
            _1_at = 1.0 - a * t
            _2_at = 2.0 - a * t

            # Output layer
            dR_db2 = _1_at
            dRa_db2 = -t

            dR_dW2 = sigma + t * (sigma_p * self.W1t) - a * t * sigma
            dRa_dW2 = (
                sigma_p * self.W1a
                + t * (sigma_pp * self.W1t * self.W1a)
                - t * sigma
                - a * t * (sigma_p * self.W1a)
            )

            # Hidden layer
            dR_db1 = self.W2 * ((_1_at) * sigma_p + t * self.W1t * sigma_pp)
            dRa_db1 = self.W2 * (
                (_1_at) * sigma_pp * self.W1a
                + t * self.W1t * self.W1a * sigma_ppp
                - t * sigma_p
            )

            dR_dW1t = t * self.W2 * ((_2_at) * sigma_p + t * self.W1t * sigma_pp)
            dRa_dW1t = t * self.W2 * (
                (_2_at) * sigma_pp * self.W1a
                + t * self.W1t * self.W1a * sigma_ppp
                - t * sigma_p
            )

            dR_dW1a = a * self.W2 * ((_1_at) * sigma_p + t * self.W1t * sigma_pp)
            dRa_dW1a = self.W2 * (
                (_1_at) * (sigma_p + a * sigma_pp * self.W1a)
                + t * self.W1t * (sigma_pp + a * sigma_ppp * self.W1a)
                - a * t * sigma_p
            )

            # Accumulate: dL/dθ = R * dR/dθ + R_a * dR_a/dθ
            dL_db2 += w_res * R * dR_db2 + w_grad * R_a * dRa_db2
            dL_dW2 += w_res * R * dR_dW2 + w_grad * R_a * dRa_dW2
            dL_db1 += w_res * R * dR_db1 + w_grad * R_a * dRa_db1
            dL_dW1t += w_res * R * dR_dW1t + w_grad * R_a * dRa_dW1t
            dL_dW1a += w_res * R * dR_dW1a + w_grad * R_a * dRa_dW1a

        # Data points
        if t_data is not None and a_data is not None and u_data is not None:
            for i in range(len(t_data)):
                t, a, u_target = float(t_data[i]), float(a_data[i]), float(u_data[i])
                N, N_t, N_a, N_ta, u, u_t, u_a, u_ta, z1, sigma, sigma_p, sigma_pp, sigma_ppp = (
                    self.forward(t, a)
                )
                err = u - u_target
                L_data += 0.5 * err * err

                # dL_data/dθ = err * du/dθ, where u = 1 + t*N
                # du/db2 = t, du/dW2 = t*sigma, du/db1 = t*W2*sigma_p, du/dW1t = t*W2*sigma_p*t = t^2*W2*sigma_p
                # Actually: du/dW1t: N depends on W1t via z1, so dN/dW1t = W2*(sigma_p*t), du/dW1t = t*dN/dW1t = t^2*W2*sigma_p
                # du/dW1a = t*W2*(sigma_p*a)
                # du/db1 = t*W2*sigma_p

                du_db2 = t
                du_dW2 = t * sigma
                du_db1 = t * self.W2 * sigma_p
                du_dW1t = t * t * self.W2 * sigma_p
                du_dW1a = t * a * self.W2 * sigma_p

                dL_db2 += w_data * err * du_db2
                dL_dW2 += w_data * err * du_dW2
                dL_db1 += w_data * err * du_db1
                dL_dW1t += w_data * err * du_dW1t
                dL_dW1a += w_data * err * du_dW1a
                # dL_data/da = err * u_a (u_a = t * N_a)
                dL_da += w_data * err * u_a

        losses = {"L_res": L_res, "L_grad": L_grad, "L_data": L_data}
        grads = {
            "W1t": dL_dW1t,
            "W1a": dL_dW1a,
            "b1": dL_db1,
            "W2": dL_dW2,
            "b2": dL_db2,
        }
        if t_data is not None and a_data is not None and u_data is not None:
            grads["a"] = dL_da
        return losses, grads

    def eval_u(self, t: np.ndarray | float, a: np.ndarray | float) -> np.ndarray | float:
        """Evaluate u(t,a) on scalar or arrays. Broadcasts if needed."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        a = np.atleast_1d(np.asarray(a, dtype=np.float64))
        t, a = np.broadcast_arrays(t, a)
        out = np.empty(t.size)
        for i in range(t.size):
            _, _, _, _, u, _, _, _, _, _, _, _, _ = self.forward(float(t.flat[i]), float(a.flat[i]))
            out.flat[i] = u
        return out[0] if out.size == 1 else out.reshape(t.shape)


try:
    import torch
    import torch.nn as nn

    class BILOModelTorch(nn.Module):
        """PyTorch version for gradient verification.

        Same architecture, uses autograd. We manually construct N, u, R, R_a
        from the formulas so we can compare gradients.
        """

        def __init__(self, n_hidden: int, seed: int = 42):
            super().__init__()
            torch.manual_seed(seed)
            self.n_hidden = n_hidden
            dtype = torch.float64
            self.W1t = nn.Parameter((torch.randn(n_hidden) * 0.5).to(dtype))
            self.W1a = nn.Parameter((torch.randn(n_hidden) * 0.5).to(dtype))
            self.b1 = nn.Parameter(torch.zeros(n_hidden, dtype=dtype))
            self.W2 = nn.Parameter((torch.randn(n_hidden) * 0.5).to(dtype))
            self.b2 = nn.Parameter(torch.tensor(0.0, dtype=dtype))

        def forward(self, t: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, ...]:
            """Returns N, u, u_t, u_a, u_ta, R, R_a, and intermediates for gradient checks."""
            z1 = self.W1t * t + self.W1a * a + self.b1
            sigma = torch.tanh(z1)
            sigma_p = 1.0 - sigma * sigma
            sigma_pp = -2.0 * sigma * sigma_p
            sigma_ppp = 2.0 * sigma_p * (3.0 * sigma * sigma - 1.0)

            N = (self.W2 * sigma).sum() + self.b2
            N_t = (self.W2 * sigma_p * self.W1t).sum()
            N_a = (self.W2 * sigma_p * self.W1a).sum()
            N_ta = (self.W2 * sigma_pp * self.W1t * self.W1a).sum()

            u = 1.0 + t * N
            u_t = N + t * N_t
            u_a = t * N_a
            u_ta = N_a + t * N_ta

            R = u_t - a * u
            R_a = u_ta - (u + a * u_a)

            return N, u, u_t, u_a, u_ta, R, R_a

        def eval_u(self, t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            """Evaluate u(t,a)."""
            z1 = self.W1t * t + self.W1a * a + self.b1
            sigma = torch.tanh(z1)
            N = (self.W2 * sigma).sum() + self.b2
            return 1.0 + t * N

except ImportError:
    BILOModelTorch = None  # type: ignore
