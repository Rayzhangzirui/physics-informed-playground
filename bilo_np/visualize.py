"""Visualization for BILO solution u(t,a;W)."""

from pathlib import Path

import numpy as np

from model import BILOModel, logistic_solution, U0_LOGISTIC


def _exact_u(
    t: np.ndarray,
    a: np.ndarray | float,
    ode_type: str,
    u0: float | None = None,
) -> np.ndarray:
    """Exact solution: exponential u=u0*exp(a*t), logistic u=u0*e^{at}/(1+u0*(e^{at}-1)). Supports arrays."""
    if ode_type == "exponential":
        u0_val = u0 if u0 is not None else 1.0
        return u0_val * np.exp(a * t)
    u0_val = u0 if u0 is not None else U0_LOGISTIC
    eat = np.exp(a * t)
    return (u0_val * eat) / (1.0 + u0_val * (eat - 1.0))


def plot_solution_multi_a(
    model: BILOModel,
    a_values: list[float],
    t_min: float = 0.0,
    t_max: float = 2.0,
    n_pts: int = 201,
    save_path: str | Path | None = None,
    show: bool = True,
    ode_type: str | None = None,
) -> None:
    """Plot u(t,a) vs t for multiple a values."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    ode = ode_type or getattr(model, "ode_type", "exponential")
    t_plot = np.linspace(t_min, t_max, n_pts)
    fig, ax = plt.subplots(figsize=(8, 5))
    for a in a_values:
        a_plot = np.full_like(t_plot, a)
        u_pred = model.eval_u(t_plot, a_plot)
        u_exact = _exact_u(t_plot, a, ode, u0=getattr(model, "u0", None))
        ax.plot(t_plot, u_pred, "-", label=f"u(t,{a:.2f};W) (BILO)")
        ax.plot(t_plot, u_exact, "--", alpha=0.6, label=f"exact a={a:.2f}")
    ax.set_xlabel("t")
    ax.set_ylabel("u")
    title = "u'=a*u*(1-u)" if ode == "logistic" else "u'=a*u"
    ax.set_title(f"BILO solution vs exact: PDE {title}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t_min, t_max)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=120)
        print(f"Saved to {save_path}")
    if show:
        plt.show()


def plot_solution(
    model: BILOModel,
    a: float,
    t_min: float = 0.0,
    t_max: float = 2.0,
    n_pts: int = 201,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot u(t,a) vs t for fixed a, and the exact solution exp(a*t).

    Args:
        model: trained BILOModel
        a: fixed parameter value
        t_min, t_max: t range
        n_pts: number of points
        save_path: path to save figure
        show: whether to plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    t_plot = np.linspace(t_min, t_max, n_pts)
    a_plot = np.full_like(t_plot, a)
    u_pred = model.eval_u(t_plot, a_plot)
    u_exact = np.exp(a * t_plot)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_plot, u_pred, "b-", label=f"u(t,{a};W) (BILO)", linewidth=2)
    ax.plot(t_plot, u_exact, "k--", label=f"exp({a}*t) (exact)", linewidth=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("u")
    ax.set_title(f"BILO solution vs exact: PDE u' = {a}*u")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t_min, t_max)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=120)
        print(f"Saved to {save_path}")
    if show:
        plt.show()


def plot_solution_after_finetune(
    model: BILOModel,
    a_learned: float,
    t_data: np.ndarray,
    u_data: np.ndarray,
    t_min: float = 0.0,
    t_max: float = 1.0,
    n_pts: int = 201,
    save_path: str | Path | None = None,
    show: bool = True,
    ode_type: str | None = None,
) -> None:
    """Plot solution after fine-tuning: data scatter, BILO u(t,a_learned), and exact u with inferred a."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    ode = ode_type or getattr(model, "ode_type", "exponential")
    t_plot = np.linspace(t_min, t_max, n_pts)
    a_plot = np.full_like(t_plot, a_learned)
    u_bilo = model.eval_u(t_plot, a_plot)
    u_a = _exact_u(t_plot, a_learned, ode, u0=getattr(model, "u0", None))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(t_data, u_data, c="k", s=30, alpha=0.8, label="data", zorder=3)
    ax.plot(t_plot, u_bilo, "-", label="u(t,a;W) (BILO)", linewidth=2)
    ax.plot(t_plot, u_a, "--", alpha=0.8, label="u_a = exp(a*t)", linewidth=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("u")
    ax.set_title(f"After fine-tuning: inferred a = {a_learned:.4f}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t_min, t_max)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=120)
        print(f"Saved to {save_path}")
    if show:
        plt.show()


def plot_solution_2d(
    model: BILOModel,
    t_min: float = 0.0,
    t_max: float = 1.2,
    a_min: float | None = None,
    a_max: float | None = None,
    a_init: float | None = None,
    a_gt: float | None = None,
    n_t: int = 101,
    n_a: int = 51,
    save_path: str | Path | None = None,
    show: bool = True,
    ode_type: str | None = None,
) -> None:
    """Plot u(t,a) as 2D heatmap. If a_init and a_gt are provided, a_min=0.8*a_init, a_max=1.2*a_gt."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    ode = ode_type or getattr(model, "ode_type", "exponential")
    if a_min is None and a_init is not None:
        a_min = 0.8 * a_init
    if a_max is None and a_gt is not None:
        a_max = 1.2 * a_gt
    if a_min is None:
        a_min = 0.5
    if a_max is None:
        a_max = 2.0
    t_plot = np.linspace(t_min, t_max, n_t)
    a_plot = np.linspace(a_min, a_max, n_a)
    T, A = np.meshgrid(t_plot, a_plot)
    U_pred = model.eval_u(T, A)
    U_exact = _exact_u(T, A, ode, u0=getattr(model, "u0", None))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    im0 = axes[0].pcolormesh(T, A, U_pred, shading="auto", cmap="viridis")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("a")
    axes[0].set_title("u(t,a;W) (BILO)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(T, A, U_exact, shading="auto", cmap="viridis")
    axes[1].set_xlabel("t")
    axes[1].set_title("exact")
    plt.colorbar(im1, ax=axes[1])

    title = "u'=a*u*(1-u)" if ode == "logistic" else "u'=a*u"
    fig.suptitle(f"BILO solution vs exact: PDE {title}")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=120)
        print(f"Saved to {save_path}")
    if show:
        plt.show()


def plot_loss_history(
    history: list[dict],
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot L_res, L_grad, L_data, L_total over training steps."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    steps = [h["step"] for h in history]
    L_res = [h["L_res"] for h in history]
    L_grad = [h["L_grad"] for h in history]
    L_data = [h["L_data"] for h in history]
    L_total = [h["L_total"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, L_res, label="L_res", alpha=0.8)
    ax.plot(steps, L_grad, label="L_grad", alpha=0.8)
    ax.plot(steps, L_data, label="L_data", alpha=0.8)
    ax.plot(steps, L_total, label="L_total", color="k", linewidth=2)

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("BILO training losses")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=120)
        print(f"Saved to {save_path}")
    if show:
        plt.show()
