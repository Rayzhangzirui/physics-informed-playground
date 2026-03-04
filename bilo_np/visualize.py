"""Visualization for BILO solution u(t,a;W)."""

from pathlib import Path

import numpy as np

from model import BILOModel


def plot_solution_multi_a(
    model: BILOModel,
    a_values: list[float],
    t_min: float = 0.0,
    t_max: float = 2.0,
    n_pts: int = 201,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot u(t,a) vs t for multiple a values."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    t_plot = np.linspace(t_min, t_max, n_pts)
    fig, ax = plt.subplots(figsize=(8, 5))
    for a in a_values:
        a_plot = np.full_like(t_plot, a)
        u_pred = model.eval_u(t_plot, a_plot)
        u_exact = np.exp(a * t_plot)
        ax.plot(t_plot, u_pred, "-", label=f"u(t,{a:.2f};W) (BILO)")
        ax.plot(t_plot, u_exact, "--", alpha=0.6, label=f"exp({a:.2f}*t) (exact)")
    ax.set_xlabel("t")
    ax.set_ylabel("u")
    ax.set_title("BILO solution vs exact: PDE u' = a*u")
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
) -> None:
    """Plot solution after fine-tuning: data scatter, BILO u(t,a_learned), and u_a = exp(a_learned*t).

    u_a is the ODE solution u' = a*u with u(0)=1 using the inferred a from BILO.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    t_plot = np.linspace(t_min, t_max, n_pts)
    a_plot = np.full_like(t_plot, a_learned)
    u_bilo = model.eval_u(t_plot, a_plot)
    u_a = np.exp(a_learned * t_plot)  # ODE solution with inferred a

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
    t_max: float = 2.0,
    a_min: float = 0.5,
    a_max: float = 2.0,
    n_t: int = 101,
    n_a: int = 51,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot u(t,a) as 2D heatmap.

    Args:
        model: trained BILOModel
        t_min, t_max: t range
        a_min, a_max: a range
        n_t, n_a: grid resolution
        save_path: path to save figure
        show: whether to plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    t_plot = np.linspace(t_min, t_max, n_t)
    a_plot = np.linspace(a_min, a_max, n_a)
    T, A = np.meshgrid(t_plot, a_plot)
    U_pred = model.eval_u(T, A)
    U_exact = np.exp(A * T)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    im0 = axes[0].pcolormesh(T, A, U_pred, shading="auto", cmap="viridis")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("a")
    axes[0].set_title("u(t,a;W) (BILO)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(T, A, U_exact, shading="auto", cmap="viridis")
    axes[1].set_xlabel("t")
    axes[1].set_title("exp(a*t) (exact)")
    plt.colorbar(im1, ax=axes[1])

    fig.suptitle("BILO solution vs exact: PDE u' = a*u")
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
