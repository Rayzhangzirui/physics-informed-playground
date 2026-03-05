"""Main entry point: train BILO and visualize.

PDEs:
  - exponential: u' = a*u, u(0)=1. Trial u = 1 + t*N(t,a;W).
  - logistic: u' = a*u*(1-u), u(0)=u0=0.1. Trial u = u0 + t*N(t,a;W).

Two-stage training:
  - Stage 1 (pretrain): fix a, t in [0,1], update W
  - Stage 2 (finetune): data from ground truth a, update W and a
"""

import argparse
from pathlib import Path

import numpy as np

from model import BILOModel, PINNModel, logistic_solution
from train import train, train_finetune, train_pinn, train_pinn_finetune
from visualize import plot_loss_history, plot_solution_multi_a, plot_solution_2d, plot_solution_after_finetune


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BILO/PINN training: u'=a*u or u'=a*u*(1-u)")
    parser.add_argument("--pde", type=str, choices=["exponential", "logistic"], default="exponential",
                        help="PDE: exponential (u'=au) or logistic (u'=au(1-u), u0=0.1)")
    parser.add_argument("--model", "-m", type=str, choices=["bilo", "pinn"], default="bilo",
                        help="Model: bilo (BILO, u(t,a;W)) or pinn (PINN, u(t;W) with a as param)")
    parser.add_argument("--n-hidden",'-n', type=int, default=8, help="Hidden layer size")
    parser.add_argument("--depth",'-d', type=int, default=2, help="Network depth (2 = one hidden layer)")
    parser.add_argument("--n-colloc", type=int, default=21, help="Number of collocation points")
    parser.add_argument("--n-pretrain",'-p', type=int, default=1000, help="Pretrain iterations (fix a=1)")
    parser.add_argument("--n-finetune",'-f', type=int, default=1000, help="Finetune iterations (update W and a)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for W")
    parser.add_argument("--lr-a", type=float, default=0.001, help="Learning rate for a (finetune)")
    parser.add_argument("--w-res", type=float, default=1.0, help="Weight for L_res")
    parser.add_argument("--w-grad", type=float, default=0.1, help="Weight for L_grad")
    parser.add_argument("--w-data", type=float, default=1.0, help="Weight for L_data (finetune)")
    parser.add_argument("--n-data", type=int, default=21, help="Number of data points for finetune")
    parser.add_argument("--std", type=float, default=0.0, help="Std of normal noise added to u_data (finetune)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-every", type=int, default=200, help="Log every N steps")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: bilo/output)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--u0", type=float, default=None,
                        help="Initial condition u(0). Default: 1.0 (exponential) or 0.1 (logistic)")
    parser.add_argument("--ainit", type=float, default=1.0, help="Initial a for pretraining")
    parser.add_argument("--agt", type=float, default=2.0, help="Ground truth a (for data generation)")
    parser.add_argument("--optimizer",'-o', type=str, choices=["gd", "adam"], default="gd",
                        help="Optimizer: gd (gradient descent) or adam")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    ode_type = args.pde
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.model}_{ode_type}"

    t_min, t_max = 0.0, 1.0
    t_colloc = np.linspace(t_min, t_max, args.n_colloc)
    a_gt = args.agt
    a_init = args.ainit
    u0 = args.u0
    if u0 is None:
        u0 = 1.0 if ode_type == "exponential" else 0.1
    t_data = np.linspace(t_min, t_max, args.n_data)
    if ode_type == "exponential":
        u_data = u0 * np.exp(a_gt * t_data)
    else:
        u_data = logistic_solution(t_data, a_gt, u0=u0)
    if args.std > 0:
        u_data = u_data + args.std * np.random.randn(args.n_data)

    a_colloc = np.full_like(t_colloc, a_init)

    # Create model
    if args.model == "bilo":
        model = BILOModel(n_hidden=args.n_hidden, depth=args.depth, ode_type=ode_type, u0=u0)
    else:
        model = PINNModel(n_hidden=args.n_hidden, depth=args.depth, ode_type=ode_type, u0=u0)

    # Stage 1: Pretrain (fix a, update W — solve PDE forward)
    print("=" * 60)
    print(f"Stage 1: Pretraining (fix a={a_init}, u0={u0}, L_res only)")
    if args.model == "bilo":
        print("  BILO: L_res + L_grad")
    else:
        print("  PINN: L_res")
    print("=" * 60)
    if args.model == "bilo":
        history_pretrain = train(
            model, t_colloc, a_colloc,
            n_iters=args.n_pretrain, lr=args.lr,
            w_res=args.w_res, w_grad=args.w_grad, w_data=0.0, log_every=args.log_every,
            optimizer=args.optimizer,
        )
    else:
        history_pretrain, _ = train_pinn(
            model, t_colloc, a_colloc,
            n_iters=args.n_pretrain, lr=args.lr, w_res=args.w_res,
            update_a=False, log_every=args.log_every,
        )

    if not args.no_plot:
        try:
            plot_solution_multi_a(
                model, a_values=[0.8, 1.0, 1.2], t_min=0.0, t_max=1.0,
                save_path=out_dir / f"{prefix}_after_pretrain.png", show=False,
                ode_type=ode_type,
            )
        except ImportError:
            pass

    # Stage 2: Finetune (update W and a with data)
    print("\n" + "=" * 60)
    print(f"Stage 2: Fine-tuning (data from a={a_gt}, update W and a)")
    print("=" * 60)
    if args.model == "bilo":
        history_finetune, a_learned = train_finetune(
            model, t_colloc, a_learned=a_init,
            t_data=t_data, u_data=u_data,
            n_iters=args.n_finetune, lr=args.lr, lr_a=args.lr_a,
            w_res=args.w_res, w_grad=args.w_grad, w_data=args.w_data, log_every=args.log_every,
            optimizer=args.optimizer,
        )
    else:
        history_finetune, a_learned = train_pinn_finetune(
            model, t_colloc, a_learned=a_init,
            t_data=t_data, u_data=u_data,
            n_iters=args.n_finetune, lr=args.lr, lr_a=args.lr_a,
            w_res=args.w_res, w_data=args.w_data, log_every=args.log_every,
        )

    history = history_pretrain + [
        {**h, "step": h["step"] + args.n_pretrain} for h in history_finetune
    ]
    for h in history_pretrain:
        h.setdefault("L_data", 0.0)

    if not args.no_plot:
        try:
            plot_loss_history(
                history,
                save_path=out_dir / f"{prefix}_loss_history.png",
                show=False,
            )
            plot_solution_after_finetune(
                model,
                a_learned=a_learned,
                t_data=t_data,
                u_data=u_data,
                t_min=0.0,
                t_max=1.0,
                save_path=out_dir / f"{prefix}_solution_after_finetune.png",
                show=False,
                ode_type=ode_type,
            )
            if args.model == "bilo":
                plot_solution_multi_a(
                    model,
                    a_values=[a_learned - 0.2, a_learned, a_learned + 0.2],
                    t_min=0.0, t_max=1.0,
                    save_path=out_dir / f"{prefix}_after_finetune.png",
                    show=False,
                    ode_type=ode_type,
                )
                plot_solution_2d(
                    model,
                    t_min=0.0, t_max=1.2,
                    a_init=a_init, a_gt=a_gt,
                    save_path=out_dir / f"{prefix}_solution_2d.png",
                    show=False,
                    ode_type=ode_type,
                )
        except ImportError:
            print("matplotlib not installed, skipping plots")

    print(f"\nLearned a = {a_learned:.4f} (ground truth a = {a_gt})")
    print(f"Final L_res={history[-1]['L_res']:.6f}  L_grad={history[-1]['L_grad']:.6f}  L_data={history[-1]['L_data']:.6f}")


if __name__ == "__main__":
    main()
