"""Main entry point: train BILO and visualize.

PDE: u' = a*u
Trial solution: u(t,a;W) = 1 + t*N(t,a;W)

Two-stage training:
  - Stage 1 (pretrain): fix a=1, t in [0,1], update W
  - Stage 2 (finetune): data from ground truth a=2, update W and a
"""

import argparse
from pathlib import Path

import numpy as np

from model import BILOModel
from train import train, train_finetune
from visualize import plot_loss_history, plot_solution_multi_a, plot_solution_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BILO training: u' = a*u")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-every", type=int, default=200, help="Log every N steps")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: bilo/output)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # t in [0, 1]
    t_min, t_max = 0.0, 1.0
    t_colloc = np.linspace(t_min, t_max, args.n_colloc)

    # Stage 1: Pretraining with a=1
    a_pretrain = 1.0
    a_colloc = np.full_like(t_colloc, a_pretrain)

    model = BILOModel(n_hidden=args.n_hidden, depth=args.depth)
    print("=" * 60)
    print("Stage 1: Pretraining (fix a=1, t in [0,1])")
    print("=" * 60)
    history_pretrain = train(
        model,
        t_colloc,
        a_colloc,
        n_iters=args.n_pretrain,
        lr=args.lr,
        w_res=args.w_res,
        w_grad=args.w_grad,
        w_data=0.0,
        log_every=args.log_every,
    )

    # Plot u(t,a) for a = 0.5, 1, 1.5 after pretraining
    if not args.no_plot:
        try:
            plot_solution_multi_a(
                model,
                a_values=[0.8, 1.0, 1.2],
                t_min=0.0,
                t_max=1.0,
                save_path=out_dir / "bilo_after_pretrain.png",
                show=False,
            )
        except ImportError:
            print("matplotlib not installed, skipping pretrain plot")

    # Stage 2: Fine-tuning with data from a=2
    a_gt = 2.0
    t_data = np.linspace(t_min, t_max, args.n_data)
    u_data = np.exp(a_gt * t_data)  # ground truth u = exp(2*t)

    print("\n" + "=" * 60)
    print(f"Stage 2: Fine-tuning (data from a={a_gt}, update W and a)")
    print("=" * 60)
    history_finetune, a_learned = train_finetune(
        model,
        t_colloc,
        a_learned=a_pretrain,
        t_data=t_data,
        u_data=u_data,
        n_iters=args.n_finetune,
        lr=args.lr,
        lr_a=args.lr_a,
        w_res=args.w_res,
        w_grad=args.w_grad,
        w_data=args.w_data,
        log_every=args.log_every,
    )

    # Combined history for loss plot
    history = history_pretrain + [
        {**h, "step": h["step"] + args.n_pretrain}
        for h in history_finetune
    ]

    # Ensure L_data in pretrain history (for plot)
    for h in history_pretrain:
        h.setdefault("L_data", 0.0)

    if not args.no_plot:
        try:
            plot_loss_history(
                history,
                save_path=out_dir / "bilo_loss_history.png",
                show=False,
            )

            plot_solution_multi_a(
                model,
                a_values=[a_learned-0.2, a_learned, a_learned+0.2],
                t_min=0.0,
                t_max=1.0,
                save_path=out_dir / "bilo_after_finetune.png",
                show=False,
            )
            plot_solution_2d(
                model,
                t_min=0.0,
                t_max=2.0,
                a_min=0.5,
                a_max=2.5,
                save_path=out_dir / "bilo_solution_2d.png",
                show=False,
            )
        except ImportError:
            print("matplotlib not installed, skipping plots")

    print(f"\nLearned a = {a_learned:.4f} (ground truth a = {a_gt})")
    print(f"Final L_res={history[-1]['L_res']:.6f}  L_grad={history[-1]['L_grad']:.6f}  L_data={history[-1]['L_data']:.6f}")


if __name__ == "__main__":
    main()
