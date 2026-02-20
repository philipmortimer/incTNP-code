"""
Unified hyperparameter sweep for GP TNP models.
Sweeps over learning rate and max_nc for any specified config.
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

# Available config files
CONFIGS = {
    "plain": "gp_plain_tnp_rangesame.yaml",
    "plain_lr_sched": "gp_plain_tnp_lr_scheduler_rangesame.yaml",
    "causal": "gp_causal_tnp_rangesame.yaml",
    "causal_lr_sched": "gp_causal_tnp_lr_scheduler_rangesame.yaml",
    "batched": "gp_batched_causal_tnp_rangesame.yaml",
    "batched_lr_sched": "gp_batched_causal_tnp_lr_scheduler_rangesame.yaml",
}


def run_experiment(config_path: str, lr: float, max_nc: int, seed: int = 1):
    """Run a single experiment with specified hyperparameters."""

    # Build command
    cmd = [
        sys.executable,  # Use current Python interpreter
        "experiments/lightning_train.py",  # Your training script
        f"--config={config_path}",
        f"optimiser.lr={lr}",
        f"params.max_nc={max_nc}",
        f"misc.seed={seed}",
    ]

    print(f"\n{'=' * 80}")
    print(f"Running: max_nc={max_nc}, lr={lr}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    # Run the experiment
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"WARNING: Experiment failed with max_nc={max_nc}, lr={lr}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep for GP TNP models"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=list(CONFIGS.keys()),
        help=f"Config to use. Available: {', '.join(CONFIGS.keys())}",
    )
    parser.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=[1e-4],
        help="Learning rates to sweep (default: 1e-4 3e-4 5e-4 1e-3 2e-3)",
    )
    parser.add_argument(
        "--max_ncs",
        nargs="+",
        type=int,
        default=[64, 512],
        help="Max context sizes to sweep (default: 64 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )

    args = parser.parse_args()

    # Get config file path
    config_filename = CONFIGS[args.config]
    config_path = f"experiments/configs/synthetic1d/{config_filename}"

    # Check config exists
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Using config: {config_filename}")
    print(f"Full path: {config_path}")

    # Generate all combinations
    total_runs = len(args.lrs) * len(args.max_ncs)
    print(f"\nStarting sweep with {total_runs} experiments")
    print(f"Learning rates: {args.lrs}")
    print(f"Max context sizes: {args.max_ncs}")

    # Run sweep
    results = []
    for i, (lr, max_nc) in enumerate(itertools.product(args.lrs, args.max_ncs), 1):
        print(f"\n{'#' * 80}")
        print(f"Experiment {i}/{total_runs}")
        print(f"{'#' * 80}")

        success = run_experiment(
            config_path=config_path,
            lr=lr,
            max_nc=max_nc,
            seed=args.seed,
        )

        results.append(
            {
                "lr": lr,
                "max_nc": max_nc,
                "success": success,
            }
        )

    # Summary
    print(f"\n{'=' * 80}")
    print("SWEEP SUMMARY")
    print(f"{'=' * 80}")

    successful = sum(1 for r in results if r["success"])
    print(f"Successful: {successful}/{total_runs}")

    if successful < total_runs:
        print("\nFailed experiments:")
        for r in results:
            if not r["success"]:
                print(f"  - max_nc={r['max_nc']}, lr={r['lr']}")


if __name__ == "__main__":
    main()
