"""Utility script to clean experiment outputs before retraining.

This is *only* for convenience when you want to re-run Stage 1/Stage 2
experiments from scratch without touching raw/preprocessed data.

It can delete:
- `experiments/retrieval/<method>/<dataset>/seed<seed>/` (retrieved.pkl, logs, ...)
- Optionally the whole `experiments/retrieval/` folder for a method.

Usage examples (from project root):

    python -m tools.cleanup_experiments --method lrurec --dataset beauty --seed 42
    python -m tools.cleanup_experiments --method lrurec --dataset beauty
    python -m tools.cleanup_experiments --method lrurec --all-datasets

NOTE: This script does *not* touch `data/` or `LlamaRec/`.
"""

import argparse
import shutil
from pathlib import Path

from config import EXPERIMENT_ROOT


def remove_dir(p: Path) -> None:
    if p.exists() and p.is_dir():
        print(f"[cleanup] Removing folder: {p}")
        shutil.rmtree(p)
    else:
        print(f"[cleanup] Skip (not found): {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean experiment folders before retraining.")
    parser.add_argument("--method", type=str, required=True, help="Retrieval method name (e.g., lrurec)")
    parser.add_argument("--dataset", type=str, help="Dataset code (e.g., beauty, games, ml_100k)")
    parser.add_argument("--seed", type=int, help="Seed used in experiments (e.g., 42)")
    parser.add_argument("--all-datasets", action="store_true", help="Remove all datasets under this method")

    args = parser.parse_args()

    base = Path(EXPERIMENT_ROOT) / "retrieval" / args.method

    if args.all_datasets:
        # Remove all datasets for this method.
        if not base.exists():
            print(f"[cleanup] No experiments found for method '{args.method}'.")
            return
        for ds_dir in base.iterdir():
            if ds_dir.is_dir():
                if args.seed is not None:
                    seed_dir = ds_dir / f"seed{args.seed}"
                    remove_dir(seed_dir)
                else:
                    remove_dir(ds_dir)
        return

    # If not all-datasets, we require dataset.
    if not args.dataset:
        raise SystemExit("--dataset is required unless you specify --all-datasets")

    ds_base = base / args.dataset
    if args.seed is not None:
        target = ds_base / f"seed{args.seed}"
        remove_dir(target)
    else:
        remove_dir(ds_base)


if __name__ == "__main__":
    main()
