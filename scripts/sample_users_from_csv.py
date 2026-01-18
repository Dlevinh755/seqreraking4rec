# python scripts/sample_users_from_csv.py \
#   --dataset_code beauty 
#   --min_rating 4 
#   --min_uc 6 --min_sc 6 \
#   --sample_users 200 
#   --sample_seed 123

import argparse
import random
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from dataset.paths import get_preprocessed_csv_path, get_preprocessed_folder_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Sample users from preprocessed CSV")
    parser.add_argument("--data_path", type=str, default="data", help="Path to data folder")
    parser.add_argument("--dataset_code", type=str, default="beauty", help="Dataset code")
    parser.add_argument("--min_rating", type=int, default=4, help="Minimum rating used in preprocessing")
    parser.add_argument("--min_uc", type=int, default=6, help="Minimum user count used in preprocessing")
    parser.add_argument("--min_sc", type=int, default=6, help="Minimum item count used in preprocessing")
    parser.add_argument("--sample_users", type=int, default=100, help="Number of users to keep; if <=0 or >= total, keeps all")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional output file path; defaults to preprocessed folder with sampled suffix")
    return parser.parse_args()


def main():
    args = _parse_args()

    csv_path = get_preprocessed_csv_path(args.dataset_code, args.min_rating, args.min_uc, args.min_sc, args.data_path)
    if not csv_path.exists():
        print(f"[sample_users] CSV not found at {csv_path}. Run data_prepare.py first.")
        return

    df = pd.read_csv(csv_path)
    if "user_id" not in df.columns:
        print("[sample_users] Column user_id not found in CSV; cannot sample.")
        return

    users = sorted(df["user_id"].unique().tolist())
    total_users = len(users)
    n = args.sample_users
    if n is None or n <= 0 or n >= total_users:
        chosen = users
    else:
        rng = random.Random(args.sample_seed)
        rng.shuffle(users)
        chosen = users[:n]

    sampled_df = df[df["user_id"].isin(chosen)].copy()

    if args.output_csv:
        out_path = Path(args.output_csv)
    else:
        folder = get_preprocessed_folder_path(args.dataset_code, args.min_rating, args.min_uc, args.min_sc, args.data_path)
        out_path = folder / f"dataset_single_export.csv"

    sampled_df.to_csv(out_path, index=False)

    split_counts = sampled_df.groupby("split")["user_id"].nunique().to_dict() if "split" in sampled_df.columns else {}

    print("[sample_users] Sampling complete")
    print(f"  Input users: {total_users}")
    print(f"  Sampled users: {len(chosen)}")
    print(f"  Output: {out_path}")
    if split_counts:
        print(f"  Users per split: {split_counts}")


if __name__ == "__main__":
    main()
