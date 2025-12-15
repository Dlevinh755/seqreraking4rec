"""Utility functions for evaluation and dataset loading.

This module provides common functions used across training and evaluation scripts.
"""

from pathlib import Path
from typing import Dict, List, Optional

from evaluation.metrics import recall_at_k, ndcg_at_k
from dataset import dataset_factory


def evaluate_split(
    recommend_fn,
    split: Dict[int, List[int]],
    k: int = 10,
    ground_truth_mode: bool = False,
    ks: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Evaluate recommendations on a split.
    
    Generic evaluation function that works with any recommendation function.
    
    Args:
        recommend_fn: Function that takes user_id and optionally ground_truth, returns List[int] recommendations
                      Can be: retriever.retrieve, pipeline.recommend, etc.
        split: Dict {user_id: [item_ids]} - ground truth
        k: Cutoff for metrics (used if ks is None)
        ground_truth_mode: If True, pass ground_truth to recommend_fn (for rerank ground_truth mode)
        ks: List of K values to evaluate (e.g., [5, 10, 20]). If None, uses [k]
        
    Returns:
        Dict with keys: recall@K, ndcg@K, hit@K for each K in ks (or just k if ks is None)
        Also includes "num_users" key.
    """
    from evaluation.metrics import hit_at_k
    
    if ks is None:
        ks = [k]
    
    users = sorted(split.keys())
    
    # Initialize lists for each K
    metrics_by_k = {k_val: {"recalls": [], "ndcgs": [], "hits": []} for k_val in ks}
    
    for user_id in users:
        gt_items = split.get(user_id, [])
        if not gt_items:
            continue
        
        # Get recommendations
        if ground_truth_mode:
            # For ground_truth mode, pass ground_truth to recommend_fn
            recs = recommend_fn(user_id, ground_truth=gt_items)
        else:
            recs = recommend_fn(user_id)
        
        if not recs:
            continue
        
        # Compute metrics for each K
        for k_val in ks:
            r = recall_at_k(recs, gt_items, k_val)
            n = ndcg_at_k(recs, gt_items, k_val)
            h = hit_at_k(recs, gt_items, k_val)
            
            metrics_by_k[k_val]["recalls"].append(r)
            metrics_by_k[k_val]["ndcgs"].append(n)
            metrics_by_k[k_val]["hits"].append(h)
    
    # Aggregate results
    result = {"num_users": len(users)}
    
    for k_val in ks:
        if metrics_by_k[k_val]["recalls"]:
            result[f"recall@{k_val}"] = float(sum(metrics_by_k[k_val]["recalls"]) / len(metrics_by_k[k_val]["recalls"]))
            result[f"ndcg@{k_val}"] = float(sum(metrics_by_k[k_val]["ndcgs"]) / len(metrics_by_k[k_val]["ndcgs"]))
            result[f"hit@{k_val}"] = float(sum(metrics_by_k[k_val]["hits"]) / len(metrics_by_k[k_val]["hits"]))
        else:
            result[f"recall@{k_val}"] = 0.0
            result[f"ndcg@{k_val}"] = 0.0
            result[f"hit@{k_val}"] = 0.0
    
    # Backward compatibility: also include "recall" and "ndcg" for the first K
    if len(ks) > 0:
        result["recall"] = result.get(f"recall@{ks[0]}", 0.0)
        result["ndcg"] = result.get(f"ndcg@{ks[0]}", 0.0)
    
    return result


def load_dataset_from_csv(
    dataset_code: str,
    min_rating: int,
    min_uc: int,
    min_sc: int,
) -> Dict:
    """Load dataset from CSV export.
    
    This function loads the dataset from the CSV file created by data_prepare.py.
    It reconstructs train/val/test splits and metadata.
    
    Args:
        dataset_code: Dataset code (beauty, games, ml-100k)
        min_rating: Minimum rating threshold
        min_uc: Minimum user count
        min_sc: Minimum item count
        
    Returns:
        Dict with keys: train, val, test, meta, smap, item_count
    """
    import pandas as pd
    from dataset.paths import get_preprocessed_csv_path
    
    csv_path = get_preprocessed_csv_path(dataset_code, min_rating, min_uc, min_sc)
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV export not found at {csv_path}. Run data_prepare.py first."
        )
    
    df = pd.read_csv(csv_path)
    df = df.reset_index(drop=False).rename(columns={"index": "row_order"})
    
    # Group by (split, user)
    grouped = (
        df.sort_values("row_order")
        .groupby(["split", "user_id"])["item_new_id"]
        .apply(lambda s: s.astype(int).tolist())
    )
    
    train, val, test = {}, {}, {}
    for (split, user), items in grouped.items():
        user = int(user)
        if split == "train":
            train[user] = items
        elif split == "val":
            val[user] = items
        else:
            test[user] = items
    
    # Build meta
    meta_df = df.drop_duplicates(subset=["item_new_id"]).set_index("item_new_id")
    meta = {}
    for item_new_id, row in meta_df.iterrows():
        text = row.get("item_text") if not pd.isna(row.get("item_text")) else None
        image_path = row.get("item_image_path") if not pd.isna(row.get("item_image_path")) else None
        caption = row.get("item_caption") if not pd.isna(row.get("item_caption")) else None
        semantic_summary = row.get("item_semantic_summary") if not pd.isna(row.get("item_semantic_summary")) else None
        meta[int(item_new_id)] = {
            "text": text,
            "image_path": image_path,
            "caption": caption,
            "semantic_summary": semantic_summary
        }
    
    # Build smap
    smap = {}
    map_df = df[~df["Item_id"].isna()].drop_duplicates(subset=["Item_id"]).copy()
    for _, row in map_df.iterrows():
        try:
            orig = row["Item_id"]
            new = int(row["item_new_id"])
            smap[orig] = new
        except Exception:
            continue
    
    item_count = max(meta.keys()) if meta else 0
    
    return {
        "train": train,
        "val": val,
        "test": test,
        "meta": meta,
        "smap": smap,
        "item_count": item_count,
    }

