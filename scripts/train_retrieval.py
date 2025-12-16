"""Training script for retrieval models (Stage 1).

This script trains retrieval models and saves candidates for Stage 2 reranking.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pickle
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from pytorch_lightning import seed_everything
import torch.nn.functional as F
# Note: config import is moved to main() to avoid argument parsing conflicts
from dataset import dataset_factory
from evaluation.metrics import recall_at_k, ndcg_at_k, absolute_recall_mrr_ndcg_for_ks
from evaluation.utils import evaluate_split
from retrieval.registry import get_retriever_class
from dataset.paths import get_clip_embeddings_path
import pandas as pd
import json


RETRIEVAL_TOP_K = 200  # how many items to retrieve per user
METRIC_K = 10          # cutoff for Recall@K, NDCG@K
METRIC_KS_FOR_RETRIEVED = [1, 5, 10, 20, 50]
RETRIEVAL_SAVE_TOP_K = 20  # how many top candidate scores to store per user


def _evaluate_split(
    retriever,
    split: Dict[int, List[int]],
    k: int = None,
    ks: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Evaluate retriever on a split. 
    
    Uses retriever._evaluate_split() if available (optimized with batching),
    otherwise falls back to evaluate_split(retriever.retrieve, ...).
    
    Args:
        retriever: Retriever instance
        split: Dict {user_id: [item_ids]} - ground truth
        k: Single cutoff (used if ks is None)
        ks: List of K values to evaluate (e.g., [5, 10, 20])
    """
    if ks is None:
        if k is None:
            ks = [10]  # Default
        else:
            ks = [k]
    
    # Check if retriever has optimized _evaluate_split method (MMGCN, VBPR, BM3)
    if hasattr(retriever, '_evaluate_split'):
        # Use retriever's optimized _evaluate_split (supports batching)
        # Note: retriever._evaluate_split typically returns a single float (recall@k)
        # We need to call it for each K value and build the result dict
        result = {"num_users": len(split)}
        
        # For efficiency, compute recall for all K values using optimized _evaluate_split
        for k_val in ks:
            recall = retriever._evaluate_split(split, k=k_val)
            result[f"recall@{k_val}"] = float(recall)
        
        # For NDCG and Hit, compute from a small sample to avoid calling retrieve() too many times
        # This is much faster than calling retrieve() for all users
        from evaluation.metrics import ndcg_at_k, hit_at_k
        
        # Sample a small subset for NDCG/Hit computation (much faster)
        sample_size = min(100, len(split))  # Only sample 100 users for NDCG/Hit
        if len(split) > sample_size:
            import random
            sample_users = random.sample(list(split.keys()), sample_size)
            sample_split = {uid: split[uid] for uid in sample_users}
        else:
            sample_split = split
        
        # Pre-compute model forward once for all sampled users (if MMGCN-like)
        # For MMGCN, we can get all scores in one batch
        if hasattr(retriever, 'model') and hasattr(retriever.model, 'forward'):
            # Try to get scores for all sampled users in one batch
            try:
                retriever.model.eval()
                with torch.no_grad():
                    if hasattr(retriever.model, 'forward'):
                        retriever.model.forward()  # Update embeddings once
                
                # Get user and item embeddings
                if hasattr(retriever.model, 'result'):
                    user_tensor = retriever.model.result[:retriever.num_user]
                    item_tensor = retriever.model.result[retriever.num_user:]
                    
                    # Get scores for all sampled users in one batch
                    sample_user_ids = list(sample_split.keys())
                    sample_user_indices = torch.tensor([uid - 1 for uid in sample_user_ids if 1 <= uid <= retriever.num_user], dtype=torch.long).to(retriever.device)
                    if len(sample_user_indices) > 0:
                        sample_user_emb = user_tensor[sample_user_indices]
                        sample_scores = torch.matmul(sample_user_emb, item_tensor.t())  # [sample_size, num_items]
                        
                        # Compute NDCG and Hit for each K
                        for k_val in ks:
                            ndcgs = []
                            hits = []
                            for idx, (user_id, gt_items) in enumerate(zip(sample_user_ids, [sample_split[uid] for uid in sample_user_ids])):
                                if idx >= len(sample_scores) or not gt_items:
                                    continue
                                
                                # Get top-K items
                                scores = sample_scores[idx].cpu().numpy()
                                top_k_indices = np.argsort(scores)[::-1][:k_val]
                                top_items = [int(idx + 1) for idx in top_k_indices]  # Convert to 1-indexed
                                
                                # Compute metrics
                                ndcgs.append(ndcg_at_k(top_items, gt_items, k_val))
                                hits.append(hit_at_k(top_items, gt_items, k_val))
                            
                            result[f"ndcg@{k_val}"] = float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0
                            result[f"hit@{k_val}"] = float(sum(hits) / len(hits)) if hits else 0.0
                    else:
                        # Fallback: use retrieve() for sample
                        for k_val in ks:
                            ndcgs = []
                            hits = []
                            for user_id, gt_items in sample_split.items():
                                if not gt_items:
                                    continue
                                recs = retriever.retrieve(user_id)
                                if not recs:
                                    continue
                                ndcgs.append(ndcg_at_k(recs, gt_items, k_val))
                                hits.append(hit_at_k(recs, gt_items, k_val))
                            
                            result[f"ndcg@{k_val}"] = float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0
                            result[f"hit@{k_val}"] = float(sum(hits) / len(hits)) if hits else 0.0
            except Exception as e:
                # Fallback: use retrieve() for sample if batch computation fails
                print(f"[_evaluate_split] Warning: Batch computation failed: {e}. Using retrieve() for sample.")
                for k_val in ks:
                    ndcgs = []
                    hits = []
                    for user_id, gt_items in sample_split.items():
                        if not gt_items:
                            continue
                        recs = retriever.retrieve(user_id)
                        if not recs:
                            continue
                        ndcgs.append(ndcg_at_k(recs, gt_items, k_val))
                        hits.append(hit_at_k(recs, gt_items, k_val))
                    
                    result[f"ndcg@{k_val}"] = float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0
                    result[f"hit@{k_val}"] = float(sum(hits) / len(hits)) if hits else 0.0
        else:
            # Fallback: use retrieve() for sample
            for k_val in ks:
                ndcgs = []
                hits = []
                for user_id, gt_items in sample_split.items():
                    if not gt_items:
                        continue
                    recs = retriever.retrieve(user_id)
                    if not recs:
                        continue
                    ndcgs.append(ndcg_at_k(recs, gt_items, k_val))
                    hits.append(hit_at_k(recs, gt_items, k_val))
                
                result[f"ndcg@{k_val}"] = float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0
                result[f"hit@{k_val}"] = float(sum(hits) / len(hits)) if hits else 0.0
        
        return result
    else:
        # Fallback: Use generic evaluate_split (calls retrieve() for each user)
        return evaluate_split(retriever.retrieve, split, k=k if k else 10, ks=ks)


def _build_edge_index(train_data: Dict[int, List[int]], num_user: int, num_item: int) -> np.ndarray:
    """Build edge_index from user-item interactions for graph-based models.
    
    Args:
        train_data: Dict {user_id: [item_ids]} - training interactions
        num_user: Number of users
        num_item: Number of items
        
    Returns:
        edge_index: np.ndarray of shape [2, E] where E is number of edges
        Format: [user_nodes, item_nodes] where item_nodes are offset by num_user
    """
    edges = []
    for user_id, items in train_data.items():
        if user_id < 1 or user_id > num_user:
            continue
        for item_id in items:
            if item_id < 1 or item_id > num_item:
                continue
            # User nodes: 0 to num_user-1
            # Item nodes: num_user to num_user+num_item-1
            user_node = user_id - 1  # 0-indexed
            item_node = num_user + item_id - 1  # 0-indexed, offset by num_user
            edges.append([user_node, item_node])
    
    if not edges:
        raise ValueError("No valid edges found in train_data")
    
    edge_index = np.array(edges, dtype=np.int64).T  # Shape: [2, E]
    return edge_index


def _load_clip_embeddings(
    dataset_code: str,
    min_rating: int,
    min_uc: int,
    min_sc: int,
    num_items: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load CLIP embeddings for visual and text features.
    
    Args:
        dataset_code: Dataset code
        min_rating: Minimum rating threshold
        min_uc: Minimum user count
        min_sc: Minimum item count
        num_items: Number of items (for validation)
        
    Returns:
        Tuple of (v_feat, t_feat) where each is np.ndarray of shape [num_items, D] or None
    """
    clip_path = get_clip_embeddings_path(dataset_code, min_rating, min_uc, min_sc)
    
    if not clip_path.exists():
        raise FileNotFoundError(
            f"CLIP embeddings not found at {clip_path}. "
            "Please run data_prepare.py with --use_image and --use_text flags first."
        )
    
    clip_payload = torch.load(clip_path, map_location="cpu")
    image_embs = clip_payload.get("image_embs")  # Shape: [num_items+1, D] (row 0 is padding)
    text_embs = clip_payload.get("text_embs")   # Shape: [num_items+1, D] (row 0 is padding)
    
    v_feat = None
    t_feat = None
    
    if image_embs is not None:
        # Skip row 0 (padding), use rows 1..num_items
        v_feat = image_embs[1:num_items+1].numpy()
    
    if text_embs is not None:
        # Skip row 0 (padding), use rows 1..num_items
        t_feat = text_embs[1:num_items+1].numpy()
    
    if v_feat is None and t_feat is None:
        raise ValueError("Both image_embs and text_embs are None in CLIP embeddings file")
    
    return v_feat, t_feat


def _build_retrieved_matrices(
    retriever,
    split: Dict[int, List[int]],
    item_count: int,
) -> Dict[str, List]:
    users = sorted(split.keys())
    # We'll store only the top-`RETRIEVAL_SAVE_TOP_K` candidate IDs and their scores
    probs: List[dict] = []
    labels: List[int] = []

    for u in users:
        gt_items = split.get(u, [])
        if not gt_items:
            continue
        label = gt_items[0]

        cands = retriever.retrieve(u)
        top_ids: List[int] = []
        top_scores: List[float] = []

        # Assign descending scores based on candidate rank and keep only top-K
        for rank, item_id in enumerate(cands):
            if 0 < item_id <= item_count:
                score = float(len(cands) - rank)
                if len(top_ids) < RETRIEVAL_SAVE_TOP_K:
                    top_ids.append(int(item_id))
                    top_scores.append(score)
                else:
                    break

        probs.append({"ids": top_ids, "scores": top_scores})
        labels.append(int(label))

    return {"probs": probs, "labels": labels}


def main() -> None:
    # Import config (it now includes --retrieval_method as optional argument)
    from config import arg, EXPERIMENT_ROOT
    
    # Get retrieval_method from arg (with default fallback)
    retrieval_method = getattr(arg, 'retrieval_method', 'lrurec')
    if retrieval_method is None:
        retrieval_method = 'lrurec'
    
    # Validate retrieval_method
    valid_methods = ["lrurec", "mmgcn", "vbpr", "bm3"]
    if retrieval_method not in valid_methods:
        raise ValueError(f"Invalid retrieval_method: {retrieval_method}. Must be one of {valid_methods}")

    seed_everything(arg.seed)

    # Load dataset using utility function
    from evaluation.utils import load_dataset_from_csv
    
    data = load_dataset_from_csv(
        arg.dataset_code,
        arg.min_rating,
        arg.min_uc,
        arg.min_sc
    )
    train = data["train"]
    val = data["val"]
    test = data["test"]
    meta = data["meta"]
    item_count = max(meta.keys()) if meta else 0
    
    # Get number of users and items
    # IMPORTANT: Use max(train.keys()) for embedding size (user_ids are 1-indexed and may not be continuous)
    # But also track actual number of users for validation
    num_users = max(train.keys()) if train else 0  # Max user_id (for embedding size)
    actual_num_users = len(train)  # Actual number of users (for validation/logging)
    num_items = item_count
    
    # Debug: Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Train users: {len(train)} (max user_id: {num_users})")
    print(f"  Val users: {len(val)}")
    print(f"  Test users: {len(test)}")
    print(f"  Items: {item_count}")
    if len(test) > 0:
        max_test_user_id = max(test.keys())
        print(f"  Max test user_id: {max_test_user_id}")
        if max_test_user_id > num_users:
            print(f"  WARNING: Max test user_id ({max_test_user_id}) > num_users ({num_users})!")
            print(f"  This will cause test users to be skipped during evaluation.")
    print()

    RetrieverCls = get_retriever_class(retrieval_method)
    
    # Prepare retriever kwargs based on method
    # Standardize hyperparameters for fair comparison
    retriever_kwargs = {
        "top_k": RETRIEVAL_TOP_K,
        "num_epochs": arg.retrieval_epochs,
        "batch_size": arg.batch_size_retrieval,
        "patience": arg.retrieval_patience,
        "lr": arg.retrieval_lr,  # Standardize learning rate across all methods
    }
    
    # For MMGCN and VBPR, we need additional dependencies
    fit_kwargs = {
        "item_count": item_count,
        "val_data": val,
    }
    
    if retrieval_method == "mmgcn":
        print("Loading CLIP embeddings for MMGCN...")
        v_feat, t_feat = _load_clip_embeddings(
            arg.dataset_code,
            arg.min_rating,
            arg.min_uc,
            arg.min_sc,
            num_items
        )
        
        if v_feat is None:
            raise ValueError("MMGCN requires image embeddings (v_feat), but image_embs is None")
        if t_feat is None:
            raise ValueError("MMGCN requires text embeddings (t_feat), but text_embs is None")
        
        print(f"Building edge_index from {len(train)} user interactions...")
        edge_index = _build_edge_index(train, num_users, num_items)
        print(f"Built edge_index with shape {edge_index.shape} ({edge_index.shape[1]} edges)")
        
        fit_kwargs.update({
            "num_user": num_users,
            "num_item": num_items,
            "v_feat": v_feat,
            "t_feat": t_feat,
            "edge_index": edge_index,
        })
    
    elif retrieval_method == "vbpr":
        print("Loading CLIP embeddings for VBPR...")
        v_feat, t_feat = _load_clip_embeddings(
            arg.dataset_code,
            arg.min_rating,
            arg.min_uc,
            arg.min_sc,
            num_items
        )
        
        if v_feat is None:
            raise ValueError("VBPR requires image embeddings (v_feat), but image_embs is None")
        
        # Convert to torch.Tensor (VBPR expects torch.Tensor)
        visual_features = torch.from_numpy(v_feat).float()
        
        fit_kwargs.update({
            "num_user": num_users,
            "num_item": num_items,
            "visual_features": visual_features,
            "dataset_code": arg.dataset_code,
            "min_rating": arg.min_rating,
            "min_uc": arg.min_uc,
            "min_sc": arg.min_sc,
        })
    
    elif retrieval_method == "bm3":
        print("Loading CLIP embeddings for BM3...")
        v_feat, t_feat = _load_clip_embeddings(
            arg.dataset_code,
            arg.min_rating,
            arg.min_uc,
            arg.min_sc,
            num_items
        )
        
        if v_feat is None:
            raise ValueError("BM3 requires image embeddings (v_feat), but image_embs is None")
        if t_feat is None:
            raise ValueError("BM3 requires text embeddings (t_feat), but text_embs is None")
        
        # Convert to torch.Tensor (BM3 expects torch.Tensor)
        visual_features = torch.from_numpy(v_feat).float()
        text_features = torch.from_numpy(t_feat).float()
        
        fit_kwargs.update({
            "num_user": num_users,
            "num_item": num_items,
            "visual_features": visual_features,
            "text_features": text_features,
            "dataset_code": arg.dataset_code,
            "min_rating": arg.min_rating,
            "min_uc": arg.min_uc,
            "min_sc": arg.min_sc,
        })
    
    retriever = RetrieverCls(**retriever_kwargs)
    retriever.fit(train, **fit_kwargs)

    # 4) Evaluate on val / test (baseline Stage 1)
    print("\n" + "=" * 80)
    best_epoch_info = f"epoch {retriever.best_state}" if hasattr(retriever, 'best_state') else "training completed"
    print(f"Load best model state from training: {best_epoch_info}")
    print("=" * 80 + "\n")
    print("Evaluating Stage 1 Retrieval on test set...")
    # Evaluate with multiple K values: 5, 10, 20
    test_metrics = _evaluate_split(retriever, test, ks=[5, 10, 20])

    print("=" * 80)
    print(f"Stage 1 Retrieval Evaluation - Method: {retrieval_method}")
    print(f"Dataset     : {arg.dataset_code}")
    print(f"min_rating  : {arg.min_rating}")
    print(f"min_uc      : {arg.min_uc}")
    print(f"min_sc      : {arg.min_sc}")
    print(f"Retrieval K : {RETRIEVAL_TOP_K}")
    print("-" * 80)
    
    # Format output as requested: Recall@10, NDCG@10, Hit@10, Recall@5, NDCG@5, Hit@5, Recall@20, NDCG@20, Hit@20
    num_users = test_metrics.get('num_users', 0)
    recall_5 = test_metrics.get('recall@5', 0.0)
    ndcg_5 = test_metrics.get('ndcg@5', 0.0)
    hit_5 = test_metrics.get('hit@5', 0.0)
    recall_10 = test_metrics.get('recall@10', 0.0)
    ndcg_10 = test_metrics.get('ndcg@10', 0.0)
    hit_10 = test_metrics.get('hit@10', 0.0)
    recall_20 = test_metrics.get('recall@20', 0.0)
    ndcg_20 = test_metrics.get('ndcg@20', 0.0)
    hit_20 = test_metrics.get('hit@20', 0.0)
    
    print(f"TEST - users: {num_users}, "
          f"Recall@10: {recall_10:.4f}, NDCG@10: {ndcg_10:.4f}, Hit@10: {hit_10:.4f}, "
          f"Recall@5: {recall_5:.4f}, NDCG@5: {ndcg_5:.4f}, Hit@5: {hit_5:.4f}, "
          f"Recall@20: {recall_20:.4f}, NDCG@20: {ndcg_20:.4f}, Hit@20: {hit_20:.4f}")
    print("=" * 80)

    # 5) Build retrieved.pkl cho Stage 2 (LlamaRec-compatible format)
    from dataset.paths import get_experiment_path
    
    export_root = get_experiment_path("retrieval", retrieval_method, arg.dataset_code, arg.seed)
    export_root.mkdir(parents=True, exist_ok=True)
    retrieved_path = export_root / "retrieved.pkl"

    val_pack = _build_retrieved_matrices(retriever, val, item_count)
    test_pack = _build_retrieved_matrices(retriever, test, item_count)

    if val_pack["probs"] and test_pack["probs"]:
        # Reconstruct full-size score tensors from the compact top-K lists for evaluation
        val_n = len(val_pack["probs"])
        test_n = len(test_pack["probs"])

        val_scores = torch.full((val_n, item_count + 1), -1e9, dtype=torch.float32)
        for i, p in enumerate(val_pack["probs"]):
            ids = p.get("ids", [])
            scores = p.get("scores", [])
            for iid, sc in zip(ids, scores):
                if 0 < iid <= item_count:
                    val_scores[i, iid] = float(sc)

        test_scores = torch.full((test_n, item_count + 1), -1e9, dtype=torch.float32)
        for i, p in enumerate(test_pack["probs"]):
            ids = p.get("ids", [])
            scores = p.get("scores", [])
            for iid, sc in zip(ids, scores):
                if 0 < iid <= item_count:
                    test_scores[i, iid] = float(sc)

        val_labels = torch.tensor(val_pack["labels"], dtype=torch.long).view(-1)
        test_labels = torch.tensor(test_pack["labels"], dtype=torch.long).view(-1)

        val_metrics_full = absolute_recall_mrr_ndcg_for_ks(val_scores, val_labels, METRIC_KS_FOR_RETRIEVED)
        test_metrics_full = absolute_recall_mrr_ndcg_for_ks(test_scores, test_labels, METRIC_KS_FOR_RETRIEVED)
    else:
        val_metrics_full = {}
        test_metrics_full = {}

    # Export retrieved candidates to CSV (one file) and metrics to JSON
    rows = []
    for split_name, pack in [("val", val_pack), ("test", test_pack)]:
        probs = pack.get("probs", [])
        labels = pack.get("labels", [])
        for i, p in enumerate(probs):
            ids = p.get("ids", [])
            scores = p.get("scores", [])
            label = labels[i] if i < len(labels) else None
            rows.append({
                "split": split_name,
                "user_index": i,
                "label": int(label) if label is not None else None,
                "candidate_ids": json.dumps(ids, ensure_ascii=False),
                "candidate_scores": json.dumps(scores, ensure_ascii=False),
            })

    df_out = pd.DataFrame(rows)
    from dataset.paths import get_retrieved_csv_path, get_retrieved_metrics_path
    
    csv_out = get_retrieved_csv_path(retrieval_method, arg.dataset_code, arg.seed)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(csv_out, index=False)

    metrics = {
        "val_metrics": val_metrics_full,
        "test_metrics": test_metrics_full,
    }
    metrics_out = get_retrieved_metrics_path(retrieval_method, arg.dataset_code, arg.seed)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved retrieved candidates to: {csv_out}")
    print(f"Saved retrieved metrics to: {metrics_out}")


if __name__ == "__main__":
    main()

