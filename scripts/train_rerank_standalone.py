"""Standalone training script for rerank models (Stage 2).

This script allows training rerank models independently from retrieval.
Can work in two modes:
1. With pre-trained retrieval: Load retrieval model and use its candidates
2. Ground truth mode: Use ground truth + random negatives (no retrieval needed)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from config import arg
from evaluation.utils import load_dataset_from_csv, evaluate_split
from rerank.registry import get_reranker_class
from retrieval.registry import get_retriever_class
from pytorch_lightning import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Train rerank model standalone (independent from retrieval)")
    
    # Dataset
    parser.add_argument("--rerank_method", type=str, required=True,
                       choices=["qwen", "qwen3vl", "vip5", "bert4rec"],
                       help="Rerank method to train")
    parser.add_argument("--rerank_top_k", type=int, default=50,
                       help="Number of final recommendations")
    
    # Training mode
    parser.add_argument("--mode", type=str, default="ground_truth",
                       choices=["ground_truth", "retrieval"],
                       help="Training mode: 'ground_truth' (gt + negatives, no retrieval) or 'retrieval' (use pre-trained retrieval)")
    
    # Retrieval config (only needed if mode=retrieval)
    parser.add_argument("--retrieval_method", type=str, default="lrurec",
                       help="Retrieval method (only used if mode=retrieval)")
    parser.add_argument("--retrieval_top_k", type=int, default=200,
                       help="Number of candidates from retrieval (only used if mode=retrieval)")
    
    # Evaluation
    parser.add_argument("--metric_k", type=int, default=10,
                       help="Cutoff for evaluation metrics")
    
    args = parser.parse_args()
    
    seed_everything(arg.seed)
    
    print("=" * 80)
    print("Standalone Rerank Training")
    print("=" * 80)
    print(f"Rerank Method: {args.rerank_method}")
    print(f"Mode: {args.mode}")
    if args.mode == "retrieval":
        print(f"Retrieval Method: {args.retrieval_method} (top_k={args.retrieval_top_k})")
    print("=" * 80)
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
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
    item_count = data["item_count"]
    
    print(f"  Train users: {len(train)}")
    print(f"  Val users: {len(val)}")
    print(f"  Test users: {len(test)}")
    print(f"  Items: {item_count}")
    
    # Create reranker
    print(f"\n[2/3] Creating reranker ({args.rerank_method})...")
    RerankerCls = get_reranker_class(args.rerank_method)
    
    # Prepare reranker kwargs
    reranker_kwargs = {"top_k": args.rerank_top_k}
    
    # Add method-specific kwargs
    if args.rerank_method == "qwen":
        max_candidates = getattr(arg, 'qwen_max_candidates', None)
        if max_candidates is not None:
            reranker_kwargs["max_candidates"] = max_candidates
    elif args.rerank_method == "qwen3vl":
        qwen3vl_mode = getattr(arg, 'qwen3vl_mode', 'raw_image')
        reranker_kwargs["mode"] = qwen3vl_mode
        max_candidates = getattr(arg, 'qwen_max_candidates', None)
        if max_candidates is not None:
            reranker_kwargs["max_candidates"] = max_candidates
    
    reranker = RerankerCls(**reranker_kwargs)
    
    # Prepare training kwargs
    training_kwargs = {
        "vocab_size": item_count + 1,  # For BERT4Rec
        "val_data": val,  # For early stopping
        "num_epochs": arg.rerank_epochs,
        "batch_size": arg.rerank_batch_size,
        "lr": arg.rerank_lr,
        "patience": arg.rerank_patience,
    }
    
    # Add text/image features if available
    if "meta" in data:
        item_id2text = {}
        user_history_text = {}
        item_meta = {}
        
        for item_id, meta in data["meta"].items():
            text = meta.get("text") if meta else None
            if text:
                item_id2text[item_id] = text
            item_meta[item_id] = meta if meta else {}
        
        # Build user history texts
        for user_id, items in train.items():
            user_history_text[user_id] = [
                item_id2text.get(item_id, f"item_{item_id}")
                for item_id in items
                if item_id in item_id2text
            ]
        
        if item_id2text:
            training_kwargs["item_id2text"] = item_id2text
            training_kwargs["user_history"] = user_history_text
        
        # Add item_meta for Qwen3-VL
        if args.rerank_method == "qwen3vl":
            training_kwargs["item_meta"] = item_meta
            if getattr(arg, 'qwen3vl_mode', 'raw_image') == "raw_image":
                user_history_images = {}
                for user_id, items in train.items():
                    user_history_images[user_id] = [
                        item_meta.get(item_id, {}).get("image_path") or item_meta.get(item_id, {}).get("image")
                        for item_id in items
                        if item_id in item_meta
                    ]
                training_kwargs["user_history"] = user_history_images
    
    # Load retrieval model if needed
    if args.mode == "retrieval":
        print(f"\n[2.5/3] Loading pre-trained retrieval model ({args.retrieval_method})...")
        RetrieverCls = get_retriever_class(args.retrieval_method)
        retriever = RetrieverCls(top_k=args.retrieval_top_k)
        retriever.fit(train, item_count=item_count, val_data=val)
        print("  Retrieval model loaded and ready")
    
    # Train reranker
    print(f"\n[3/3] Training reranker...")
    reranker.fit(train, **training_kwargs)
    print("  Rerank training completed!")
    
    # Evaluate
    print(f"\n[4/4] Evaluating reranker...")
    ks = [5, 10, 20]
    
    def recommend_fn(user_id, ground_truth=None):
        """Recommendation function for evaluation."""
        if args.mode == "ground_truth":
            # Ground truth mode: use gt + random negatives
            if ground_truth is None:
                return []
            
            import random
            # Get all items
            all_items = set(range(1, item_count + 1))
            # Exclude user's history
            user_history = set(train.get(user_id, []))
            exclude_set = user_history - set(ground_truth)
            candidate_pool = all_items - exclude_set - set(ground_truth)
            
            # Sample negatives
            num_negatives = 19
            num_negatives = min(num_negatives, len(candidate_pool))
            if num_negatives > 0:
                negatives = random.sample(list(candidate_pool), num_negatives)
                candidates = list(ground_truth) + negatives
            else:
                candidates = list(ground_truth)
            
            if not candidates:
                return []
            
            scored = reranker.rerank(user_id, candidates)
            return [item_id for item_id, _ in scored]
        
        else:  # retrieval mode
            # Use retrieval to get candidates
            exclude_set = set(train.get(user_id, []))
            candidates = retriever.retrieve(user_id, exclude_items=exclude_set)
            
            if not candidates:
                return []
            
            scored = reranker.rerank(user_id, candidates)
            return [item_id for item_id, _ in scored]
    
    # Evaluate on val and test
    val_metrics = evaluate_split(recommend_fn, val, k=args.metric_k, ks=ks, ground_truth_mode=(args.mode == "ground_truth"))
    test_metrics = evaluate_split(recommend_fn, test, k=args.metric_k, ks=ks, ground_truth_mode=(args.mode == "ground_truth"))
    
    # Print results
    print("=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Rerank Method: {args.rerank_method}")
    print("-" * 80)
    print(f"Val Metrics:")
    print(f"  {'Metric':<12} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [val_metrics.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"  {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  {'Metric':<12} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [test_metrics.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"  {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

