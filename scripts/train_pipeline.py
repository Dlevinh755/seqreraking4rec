"""Training script for end-to-end two-stage pipeline.

This script trains both retrieval (Stage 1) and reranking (Stage 2) models
in sequence, then evaluates the full pipeline.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from config import arg, EXPERIMENT_ROOT
from dataset import dataset_factory
from evaluation.metrics import recall_at_k, ndcg_at_k
from evaluation.utils import evaluate_split, load_dataset_from_csv
from pipelines.base import PipelineConfig, RetrievalConfig, RerankConfig, TwoStagePipeline
from pytorch_lightning import seed_everything


def evaluate_pipeline(
    pipeline: TwoStagePipeline,
    split: Dict[int, List[int]],
    k: int = 10,
    ground_truth_mode: bool = False
) -> Dict[str, float]:
    """Evaluate pipeline on a split. Wrapper for evaluation.utils.evaluate_split.
    
    Args:
        pipeline: TwoStagePipeline instance
        split: Dict {user_id: [item_ids]} - ground truth
        k: Cutoff for metrics
        ground_truth_mode: If True, use ground_truth rerank mode
    """
    return evaluate_split(pipeline.recommend, split, k, ground_truth_mode=ground_truth_mode)


def main():
    parser = argparse.ArgumentParser(description="Train two-stage recommendation pipeline")
    parser.add_argument("--retrieval_method", type=str, default="lrurec",
                       help="Retrieval method (lrurec, mmgcn)")
    parser.add_argument("--retrieval_top_k", type=int, default=200,
                       help="Number of candidates from Stage 1")
    parser.add_argument("--rerank_method", type=str, default="qwen",
                       help="Rerank method (qwen, vip5, bert4rec)")
    parser.add_argument("--rerank_top_k", type=int, default=50,
                       help="Number of final recommendations")
    parser.add_argument("--metric_k", type=int, default=10,
                       help="Cutoff for evaluation metrics")
    parser.add_argument("--rerank_mode", type=str, default="retrieval",
                       choices=["retrieval", "ground_truth"],
                       help="Rerank mode: 'retrieval' (use Stage 1 candidates) or 'ground_truth' (gt + 19 negatives)")
    
    args = parser.parse_args()
    
    seed_everything(arg.seed)
    
    print("=" * 80)
    print("Two-Stage Pipeline Training")
    print("=" * 80)
    print(f"Retrieval: {args.retrieval_method} (top_k={args.retrieval_top_k})")
    print(f"Rerank: {args.rerank_method} (top_k={args.rerank_top_k}, mode={args.rerank_mode})")
    
    print("=" * 80)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
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
    
    # Create pipeline config
    retrieval_cfg = RetrievalConfig(
        method=args.retrieval_method,
        top_k=args.retrieval_top_k
    )
    rerank_cfg = RerankConfig(
        method=args.rerank_method,
        top_k=args.rerank_top_k,
        mode=args.rerank_mode,
        num_negatives=19  # Ground truth + 19 negatives
    )
    pipeline_cfg = PipelineConfig(
        retrieval=retrieval_cfg,
        rerank=rerank_cfg
    )
    
    # Create pipeline
    pipeline = TwoStagePipeline(pipeline_cfg)
    
    # Prepare reranker kwargs (standardize hyperparameters for fair comparison)
    reranker_kwargs = {
        "vocab_size": item_count + 1,  # For BERT4Rec
        "val_data": val,  # For early stopping
        # Standardize hyperparameters from config (for trainable rerankers)
        "num_epochs": arg.rerank_epochs,
        "batch_size": arg.rerank_batch_size,
        "lr": arg.rerank_lr,
        "patience": arg.rerank_patience,
    }
    
    # Add text/image features if available (for Qwen/Qwen3-VL)
    if "meta" in data:
        item_id2text = {}
        user_history_text = {}
        item_meta = {}  # For Qwen3-VL
        
        for item_id, meta in data["meta"].items():
            text = meta.get("text") if meta else None
            if text:
                item_id2text[item_id] = text
            
            # Store full meta for Qwen3-VL
            item_meta[item_id] = meta if meta else {}
        
        # Build user history texts
        for user_id, items in train.items():
            user_history_text[user_id] = [
                item_id2text.get(item_id, f"item_{item_id}")
                for item_id in items
                if item_id in item_id2text
            ]
        
        if item_id2text:
            reranker_kwargs["item_id2text"] = item_id2text
            reranker_kwargs["user_history"] = user_history_text
        
        # Add item_meta for Qwen3-VL
        if args.rerank_method.lower() == "qwen3vl":
            reranker_kwargs["item_meta"] = item_meta
            # For raw_image mode, also pass images in user_history
            if arg.qwen3vl_mode == "raw_image":
                # Build user history with image paths for raw_image mode
                user_history_images = {}
                for user_id, items in train.items():
                    user_history_images[user_id] = [
                        item_meta.get(item_id, {}).get("image_path") or item_meta.get(item_id, {}).get("image")
                        for item_id in items
                        if item_id in item_meta
                    ]
                reranker_kwargs["user_history"] = user_history_images
    
    # Train Stage 1
    print(f"\n[2/4] Training Stage 1 ({args.retrieval_method})...")
    retriever_kwargs = {
        "item_count": item_count,
        "val_data": val,
    }
    pipeline.fit(
        train,
        retriever_kwargs={"item_count": item_count, "val_data": val},
        reranker_kwargs=reranker_kwargs  # Pass standardized reranker kwargs
    )
    
    # Evaluate Stage 1
    print(f"\n[3/4] Evaluating Stage 1...")
    ks = [5, 10, 20]
    val_metrics_stage1 = evaluate_pipeline(pipeline, val, k=args.metric_k, ks=ks)
    test_metrics_stage1 = evaluate_pipeline(pipeline, test, k=args.metric_k, ks=ks)
    
    print(f"  Val Metrics:")
    print(f"    {'Metric':<12} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [val_metrics_stage1.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"    {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
    
    print(f"  Test Metrics:")
    print(f"    {'Metric':<12} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [test_metrics_stage1.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"    {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
    
    # Note: Stage 2 training is handled in pipeline.fit() via reranker_kwargs
    if args.rerank_method.lower() not in ("none", ""):
        print(f"\n[4/4] Stage 2 ({args.rerank_method}) training completed in pipeline.fit()")
    
    # Evaluate full pipeline
    print(f"\n[5/5] Evaluating Full Pipeline...")
    ground_truth_mode = (args.rerank_mode == "ground_truth")
    val_metrics_full = evaluate_pipeline(pipeline, val, k=args.metric_k, ks=ks, ground_truth_mode=ground_truth_mode)
    test_metrics_full = evaluate_pipeline(pipeline, test, k=args.metric_k, ks=ks, ground_truth_mode=ground_truth_mode)
    
    print("=" * 80)
    print("Final Results")
    print("=" * 80)
    print(f"Stage 1 Only - Val:")
    print(f"  {'Metric':<12} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [val_metrics_stage1.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"  {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
    
    print(f"\nStage 1 Only - Test:")
    print(f"  {'Metric':<12} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [test_metrics_stage1.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"  {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
    
    print(f"\nFull Pipeline (Stage 1 + Stage 2) - Val:")
    print(f"  {'Metric':<12} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [val_metrics_full.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"  {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
    
    print(f"\nFull Pipeline (Stage 1 + Stage 2) - Test:")
    print(f"  {'Metric':<12} {'@5':>10} {'@10':>10} {'@20':>10}")
    for metric_name in ["recall", "ndcg", "hit"]:
        values = [test_metrics_full.get(f"{metric_name}@{k}", 0.0) for k in ks]
        print(f"  {metric_name.capitalize():<12} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

