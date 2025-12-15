"""Training script for end-to-end two-stage pipeline.

This script trains both retrieval (Stage 1) and reranking (Stage 2) models
in sequence, then evaluates the full pipeline.
"""

import argparse
from pathlib import Path
from typing import Dict, List

from config import arg, EXPERIMENT_ROOT
from dataset import dataset_factory
from evaluation.metrics import recall_at_k, ndcg_at_k
from evaluation.utils import evaluate_split, load_dataset_from_csv
from pipelines.base import PipelineConfig, RetrievalConfig, RerankConfig, TwoStagePipeline
from pytorch_lightning import seed_everything


def evaluate_pipeline(
    pipeline: TwoStagePipeline,
    split: Dict[int, List[int]],
    k: int = 10
) -> Dict[str, float]:
    """Evaluate pipeline on a split. Wrapper for evaluation.utils.evaluate_split."""
    return evaluate_split(pipeline.recommend, split, k)


def main():
    parser = argparse.ArgumentParser(description="Train two-stage recommendation pipeline")
    parser.add_argument("--retrieval_method", type=str, default="lrurec",
                       help="Retrieval method (lrurec, mmgcn)")
    parser.add_argument("--retrieval_top_k", type=int, default=200,
                       help="Number of candidates from Stage 1. "
                            "Note: If using 'qwen' reranker, recommend <= 20 (LLM limit)")
    parser.add_argument("--rerank_method", type=str, default="identity",
                       help="Rerank method (identity, random, qwen, vip5). "
                            "Note: 'qwen' only supports up to 20 candidates")
    parser.add_argument("--rerank_top_k", type=int, default=50,
                       help="Number of final recommendations")
    parser.add_argument("--metric_k", type=int, default=10,
                       help="Cutoff for evaluation metrics")
    
    args = parser.parse_args()
    
    seed_everything(arg.seed)
    
    print("=" * 80)
    print("Two-Stage Pipeline Training")
    print("=" * 80)
    print(f"Retrieval: {args.retrieval_method} (top_k={args.retrieval_top_k})")
    print(f"Rerank: {args.rerank_method} (top_k={args.rerank_top_k})")
    
    # Warning if using Qwen reranker with > 20 candidates
    if args.rerank_method.lower() == "qwen" and args.retrieval_top_k > 20:
        print("\n⚠️  WARNING: Qwen reranker only supports up to 20 candidates.")
        print(f"   Retrieval top_k={args.retrieval_top_k} will be truncated to 20.")
        print("   Consider setting --retrieval_top_k <= 20 for better results.")
    
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
        top_k=args.rerank_top_k
    )
    pipeline_cfg = PipelineConfig(
        retrieval=retrieval_cfg,
        rerank=rerank_cfg
    )
    
    # Create pipeline
    pipeline = TwoStagePipeline(pipeline_cfg)
    
    # Train Stage 1
    print(f"\n[2/4] Training Stage 1 ({args.retrieval_method})...")
    retriever_kwargs = {
        "item_count": item_count,
        "val_data": val,
    }
    pipeline.fit(train, retriever_kwargs={"item_count": item_count, "val_data": val})
    
    # Evaluate Stage 1
    print(f"\n[3/4] Evaluating Stage 1...")
    val_metrics_stage1 = evaluate_pipeline(pipeline, val, k=args.metric_k)
    test_metrics_stage1 = evaluate_pipeline(pipeline, test, k=args.metric_k)
    
    print(f"  Val  - Recall@{args.metric_k}: {val_metrics_stage1['recall']:.4f}, "
          f"NDCG@{args.metric_k}: {val_metrics_stage1['ndcg']:.4f}")
    print(f"  Test - Recall@{args.metric_k}: {test_metrics_stage1['recall']:.4f}, "
          f"NDCG@{args.metric_k}: {test_metrics_stage1['ndcg']:.4f}")
    
    # Train Stage 2 (if not identity)
    if args.rerank_method.lower() not in ("identity", "none"):
        print(f"\n[4/4] Training Stage 2 ({args.rerank_method})...")
        # TODO: Add reranker training logic here
        # For now, identity reranker doesn't need training
        pass
    
    # Evaluate full pipeline
    print(f"\n[5/5] Evaluating Full Pipeline...")
    val_metrics_full = evaluate_pipeline(pipeline, val, k=args.metric_k)
    test_metrics_full = evaluate_pipeline(pipeline, test, k=args.metric_k)
    
    print("=" * 80)
    print("Final Results")
    print("=" * 80)
    print(f"Stage 1 Only:")
    print(f"  Val  - Recall@{args.metric_k}: {val_metrics_stage1['recall']:.4f}, "
          f"NDCG@{args.metric_k}: {val_metrics_stage1['ndcg']:.4f}")
    print(f"  Test - Recall@{args.metric_k}: {test_metrics_stage1['recall']:.4f}, "
          f"NDCG@{args.metric_k}: {test_metrics_stage1['ndcg']:.4f}")
    print(f"\nFull Pipeline (Stage 1 + Stage 2):")
    print(f"  Val  - Recall@{args.metric_k}: {val_metrics_full['recall']:.4f}, "
          f"NDCG@{args.metric_k}: {val_metrics_full['ndcg']:.4f}")
    print(f"  Test - Recall@{args.metric_k}: {test_metrics_full['recall']:.4f}, "
          f"NDCG@{args.metric_k}: {test_metrics_full['ndcg']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

