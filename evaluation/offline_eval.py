"""Offline evaluation for retrieval / rerank / full pipeline.

Modes:
- retrieval : Stage 1 only (retrieval-only)
- full      : Stage 1 + Stage 2
- rerank_only : Stage 2 only, using Stage 1 to generate candidates + inject ground truth

Metrics: Recall@K, NDCG@K on val/test split.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from evaluation.metrics import recall_at_k, ndcg_at_k
from pipelines.base import PipelineConfig, RetrievalConfig, RerankConfig, TwoStagePipeline
from retrieval import get_retriever_class
from rerank import get_reranker_class


def load_dataset(
    dataset_code: str,
    min_rating: int,
    min_uc: int,
    min_sc: int,
) -> Dict:
    """Load dataset.pkl dựa trên tham số filter.

    Thư mục: data/preprocessed/{code}_min_rating{min_rating}-min_uc{min_uc}-min_sc{min_sc}/dataset.pkl
    """
    preprocessed_root = Path("data/preprocessed")
    folder_name = f"{dataset_code}_min_rating{min_rating}-min_uc{min_uc}-min_sc{min_sc}"
    dataset_path = preprocessed_root.joinpath(folder_name, "dataset.pkl")

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset.pkl not found at {dataset_path}. "
            "Hãy chạy data_prepare.py với cùng tham số dataset/filter trước."
        )

    with dataset_path.open("rb") as f:
        dataset = pickle.load(f)
    return dataset


def evaluate_users(
    users: List[int],
    recs_by_user: Dict[int, List[int]],
    ground_truth: Dict[int, List[int]],
    k: int,
) -> Tuple[float, float]:
    """Tính trung bình Recall@K và NDCG@K trên một tập user."""
    recalls = []
    ndcgs = []
    for u in users:
        gt_items = ground_truth.get(u, [])
        recs = recs_by_user.get(u, [])
        if not gt_items:
            continue
        r = recall_at_k(recs, gt_items, k)
        n = ndcg_at_k(recs, gt_items, k)
        recalls.append(r)
        ndcgs.append(n)

    if not recalls:
        return 0.0, 0.0
    return float(sum(recalls) / len(recalls)), float(sum(ndcgs) / len(ndcgs))


def run_retrieval_only(
    train: Dict[int, List[int]],
    eval_split: Dict[int, List[int]],
    retrieval_method: str,
    retrieval_top_k: int,
    k: int,
) -> Tuple[float, float]:
    """Stage 1 only: dùng TwoStagePipeline với rerank_method = none."""
    retrieval_cfg = RetrievalConfig(method=retrieval_method, top_k=retrieval_top_k)
    rerank_cfg = RerankConfig(method="none", top_k=k)
    cfg = PipelineConfig(retrieval=retrieval_cfg, rerank=rerank_cfg)
    pipeline = TwoStagePipeline(cfg)
    pipeline.fit(train)

    users = sorted(eval_split.keys())
    recs_by_user: Dict[int, List[int]] = {}
    for u in users:
        recs_by_user[u] = pipeline.recommend(u)

    return evaluate_users(users, recs_by_user, eval_split, k)


def run_full_pipeline(
    train: Dict[int, List[int]],
    eval_split: Dict[int, List[int]],
    retrieval_method: str,
    retrieval_top_k: int,
    rerank_method: str,
    rerank_top_k: int,
    k: int,
) -> Tuple[float, float]:
    """Stage 1 + Stage 2: dùng TwoStagePipeline bình thường."""
    retrieval_cfg = RetrievalConfig(method=retrieval_method, top_k=retrieval_top_k)
    rerank_cfg = RerankConfig(method=rerank_method, top_k=rerank_top_k)
    cfg = PipelineConfig(retrieval=retrieval_cfg, rerank=rerank_cfg)
    pipeline = TwoStagePipeline(cfg)
    pipeline.fit(train)

    users = sorted(eval_split.keys())
    recs_by_user: Dict[int, List[int]] = {}
    for u in users:
        recs_by_user[u] = pipeline.recommend(u)

    return evaluate_users(users, recs_by_user, eval_split, k)


def run_rerank_only(
    train: Dict[int, List[int]],
    eval_split: Dict[int, List[int]],
    retrieval_method: str,
    retrieval_top_k: int,
    rerank_method: str,
    rerank_top_k: int,
    k: int,
) -> Tuple[float, float]:
    """Stage 2 only: dùng Stage 1 để tạo candidates + inject ground truth.

    - Stage 1 chỉ để tạo pool candidates (top-K Stage 1).
    - Sau đó union với ground_truth để đảm bảo reranker luôn nhìn thấy item đúng.
    - Metrics đo trên ranking cuối của Stage 2.
    """
    RetrieverCls = get_retriever_class(retrieval_method)
    RerankerCls = get_reranker_class(rerank_method)

    retriever = RetrieverCls(top_k=retrieval_top_k)
    reranker = RerankerCls(top_k=rerank_top_k)

    retriever.fit(train)
    reranker.fit(train)

    users = sorted(eval_split.keys())
    recs_by_user: Dict[int, List[int]] = {}

    for u in users:
        gt_items = eval_split.get(u, [])
        base_cands = retriever.retrieve(u)
        # Union base candidates + ground truth, giữ thứ tự, loại trùng
        seen = set()
        merged: List[int] = []
        for item_id in base_cands + gt_items:
            if item_id not in seen:
                seen.add(item_id)
                merged.append(item_id)

        if not merged:
            recs_by_user[u] = []
            continue

        scored = reranker.rerank(u, merged)
        ranked_items = [item_id for item_id, _ in scored]
        recs_by_user[u] = ranked_items

    return evaluate_users(users, recs_by_user, eval_split, k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation for two-stage recommendation")

    # Dataset / filtering (cần khớp với data_prepare.py)
    parser.add_argument("--dataset_code", type=str, default="beauty")
    parser.add_argument("--min_rating", type=int, default=3)
    parser.add_argument("--min_uc", type=int, default=5)
    parser.add_argument("--min_sc", type=int, default=5)

    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dùng val hay test để eval")
    parser.add_argument("--K", type=int, default=50, help="Cutoff cho Recall@K, NDCG@K")

    parser.add_argument("--mode", type=str, default="full", choices=["retrieval", "full", "rerank_only"], help="Chế độ eval")

    # Retrieval config
    parser.add_argument("--retrieval_method", type=str, default="lrurec")
    parser.add_argument("--retrieval_top_k", type=int, default=200)

    # Rerank config
    parser.add_argument("--rerank_method", type=str, default="identity")
    parser.add_argument("--rerank_top_k", type=int, default=50)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_dataset(
        dataset_code=args.dataset_code,
        min_rating=args.min_rating,
        min_uc=args.min_uc,
        min_sc=args.min_sc,
    )

    train: Dict[int, List[int]] = dataset["train"]
    val: Dict[int, List[int]] = dataset["val"]
    test: Dict[int, List[int]] = dataset["test"]

    eval_split = val if args.split == "val" else test

    if args.mode == "retrieval":
        recall, ndcg = run_retrieval_only(
            train=train,
            eval_split=eval_split,
            retrieval_method=args.retrieval_method,
            retrieval_top_k=args.retrieval_top_k,
            k=args.K,
        )
        mode_name = "Stage 1 only (retrieval)"

    elif args.mode == "full":
        recall, ndcg = run_full_pipeline(
            train=train,
            eval_split=eval_split,
            retrieval_method=args.retrieval_method,
            retrieval_top_k=args.retrieval_top_k,
            rerank_method=args.rerank_method,
            rerank_top_k=args.rerank_top_k,
            k=args.K,
        )
        mode_name = "Full pipeline (Stage 1 + Stage 2)"

    else:  # rerank_only
        recall, ndcg = run_rerank_only(
            train=train,
            eval_split=eval_split,
            retrieval_method=args.retrieval_method,
            retrieval_top_k=args.retrieval_top_k,
            rerank_method=args.rerank_method,
            rerank_top_k=args.rerank_top_k,
            k=args.K,
        )
        mode_name = "Stage 2 only (rerank-only với pool từ Stage 1 + ground truth)"

    print("=" * 80)
    print(f"Mode      : {mode_name}")
    print(f"Dataset   : {args.dataset_code}")
    print(f"Split     : {args.split}")
    print(f"K         : {args.K}")
    print(f"Retriever : {args.retrieval_method} (top_k={args.retrieval_top_k})")
    print(f"Reranker  : {args.rerank_method} (top_k={args.rerank_top_k})")
    print("-" * 80)
    print(f"Recall@{args.K}: {recall:.4f}")
    print(f"NDCG@{args.K}  : {ndcg:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
