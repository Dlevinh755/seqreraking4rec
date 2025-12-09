"""Stage 1 training & evaluation script for LRURec retrieval.

Features:
- Uses the existing preprocessing pipeline (dataset.pkl via `datasets` + `config`).
- Uses the simple `LRURecRetriever` (reverse-chronological heuristic) as Stage 1 model.
- Computes Recall@K and NDCG@K on val / test splits.
- Generates a `retrieved.pkl` file (LlamaRec-compatible) containing per-item scores
    for val/test users, so Stage 2 rerankers (LLM, VIP4, ...) có thể sử dụng lại
    mà không phải chạy lại Stage 1.

Lưu ý: đây là bản đơn giản, không phải bản LRURec neural đầy đủ của LlamaRec,
nhưng format của `retrieved.pkl` giống với `LlamaRec/trainer/lru.py`.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List

import torch
from pytorch_lightning import seed_everything

from config import arg, EXPERIMENT_ROOT
from datasets import dataset_factory
from evaluation.metrics import recall_at_k, ndcg_at_k
from retrieval.registry import get_retriever_class


RETRIEVAL_METHOD = "lrurec"
RETRIEVAL_TOP_K = 200  # how many items to retrieve per user
METRIC_K = 10          # cutoff for Recall@K, NDCG@K
METRIC_KS_FOR_RETRIEVED = [1, 5, 10, 20, 50]


def _evaluate_split(
    retriever,
    split: Dict[int, List[int]],
    k: int,
) -> Dict[str, float]:
    """Compute average Recall@K and NDCG@K for a given split.

    Args:
        retriever: fitted retriever with `retrieve(user_id)` method.
        split: dict {user_id: [item_ids]} for val or test.
        k: cutoff K.
    """
    users = sorted(split.keys())
    recalls, ndcgs = [], []

    for u in users:
        gt_items = split.get(u, [])
        if not gt_items:
            continue
        recs = retriever.retrieve(u)
        if not recs:
            continue
        r = recall_at_k(recs, gt_items, k)
        n = ndcg_at_k(recs, gt_items, k)
        recalls.append(r)
        ndcgs.append(n)

    if not recalls:
        return {"recall": 0.0, "ndcg": 0.0, "num_users": 0}

    return {
        "recall": float(sum(recalls) / len(recalls)),
        "ndcg": float(sum(ndcgs) / len(ndcgs)),
        "num_users": len(recalls),
    }


def absolute_recall_mrr_ndcg_for_ks(scores: torch.Tensor, labels: torch.Tensor, ks) -> Dict[str, float]:
    """Compute Recall, MRR, NDCG cho nhiều K (giống LlamaRec.trainer.utils).

    Args:
        scores: tensor [B, N] điểm cho từng item.
        labels: tensor [B] chứa item_id đúng.
        ks: iterable các giá trị K.
    """
    import torch.nn.functional as F

    metrics: Dict[str, float] = {}
    one_hot = F.one_hot(labels, num_classes=scores.size(1))
    answer_count = one_hot.sum(1)

    labels_float = one_hot.float()
    rank = (-scores).argsort(dim=1)

    cut = rank
    device = scores.device
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)

        # Recall@K
        denom = torch.min(torch.tensor([k], device=device), labels_float.sum(1).float())
        metrics[f"Recall@{k}"] = (hits.sum(1) / denom).mean().cpu().item()

        # MRR@K
        positions = torch.arange(1, k + 1, device=device).unsqueeze(0)
        metrics[f"MRR@{k}"] = (hits / positions).sum(1).mean().cpu().item()

        # NDCG@K
        position = torch.arange(2, 2 + k, device=device)
        weights = 1.0 / torch.log2(position.float())
        dcg = (hits * weights.to(device)).sum(1)
        idcg = torch.tensor([
            weights[: int(min(int(n.item()), k))].sum().item() for n in answer_count
        ], device=device)
        ndcg = (dcg / idcg).mean()
        metrics[f"NDCG@{k}"] = ndcg.cpu().item()

    return metrics


def _build_retrieved_matrices(
    retriever,
    split: Dict[int, List[int]],
    item_count: int,
) -> Dict[str, List]:
    """Build score matrices & labels cho val/test giống LlamaRec LRUTrainer.

    - Với mỗi user có ground-truth trong split:
      - Lấy danh sách candidates từ retriever (top-K).
      - Tạo vector scores dài (item_count+1):
        * index 0 = -1e9 (padding).
        * các item trong candidates được gán điểm giảm dần theo thứ hạng.
        * các item khác giữ -1e9.
    - Trả về dict với `probs` (list[list[float]]) và `labels` (list[int]).
    """
    users = sorted(split.keys())
    probs: List[List[float]] = []
    labels: List[int] = []

    for u in users:
        gt_items = split.get(u, [])
        if not gt_items:
            continue
        label = gt_items[0]

        cands = retriever.retrieve(u)
        # Khởi tạo tất cả = -1e9, giống cách LlamaRec mask padding/history
        scores = torch.full((item_count + 1,), -1e9, dtype=torch.float32)
        # Gán điểm giảm dần cho candidates (vị trí đầu có score cao nhất)
        for rank, item_id in enumerate(cands):
            if 0 < item_id <= item_count:
                scores[item_id] = float(len(cands) - rank)

        probs.append(scores.tolist())
        labels.append(int(label))

    return {"probs": probs, "labels": labels}


def main() -> None:
    # 1) Seed
    seed_everything(arg.seed)

    # 2) Ensure dataset is preprocessed and load dataset.pkl
    dataset = dataset_factory(arg)
    data = dataset.load_dataset()
    train: Dict[int, List[int]] = data["train"]
    val: Dict[int, List[int]] = data["val"]
    test: Dict[int, List[int]] = data["test"]
    item_count: int = len(data["smap"])

    # 3) Build & fit retriever
    RetrieverCls = get_retriever_class(RETRIEVAL_METHOD)
    retriever = RetrieverCls(top_k=RETRIEVAL_TOP_K)
    # Neural LRURecRetriever cần biết tổng số item để khởi tạo embedding.
    retriever.fit(train, item_count=item_count)

    # 4) Evaluate on val / test (baseline Stage 1)
    val_metrics = _evaluate_split(retriever, val, METRIC_K)
    test_metrics = _evaluate_split(retriever, test, METRIC_K)

    print("=" * 80)
    print(f"Stage 1 Retrieval Evaluation - Method: {RETRIEVAL_METHOD}")
    print(f"Dataset     : {arg.dataset_code}")
    print(f"min_rating  : {arg.min_rating}")
    print(f"min_uc      : {arg.min_uc}")
    print(f"min_sc      : {arg.min_sc}")
    print(f"Retrieval K : {RETRIEVAL_TOP_K}")
    print(f"Metric K    : {METRIC_K}")
    print("-" * 80)
    print(f"VAL  - users: {val_metrics['num_users']}, Recall@{METRIC_K}: {val_metrics['recall']:.4f}, NDCG@{METRIC_K}: {val_metrics['ndcg']:.4f}")
    print(f"TEST - users: {test_metrics['num_users']}, Recall@{METRIC_K}: {test_metrics['recall']:.4f}, NDCG@{METRIC_K}: {test_metrics['ndcg']:.4f}")
    print("=" * 80)

    # 5) Build retrieved.pkl cho Stage 2 (LlamaRec-compatible format)
    export_root = Path(EXPERIMENT_ROOT) / "retrieval" / RETRIEVAL_METHOD / arg.dataset_code / f"seed{arg.seed}"
    export_root.mkdir(parents=True, exist_ok=True)
    retrieved_path = export_root / "retrieved.pkl"

    val_pack = _build_retrieved_matrices(retriever, val, item_count)
    test_pack = _build_retrieved_matrices(retriever, test, item_count)

    if val_pack["probs"] and test_pack["probs"]:
        val_scores = torch.tensor(val_pack["probs"], dtype=torch.float32)
        val_labels = torch.tensor(val_pack["labels"], dtype=torch.long).view(-1)
        test_scores = torch.tensor(test_pack["probs"], dtype=torch.float32)
        test_labels = torch.tensor(test_pack["labels"], dtype=torch.long).view(-1)

        val_metrics_full = absolute_recall_mrr_ndcg_for_ks(val_scores, val_labels, METRIC_KS_FOR_RETRIEVED)
        test_metrics_full = absolute_recall_mrr_ndcg_for_ks(test_scores, test_labels, METRIC_KS_FOR_RETRIEVED)
    else:
        val_metrics_full = {}
        test_metrics_full = {}

    retrieved_payload = {
        "val_probs": val_pack["probs"],
        "val_labels": val_pack["labels"],
        "val_metrics": val_metrics_full,
        "test_probs": test_pack["probs"],
        "test_labels": test_pack["labels"],
        "test_metrics": test_metrics_full,
    }

    with retrieved_path.open("wb") as f:
        pickle.dump(retrieved_payload, f)

    print(f"Saved retrieved candidates to: {retrieved_path}")


if __name__ == "__main__":
    main()
