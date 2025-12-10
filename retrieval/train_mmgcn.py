import os
import sys
import importlib.util
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from pytorch_lightning import seed_everything

# Load trực tiếp config.py ở project root để tránh đụng retrieval/config.py
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.py")
spec = importlib.util.spec_from_file_location("root_config", CONFIG_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load root config from {CONFIG_PATH}")
_config = importlib.util.module_from_spec(spec)
sys.modules["root_config"] = _config
spec.loader.exec_module(_config)

arg = _config.arg
EXPERIMENT_ROOT = _config.EXPERIMENT_ROOT
from datasets import dataset_factory
from evaluation.metrics import recall_at_k, ndcg_at_k
from retrieval.registry import get_retriever_class


RETRIEVAL_METHOD = "mmgcn"
RETRIEVAL_TOP_K = 200  # how many items to retrieve per user
METRIC_K = 10          # cutoff for Recall@K, NDCG@K
METRIC_KS_FOR_RETRIEVED = [1, 5, 10, 20, 50]


def _evaluate_split(
    retriever,
    split: Dict[int, List[int]],
    k: int,
) -> Dict[str, float]:
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
    train = data["train"]  # Dict[int, List[int]]

    # Mỗi user_id -> tập các item đã tương tác
    user_item_dict = {u: set(items) for u, items in train.items()}

    val: Dict[int, List[int]] = data["val"]
    test: Dict[int, List[int]] = data["test"]
    item_count: int = len(data["smap"])
    user_count: int = len(data["umap"])

    # Load CLIP embeddings (visual + text) từ thư mục preprocessed giống các script khác
    preprocessed_folder = dataset._get_preprocessed_folder_path()
    clip_path = preprocessed_folder / "clip_embeddings.pt"
    payload = torch.load(clip_path)
    # Bỏ index 0 (padding) để chỉ giữ đúng num_item embedding
    v_feat = payload["image_embs"][1:]
    t_feat = payload["text_embs"][1:]

    user_ids = []
    item_ids = []

    for u, items in train.items():
        for i in items:
            # User node: 0..user_count-1, Item node: user_count..user_count+item_count-1
            user_ids.append(u - 1)
            item_ids.append(user_count + i - 1)

    train_edge = torch.tensor([user_ids, item_ids], dtype=torch.long)
    

    # 3) Build & fit retriever
    RetrieverCls = get_retriever_class(RETRIEVAL_METHOD)
    retriever = RetrieverCls(
        top_k=RETRIEVAL_TOP_K,
        num_epochs=arg.retrieval_epochs,
        batch_size=arg.batch_size_retrieval,
        num_workers=arg.num_workers_retrieval,
    )

    # MMGCNRetriever cần biết tổng số user, item và graph + CLIP features.
    retriever.fit(
        train,
        item_count=item_count,
        user_count=user_count,
        train_edge=train_edge,
        v_feat=v_feat,
        t_feat=t_feat,
        user_item_dict=user_item_dict,
    )

    # 4) Evaluate on val / test (giống flow của LRURec)
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
