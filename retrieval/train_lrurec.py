import os
import pickle
from pathlib import Path
from typing import Dict, List

import torch
from pytorch_lightning import seed_everything
import torch.nn.functional as F
from config import arg, EXPERIMENT_ROOT
from datasets import dataset_factory
from evaluation.metrics import recall_at_k, ndcg_at_k
from retrieval.registry import get_retriever_class
import pandas as pd
import json


RETRIEVAL_METHOD = "lrurec"
RETRIEVAL_TOP_K = 200  # how many items to retrieve per user
METRIC_K = 10          # cutoff for Recall@K, NDCG@K
METRIC_KS_FOR_RETRIEVED = [1, 5, 10, 20, 50]
RETRIEVAL_SAVE_TOP_K = 20  # how many top candidate scores to store per user


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

    seed_everything(arg.seed)

    dataset = dataset_factory(arg)
    # Require CSV export (produced by `data_prepare.py`) and use it for training
    preproc_folder = Path(dataset._get_preprocessed_folder_path())
    csv_path = preproc_folder.joinpath("dataset_single_export.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Vectorized reconstruction: preserve file order, group by (split,user)
        import numpy as np

        df = df.reset_index(drop=False).rename(columns={"index": "row_order"})

        # Aggregate item lists per (split, user) preserving original row order
        grouped = (
            df.sort_values("row_order")
              .groupby(["split", "user_id"]) ["item_new_id"]
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

        # Build meta by taking first occurrence per item_new_id
        meta_df = df.drop_duplicates(subset=["item_new_id"]).set_index("item_new_id")
        meta = {}
        for item_new_id, row in meta_df.iterrows():
            text = row.get("item_text") if not pd.isna(row.get("item_text")) else None
            image_path = row.get("item_image_path") if not pd.isna(row.get("item_image_path")) else None
            meta[int(item_new_id)] = {"text": text, "image_path": image_path}

        # Build smap (original_id -> new_id) from rows where Item_id present
        smap = {}
        map_df = df[~df["Item_id"].isna()].drop_duplicates(subset=["Item_id"]).copy()
        for _, row in map_df.iterrows():
            try:
                orig = row["Item_id"]
                new = int(row["item_new_id"])
                smap[orig] = new
            except Exception:
                continue

        data = {"train": train, "val": val, "test": test, "meta": meta, "smap": smap}
        item_count = max(meta.keys()) if meta else 0
    else:
        raise FileNotFoundError(
            f"CSV export not found at {csv_path}. Run data_prepare.py to create dataset_single_export.csv"
        )

    RetrieverCls = get_retriever_class(RETRIEVAL_METHOD)
    retriever = RetrieverCls(
        top_k=RETRIEVAL_TOP_K,
        num_epochs=arg.retrieval_epochs,
        batch_size=arg.batch_size_retrieval,
        num_workers=arg.num_workers_retrieval,
        patience=arg.retrieval_patience,
    )
    retriever.fit(train, item_count=item_count, val_data=val)

    # 4) Evaluate on val / test (baseline Stage 1)
    print("\n" + "=" * 80)
    print(f"Load best model state from training: epoch {retriever.best_state}")
    print("=" * 80 + "\n")
    print("Evaluating Stage 1 Retrieval on test set...")
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
    print(f"TEST - users: {test_metrics['num_users']}, Recall@{METRIC_K}: {test_metrics['recall']:.4f}, NDCG@{METRIC_K}: {test_metrics['ndcg']:.4f}")
    print("=" * 80)

    # 5) Build retrieved.pkl cho Stage 2 (LlamaRec-compatible format)
    export_root = Path(EXPERIMENT_ROOT) / "retrieval" / RETRIEVAL_METHOD / arg.dataset_code / f"seed{arg.seed}"
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
    csv_out = export_root / "retrieved.csv"
    df_out.to_csv(csv_out, index=False)

    metrics = {
        "val_metrics": val_metrics_full,
        "test_metrics": test_metrics_full,
    }
    metrics_out = export_root / "retrieved_metrics.json"
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved retrieved candidates to: {csv_out}")
    print(f"Saved retrieved metrics to: {metrics_out}")


if __name__ == "__main__":
    main()
