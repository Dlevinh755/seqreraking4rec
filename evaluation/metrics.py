"""Ranking metrics: Recall@K and NDCG@K."""

from math import log2
from typing import Iterable, List


def recall_at_k(recommended: List[int], ground_truth: Iterable[int], k: int) -> float:
    """Compute Recall@K for a single user.

    Args:
        recommended: ranked list of item_ids.
        ground_truth: iterable of relevant item_ids (e.g., validation/test items).
        k: cutoff.
    """
    if k <= 0:
        return 0.0
    gt = set(ground_truth)
    if not gt:
        return 0.0
    rec_k = recommended[:k]
    hits = len(gt.intersection(rec_k))
    return hits / float(len(gt))


def ndcg_at_k(recommended: List[int], ground_truth: Iterable[int], k: int) -> float:
    """Compute NDCG@K for a single user.

    Uses binary relevance (item is either relevant or not).
    """
    if k <= 0:
        return 0.0
    gt = set(ground_truth)
    if not gt:
        return 0.0

    rec_k = recommended[:k]
    dcg = 0.0
    for idx, item_id in enumerate(rec_k):
        if item_id in gt:
            dcg += 1.0 / log2(idx + 2.0)  # log2(rank+1)

    # Ideal DCG: all relevant items at the top
    ideal_hits = min(len(gt), k)
    idcg = sum(1.0 / log2(i + 2.0) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
