# VIP5 Reranker Quick Start

## Tổng quan

VIP5 (Visual Item Preference) là một reranker multimodal sử dụng visual và textual features từ CLIP embeddings.

## Cài đặt nhanh

### 1. Chuẩn bị CLIP embeddings

```bash
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image \
    --use_text
```

### 2. Sử dụng VIP5 trong pipeline

```bash
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method vip5 \
    --rerank_top_k 50
```

## Sử dụng trong code

```python
from rerank.registry import get_reranker_class

# Get VIP5 reranker
RerankerCls = get_reranker_class("vip5")
reranker = RerankerCls(top_k=50)

# Fit với dataset info
reranker.fit(
    train_data,
    dataset_code="beauty",
    min_rating=3,
    min_uc=20,
    min_sc=20,
    num_items=item_count,
)

# Rerank
ranked = reranker.rerank(user_id=1, candidates=[1, 2, 3, ...])
```

## Yêu cầu

- CLIP embeddings (cả image và text)
- PyTorch
- NumPy

Xem `VIP5_INTEGRATION.md` để biết thêm chi tiết.

