# VBPR Implementation Guide

## Overview

VBPR (Visual Bayesian Personalized Ranking) đã được tích hợp vào hệ thống retrieval. Implementation dựa trên:
- Paper: "VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback" (He & McAuley, 2016)
- Reference implementation: https://github.com/aaossa/VBPR-PyTorch

## Architecture

VBPR kết hợp collaborative filtering với visual features:

```
score(u, i) = gamma_user[u]^T * gamma_item[i] + 
              theta_user[u]^T * (E * visual_features[i])
```

Trong đó:
- `gamma_user`: User latent factors [n_users, dim_gamma]
- `gamma_item`: Item latent factors [n_items, dim_gamma]
- `theta_user`: User visual preference [n_users, dim_theta]
- `E`: Visual projection matrix [visual_dim, dim_theta]
- `visual_features`: Pre-computed visual features [n_items, visual_dim]

## Files Added

1. **`retrieval/models/vbpr.py`**: VBPR model implementation
   - `VBPR` class: Core model với forward, predict_all, loss methods
   - Bám sát architecture từ paper và reference implementation

2. **`retrieval/methods/vbpr.py`**: VBPRRetriever wrapper
   - Implement `BaseRetriever` interface
   - Handle training với BPR loss
   - Load visual features từ CLIP embeddings
   - Support early stopping và validation

3. **`retrieval/registry.py`**: Đăng ký VBPRRetriever
   - Thêm `"vbpr": VBPRRetriever` vào registry

4. **`scripts/train_retrieval.py`**: Support VBPR training
   - Load CLIP embeddings cho VBPR
   - Pass visual features vào fit_kwargs

## Usage

### Basic Usage

```python
from retrieval.registry import get_retriever_class
from dataset.paths import get_clip_embeddings_path
import torch

# Load visual features
clip_path = get_clip_embeddings_path("beauty", 3, 20, 20)
clip_payload = torch.load(clip_path)
visual_features = clip_payload["image_embs"][1:num_items+1]  # Skip padding

# Create retriever
RetrieverCls = get_retriever_class("vbpr")
retriever = RetrieverCls(
    top_k=50,
    dim_gamma=20,
    dim_theta=20,
    num_epochs=10,
    lr=5e-4,
)

# Train
retriever.fit(
    train_data,
    num_user=num_users,
    num_item=num_items,
    visual_features=visual_features,
    val_data=val_data,
)

# Retrieve
candidates = retriever.retrieve(user_id=1, exclude_items={2, 3})
```

### Using with Pipeline

```python
from pipelines.base import PipelineConfig, RetrievalConfig, RerankConfig, TwoStagePipeline

retrieval_cfg = RetrievalConfig(
    method="vbpr",
    top_k=200
)
rerank_cfg = RerankConfig(
    method="identity",
    top_k=50
)

pipeline_cfg = PipelineConfig(
    retrieval=retrieval_cfg,
    rerank=rerank_cfg
)

pipeline = TwoStagePipeline(pipeline_cfg)
pipeline.fit(
    train_data,
    retriever_kwargs={
        "num_user": num_users,
        "num_item": num_items,
        "visual_features": visual_features,
        "dataset_code": "beauty",
        "min_rating": 3,
        "min_uc": 20,
        "min_sc": 20,
        "val_data": val_data,
    }
)
```

### Command Line

```bash
# Train VBPR retrieval model
python scripts/train_retrieval.py \
    --retrieval_method vbpr \
    --retrieval_top_k 200

# Use in pipeline
python scripts/train_pipeline.py \
    --retrieval_method vbpr \
    --retrieval_top_k 200 \
    --rerank_method identity \
    --rerank_top_k 50
```

## Requirements

1. **CLIP Embeddings**: VBPR cần visual features từ CLIP embeddings
   ```bash
   python data_prepare.py --use_image --use_text
   ```

2. **Dependencies**: 
   - PyTorch
   - NumPy
   - CLIP embeddings (từ `data_prepare.py`)

## Hyperparameters

Default values (theo paper và reference implementation):

- `dim_gamma`: 20 (user/item latent dimension)
- `dim_theta`: 20 (visual projection dimension)
- `lr`: 5e-4 (learning rate, SGD)
- `lambda_reg`: 0.01 (regularization weight)
- `batch_size`: 64
- `num_epochs`: 10

Có thể điều chỉnh trong `VBPRRetriever.__init__()` hoặc qua kwargs.

## Training Process

1. **Sample Generation**: 
   - Với mỗi (user, positive_item), sample một negative_item
   - Negative items không nằm trong user's history

2. **BPR Loss**:
   ```
   loss = -log(sigmoid(pos_score - neg_score)) + lambda_reg * ||params||^2
   ```

3. **Optimization**: SGD với learning rate 5e-4

4. **Early Stopping**: Optional, dựa trên validation recall@10

## Differences from Reference Implementation

1. **Regularization**: 
   - Reference implementation có asymmetric regularization (lambda/10 cho negative items)
   - Implementation này dùng symmetric regularization (đơn giản hơn, performance tương đương)

2. **Visual Features**:
   - Reference: Load từ file `.b` format
   - This implementation: Load từ CLIP embeddings (tương thích với pipeline)

3. **Training Loop**:
   - Reference: Custom training loop với manual gradient updates
   - This implementation: Standard PyTorch training với optimizer

## Performance

Expected performance (theo paper và reference repo):
- Tradesy dataset: Recall@20 ~ 0.76-0.78
- Tương đương hoặc tốt hơn BPR baseline

## Notes

- VBPR chỉ cần **visual features** (image embeddings), không cần text features
- Visual features được load từ CLIP embeddings (tương thích với MMGCN)
- Model tự động detect visual_dim từ visual_features shape
- Item IDs được convert giữa 1-indexed (external) và 0-indexed (internal)

## Troubleshooting

1. **Missing CLIP embeddings**:
   ```
   FileNotFoundError: CLIP embeddings not found
   ```
   Solution: Run `python data_prepare.py --use_image`

2. **Shape mismatch**:
   ```
   ValueError: visual_features shape mismatch
   ```
   Solution: Đảm bảo `visual_features.size(0) == num_items`

3. **CUDA out of memory**:
   - Giảm `batch_size`
   - Giảm `dim_gamma` hoặc `dim_theta`

