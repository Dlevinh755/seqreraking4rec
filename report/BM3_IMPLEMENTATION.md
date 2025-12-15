# BM3 Implementation Guide

## Overview

BM3 (Bootstrap Latent Representations for Multi-modal Recommendation) đã được tích hợp vào hệ thống retrieval. Implementation dựa trên:
- Paper: "Bootstrap Latent Representations for Multi-Modal Recommendation" (Zhou et al., WWW'23)
- Reference implementation: https://github.com/enoche/BM3

## Architecture

BM3 sử dụng bootstrap mechanism để học latent representations từ multimodal data:

```
score(u, i) = CF_score(u, i) + Multimodal_score(u, i)

CF_score = user_emb[u]^T * item_emb[i]
Multimodal_score = user_emb[u]^T * fused_multimodal_emb[i]

fused_multimodal_emb = MLP(concat(visual_proj, text_proj))
```

Trong đó:
- `user_emb`: User embeddings [n_users, embed_dim]
- `item_emb`: Item embeddings [n_items, embed_dim]
- `visual_proj`: Visual features projected to embed_dim
- `text_proj`: Text features projected to embed_dim
- `MLP`: Multi-layer perceptron để fuse multimodal features (bootstrap mechanism)

## Files Added

1. **`retrieval/models/bm3.py`**: BM3 model implementation
   - `BM3` class: Core model với forward, predict_all, loss methods
   - Bootstrap mechanism: MLP layers để fuse visual và text features
   - Bám sát architecture từ paper và reference implementation

2. **`retrieval/methods/bm3.py`**: BM3Retriever wrapper
   - Implement `BaseRetriever` interface
   - Handle training với BPR loss
   - Load visual và text features từ CLIP embeddings
   - Support early stopping và validation

3. **`retrieval/registry.py`**: Đăng ký BM3Retriever
   - Thêm `"bm3": BM3Retriever` vào registry

4. **`scripts/train_retrieval.py`**: Support BM3 training
   - Load CLIP embeddings cho BM3 (cả visual và text)
   - Pass features vào fit_kwargs

## Usage

### Basic Usage

```python
from retrieval.registry import get_retriever_class
from dataset.paths import get_clip_embeddings_path
import torch

# Load multimodal features
clip_path = get_clip_embeddings_path("beauty", 3, 20, 20)
clip_payload = torch.load(clip_path)
visual_features = clip_payload["image_embs"][1:num_items+1]  # Skip padding
text_features = clip_payload["text_embs"][1:num_items+1]     # Skip padding

# Create retriever
RetrieverCls = get_retriever_class("bm3")
retriever = RetrieverCls(
    top_k=50,
    embed_dim=64,
    layers=1,
    dropout=0.5,
    num_epochs=10,
    lr=1e-3,
    reg_weight=0.1,
)

# Train
retriever.fit(
    train_data,
    num_user=num_users,
    num_item=num_items,
    visual_features=visual_features,
    text_features=text_features,
    val_data=val_data,
)

# Retrieve
candidates = retriever.retrieve(user_id=1, exclude_items={2, 3})
```

### Using with Pipeline

```python
from pipelines.base import PipelineConfig, RetrievalConfig, RerankConfig, TwoStagePipeline

retrieval_cfg = RetrievalConfig(
    method="bm3",
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
        "text_features": text_features,
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
# Train BM3 retrieval model
python scripts/train_retrieval.py \
    --retrieval_method bm3 \
    --retrieval_top_k 200

# Use in pipeline
python scripts/train_pipeline.py \
    --retrieval_method bm3 \
    --retrieval_top_k 200 \
    --rerank_method identity \
    --rerank_top_k 50
```

## Requirements

1. **CLIP Embeddings**: BM3 cần cả visual và text features từ CLIP embeddings
   ```bash
   python data_prepare.py --use_image --use_text
   ```

2. **Dependencies**: 
   - PyTorch
   - NumPy
   - CLIP embeddings (từ `data_prepare.py`)

## Hyperparameters

Default values (theo paper và reference implementation):

- `embed_dim`: 64 (embedding dimension)
- `layers`: 1 (number of MLP layers for fusion)
- `dropout`: 0.5 (dropout rate)
- `lr`: 1e-3 (learning rate, Adam)
- `reg_weight`: 0.1 (regularization weight)
- `batch_size`: 64
- `num_epochs`: 10

**Best hyperparameters** (theo paper, dataset-dependent):

| Dataset | layers | dropout | reg_weight |
|---------|--------|---------|------------|
| Baby    | 1      | 0.5     | 0.1        |
| Sports  | 1      | 0.5     | 0.01       |
| Elec    | 2      | 0.3     | 0.1        |

Có thể điều chỉnh trong `BM3Retriever.__init__()` hoặc qua kwargs.

## Training Process

1. **Sample Generation**: 
   - Với mỗi (user, positive_item), sample một negative_item
   - Negative items không nằm trong user's history

2. **BPR Loss**:
   ```
   loss = -log(sigmoid(pos_score - neg_score)) + lambda_reg * ||params||^2
   ```

3. **Optimization**: Adam với learning rate 1e-3

4. **Early Stopping**: Optional, dựa trên validation recall@10

## Bootstrap Mechanism

BM3 sử dụng bootstrap mechanism để học latent representations:

1. **Project multimodal features** to embedding space:
   - `visual_proj = Linear(visual_dim -> embed_dim)(visual_features)`
   - `text_proj = Linear(text_dim -> embed_dim)(text_features)`

2. **Concatenate** projected features:
   - `multimodal_input = concat(visual_proj, text_proj)`

3. **Fuse through MLP** (bootstrap):
   - `fused_emb = MLP(multimodal_input)`
   - MLP có thể có 1-2 layers tùy dataset

4. **Combine with CF**:
   - `score = CF_score + Multimodal_score`

## Differences from Reference Implementation

1. **Feature Loading**:
   - Reference: Load từ pre-processed files
   - This implementation: Load từ CLIP embeddings (tương thích với pipeline)

2. **Training Loop**:
   - Reference: Custom training loop
   - This implementation: Standard PyTorch training với optimizer

3. **Architecture**:
   - Bám sát paper và reference implementation
   - Bootstrap mechanism được implement qua MLP fusion layers

## Performance

Expected performance (theo paper):
- Baby dataset: Recall@20 ~ competitive với SOTA
- Sports dataset: Recall@20 ~ competitive với SOTA
- Elec dataset: Recall@20 ~ competitive với SOTA

## Notes

- BM3 cần cả **visual và text features** (khác với VBPR chỉ cần visual)
- Features được load từ CLIP embeddings (tương thích với MMGCN)
- Model tự động detect feature dimensions từ input tensors
- Item IDs được convert giữa 1-indexed (external) và 0-indexed (internal)
- Bootstrap mechanism giúp model học better representations từ multimodal data

## Troubleshooting

1. **Missing CLIP embeddings**:
   ```
   FileNotFoundError: CLIP embeddings not found
   ```
   Solution: Run `python data_prepare.py --use_image --use_text`

2. **Shape mismatch**:
   ```
   ValueError: visual_features/text_features shape mismatch
   ```
   Solution: Đảm bảo `features.size(0) == num_items`

3. **CUDA out of memory**:
   - Giảm `batch_size`
   - Giảm `embed_dim` hoặc `layers`

4. **Poor performance**:
   - Thử điều chỉnh hyperparameters theo best settings trong paper
   - Tăng `layers` nếu dataset lớn (như Elec)
   - Điều chỉnh `dropout` và `reg_weight` theo dataset

