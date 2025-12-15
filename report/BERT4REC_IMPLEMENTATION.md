# BERT4Rec Implementation Guide

## Overview

BERT4Rec (Bidirectional Encoder Representations for Sequential Recommendation) đã được tích hợp vào hệ thống rerank. Implementation dựa trên:
- Paper: "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer" (Sun et al., CIKM'19)
- Reference implementation: https://github.com/FeiSun/BERT4Rec

## Architecture

BERT4Rec sử dụng bidirectional BERT encoder để học sequential patterns:

```
Input: [item_1, item_2, ..., item_n, candidate]
       ↓
BERT Encoder (bidirectional attention)
       ↓
Output: scores for candidate
```

**Key Components**:
- **BertEmbedding**: Item embedding + position embedding + token type embedding
- **BertEncoder**: Stack of transformer layers với bidirectional attention
- **Masked Language Modeling**: Train bằng cách mask một số items và predict chúng

## Files Added

1. **`rerank/models/bert4rec.py`**: BERT4Rec model implementation
   - `BertEmbedding`: Embedding layer
   - `BertSelfAttention`: Multi-head self-attention
   - `BertLayer`: Single transformer layer
   - `BertEncoder`: Stack of transformer layers
   - `BERT4Rec`: Main model với forward và predict_scores methods
   - Bám sát architecture từ paper và reference implementation

2. **`rerank/methods/bert4rec_reranker.py`**: BERT4RecReranker wrapper
   - Implement `BaseReranker` interface
   - Handle training với masked language modeling (MLM) loss
   - Support early stopping và validation
   - Efficient batch processing cho reranking

3. **`rerank/registry.py`**: Đăng ký BERT4RecReranker
   - Thêm `"bert4rec": BERT4RecReranker` vào registry

4. **`rerank/models/__init__.py`**: Export BERT4Rec model

5. **`scripts/train_pipeline.py`**: Support BERT4Rec reranker
   - Thêm `"bert4rec"` vào help text

## Usage

### Basic Usage

```python
from retrieval.registry import get_retriever_class
from rerank.registry import get_reranker_class

# Create retriever
RetrieverCls = get_retriever_class("lrurec")
retriever = RetrieverCls(top_k=200)
retriever.fit(train_data, item_count=num_items)

# Create reranker
RerankerCls = get_reranker_class("bert4rec")
reranker = RerankerCls(
    top_k=50,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    max_seq_length=200,
    num_epochs=10,
    lr=1e-4,
)

# Train
reranker.fit(
    train_data,  # Sequential interactions
    vocab_size=num_items + 1,  # +1 for padding token
    val_data=val_data,
)

# Rerank
candidates = retriever.retrieve(user_id=1)
ranked = reranker.rerank(user_id=1, candidates=candidates)
```

### Using with Pipeline

```python
from pipelines.base import PipelineConfig, RetrievalConfig, RerankConfig, TwoStagePipeline

retrieval_cfg = RetrievalConfig(
    method="lrurec",
    top_k=200
)
rerank_cfg = RerankConfig(
    method="bert4rec",
    top_k=50
)

pipeline_cfg = PipelineConfig(
    retrieval=retrieval_cfg,
    rerank=rerank_cfg
)

pipeline = TwoStagePipeline(pipeline_cfg)
pipeline.fit(
    train_data,
    retriever_kwargs={"item_count": num_items},
    reranker_kwargs={
        "vocab_size": num_items + 1,
        "val_data": val_data,
    }
)
```

### Command Line

```bash
# Train BERT4Rec reranker
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method bert4rec \
    --rerank_top_k 50
```

## Requirements

1. **Sequential Data**: BERT4Rec cần sequential interaction data
   - `train_data`: Dict {user_id: [item_ids]} - items theo thứ tự thời gian
   - Model học patterns từ sequence order

2. **Dependencies**: 
   - PyTorch
   - NumPy

## Hyperparameters

Default values (theo paper và reference implementation):

- `hidden_size`: 64 (embedding dimension)
- `num_hidden_layers`: 2 (number of transformer layers)
- `num_attention_heads`: 2 (attention heads)
- `intermediate_size`: 256 (feed-forward size)
- `max_seq_length`: 200 (maximum sequence length)
- `attention_probs_dropout_prob`: 0.2
- `hidden_dropout_prob`: 0.2
- `hidden_act`: "gelu" (activation function)
- `lr`: 1e-4 (learning rate)
- `num_warmup_steps`: 100 (warmup steps for LR scheduler)
- `batch_size`: 32
- `num_epochs`: 10

Có thể điều chỉnh trong `BERT4RecReranker.__init__()` hoặc qua kwargs.

## Training Process

1. **Masked Language Modeling (MLM)**:
   - Mask 15% of items in sequence
   - 80% replace with [MASK] token
   - 10% keep original
   - 10% replace with random item
   - Predict masked items

2. **Loss**: Cross-entropy loss cho masked positions

3. **Optimization**: Adam với learning rate 1e-4 và warmup

4. **Early Stopping**: Optional, dựa trên validation recall@10

## Reranking Process

1. **Input**: User history + candidates từ retrieval stage
2. **For each candidate**: Create sequence [history, candidate]
3. **Forward pass**: BERT encoder với bidirectional attention
4. **Score**: Extract logit for candidate position
5. **Rank**: Sort candidates by scores

## Differences from Reference Implementation

1. **Framework**:
   - Reference: TensorFlow 1.12
   - This implementation: PyTorch (modern, easier to integrate)

2. **Training**:
   - Reference: Custom training loop với TFRecord data
   - This implementation: Standard PyTorch training với in-memory data

3. **Reranking Adaptation**:
   - Reference: Designed for next-item prediction
   - This implementation: Adapted for reranking task (history + candidates)

4. **Data Format**:
   - Reference: TFRecord format
   - This implementation: In-memory Python dicts (tương thích với pipeline)

## Performance

Expected performance (theo paper):
- Sequential recommendation: Competitive với SOTA sequential models
- Reranking: Should improve over retrieval-only baseline

## Notes

- BERT4Rec là **sequential model**, cần user history để predict
- Model sử dụng **bidirectional attention**, có thể nhìn cả past và future trong sequence
- Training sử dụng **masked language modeling**, tương tự BERT
- Item IDs được sử dụng trực tiếp (không cần text/image features)
- Padding token = 0, [MASK] token = vocab_size - 1

## Troubleshooting

1. **Out of memory với nhiều candidates**:
   - `predict_scores` đã được optimize để batch process tất cả candidates
   - Nếu vẫn OOM, có thể process candidates in chunks

2. **Poor performance**:
   - Đảm bảo sequential data có thứ tự đúng (theo timestamp)
   - Tăng `max_seq_length` nếu user history dài
   - Điều chỉnh `num_hidden_layers` và `hidden_size`

3. **Slow training**:
   - Giảm `batch_size`
   - Giảm `max_seq_length`
   - Giảm `num_hidden_layers`

4. **No history users**:
   - Model trả về candidates với uniform scores nếu không có history
   - Có thể fallback về retrieval scores

