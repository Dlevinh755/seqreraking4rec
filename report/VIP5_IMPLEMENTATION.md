# VIP5 Implementation - Following Original Source Code

## Tổng quan

Đã tích hợp VIP5 (Visual Item Preference) reranker theo sát source code gốc từ repository: https://github.com/jeykigung/VIP5

## Cấu trúc Implementation

### 1. `rerank/models/vip5_modeling.py`
**Complete VIP5 model implementation từ original repo**

- Copy toàn bộ `modeling_vip5.py` từ VIP5 repo
- Chỉ update imports để dùng relative imports cho adapters
- Bao gồm:
  - `MLP`: Multi-layer perceptron
  - `VisualEmbedding`: Visual feature encoding
  - `T5LayerFF`, `T5Attention`, `T5LayerSelfAttention`, `T5LayerCrossAttention`
  - `T5Block`, `T5Stack`
  - `JointEncoder`: Encoder kết hợp text và visual
  - `VIP5`: Main model class (T5-based seq2seq)
  - `VIP5Seq2SeqLMOutput`: Output dataclass

### 2. `rerank/models/vip5_utils.py`
**Utility functions để prepare input cho VIP5**

- `prepare_vip5_input()`: Prepare input theo format VIP5 (input_ids, whole_word_ids, category_ids, vis_feats)
- `calculate_whole_word_ids()`: Calculate whole word IDs cho whole word embeddings
- `build_rerank_prompt()`: Build prompt theo VIP5 templates

### 3. `rerank/methods/vip5_reranker.py`
**VIP5Reranker wrapper implement BaseReranker interface**

- Sử dụng VIP5 model từ `vip5_modeling.py`
- Load CLIP embeddings
- Prepare input theo đúng format VIP5
- Encode với VIP5 encoder để rerank

### 4. `rerank/models/adapters/`
**Adapters module từ VIP5 (cần copy từ original repo)**

- Cần copy toàn bộ folder `adapters/` từ `VIP5/src/adapters/` vào `rerank/models/adapters/`
- Bao gồm: adapter_controller, adapter_modeling, adapter_configuration, etc.

## Setup

### Bước 1: Clone VIP5 repository

```bash
git clone https://github.com/jeykigung/VIP5.git
```

### Bước 2: Copy VIP5 source code

```bash
# Copy modeling file (đã được copy trong vip5_modeling.py)
# Copy adapters module
cp -r VIP5/src/adapters rerank/models/adapters

# Copy tokenization (optional, có thể dùng T5Tokenizer)
cp VIP5/src/tokenization.py rerank/models/
```

### Bước 3: Install dependencies

```bash
pip install torch transformers tqdm numpy sentencepiece pyyaml
pip install git+https://github.com/openai/CLIP.git
```

## Cách sử dụng

### Basic Usage

```python
from rerank.registry import get_reranker_class

# Get VIP5 reranker
RerankerCls = get_reranker_class("vip5")
reranker = RerankerCls(
    top_k=50,
    checkpoint_path="path/to/vip5/checkpoint.pth",  # Optional
    backbone="t5-small",  # or "t5-base", "t5-large"
    image_feature_type="vitb32",  # vitb32, vitb16, vitl14, rn50, rn101
    image_feature_size_ratio=2,  # Number of visual tokens per item
)

# Fit reranker
reranker.fit(
    train_data,
    dataset_code="beauty",
    min_rating=3,
    min_uc=20,
    min_sc=20,
    num_items=item_count,
    item_id2text=item_id2text,  # Optional: for better tokenization
)

# Rerank
ranked = reranker.rerank(
    user_id=1,
    candidates=[1, 2, 3, ...],
    user_history=[10, 20, 30],  # Optional: user's interaction history
)
```

### Trong Pipeline

```bash
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method vip5 \
    --rerank_top_k 50
```

## VIP5 Model Architecture

VIP5 là một T5-based sequence-to-sequence model với:

1. **JointEncoder**: 
   - Kết hợp text và visual features
   - Sử dụng `VisualEmbedding` để project CLIP features
   - Whole word embeddings và category embeddings

2. **Input Format**:
   - `input_ids`: Tokenized text
   - `whole_word_ids`: Whole word boundaries
   - `category_ids`: 1 cho visual tokens (`<extra_id_0>`), 0 cho text tokens
   - `vis_feats`: Visual features [B, V_W_L, feat_dim]

3. **Forward Pass**:
   - Encode với JointEncoder
   - Decode với T5Stack decoder
   - Generate hoặc compute scores

## Reranking Strategy

Hiện tại implementation sử dụng:

1. Encode user history + từng candidate riêng lẻ
2. Sử dụng encoder output (mean pooling) để tính scores
3. Rank candidates theo scores

**Note**: Đây là simplified approach. Full implementation có thể:
- Sử dụng decoder để generate và score
- Batch encode multiple candidates
- Sử dụng VIP5's generation mechanism

## Checkpoint Loading

VIP5 checkpoints từ original repo có format:

```python
checkpoint = {
    "config": T5Config,
    "state_dict": model.state_dict(),
    # hoặc
    "model": model.state_dict(),
}
```

VIP5Reranker tự động detect và load checkpoint format.

## Dependencies

### Required:
- PyTorch
- transformers (T5)
- CLIP embeddings (từ data_prepare.py)

### Optional (cho full VIP5):
- P5Tokenizer (từ VIP5 repo) - fallback to T5Tokenizer nếu không có
- Adapters module (từ VIP5 repo)

## Lưu ý quan trọng

1. **Adapters Module**: Cần copy `adapters/` từ VIP5 repo vào `rerank/models/adapters/`
   - Nếu không có, VIP5 sẽ không hoạt động với `use_adapter=True`

2. **Tokenizer**: 
   - Ưu tiên dùng `P5Tokenizer` từ VIP5 repo (hỗ trợ `<user_id_X>`, `<item_id_X>`)
   - Fallback to `T5Tokenizer` nếu không có

3. **Checkpoint Format**: 
   - Cần đảm bảo checkpoint format tương thích
   - Có thể cần điều chỉnh loading logic tùy version

4. **Input Format**:
   - VIP5 cần đúng format: whole_word_ids, category_ids, vis_feats
   - Visual features phải match với số lượng visual tokens trong text

## Troubleshooting

### Lỗi: "AdapterController not available"

**Giải pháp**: Copy `adapters/` từ VIP5 repo:
```bash
cp -r VIP5/src/adapters rerank/models/adapters
```

### Lỗi: "P5Tokenizer not found"

**Giải pháp**: Sử dụng T5Tokenizer (đã có fallback) hoặc copy `tokenization.py` từ VIP5 repo

### Lỗi: "Checkpoint format không tương thích"

**Giải pháp**: Kiểm tra checkpoint format và điều chỉnh loading logic trong `VIP5Reranker.fit()`

## Next Steps

Để hoàn thiện implementation:

1. [ ] Copy adapters module từ VIP5 repo
2. [ ] Implement batch encoding cho multiple candidates
3. [ ] Sử dụng VIP5 decoder để generate và score (thay vì chỉ encoder)
4. [ ] Support đầy đủ VIP5 templates và tasks
5. [ ] Optimize inference performance

