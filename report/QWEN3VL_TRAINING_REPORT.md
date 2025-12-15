# Qwen3VLReranker Training Report

## Tổng quan

Qwen3VLReranker hỗ trợ training cho **tất cả 4 modes**:
1. `raw_image`: Sử dụng raw images trực tiếp
2. `caption`: Sử dụng image captions
3. `semantic_summary`: Sử dụng semantic summaries với Qwen3-VL
4. `semantic_summary_small`: Sử dụng semantic summaries với model nhỏ hơn (Qwen3-0.6B)

## Training Architecture

### 1. Text Model Training (`semantic_summary_small`)

**Model**: `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`
- **Framework**: Unsloth với LoRA adapters
- **Memory**: ~4-8GB GPU
- **Training**: Parameter-efficient fine-tuning (chỉ train adapters)

**Process**:
1. Load model với Unsloth `FastLanguageModel`
2. Setup LoRA adapters (r=8, alpha=16)
3. Prepare training data: (prompt, target_number)
4. Train với transformers `Trainer`
5. Validation và early stopping

### 2. VL Model Training (`raw_image`, `caption`, `semantic_summary`)

**Model**: `unsloth/Qwen3-VL-2B-Instruct`
- **Framework**: Transformers Trainer
- **Memory**: ~8-16GB GPU
- **Training**: Full model fine-tuning (hoặc có thể dùng LoRA nếu cần)

**Process**:
1. Load Qwen3-VL model với `Qwen3VLForConditionalGeneration`
2. Prepare training data với proper format
3. Custom data collator để handle multimodal inputs
4. Train với transformers `Trainer`
5. Validation và early stopping

**Lưu ý cho `raw_image` mode**:
- Training sử dụng **raw images trực tiếp** (giống như inference)
- Images được load từ `item_meta[item_id]["image_path"]` và đưa vào messages
- Format: messages với content là list chứa `{"type": "image", "image": PIL.Image}` và `{"type": "text", "text": "..."}`

## Data Requirements

### 1. `raw_image` Mode

**Required**:
- `item_meta[item_id]["image_path"]`: Đường dẫn đến image file (bắt buộc cho cả training và inference)
- `item_meta[item_id]["text"]`: Item text/description

**Data Preparation**:
```bash
# Download images (no caption generation needed for raw_image mode)
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image
```

**Note**: `raw_image` mode sử dụng trực tiếp raw images cho cả training và inference, không cần captions.

### 2. `caption` Mode

**Required**:
- `item_meta[item_id]["caption"]`: Image caption
- `item_meta[item_id]["text"]`: Item text/description

**Data Preparation**:
```bash
# Generate captions
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image \
    --generate_caption
```

### 3. `semantic_summary` Mode

**Required**:
- `item_meta[item_id]["semantic_summary"]`: Semantic summary
- `item_meta[item_id]["text"]`: Item text/description

**Data Preparation**:
```bash
# Generate semantic summaries
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image \
    --generate_semantic_summary
```

### 4. `semantic_summary_small` Mode

**Required**:
- `item_meta[item_id]["semantic_summary"]`: Semantic summary
- `item_meta[item_id]["text"]`: Item text/description

**Data Preparation**:
```bash
# Generate semantic summaries
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image \
    --generate_semantic_summary
```

## Training Setup

### Hyperparameters

Tất cả modes sử dụng cùng hyperparameters từ config:

```python
# From config.py
--rerank_epochs: int = 10          # Số epochs
--rerank_batch_size: int = 32      # Batch size
--rerank_lr: float = 1e-4          # Learning rate
--rerank_patience: int = 5         # Early stopping patience
```

### Training Data Format

**Input**: `train_data: Dict[int, List[int]]`
- Key: user_id (1-indexed)
- Value: List of item_ids in chronological order

**Training Samples**:
- History: `items[0:end_pos]` (randomly selected split point)
- Target: `items[end_pos]` (next item)
- Candidates: `[target_item] + 19 random negatives`
- Target label: Index của target trong candidates (1-indexed)

**Example**:
```python
# User history: [1, 2, 3, 4, 5]
# Split at position 3
# History: [1, 2, 3]
# Target: 4
# Candidates: [4, 10, 15, 20, ...] (shuffled)
# Target label: "2" (if target is at position 2 after shuffle)
```

### Training Process

#### Step 1: Prepare Training Samples

```python
def _prepare_training_samples(train_data):
    # For each user:
    # 1. Randomly select split point
    # 2. History = items[:split_point]
    # 3. Target = items[split_point]
    # 4. Candidates = [target] + 19 negatives
    # 5. Shuffle candidates
    # 6. Find target index
    return samples
```

#### Step 2: Build Training Prompts

**Format cho text-only modes** (`caption`, `semantic_summary`, `semantic_summary_small`):
```
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
- item_1 (Image: caption_1)
- item_2 (Image: caption_2)
...

Candidate items:
1. item_10 (Image: caption_10)
2. item_4 (Image: caption_4)  # Target
3. item_15 (Image: caption_15)
...

Answer with only one number (1-20).
```

**Format cho `raw_image` mode**:
Messages với content là list chứa images và text:
```python
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "You are a recommendation ranking assistant.\n\nChoose exactly ONE item..."},
        {"type": "image", "image": PIL.Image},  # Candidate 1 image
        {"type": "text", "text": "1. item_10"},
        {"type": "image", "image": PIL.Image},  # Candidate 2 image
        {"type": "text", "text": "2. item_4"},  # Target
        ...
        {"type": "text", "text": "\nAnswer with only one number (1-20)."}
    ]
}]
```

**Target**: `"2"` (1-indexed position of target item)

#### Step 3: Training Loop

**Text Model** (`semantic_summary_small`):
```python
# 1. Tokenize prompts + targets
# 2. Create labels (same as input_ids for causal LM)
# 3. Train with Unsloth Trainer
# 4. Validation after each epoch
# 5. Early stopping based on val_recall
```

**VL Model** (`raw_image`, `caption`, `semantic_summary`):
```python
# 1. Apply chat template to prompts
# 2. Tokenize targets
# 3. Create labels (-100 for input, target tokens for output)
# 4. Custom collate function for batching
# 5. Train with transformers Trainer
# 6. Validation after each epoch
# 7. Early stopping based on val_recall
```

## Training Examples

### Example 1: Train `semantic_summary_small` Mode

```bash
# 1. Prepare data with semantic summaries
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image \
    --generate_semantic_summary

# 2. Train reranker
python scripts/train_rerank_standalone.py \
    --rerank_method qwen3vl \
    --qwen3vl_mode semantic_summary_small \
    --mode ground_truth \
    --rerank_top_k 50 \
    --rerank_epochs 10 \
    --rerank_batch_size 32 \
    --rerank_lr 1e-4 \
    --rerank_patience 5
```

### Example 2: Train `caption` Mode

```bash
# 1. Prepare data with captions
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image \
    --generate_caption

# 2. Train reranker
python scripts/train_rerank_standalone.py \
    --rerank_method qwen3vl \
    --qwen3vl_mode caption \
    --mode ground_truth \
    --rerank_top_k 50 \
    --rerank_epochs 10 \
    --rerank_batch_size 16 \
    --rerank_lr 1e-4 \
    --rerank_patience 5
```

### Example 3: Train `raw_image` Mode

```bash
# 1. Prepare data with images (no captions needed)
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image

# 2. Train reranker
# Note: Training uses raw images directly (same as inference)
python scripts/train_rerank_standalone.py \
    --rerank_method qwen3vl \
    --qwen3vl_mode raw_image \
    --mode ground_truth \
    --rerank_top_k 50 \
    --rerank_epochs 10 \
    --rerank_batch_size 8 \
    --rerank_lr 1e-4 \
    --rerank_patience 5
```

**Note**: `raw_image` mode sử dụng raw images trực tiếp cho cả training và inference. Images được load từ `image_path` và đưa vào messages với format `{"type": "image", "image": PIL.Image}`.

### Example 4: Train `semantic_summary` Mode

```bash
# 1. Prepare data with semantic summaries
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image \
    --generate_semantic_summary

# 2. Train reranker
python scripts/train_rerank_standalone.py \
    --rerank_method qwen3vl \
    --qwen3vl_mode semantic_summary \
    --mode ground_truth \
    --rerank_top_k 50 \
    --rerank_epochs 10 \
    --rerank_batch_size 16 \
    --rerank_lr 1e-4 \
    --rerank_patience 5
```

## Training Configuration

### Batch Size Recommendations

| Mode | Recommended Batch Size | Memory Usage |
|------|------------------------|--------------|
| `semantic_summary_small` | 32 | ~4-8GB |
| `caption` | 16 | ~8-12GB |
| `semantic_summary` | 16 | ~8-12GB |
| `raw_image` | 8 | ~12-16GB |

**Note**: Batch sizes có thể cần điều chỉnh dựa trên GPU memory available.

### Learning Rate

- **Default**: `1e-4`
- **Range**: `5e-5` đến `2e-4`
- **Recommendation**: 
  - Text model: `1e-4` hoặc `2e-4`
  - VL model: `1e-4` hoặc `5e-5`

### Epochs và Patience

- **Default epochs**: 10
- **Default patience**: 5
- **Recommendation**: 
  - Nếu dataset nhỏ: epochs=20, patience=10
  - Nếu dataset lớn: epochs=5, patience=3

## Validation Process

### Validation Metrics

- **Metric**: Recall@K (K = min(10, top_k))
- **Process**:
  1. For each user in validation set
  2. Get user history from training data
  3. Rerank all items (or sample 100 items for efficiency)
  4. Compute Recall@K
  5. Average across all users

### Early Stopping

- **Criterion**: Validation Recall@K
- **Patience**: Number of epochs without improvement
- **Best model**: Model với highest validation Recall@K

## Training Output

### Console Output

```
[Qwen3VLReranker] Training epoch 1/10...
Training loss: 2.3456
[Qwen3VLReranker] Epoch 1/10 - val_Recall@10: 0.1234

[Qwen3VLReranker] Training epoch 2/10...
Training loss: 1.9876
[Qwen3VLReranker] Epoch 2/10 - val_Recall@10: 0.1456 [BEST]

...

[Qwen3VLReranker] Training epoch 6/10...
Training loss: 1.2345
[Qwen3VLReranker] Epoch 6/10 - val_Recall@10: 0.1456
Early stopping at epoch 6

Loaded best model with val_Recall@10: 0.1456
```

### Saved Checkpoints

- **Text model**: `./qwen3vl_rerank/checkpoint-{step}/`
- **VL model**: `./qwen3vl_rerank_vl/checkpoint-{step}/`

## Troubleshooting

### 1. Out of Memory

**Problem**: GPU out of memory during training

**Solutions**:
- Giảm `--rerank_batch_size`
- Tăng `gradient_accumulation_steps` (trong code)
- Sử dụng `semantic_summary_small` mode (nhẹ hơn)
- Giảm `max_history` trong Qwen3VLReranker

### 2. Training Loss không giảm

**Problem**: Loss không giảm hoặc tăng

**Solutions**:
- Giảm learning rate (`--rerank_lr 5e-5`)
- Tăng batch size (nếu có memory)
- Kiểm tra data quality (captions/semantic summaries có đúng không)
- Kiểm tra training samples có đúng format không

### 3. Validation Recall thấp

**Problem**: Validation Recall@K thấp hoặc không cải thiện

**Solutions**:
- Tăng số epochs
- Tăng patience
- Kiểm tra validation data có đúng không
- Kiểm tra model có được load đúng không

### 4. Missing Dependencies

**Problem**: Import errors

**Solutions**:
```bash
# Install transformers from source (required for Qwen3-VL)
pip install git+https://github.com/huggingface/transformers

# Install unsloth
pip install unsloth[colab-new]

# Install other dependencies
pip install -r requirements.txt
```

## Best Practices

### 1. Data Preparation

- **Luôn generate captions/semantic summaries trước khi train**
- **Kiểm tra data quality**: Đảm bảo captions/semantic summaries có ý nghĩa
- **Filter items**: Chỉ train với items có đầy đủ features

### 2. Hyperparameter Tuning

- **Bắt đầu với defaults**: epochs=10, batch_size=32, lr=1e-4
- **Tune batch size trước**: Tăng/giảm dựa trên GPU memory
- **Tune learning rate**: Nếu loss không giảm, thử giảm lr
- **Tune epochs/patience**: Dựa trên validation performance

### 3. Training Strategy

- **Ground truth mode**: Dùng để train và evaluate rerank quality
- **Retrieval mode**: Dùng để train với candidates từ retrieval stage
- **Validation**: Luôn enable validation để monitor training

### 4. Model Selection

- **Memory constrained**: Dùng `semantic_summary_small` mode
- **Best quality**: Dùng `raw_image` hoặc `semantic_summary` mode
- **Fast training**: Dùng `caption` mode (text-only, nhanh hơn)

## Comparison Table

| Mode | Model | Training | Memory | Speed | Quality |
|------|-------|----------|--------|-------|---------|
| `semantic_summary_small` | Qwen3-0.6B | ✅ Unsloth LoRA | ~4-8GB | ⚡ Fast | ⭐⭐⭐ |
| `caption` | Qwen3-VL-2B | ✅ Transformers | ~8-12GB | 🐢 Medium | ⭐⭐⭐⭐ |
| `semantic_summary` | Qwen3-VL-2B | ✅ Transformers | ~8-12GB | 🐢 Medium | ⭐⭐⭐⭐⭐ |
| `raw_image` | Qwen3-VL-2B | ✅ Transformers* | ~12-16GB | 🐌 Slow | ⭐⭐⭐⭐⭐ |

*Note: `raw_image` training uses captions, but inference uses raw images.

## Summary

Qwen3VLReranker hỗ trợ training cho **tất cả 4 modes**:

1. ✅ **semantic_summary_small**: Text model với Unsloth (nhẹ nhất, nhanh nhất)
2. ✅ **caption**: VL model với text-only training (cân bằng)
3. ✅ **semantic_summary**: VL model với semantic summaries (chất lượng cao)
4. ✅ **raw_image**: VL model với caption-based training (chất lượng cao nhất, inference dùng raw images)

Tất cả modes đều:
- Support validation và early stopping
- Sử dụng hyperparameters từ config
- Fair comparison với các rerank methods khác
- Được fine-tune trên dataset hiện tại

