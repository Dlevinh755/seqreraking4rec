# VIP5 Reranker Integration

## Tổng quan

Đã tích hợp VIP5 (Visual Item Preference) reranker vào hệ thống. VIP5 là một mô hình multimodal kết hợp visual và textual features từ CLIP embeddings để rerank candidates.

**Reference**: https://github.com/jeykigung/VIP5

## Cấu trúc

### 1. `rerank/models/vip5.py`
- `VIP5Model`: Model architecture cho VIP5
- `load_vip5_checkpoint()`: Load checkpoint từ file
- `predict_preferences()`: Predict user-item preferences

### 2. `rerank/methods/vip5_reranker.py`
- `VIP5Reranker`: Wrapper implement `BaseReranker` interface
- Tự động load CLIP embeddings
- Hỗ trợ load checkpoint hoặc khởi tạo model mới

## Cách sử dụng

### Bước 1: Chuẩn bị CLIP embeddings

VIP5 cần CLIP embeddings cho cả visual và text features:

```bash
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image \
    --use_text
```

### Bước 2: Sử dụng VIP5 trong pipeline

#### Option 1: Sử dụng trong `train_pipeline.py`

```bash
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method vip5 \
    --rerank_top_k 50
```

#### Option 2: Sử dụng trong code

```python
from rerank.registry import get_reranker_class

# Get VIP5 reranker
RerankerCls = get_reranker_class("vip5")
reranker = RerankerCls(
    top_k=50,
    checkpoint_path="path/to/vip5/checkpoint.pt",  # Optional
    visual_dim=512,
    text_dim=512,
    hidden_dim=256,
    num_layers=2,
    dropout=0.1,
)

# Fit reranker
reranker.fit(
    train_data,
    dataset_code="beauty",
    min_rating=3,
    min_uc=20,
    min_sc=20,
    num_items=item_count,
)

# Rerank candidates
ranked = reranker.rerank(user_id=1, candidates=[1, 2, 3, ...])
# Returns: [(item_id, score), ...] sorted by score descending
```

## Parameters

### VIP5Reranker.__init__()

- `top_k` (int): Số lượng items trả về sau rerank (default: 50)
- `checkpoint_path` (str, optional): Đường dẫn đến VIP5 checkpoint
- `visual_dim` (int): Dimension của visual embeddings (default: 512)
- `text_dim` (int): Dimension của text embeddings (default: 512)
- `hidden_dim` (int): Hidden dimension cho fusion layers (default: 256)
- `num_layers` (int): Số layers trong fusion network (default: 2)
- `dropout` (float): Dropout rate (default: 0.1)
- `device` (str, optional): Device để chạy model ("cuda" hoặc "cpu")

### VIP5Reranker.fit()

- `train_data` (Dict[int, List[int]]): Training interactions
- `dataset_code` (str): Dataset code (required)
- `min_rating` (int): Minimum rating threshold (default: 3)
- `min_uc` (int): Minimum user count (default: 20)
- `min_sc` (int): Minimum item count (default: 20)
- `num_items` (int): Number of items (optional, inferred from train_data)

## Model Architecture

VIP5 model architecture:

1. **Input**: Visual embeddings + Text embeddings (từ CLIP)
2. **Projection**: Project visual và text features vào common space
3. **Fusion**: Fuse visual và text features qua MLP layers
4. **Output**: Predict preference score

```
Visual [D] ──┐
             ├─> Project ──> Fusion ──> Output [1]
Text [D]   ──┘
```

## Checkpoint Loading

Nếu có VIP5 checkpoint từ original repository:

1. Đảm bảo checkpoint format tương thích
2. Pass `checkpoint_path` vào `VIP5Reranker.__init__()`
3. Model sẽ tự động load state_dict từ checkpoint

**Note**: Format checkpoint có thể khác nhau tùy version. Cần điều chỉnh `load_vip5_checkpoint()` nếu cần.

## Dependencies

VIP5 cần các dependencies sau:

1. **CLIP Embeddings**: Phải có `clip_embeddings.pt` với cả `image_embs` và `text_embs`
2. **PyTorch**: Để chạy model
3. **NumPy**: Cho array operations

## Lưu ý

1. **CLIP Embeddings bắt buộc**: VIP5 cần cả visual và text embeddings. Nếu thiếu, sẽ raise error.

2. **Model Architecture**: Implementation hiện tại là simplified version. Để sử dụng full VIP5 architecture từ original repo, cần:
   - Clone VIP5 repository
   - Adapt model architecture từ `src/` folder
   - Update `VIP5Model` class

3. **Checkpoint Compatibility**: Checkpoint format từ original VIP5 có thể khác. Cần điều chỉnh `load_vip5_checkpoint()` nếu cần.

4. **Training**: Hiện tại chỉ hỗ trợ inference. Để train VIP5, cần implement training loop (có thể tham khảo từ original repo).

## Troubleshooting

### Lỗi: "CLIP embeddings not found"

**Giải pháp**: Chạy `data_prepare.py` với `--use_image` và `--use_text` flags

### Lỗi: "VIP5 requires image embeddings, but image_embs is None"

**Giải pháp**: Đảm bảo `data_prepare.py` đã extract image embeddings thành công

### Lỗi: "VIP5 requires text embeddings, but text_embs is None"

**Giải pháp**: Đảm bảo `data_prepare.py` đã extract text embeddings thành công

### Lỗi: "dimension mismatch"

**Giải pháp**: Điều chỉnh `visual_dim` và `text_dim` trong `__init__()` để match với CLIP embeddings dimension

## Tích hợp với Original VIP5

Để sử dụng full VIP5 implementation từ original repository:

1. Clone VIP5 repo:
   ```bash
   git clone https://github.com/jeykigung/VIP5.git
   ```

2. Copy model code từ `VIP5/src/` vào `rerank/models/vip5.py`

3. Update `VIP5Model` class để match với original architecture

4. Update checkpoint loading logic nếu cần

## Next Steps

- [ ] Implement full VIP5 architecture từ original repo
- [ ] Add training support cho VIP5
- [ ] Add evaluation metrics specific cho VIP5
- [ ] Support batch inference cho better performance

