# MMGCN Retrieval Support

## Tổng quan

Script `scripts/train_retrieval.py` đã được cập nhật để hỗ trợ training MMGCN retrieval model thay vì chỉ LRURec.

## Thay đổi chính

### 1. Thêm argument `--retrieval_method`

Script hiện hỗ trợ chọn retrieval method qua command line:

```bash
# Train với LRURec (mặc định)
python scripts/train_retrieval.py

# Train với MMGCN
python scripts/train_retrieval.py --retrieval_method mmgcn
```

### 2. Thêm function `_build_edge_index()`

Function này build graph edge_index từ user-item interactions:

- Input: `train_data` (Dict {user_id: [item_ids]}), `num_user`, `num_item`
- Output: `edge_index` (np.ndarray shape [2, E])
- Format: User nodes (0..num_user-1), Item nodes (num_user..num_user+num_item-1)

### 3. Thêm function `_load_clip_embeddings()`

Function này load CLIP embeddings cho visual và text features:

- Load từ `data/preprocessed/{dataset}/clip_embeddings.pt`
- Trả về `v_feat` (image embeddings) và `t_feat` (text embeddings)
- Shape: [num_items, D] (skip row 0 padding)

### 4. Cập nhật `main()` function

- Tự động detect method và load dependencies tương ứng
- Với MMGCN: load CLIP embeddings, build edge_index, pass vào `retriever.fit()`

## Dependencies cho MMGCN

MMGCN cần các dependencies sau:

1. **CLIP Embeddings**: Phải chạy `data_prepare.py` với flags `--use_image` và `--use_text` trước
2. **Graph Edges**: Tự động build từ `train_data`
3. **User/Item Counts**: Tự động tính từ dataset

## Cách sử dụng

### Bước 1: Chuẩn bị data với CLIP embeddings

```bash
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image \
    --use_text
```

Điều này sẽ tạo file `data/preprocessed/.../clip_embeddings.pt`

### Bước 2: Train MMGCN retrieval

```bash
python scripts/train_retrieval.py \
    --retrieval_method mmgcn \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --retrieval_epochs 10 \
    --batch_size_retrieval 128
```

## Lưu ý

1. **CLIP Embeddings bắt buộc**: Nếu chưa có CLIP embeddings, script sẽ raise `FileNotFoundError` với hướng dẫn
2. **Memory**: MMGCN cần nhiều memory hơn LRURec do phải load CLIP embeddings và graph
3. **Training time**: MMGCN training có thể lâu hơn do GCN forward pass

## Kiểm tra

Để kiểm tra xem script có chạy được với MMGCN không:

1. Đảm bảo đã có CLIP embeddings:
   ```bash
   ls data/preprocessed/*/clip_embeddings.pt
   ```

2. Chạy với `--retrieval_method mmgcn`:
   ```bash
   python scripts/train_retrieval.py --retrieval_method mmgcn
   ```

3. Kiểm tra output:
   - Script sẽ in: "Loading CLIP embeddings for MMGCN..."
   - Script sẽ in: "Building edge_index from X user interactions..."
   - Script sẽ in: "Built edge_index with shape (2, E) (E edges)"

## Troubleshooting

### Lỗi: "CLIP embeddings not found"

**Giải pháp**: Chạy `data_prepare.py` với `--use_image` và `--use_text` flags

### Lỗi: "MMGCN requires image embeddings (v_feat), but image_embs is None"

**Giải pháp**: Đảm bảo `data_prepare.py` đã extract image embeddings thành công

### Lỗi: "MMGCN requires text embeddings (t_feat), but text_embs is None"

**Giải pháp**: Đảm bảo `data_prepare.py` đã extract text embeddings thành công

### Lỗi: "No valid edges found in train_data"

**Giải pháp**: Kiểm tra xem `train_data` có hợp lệ không, `num_user` và `num_item` có đúng không

