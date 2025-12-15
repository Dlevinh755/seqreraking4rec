# Phân tích tính khách quan khi so sánh các Rerank Methods

## Tổng quan

Báo cáo này phân tích xem các rerank methods (identity, random, qwen, vip5, bert4rec) đã được đánh giá một cách khách quan và công bằng hay chưa.

## ✅ Điểm tốt (Đã đảm bảo tính khách quan)

### 1. **Evaluation Metrics**
- ✅ Tất cả methods đều sử dụng cùng metrics: **Recall@K** và **NDCG@K**
- ✅ Tất cả đều dùng cùng evaluation function: `evaluate_split()` từ `evaluation.utils`
- ✅ Cùng evaluation protocol: evaluate trên cùng test set với cùng K
- ✅ Tất cả đều implement `BaseReranker` interface với cùng `rerank()` signature

### 2. **Data Splits**
- ✅ Tất cả methods đều dùng cùng train/val/test splits từ `load_dataset_from_csv()`
- ✅ Cùng data filtering: `min_rating`, `min_uc`, `min_sc`
- ✅ Cùng seed: `seed_everything(arg.seed)` được gọi trước khi train

### 3. **Pipeline Integration**
- ✅ Tất cả methods đều được tích hợp vào `TwoStagePipeline` cùng cách
- ✅ Cùng input: nhận candidates từ retrieval stage
- ✅ Cùng output format: `List[Tuple[int, float]]` (item_id, score)

### 4. **Rerank Top-K**
- ✅ Tất cả methods đều dùng cùng `rerank_top_k` từ `RerankConfig`

## ⚠️ Vấn đề cần cải thiện

### 1. **Training Requirements khác nhau**

**Vấn đề**: Các methods có training requirements rất khác nhau:

| Method | Training Required | Input Requirements | Training Time |
|--------|------------------|-------------------|---------------|
| **identity** | ❌ No | None | N/A |
| **random** | ❌ No | None | N/A |
| **qwen** | ✅ Yes (LLM fine-tuning) | item_id2text, user_history | Long |
| **vip5** | ⚠️ Optional (checkpoint) | CLIP embeddings (visual + text) | Long (if training) |
| **bert4rec** | ✅ Yes | Sequential data, vocab_size | Medium |

**Vấn đề**:
- Identity và Random không cần training → không công bằng khi so sánh với methods cần training
- Qwen và VIP5 có thể cần pre-training hoặc fine-tuning → training time khác nhau
- BERT4Rec cần training từ scratch → có thể chưa được train đủ

### 2. **Hyperparameters không được standardize**

**Vấn đề**: Các methods có hyperparameters riêng và không được quản lý từ config:

| Method | Key Hyperparameters | Default Values | From Config? |
|--------|---------------------|----------------|--------------|
| **bert4rec** | `num_epochs`, `batch_size`, `lr`, `patience` | 10, 32, 1e-4, None | ❌ No |
| **qwen** | `model_name`, `max_history` | "Qwen/Qwen3-0.6B", 10 | ❌ No |
| **vip5** | `backbone`, `checkpoint_path` | "t5-small", None | ❌ No |

**Hiện tại trong `scripts/train_pipeline.py`**:
```python
# Không có reranker_kwargs được truyền!
pipeline.fit(train, retriever_kwargs={"item_count": item_count, "val_data": val})
# reranker_kwargs không được truyền → methods dùng default values
```

**Vấn đề**: 
- ❌ Không có config cho rerank hyperparameters
- ❌ Mỗi method dùng default values riêng → **KHÔNG CÔNG BẰNG**
- ❌ Không có cách để standardize training settings

### 3. **Input Requirements khác nhau**

**Vấn đề**: Các methods cần input khác nhau:

| Method | Required Input | Optional Input |
|--------|---------------|----------------|
| **identity** | None | None |
| **random** | None | None |
| **qwen** | `item_id2text`, `user_history` | `train_data_for_llm` |
| **vip5** | CLIP embeddings (visual + text) | `checkpoint_path` |
| **bert4rec** | Sequential `train_data` | `vocab_size`, `val_data` |

**Vấn đề**:
- Qwen cần text features → chỉ hoạt động với datasets có text
- VIP5 cần CLIP embeddings → chỉ hoạt động với datasets có images
- BERT4Rec cần sequential data → cần data theo thứ tự thời gian
- Identity và Random không cần gì → luôn hoạt động

**Kết quả**: Không phải tất cả methods đều có thể chạy trên cùng dataset!

### 4. **Candidate Limit Issues**

**Vấn đề**: Qwen reranker có giới hạn 20 candidates:
- Nếu `retrieval_top_k > 20`, Qwen sẽ truncate về 20
- Các methods khác không có giới hạn này
- → **KHÔNG CÔNG BẰNG** khi so sánh với methods khác

### 5. **Training Logic chưa hoàn chỉnh**

**Vấn đề**: Trong `scripts/train_pipeline.py`:
```python
# Train Stage 2 (if not identity)
if args.rerank_method.lower() not in ("identity", "none"):
    print(f"\n[4/4] Training Stage 2 ({args.rerank_method})...")
    # TODO: Add reranker training logic here
    # For now, identity reranker doesn't need training
    pass  # ❌ Không có training logic!
```

**Vấn đề**:
- ❌ Rerankers không được train trong `train_pipeline.py`
- ❌ Chỉ được fit trong `TwoStagePipeline.fit()` nhưng không có kwargs
- ❌ Không có validation data cho rerankers
- ❌ Không có early stopping cho rerankers

### 6. **Model Selection**

**Vấn đề**:
- BERT4Rec có early stopping và model selection
- Qwen và VIP5 có thể không có (tùy implementation)
- Identity và Random không cần

## 📋 Khuyến nghị cải thiện

### 1. **Thêm Rerank Config vào config.py**

```python
# config.py - Thêm vào
parser.add_argument('--rerank_epochs', type=int, default=10,
                    help='Number of training epochs for rerank models')
parser.add_argument('--rerank_batch_size', type=int, default=32,
                    help='Batch size for rerank model training')
parser.add_argument('--rerank_lr', type=float, default=1e-4,
                    help='Learning rate for rerank models')
parser.add_argument('--rerank_patience', type=int, default=5,
                    help='Early stopping patience for rerank models')
```

### 2. **Cập nhật train_pipeline.py để truyền reranker_kwargs**

```python
# scripts/train_pipeline.py
reranker_kwargs = {
    "vocab_size": item_count + 1,  # For BERT4Rec
    "val_data": val,  # For early stopping
    "item_id2text": build_item_id2text(data["meta"]),  # For Qwen
    "user_history": build_user_history(train),  # For Qwen
}

pipeline.fit(
    train,
    retriever_kwargs={"item_count": item_count, "val_data": val},
    reranker_kwargs=reranker_kwargs  # NEW
)
```

### 3. **Standardize Training Settings**

Đảm bảo tất cả trainable rerankers (BERT4Rec, Qwen, VIP5) nhận cùng hyperparameters từ config:
- `num_epochs` từ `--rerank_epochs`
- `batch_size` từ `--rerank_batch_size`
- `lr` từ `--rerank_lr`
- `patience` từ `--rerank_patience`

### 4. **Xử lý Input Requirements**

Tạo helper functions để build required inputs:
```python
def build_reranker_kwargs(data, train, val, item_count):
    """Build kwargs for rerankers based on available data."""
    kwargs = {
        "vocab_size": item_count + 1,
        "val_data": val,
    }
    
    # Add text features if available
    if "meta" in data:
        item_id2text = {item_id: meta.get("text", f"item_{item_id}") 
                        for item_id, meta in data["meta"].items()}
        kwargs["item_id2text"] = item_id2text
    
    # Add user history for sequential methods
    kwargs["user_history"] = train
    
    return kwargs
```

### 5. **Documentation về Compatibility**

Tạo file `RERANK_METHOD_COMPATIBILITY.md` ghi rõ:
- Methods nào cần gì (text, images, sequential data)
- Methods nào có thể chạy trên dataset nào
- Giới hạn của từng method (ví dụ: Qwen 20 candidates)

### 6. **Tạo Comparison Script**

Tạo script để so sánh tất cả rerankers với cùng settings:
```python
# scripts/compare_rerank_methods.py
methods = ["identity", "random", "bert4rec"]  # Only methods that can run on same data
results = {}

for method in methods:
    pipeline = create_pipeline(retrieval_method, method, **common_kwargs)
    pipeline.fit(train, reranker_kwargs=standard_reranker_kwargs)
    metrics = evaluate_pipeline(pipeline, test)
    results[method] = metrics
```

## 🔍 Kiểm tra chi tiết

### Identity
- ✅ Không cần training
- ✅ Không cần input đặc biệt
- ✅ Luôn hoạt động
- ✅ Baseline tốt

### Random
- ✅ Không cần training
- ✅ Không cần input đặc biệt
- ✅ Luôn hoạt động
- ✅ Baseline tốt

### Qwen
- ⚠️ Cần training (LLM fine-tuning)
- ⚠️ Cần `item_id2text` và `user_history`
- ⚠️ Giới hạn 20 candidates
- ❌ Hyperparameters không từ config
- ❌ Không có validation trong training

### VIP5
- ⚠️ Có thể cần training hoặc checkpoint
- ⚠️ Cần CLIP embeddings (visual + text)
- ⚠️ Chỉ hoạt động với datasets có images
- ❌ Hyperparameters không từ config
- ❌ Training logic phức tạp

### BERT4Rec
- ✅ Cần training
- ✅ Cần sequential data (có sẵn)
- ✅ Có early stopping
- ❌ Hyperparameters không từ config (dùng defaults)
- ⚠️ Cần `vocab_size` (có thể infer)

## 📊 Kết luận

### Tính khách quan hiện tại: **50%**

**Đã đảm bảo**:
- ✅ Cùng evaluation metrics và protocol
- ✅ Cùng data splits và seed
- ✅ Cùng rerank top-K
- ✅ Cùng pipeline integration

**Chưa đảm bảo**:
- ❌ Training requirements khác nhau (một số không cần training)
- ❌ Hyperparameters không được standardize
- ❌ Input requirements khác nhau (không phải tất cả methods chạy được trên cùng dataset)
- ❌ Qwen có giới hạn 20 candidates
- ❌ Training logic chưa hoàn chỉnh trong `train_pipeline.py`
- ❌ Không có config cho rerank hyperparameters

### Hành động cần thiết

1. **Ngắn hạn** (Quan trọng):
   - Thêm rerank config vào `config.py`
   - Cập nhật `train_pipeline.py` để truyền `reranker_kwargs`
   - Standardize hyperparameters cho trainable rerankers

2. **Trung hạn**:
   - Hoàn thiện training logic trong `train_pipeline.py`
   - Tạo helper functions để build reranker inputs
   - Tạo compatibility documentation

3. **Dài hạn**:
   - Tạo comparison script cho methods tương thích
   - Hyperparameter tuning cho từng method
   - Report best settings cho từng method

## ⚠️ Lưu ý đặc biệt

**Không thể so sánh trực tiếp tất cả methods** vì:
1. **Qwen** chỉ hoạt động với datasets có text
2. **VIP5** chỉ hoạt động với datasets có images
3. **BERT4Rec** cần sequential data (có sẵn)
4. **Identity/Random** luôn hoạt động nhưng là baselines

**Khuyến nghị**: Chia thành các nhóm so sánh:
- **Group 1 (Baselines)**: identity, random
- **Group 2 (Text-based)**: qwen (nếu có text)
- **Group 3 (Multimodal)**: vip5 (nếu có images)
- **Group 4 (Sequential)**: bert4rec

So sánh trong từng group, không so sánh cross-group.

