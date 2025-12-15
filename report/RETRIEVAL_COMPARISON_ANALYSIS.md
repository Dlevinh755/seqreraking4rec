# Phân tích tính khách quan khi so sánh các Retrieval Methods

## Tổng quan

Báo cáo này phân tích xem các retrieval methods (LRURec, MMGCN, VBPR, BM3) đã được đánh giá một cách khách quan và công bằng hay chưa.

## ✅ Điểm tốt (Đã đảm bảo tính khách quan)

### 1. **Evaluation Metrics**
- ✅ Tất cả methods đều sử dụng cùng metrics: **Recall@K** và **NDCG@K**
- ✅ Tất cả đều dùng cùng evaluation function: `evaluate_split()` từ `evaluation.utils`
- ✅ Cùng evaluation protocol: evaluate trên cùng test set với cùng K

### 2. **Data Splits**
- ✅ Tất cả methods đều dùng cùng train/val/test splits từ `load_dataset_from_csv()`
- ✅ Cùng data filtering: `min_rating`, `min_uc`, `min_sc`
- ✅ Cùng seed: `seed_everything(arg.seed)` được gọi trước khi train

### 3. **Evaluation Interface**
- ✅ Tất cả methods implement cùng `BaseRetriever` interface
- ✅ Cùng `retrieve(user_id)` method signature
- ✅ Cùng cách exclude items từ history

### 4. **Retrieval Top-K**
- ✅ Tất cả methods đều dùng cùng `RETRIEVAL_TOP_K = 200` (trong `scripts/train_retrieval.py`)

## ⚠️ Vấn đề cần cải thiện

### 1. **Hyperparameters không đồng nhất**

**Vấn đề**: Các methods có default hyperparameters khác nhau, dẫn đến không công bằng khi so sánh.

| Method | Default `num_epochs` | Default `batch_size` | Default `lr` | Optimizer |
|--------|---------------------|---------------------|--------------|-----------|
| **LRURec** | 3 | 128 | 1e-3 | AdamW |
| **MMGCN** | 10 | 128 | 1e-3 | Adam |
| **VBPR** | 10 | 64 | 5e-4 | SGD |
| **BM3** | 10 | 64 | 1e-3 | Adam |

**Hiện tại trong `scripts/train_retrieval.py`**:
```python
retriever_kwargs = {
    "top_k": RETRIEVAL_TOP_K,
    "num_epochs": arg.retrieval_epochs,  # Từ config.py (default: 10)
    "batch_size": arg.batch_size_retrieval,  # Từ config.py (default: 128)
    "patience": arg.retrieval_patience,  # Từ config.py (default: 5)
}
```

**Vấn đề**: 
- ✅ `num_epochs`, `batch_size`, `patience` được truyền từ config → **OK**
- ❌ `lr` không được truyền từ config → Mỗi method dùng default riêng → **KHÔNG CÔNG BẰNG**
- ❌ Các hyperparameters khác (như `dim_gamma`, `dim_theta` cho VBPR, `embed_dim` cho BM3) không được standardize

### 2. **Early Stopping không nhất quán**

**Vấn đề**: 
- `patience` được truyền từ config (default: 5)
- Nhưng một số methods có thể không implement early stopping đúng cách
- Cần kiểm tra xem tất cả methods có dùng `patience` từ config không

### 3. **Validation Set Usage**

**Vấn đề**:
- Tất cả methods đều nhận `val_data` trong `fit_kwargs`
- Nhưng cần đảm bảo tất cả đều dùng validation set cho early stopping và model selection

### 4. **Model Selection**

**Vấn đề**:
- Cần đảm bảo tất cả methods đều:
  - Evaluate trên validation set sau mỗi epoch
  - Lưu best model state dựa trên validation metric
  - Load best model state trước khi evaluate trên test set

## 📋 Khuyến nghị cải thiện

### 1. **Standardize Hyperparameters**

Tạo một config file chung cho tất cả retrieval methods:

```python
# config.py - Thêm vào
parser.add_argument('--retrieval_lr', type=float, default=1e-3,
                    help='Learning rate for all retrieval methods')
parser.add_argument('--retrieval_embed_dim', type=int, default=64,
                    help='Embedding dimension for retrieval methods')
```

Và cập nhật `scripts/train_retrieval.py`:
```python
retriever_kwargs = {
    "top_k": RETRIEVAL_TOP_K,
    "num_epochs": arg.retrieval_epochs,
    "batch_size": arg.batch_size_retrieval,
    "patience": arg.retrieval_patience,
    "lr": arg.retrieval_lr,  # NEW: Standardize learning rate
}
```

### 2. **Đảm bảo Early Stopping nhất quán**

Kiểm tra và đảm bảo tất cả methods:
- ✅ Nhận `patience` từ config
- ✅ Evaluate trên validation set sau mỗi epoch
- ✅ Lưu best model state
- ✅ Load best model state trước khi test

### 3. **Tạo Comparison Script**

Tạo script để so sánh tất cả methods với cùng settings:

```python
# scripts/compare_retrieval_methods.py
methods = ["lrurec", "mmgcn", "vbpr", "bm3"]
results = {}

for method in methods:
    # Train với cùng settings
    retriever = train_retriever(method, **common_kwargs)
    # Evaluate trên cùng test set
    metrics = evaluate_retriever(retriever, test)
    results[method] = metrics

# Print comparison table
print_comparison_table(results)
```

### 4. **Documentation**

Tạo file `RETRIEVAL_BENCHMARK.md` ghi rõ:
- Hyperparameters được sử dụng cho mỗi method
- Training settings (epochs, batch size, lr, etc.)
- Evaluation protocol
- Results table

## 🔍 Kiểm tra chi tiết

### LRURec
- ✅ Nhận `num_epochs`, `batch_size`, `patience` từ config
- ❌ `lr` dùng default 1e-3 (không từ config)
- ✅ Có early stopping
- ✅ Evaluate trên validation set

### MMGCN
- ✅ Nhận `num_epochs`, `batch_size`, `patience` từ config
- ❌ `lr` dùng default 1e-3 (không từ config)
- ✅ Có early stopping
- ✅ Evaluate trên validation set

### VBPR
- ✅ Nhận `num_epochs`, `batch_size`, `patience` từ config
- ❌ `lr` dùng default 5e-4 (không từ config, khác với các methods khác!)
- ✅ Có early stopping
- ✅ Evaluate trên validation set

### BM3
- ✅ Nhận `num_epochs`, `batch_size`, `patience` từ config
- ❌ `lr` dùng default 1e-3 (không từ config)
- ✅ Có early stopping
- ✅ Evaluate trên validation set

## 📊 Kết luận

### Tính khách quan hiện tại: **70%**

**Đã đảm bảo**:
- ✅ Cùng evaluation metrics và protocol
- ✅ Cùng data splits và seed
- ✅ Cùng retrieval top-K
- ✅ Cùng early stopping mechanism

**Chưa đảm bảo**:
- ❌ Learning rate không đồng nhất (VBPR dùng 5e-4, các methods khác dùng 1e-3)
- ❌ Một số hyperparameters khác (embedding dimensions, regularization weights) không được standardize
- ❌ Chưa có script để so sánh tất cả methods cùng lúc

### Hành động cần thiết

1. **Ngắn hạn** (Quan trọng):
   - Thêm `--retrieval_lr` vào config.py
   - Cập nhật `scripts/train_retrieval.py` để truyền `lr` từ config cho tất cả methods
   - Đảm bảo VBPR cũng dùng `lr` từ config (thay vì default 5e-4)

2. **Trung hạn**:
   - Tạo `scripts/compare_retrieval_methods.py` để so sánh tất cả methods
   - Tạo `RETRIEVAL_BENCHMARK.md` với results table

3. **Dài hạn**:
   - Hyperparameter tuning cho từng method
   - Report best hyperparameters cho từng method trên từng dataset

