# MMGCN Hyperparameters - Hướng dẫn điều chỉnh để tăng hiệu suất

## Tổng hợp các thông số có thể điều chỉnh

### 1. **Training Hyperparameters** (có thể điều chỉnh qua config.py)

| Thông số | Giá trị mặc định | Giá trị đề xuất | Mô tả |
|----------|------------------|-----------------|-------|
| `num_epochs` | 10 | **50-100** | Số epochs training (hiện tại: 100 từ config) |
| `batch_size` | 512 | **256-512** | Batch size cho training (hiện tại: 512 từ config) |
| `lr` | 1e-3 | **5e-4 đến 2e-3** | Learning rate (hiện tại: 1e-4 từ config - có thể quá nhỏ!) |
| `patience` | None | **10-20** | Early stopping patience (hiện tại: 10 từ config) |

**⚠️ Vấn đề hiện tại:**
- `lr=1e-4` từ config có thể **quá nhỏ** cho MMGCN
- Đề xuất: **1e-3 đến 2e-3** (giống VBPR paper: 5e-4, nhưng MMGCN có thể cần cao hơn)

### 2. **Model Architecture Hyperparameters** (hiện tại dùng giá trị mặc định)

| Thông số | Giá trị mặc định | Giá trị đề xuất | Mô tả | Ảnh hưởng |
|----------|------------------|-----------------|-------|-----------|
| `dim_x` | 64 | **128-256** | Embedding dimension | ⬆️ Tăng → model phức tạp hơn, có thể tốt hơn nhưng chậm hơn |
| `num_layer` | 2 | **2-3** | Số GCN layers | ⬆️ Tăng → học được patterns phức tạp hơn (nhưng có thể overfit) |
| `aggr_mode` | "add" | **"add" hoặc "mean"** | Aggregation mode | "add" thường tốt hơn "mean" cho recommendation |
| `concate` | False | **True** | Có concat features không | True → thêm thông tin, có thể tốt hơn |
| `has_id` | True | **True** | Có dùng ID embedding không | True → quan trọng cho recommendation |
| `reg_weight` | 1e-4 | **1e-5 đến 1e-3** | Regularization weight | ⬆️ Tăng → tránh overfit nhưng có thể underfit |

### 3. **Đề xuất điều chỉnh để tăng hiệu suất**

#### **Ưu tiên cao (ảnh hưởng lớn):**

1. **Learning Rate**: Tăng từ `1e-4` lên `1e-3` hoặc `2e-3`
   - Hiện tại `lr=1e-4` từ config có thể quá nhỏ
   - MMGCN cần learning rate cao hơn để học tốt

2. **Embedding Dimension (`dim_x`)**: Tăng từ `64` lên `128` hoặc `256`
   - Tăng capacity của model
   - Trade-off: chậm hơn nhưng có thể tốt hơn

3. **Number of Epochs**: Đảm bảo đủ epochs (hiện tại: 100 từ config - OK)
   - Với training đầy đủ (tất cả positive items), model cần nhiều epochs hơn

4. **Regularization Weight (`reg_weight`)**: Điều chỉnh từ `1e-4` xuống `1e-5` hoặc lên `1e-3`
   - Nếu overfit → tăng `reg_weight`
   - Nếu underfit → giảm `reg_weight`

#### **Ưu tiên trung bình:**

5. **Concate**: Thử `concate=True`
   - Có thể cải thiện performance bằng cách concat graph features và self-transformed features

6. **Number of Layers (`num_layer`)**: Thử `3` layers (hiện tại chỉ có 2 active)
   - Cần uncomment layer 3 trong code
   - Cẩn thận với overfitting

#### **Ưu tiên thấp:**

7. **Aggregation Mode**: Giữ `"add"` (thường tốt hơn `"mean"` cho recommendation)

8. **Batch Size**: Giữ `512` (đã tối ưu cho GPU)

## Cách điều chỉnh

### Option 1: Thêm vào config.py (khuyến nghị)
Thêm các thông số MMGCN-specific vào `config.py`:

```python
# MMGCN-specific hyperparameters
parser.add_argument('--mmgcn_dim_x', type=int, default=128,
                    help='MMGCN embedding dimension (default: 64, recommended: 128-256)')
parser.add_argument('--mmgcn_num_layer', type=int, default=2,
                    help='MMGCN number of GCN layers (default: 2, max: 3)')
parser.add_argument('--mmgcn_concate', action='store_true', default=True,
                    help='MMGCN concat features (default: False, recommended: True)')
parser.add_argument('--mmgcn_reg_weight', type=float, default=1e-4,
                    help='MMGCN regularization weight (default: 1e-4, range: 1e-5 to 1e-3)')
parser.add_argument('--mmgcn_aggr_mode', type=str, default='add',
                    choices=['add', 'mean', 'max'],
                    help='MMGCN aggregation mode (default: add)')
```

### Option 2: Sửa trực tiếp trong train_retrieval.py
Thêm vào `retriever_kwargs` khi tạo MMGCN:

```python
if retrieval_method == "mmgcn":
    retriever_kwargs.update({
        "dim_x": 128,  # Tăng từ 64
        "concate": True,  # Thử concat
        "num_layer": 2,  # Giữ 2 (hoặc thử 3 nếu uncomment layer 3)
        "reg_weight": 1e-4,  # Điều chỉnh nếu cần
        "aggr_mode": "add",  # Giữ "add"
    })
```

## So sánh với các methods khác

| Method | Learning Rate | Embedding Dim | Epochs | Batch Size |
|--------|---------------|---------------|--------|------------|
| **LRURec** | 1e-3 | 64 | 3-10 | 128 |
| **VBPR** | 5e-4 | 20 | 10-50 | 64 |
| **BM3** | 1e-3 | 64 | 10-100 | 64 |
| **MMGCN** (hiện tại) | **1e-4** ⚠️ | 64 | 100 | 512 |
| **MMGCN** (đề xuất) | **1e-3** | **128** | 100 | 512 |

## Lưu ý

1. **Learning Rate quá nhỏ**: `1e-4` có thể khiến model học chậm hoặc không học được
2. **Training đầy đủ**: Sau khi sửa training loop, model sẽ train nhiều samples hơn → cần điều chỉnh learning rate và epochs
3. **Regularization**: Với nhiều samples hơn, có thể cần tăng `reg_weight` để tránh overfit
4. **Memory**: Tăng `dim_x` sẽ tăng memory usage, cần kiểm tra GPU memory

