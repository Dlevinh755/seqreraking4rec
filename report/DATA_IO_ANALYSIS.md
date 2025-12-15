# Phân tích Save/Load Data - Đánh giá và Cải thiện

## 🔍 Phát hiện Vấn đề

### 1. **Code trùng lặp: Load CSV Dataset** ❌ CRITICAL

**Vấn đề:**
- `dataset/base.py` có code load CSV **2 LẦN GIỐNG HỆT** (lines 65-107 và 113-155)
- `scripts/train_retrieval.py` có code load CSV tương tự (lines 117-164)
- `evaluation/utils.py` đã có `load_dataset_from_csv()` nhưng **KHÔNG được dùng** trong `dataset/base.py`

**Impact:**
- ~100 lines code trùng lặp
- Khó maintain - sửa bug phải sửa nhiều chỗ
- Inconsistent behavior

**Đề xuất:**
- Refactor `dataset/base.py` để dùng `evaluation.utils.load_dataset_from_csv()`
- Hoặc tạo `dataset/utils.py` với hàm chung

---

### 2. **Hardcoded Paths** ❌ CRITICAL

**Vấn đề:**
- `scripts/train_rerank.py`:
  - Line 112: `Path("data/preprocessed/beauty_min_rating3-min_uc20-min_sc20/dataset_single_export.csv")`
  - Line 142: `Path("experiments/retrieval/lrurec/beauty/seed42/retrieved.csv")`
- `rerank/train_qwen.py`:
  - Line 106, 125, 130, 134: Hardcoded paths với `beauty_min_rating3-min_uc20-min_sc20`

**Impact:**
- Không flexible - chỉ chạy được với dataset/seed cụ thể
- Khó test với datasets khác
- Không dùng config

**Đề xuất:**
- Tạo utility functions để get paths từ config
- Sử dụng `dataset._get_preprocessed_folder_path()` và `EXPERIMENT_ROOT`

---

### 3. **Inconsistent Data Flow** ⚠️

**Data Flow hiện tại:**
```
1. data_prepare.py
   → Save: dataset_single_export.csv
   → Save: clip_embeddings.pt (optional)

2. dataset/base.py
   → Load: dataset_single_export.csv (2 lần code giống nhau!)
   → Fallback: dataset.pkl (legacy)

3. scripts/train_retrieval.py
   → Load: dataset_single_export.csv (code trùng lặp)
   → Save: retrieved.csv, retrieved_metrics.json

4. scripts/train_rerank.py
   → Load: dataset_single_export.csv (hardcoded path)
   → Load: retrieved.csv (hardcoded path)
```

**Vấn đề:**
- Không có standard interface cho save/load
- Mỗi script tự implement load logic
- Không có validation

---

### 4. **Missing Path Utilities** ⚠️

**Thiếu:**
- Utility để get experiment paths: `get_experiment_path(method, dataset, seed)`
- Utility để get retrieved CSV path
- Utility để save/load model checkpoints

**Hiện tại:**
- Mỗi script tự build paths
- Code lặp lại: `Path(EXPERIMENT_ROOT) / "retrieval" / method / dataset / f"seed{seed}"`

---

### 5. **Error Handling không đầy đủ** ⚠️

**Vấn đề:**
- Một số nơi check `exists()` nhưng không có clear error message
- Không validate data format sau khi load
- Không có fallback strategies

---

### 6. **File Format Inconsistency** ⚠️

**Hiện tại:**
- Dataset: CSV (primary), pickle (fallback)
- Retrieved: CSV + JSON
- CLIP embeddings: `.pt` (torch.save)
- Model checkpoints: Không rõ (có thể trong training scripts)

**Vấn đề:**
- Không có documentation về format
- Không có schema validation

---

## 📊 Tóm tắt Vấn đề

### Critical Issues:
1. ❌ Code trùng lặp load CSV (3 nơi, ~100 lines)
2. ❌ Hardcoded paths trong training scripts
3. ❌ `dataset/base.py` có code giống nhau 2 lần

### Important Issues:
4. ⚠️ Không có path utilities
5. ⚠️ Inconsistent error handling
6. ⚠️ Không có data validation

---

## 🎯 Đề xuất Cải thiện

### Priority 1 (Critical):

1. **Refactor `dataset/base.py`**
   - Gộp 2 đoạn code load CSV thành 1 hàm
   - Dùng `evaluation.utils.load_dataset_from_csv()` hoặc tạo `dataset/utils.py`

2. **Fix Hardcoded Paths**
   - `scripts/train_rerank.py`: Dùng config và utility functions
   - `rerank/train_qwen.py`: Dùng config (hoặc deprecate)

3. **Tạo Path Utilities**
   - `dataset/paths.py` hoặc `config.py` với helper functions
   - `get_experiment_path()`, `get_retrieved_csv_path()`, etc.

### Priority 2 (Important):

4. **Tạo Data I/O Utilities**
   - `dataset/io.py` với functions:
     - `save_dataset_csv()`
     - `load_dataset_csv()`
     - `validate_dataset_format()`

5. **Improve Error Handling**
   - Clear error messages
   - Validation after load
   - Fallback strategies

6. **Documentation**
   - Document data formats
   - Document save/load flow
   - Create data schema

---

## 💾 Data Flow Đề xuất (Sau cải thiện)

```
1. data_prepare.py
   → dataset.io.save_dataset_csv()  # Standardized save
   → dataset.io.save_clip_embeddings()  # If needed

2. dataset/base.py
   → dataset.io.load_dataset_csv()  # Single function, no duplication

3. scripts/train_retrieval.py
   → dataset.io.load_dataset_csv()  # Reuse utility
   → dataset.io.save_retrieved_csv()  # Standardized save

4. scripts/train_rerank.py
   → dataset.io.load_dataset_csv()  # From config, not hardcoded
   → dataset.io.load_retrieved_csv()  # From config, not hardcoded
```

---

## 📝 Implementation Plan

### Step 1: Tạo Path Utilities
```python
# dataset/paths.py hoặc config.py
def get_preprocessed_csv_path(dataset_code, min_rating, min_uc, min_sc):
    ...

def get_experiment_path(stage, method, dataset_code, seed):
    ...

def get_retrieved_csv_path(method, dataset_code, seed):
    ...
```

### Step 2: Refactor dataset/base.py
- Extract CSV loading logic thành helper function
- Dùng helper function ở cả 2 chỗ

### Step 3: Fix Hardcoded Paths
- Update `scripts/train_rerank.py`
- Update `rerank/train_qwen.py` (hoặc deprecate)

### Step 4: Tạo I/O Utilities
- `dataset/io.py` với standardized save/load functions

---

**Status**: ⚠️ Cần cải thiện ngay

