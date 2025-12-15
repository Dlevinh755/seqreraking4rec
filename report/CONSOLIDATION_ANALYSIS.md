# Phân tích Các File Có Thể Gộp và Thu Gọn

## 🔍 Phát hiện

### 1. **Code trùng lặp: Hàm `_evaluate_split`** ⚠️

**Các file có code giống nhau:**
- `scripts/train_retrieval.py` - `_evaluate_split()` (lines 29-56)
- `retrieval/train_lrurec.py` - `_evaluate_split()` (lines 38-65) - **GIỐNG HỆT**
- `scripts/train_pipeline.py` - `evaluate_pipeline()` (lines 93-134) - **TƯƠNG TỰ**

**Đề xuất:** Tạo hàm chung trong `evaluation/utils.py` hoặc thêm vào `evaluation/metrics.py`

---

### 2. **Code trùng lặp: Hàm `load_dataset_from_csv`** ⚠️

**Các file có code tương tự:**
- `scripts/train_pipeline.py` - `load_dataset_from_csv()` (lines 18-90)
- `evaluation/offline_eval.py` - `load_dataset()` (lines 22-41) - **TƯƠNG TỰ**

**Đề xuất:** Tạo utility function chung trong `dataset/utils.py` hoặc `evaluation/utils.py`

---

### 3. **Cleanup Tools có thể gộp** ⚠️

**Các file:**
- `tools/clean_preprocessed.py` - Xóa preprocessed data
- `tools/cleanup_experiments.py` - Xóa experiment results

**Đề xuất:** Gộp thành `tools/clean.py` với subcommands:
```bash
python tools/clean.py preprocessed
python tools/clean.py experiments --method lrurec --dataset beauty
```

---

### 4. **File Deprecated có thể xóa** ✅

**File:**
- `retrieval/train_lrurec.py` - Đã deprecated, có warning
- Đã có `scripts/train_retrieval.py` thay thế

**Đề xuất:** Xóa hoàn toàn file này

---

### 5. **__init__.py files nhỏ có thể đơn giản hóa** ℹ️

**Các file:**
- `retrieval/methods/__init__.py` - Chỉ có docstring (125 bytes)
- `rerank/methods/__init__.py` - Chỉ có docstring (125 bytes)

**Đề xuất:** Giữ nguyên (không cần thiết phải gộp, nhưng có thể thêm exports nếu cần)

---

### 6. **Hàm `absolute_recall_mrr_ndcg_for_ks` trùng lặp** ⚠️

**Các file:**
- `scripts/train_retrieval.py` - `absolute_recall_mrr_ndcg_for_ks()` (lines 59-88)
- `retrieval/train_lrurec.py` - `absolute_recall_mrr_ndcg_for_ks()` (lines 68-97) - **GIỐNG HỆT**

**Đề xuất:** Di chuyển vào `evaluation/metrics.py`

---

## 📊 Tóm tắt

### Code trùng lặp:
1. ✅ `_evaluate_split()` - 3 nơi (có thể gộp)
2. ✅ `load_dataset_from_csv()` - 2 nơi (có thể gộp)
3. ✅ `absolute_recall_mrr_ndcg_for_ks()` - 2 nơi (có thể gộp)

### Files có thể xóa:
1. ✅ `retrieval/train_lrurec.py` - Deprecated

### Tools có thể gộp:
1. ✅ `clean_preprocessed.py` + `cleanup_experiments.py` → `clean.py`

---

## 🎯 Kế hoạch Thực hiện

### Priority 1 (Quan trọng):
1. Tạo `evaluation/utils.py` với các hàm chung:
   - `evaluate_split()` - Gộp `_evaluate_split` và `evaluate_pipeline`
   - `load_dataset_from_csv()` - Utility function chung
2. Di chuyển `absolute_recall_mrr_ndcg_for_ks()` vào `evaluation/metrics.py`
3. Xóa `retrieval/train_lrurec.py` (deprecated)

### Priority 2 (Nên làm):
4. Gộp cleanup tools thành `tools/clean.py` với subcommands
5. Cập nhật imports trong các file sử dụng

---

## 💾 Tiết kiệm

Sau khi gộp:
- **Giảm ~200 lines** code trùng lặp
- **Xóa 1 file** deprecated (~300 lines)
- **Gộp 2 tools** thành 1 file
- **Tổng cộng**: Giảm ~500 lines code, dễ maintain hơn

