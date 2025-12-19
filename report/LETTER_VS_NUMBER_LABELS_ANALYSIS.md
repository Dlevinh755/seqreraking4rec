# Phân tích Letter vs Number Labels cho LLM Reranking

## 📊 Giới hạn số lượng chữ cái

### Bảng chữ cái tiếng Anh
- **Chữ hoa (uppercase)**: 26 chữ (A-Z)
- **Chữ thường (lowercase)**: 26 chữ (a-z)
- **Tổng cộng**: 52 chữ cái

### Số lượng candidates hiện tại
- **Config default**: `rerank_eval_candidates = 50`
- **Config default**: `qwen_max_candidates = 50`
- **Có thể tăng lên**: 100, 200, hoặc nhiều hơn

---

## ❌ Vấn đề với Letter Labels

### 1. **Không đủ chữ cái cho 50+ candidates**

**Nếu chỉ dùng chữ hoa (A-Z)**:
- ✅ Đủ cho: 26 candidates
- ❌ Không đủ cho: 50 candidates (thiếu 24 chữ)

**Nếu dùng cả chữ hoa và chữ thường (A-Z, a-z)**:
- ✅ Đủ cho: 52 candidates
- ⚠️ Có thể đủ cho: 50 candidates (dư 2 chữ)
- ❌ Không đủ cho: 100+ candidates

### 2. **Code hiện tại chỉ hỗ trợ 20 chữ cái**

**Location**: `rerank/models/llm.py:14`
```python
LETTERS = list(string.ascii_uppercase[:20])  # A-T (chỉ 20 chữ)
```

**Vấn đề**:
- Chỉ hỗ trợ tối đa 20 candidates
- Config hiện tại yêu cầu 50 candidates
- Không đủ!

---

## ✅ Giải pháp

### Option 1: Dùng cả chữ hoa và chữ thường (Recommended cho ≤ 52 candidates)

**Ưu điểm**:
- Đủ cho 50 candidates (config default)
- Vẫn tránh confusion với numbers trong item text
- Giống LlamaRec (dùng letter)

**Nhược điểm**:
- Không đủ cho > 52 candidates
- Cần phân biệt chữ hoa/chữ thường (có thể gây confusion)

**Implementation**:
```python
# Dùng cả chữ hoa và chữ thường
LETTERS = list(string.ascii_uppercase) + list(string.ascii_lowercase)  # A-Z, a-z (52 chữ)
# A, B, C, ..., Z, a, b, c, ..., z
```

### Option 2: Dùng Numbers (Hiện tại - Đã implement) ✅

**Ưu điểm**:
- ✅ Không giới hạn số lượng (1, 2, 3, ..., 100, 200, ...)
- ✅ Đã được implement và test
- ✅ Hỗ trợ unlimited candidates

**Nhược điểm**:
- ⚠️ Có thể confusion với numbers trong item text
- ⚠️ Không giống LlamaRec (LlamaRec dùng letter)

**Implementation** (đã có):
```python
# Dùng numbers
cand_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
answer_format = f"Answer with only one number (1-{num_candidates})."
```

### Option 3: Hybrid - Letter cho ≤ 26, Number cho > 26

**Ưu điểm**:
- Dùng letter khi có ít candidates (giống LlamaRec)
- Dùng number khi có nhiều candidates (flexible)

**Nhược điểm**:
- Phức tạp hơn (cần 2 logic)
- Không consistent

---

## 📊 So sánh

| Approach | Max Candidates | LlamaRec Compatible | Confusion Risk |
|----------|----------------|---------------------|----------------|
| **Letter (A-Z only)** | 26 | ✅ Yes | ✅ Low |
| **Letter (A-Z, a-z)** | 52 | ✅ Yes | ⚠️ Medium (case sensitivity) |
| **Number (1, 2, 3, ...)** | Unlimited | ❌ No | ⚠️ Medium (numbers in text) |
| **Hybrid** | Unlimited | ⚠️ Partial | ⚠️ Medium |

---

## 🎯 Khuyến nghị

### Nếu muốn giống LlamaRec (≤ 52 candidates):

**Sử dụng Letter (A-Z, a-z)**:
```python
# Sửa trong rerank/models/llm.py
LETTERS = list(string.ascii_uppercase) + list(string.ascii_lowercase)  # 52 chữ
# A-Z (26) + a-z (26) = 52 chữ cái
```

**Giới hạn**:
- Set `rerank_eval_candidates <= 52`
- Set `qwen_max_candidates <= 52`

### Nếu cần > 52 candidates:

**Giữ nguyên Numbers (hiện tại)**:
- Đã implement và test
- Không giới hạn số lượng
- Chấp nhận risk confusion với numbers trong text

---

## ✅ Kết luận

**Câu trả lời**: 
- ❌ **Chỉ dùng chữ hoa (A-Z)**: Không đủ cho 50 candidates (chỉ có 26 chữ)
- ✅ **Dùng cả chữ hoa và chữ thường (A-Z, a-z)**: Đủ cho 50 candidates (có 52 chữ)
- ✅ **Dùng numbers (1, 2, 3, ...)**: Đủ cho unlimited candidates (hiện tại đang dùng)

**Khuyến nghị**:
- Nếu muốn giống LlamaRec và ≤ 52 candidates: Dùng letter (A-Z, a-z)
- Nếu cần > 52 candidates: Giữ nguyên numbers (hiện tại)

