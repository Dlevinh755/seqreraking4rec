# So sánh LLM Training với LlamaRec

## 📚 LlamaRec Training Approach

### 1. **Training Objective** ✅
- **Next-token prediction** (giống LLM chuẩn)
- **Label = index letter của ground-truth item**
  - Ví dụ: Ground truth = item (D) → Label token = "D"
- **Loss function**: Cross-entropy loss trên token label
  - `L = -log P(token = GT_letter)`
- **Chỉ tính loss ở phần Response** (token index + EOS)
  - Không tính loss cho: instruction, user history, candidate list

---

## 🔍 Kiểm tra Implementation hiện tại

### 1. **Training Objective** ✅ ĐÚNG

**Location**: `rerank/models/llm.py:149-241`

**Code**:
```python
# ✅ Sử dụng SFTTrainer với next-token prediction
trainer = SFTTrainer(
    model=self.model,
    tokenizer=self.tokenizer,
    train_dataset=hf_train_dataset,
    args=training_args,
)

# ✅ Use train_on_responses_only to automatically mask prompt tokens
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
```

**Kết luận**: ✅ **ĐÚNG** - Sử dụng next-token prediction và mask prompt tokens

---

### 2. **Label Format** ❌ SAI

**LlamaRec**: Label = **letter index** (A, B, C, D, ...)

**Implementation hiện tại**: Label = **number index** (1, 2, 3, 4, ...)

**Location**: 
- `rerank/methods/qwen_reranker_unified.py:702`
- `scripts/train_rerank_standalone.py:254`

**Code hiện tại**:
```python
# ❌ SAI: Dùng number index
target = str(sample["target_idx"] + 1)  # "1", "2", "3", ...
```

**Code LlamaRec (đúng)**:
```python
# ✅ ĐÚNG: Dùng letter index
label = LETTERS[label_idx]  # "A", "B", "C", "D", ...
```

**Vấn đề**:
- Numbers (1, 2, 3, ...) có thể xuất hiện trong item text/description
- Dễ gây confusion khi model predict
- Letter index (A, B, C, ...) ít xuất hiện trong item text, tránh confusion

---

### 3. **Loss Function** ✅ ĐÚNG

**Location**: `rerank/models/llm.py:232-236`

**Code**:
```python
# ✅ train_on_responses_only tự động mask prompt tokens
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
```

**Kết luận**: ✅ **ĐÚNG** - Chỉ tính loss ở phần Response (assistant response)

---

### 4. **Prompt Format** ⚠️ KHÁC

**LlamaRec**: 
- Prompt có thể dùng letter labels (A, B, C, ...)
- Answer format: "Answer with only one letter (A-T)."

**Implementation hiện tại**:
- Prompt dùng number labels (1, 2, 3, ...)
- Answer format: "Answer with only one number (1-20)."

**Location**: `rerank/models/llm.py:17-57`

**Code hiện tại**:
```python
# ⚠️ Dùng numbers
cand_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
answer_format = f"Answer with only one number (1-{num_candidates})."
```

---

### 5. **Rerank/Inference** ⚠️ KHÁC

**LlamaRec**: 
- Predict letter token (A, B, C, ...)
- Map letter → candidate index

**Implementation hiện tại**:
- Predict number token (1, 2, 3, ...)
- Map number → candidate index

**Location**: `rerank/models/llm.py:266-369`

**Code hiện tại**:
```python
# ⚠️ Tìm number tokens
for i in range(1, num_candidates + 1):
    num_str = str(i)
    token_id = self.tokenizer.convert_tokens_to_ids(num_str)
    # ...
```

---

## 📊 Tóm tắt so sánh

| Aspect | LlamaRec | Implementation hiện tại | Status |
|--------|----------|-------------------------|--------|
| **Training Objective** | Next-token prediction | ✅ Next-token prediction | ✅ ĐÚNG |
| **Label Format** | Letter index (A, B, C, ...) | ❌ Number index (1, 2, 3, ...) | ❌ SAI |
| **Loss Function** | Cross-entropy on token label | ✅ Cross-entropy on token label | ✅ ĐÚNG |
| **Loss Masking** | Chỉ tính loss ở Response | ✅ Chỉ tính loss ở Response | ✅ ĐÚNG |
| **Prompt Format** | Letter labels (A-T) | ⚠️ Number labels (1-20) | ⚠️ KHÁC |
| **Rerank/Inference** | Predict letter token | ⚠️ Predict number token | ⚠️ KHÁC |

---

## 🔧 Đề xuất sửa

### Priority 1: Sửa Label Format từ Number → Letter

**Lý do**:
- LlamaRec dùng letter để tránh confusion với numbers trong item text
- Numbers (1, 2, 3, ...) có thể xuất hiện trong item descriptions
- Letter index (A, B, C, ...) ít xuất hiện hơn, tránh confusion

**Cần sửa**:

1. **Training data preparation**:
   - `rerank/methods/qwen_reranker_unified.py:702`
   - `scripts/train_rerank_standalone.py:254`

2. **Prompt format**:
   - `rerank/models/llm.py:17-57` (build_prompt_from_candidates)

3. **Rerank/Inference**:
   - `rerank/models/llm.py:266-369` (predict_probs)

**Code đề xuất**:
```python
# Thay vì:
target = str(sample["target_idx"] + 1)  # "1", "2", "3", ...

# Nên dùng:
LETTERS = list(string.ascii_uppercase[:20])  # A-T
target = LETTERS[sample["target_idx"]]  # "A", "B", "C", ...
```

---

## ✅ Kết luận

**Đã đúng với LlamaRec**:
- ✅ Training objective: Next-token prediction
- ✅ Loss function: Cross-entropy on token label
- ✅ Loss masking: Chỉ tính loss ở Response

**Chưa đúng với LlamaRec**:
- ❌ Label format: Dùng number (1, 2, 3, ...) thay vì letter (A, B, C, ...)
- ⚠️ Prompt format: Dùng number labels thay vì letter labels
- ⚠️ Rerank/Inference: Predict number token thay vì letter token

**Khuyến nghị**: Sửa label format từ number → letter để giống LlamaRec và tránh confusion với numbers trong item text.

