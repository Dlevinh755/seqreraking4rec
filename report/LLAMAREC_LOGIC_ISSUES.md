# Vấn đề Logic Training và Rerank so với LlamaRec

## 📊 Tổng quan

Sau khi so sánh chi tiết với LlamaRec, tôi phát hiện **logic code đã đúng**, nhưng có **1 vấn đề tiềm ẩn** và **1 vấn đề hyperparameters**.

---

## ✅ Logic đã đúng với LlamaRec

### **1. Training Data Preparation** ✅

- ✅ Dùng **letter labels** (A, B, C, ...) thay vì numbers
- ✅ Random split point
- ✅ Candidates = [target] + negatives (shuffled)
- ✅ Target index = 0-indexed

**Code**: `rerank/methods/qwen_reranker_unified.py:652-704`

---

### **2. Training Process** ✅

- ✅ Next-token prediction
- ✅ Mask prompt tokens với `train_on_responses_only`
- ✅ Chat template format
- ✅ LoRA fine-tuning

**Code**: `rerank/models/llm.py:164-263`

---

### **3. Rerank/Inference** ✅

- ✅ Extract logits: `logits[:, -1]`
- ✅ Extract letter token IDs (với fallback strategies)
- ✅ Softmax trên letter tokens
- ✅ Map về candidate indices
- ✅ Sort by probability

**Code**: `rerank/models/llm.py:290-414`

---

### **4. Prompt Format** ✅

- ✅ Dùng letter labels (A, B, C, ...)
- ✅ Answer format với letters
- ✅ Format giống LlamaRec

**Code**: `rerank/models/llm.py:19-69`

---

## ⚠️ Vấn đề Logic Phát hiện

### **1. Text-only mode có thể không train** 🔴

**Vấn đề**:

**Code** (`rerank/methods/qwen_reranker_unified.py:296-403`):
```python
train_data_for_llm = kwargs.get("train_data_for_llm")

# For caption/semantic_summary modes, prepare training data from item_meta
if self.mode in ["caption", "semantic_summary"] and train_data_for_llm is None:
    train_samples = self._prepare_training_samples(train_data)
    # ... prepare train_data_for_llm ...
    
if train_data_for_llm is not None:
    # Train với pre-prepared data
    self.llm_model = LLMModel(train_data=train_data_for_llm, ...)
    self.llm_model.train(...)
else:
    # ❌ Chỉ load model, KHÔNG TRAIN!
    self.llm_model = LLMModel(train_data=None, ...)
    self.llm_model.load_model(...)
```

**Phân tích**:
- ✅ `caption/semantic_summary` modes: Tự động prepare training data nếu không có
- ❌ `text_only` mode: **KHÔNG tự động prepare** → nếu không có `train_data_for_llm` từ kwargs → **model không được train!**

**Giải pháp**:
```python
# Thêm auto-prepare cho text_only mode
if self.mode == "text_only" and train_data_for_llm is None:
    train_samples = self._prepare_training_samples(train_data)
    if len(train_samples) > 0:
        # Convert to LLM training format với letter labels
        from rerank.models.llm import build_prompt_from_candidates, LETTERS
        train_data_for_llm = []
        for sample in train_samples:
            history = sample["history"]
            candidates = sample["candidates"]
            target_idx = sample["target_idx"]
            
            # Build prompt
            history_texts = [self.item_id2text.get(item_id, f"item_{item_id}") 
                           for item_id in history[-self.max_history:]]
            candidate_texts = [self.item_id2text.get(item_id, f"item_{item_id}") 
                             for item_id in candidates]
            
            prompt = build_prompt_from_candidates(
                history_texts,
                candidates,  # IDs for mapping
                self.item_id2text,
                max_candidates=self.max_candidates
            )
            
            # Use letter index (LlamaRec style)
            if target_idx >= len(LETTERS):
                raise ValueError(f"Target index {target_idx} exceeds max letters")
            target = LETTERS[target_idx]  # Letter (A, B, C, ...)
            
            train_data_for_llm.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target}
                ]
            })
```

---

### **2. Epochs quá ít (CRITICAL)** 🔴

**Vấn đề**:
- `rerank_epochs = 1` (default) → quá ít để model học được pattern
- Với 1635 samples, batch_size=16, gradient_accumulation=2:
  - Steps per epoch = 1635 / 32 ≈ 51 steps
  - Total steps = 51 × 1 = **51 steps** (quá ít!)

**Giải pháp**:
```python
# config.py
parser.add_argument('--rerank_epochs', type=int, default=5,  # ✅ Tăng từ 1 lên 5
```

---

## 📊 So sánh với LlamaRec

| Aspect | LlamaRec | Project hiện tại | Status |
|--------|----------|------------------|--------|
| **Training Data Prep** | Letter labels | ✅ Letter labels | ✅ ĐÚNG |
| **Training Process** | Next-token prediction | ✅ Next-token prediction | ✅ ĐÚNG |
| **Loss Masking** | Mask prompt tokens | ✅ `train_on_responses_only` | ✅ ĐÚNG |
| **Prompt Format** | Letter labels | ✅ Letter labels | ✅ ĐÚNG |
| **Rerank Process** | Verbalizer approach | ✅ Verbalizer approach | ✅ ĐÚNG |
| **Text-only Auto-prepare** | N/A | ❌ Không tự động | ⚠️ VẤN ĐỀ |
| **Epochs** | 3-10 epochs | ❌ 1 epoch (default) | ⚠️ VẤN ĐỀ |

---

## 🎯 Kết luận

### **Logic Code**: ✅ **ĐÚNG với LlamaRec**

**Tất cả các aspects chính đều đúng**:
- ✅ Training data preparation
- ✅ Training process
- ✅ Rerank process
- ✅ Prompt format

### **Vấn đề**:

1. **Text-only mode** có thể không train nếu không có `train_data_for_llm` từ kwargs
2. **Epochs quá ít** (1 epoch) → Model chưa học được pattern

### **Recommendation**:

1. ✅ **Tăng epochs lên 5-10** (CRITICAL)
2. ✅ **Thêm auto-prepare cho text_only mode** (nếu đang dùng text_only mode)
3. ✅ **Kiểm tra xem model có được train không** (check training loss)

**Logic code đã đúng, vấn đề chính là hyperparameters (epochs quá ít)!**

