# Kiểm tra: Training và Evaluation có đang được thực hiện đúng không?

## ✅ Training Process

### 1. **Data Preparation** (`_prepare_training_samples`)

**File**: `rerank/methods/qwen_reranker_unified.py:651-703`

**Quy trình**:
1. ✅ Lặp qua `train_data` (user_id -> [item_ids])
2. ✅ Skip users có < 2 items
3. ✅ Chọn target item: `end_pos = random.randint(1, len(items) - 1)` hoặc `len(items) - 1`
4. ✅ History: `items[:end_pos]`
5. ✅ Target: `items[end_pos]`
6. ✅ Sample negatives: `num_negatives = rerank_eval_candidates - 1` (từ config)
7. ✅ Candidates: `[target_item] + negatives`, shuffle
8. ✅ Target index: `candidates.index(target_item)` (0-indexed)

**✅ Đúng**: 
- Target index là 0-indexed
- Số lượng negatives lấy từ config
- Candidates được shuffle

### 2. **Training Data Format** (`_train_text_model`)

**File**: `rerank/methods/qwen_reranker_unified.py:720-800`

**Quy trình**:
1. ✅ Build prompt từ sample: `_build_training_prompt(sample)`
2. ✅ Convert target_idx → letter: `LETTERS[target_idx]` (LlamaRec style)
3. ✅ Format messages:
   ```python
   {
       "messages": [
           {"role": "system", "content": "You are a recommendation ranking assistant."},
           {"role": "user", "content": prompt},
           {"role": "assistant", "content": target_letter}  # ✅ Letter (A, B, C, ...)
       ]
   }
   ```

**✅ Đúng**:
- Target là letter (A, B, C, ...) thay vì number
- Messages format đúng cho chat template
- Prompt có letters cho candidates

### 3. **LLM Training** (`LLMModel.train()`)

**File**: `rerank/models/llm.py:162-254`

**Quy trình**:
1. ✅ Convert messages → text: `apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`
2. ✅ Format: `'\n<user_content><|im_end|>\n<|im_start|>assistant\n<response><|im_end|>\n'`
3. ✅ Use `train_on_responses_only`: Mask prompt tokens, chỉ tính loss trên response tokens
4. ✅ Training với SFTTrainer:
   - `num_epochs`: từ config (default: 1)
   - `batch_size`: từ config (default: 16)
   - `learning_rate`: từ config (default: 1e-4) ✅ ĐÃ SỬA
   - `gradient_accumulation_steps`: 4
   - `fp16`: True
   - `optim`: "adamw_8bit"

**✅ Đúng**:
- Chat template format đúng
- Loss chỉ tính trên response tokens (LlamaRec style)
- Hyperparameters lấy từ config
- Learning rate đã được sửa để lấy từ config

**⚠️ Vấn đề**:
- `num_epochs=1` (default) → quá ít, cần tăng lên 10-20

### 4. **Training Loop với Validation**

**File**: `rerank/methods/qwen_reranker_unified.py:1300-1330`

**Quy trình**:
1. ✅ Lặp qua epochs
2. ✅ Train model: `trainer.train()`
3. ✅ Validate: `_evaluate_split(val_data, k=min(10, self.top_k))`
4. ✅ Early stopping: nếu `epochs_no_improve >= patience`
5. ✅ Load best model: `trainer.model.load_state_dict(best_model_state)`

**✅ Đúng**:
- Validation sau mỗi epoch
- Early stopping hoạt động
- Best model được lưu và load

## ✅ Evaluation Process

### 1. **Candidate Loading** (`_evaluate_split`)

**File**: `rerank/methods/qwen_reranker_unified.py:1332-1373`

**Quy trình**:
1. ✅ Load pre-generated candidates: `load_rerank_candidates()`
2. ✅ Lấy candidates cho user từ val/test split
3. ✅ Skip nếu không có candidates

**✅ Đúng**:
- Sử dụng pre-generated candidates (từ data_prepare.py)
- Đúng split (val hoặc test)

### 2. **Reranking** (`rerank()`)

**File**: `rerank/methods/qwen_reranker_unified.py:425-649`

**Quy trình**:
1. ✅ Get user history: `history[-self.max_history:]` (truncate)
2. ✅ Build prompt: `build_prompt_from_candidates()` hoặc `_build_test_prompt_sample()`
3. ✅ Apply chat template: `apply_chat_template(messages, add_generation_prompt=True)` ✅ ĐÃ SỬA
4. ✅ Predict probabilities: `predict_probs(prompt, num_candidates)`
5. ✅ Rank candidates: `rank_candidates(probs, candidates)`
6. ✅ Return top_k: `ranked_items[:self.top_k]`

**✅ Đúng**:
- History được truncate theo `max_history`
- Prompt dùng letters (A, B, C, ...)
- Chat template format được sử dụng cho inference ✅ ĐÃ SỬA
- Probabilities được normalize

### 3. **Metric Calculation**

**File**: `rerank/methods/qwen_reranker_unified.py:1365-1373`

**Quy trình**:
1. ✅ Rerank candidates: `rerank(user_id, candidates)`
2. ✅ Get top_k items: `[item_id for item_id, _ in reranked[:k]]`
3. ✅ Calculate hits: `len(set(top_k_items) & set(gt_items))`
4. ✅ Calculate recall: `hits / len(gt_items)`
5. ✅ Average recall: `np.mean(recalls)`

**✅ Đúng**:
- Recall calculation đúng
- Top-k items được lấy đúng
- Average across users

## 🔍 Kiểm tra chi tiết

### **Training Data Format**

**Expected**:
```python
{
    "messages": [
        {"role": "system", "content": "You are a recommendation ranking assistant."},
        {"role": "user", "content": "You are a recommendation ranking assistant.\n\nChoose exactly ONE item...\n\nCandidate items:\nA. item1\nB. item2\n...\n\nAnswer with only one letter (A-Z, a-z)."},
        {"role": "assistant", "content": "E"}  # ✅ Letter
    ]
}
```

**Actual**: ✅ Đúng format

### **Inference Prompt Format**

**Expected** (sau chat template):
```
<|im_start|>user
You are a recommendation ranking assistant.
...
Answer with only one letter (A-Z, a-z).<|im_end|>
<|im_start|>assistant
```

**Actual**: ✅ Đúng format (đã sửa)

### **Target Label Format**

**Expected**: Letter (A, B, C, ...) - LlamaRec style

**Actual**: ✅ `LETTERS[target_idx]` - Đúng

### **Loss Calculation**

**Expected**: 
- Loss chỉ tính trên response tokens (letter + EOS)
- Không tính loss cho prompt tokens

**Actual**: ✅ `train_on_responses_only` - Đúng

## ⚠️ Vấn đề phát hiện

### 1. **Quá ít Epochs** (CRITICAL)

**Vấn đề**:
- Default `rerank_epochs=1` → quá ít
- Model chưa kịp học

**Giải pháp**:
```bash
--rerank_epochs 10  # Hoặc 20
```

### 2. **Learning Rate đã được sửa** ✅

**Trước**: Hardcode `learning_rate=2e-5`
**Sau**: Lấy từ config `rerank_lr=1e-4` ✅

### 3. **Training Loss cao** (4.25)

**Nguyên nhân**:
- Quá ít epochs (1)
- Model chưa converge

**Giải pháp**:
- Tăng epochs lên 10-20
- Monitor training loss

## ✅ Kết luận

### **Training Process**: ✅ ĐÚNG

1. ✅ Data preparation đúng
2. ✅ Training data format đúng (letters, chat template)
3. ✅ Loss calculation đúng (chỉ trên response tokens)
4. ✅ Training loop đúng (với validation và early stopping)
5. ⚠️ Cần tăng epochs (1 → 10-20)

### **Evaluation Process**: ✅ ĐÚNG

1. ✅ Candidate loading đúng (pre-generated)
2. ✅ Prompt building đúng (letters, chat template)
3. ✅ Reranking đúng (predict_probs → rank)
4. ✅ Metric calculation đúng (Recall@K)

### **Cần sửa**:

1. ✅ **Đã sửa**: Learning rate lấy từ config
2. ⚠️ **Cần sửa**: Tăng `--rerank_epochs` lên 10-20
3. ⚠️ **Cần monitor**: Training loss phải giảm xuống < 2.0

## 📝 Recommendations

1. **Tăng epochs**: `--rerank_epochs 10` hoặc `20`
2. **Monitor training loss**: Phải giảm từ 4.25 xuống < 2.0
3. **Check validation recall**: Phải tăng dần qua các epochs
4. **Verify predictions**: Kiểm tra xem model có predict đúng letter không

