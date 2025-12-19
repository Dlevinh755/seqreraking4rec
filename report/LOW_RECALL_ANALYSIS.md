# Phân tích: Tại sao Recall@20 thấp (0.4 - ngang với random)?

## 📊 Vấn đề

**Recall@20 ≈ 0.4** - Gần như random performance

**Random baseline**:
- Với 20 candidates và 1 GT item: Recall@20 = 1/20 = **0.05** (5%)
- Với 20 candidates và nhiều GT items: Recall@20 có thể cao hơn
- **0.4 recall@20** cho thấy model gần như không học được gì

---

## 🔍 Nguyên nhân chính

### **1. Epochs quá ít (CRITICAL)** ⚠️

**Config hiện tại** (`config.py:122`):
```python
parser.add_argument('--rerank_epochs', type=int, default=1,  # ❌ QUÁ ÍT!
```

**Vấn đề**:
- **1 epoch** là quá ít để model học được pattern
- Với LLM fine-tuning, thường cần **3-10 epochs**
- Model chưa kịp converge

**Phân tích**:
- Training steps = `(num_samples / batch_size) * epochs`
- Với 1635 samples, batch_size=16, gradient_accumulation=2:
  - Effective batch size = 16 * 2 = 32
  - Steps per epoch = 1635 / 32 ≈ **51 steps**
  - Total steps = 51 * 1 = **51 steps** (quá ít!)

**Giải pháp**:
```python
# config.py
parser.add_argument('--rerank_epochs', type=int, default=5,  # ✅ Tăng lên 5-10
```

---

### **2. Learning Rate có thể chưa tối ưu** ⚠️

**Config hiện tại** (`config.py:126`):
```python
parser.add_argument('--rerank_lr', type=float, default=1e-4,
```

**Phân tích**:
- `1e-4` có thể OK cho LoRA fine-tuning
- Nhưng với chỉ 1 epoch, learning rate cần cao hơn để model học nhanh hơn
- LlamaRec thường dùng `1e-4` đến `5e-4`

**Giải pháp**:
```python
# Thử tăng learning rate nếu epochs vẫn ít
parser.add_argument('--rerank_lr', type=float, default=2e-4,  # ✅ Tăng lên 2e-4
```

---

### **3. Model chưa được train đủ** ⚠️

**Triệu chứng**:
- Recall@20 ≈ 0.4 (gần random)
- Model có thể đang predict gần như uniform distribution

**Kiểm tra**:
```python
# Kiểm tra training loss
# Nếu loss > 3.0 sau training → model chưa học được gì
# Với 50 candidates: random loss ≈ -log(1/50) ≈ 3.9
```

**Giải pháp**:
- Tăng epochs
- Kiểm tra training loss có giảm không
- Kiểm tra validation loss

---

### **4. LoRA Config có thể chưa tối ưu** ⚠️

**Code hiện tại** (`rerank/models/llm.py:146-154`):
```python
self.model = FastLanguageModel.get_peft_model(
    self.model,
    r = 8,              # LoRA rank
    target_modules = ["q_proj","k_proj","v_proj","o_proj"],
    lora_alpha = 16,   # LoRA alpha
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
)
```

**Phân tích**:
- `r=8, alpha=16` là khá nhỏ
- LlamaRec và các paper thường dùng `r=16-32, alpha=32-64`
- Với model nhỏ (Qwen3-0.6B), `r=8` có thể đủ, nhưng có thể thử tăng

**Giải pháp**:
```python
# Thử tăng LoRA rank
r = 16,              # Tăng từ 8 lên 16
lora_alpha = 32,     # Tăng từ 16 lên 32
```

---

### **5. Evaluation Setup có thể có vấn đề** ⚠️

**Kiểm tra evaluation process**:

1. **Candidates sampling**:
   - Có đảm bảo GT item có trong candidates không?
   - Có shuffle candidates để tránh bias không?

2. **History exclusion**:
   - Có exclude history items khỏi candidates không?

3. **Prompt format**:
   - Prompt có đúng format không?
   - Model có hiểu được task không?

**Code evaluation** (`rerank/methods/qwen_reranker_unified.py:1333-1374`):
```python
def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> float:
    recalls = []
    for user_id, gt_items in split.items():
        # ... get candidates ...
        # Rerank
        reranked = self.rerank(user_id, candidates)
        # Compute recall
        top_k_items = [item_id for item_id, _ in reranked[:k]]
        hits = len(set(top_k_items) & set(gt_items))
        recalls.append(hits / len(gt_items))
    return float(np.mean(recalls))
```

**Vấn đề có thể có**:
- Candidates có thể không chứa GT items
- Model có thể không predict đúng letter tokens
- Probabilities có thể bị uniform

---

### **6. Model Prediction có vấn đề** ⚠️

**Kiểm tra logits extraction**:

1. **Letter tokens có được tìm thấy không?**
```python
# rerank/models/llm.py:386-393
if len(letter_tokens) < num_candidates:
    print(f"[WARNING] Only found {len(letter_tokens)}/{num_candidates} letter tokens!")
    # Nếu không tìm thấy letter tokens → fallback to uniform
```

2. **Probabilities có bị uniform không?**
```python
# Kiểm tra output probabilities
probs = self.llm_model.predict_probs(prompt, num_candidates=len(candidates))
print(f"Probs: {probs}")  # Nếu gần uniform → model chưa học được gì
```

3. **Model có predict đúng letter không?**
```python
# Kiểm tra predicted letter
predicted_letter = LETTERS[np.argmax(probs)]
print(f"Predicted: {predicted_letter}, GT: {gt_letter}")
```

---

## 🎯 Giải pháp ưu tiên

### **Priority 1: Tăng Epochs (CRITICAL)** 🔴

```python
# config.py
parser.add_argument('--rerank_epochs', type=int, default=5,  # ✅ Tăng từ 1 lên 5
                    help='Number of training epochs for rerank models.')
```

**Lý do**:
- 1 epoch quá ít để model học được pattern
- Với 1635 samples, cần ít nhất 3-5 epochs
- LlamaRec thường train 3-10 epochs

---

### **Priority 2: Kiểm tra Training Loss** 🟡

```python
# Thêm logging để kiểm tra training loss
# Nếu loss không giảm → có vấn đề với training
```

**Expected behavior**:
- Initial loss: ~3.9 (random với 50 candidates)
- After 1 epoch: ~2.0-3.0 (nếu model học được một chút)
- After 5 epochs: ~1.0-2.0 (nếu model học tốt)

---

### **Priority 3: Tăng LoRA Rank (nếu cần)** 🟡

```python
# rerank/models/llm.py
self.model = FastLanguageModel.get_peft_model(
    self.model,
    r = 16,              # ✅ Tăng từ 8 lên 16
    lora_alpha = 32,     # ✅ Tăng từ 16 lên 32
    ...
)
```

**Lý do**:
- Tăng model capacity
- Có thể improve performance
- Trade-off: chậm hơn một chút

---

### **Priority 4: Kiểm tra Evaluation** 🟢

1. **Debug prediction**:
```python
# Thêm debug output
probs = self.llm_model.predict_probs(prompt, num_candidates=len(candidates))
print(f"Probs: {probs[:5]}")  # Top 5 probabilities
print(f"Max prob: {np.max(probs)}, Min prob: {np.min(probs)}")
```

2. **Kiểm tra letter tokens**:
```python
# Kiểm tra xem letter tokens có được tìm thấy không
# Nếu không → có vấn đề với tokenizer
```

3. **Kiểm tra candidates**:
```python
# Đảm bảo GT items có trong candidates
# Nếu không → recall sẽ = 0
```

---

## 📊 Expected Results sau khi sửa

### **Sau khi tăng epochs lên 5**:

**Expected**:
- Training loss: ~1.5-2.5 (giảm từ ~3.9)
- Recall@20: **0.6-0.8** (tăng từ 0.4)
- NDCG@10: **0.3-0.5** (tăng từ ~0.1)

**Nếu vẫn thấp**:
- Kiểm tra training data quality
- Kiểm tra model size (có thể cần model lớn hơn)
- Kiểm tra evaluation setup

---

## 🔧 Action Items

### **Immediate (CRITICAL)**:

1. ✅ **Tăng epochs lên 5-10**:
   ```python
   # config.py
   --rerank_epochs 5
   ```

2. ✅ **Kiểm tra training loss**:
   - Xem loss có giảm không
   - Nếu không giảm → có vấn đề với training

3. ✅ **Debug prediction**:
   - Kiểm tra probabilities có uniform không
   - Kiểm tra letter tokens có được tìm thấy không

### **Next Steps**:

4. ✅ **Tăng LoRA rank** (nếu epochs không đủ):
   ```python
   r = 16, alpha = 32
   ```

5. ✅ **Tăng learning rate** (nếu cần):
   ```python
   --rerank_lr 2e-4
   ```

6. ✅ **Kiểm tra evaluation setup**:
   - Đảm bảo GT items có trong candidates
   - Kiểm tra prompt format

---

## 📚 References

- **LlamaRec**: Thường train 3-10 epochs
- **LoRA best practices**: r=16-32, alpha=32-64 cho better performance
- **Learning rate**: 1e-4 đến 5e-4 cho LoRA fine-tuning

---

## ✅ Tóm tắt

**Nguyên nhân chính**: **Epochs quá ít (1 epoch)**

**Giải pháp**:
1. ✅ Tăng `--rerank_epochs` lên 5-10
2. ✅ Kiểm tra training loss
3. ✅ Debug prediction để đảm bảo model học được

**Expected improvement**: Recall@20 từ 0.4 → **0.6-0.8** sau khi tăng epochs

