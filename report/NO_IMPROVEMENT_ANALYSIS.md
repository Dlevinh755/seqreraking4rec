# Phân tích: Tại sao kết quả không cải thiện sau 4 epochs?

## 📊 Vấn đề

**Đã thử training với 4 epochs nhưng kết quả không cải thiện** (Recall@20 vẫn ~0.4)

Điều này cho thấy có vấn đề khác ngoài epochs. Cần kiểm tra:

---

## 🔍 Các Nguyên nhân Có thể

### **1. Training Loss không giảm** 🔴

**Kiểm tra**:
- Training loss có giảm không?
- Nếu loss không giảm → model không học được gì

**Expected behavior**:
- Initial loss: ~3.9 (random với 50 candidates: -log(1/50) ≈ 3.9)
- Sau 1 epoch: ~2.5-3.5 (nếu học được một chút)
- Sau 4 epochs: ~1.5-2.5 (nếu học tốt)

**Nếu loss không giảm**:
- ❌ Learning rate quá thấp
- ❌ Model không được train (checkpoint không được save/load)
- ❌ Training data format sai
- ❌ Loss masking không đúng

**Cách kiểm tra**:
```python
# Thêm logging trong training
training_args = SFTConfig(
    ...
    logging_steps=10,  # ✅ Log mỗi 10 steps
    report_to="tensorboard",  # Hoặc "wandb"
)
```

---

### **2. Model không Predict đúng Letter Tokens** 🔴

**Vấn đề**:
- Model có thể không tìm thấy letter tokens (A, B, C, ...)
- Fallback về uniform distribution → recall = random

**Code kiểm tra** (`rerank/models/llm.py:386-393`):
```python
# Debug: Check if we found letter tokens
if len(letter_tokens) < num_candidates:
    print(f"[WARNING] Only found {len(letter_tokens)}/{num_candidates} letter tokens!")
    # Nếu không tìm thấy → fallback to uniform
    if len(letter_tokens) == 0:
        return np.ones(num_candidates) / num_candidates  # ❌ Uniform!
```

**Cách kiểm tra**:
1. **Thêm debug output**:
```python
# Trong predict_probs()
print(f"[DEBUG] Found {len(letter_tokens)}/{num_candidates} letter tokens")
print(f"[DEBUG] Letter tokens: {[l for _, l, _ in letter_tokens[:5]]}")
print(f"[DEBUG] Probabilities: {prob_array[:5]}")
```

2. **Kiểm tra tokenizer**:
```python
# Test tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
for letter in ["A", "B", "C", "D", "E"]:
    token_id = tokenizer.convert_tokens_to_ids(letter)
    print(f"Letter {letter}: token_id={token_id}, unk={tokenizer.unk_token_id}")
```

**Nếu không tìm thấy letter tokens**:
- ❌ Tokenizer không hỗ trợ single letter tokens
- ❌ Cần dùng strategy khác (space prefix, encoding)

---

### **3. Model Predict Uniform Distribution** 🔴

**Vấn đề**:
- Model có thể predict gần như uniform distribution
- Tất cả candidates có probability ≈ 1/num_candidates
- → Recall = random

**Cách kiểm tra**:
```python
# Trong rerank()
probs = self.llm_model.predict_probs(prompt, num_candidates=len(candidates))
print(f"[DEBUG] Probabilities: {probs}")
print(f"[DEBUG] Max prob: {np.max(probs)}, Min prob: {np.min(probs)}")
print(f"[DEBUG] Std: {np.std(probs)}")

# Nếu std rất nhỏ → uniform distribution
if np.std(probs) < 0.01:
    print("[WARNING] Probabilities are nearly uniform!")
```

**Nguyên nhân**:
- ❌ Model chưa học được pattern (loss không giảm)
- ❌ Training data không đủ quality
- ❌ Model quá nhỏ (Qwen3-0.6B có thể không đủ)

---

### **4. Evaluation Setup có vấn đề** 🟡

**Vấn đề**:
- GT items có thể không có trong candidates
- Candidates có thể không được shuffle đúng
- Evaluation có thể không đúng

**Code kiểm tra** (`rerank/methods/qwen_reranker_unified.py:1333-1374`):
```python
def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> float:
    recalls = []
    for user_id, gt_items in split.items():
        # ... get candidates ...
        
        # ✅ Đảm bảo GT item có trong candidates
        if not any(item in candidates for item in gt_items):
            candidates[0] = gt_items[0]  # ✅ Force GT vào candidates
        
        # Rerank
        reranked = self.rerank(user_id, candidates)
        
        # Compute recall
        top_k_items = [item_id for item_id, _ in reranked[:k]]
        hits = len(set(top_k_items) & set(gt_items))
        recalls.append(hits / len(gt_items))
```

**Cách kiểm tra**:
```python
# Thêm debug trong evaluation
print(f"[DEBUG] User {user_id}: GT={gt_items}, Candidates={candidates[:5]}")
print(f"[DEBUG] Reranked top-5: {[item_id for item_id, _ in reranked[:5]]}")
print(f"[DEBUG] Hits: {hits}, Recall: {hits / len(gt_items)}")
```

---

### **5. Training Data Quality** 🟡

**Vấn đề**:
- Training data có thể không đủ quality
- History có thể quá ngắn
- Candidates có thể không đa dạng

**Cách kiểm tra**:
```python
# Kiểm tra training data
print(f"Training samples: {len(train_data_for_llm)}")
print(f"Sample prompt length: {len(train_data_for_llm[0]['messages'][1]['content'])}")
print(f"Sample target: {train_data_for_llm[0]['messages'][2]['content']}")

# Kiểm tra distribution của target letters
from collections import Counter
targets = [sample['messages'][2]['content'] for sample in train_data_for_llm]
target_counts = Counter(targets)
print(f"Target distribution: {target_counts}")
# Nếu quá imbalanced → có thể ảnh hưởng training
```

---

### **6. Model Size có thể quá nhỏ** 🟡

**Vấn đề**:
- Qwen3-0.6B có thể quá nhỏ cho task này
- LlamaRec dùng Llama 2-7B (lớn hơn 10x)

**So sánh**:
- **LlamaRec**: Llama 2-7B (7B parameters)
- **Project hiện tại**: Qwen3-0.6B (0.6B parameters)

**Giải pháp**:
- Thử model lớn hơn: Qwen3-1.7B, Qwen3-4B
- Hoặc tăng LoRA rank: r=16, alpha=32

---

### **7. Learning Rate có thể chưa tối ưu** 🟡

**Vấn đề**:
- Learning rate 1e-4 có thể quá thấp hoặc quá cao
- Cần điều chỉnh dựa trên training loss

**Cách kiểm tra**:
- Nếu loss không giảm → tăng learning rate (2e-4, 5e-4)
- Nếu loss oscillate → giảm learning rate (5e-5)

---

## 🔧 Debugging Steps

### **Step 1: Kiểm tra Training Loss**

```python
# Thêm vào training
training_args = SFTConfig(
    ...
    logging_steps=1,  # Log mỗi step để debug
    report_to="none",  # Hoặc "tensorboard"
)

# Sau training, check logs
# Nếu loss không giảm → có vấn đề với training
```

---

### **Step 2: Kiểm tra Letter Token Extraction**

```python
# Thêm debug trong predict_probs()
def predict_probs(self, prompt, num_candidates=None):
    # ... existing code ...
    
    # ✅ DEBUG: Check letter tokens
    print(f"[DEBUG] Looking for {num_candidates} letter tokens")
    print(f"[DEBUG] Found {len(letter_tokens)} letter tokens")
    if len(letter_tokens) < num_candidates:
        print(f"[WARNING] Missing {num_candidates - len(letter_tokens)} letter tokens!")
        print(f"[DEBUG] Found letters: {[l for _, l, _ in letter_tokens]}")
    
    # ✅ DEBUG: Check probabilities
    print(f"[DEBUG] Probabilities: max={np.max(prob_array):.4f}, min={np.min(prob_array):.4f}, std={np.std(prob_array):.4f}")
    
    return prob_array
```

---

### **Step 3: Kiểm tra Model Prediction**

```python
# Test prediction trên một sample
prompt = """You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
- item1
- item2

Candidate items:
A. candidate1
B. candidate2
C. candidate3

Answer with only one letter (A-C)."""

probs = model.predict_probs(prompt, num_candidates=3)
print(f"Probabilities: {probs}")
print(f"Predicted letter: {LETTERS[np.argmax(probs)]}")

# Nếu probabilities gần uniform → model chưa học được gì
```

---

### **Step 4: Kiểm tra Evaluation**

```python
# Thêm debug trong _evaluate_split()
def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> float:
    recalls = []
    for i, (user_id, gt_items) in enumerate(split.items()):
        # ... existing code ...
        
        # ✅ DEBUG: Check first few samples
        if i < 3:
            print(f"\n[DEBUG] User {user_id}:")
            print(f"  GT items: {gt_items}")
            print(f"  Candidates: {candidates[:10]}")
            print(f"  GT in candidates: {any(item in candidates for item in gt_items)}")
            
            reranked = self.rerank(user_id, candidates)
            print(f"  Reranked top-5: {[item_id for item_id, _ in reranked[:5]]}")
            
            top_k_items = [item_id for item_id, _ in reranked[:k]]
            hits = len(set(top_k_items) & set(gt_items))
            recall = hits / len(gt_items)
            print(f"  Hits: {hits}, Recall@{k}: {recall:.4f}")
    
    return float(np.mean(recalls))
```

---

## 🎯 Action Plan

### **Priority 1: Debug Training Loss** 🔴

1. **Thêm logging**:
```python
training_args = SFTConfig(
    logging_steps=1,  # Log mỗi step
    ...
)
```

2. **Kiểm tra loss có giảm không**:
- Nếu không giảm → có vấn đề với training
- Nếu giảm nhưng chậm → tăng learning rate

---

### **Priority 2: Debug Letter Token Extraction** 🔴

1. **Thêm debug output** trong `predict_probs()`
2. **Kiểm tra xem letter tokens có được tìm thấy không**
3. **Nếu không tìm thấy → sửa token extraction**

---

### **Priority 3: Debug Model Prediction** 🟡

1. **Test prediction trên sample cụ thể**
2. **Kiểm tra probabilities có uniform không**
3. **Nếu uniform → model chưa học được gì**

---

### **Priority 4: Kiểm tra Evaluation** 🟡

1. **Thêm debug trong evaluation**
2. **Kiểm tra GT items có trong candidates không**
3. **Kiểm tra reranking có đúng không**

---

## 📊 Expected Results sau khi Debug

### **Nếu Training Loss giảm nhưng Recall không cải thiện**:

- ❌ Vấn đề với letter token extraction
- ❌ Vấn đề với evaluation setup
- ❌ Model predict uniform distribution

### **Nếu Training Loss không giảm**:

- ❌ Learning rate quá thấp/cao
- ❌ Training data format sai
- ❌ Model không được train đúng

### **Nếu Letter Tokens không được tìm thấy**:

- ❌ Tokenizer không hỗ trợ single letters
- ❌ Cần sửa token extraction strategy

---

## ✅ Tóm tắt

**Vấn đề**: Training 4 epochs nhưng kết quả không cải thiện

**Các nguyên nhân có thể**:
1. 🔴 Training loss không giảm
2. 🔴 Letter tokens không được tìm thấy
3. 🔴 Model predict uniform distribution
4. 🟡 Evaluation setup có vấn đề
5. 🟡 Training data quality
6. 🟡 Model size quá nhỏ
7. 🟡 Learning rate chưa tối ưu

**Next Steps**:
1. ✅ Debug training loss
2. ✅ Debug letter token extraction
3. ✅ Debug model prediction
4. ✅ Debug evaluation setup

