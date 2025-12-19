# Deep Debug: Tại sao Recall thấp dù Tests Pass?

## 📊 Tình huống

**Debug tests đều PASS** nhưng **Recall@20 vẫn ~0.4** (gần random)

**Tests passed**:
- ✅ Letter tokens được tìm thấy
- ✅ Model prediction không uniform (std=0.34)
- ✅ Training data format đúng
- ✅ Evaluation setup đúng

**Vấn đề**: Model có thể không được train đúng hoặc không được sử dụng đúng sau training.

---

## 🔍 Các Vấn đề Tiềm ẩn

### **1. Model không được Save/Load sau Training** 🔴

**Vấn đề**:
- Model được train nhưng không được save
- Hoặc model được save nhưng không được load lại
- Inference dùng model chưa được train

**Kiểm tra**:
```python
# Trong fit() method
# Sau khi train, model có được save không?
# Khi rerank(), model có được load từ checkpoint không?
```

**Code hiện tại** (`rerank/models/llm.py:228-246`):
```python
training_args = SFTConfig(
    output_dir="./qwen_rerank",
    save_steps=500,  # ✅ Save mỗi 500 steps
    load_best_model_at_end=False,  # ❌ KHÔNG load best model!
    ...
)
```

**Vấn đề**:
- `load_best_model_at_end=False` → Model không tự động load best checkpoint
- Model sau training có thể không phải là best model
- Cần manually load checkpoint hoặc save final model

---

### **2. Training Loss không được Log đầy đủ** 🟡

**Vấn đề**:
- `logging_steps=1` (đã sửa) nhưng có thể không đủ
- Cần kiểm tra loss có giảm không

**Cách kiểm tra**:
```python
# Thêm vào training
training_args = SFTConfig(
    logging_steps=1,  # ✅ Log mỗi step
    report_to="tensorboard",  # Hoặc "wandb" để track loss
    ...
)
```

---

### **3. Model được Test trên Sample đơn giản** 🟡

**Vấn đề**:
- Debug script test trên sample đơn giản (5 candidates)
- Trong thực tế có thể có 20-50 candidates
- Model có thể predict tốt với ít candidates nhưng kém với nhiều candidates

**Kiểm tra**:
```python
# Test với số lượng candidates giống evaluation
probs = model.predict_probs(prompt, num_candidates=20)  # Hoặc 50
print(f"Std với 20 candidates: {np.std(probs)}")
# Nếu std giảm → model kém với nhiều candidates
```

---

### **4. Training Data Quality** 🟡

**Vấn đề**:
- Training data có thể không đủ quality
- History quá ngắn
- Candidates không đa dạng

**Kiểm tra**:
```python
# Kiểm tra training data distribution
from collections import Counter
targets = [sample['messages'][2]['content'] for sample in train_data_for_llm]
target_counts = Counter(targets)
print(f"Target distribution: {target_counts}")

# Nếu quá imbalanced → có thể ảnh hưởng training
# Ví dụ: 80% là "A", 20% là các letters khác
```

---

### **5. Model Size có thể quá nhỏ** 🟡

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

### **6. Evaluation Candidates khác Training** 🔴

**Vấn đề**:
- Training: Candidates được sample từ all_items
- Evaluation: Candidates có thể từ pre-generated list
- Distribution khác nhau → model không generalize

**Kiểm tra**:
```python
# So sánh training candidates vs evaluation candidates
# Training: random sample từ all_items
# Evaluation: pre-generated candidates (có thể có bias)
```

---

## 🔧 Debugging Steps

### **Step 1: Kiểm tra Model có được Save/Load không**

```python
# Thêm vào fit() sau training
print(f"[DEBUG] Model state after training:")
print(f"  Output dir: {training_args.output_dir}")
print(f"  Checkpoints saved: {os.listdir(training_args.output_dir) if os.path.exists(training_args.output_dir) else 'None'}")

# Kiểm tra xem model có được load lại không
print(f"[DEBUG] Model device: {self.model.device}")
print(f"[DEBUG] Model dtype: {self.model.dtype}")
```

---

### **Step 2: Kiểm tra Training Loss**

```python
# Thêm callback để track loss
from transformers import TrainerCallback

class LossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            print(f"Step {state.global_step}: loss={logs['loss']:.4f}")

# Thêm vào trainer
trainer.add_callback(LossCallback())
```

---

### **Step 3: Test với số lượng Candidates giống Evaluation**

```python
# Test với 20-50 candidates (giống evaluation)
prompt = build_prompt_from_candidates(
    history,
    list(range(20)),  # 20 candidates
    item_id2text,
    max_candidates=20
)

probs = model.predict_probs(prompt, num_candidates=20)
print(f"Std với 20 candidates: {np.std(probs)}")
print(f"Max prob: {np.max(probs):.4f}, Min prob: {np.min(probs):.4f}")

# Nếu std < 0.1 với 20 candidates → model kém với nhiều candidates
```

---

### **Step 4: Kiểm tra Training Data Distribution**

```python
# Kiểm tra target distribution
from collections import Counter
targets = [sample['messages'][2]['content'] for sample in train_data_for_llm]
target_counts = Counter(targets)

print(f"Target distribution:")
for letter, count in sorted(target_counts.items()):
    print(f"  {letter}: {count} ({count/len(targets)*100:.1f}%)")

# Nếu quá imbalanced → có thể ảnh hưởng training
```

---

### **Step 5: So sánh Training vs Evaluation Candidates**

```python
# Kiểm tra distribution của candidates
# Training: random sample
# Evaluation: pre-generated

# Training candidates
train_candidates = [sample['candidates'] for sample in train_samples]
train_candidate_items = set()
for cands in train_candidates:
    train_candidate_items.update(cands)

# Evaluation candidates
eval_candidates = load_rerank_candidates(...)
eval_candidate_items = set()
for user_cands in eval_candidates.values():
    eval_candidate_items.update(user_cands)

print(f"Training candidate items: {len(train_candidate_items)}")
print(f"Evaluation candidate items: {len(eval_candidate_items)}")
print(f"Overlap: {len(train_candidate_items & eval_candidate_items)}")

# Nếu overlap thấp → distribution khác nhau
```

---

## 🎯 Action Plan

### **Priority 1: Kiểm tra Model Save/Load** 🔴

1. **Thêm logging**:
```python
# Sau training
print(f"[DEBUG] Model output dir: {training_args.output_dir}")
if os.path.exists(training_args.output_dir):
    checkpoints = [f for f in os.listdir(training_args.output_dir) if 'checkpoint' in f]
    print(f"[DEBUG] Checkpoints: {checkpoints}")
else:
    print(f"[DEBUG] Output dir does not exist!")
```

2. **Kiểm tra model có được save không**:
```python
# Save model manually sau training
trainer.save_model("./qwen_rerank_final")
print(f"[DEBUG] Model saved to ./qwen_rerank_final")
```

3. **Kiểm tra model có được load lại không**:
```python
# Khi rerank, kiểm tra model state
print(f"[DEBUG] Model device: {self.model.device}")
print(f"[DEBUG] Model is training mode: {self.model.training}")
self.model.eval()  # ✅ Đảm bảo eval mode
```

---

### **Priority 2: Kiểm tra Training Loss** 🔴

1. **Thêm loss tracking**:
```python
# Thêm callback
from transformers import TrainerCallback

class LossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            print(f"[TRAINING] Step {state.global_step}: loss={logs['loss']:.4f}")

trainer.add_callback(LossCallback())
```

2. **Kiểm tra loss có giảm không**:
- Initial loss: ~3.9 (random)
- Sau 1 epoch: ~2.5-3.5
- Sau 4 epochs: ~1.5-2.5

---

### **Priority 3: Test với nhiều Candidates** 🟡

```python
# Test với 20-50 candidates
for num_cand in [5, 10, 20, 50]:
    prompt = build_prompt_with_n_candidates(num_cand)
    probs = model.predict_probs(prompt, num_candidates=num_cand)
    print(f"Candidates={num_cand}: std={np.std(probs):.4f}")
    
    # Nếu std giảm khi tăng candidates → model kém với nhiều candidates
```

---

### **Priority 4: Kiểm tra Training Data Quality** 🟡

```python
# Kiểm tra distribution
targets = [sample['messages'][2]['content'] for sample in train_data_for_llm]
target_counts = Counter(targets)

# Nếu quá imbalanced → có thể ảnh hưởng
if max(target_counts.values()) / len(targets) > 0.5:
    print(f"[WARNING] Training data is imbalanced!")
    print(f"  Most common target: {max(target_counts, key=target_counts.get)} ({max(target_counts.values())/len(targets)*100:.1f}%)")
```

---

## 📊 Expected Results

### **Nếu Model được Train đúng**:

- Training loss giảm từ ~3.9 → ~1.5-2.5
- Model predict không uniform (std > 0.1)
- Recall@20 > 0.6 (không phải 0.4)

### **Nếu Model không được Train đúng**:

- Training loss không giảm hoặc giảm rất ít
- Model predict gần uniform
- Recall@20 ≈ 0.4 (random)

---

## ✅ Tóm tắt

**Tests pass nhưng recall thấp** → Vấn đề có thể là:

1. 🔴 **Model không được save/load đúng** (CRITICAL)
2. 🔴 **Training loss không giảm** (CRITICAL)
3. 🟡 **Model kém với nhiều candidates** (20-50)
4. 🟡 **Training data quality** (imbalanced)
5. 🟡 **Model size quá nhỏ** (Qwen3-0.6B)

**Next Steps**:
1. ✅ Kiểm tra model save/load
2. ✅ Kiểm tra training loss
3. ✅ Test với nhiều candidates
4. ✅ Kiểm tra training data quality

