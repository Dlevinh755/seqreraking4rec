# Hướng dẫn Kiểm tra Training Loss

## 🔍 Vấn đề

**Tests pass nhưng recall thấp (0.4)** → Có thể training loss không giảm

---

## 📊 Cách Kiểm tra Training Loss

### **1. Xem Training Logs**

Khi training, bạn sẽ thấy logs như:

```
[TRAINING] Step 1: loss=3.9123
[TRAINING] Step 2: loss=3.8456
[TRAINING] Step 3: loss=3.7891
...
```

**Expected behavior**:
- **Initial loss**: ~3.9 (random với 50 candidates: -log(1/50) ≈ 3.9)
- **Sau 1 epoch**: ~2.5-3.5 (nếu học được một chút)
- **Sau 4 epochs**: ~1.5-2.5 (nếu học tốt)

**Nếu loss không giảm**:
- ❌ Model không học được gì
- ❌ Có vấn đề với training process

---

### **2. Kiểm tra Loss Progression**

**Good training**:
```
Step 1: loss=3.9123
Step 10: loss=3.5123
Step 20: loss=3.1234
Step 30: loss=2.7891
...
Step 100: loss=2.1234
Step 200: loss=1.7891
```

**Bad training (loss không giảm)**:
```
Step 1: loss=3.9123
Step 10: loss=3.9123
Step 20: loss=3.9123
Step 30: loss=3.9123
...
```

---

### **3. Nếu Loss không giảm**

**Nguyên nhân có thể**:

1. **Learning rate quá thấp**:
   - Current: 1e-4
   - Thử: 2e-4, 5e-4

2. **Learning rate quá cao**:
   - Loss oscillate hoặc NaN
   - Thử: 5e-5

3. **Training data format sai**:
   - Check training data có đúng format không
   - Check target labels có đúng không

4. **Model không được train**:
   - Check model có được save/load không
   - Check model state (training vs eval mode)

---

## 🔧 Action Plan

### **Step 1: Kiểm tra Training Logs**

Khi chạy training, xem logs:
- Loss có giảm không?
- Loss có oscillate không?
- Loss có NaN không?

### **Step 2: Nếu Loss không giảm**

1. **Tăng learning rate**:
   ```python
   # config.py
   --rerank_lr 2e-4  # Hoặc 5e-4
   ```

2. **Kiểm tra training data**:
   ```python
   # Check training data format
   print(f"Training samples: {len(train_data_for_llm)}")
   print(f"Sample: {train_data_for_llm[0]}")
   ```

3. **Kiểm tra model state**:
   ```python
   # Sau training
   print(f"Model training mode: {model.training}")
   model.eval()  # ✅ Đảm bảo eval mode cho inference
   ```

### **Step 3: Nếu Loss giảm nhưng Recall vẫn thấp**

1. **Kiểm tra evaluation**:
   - GT items có trong candidates không?
   - Candidates có được shuffle không?

2. **Test với nhiều candidates**:
   - Model có thể tốt với ít candidates nhưng kém với nhiều candidates

3. **Kiểm tra model size**:
   - Qwen3-0.6B có thể quá nhỏ
   - Thử model lớn hơn: Qwen3-1.7B, Qwen3-4B

---

## ✅ Tóm tắt

**Nếu tests pass nhưng recall thấp**:

1. ✅ **Kiểm tra training loss có giảm không** (CRITICAL)
2. ✅ **Nếu loss không giảm** → Tăng learning rate hoặc kiểm tra training data
3. ✅ **Nếu loss giảm nhưng recall thấp** → Kiểm tra evaluation setup hoặc model size

**Next step**: Chạy training lại và xem training logs để kiểm tra loss có giảm không!

