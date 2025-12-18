# Phân tích: Tại sao kết quả thấp?

## 📊 Kết quả hiện tại

- **Val Recall@1**: 0.0179 (1.79%) - Rất thấp
- **Test Recall@1**: 0.0212 (2.12%) - Rất thấp
- **Val NDCG@10**: 0.0875
- **Test NDCG@10**: 0.1019
- **Training Loss**: 4.25 (Rất cao, thường nên < 2)

## 🔍 Nguyên nhân chính

### 1. **Quá ít Training Data và Epochs** ⚠️ CRITICAL

**Vấn đề**:
- **Training samples**: 614 (rất ít)
- **Epochs**: 1 (quá ít)
- **Training steps**: 5 (614 / 128 batch size = ~5 steps)
- **Training loss**: 4.25 (rất cao, model chưa học được gì)

**Phân tích**:
- Với 614 samples và 1 epoch, model chỉ thấy mỗi sample 1 lần
- 5 steps là quá ít để model học được pattern
- Loss 4.25 cho thấy model gần như random (cross-entropy với 50 classes ≈ -log(1/50) ≈ 3.9)

**Giải pháp**:
```python
# Tăng epochs trong config.py
--rerank_epochs 10  # Thay vì 1

# Hoặc trong code
num_epochs = 10  # Thay vì 1
```

### 2. **Learning Rate không được sử dụng từ Config** ⚠️

**Vấn đề**:
- Code hardcode `learning_rate=2e-5` trong `SFTConfig` (line 228)
- Config có `--rerank_lr=1e-4` nhưng không được sử dụng
- Learning rate 2e-5 có thể quá thấp cho fine-tuning

**Code hiện tại** (`rerank/models/llm.py:228`):
```python
training_args = SFTConfig(
    ...
    learning_rate=2e-5,  # ❌ Hardcoded, không dùng từ config
    ...
)
```

**Giải pháp**:
```python
# Lấy learning rate từ config
try:
    from config import arg
    lr = getattr(arg, 'rerank_lr', 2e-5)
except ImportError:
    lr = 2e-5

training_args = SFTConfig(
    ...
    learning_rate=lr,  # ✅ Dùng từ config
    ...
)
```

### 3. **Dataset quá nhỏ** ⚠️

**Vấn đề**:
- **Users**: 614 (rất ít)
- **Items**: 474 (rất ít)
- **Training samples**: 614 (1 sample/user)
- **Diversity**: Thấp, model khó học pattern

**Phân tích**:
- Với dataset nhỏ, model dễ overfit hoặc underfit
- Cần nhiều epochs hơn để model học được pattern
- Có thể cần data augmentation hoặc transfer learning

### 4. **Model quá nhỏ** ⚠️

**Vấn đề**:
- **Model**: Qwen3-0.6B (600M parameters)
- **Task**: Rerank 50 candidates (khá phức tạp)
- **Capacity**: Có thể không đủ để học pattern tốt

**Giải pháp**:
- Thử model lớn hơn: `qwen3-1.6b` hoặc `qwen3-2b`
- Hoặc tăng LoRA rank để tăng capacity

### 5. **Training Loss quá cao** ⚠️

**Vấn đề**:
- **Training loss**: 4.25 (rất cao)
- **Expected loss**: < 2.0 (cho 50 classes)
- **Random baseline**: ~3.9 (-log(1/50))

**Phân tích**:
- Loss 4.25 gần với random (3.9)
- Model chưa học được gì hữu ích
- Cần nhiều epochs để loss giảm xuống

### 6. **Format mới (Letters) có thể chưa được train đủ** ⚠️

**Vấn đề**:
- Mới chuyển từ numbers sang letters
- Model cần học lại cách predict letters
- Với 1 epoch, model chưa kịp học

**Giải pháp**:
- Tăng epochs để model học được letter prediction
- Hoặc thử lại với numbers để so sánh

### 7. **Chat Template Format mới** ⚠️

**Vấn đề**:
- Mới sửa để dùng chat template format cho inference
- Training và inference giờ đã consistent
- Nhưng model cần thời gian để học format mới

**Giải pháp**:
- Tăng epochs để model học được format mới
- Monitor training loss để đảm bảo đang giảm

## 🎯 Giải pháp đề xuất

### **Priority 1: Tăng Epochs** (QUAN TRỌNG NHẤT)

```bash
# Sửa config.py hoặc command line
--rerank_epochs 10  # Thay vì 1
```

**Lý do**:
- 1 epoch là quá ít, model chưa kịp học
- Với 614 samples, cần ít nhất 5-10 epochs
- Loss 4.25 cho thấy model chưa converge

### **Priority 2: Sửa Learning Rate từ Config**

**Sửa code** (`rerank/models/llm.py:228`):
```python
# Lấy learning rate từ config
try:
    from config import arg
    lr = getattr(arg, 'rerank_lr', 2e-5)
except ImportError:
    lr = 2e-5

training_args = SFTConfig(
    ...
    learning_rate=lr,  # ✅ Dùng từ config (default: 1e-4)
    ...
)
```

**Lý do**:
- Config có `--rerank_lr=1e-4` nhưng không được dùng
- 1e-4 thường tốt hơn 2e-5 cho fine-tuning
- Cần consistency giữa config và code

### **Priority 3: Tăng Batch Size (nếu GPU cho phép)**

```bash
--rerank_batch_size 32  # Thay vì 16
```

**Lý do**:
- Batch size lớn hơn → training ổn định hơn
- Gradient accumulation steps = 4, nên effective batch = 128
- Có thể tăng lên 64 nếu GPU memory cho phép

### **Priority 4: Monitor Training Loss**

**Kiểm tra**:
- Training loss có giảm không?
- Nếu không giảm → learning rate quá thấp hoặc model không học được
- Nếu giảm quá nhanh → có thể overfit

### **Priority 5: Thử Model lớn hơn**

```bash
--qwen_model qwen3-1.6b  # Thay vì qwen3-0.6b
```

**Lý do**:
- Model lớn hơn có capacity tốt hơn
- Có thể học được pattern phức tạp hơn
- Trade-off: chậm hơn, tốn memory hơn

## 📈 Expected Results sau khi sửa

**Với epochs=10, lr=1e-4**:
- Training loss: < 2.0 (sau 10 epochs)
- Val Recall@1: > 0.05 (5%)
- Test Recall@1: > 0.05 (5%)
- NDCG@10: > 0.15

**Với epochs=20, lr=1e-4, model lớn hơn**:
- Training loss: < 1.5
- Val Recall@1: > 0.10 (10%)
- Test Recall@1: > 0.10 (10%)
- NDCG@10: > 0.20

## 🔧 Quick Fix

**Command line**:
```bash
python scripts/train_rerank_standalone.py \
    --rerank_method qwen3vl \
    --rerank_mode ground_truth \
    --rerank_epochs 10 \
    --rerank_lr 1e-4 \
    --rerank_batch_size 32
```

**Hoặc sửa config.py**:
```python
parser.add_argument('--rerank_epochs', type=int, default=10,  # Thay vì 1
parser.add_argument('--rerank_lr', type=float, default=1e-4,  # Đã đúng
parser.add_argument('--rerank_batch_size', type=int, default=32,  # Thay vì 16
```

## 📝 Kết luận

**Nguyên nhân chính**: 
1. **Quá ít epochs (1)** → Model chưa kịp học
2. **Learning rate không được dùng từ config** → Có thể không tối ưu
3. **Training loss cao (4.25)** → Model gần như random

**Giải pháp**:
1. ✅ Tăng epochs lên 10-20
2. ✅ Sửa code để dùng learning rate từ config
3. ✅ Monitor training loss để đảm bảo đang giảm
4. ⚠️ Cân nhắc model lớn hơn nếu vẫn thấp

