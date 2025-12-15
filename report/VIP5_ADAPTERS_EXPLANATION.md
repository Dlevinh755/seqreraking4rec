# VIP5 Adapters - Giải thích và Tác dụng

## ✅ Trạng thái hiện tại

**Adapters đã có trong project!** 

Folder `rerank/models/adapters/` đã được copy từ VIP5 repository và bao gồm:
- `adapter_controller.py`: Controller quản lý adapters
- `adapter_modeling.py`: Các loại adapter layers
- `adapter_configuration.py`: Configuration cho adapters
- `adapter_hypernetwork.py`: Hypernetwork adapters
- `adapter_utils.py`: Utilities
- `config.py`: Config classes
- `low_rank_layer.py`: Low-rank adapters
- `hypercomplex/`: Hypercomplex adapters

## 🎯 Tác dụng của Adapters trong VIP5

### 1. **Parameter-Efficient Fine-Tuning (PEFT)**

Adapters cho phép fine-tune VIP5 model với **rất ít parameters**:

- **Không cần fine-tune toàn bộ model**: Chỉ train một số lượng nhỏ parameters trong adapter layers
- **Giảm memory và computation**: Chỉ cần lưu và update adapter weights, không phải toàn bộ model
- **Nhanh hơn**: Training và inference nhanh hơn so với full fine-tuning

### 2. **Multi-Task Learning**

VIP5 sử dụng adapters để hỗ trợ **nhiều tasks khác nhau**:

- **Sequential recommendation**: Dự đoán item tiếp theo
- **Direct recommendation**: Đánh giá user-item preference
- **Explanation generation**: Tạo explanation cho recommendations

Mỗi task có thể có adapter riêng, cho phép model học task-specific features mà không làm ảnh hưởng đến base model.

### 3. **Task-Specific Adaptation**

Adapters cho phép model **adapt** cho từng task cụ thể:

```python
# Trong VIP5 forward pass
if self.ff_adapter is not None:
    forwarded_states = self.ff_adapter(forwarded_states, task)  # task = "sequential", "direct", etc.
```

Model tự động chọn adapter phù hợp với task hiện tại.

## 🔧 Cách Adapters hoạt động trong VIP5

### Architecture

Adapter là một **bottleneck layer** với cấu trúc:

```
Input (d_model) 
  ↓
Down-sampling (d_model → d_model/reduction_factor)
  ↓
Activation (GELU)
  ↓
Up-sampling (d_model/reduction_factor → d_model)
  ↓
Output (d_model)
```

**Ví dụ**: Với `d_model=512` và `reduction_factor=16`:
- Down-sampling: 512 → 32
- Up-sampling: 32 → 512
- **Chỉ train 512×32 + 32×512 = 32,768 parameters** thay vì toàn bộ layer

### Vị trí trong VIP5

Adapters được thêm vào **3 vị trí chính**:

1. **Feed-Forward Layer** (`T5LayerFF`):
   ```python
   # Trong T5LayerFF.forward()
   forwarded_states = self.DenseReluDense(forwarded_states)
   if self.ff_adapter is not None:
       forwarded_states = self.ff_adapter(forwarded_states, task)  # ← Adapter ở đây
   ```

2. **Self-Attention Layer** (`T5LayerSelfAttention`):
   ```python
   # Trong T5LayerSelfAttention.forward()
   y = attention_output[0]
   if self.attn_adapter is not None:
       y = self.attn_adapter(y, task)  # ← Adapter ở đây
   ```

3. **Cross-Attention Layer** (`T5LayerCrossAttention`):
   ```python
   # Trong T5LayerCrossAttention.forward()
   y = attention_output[0]
   if self.enc_attn_adapter is not None:
       y = self.enc_attn_adapter(y, task)  # ← Adapter ở đây
   ```

4. **LM Head** (`VIP5`):
   ```python
   # Trong VIP5.__init__()
   if config.use_lm_head_adapter:
       self.output_adapter = OutputParallelAdapterLayer(...)  # ← Adapter cho output
   ```

### AdapterController

`AdapterController` quản lý nhiều adapters cho nhiều tasks:

```python
class AdapterController(nn.Module):
    def __init__(self, config):
        self.adapters = nn.ModuleDict()
        # Tạo adapter cho mỗi task
        for task in tasks:
            self.adapters[task] = Adapter(config)
    
    def forward(self, inputs, task):
        # Chọn adapter phù hợp với task
        adapter = self.get_adapter(task)
        return adapter(inputs)
```

## 📊 So sánh: Có Adapter vs Không có Adapter

### Không có Adapter (`use_adapter=False`):
- Fine-tune toàn bộ model
- Cần nhiều memory và computation
- Chậm hơn
- Khó multi-task learning

### Có Adapter (`use_adapter=True`):
- Chỉ fine-tune adapter layers (~1-5% parameters)
- Tiết kiệm memory và computation
- Nhanh hơn
- Dễ dàng multi-task learning
- Có thể share base model cho nhiều tasks

## 🎨 Các loại Adapters trong VIP5

### 1. **Standard Adapter** (Adapter)
- Bottleneck architecture: down → activation → up
- Reduction factor: 16 (default)
- Parameters: ~1/16 của full layer

### 2. **Low-Rank Adapter** (LowRankAdapter)
- Sử dụng low-rank matrices
- Còn ít parameters hơn
- Phù hợp cho resource-constrained environments

### 3. **HyperComplex Adapter** (HyperComplexAdapter)
- Sử dụng hypercomplex multiplication
- Parameters: 1/n so với standard adapter (n = hypercomplex_division)
- Hiệu quả hơn về memory

### 4. **Output Adapter** (OutputAdapter)
- Dùng cho LM head
- Output dimension có thể khác input dimension

## 💡 Lợi ích cụ thể cho VIP5

1. **Efficient Training**:
   - Chỉ train adapters thay vì toàn bộ T5 model
   - Giảm training time và memory usage

2. **Task Specialization**:
   - Mỗi task (sequential, direct, explanation) có adapter riêng
   - Model có thể học task-specific patterns

3. **Transfer Learning**:
   - Base T5 model được giữ nguyên (pretrained weights)
   - Chỉ adapters được fine-tune cho recommendation tasks

4. **Scalability**:
   - Dễ dàng thêm tasks mới bằng cách thêm adapter mới
   - Không cần retrain toàn bộ model

## 🔍 Kiểm tra Adapters trong Code

### Trong `vip5_modeling.py`:

```python
# T5LayerFF - Feed-forward adapter
if config.use_adapter:
    self.ff_adapter = AdapterController(config.adapter_config)

# T5LayerSelfAttention - Self-attention adapter  
if config.use_adapter:
    self.attn_adapter = AdapterController(config.adapter_config)

# T5LayerCrossAttention - Cross-attention adapter
if config.use_adapter and config.add_adapter_cross_attn:
    self.enc_attn_adapter = AdapterController(config.adapter_config)

# VIP5 - Output adapter
if config.use_lm_head_adapter:
    self.output_adapter = OutputParallelAdapterLayer(...)
```

### Sử dụng trong Forward Pass:

```python
# Feed-forward
forwarded_states = self.DenseReluDense(forwarded_states)
if self.ff_adapter is not None:
    forwarded_states = self.ff_adapter(forwarded_states, task)  # task-aware

# Self-attention
y = attention_output[0]
if self.attn_adapter is not None:
    y = self.attn_adapter(y, task)  # task-aware
```

## ⚙️ Configuration

Để sử dụng adapters trong VIP5:

```python
from rerank.models.adapters.config import AdapterConfig

adapter_config = AdapterConfig(
    reduction_factor=16,  # Giảm 16 lần parameters
    non_linearity="gelu_new",
    tasks=["sequential", "direct", "explanation"],  # Các tasks
    use_single_adapter=False,  # Mỗi task có adapter riêng
)

# Trong VIP5Reranker
reranker = VIP5Reranker(
    use_adapter=True,  # Enable adapters
    adapter_config=adapter_config,
    add_adapter_cross_attn=True,  # Thêm adapter cho cross-attention
    use_lm_head_adapter=True,  # Adapter cho LM head
)
```

## 📈 Performance Impact

Với adapters:
- **Parameters to train**: ~1-5% của full model
- **Memory usage**: Giảm ~80-90%
- **Training speed**: Nhanh hơn 2-5x
- **Inference speed**: Gần như không đổi (chỉ thêm một vài operations)

## 🎯 Kết luận

Adapters trong VIP5 là một **parameter-efficient fine-tuning technique** cho phép:

1. ✅ Fine-tune model với ít parameters
2. ✅ Hỗ trợ multi-task learning
3. ✅ Task-specific adaptation
4. ✅ Tiết kiệm memory và computation
5. ✅ Dễ dàng scale và thêm tasks mới

**Adapters đã có sẵn trong project** và sẽ tự động được sử dụng nếu `use_adapter=True` trong config!

