# Optimization Guide - Tăng tốc Semantic Summary và LLM Inference

## 📊 Tổng quan

Hướng dẫn này mô tả các cách để tăng tốc độ:
1. **Semantic Summary Generation** (Qwen3-VL)
2. **LLM Inference** (Qwen reranker)

## 🚀 Optimizations cho Semantic Summary Generation

### 1. Tăng Batch Size

**Hiện tại**: `BATCH_SIZE = 4`, nhưng chỉ group images, không batch inference thực sự.

**Optimization**: 
- Tăng `BATCH_SIZE` nếu GPU memory cho phép (8, 16, 32)
- Implement batch inference thực sự nếu model support

**Cách sử dụng**:
```python
# Trong config.py hoặc command line
--semantic_summary_batch_size 8  # Tăng từ 4 lên 8
```

### 2. Model Quantization

**8-bit Quantization**:
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
)
```

**4-bit Quantization** (tiết kiệm memory hơn):
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

### 3. torch.compile() (PyTorch 2.0+)

**Compile model để tăng tốc**:
```python
model = torch.compile(model, mode="reduce-overhead")
```

**Lưu ý**: Cần PyTorch 2.0+ và có thể mất thời gian compile lần đầu.

### 4. Flash Attention 2

**Sử dụng Flash Attention 2** (nếu model support):
```python
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
)
```

**Cài đặt**:
```bash
pip install flash-attn --no-build-isolation
```

### 5. Parallel Processing

**Process multiple images song song** (nếu không thể batch):
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(process_image, img) for img in images]
    results = [f.result() for f in futures]
```

### 6. Caching Processed Images

**Cache resized images** để tránh resize lại:
```python
from functools import lru_cache
from PIL import Image

@lru_cache(maxsize=1000)
def load_and_resize_image(image_path, max_size=448):
    img = Image.open(image_path).convert("RGB")
    # Resize logic...
    return img
```

## 🚀 Optimizations cho LLM Inference

### 1. Batch Inference

**Hiện tại**: Process từng prompt một.

**Optimization**: Batch multiple prompts:
```python
def predict_probs_batch(self, prompts, num_candidates_list):
    """Batch inference cho multiple prompts."""
    # Tokenize all prompts
    inputs = self.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(self.model.device)
    
    with torch.no_grad():
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1]  # [batch_size, vocab_size]
    
    # Process each prompt's logits
    results = []
    for i, num_candidates in enumerate(num_candidates_list):
        # Extract probabilities for this prompt
        # ... (similar to predict_probs)
        results.append(prob_array)
    
    return results
```

### 2. Model Quantization

**8-bit hoặc 4-bit quantization**:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-0.6B",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,  # 4-bit quantization
)
```

### 3. torch.compile()

**Compile model**:
```python
self.model = torch.compile(self.model, mode="reduce-overhead")
```

### 4. vLLM (Very Fast LLM Inference)

**Sử dụng vLLM** cho faster inference:
```bash
pip install vllm
```

**Sử dụng**:
```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B", quantization="awq")
sampling_params = SamplingParams(temperature=0, max_tokens=1)

prompts = ["prompt1", "prompt2", ...]
outputs = llm.generate(prompts, sampling_params)
```

**Lưu ý**: vLLM yêu cầu GPU và có thể không support tất cả models.

### 5. Text Generation Inference (TGI)

**Sử dụng TGI** từ Hugging Face:
```bash
# Docker
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id Qwen/Qwen3-0.6B
```

### 6. Caching Prompts

**Cache tokenized prompts** nếu prompts lặp lại:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def tokenize_prompt(prompt):
    return self.tokenizer(prompt, return_tensors="pt")
```

## 📈 Expected Speedup

| Optimization | Semantic Summary | LLM Inference | Memory Impact |
|--------------|------------------|---------------|--------------|
| Increase Batch Size | 2-4x | 3-5x | +50-100% |
| 8-bit Quantization | 1.5-2x | 1.5-2x | -50% |
| 4-bit Quantization | 2-3x | 2-3x | -75% |
| torch.compile() | 1.2-1.5x | 1.2-1.5x | 0% |
| Flash Attention 2 | 1.5-2x | N/A | -20% |
| vLLM | N/A | 5-10x | +20% |
| Batch Inference | N/A | 3-5x | +30% |

## ⚙️ Implementation Priority

### Quick Wins (Dễ implement, hiệu quả cao):
1. ✅ Tăng batch size cho semantic summary
2. ✅ 4-bit quantization cho cả hai
3. ✅ torch.compile() cho cả hai
4. ✅ Batch inference cho LLM

### Medium Effort (Cần thay đổi code nhiều hơn):
1. Flash Attention 2 cho Qwen3-VL
2. Parallel processing cho semantic summary
3. Caching mechanisms

### Advanced (Cần setup phức tạp):
1. vLLM cho LLM inference
2. TGI server
3. Multi-GPU inference

## 🔧 Configuration

Thêm vào `config.py`:
```python
parser.add_argument('--semantic_summary_batch_size', type=int, default=4,
                   help='Batch size for semantic summary generation')
parser.add_argument('--llm_batch_size', type=int, default=1,
                   help='Batch size for LLM inference')
parser.add_argument('--use_quantization', action='store_true',
                   help='Use 4-bit quantization for models')
parser.add_argument('--use_torch_compile', action='store_true',
                   help='Use torch.compile() for faster inference')
parser.add_argument('--use_flash_attention', action='store_true',
                   help='Use Flash Attention 2 (if supported)')
```

## 📝 Notes

- **Memory vs Speed**: Quantization giảm memory nhưng có thể giảm accuracy nhẹ
- **GPU Required**: Hầu hết optimizations cần GPU
- **Compatibility**: Một số optimizations không work với tất cả models
- **Testing**: Luôn test accuracy sau khi apply optimizations

