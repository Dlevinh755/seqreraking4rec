# Phân tích EOS Token trong Prompt

## ✅ Kết quả kiểm tra

### 1. **Training (LLM.train())**

**File**: `rerank/models/llm.py:186-200`

**Cách hoạt động**:
```python
text = self.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
```

**Kết quả**:
- `apply_chat_template` tự động format messages với chat template của Qwen
- Output format: `'\n<user_content><|im_end|>\n<|im_start|>assistant\n<response><|im_end|>\n'`
- **EOS token (`<|im_end|>`) được tự động thêm vào sau response**

**Ví dụ**:
- Input: `messages = [{'role': 'user', 'content': 'test'}, {'role': 'assistant', 'content': 'A'}]`
- Output: `'\ntest<|im_end|>\n<|im_start|>assistant\nA<|im_end|>\n'`
- ✅ **EOS token có trong training data**

### 2. **Training với `train_on_responses_only`**

**File**: `rerank/models/llm.py:244-248`

**Cách hoạt động**:
```python
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
```

**Kết quả**:
- `train_on_responses_only` mask prompt tokens, chỉ tính loss trên response tokens
- Response part: `"<|im_start|>assistant\n"` - đây là prefix cho response
- **EOS token (`<|im_end|>`) đã được thêm vào bởi `apply_chat_template` trước đó**
- ✅ **Loss được tính trên response tokens (bao gồm cả EOS token)**

### 3. **Inference (`predict_probs()`)**

**File**: `rerank/models/llm.py:296-300`

**Cách hoạt động**:
```python
inputs = self.tokenizer(
    prompt, 
    return_tensors="pt",
    truncation=True,
    max_length=max_length,
)
```

**Kết quả**:
- `tokenizer()` với default `add_special_tokens=True` (không explicit)
- Prompt là plain text (không phải chat template format)
- **EOS token KHÔNG được thêm vào prompt** (vì prompt không phải chat format)
- Model sẽ predict next token sau prompt (không có EOS)

### 4. **So sánh với LlamaRec**

**LlamaRec yêu cầu**:
- Training: Loss chỉ tính trên response tokens (bao gồm EOS)
- Inference: Model predict next token (letter) sau prompt

**Code hiện tại**:
- ✅ Training: EOS token có trong response, loss chỉ tính trên response (bao gồm EOS)
- ✅ Inference: Model predict next token (letter) sau prompt (không có EOS trong prompt)

## 📝 Kết luận

### ✅ **EOS token ĐƯỢC SỬ DỤNG trong Training**
- `apply_chat_template` tự động thêm `<|im_end|>` sau response
- `train_on_responses_only` tính loss trên response tokens (bao gồm EOS)
- Đúng với LlamaRec: loss chỉ tính trên response (token label + EOS)

### ⚠️ **EOS token KHÔNG có trong Inference Prompt**
- Prompt là plain text, không phải chat template format
- `tokenizer()` không thêm EOS vào prompt (vì không phải chat format)
- Model predict next token (letter) sau prompt
- **Điều này là ĐÚNG** - model cần predict letter, không cần EOS trong prompt

## 🔍 Kiểm tra thêm

Cần kiểm tra xem:
1. Model có tự động thêm EOS khi generate không?
2. `predict_probs()` có cần thêm EOS vào prompt không?

**Kết luận**: Code hiện tại đã đúng - EOS token được sử dụng trong training (tự động bởi `apply_chat_template`), nhưng không cần trong inference prompt (vì model chỉ cần predict letter).

