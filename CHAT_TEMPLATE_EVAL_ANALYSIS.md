# Phân tích: Có nên sử dụng Chat Template Format cho Prompt khi Eval?

## 🔍 Vấn đề hiện tại

### 1. **Training Format**
**File**: `rerank/models/llm.py:186-200`

```python
text = self.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
```

**Format**:
```
\n<user_content><|im_end|>\n<|im_start|>assistant\n<response><|im_end|>\n
```

**Đặc điểm**:
- ✅ Có special tokens: `<|im_start|>user\n`, `<|im_start|>assistant\n`, `<|im_end|>`
- ✅ Model được train với chat template format
- ✅ Model học cách predict response sau `<|im_start|>assistant\n`

### 2. **Inference Format (Hiện tại)**
**File**: `rerank/models/llm.py:296-301`

```python
prompt = build_prompt_from_candidates(...)  # Plain text
inputs = self.tokenizer(
    prompt, 
    return_tensors="pt",
    truncation=True,
    max_length=max_length,
)
```

**Format**:
```
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
- item1
- item2

Candidate items:
A. candidate1
B. candidate2

Answer with only one letter (A-B).
```

**Đặc điểm**:
- ❌ Không có special tokens
- ❌ Không có chat template format
- ❌ Format khác với training

## ⚠️ Vấn đề

### **Format Mismatch giữa Training và Inference**

1. **Training**: Model được train với chat template format
   - Input: `'\n<user_content><|im_end|>\n<|im_start|>assistant\n'`
   - Model học predict next token sau `<|im_start|>assistant\n`

2. **Inference**: Model nhận plain text prompt
   - Input: `'You are a recommendation ranking assistant. ... Answer with only one letter (A-B).'`
   - Model không thấy `<|im_start|>assistant\n` → có thể bị confusion

3. **Hậu quả**:
   - Model có thể không hiểu context đúng
   - Performance có thể giảm do format mismatch
   - Model không biết đang ở đâu trong conversation flow

## ✅ Giải pháp: Sử dụng Chat Template Format cho Inference

### **Lý do nên sử dụng**:

1. **Consistency với Training**
   - Training và inference dùng cùng format
   - Model quen với format này
   - Giảm distribution shift

2. **Đúng với Model Design**
   - Qwen models được train với chat template format
   - Special tokens (`<|im_start|>`, `<|im_end|>`) giúp model hiểu context
   - Model biết đang ở đâu trong conversation

3. **Better Performance**
   - Model predict next token trong context đúng
   - Special tokens giúp model focus vào response part
   - Giảm confusion về format

4. **Đúng với LlamaRec**
   - LlamaRec cũng sử dụng chat template format
   - Consistency với best practices

### **Cách implement**:

```python
def predict_probs(self, prompt, num_candidates=None):
    # Convert plain text prompt to chat template format
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template with generation prompt
    text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # ✅ Add <|im_start|>assistant\n
    )
    
    # Tokenize
    inputs = self.tokenizer(
        text, 
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(self.model.device)
    
    # Predict
    with torch.no_grad():
        outputs = self.model(**inputs)
    
    logits = outputs.logits[:, -1]  # [vocab_size]
    # ... rest of the code
```

### **Format sau khi apply chat template**:

```
<|im_start|>user
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
- item1
- item2

Candidate items:
A. candidate1
B. candidate2

Answer with only one letter (A-B).<|im_end|>
<|im_start|>assistant
```

**Đặc điểm**:
- ✅ Có `<|im_start|>user\n` ở đầu
- ✅ Có `<|im_end|>` sau user content
- ✅ Có `<|im_start|>assistant\n` ở cuối (generation prompt)
- ✅ Model predict next token sau `<|im_start|>assistant\n` (giống training)

## 📊 So sánh

| Aspect | Plain Text (Hiện tại) | Chat Template (Đề xuất) |
|--------|----------------------|-------------------------|
| **Consistency với Training** | ❌ Khác format | ✅ Cùng format |
| **Special Tokens** | ❌ Không có | ✅ Có (`<|im_start|>`, `<|im_end|>`) |
| **Model Context** | ⚠️ Không rõ ràng | ✅ Rõ ràng (user → assistant) |
| **Performance** | ⚠️ Có thể giảm | ✅ Tốt hơn |
| **LlamaRec Alignment** | ❌ Không | ✅ Có |

## 🎯 Kết luận

### ✅ **NÊN sử dụng Chat Template Format cho Inference**

**Lý do**:
1. **Consistency**: Training và inference dùng cùng format
2. **Performance**: Model hoạt động tốt hơn với format quen thuộc
3. **Best Practice**: Đúng với cách Qwen models được design
4. **LlamaRec Alignment**: Phù hợp với LlamaRec approach

**Cách implement**:
- Sửa `predict_probs()` để convert plain text prompt → chat template format
- Sử dụng `apply_chat_template()` với `add_generation_prompt=True`
- Đảm bảo format giống với training

**Lưu ý**:
- Cần test để đảm bảo performance không giảm
- Có thể cần điều chỉnh `max_length` nếu prompt dài hơn (do thêm special tokens)

