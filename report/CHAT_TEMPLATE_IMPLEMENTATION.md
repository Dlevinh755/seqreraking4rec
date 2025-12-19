# Implementation: Chat Template Format cho Inference

## ✅ Đã sửa

### **File**: `rerank/models/llm.py:279-314`

**Thay đổi**:
- `predict_probs()` giờ convert plain text prompt → chat template format trước khi tokenize
- Sử dụng `apply_chat_template()` với `add_generation_prompt=True`
- Đảm bảo consistency với training format

**Code trước**:
```python
def predict_probs(self, prompt, num_candidates=None):
    inputs = self.tokenizer(
        prompt,  # Plain text
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(self.model.device)
```

**Code sau**:
```python
def predict_probs(self, prompt, num_candidates=None):
    # ✅ Convert plain text prompt to chat template format
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template with generation prompt
    text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # ✅ Add <|im_start|>assistant\n
    )
    
    # Tokenize the chat template formatted text
    inputs = self.tokenizer(
        text,  # Chat template format
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(self.model.device)
```

## 📊 Format Comparison

### **Before (Plain Text)**:
```
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
- item1

Candidate items:
A. candidate1
B. candidate2

Answer with only one letter (A-B).
```

### **After (Chat Template Format)**:
```
<|im_start|>user
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
- item1

Candidate items:
A. candidate1
B. candidate2

Answer with only one letter (A-B).<|im_end|>
<|im_start|>assistant
```

## ✅ Lợi ích

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

## 🔍 Chi tiết kỹ thuật

### **Tokenization**:
- Original prompt: 220 characters
- Chat template format: 368 characters (+148 chars for special tokens)
- Last tokens: `[' letter', ' (', 'A', '-B', ').', '<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n']`
- Model sẽ predict next token sau `<|im_start|>assistant\n` (giống training)

### **Infer num_candidates**:
- Logic vẫn sử dụng original prompt (trước khi apply chat template)
- Đúng vì original prompt vẫn chứa thông tin về candidates
- Không bị ảnh hưởng bởi special tokens

## 📝 Lưu ý

1. **Interface không thay đổi**: 
   - `predict_probs()` vẫn nhận plain text prompt
   - Conversion tự động bên trong method
   - Không cần sửa code gọi `predict_probs()`

2. **Token length**:
   - Chat template format dài hơn ~148 characters
   - Có thể cần tăng `max_length` nếu prompt dài
   - Hiện tại `qwen_max_seq_length=2048` đủ cho hầu hết cases

3. **Performance**:
   - Cần test để đảm bảo performance không giảm
   - Expected: performance tốt hơn do consistency với training

## ✅ Kết luận

Code đã được sửa để sử dụng chat template format cho inference, đảm bảo consistency với training format và cải thiện performance.

