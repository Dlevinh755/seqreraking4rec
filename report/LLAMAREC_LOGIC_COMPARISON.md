# So sánh Logic Training và Rerank với LlamaRec

## 📊 Tổng quan

So sánh chi tiết logic training và rerank của project hiện tại với LlamaRec để tìm vấn đề.

---

## 🔍 1. Training Data Preparation

### **LlamaRec Approach**

```python
# LlamaRec training sample format
{
    "messages": [
        {
            "role": "user",
            "content": """
Candidate items:
A. candidate1
B. candidate2
C. candidate3
D. candidate4

Answer with only one letter (A-D).
"""
        },
        {
            "role": "assistant",
            "content": "D"  # ✅ Letter index
        }
    ]
}
```

**Key Points**:
- ✅ Dùng **letter labels** (A, B, C, D, ...)
- ✅ Target = letter của ground-truth item
- ✅ Prompt format với letter labels

---

### **Project hiện tại - Kiểm tra Code**

#### **1. Text-only mode** (`rerank/methods/qwen_reranker_unified.py`)

**Code hiện tại** (line 300-389):
```python
# For caption/semantic_summary modes
if self.mode in ["caption", "semantic_summary"]:
    train_samples = self._prepare_training_samples(train_data)
    # ...
    target = LETTERS[target_idx]  # ✅ Letter index (A, B, C, ...)
    train_data_for_llm.append({
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target}  # ✅ Letter
        ]
    })
```

**Status**: ✅ **ĐÚNG** - Dùng letter labels

**Nhưng**: Cần kiểm tra `text_only` mode có dùng `train_data_for_llm` từ kwargs không?

---

#### **2. Training Sample Preparation** (`_prepare_training_samples`)

**Code hiện tại** (`rerank/methods/qwen_reranker_unified.py:652-738`):
```python
def _prepare_training_samples(
    self,
    train_data: Dict[int, List[int]],
) -> List[Dict]:
    """Prepare training samples for multimodal modes."""
    samples = []
    
    for user_id, items in train_data.items():
        if len(items) < 2:
            continue
        
        # Randomly select split point
        split_point = random.randint(1, len(items) - 1)
        history = items[:split_point]
        target_item = items[split_point]
        
        # Get candidates: target + negatives
        all_items = set()
        for user_items in train_data.values():
            all_items.update(user_items)
        
        # Exclude history and target
        candidate_pool = [item for item in all_items 
                         if item not in history and item != target_item]
        
        # Sample negatives
        num_negatives = min(self.top_k - 1, len(candidate_pool))
        negatives = random.sample(candidate_pool, num_negatives)
        
        # Candidates = [target] + negatives (shuffled)
        candidates = [target_item] + negatives
        random.shuffle(candidates)
        
        # Find target index
        target_idx = candidates.index(target_item)
        
        samples.append({
            "user_id": user_id,
            "history": history,
            "candidates": candidates,
            "target_item": target_item,
            "target_idx": target_idx,  # ✅ 0-indexed
        })
    
    return samples
```

**Phân tích**:
- ✅ **ĐÚNG**: Random split point
- ✅ **ĐÚNG**: History = items[:split_point]
- ✅ **ĐÚNG**: Target = items[split_point]
- ✅ **ĐÚNG**: Candidates = [target] + negatives (shuffled)
- ✅ **ĐÚNG**: Target index = 0-indexed

**Status**: ✅ **Logic đúng với LlamaRec**

---

## 🔍 2. Training Process

### **LlamaRec Approach**

```python
# 1. Load model với LoRA
model = get_peft_model(model, LoraConfig(...))

# 2. Prepare dataset với letter labels
train_dataset = prepare_dataset(train_data)

# 3. Train với next-token prediction
trainer = Trainer(...)
trainer.train()

# 4. Mask prompt tokens (chỉ tính loss ở response)
# LlamaRec sử dụng mask để chỉ tính loss ở phần assistant response
```

---

### **Project hiện tại**

**Code** (`rerank/models/llm.py:164-263`):
```python
def train(self, batch_size=None):
    # 1. Load dataset
    hf_train_dataset = Dataset.from_list(self.train_data)
    
    # 2. Format messages to text using chat template
    hf_train_dataset = hf_train_dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    
    # 3. Setup training args
    training_args = SFTConfig(
        learning_rate=learning_rate,  # ✅ From config
        num_train_epochs=num_epochs,   # ✅ From config
        ...
    )
    
    # 4. Create trainer
    trainer = SFTTrainer(...)
    
    # 5. ✅ Mask prompt tokens (chỉ tính loss ở response)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    
    # 6. Train
    trainer.train()
```

**Phân tích**:
- ✅ **ĐÚNG**: Next-token prediction
- ✅ **ĐÚNG**: Mask prompt tokens với `train_on_responses_only`
- ✅ **ĐÚNG**: Chat template format
- ✅ **ĐÚNG**: LoRA fine-tuning

**Status**: ✅ **Logic đúng với LlamaRec**

---

## 🔍 3. Rerank/Inference Process

### **LlamaRec Approach**

```python
# 1. Build prompt với letter labels
prompt = build_prompt(user_history, candidates)  # A, B, C, ...

# 2. Forward pass
outputs = model(**inputs)
logits = outputs.logits[:, -1]  # [vocab_size]

# 3. Extract letter token IDs
letter_tokens = []
for letter in ["A", "B", "C", ...]:
    token_id = tokenizer.convert_tokens_to_ids(letter)
    letter_tokens.append((idx, letter, token_id))

# 4. Extract probabilities
token_ids = [tid for _, _, tid in letter_tokens]
probs = F.softmax(logits[:, token_ids], dim=-1)

# 5. Map to candidates
prob_array = np.zeros(num_candidates)
for idx, (cand_idx, letter, token_id) in enumerate(letter_tokens):
    prob_array[cand_idx] = probs[0, idx].item()

# 6. Rank by probability
ranked = sorted(zip(candidates, prob_array), key=lambda x: x[1], reverse=True)
```

---

### **Project hiện tại**

**Code** (`rerank/models/llm.py:290-414`):
```python
def predict_probs(self, prompt, num_candidates=None):
    # 1. Convert to chat template format
    messages = [{"role": "user", "content": prompt}]
    text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 2. Tokenize
    inputs = self.tokenizer(text, return_tensors="pt", ...).to(self.model.device)
    
    # 3. Forward pass
    with torch.no_grad():
        outputs = self.model(**inputs)
    
    # 4. Extract logits
    logits = outputs.logits[:, -1]  # [vocab_size] ✅ ĐÚNG
    
    # 5. Extract letter token IDs
    letter_tokens = []
    for i in range(num_candidates):
        letter = LETTERS[i]  # ✅ Letter (A, B, C, ...)
        
        # Strategy 1: Direct letter token
        token_id = self.tokenizer.convert_tokens_to_ids(letter)
        if token_id != self.tokenizer.unk_token_id:
            letter_tokens.append((i, letter, token_id))
            continue
        
        # Strategy 2: With space prefix
        token_id = self.tokenizer.convert_tokens_to_ids(" " + letter)
        if token_id != self.tokenizer.unk_token_id:
            letter_tokens.append((i, letter, token_id))
            continue
        
        # Strategy 3: Encoding
        encoded = self.tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) > 0:
            letter_tokens.append((i, letter, encoded[0]))
    
    # 6. Extract probabilities
    token_ids = [tid for _, _, tid in letter_tokens]
    probs = F.softmax(logits[:, token_ids], dim=-1)  # ✅ ĐÚNG
    
    # 7. Map to candidates
    prob_array = np.zeros(num_candidates)
    for idx, (cand_idx, letter, token_id) in enumerate(letter_tokens):
        if cand_idx < num_candidates:
            prob_array[cand_idx] = probs[0, idx].item()
    
    # 8. Normalize
    if prob_array.sum() > 0:
        prob_array = prob_array / prob_array.sum()
    
    return prob_array
```

**Phân tích**:
- ✅ **ĐÚNG**: Extract logits của token cuối cùng
- ✅ **ĐÚNG**: Extract letter token IDs (với fallback strategies)
- ✅ **ĐÚNG**: Softmax trên letter tokens
- ✅ **ĐÚNG**: Map về candidate indices
- ✅ **ĐÚNG**: Normalize probabilities

**Status**: ✅ **Logic đúng với LlamaRec**

---

## 🔍 4. Prompt Format

### **LlamaRec**

```
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
- item1
- item2

Candidate items:
A. candidate1
B. candidate2
C. candidate3

Answer with only one letter (A-C).
```

---

### **Project hiện tại**

**Code** (`rerank/models/llm.py:19-69`):
```python
def build_prompt_from_candidates(user_history, candidate_ids, item_id2text, max_candidates=None):
    # ...
    # Use letters (A-Z, a-z) for up to 52 candidates (LlamaRec style)
    cand_text = "\n".join(
        [f"{LETTERS[i]}. {c}" for i, c in enumerate(candidates)]  # ✅ Letter labels
    )
    
    # Answer format with letters
    if num_candidates <= 26:
        answer_format = f"Answer with only one letter (A-{LETTERS[num_candidates-1]})."  # ✅
    else:
        answer_format = f"Answer with only one letter (A-Z, a-{LETTERS[num_candidates-1]})."  # ✅
    
    prompt = f"""
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
{history_text}

Candidate items:
{cand_text}

{answer_format}
"""
    return prompt
```

**Phân tích**:
- ✅ **ĐÚNG**: Dùng letter labels (A, B, C, ...)
- ✅ **ĐÚNG**: Answer format với letters
- ✅ **ĐÚNG**: Format giống LlamaRec

**Status**: ✅ **Logic đúng với LlamaRec**

---

## ⚠️ Vấn đề Logic Phát hiện

### **1. Text-only mode training data** ⚠️

**Vấn đề**: 
- Code hiện tại có 2 paths cho training:
  1. `train_data_for_llm` từ kwargs (pre-prepared)
  2. Tự prepare từ `_prepare_training_samples` (cho caption/semantic_summary)

**Kiểm tra**: 
- Nếu `text_only` mode dùng `train_data_for_llm` từ kwargs, cần đảm bảo format đúng
- Nếu không có `train_data_for_llm`, code sẽ không train (chỉ load model)

**Code** (`rerank/methods/qwen_reranker_unified.py:296-403`):
```python
train_data_for_llm = kwargs.get("train_data_for_llm")

if train_data_for_llm is not None:
    # Train với pre-prepared data
    self.llm_model = LLMModel(train_data=train_data_for_llm, ...)
    self.llm_model.train(...)
else:
    # Chỉ load model, không train
    self.llm_model = LLMModel(train_data=None, ...)
    self.llm_model.load_model(...)
```

**Vấn đề tiềm ẩn**:
- Nếu `train_data_for_llm` không được cung cấp cho `text_only` mode → model không được train
- Cần kiểm tra xem `train_data_for_llm` có được prepare đúng format không

---

### **2. Chat template format consistency** ⚠️

**Vấn đề**:
- Training: Dùng chat template format (`apply_chat_template`)
- Inference: Cũng dùng chat template format (`apply_chat_template`)

**Phân tích**:
- ✅ **ĐÚNG**: Consistency giữa training và inference
- ✅ **ĐÚNG**: Dùng `add_generation_prompt=True` cho inference

**Status**: ✅ **Logic đúng**

---

### **3. Letter token extraction fallback** ✅

**Code** (`rerank/models/llm.py:363-384`):
```python
# Strategy 1: Direct letter token
token_id = self.tokenizer.convert_tokens_to_ids(letter)

# Strategy 2: With space prefix
token_id = self.tokenizer.convert_tokens_to_ids(" " + letter)

# Strategy 3: Encoding
encoded = self.tokenizer.encode(letter, add_special_tokens=False)
```

**Phân tích**:
- ✅ **ĐÚNG**: Có fallback strategies để handle different tokenizers
- ✅ **ĐÚNG**: LlamaRec có thể không có fallback này, nhưng đây là improvement

**Status**: ✅ **Logic đúng (thậm chí tốt hơn LlamaRec)**

---

## 📊 Tóm tắt So sánh

| Aspect | LlamaRec | Project hiện tại | Status |
|--------|----------|------------------|--------|
| **Training Data Prep** | Letter labels | ✅ Letter labels | ✅ ĐÚNG |
| **Training Objective** | Next-token prediction | ✅ Next-token prediction | ✅ ĐÚNG |
| **Loss Masking** | Mask prompt tokens | ✅ `train_on_responses_only` | ✅ ĐÚNG |
| **Prompt Format** | Letter labels | ✅ Letter labels | ✅ ĐÚNG |
| **Logits Extraction** | `logits[:, -1]` | ✅ `logits[:, -1]` | ✅ ĐÚNG |
| **Token Extraction** | Letter tokens | ✅ Letter tokens (với fallback) | ✅ ĐÚNG |
| **Probability Mapping** | Map letters → candidates | ✅ Map letters → candidates | ✅ ĐÚNG |
| **Reranking** | Sort by probability | ✅ Sort by probability | ✅ ĐÚNG |

---

## ⚠️ Vấn đề Logic Tiềm ẩn

### **1. Text-only mode training data source** 🔴

**Vấn đề**:
- `text_only` mode phụ thuộc vào `train_data_for_llm` từ kwargs
- Nếu không có → model không được train

**Giải pháp**:
```python
# Thêm auto-prepare cho text_only mode nếu không có train_data_for_llm
if self.mode == "text_only" and train_data_for_llm is None:
    train_samples = self._prepare_training_samples(train_data)
    # Convert to LLM training format với letter labels
    train_data_for_llm = []
    for sample in train_samples:
        # ... build prompt và target với letter labels ...
```

---

### **2. Epochs quá ít** 🔴

**Vấn đề**:
- `rerank_epochs = 1` (default) → quá ít để model học được pattern

**Giải pháp**:
```python
# config.py
parser.add_argument('--rerank_epochs', type=int, default=5,  # ✅ Tăng lên 5
```

---

### **3. LoRA config có thể chưa tối ưu** 🟡

**Vấn đề**:
- `r=8, alpha=16` có thể nhỏ cho better performance

**Giải pháp**:
```python
# Thử tăng lên r=16, alpha=32
```

---

## ✅ Kết luận

### **Logic Training và Rerank**: ✅ **ĐÚNG với LlamaRec**

**Tất cả các aspects chính đều đúng**:
- ✅ Training data preparation với letter labels
- ✅ Training process với next-token prediction và loss masking
- ✅ Rerank process với verbalizer approach
- ✅ Prompt format với letter labels

### **Vấn đề chính**: 

1. **Epochs quá ít** (1 epoch) → Model chưa học được pattern
2. **Text-only mode** có thể không train nếu không có `train_data_for_llm`

### **Recommendation**:

1. ✅ **Tăng epochs lên 5-10**
2. ✅ **Kiểm tra text_only mode có train không**
3. ✅ **Optional: Tăng LoRA rank nếu cần**

**Logic code đã đúng, vấn đề là hyperparameters (epochs quá ít)!**

