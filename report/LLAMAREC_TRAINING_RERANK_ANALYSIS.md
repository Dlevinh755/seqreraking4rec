# Phân tích Cách LlamaRec Train, Lấy Logits và Rerank

## 📚 Tổng quan LlamaRec

**Repository**: [LlamaRec](https://github.com/Yueeeeeeee/LlamaRec)

**Paper**: "LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking" (PGAI@CIKM 2023)

**Approach**: 
- **Stage 1**: Sequential recommender (LRURec) để retrieve candidates
- **Stage 2**: LLM (Llama 2) để rank candidates
- **Method**: Verbalizer-based approach - transform output logits thành probability distributions

---

## 🔍 1. Training Process

### **LlamaRec Training Approach**

Theo paper và README:

#### **Training Objective**:
- **Next-token prediction** (giống LLM chuẩn)
- **Label = letter index** của ground-truth item (A, B, C, D, ...)
- **Loss function**: Cross-entropy loss trên token label
- **Chỉ tính loss ở phần Response** (token index + EOS)

#### **Training Data Format**:

```python
# LlamaRec training sample format
{
    "messages": [
        {
            "role": "user",
            "content": """
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
- item1
- item2
- item3

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
            "content": "D"  # ✅ Letter index (not number)
        }
    ]
}
```

#### **Training Process**:

```python
# Pseudocode từ LlamaRec
# 1. Load base model (Llama 2)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Setup LoRA (parameter-efficient fine-tuning)
model = get_peft_model(model, LoraConfig(...))

# 3. Prepare training data với letter labels
train_dataset = prepare_dataset(train_data)  # Format với letter labels (A, B, C, ...)

# 4. Train với next-token prediction
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
)

# 5. Mask prompt tokens (chỉ tính loss ở response)
# LlamaRec sử dụng mask để chỉ tính loss ở phần assistant response
trainer.train()
```

**Key Points**:
- ✅ Dùng **letter labels** (A, B, C, ...) thay vì numbers (1, 2, 3, ...)
- ✅ **Mask prompt tokens** - chỉ tính loss ở response
- ✅ **Next-token prediction** - model predict letter token

---

## 🔍 2. Logits Extraction (Verbalizer Approach)

### **LlamaRec Verbalizer Approach**

Theo paper, LlamaRec sử dụng **verbalizer-based approach**:

> "Instead of generating next-item titles, we adopt a verbalizer-based approach that transforms output logits into probability distributions over the candidate items."

#### **Process**:

```python
# LlamaRec logits extraction (pseudocode)
def extract_logits_and_rerank(model, tokenizer, prompt, candidates):
    """
    Extract logits using verbalizer approach.
    
    Args:
        model: Fine-tuned Llama 2 model
        tokenizer: Tokenizer
        prompt: Input prompt with candidates labeled A, B, C, ...
        candidates: List of candidate items
    
    Returns:
        probabilities: [num_candidates] - probability distribution
    """
    # 1. Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 2. Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 3. Extract logits của token cuối cùng (next token prediction)
    logits = outputs.logits[:, -1]  # [vocab_size]
    
    # 4. Extract token IDs cho letters (A, B, C, ...)
    letter_tokens = []
    for i, letter in enumerate(["A", "B", "C", "D", ...]):  # Up to num_candidates
        token_id = tokenizer.convert_tokens_to_ids(letter)
        if token_id != tokenizer.unk_token_id:
            letter_tokens.append((i, letter, token_id))
    
    # 5. Extract probabilities cho letter tokens
    token_ids = [tid for _, _, tid in letter_tokens]
    probs = F.softmax(logits[:, token_ids], dim=-1)  # [1, num_letters]
    
    # 6. Map về candidate indices
    prob_array = np.zeros(len(candidates))
    for idx, (cand_idx, letter, token_id) in enumerate(letter_tokens):
        if cand_idx < len(candidates):
            prob_array[cand_idx] = probs[0, idx].item()
    
    # 7. Normalize
    if prob_array.sum() > 0:
        prob_array = prob_array / prob_array.sum()
    else:
        prob_array = np.ones(len(candidates)) / len(candidates)
    
    return prob_array
```

**Key Points**:
- ✅ **Next-token prediction**: `logits = outputs.logits[:, -1]`
- ✅ **Letter tokens**: Extract token IDs cho letters (A, B, C, ...)
- ✅ **Softmax**: Convert logits → probabilities
- ✅ **Map to candidates**: Map letter probabilities → candidate indices

---

## 🔍 3. Reranking Process

### **LlamaRec Reranking**

#### **Two-Stage Process**:

```python
# Stage 1: Retrieval (LRURec)
def retrieve_candidates(user_history, retriever_model, top_k=50):
    """
    Retrieve top-K candidates using sequential recommender.
    """
    candidates = retriever_model.retrieve(user_history, top_k=top_k)
    return candidates

# Stage 2: Ranking (Llama 2)
def rerank_candidates(user_history, candidates, ranker_model, tokenizer):
    """
    Rerank candidates using LLM.
    """
    # 1. Build prompt với candidates labeled A, B, C, ...
    prompt = build_prompt(user_history, candidates)
    
    # 2. Extract probabilities using verbalizer approach
    probs = extract_logits_and_rerank(ranker_model, tokenizer, prompt, candidates)
    
    # 3. Sort candidates by probability
    ranked_indices = np.argsort(probs)[::-1]  # Descending order
    ranked_candidates = [candidates[i] for i in ranked_indices]
    ranked_scores = [probs[i] for i in ranked_indices]
    
    return list(zip(ranked_candidates, ranked_scores))
```

#### **Prompt Format**:

```
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
- item1
- item2
- item3

Candidate items:
A. candidate1
B. candidate2
C. candidate3
D. candidate4
...

Answer with only one letter (A-{last_letter}).
```

**Key Points**:
- ✅ **Two-stage**: Retrieval → Ranking
- ✅ **Letter labels**: Candidates labeled A, B, C, ...
- ✅ **Single token output**: Model chỉ predict một letter
- ✅ **Probability distribution**: Transform logits → probabilities → ranking

---

## 📊 So sánh với Project hiện tại

### **1. Training**

| Aspect | LlamaRec | Project hiện tại |
|--------|----------|------------------|
| **Base Model** | Llama 2-7B | Qwen3-0.6B, Qwen3-2BVL |
| **Fine-tuning** | LoRA | LoRA (r=8, alpha=16) |
| **Label Format** | ✅ Letters (A, B, C, ...) | ✅ Letters (A, B, C, ...) - **Đã sửa** |
| **Loss Masking** | ✅ Mask prompt tokens | ✅ `train_on_responses_only` |
| **Training Objective** | ✅ Next-token prediction | ✅ Next-token prediction |

**Status**: ✅ **Đã align với LlamaRec** (sau khi sửa sang letter labels)

---

### **2. Logits Extraction**

| Aspect | LlamaRec | Project hiện tại |
|--------|----------|------------------|
| **Method** | ✅ Verbalizer approach | ✅ Verbalizer approach |
| **Logits Source** | ✅ `logits[:, -1]` | ✅ `logits[:, -1]` |
| **Token Type** | ✅ Letter tokens (A, B, C, ...) | ✅ Letter tokens (A, B, C, ...) |
| **Extraction** | ✅ Extract letter token IDs | ✅ Extract letter token IDs |
| **Softmax** | ✅ Apply on letter tokens | ✅ Apply on letter tokens |
| **Mapping** | ✅ Map letters → candidates | ✅ Map letters → candidates |

**Status**: ✅ **Giống LlamaRec**

**Code hiện tại** (`rerank/models/llm.py:363-410`):
```python
# Get token IDs for letters A-Z, a-z (LlamaRec style)
letter_tokens = []
for i in range(num_candidates):
    letter = LETTERS[i]  # "A", "B", "C", ...
    
    # Strategy 1: Try direct letter token
    token_id = self.tokenizer.convert_tokens_to_ids(letter)
    if token_id != self.tokenizer.unk_token_id:
        letter_tokens.append((i, letter, token_id))
        continue
    
    # Strategy 2: Try with space prefix
    token_id = self.tokenizer.convert_tokens_to_ids(" " + letter)
    if token_id != self.tokenizer.unk_token_id:
        letter_tokens.append((i, letter, token_id))
        continue
    
    # Strategy 3: Try encoding and taking first token
    encoded = self.tokenizer.encode(letter, add_special_tokens=False)
    if len(encoded) > 0 and encoded[0] != self.tokenizer.unk_token_id:
        letter_tokens.append((i, letter, encoded[0]))
        continue

# Extract probabilities for letter tokens
token_ids = [tid for _, _, tid in letter_tokens]
probs = F.softmax(logits[:, token_ids], dim=-1)

# Map back to candidate indices
prob_array = np.zeros(num_candidates)
for idx, (cand_idx, letter, token_id) in enumerate(letter_tokens):
    if cand_idx < num_candidates:
        prob_array[cand_idx] = probs[0, idx].item()
```

**Kết luận**: ✅ **Code hiện tại đã implement đúng verbalizer approach của LlamaRec**

---

### **3. Reranking**

| Aspect | LlamaRec | Project hiện tại |
|--------|----------|------------------|
| **Two-stage** | ✅ Retrieval + Ranking | ✅ Retrieval + Ranking |
| **Prompt Format** | ✅ Letter labels (A, B, C, ...) | ✅ Letter labels (A, B, C, ...) |
| **Output** | ✅ Single letter token | ✅ Single letter token |
| **Ranking** | ✅ Sort by probability | ✅ Sort by probability |

**Status**: ✅ **Giống LlamaRec**

**Code hiện tại** (`rerank/methods/qwen_reranker_unified.py:426-500`):
```python
def rerank(self, user_id: int, candidates: List[int], **kwargs: Any) -> List[Tuple[int, float]]:
    """Rerank candidates using LLM."""
    # 1. Build prompt với letter labels
    prompt = build_prompt_from_candidates(
        user_history, 
        candidates, 
        item_id2text,
        max_candidates=self.max_candidates
    )
    
    # 2. Extract probabilities
    probs = self.llm_model.predict_probs(prompt, num_candidates=len(candidates))
    
    # 3. Sort by probability
    ranked = sorted(
        zip(candidates, probs),
        key=lambda x: x[1],
        reverse=True
    )
    
    return ranked
```

**Kết luận**: ✅ **Code hiện tại đã implement đúng reranking process của LlamaRec**

---

## ✅ Tóm tắt

### **Training**:
- ✅ **Đã align**: Dùng letter labels, next-token prediction, mask prompt tokens
- ✅ **Code**: `rerank/models/llm.py:164-263`

### **Logits Extraction**:
- ✅ **Đã implement đúng**: Verbalizer approach với letter tokens
- ✅ **Code**: `rerank/models/llm.py:363-410`

### **Reranking**:
- ✅ **Đã implement đúng**: Two-stage, letter labels, probability-based ranking
- ✅ **Code**: `rerank/methods/qwen_reranker_unified.py:426-500`

---

## 🎯 Kết luận

**Project hiện tại đã implement đúng cách LlamaRec train, lấy logits và rerank:**

1. ✅ **Training**: Next-token prediction với letter labels, mask prompt tokens
2. ✅ **Logits Extraction**: Verbalizer approach - extract letter token logits
3. ✅ **Reranking**: Two-stage process, probability-based ranking

**Khác biệt duy nhất**:
- **Base Model**: LlamaRec dùng Llama 2-7B, project hiện tại dùng Qwen3-0.6B/Qwen3-2BVL
- **LoRA Config**: Khác nhau về rank và alpha (nhưng không ảnh hưởng đến approach)

**Recommendation**: ✅ **Code hiện tại đã đúng theo LlamaRec approach**

---

## 📚 References

- **LlamaRec Repository**: [https://github.com/Yueeeeeeee/LlamaRec](https://github.com/Yueeeeeeee/LlamaRec)
- **Paper**: "LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking" (PGAI@CIKM 2023)
- **Verbalizer Approach**: Transform output logits into probability distributions over candidate items

