# Phân tích cơ chế Rerank của các LLM Rerankers

## 📊 Tổng quan

Có 2 LLM rerankers trong project:
1. **QwenReranker** - Text-only LLM (Qwen3-0.6B)
2. **Qwen3VLReranker** - Multimodal LLM (Qwen3-VL-2B) với 4 modes

## 🔍 Cơ chế Rerank

### 1. QwenReranker (Text-only)

**Location**: `rerank/methods/qwen_reranker.py:108-178`

#### Quy trình:

```python
def rerank(self, user_id: int, candidates: List[int]) -> List[Tuple[int, float]]:
    # 1. Truncate history xuống 5 items cuối cùng
    history = self.user_history.get(user_id, [])
    history = history[-self.max_history:]  # max_history = 5
    
    # 2. Build prompt với format:
    prompt = build_prompt_from_candidates(
        history,           # User history (5 items cuối)
        candidates,        # Candidate item IDs
        self.item_id2text, # Mapping item_id -> text
        max_candidates=self.max_candidates
    )
    
    # 3. Predict probabilities từ LLM
    probs = self.llm_model.predict_probs(prompt, num_candidates=len(candidates))
    
    # 4. Rank candidates theo probabilities
    ranked_items = rank_candidates(probs, candidates)
    
    # 5. Return top-K với scores
    return [(item_id, score) for item_id in ranked_items[:self.top_k]]
```

#### Prompt Format:

```
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
- item_text_1
- item_text_2
- item_text_3
- item_text_4
- item_text_5

Candidate items:
1. candidate_text_1
2. candidate_text_2
3. candidate_text_3
...
N. candidate_text_N

Answer with only one number (1-N).
```

#### Cách Extract Probabilities:

**Location**: `rerank/models/llm.py:201-272`

```python
def predict_probs(self, prompt, num_candidates=None):
    # 1. Tokenize prompt
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    
    # 2. Forward pass
    with torch.no_grad():
        outputs = self.model(**inputs)
    
    # 3. Lấy logits của token cuối cùng (next token prediction)
    logits = outputs.logits[:, -1]  # [vocab_size]
    
    # 4. Extract token IDs cho các số 1, 2, 3, ..., num_candidates
    number_tokens = []
    for i in range(1, num_candidates + 1):
        num_str = str(i)
        token_id = self.tokenizer.convert_tokens_to_ids(num_str)
        if token_id != self.tokenizer.unk_token_id:
            number_tokens.append((i, token_id))
    
    # 5. Extract probabilities cho number tokens
    token_ids = [tid for _, tid in number_tokens]
    probs = F.softmax(logits[:, token_ids], dim=-1)  # [1, num_tokens]
    
    # 6. Map về candidate indices (1-indexed)
    prob_array = np.zeros(num_candidates)
    for idx, (cand_num, token_id) in enumerate(number_tokens):
        if cand_num <= num_candidates:
            prob_array[cand_num - 1] = probs[0, idx].item()
    
    return prob_array  # [num_candidates]
```

**Cơ chế**:
- Model được yêu cầu output một số (1-N) để chọn candidate
- Extract logits của token cuối cùng (next token prediction)
- Lấy probabilities của các number tokens (1, 2, 3, ..., N)
- Mỗi probability tương ứng với một candidate

#### Ranking:

```python
def rank_candidates(probs, candidate_ids):
    # Sort theo probability giảm dần
    ranked = sorted(
        zip(candidate_ids, probs),
        key=lambda x: x[1],
        reverse=True
    )
    return [cid for cid, _ in ranked]
```

---

### 2. Qwen3VLReranker (Multimodal)

**Location**: `rerank/methods/qwen3vl_reranker.py:154-220`

#### Quy trình:

```python
def _rerank_internal(self, user_id: int, candidates: List[int], user_history=None):
    # 1. Truncate history xuống 5 items cuối cùng
    history = history[-self.max_history:]  # max_history = 5
    
    # 2. Predict probabilities (tùy mode)
    probs = self.qwen3vl_model.predict_probs(
        user_history=history,
        candidates=candidates,
        item_meta=self.item_meta,
        num_candidates=len(candidates)
    )
    
    # 3. Rank candidates
    ranked_items = rank_candidates(probs, candidates)
    
    # 4. Return với scores
    return [(item_id, score) for item_id in ranked_items[:self.top_k]]
```

#### 4 Modes:

##### Mode 1: `raw_image`

**Location**: `rerank/models/qwen3vl.py:280-465`

```python
def _predict_probs_raw_image(self, user_history, candidates, item_meta, num_candidates):
    # 1. Load images và texts cho history items
    history_images = []
    history_texts = []
    for item_id in user_history:
        meta = item_meta.get(item_id, {})
        text = meta.get("text", f"item_{item_id}")
        image_path = meta.get("image_path")
        if image_path:
            img = Image.open(image_path).convert("RGB")
            img = resize_image_for_qwen3vl(img, max_size=448)
            history_images.append(img)
        else:
            history_images.append(None)
        history_texts.append(text)
    
    # 2. Load images và texts cho candidates
    candidate_images = []
    candidate_texts = []
    for item_id in candidates:
        meta = item_meta.get(item_id, {})
        text = meta.get("text", f"item_{item_id}")
        image_path = meta.get("image_path")
        if image_path:
            img = Image.open(image_path).convert("RGB")
            img = resize_image_for_qwen3vl(img, max_size=448)
            candidate_images.append(img)
        else:
            candidate_images.append(None)
        candidate_texts.append(text)
    
    # 3. Build messages với images
    messages = []
    # History với images
    for img, text in zip(history_images, history_texts):
        if img:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": text}
                ]
            })
        else:
            messages.append({"role": "user", "content": text})
    
    # Candidates với images
    cand_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidate_texts)])
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": img} for img in candidate_images if img
        ] + [{"type": "text", "text": f"Candidate items:\n{cand_text}\nAnswer with only one number (1-{num_candidates})."}]
    })
    
    # 4. Apply chat template và tokenize
    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = self.processor(
        text=text,
        images=[img for img in history_images + candidate_images if img],
        return_tensors="pt",
        padding=True
    ).to(self.device)
    
    # 5. Forward pass
    with torch.no_grad():
        outputs = self.model(**inputs)
    
    # 6. Extract logits và probabilities (giống QwenReranker)
    logits = outputs.logits[:, -1]  # [vocab_size]
    # Extract number token probabilities...
    return prob_array
```

##### Mode 2: `caption`

**Location**: `rerank/models/qwen3vl.py:467-553`

```python
def _predict_probs_caption(self, user_history, candidates, item_meta, num_candidates):
    # 1. Build history text với captions
    history_texts = []
    for item_id in user_history:
        meta = item_meta.get(item_id, {})
        text = meta.get("text", f"item_{item_id}")
        caption = meta.get("caption", "")
        if caption:
            history_texts.append(f"{text} (Image: {caption})")
        else:
            history_texts.append(text)
    
    # 2. Build candidate texts với captions
    candidate_texts = []
    for item_id in candidates:
        meta = item_meta.get(item_id, {})
        text = meta.get("text", f"item_{item_id}")
        caption = meta.get("caption", "")
        if caption:
            candidate_texts.append(f"{text} (Image: {caption})")
        else:
            candidate_texts.append(text)
    
    # 3. Build prompt (text-only, không có images)
    prompt = self._build_rerank_prompt(history_texts, candidate_texts)
    
    # 4. Tokenize và predict (giống QwenReranker)
    inputs = self.processor.tokenizer(prompt, return_tensors="pt").to(self.device)
    # ... extract probabilities
```

##### Mode 3: `semantic_summary`

**Location**: `rerank/models/qwen3vl.py:554-626`

```python
def _predict_probs_semantic_summary_vl(self, user_history, candidates, item_meta, num_candidates):
    # Tương tự caption mode nhưng dùng semantic_summary thay vì caption
    # Format: "{text} (Semantic: {semantic_summary})"
    # Có thể dùng images nếu cần (tùy implementation)
```

##### Mode 4: `semantic_summary_small`

**Location**: `rerank/models/qwen3vl.py:627-700`

```python
def _predict_probs_semantic_summary_text(self, user_history, candidates, item_meta, num_candidates):
    # Text-only mode với semantic summaries
    # Format: "{text} (Semantic: {semantic_summary})"
    # Không dùng images
```

#### Prompt Format (chung cho tất cả modes):

```
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
- history_item_1
- history_item_2
- history_item_3
- history_item_4
- history_item_5

Candidate items:
1. candidate_item_1
2. candidate_item_2
3. candidate_item_3
...
N. candidate_item_N

Answer with only one number (1-N).
```

**Khác biệt giữa modes**:
- `raw_image`: History và candidates có images trong messages
- `caption`: History và candidates có format `{text} (Image: {caption})`
- `semantic_summary`: History và candidates có format `{text} (Semantic: {semantic_summary})`
- `semantic_summary_small`: Giống semantic_summary nhưng text-only

#### Cách Extract Probabilities:

Tương tự QwenReranker:
1. Forward pass để lấy logits
2. Extract logits của token cuối cùng
3. Lấy probabilities của number tokens (1, 2, 3, ..., N)
4. Map về candidate indices

---

## 🔑 Điểm quan trọng

### 1. Single-Token Output

- Model được yêu cầu output **chỉ một số** (1-N) để chọn candidate
- Không phải ranking toàn bộ candidates trong một output
- Extract probabilities từ **next token prediction** (logits của token cuối cùng)

### 2. Number Tokens

- Sử dụng số (1, 2, 3, ..., N) thay vì chữ cái (A, B, C, ...)
- Hỗ trợ nhiều candidates hơn (không giới hạn ở 20 như chữ cái)
- Extract token IDs cho các số từ vocabulary

### 3. Probability Extraction

- Lấy logits của token cuối cùng: `logits = outputs.logits[:, -1]`
- Extract probabilities cho number tokens: `probs = F.softmax(logits[:, token_ids], dim=-1)`
- Map về candidate indices: `prob_array[cand_num - 1] = probs[0, idx].item()`

### 4. Ranking

- Sort candidates theo probability giảm dần
- Return top-K items với scores

### 5. History Truncation

- Chỉ giữ lại **5 items cuối cùng** trong history
- Áp dụng cho cả QwenReranker và Qwen3VLReranker

---

## 📝 So sánh

| Aspect | QwenReranker | Qwen3VLReranker |
|--------|--------------|-----------------|
| **Input** | Text only | Text + Images (tùy mode) |
| **History** | 5 items cuối | 5 items cuối |
| **Prompt** | Text-only | Multimodal (tùy mode) |
| **Extract Probs** | Number tokens | Number tokens |
| **Ranking** | Sort by prob | Sort by prob |
| **Modes** | 1 mode | 4 modes (raw_image, caption, semantic_summary, semantic_summary_small) |

---

## ⚠️ Lưu ý

1. **Tokenization**: Number tokens có thể không tồn tại trong vocabulary → fallback về letters
2. **Image Processing**: Images được resize về 448px (max_size) để tiết kiệm memory
3. **Batch Processing**: Hiện tại process từng user một, không batch
4. **Max Candidates**: Có thể giới hạn số candidates qua `max_candidates` parameter

