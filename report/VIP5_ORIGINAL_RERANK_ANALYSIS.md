# Phân tích cách VIP5 gốc thực hiện Rerank

## 📚 Nguồn tham khảo

Repo gốc: https://github.com/jeykigung/VIP5/tree/main/src

File tham khảo:
- `retrieval/vip5_temp/notebooks/evaluate_VIP5.ipynb` (Cell 17, 18, 20)
- `retrieval/vip5_temp/src/model.py` (method `generate_step`)

---

## 🔍 Cách VIP5 gốc thực hiện Rerank

### 1. **Sử dụng Beam Search Generation** ✅

**Code từ notebook (Cell 17, 20)**:
```python
# Generate top-K items sử dụng beam search
beam_outputs = model.generate(
    input_ids=batch['input_ids'].to('cuda'), 
    whole_word_ids=batch['whole_word_ids'].to('cuda'), 
    category_ids=batch['category_ids'].to('cuda'), 
    vis_feats=batch['vis_feats'].to('cuda'), 
    task=batch["task"][0],
    max_length=50, 
    num_beams=20,                    # ✅ Beam size = 20
    no_repeat_ngram_size=0, 
    num_return_sequences=20,          # ✅ Return top-20 sequences
    early_stopping=True
)

# Decode generated sequences
generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
```

**Đặc điểm**:
- Sử dụng **beam search** để generate top-K items
- `num_beams=20`: Beam size = 20
- `num_return_sequences=20`: Return top-20 sequences
- Model **generate** toàn bộ sequence (không chỉ tính logit)

### 2. **Scoring dựa trên Rank trong Beam Search** ✅

**Code từ notebook**:
```python
gt = {}
ui_scores = {}
for i, info in enumerate(all_info):
    gt[i] = [int(info['target_item'])]
    pred_dict = {}
    for j in range(len(info['gen_item_list'])):
        try:
            # ✅ Score = negative rank (rank 1 -> score -1, rank 2 -> score -2, ...)
            pred_dict[int(info['gen_item_list'][j])] = -(j+1)
        except:
            pass
    ui_scores[i] = pred_dict

# Evaluate
evaluate_all(ui_scores, gt, 5)
evaluate_all(ui_scores, gt, 10)
```

**Đặc điểm**:
- **KHÔNG tính logit trực tiếp**
- Score = **negative rank** trong beam search results
  - Item đầu tiên (rank 1) → score = -1
  - Item thứ 2 (rank 2) → score = -2
  - ...
  - Item thứ 20 (rank 20) → score = -20
- Score càng cao (ít âm hơn) = rank càng cao = item càng tốt

### 3. **Direct Task (B-5) cho Reranking** ✅

**Từ `data.py` (line 452-471)**:
```python
# Direct Task template B-5
template = "Which item of the following to recommend for user_{} ? \n {}"

# Source text: chứa TẤT CẢ candidates với visual token placeholders
candidates_with_visual = ' {}, '.format('<extra_id_0> ' * image_feature_size_ratio).join(candidate_samples) + ' <extra_id_0>' * image_feature_size_ratio
source_text = template.format(user_id, candidates_with_visual)

# Target text: chỉ item target
target_text = f"item_{target_item}"
```

**Đặc điểm**:
- Prompt chứa **TẤT CẢ candidates** trong một prompt
- Visual features cho TẤT CẢ candidates
- Model generate **một item** từ danh sách candidates

---

## ⚠️ So sánh với Implementation hiện tại

### Implementation hiện tại (SAI) ❌

**Location**: `rerank/methods/vip5_reranker.py:519-685`

**Cách làm**:
1. Encode prompt với TẤT CẢ candidates (✅ Đúng)
2. Decode từng candidate riêng lẻ để tính logit (❌ SAI)
3. Score = logit của item_id token (❌ SAI)

**Code hiện tại**:
```python
# ❌ SAI: Decode từng candidate riêng lẻ
decoder_input_texts = [f"item_{item_id}" for item_id in valid_candidates]
decoder_inputs_tokenized = self.tokenizer(...)
decoder_outputs = self.model.decoder(...)
logits = self.model.lm_head(decoder_hidden)

# ❌ SAI: Lấy logit trực tiếp
score = float(logits[i, first_token_idx, item_token_id].item())
```

**Vấn đề**:
- Không sử dụng **beam search generation**
- Tính logit trực tiếp (không đúng với cách VIP5 gốc)
- Không generate sequence, chỉ decode một lần

### Cách VIP5 gốc (ĐÚNG) ✅

**Cách làm**:
1. Encode prompt với TẤT CẢ candidates (✅ Đúng)
2. **Generate** top-K items sử dụng **beam search** (✅ Đúng)
3. Score = **negative rank** trong beam search results (✅ Đúng)

**Code gốc**:
```python
# ✅ ĐÚNG: Generate với beam search
beam_outputs = model.generate(
    input_ids=input_ids,
    whole_word_ids=whole_word_ids,
    category_ids=category_ids,
    vis_feats=vis_feats,
    task="direct",
    max_length=50,
    num_beams=20,
    num_return_sequences=20,
    early_stopping=True
)

# ✅ ĐÚNG: Score = negative rank
for j in range(len(generated_items)):
    pred_dict[int(generated_items[j])] = -(j+1)
```

---

## 🔧 Cách sửa Implementation

### Option 1: Sửa theo cách VIP5 gốc (Recommended) ✅

**Sửa `rerank()` method**:
```python
def rerank(
    self,
    user_id: int,
    candidates: List[int],
    **kwargs: Any
) -> List[Tuple[int, float]]:
    """Rerank candidates sử dụng VIP5 beam search generation (theo cách gốc)."""
    self._validate_fitted()
    
    if not candidates:
        return []
    
    # Get visual features for candidates
    valid_candidates = []
    candidate_visual = []
    for item_id in candidates:
        if item_id in self.item_id_to_idx:
            idx = self.item_id_to_idx[item_id]
            valid_candidates.append(item_id)
            candidate_visual.append(self.visual_embeddings[idx])
    
    if not valid_candidates:
        return []
    
    # Build prompt với Direct Task template (B-5)
    visual_token_placeholder = " <extra_id_0>" * self.image_feature_size_ratio
    candidates_with_visual = visual_token_placeholder.join([f"item_{c}" for c in valid_candidates]) + visual_token_placeholder
    direct_prompt = f"Which item of the following to recommend for user_{user_id} ? \n {candidates_with_visual}"
    
    # Prepare visual features
    all_candidates_visual_tensor = torch.stack(candidate_visual)  # [num_candidates, feat_dim]
    
    # Prepare VIP5 input
    vip5_input = prepare_vip5_input(
        direct_prompt,
        all_candidates_visual_tensor,
        self.tokenizer,
        max_length=self.max_text_length,
        image_feature_size_ratio=self.image_feature_size_ratio,
    )
    
    # Move to device
    input_ids = vip5_input["input_ids"].to(self.device)
    whole_word_ids = vip5_input["whole_word_ids"].to(self.device)
    category_ids = vip5_input["category_ids"].to(self.device)
    vis_feats = vip5_input["vis_feats"].to(self.device)
    
    # ✅ Generate với beam search (theo cách gốc)
    self.model.eval()
    with torch.no_grad():
        # Generate top-K items
        num_beams = min(len(valid_candidates), 20)  # Beam size = min(num_candidates, 20)
        num_return_sequences = min(len(valid_candidates), self.top_k)  # Return top-K
        
        beam_outputs = self.model.generate(
            input_ids=input_ids,
            whole_word_ids=whole_word_ids,
            category_ids=category_ids,
            vis_feats=vis_feats,
            task="direct",
            max_length=64,  # gen_max_length from VIP5
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=0,
            early_stopping=True,
        )
        
        # Decode generated sequences
        generated_sents = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
    
    # ✅ Score = negative rank (theo cách gốc)
    scores = []
    for rank, generated_text in enumerate(generated_sents):
        try:
            # Extract item_id from generated text (e.g., "item_123" -> 123)
            item_id = int(generated_text.replace("item_", "").strip())
            if item_id in valid_candidates:
                # Score = negative rank (rank 0 -> score -1, rank 1 -> score -2, ...)
                score = -(rank + 1)
                scores.append((item_id, score))
        except ValueError:
            # Skip invalid generated text
            continue
    
    # Sort by score descending (higher score = better rank)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Fill missing candidates with worst scores
    missing_candidates = set(valid_candidates) - {item_id for item_id, _ in scores}
    for item_id in missing_candidates:
        scores.append((item_id, -len(valid_candidates) - 1))
    
    # Return top_k
    return scores[:self.top_k]
```

### Option 2: Hybrid Approach (Nếu beam search quá chậm)

Nếu beam search quá chậm với nhiều candidates, có thể:
1. Dùng beam search cho top-K candidates (sau khi filter)
2. Hoặc dùng greedy generation thay vì beam search
3. Hoặc tính logit trực tiếp nhưng **đúng cách** (sum logits cho tất cả tokens)

---

## 📊 So sánh Performance

| Aspect | VIP5 Gốc | Implementation hiện tại |
|--------|----------|------------------------|
| **Method** | Beam search generation | Direct logit calculation |
| **Scoring** | Negative rank | Logit value |
| **Speed** | Chậm hơn (beam search) | Nhanh hơn (direct decode) |
| **Accuracy** | ✅ Đúng (theo paper) | ❌ Có thể sai |
| **Consistency** | ✅ Giống training | ⚠️ Khác với training |

---

## ✅ Kết luận

**VIP5 gốc sử dụng beam search generation để rerank**, không phải tính logit trực tiếp. 

**Cách đúng**:
1. Encode prompt với TẤT CẢ candidates
2. Generate top-K items sử dụng **beam search**
3. Score = **negative rank** trong beam search results

**Implementation hiện tại SAI** vì:
- Không sử dụng beam search
- Tính logit trực tiếp (không đúng với cách gốc)
- Không generate sequence

**Khuyến nghị**: Sửa `rerank()` method để sử dụng beam search generation như VIP5 gốc.

