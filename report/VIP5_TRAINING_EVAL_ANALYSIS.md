# Phân tích VIP5 Training và Evaluation

## ✅ Những gì đang hoạt động ĐÚNG

### 1. **Training Loop** ✅
- Có training loop đầy đủ với epochs, batches
- Loss calculation đúng: reshape loss, mask padding tokens, apply loss weights
- Có validation và early stopping
- Có optimizer và gradient clipping
- Training format đúng: Direct Task (B-5) template

### 2. **Loss Calculation** ✅
- Loss được tính đúng: `reduce_loss=False` để lấy per-token loss
- Mask padding tokens: `target_mask = (target_ids != -100).float()`
- Per-sample loss: `(loss * target_mask).sum(dim=1) / target_mask.sum(dim=1)`
- Apply loss weights: `(per_sample_loss * loss_weights).mean()`
- Giống với reference implementation trong `retrieval/vip5_temp/src/model.py`

### 3. **Evaluation Setup** ✅
- Có `_evaluate_split()` method
- Load pre-generated candidates từ `evaluation.utils.load_rerank_candidates`
- Recall calculation đúng: `hits / len(gt_items)` (công thức chuẩn)
- Batch processing để tối ưu tốc độ

### 4. **Model Initialization** ✅
- Load checkpoint nếu có
- Initialize từ T5 backbone nếu không có checkpoint
- Adapter support (default: enabled)
- Visual features được load đúng

---

## ⚠️ Vấn đề tiềm ẩn cần kiểm tra

### 1. **Scoring Logic trong Rerank() - CÓ THỂ SAI** ⚠️

**Location**: `rerank/methods/vip5_reranker.py:655-679` và `1078-1102`

**Vấn đề**:
- Code đang lấy logit tại position `first_token_idx` để predict token tại position `first_token_idx`
- Nhưng trong seq2seq T5, logit tại position `t` dự đoán token tại position `t+1`
- Decoder input: `[pad_token, item_token1, item_token2, ...]`
- Logit tại position 0 dự đoán token tại position 1
- Logit tại position 1 dự đoán token tại position 2

**Code hiện tại**:
```python
# Line 668-669
if first_token_idx < logits.size(1) - 1:
    score = float(logits[i, first_token_idx, item_token_id].item())
```

**Vấn đề**: 
- `decoder_input_ids[i, first_token_idx]` là token tại position `first_token_idx`
- Nhưng `logits[i, first_token_idx, ...]` dự đoán token tại position `first_token_idx + 1`
- Nên lấy logit tại position `first_token_idx` để predict token tại position `first_token_idx + 1`

**Giải pháp đề xuất**:
```python
# Option 1: Lấy logit tại position 0 để predict token tại position 1 (first token của item)
# Decoder input: [pad_token, item_token1, item_token2, ...]
# Logit[0] predicts item_token1
first_token_idx = 0  # Always use position 0
item_token_id = decoder_input_ids[i, 1].item()  # Token tại position 1
score = float(logits[i, 0, item_token_id].item())  # Logit tại position 0

# Option 2: Sum logits cho tất cả tokens của item_id
# Lấy tất cả tokens của "item_{item_id}" và sum logits
item_tokens = decoder_input_ids[i][decoder_attention_mask[i] == 1]  # All non-padding tokens
scores = []
for pos in range(len(item_tokens) - 1):  # -1 vì logit tại pos predicts token tại pos+1
    token_id = item_tokens[pos + 1].item()
    logit = logits[i, pos, token_id].item()
    scores.append(logit)
score = sum(scores) / len(scores)  # Average logit
```

### 2. **Decoder Input Format trong Inference** ⚠️

**Vấn đề**:
- Trong training, decoder_input_ids được tự động shift từ labels (thêm pad_token_id ở đầu)
- Trong inference (rerank), code đang pass decoder_input_ids trực tiếp từ tokenizer
- Tokenizer có thể không thêm pad_token_id ở đầu

**Kiểm tra**:
- T5 tokenizer thường thêm pad_token_id ở đầu khi tokenize với `add_special_tokens=True`
- Nhưng code đang dùng `add_special_tokens=False` (line 627, 1055)
- Cần đảm bảo decoder_input_ids bắt đầu với pad_token_id

**Giải pháp**:
```python
# Đảm bảo decoder_input_ids bắt đầu với pad_token_id
decoder_input_ids = decoder_inputs_tokenized["input_ids"].to(self.device)
pad_token_id = self.tokenizer.pad_token_id
# Prepend pad_token_id nếu chưa có
if decoder_input_ids[0, 0] != pad_token_id:
    pad_tokens = torch.full((decoder_input_ids.size(0), 1), pad_token_id, device=self.device)
    decoder_input_ids = torch.cat([pad_tokens, decoder_input_ids], dim=1)
```

### 3. **Training Sample Preparation** ✅ (Có thể cải thiện)

**Hiện tại**:
- Target item là item cuối cùng trong history: `target_item = items[-1]`
- Sample negatives từ tất cả items (trừ user history)
- Shuffle candidates để tránh bias

**Có thể cải thiện**:
- Có thể thử các strategies khác: random target từ history, hoặc target từ middle
- Nhưng hiện tại đã đúng với sequential recommendation

---

## 🔍 Kiểm tra cần thực hiện

### 1. **Kiểm tra Decoder Input Format**
```python
# Trong rerank(), sau khi tokenize:
print(f"Decoder input IDs shape: {decoder_input_ids.shape}")
print(f"First token IDs: {decoder_input_ids[0, :5]}")
print(f"Pad token ID: {self.tokenizer.pad_token_id}")
# Đảm bảo decoder_input_ids[0, 0] == pad_token_id
```

### 2. **Kiểm tra Scoring Logic**
```python
# So sánh 2 cách tính score:
# Cách 1: Hiện tại (có thể sai)
score1 = logits[i, first_token_idx, item_token_id]

# Cách 2: Đề xuất (sum logits cho tất cả tokens)
item_tokens = decoder_input_ids[i][decoder_attention_mask[i] == 1]
scores = []
for pos in range(len(item_tokens) - 1):
    token_id = item_tokens[pos + 1].item()
    logit = logits[i, pos, token_id].item()
    scores.append(logit)
score2 = sum(scores) / len(scores)

# So sánh score1 vs score2
```

### 3. **Kiểm tra Training vs Inference Consistency**
- Training: decoder_input_ids được shift từ labels
- Inference: decoder_input_ids được tokenize trực tiếp
- Đảm bảo format giống nhau

---

## 📊 Tóm tắt

| Aspect | Status | Notes |
|--------|--------|-------|
| **Training Loop** | ✅ Đúng | Có đầy đủ epochs, batches, validation |
| **Loss Calculation** | ✅ Đúng | Giống reference implementation |
| **Evaluation Setup** | ✅ Đúng | Load candidates, tính recall đúng |
| **Scoring Logic** | ⚠️ Có thể sai | Cần kiểm tra decoder input format và logit indexing |
| **Decoder Input Format** | ⚠️ Cần kiểm tra | Đảm bảo có pad_token_id ở đầu |

---

## 🎯 Đề xuất sửa

### Priority 1: Sửa Scoring Logic
1. Kiểm tra decoder input format (có pad_token_id ở đầu không)
2. Sửa cách lấy logit: lấy logit tại position `t` để predict token tại position `t+1`
3. Hoặc sum logits cho tất cả tokens của item_id

### Priority 2: Đảm bảo Consistency
1. Đảm bảo decoder input format giống nhau giữa training và inference
2. Test với một vài samples để verify scoring logic

### Priority 3: Thêm Debug Logging
1. Log decoder input IDs format
2. Log scoring values để debug
3. So sánh scores giữa các candidates

---

## ✅ Kết luận

**VIP5 training và evaluation đang hoạt động ĐÚNG về cơ bản**, nhưng có **một vấn đề tiềm ẩn với scoring logic** trong `rerank()` method:

1. **Scoring logic có thể sai**: Cách lấy logit để tính score có thể không đúng với seq2seq model
2. **Decoder input format cần kiểm tra**: Đảm bảo có pad_token_id ở đầu

**Khuyến nghị**: 
- Kiểm tra và sửa scoring logic
- Test lại với một vài samples để verify
- So sánh performance trước và sau khi sửa

