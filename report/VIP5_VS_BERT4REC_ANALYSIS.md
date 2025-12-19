# Phân tích tại sao VIP5 có Recall thấp hơn BERT4Rec

## 🔍 Vấn đề phát hiện

### 1. **Cách tính Score của VIP5 - KHÔNG ĐÚNG** ❌

**Location**: `rerank/methods/vip5_reranker.py:855` và `567`

```python
# VIP5 hiện tại:
encoder_hidden = encoder_outputs.last_hidden_state  # [1, seq_len, d_model]
score = float(encoder_hidden.mean(dim=1).squeeze(0).norm().item())
```

**Vấn đề**:
- VIP5 là một **seq2seq model** (T5-based), nhưng chỉ dùng **encoder output**
- Score được tính bằng **norm của mean pooling** - không có ý nghĩa về mặt recommendation
- Không sử dụng **decoder** để predict probability của item_id
- Score này không phản ánh khả năng model predict item đó

**So sánh với BERT4Rec**:
```python
# BERT4Rec - ĐÚNG:
scores = self.model.predict_scores(history_tensor, candidates_tensor)  # [1, num_candidates]
# predict_scores() tính score thực sự cho từng candidate dựa trên model output
```

### 2. **Công thức Recall - Có thể cải thiện** ⚠️

**VIP5**:
```python
recalls.append(hits / min(k, len(gt_items)))  # Line 868
```

**BERT4Rec**:
```python
recalls.append(hits / min(k, len(valid_gt_items)))  # Line 414
```

**Qwen3VL** (đã sửa):
```python
recalls.append(hits / len(gt_items))  # Công thức chuẩn
```

**Phân tích**:
- Cả VIP5 và BERT4Rec đều dùng `min(k, len(gt_items))` - có thể không cần thiết
- Công thức chuẩn nên là: `hits / len(gt_items)`
- Tuy nhiên, trong thực tế thường `k >= len(gt_items)`, nên không ảnh hưởng nhiều

### 3. **Cách Rerank - VIP5 chậm hơn** ⚠️

**VIP5**:
- Encode **từng candidate một cách riêng lẻ** (loop qua từng item)
- Mỗi candidate cần một forward pass riêng
- Rất chậm với nhiều candidates

**BERT4Rec**:
- Batch processing: encode history một lần, predict scores cho tất cả candidates cùng lúc
- Nhanh hơn nhiều

### 4. **Training Process - Có thể khác biệt** ⚠️

Cần kiểm tra:
- VIP5 training có đúng không?
- Loss function có phù hợp không?
- Model có converge không?

---

## 🔧 Đề xuất sửa

### Priority 1: Sửa cách tính Score của VIP5

VIP5 nên dùng **decoder** để tính score thực sự cho item_id:

```python
# Thay vì:
score = float(encoder_hidden.mean(dim=1).squeeze(0).norm().item())

# Nên dùng:
# Option 1: Generate và score
decoder_input_ids = tokenizer.encode(f"item_{item_id}", return_tensors="pt").to(device)
decoder_outputs = model.decoder(
    input_ids=decoder_input_ids,
    encoder_hidden_states=encoder_hidden,
    encoder_attention_mask=attention_mask,
    return_dict=True
)
logits = model.lm_head(decoder_outputs.last_hidden_state)  # [1, seq_len, vocab_size]
# Score = logit của item_id token

# Option 2: Direct prediction (nếu có method)
score = model.predict_item_score(encoder_hidden, item_id)
```

### Priority 2: Sửa công thức Recall

```python
# Thay vì:
recalls.append(hits / min(k, len(gt_items)))

# Nên dùng:
recalls.append(hits / len(gt_items))  # Công thức chuẩn
```

### Priority 3: Batch Processing (Optional)

Có thể batch encode nhiều candidates cùng lúc để tăng tốc, nhưng cần đảm bảo đúng logic.

---

## 📊 So sánh chi tiết

| Aspect | VIP5 | BERT4Rec |
|--------|------|----------|
| **Score Calculation** | ❌ Norm của encoder output (không đúng) | ✅ predict_scores() method (đúng) |
| **Decoder Usage** | ❌ Không dùng | ✅ Dùng trong predict_scores() |
| **Batch Processing** | ❌ Loop từng candidate | ✅ Batch tất cả candidates |
| **Recall Formula** | ⚠️ `hits / min(k, len(gt))` | ⚠️ `hits / min(k, len(gt))` |
| **Speed** | ❌ Chậm (loop từng item) | ✅ Nhanh (batch) |
| **Model Type** | Seq2Seq (T5) | Encoder-only (BERT) |

---

## 🎯 Kết luận

**Nguyên nhân chính**: VIP5 đang dùng cách tính score **KHÔNG ĐÚNG**:
- Chỉ dùng encoder output
- Tính norm của mean pooling - không có ý nghĩa
- Không dùng decoder để predict probability của item_id

**Giải pháp**: Đã sửa cách tính score để dùng decoder và tính logit thực sự của item_id token.

## ✅ Đã sửa

1. **Cách tính Score**: 
   - Trước: `score = encoder_hidden.mean().norm()` ❌
   - Sau: Dùng decoder để predict `item_{item_id}`, lấy logit của item_id token ✅

2. **Prompt Format**:
   - Trước: Prompt có cả history + candidate ❌
   - Sau: Prompt chỉ có history (giống training format) ✅

3. **Công thức Recall**:
   - Trước: `hits / min(k, len(gt_items))` ⚠️
   - Sau: `hits / len(gt_items)` (công thức chuẩn) ✅

**Kỳ vọng**: Sau khi sửa, VIP5 recall sẽ tăng đáng kể và có thể ngang bằng hoặc cao hơn BERT4Rec.

