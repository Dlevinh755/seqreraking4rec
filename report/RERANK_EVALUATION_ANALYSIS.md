# Phân tích quá trình Evaluation của các Rerank Models

## Tổng quan

Quá trình evaluation của các rerank models được thực hiện ở 2 giai đoạn:
1. **Trong training (validation)**: Sử dụng `_evaluate_split()` method sau mỗi epoch
2. **Sau training (final evaluation)**: Sử dụng `evaluate_split()` từ `evaluation/utils.py` với `recommend_fn`

---

## 1. Evaluation trong Training (Validation)

### 1.1. Các Rerankers có `_evaluate_split()` method

Tất cả các rerankers đều có method `_evaluate_split()` được gọi sau mỗi epoch để tính validation recall:

- **VIP5Reranker** (`rerank/methods/vip5_reranker.py:747`)
- **BERT4RecReranker** (`rerank/methods/bert4rec_reranker.py:327`)
- **Qwen3VLReranker** (`rerank/methods/qwen3vl_reranker.py:996`)

### 1.2. Quy trình Evaluation trong Training

```python
# Sau mỗi epoch:
val_recall = self._evaluate_split(val_data, k=min(10, self.top_k))
```

**Các bước chung trong `_evaluate_split()`:**

1. **Lấy user history**: Từ `self.user_history[user_id]` hoặc `self.train_user_history[user_id]`
2. **Lấy candidates**: 
   - Lấy tất cả items từ dataset
   - Sample ngẫu nhiên `rerank_eval_candidates` items (default: 20)
   - Đảm bảo ít nhất 1 ground truth item có trong candidates
3. **Rerank candidates**: Gọi `self.rerank(user_id, candidates)`
4. **Tính Recall@K**: So sánh top-K items với ground truth

### 1.3. ⚠️ Vấn đề: Không mask history items

**Vấn đề**: Trong `_evaluate_split()`, các rerankers **KHÔNG loại bỏ history items** khỏi candidates pool trước khi sample.

**Ví dụ từ VIP5Reranker:**
```python
# Get all items as candidates (for evaluation)
all_items = list(self.item_id_to_idx.keys())  # ❌ Bao gồm cả history items

# Sample candidates
candidates = random.sample(all_items, max_eval_candidates)  # ❌ Có thể sample history items
```

**Hậu quả**:
- Model có thể recommend lại items đã mua (history items)
- Validation recall có thể bị inflated nếu history items được sample và recommend
- Không phản ánh đúng performance thực tế

**Tương tự với BERT4Rec và Qwen3VL:**
- BERT4Rec: `all_items = list(range(1, self.vocab_size))` - không exclude history
- Qwen3VL: `all_items = set()` từ `train_user_history.values()` - không exclude history của user hiện tại

---

## 2. Evaluation sau Training (Final Evaluation)

### 2.1. Script: `scripts/train_rerank_standalone.py`

Sau khi training xong, script sử dụng `recommend_fn` để evaluate:

```python
def recommend_fn(user_id, ground_truth=None):
    if args.mode == "ground_truth":
        # Ground truth mode: use gt + random negatives
        # ✅ CÓ exclude history items
        user_history = set(train.get(user_id, []))
        exclude_set = user_history - set(ground_truth)
        candidate_pool = all_items - exclude_set - set(ground_truth)
        
    else:  # retrieval mode
        # ✅ CÓ exclude history items
        exclude_set = set(train.get(user_id, []))
        candidates = retriever.retrieve(user_id, exclude_items=exclude_set)
```

### 2.2. Ground Truth Mode

**Quy trình:**
1. ✅ **Exclude history items**: `exclude_set = user_history - set(ground_truth)`
2. ✅ **Sample candidates**: 1 GT item + (max_candidates - 1) negatives
3. ✅ **Shuffle candidates**: Tránh bias
4. ✅ **Rerank**: Gọi `reranker.rerank(user_id, candidates)`

**Điểm tốt:**
- ✅ Loại bỏ history items khỏi candidate pool
- ✅ Chỉ đảm bảo 1 GT item trong candidates (realistic)
- ✅ Shuffle để tránh bias

### 2.3. Retrieval Mode

**Quy trình:**
1. ✅ **Exclude history items**: `exclude_set = set(train.get(user_id, []))`
2. ✅ **Retrieve candidates**: `retriever.retrieve(user_id, exclude_items=exclude_set)`
3. ✅ **Rerank**: Gọi `reranker.rerank(user_id, candidates)`

**Điểm tốt:**
- ✅ Loại bỏ history items
- ✅ Sử dụng retrieval stage để lấy candidates

---

## 3. So sánh: Training vs Final Evaluation

| Aspect | Training (`_evaluate_split`) | Final Evaluation (`recommend_fn`) |
|--------|------------------------------|-----------------------------------|
| **History masking** | ❌ **KHÔNG** | ✅ **CÓ** |
| **Candidate sampling** | Random từ all_items | GT + negatives (ground_truth mode) hoặc retrieval (retrieval mode) |
| **Number of candidates** | `rerank_eval_candidates` (default: 20) | `rerank_eval_candidates` (default: 20) |
| **GT guarantee** | Đảm bảo ít nhất 1 GT | Đảm bảo 1 GT (ground_truth mode) |
| **Metrics** | Chỉ Recall@K | Recall@K, NDCG@K, Hit@K |

---

## 4. ⚠️ Vấn đề cần sửa

### 4.1. Vấn đề chính: Không mask history items trong `_evaluate_split()`

**Ảnh hưởng:**
- Validation recall có thể bị inflated
- Model có thể được đánh giá cao hơn thực tế
- Không nhất quán với final evaluation (có mask history)

**Giải pháp:**
Cần sửa `_evaluate_split()` của tất cả rerankers để:
1. Exclude history items khỏi candidate pool
2. Chỉ sample từ items chưa được user tương tác

### 4.2. Các rerankers cần sửa:

1. **VIP5Reranker** (`rerank/methods/vip5_reranker.py:747`)
2. **BERT4RecReranker** (`rerank/methods/bert4rec_reranker.py:327`)
3. **Qwen3VLReranker** (`rerank/methods/qwen3vl_reranker.py:996`)

---

## 5. Đề xuất sửa lỗi

### 5.1. Pattern chung để sửa:

```python
def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> float:
    # ... existing code ...
    
    for user_id, gt_items in split.items():
        # Get user history
        history = self.user_history[user_id]
        
        # ✅ FIX: Exclude history items from candidate pool
        all_items = list(self.item_id_to_idx.keys())  # or range(1, vocab_size)
        history_set = set(history)
        candidate_pool = [item for item in all_items if item not in history_set]
        
        # Sample from candidate_pool (not all_items)
        candidates = random.sample(candidate_pool, max_eval_candidates)
        
        # ... rest of evaluation ...
```

### 5.2. Lợi ích:

- ✅ Validation recall phản ánh đúng performance
- ✅ Nhất quán với final evaluation
- ✅ Tránh recommend lại items đã mua
- ✅ Evaluation công bằng hơn

---

## 6. Tóm tắt

### ✅ Điểm tốt:
- Final evaluation có mask history items
- Ground truth mode chỉ đảm bảo 1 GT item (realistic)
- Sử dụng cùng metrics (Recall, NDCG, Hit) cho tất cả methods

### ⚠️ Vấn đề:
- **Training validation không mask history items** → có thể bị inflated recall
- Không nhất quán giữa training validation và final evaluation

### 🔧 Cần sửa:
- Thêm history masking vào `_evaluate_split()` của tất cả rerankers

