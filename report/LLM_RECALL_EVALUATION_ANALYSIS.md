# Phân tích cách tính Recall cho LLM Rerankers

## 📊 Tổng quan

Có 2 LLM rerankers trong project:
1. **QwenReranker** (`rerank/methods/qwen_reranker.py`) - Text-only LLM
2. **Qwen3VLReranker** (`rerank/methods/qwen3vl_reranker.py`) - Multimodal LLM

## 🔍 Cách tính Recall

### 1. Qwen3VLReranker - Có `_evaluate_split()` method

**Location**: `rerank/methods/qwen3vl_reranker.py:1131-1204`

**Quy trình**:
```python
def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> float:
    recalls = []
    
    for user_id, gt_items in split.items():
        # 1. Lấy user history (đã được truncate xuống 5 items cuối cùng)
        history = self.train_user_history[user_id]
        
        # 2. Tạo candidate pool từ tất cả items trong dataset
        all_items = set()
        for items in self.train_user_history.values():
            all_items.update(items)
        all_items = list(all_items)
        
        # 3. ✅ EXCLUDE history items khỏi candidate pool
        history_set = set(history)
        candidate_pool = [item for item in all_items if item not in history_set]
        
        # 4. Sample candidates (default: 20 candidates từ config)
        max_eval_candidates = getattr(arg, 'rerank_eval_candidates', 20)
        candidates = random.sample(candidate_pool, max_eval_candidates)
        
        # 5. ✅ Đảm bảo ít nhất 1 GT item có trong candidates
        if not any(item in candidates for item in gt_items):
            candidates[0] = gt_items[0]
        
        # 6. ✅ Shuffle candidates để tránh bias
        random.shuffle(candidates)
        
        # 7. Rerank candidates
        reranked = self._rerank_internal(user_id, candidates, user_history=history)
        
        # 8. Lấy top-K items
        top_k_items = [item_id for item_id, _ in reranked[:k]]
        
        # 9. Tính recall
        hits = len(set(top_k_items) & set(gt_items))
        if len(gt_items) > 0:
            recalls.append(hits / min(k, len(gt_items)))
    
    return float(np.mean(recalls)) if recalls else 0.0
```

**Công thức Recall**:
```python
recall = hits / min(k, len(gt_items))
```
- `hits`: Số GT items có trong top-K
- `k`: Cutoff (thường là 10)
- `len(gt_items)`: Số lượng GT items

**⚠️ Vấn đề tiềm ẩn**:
- Công thức `hits / min(k, len(gt_items))` có thể không chuẩn
- Recall chuẩn nên là: `hits / len(gt_items)` (không cần `min(k, len(gt_items))`)
- Nếu `len(gt_items) > k`, công thức hiện tại sẽ cho recall cao hơn thực tế

**Ví dụ**:
- GT items: [1, 2, 3, 4, 5] (5 items)
- Top-10: [1, 2, 6, 7, 8, 9, 10, 11, 12, 13] (hits = 2)
- Công thức hiện tại: `2 / min(10, 5) = 2/5 = 0.4` ✅ ĐÚNG
- Công thức chuẩn: `2 / 5 = 0.4` ✅ CŨNG ĐÚNG

**Kết luận**: Công thức hiện tại **ĐÚNG** vì `min(k, len(gt_items))` chỉ có tác dụng khi `k < len(gt_items)`, nhưng trong trường hợp đó thì `min(k, len(gt_items)) = k`, và recall vẫn đúng.

Tuy nhiên, công thức chuẩn hơn là: `hits / len(gt_items)` (không cần min).

### 2. QwenReranker - KHÔNG có `_evaluate_split()` method

**Location**: `rerank/methods/qwen_reranker.py`

**Quan sát**:
- QwenReranker **KHÔNG có** `_evaluate_split()` method
- Có thể nó không có validation trong training, hoặc dùng evaluation từ bên ngoài

**Evaluation từ bên ngoài**:
- Sử dụng `evaluation/utils.py:evaluate_split()` với `recommend_fn = reranker.rerank`
- Hoặc sử dụng `evaluation/offline_eval.py` cho offline evaluation

### 3. Evaluation từ bên ngoài (evaluation/utils.py)

**Location**: `evaluation/utils.py:13-106`

**Quy trình**:
```python
def evaluate_split(recommend_fn, split: Dict[int, List[int]], k: int = 10, ...):
    for user_id in users:
        gt_items = split.get(user_id, [])
        
        # Get recommendations
        recs = recommend_fn(user_id)  # Gọi reranker.rerank(user_id, candidates)
        
        # Compute metrics
        r = recall_at_k(recs, gt_items, k_val)
        n = ndcg_at_k(recs, gt_items, k_val)
```

**Công thức Recall** (từ `evaluation/metrics.py:10-25`):
```python
def recall_at_k(recommended: List[int], ground_truth: Iterable[int], k: int) -> float:
    gt = set(ground_truth)
    rec_k = recommended[:k]
    hits = len(gt.intersection(rec_k))
    return hits / float(len(gt))  # ✅ Công thức chuẩn
```

**✅ Công thức này ĐÚNG**: `hits / len(gt)` - không có `min(k, len(gt))`

## ⚠️ Vấn đề phát hiện

### 1. Inconsistency trong công thức Recall

**Qwen3VLReranker._evaluate_split()**:
```python
recalls.append(hits / min(k, len(gt_items)))  # ⚠️ Có min()
```

**evaluation/metrics.py:recall_at_k()**:
```python
return hits / float(len(gt))  # ✅ Không có min()
```

**Phân tích**:
- Công thức có `min()` chỉ khác khi `k < len(gt_items)`
- Trong thực tế, thường `k >= len(gt_items)` (k=10, gt_items thường 1-3 items)
- Nhưng để consistency, nên dùng công thức chuẩn: `hits / len(gt_items)`

### 2. QwenReranker không có validation trong training

- QwenReranker không có `_evaluate_split()` method
- Không có validation sau mỗi epoch trong training
- Chỉ có evaluation sau khi training xong

## ✅ Điểm tốt

1. **History exclusion**: Qwen3VLReranker đã exclude history items khỏi candidate pool ✅
2. **GT guarantee**: Đảm bảo ít nhất 1 GT item có trong candidates ✅
3. **Shuffle candidates**: Shuffle để tránh bias ✅
4. **History truncation**: History được truncate xuống 5 items cuối cùng ✅

## 🔧 Đề xuất sửa

1. **Sửa công thức Recall trong Qwen3VLReranker**:
   ```python
   # Trước:
   recalls.append(hits / min(k, len(gt_items)))
   
   # Sau:
   recalls.append(hits / len(gt_items))  # Công thức chuẩn
   ```

2. **Thêm validation cho QwenReranker** (optional):
   - Thêm `_evaluate_split()` method tương tự Qwen3VLReranker
   - Hoặc giữ nguyên và chỉ dùng evaluation từ bên ngoài

## 📝 Tóm tắt

| Reranker | Có `_evaluate_split()`? | Công thức Recall | History Exclusion | GT Guarantee |
|----------|------------------------|------------------|-------------------|--------------|
| QwenReranker | ❌ Không | `hits / len(gt)` (từ metrics.py) | N/A | N/A |
| Qwen3VLReranker | ✅ Có | `hits / min(k, len(gt))` ⚠️ | ✅ Có | ✅ Có |

**Kết luận**: 
- Qwen3VLReranker có evaluation logic đầy đủ nhưng công thức recall hơi khác chuẩn
- QwenReranker không có validation trong training, chỉ dùng evaluation từ bên ngoài
- Cả hai đều sử dụng `evaluation/metrics.py:recall_at_k()` cho final evaluation (công thức chuẩn)

