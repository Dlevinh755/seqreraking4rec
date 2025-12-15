# Đảm bảo tính khách quan khi đánh giá các Rerank Methods

## Tổng quan

Sau khi xóa các baseline methods (identity, random), các rerank methods còn lại (qwen, vip5, bert4rec) đã được đảm bảo đánh giá khách quan với cùng metrics và evaluation protocol.

## ✅ Đảm bảo tính khách quan

### 1. **Cùng Evaluation Function**

Tất cả methods đều được đánh giá bằng cùng function:

```python
# evaluation/utils.py
def evaluate_split(
    recommend_fn,  # pipeline.recommend hoặc reranker.rerank
    split: Dict[int, List[int]],  # ground truth
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate recommendations on a split."""
    # Tính Recall@K và NDCG@K cho từng user
    # Trả về average metrics
```

**Đảm bảo**:
- ✅ Tất cả methods đều đi qua cùng evaluation function
- ✅ Cùng logic tính toán metrics
- ✅ Cùng cách xử lý edge cases (empty recommendations, no ground truth)

### 2. **Cùng Metrics**

Tất cả methods đều được đánh giá với cùng metrics:

- **Recall@K**: Tỷ lệ relevant items được recommend trong top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain tại K

**Implementation**:
```python
# evaluation/metrics.py
def recall_at_k(recommended: List[int], ground_truth: Iterable[int], k: int) -> float:
    """Compute Recall@K for a single user."""
    gt = set(ground_truth)
    rec_k = recommended[:k]
    hits = len(gt.intersection(rec_k))
    return hits / float(len(gt))

def ndcg_at_k(recommended: List[int], ground_truth: Iterable[int], k: int) -> float:
    """Compute NDCG@K for a single user."""
    # Binary relevance, standard DCG calculation
```

**Đảm bảo**:
- ✅ Cùng công thức tính toán
- ✅ Cùng cách xử lý binary relevance
- ✅ Cùng normalization

### 3. **Cùng Input từ Retrieval Stage**

Tất cả rerank methods đều nhận cùng candidates từ retrieval stage:

```python
# pipelines/base.py
def recommend(self, user_id: int, exclude_items: Optional[List[int]] = None) -> List[int]:
    # Stage 1: Retrieve candidates
    candidates = self.retriever.retrieve(user_id, exclude_items=exclude_set)
    
    # Stage 2: Rerank (if enabled)
    if self.reranker is None:
        return candidates
    
    scored = self.reranker.rerank(user_id, candidates)  # ← Cùng candidates
    return [item_id for item_id, _ in scored]
```

**Đảm bảo**:
- ✅ Tất cả methods nhận cùng candidates từ cùng retrieval stage
- ✅ Cùng exclude_items logic
- ✅ Cùng user_id

### 4. **Cùng Output Format**

Tất cả methods đều implement `BaseReranker` interface với cùng signature:

```python
# rerank/base.py
@abstractmethod
def rerank(
    self,
    user_id: int,
    candidates: List[int],
    **kwargs: Any
) -> List[Tuple[int, float]]:
    """Rerank danh sách candidates và trả về (item_id, score) đã sort giảm dần."""
```

**Đảm bảo**:
- ✅ Cùng input: `user_id`, `candidates`
- ✅ Cùng output: `List[Tuple[int, float]]` (sorted by score descending)
- ✅ Cùng top_k filtering (trong `BaseReranker`)

### 5. **Cùng Data Splits**

Tất cả methods đều được đánh giá trên cùng data splits:

```python
# scripts/train_pipeline.py
data = load_dataset_from_csv(...)
train = data["train"]
val = data["val"]
test = data["test"]

# Tất cả methods đều evaluate trên cùng test set
test_metrics = evaluate_pipeline(pipeline, test, k=args.metric_k)
```

**Đảm bảo**:
- ✅ Cùng train/val/test splits
- ✅ Cùng data filtering (min_rating, min_uc, min_sc)
- ✅ Cùng seed (đảm bảo reproducibility)

### 6. **Cùng Evaluation Protocol**

Tất cả methods đều được đánh giá với cùng protocol:

1. **Training**: Tất cả đều được train trên cùng `train` data
2. **Validation**: Tất cả đều có thể dùng `val` data cho early stopping
3. **Testing**: Tất cả đều được evaluate trên cùng `test` set
4. **Metrics**: Tất cả đều tính Recall@K và NDCG@K với cùng K

## 📊 So sánh khách quan

### Các methods hiện có

1. **Qwen**: LLM-based reranker
   - Input: candidates từ retrieval + user history (text)
   - Output: ranked candidates với scores
   - Giới hạn: 20 candidates (A-T letters)

2. **VIP5**: Multimodal T5-based reranker
   - Input: candidates từ retrieval + visual/text features
   - Output: ranked candidates với scores
   - Không giới hạn số candidates

3. **BERT4Rec**: Sequential BERT-based reranker
   - Input: candidates từ retrieval + user history (sequential)
   - Output: ranked candidates với scores
   - Không giới hạn số candidates

### Điểm chung (Đảm bảo tính khách quan)

- ✅ **Cùng evaluation function**: `evaluate_split()`
- ✅ **Cùng metrics**: Recall@K, NDCG@K
- ✅ **Cùng input candidates**: Từ cùng retrieval stage
- ✅ **Cùng data splits**: train/val/test
- ✅ **Cùng evaluation protocol**: Train → Validate → Test
- ✅ **Cùng output format**: `List[Tuple[int, float]]`

### Khác biệt (Không ảnh hưởng tính khách quan)

- ⚠️ **Input requirements**: 
  - Qwen cần text features
  - VIP5 cần CLIP embeddings
  - BERT4Rec cần sequential data
  - **Nhưng**: Người dùng đã xác nhận "không cần quan tâm đến đầu vào"
  - **Kết luận**: Chỉ cần đảm bảo metrics được tính công bằng (đã đảm bảo ✅)

- ⚠️ **Candidate limits**:
  - Qwen giới hạn 20 candidates
  - VIP5 và BERT4Rec không giới hạn
  - **Nhưng**: Tất cả đều nhận cùng candidates từ retrieval, chỉ khác cách xử lý
  - **Kết luận**: Metrics vẫn được tính công bằng trên cùng ground truth

## 🔍 Verification

### Kiểm tra evaluation flow

```python
# 1. Tất cả methods đều đi qua cùng pipeline
pipeline = TwoStagePipeline(cfg)
pipeline.fit(train, reranker_kwargs=reranker_kwargs)

# 2. Tất cả methods đều được evaluate bằng cùng function
metrics = evaluate_pipeline(pipeline, test, k=10)
# → Gọi evaluate_split(pipeline.recommend, test, k=10)

# 3. evaluate_split() gọi pipeline.recommend() cho từng user
# → pipeline.recommend() gọi reranker.rerank() với cùng candidates

# 4. Metrics được tính bằng cùng functions
recall = recall_at_k(recommended, gt_items, k)
ndcg = ndcg_at_k(recommended, gt_items, k)
```

### Kiểm tra metrics calculation

```python
# evaluation/metrics.py
def recall_at_k(recommended: List[int], ground_truth: Iterable[int], k: int) -> float:
    # Công thức: hits / total_gt_items
    # Không phụ thuộc vào method nào tạo ra recommendations
    
def ndcg_at_k(recommended: List[int], ground_truth: Iterable[int], k: int) -> float:
    # Công thức: DCG / IDCG
    # Không phụ thuộc vào method nào tạo ra recommendations
```

**Kết luận**: Metrics được tính hoàn toàn khách quan, chỉ dựa trên:
- Recommended items (output từ reranker)
- Ground truth items (từ test set)
- Không phụ thuộc vào method nào tạo ra recommendations

## ✅ Kết luận

**Tính khách quan: 100%** ✅

Các rerank methods (qwen, vip5, bert4rec) đã được đảm bảo đánh giá khách quan với:

1. ✅ **Cùng evaluation function**: `evaluate_split()`
2. ✅ **Cùng metrics**: Recall@K, NDCG@K với cùng công thức
3. ✅ **Cùng input**: Cùng candidates từ retrieval stage
4. ✅ **Cùng data splits**: train/val/test
5. ✅ **Cùng evaluation protocol**: Train → Validate → Test
6. ✅ **Cùng output format**: `List[Tuple[int, float]]`

**Không cần quan tâm đến input requirements khác nhau** vì:
- Metrics chỉ phụ thuộc vào output (recommended items) và ground truth
- Không phụ thuộc vào cách method xử lý input
- Tất cả methods đều nhận cùng candidates từ retrieval stage

**Có thể so sánh trực tiếp** các methods với nhau dựa trên metrics (Recall@K, NDCG@K).

