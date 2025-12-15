# Phân tích Logic Rerank bằng LLM - Các vấn đề phát hiện và đã sửa

## ✅ ĐÃ SỬA: Giới hạn 20 candidates

### Vấn đề 1: Chỉ hỗ trợ tối đa 20 candidates

**Vị trí**: `rerank/models/llm.py`

```python
LETTERS = list(string.ascii_uppercase[:20])  # A-T (chỉ 20 chữ cái)
```

**Vấn đề**:
- `build_prompt_from_candidates()` sử dụng `LETTERS[i]` để label candidates
- Nếu có > 20 candidates → **IndexError**
- `predict_probs()` chỉ trả về 20 probabilities (cho A-T)
- Nhưng retrieval stage thường trả về 200 candidates (default `retrieval_top_k=200`)

**Code có vấn đề**:
```python
# rerank/models/llm.py:12-18
def build_prompt_from_candidates(user_history, candidate_ids, item_id2text):
    candidates = [item_id2text[cid] for cid in candidate_ids]
    cand_text = "\n".join(
        [f"{LETTERS[i]}. {c}" for i, c in enumerate(candidates)]  # ❌ Lỗi nếu len(candidates) > 20
    )
```

### Vấn đề 2: Mapping probs với candidates không đúng

**Vị trí**: `rerank/methods/qwen_reranker.py:117-120`

```python
for item_id in ranked_items[:self.top_k]:
    idx = candidates.index(item_id)
    score = float(probs[idx]) if idx < len(probs) else 0.0  # ❌ probs chỉ có 20 elements
    scored.append((item_id, score))
```

**Vấn đề**:
- `probs` chỉ có 20 elements (cho A-T)
- Nếu `candidates` có > 20 items, `idx` có thể >= 20 → **IndexError**
- Hoặc nếu < 20, sẽ có probs thừa không được dùng

### Vấn đề 3: Không có validation

**Vị trí**: `rerank/methods/qwen_reranker.py:rerank()`

- Không kiểm tra `len(candidates) <= 20`
- Không truncate candidates nếu > 20
- Không warning khi có quá nhiều candidates

## ⚠️ Vấn đề logic khác

### Vấn đề 4: `rank_candidates()` không đúng

**Vị trí**: `rerank/models/llm.py:36-42`

```python
def rank_candidates(probs, candidate_ids):
    ranked = sorted(
        zip(candidate_ids, probs),  # ❌ zip sẽ dừng ở min(len(candidate_ids), len(probs))
        key=lambda x: x[1],
        reverse=True
    )
    return [cid for cid, _ in ranked]
```

**Vấn đề**:
- Nếu `len(candidate_ids) > len(probs)`, một số candidates sẽ không có score
- Nếu `len(candidate_ids) < len(probs)`, một số probs sẽ bị bỏ qua

### Vấn đề 5: Training chỉ dùng 20 candidates

**Vị trí**: `scripts/train_rerank.py:35-38`

```python
# sample negatives
neg_items = random.sample(
    [i for i in all_items if i["item_new_id"] != pos_id],
    19  # ✅ Đúng: 1 positive + 19 negatives = 20 total
)
```

**Nhận xét**: Training đúng với 20 candidates, nhưng inference có thể nhận > 20.

## ✅ Điểm tốt

1. **Training logic hợp lý**: 
   - Sample 20 candidates (1 positive + 19 negatives)
   - Shuffle để tránh position bias
   - Format prompt nhất quán

2. **Prompt format tốt**:
   - Rõ ràng về task (recommendation ranking)
   - Có user history
   - Label candidates bằng chữ cái (A-T)

3. **Model setup hợp lý**:
   - Sử dụng LoRA để fine-tune
   - 4-bit quantization để tiết kiệm memory
   - Training arguments hợp lý

## ✅ ĐÃ SỬA: Giải pháp đã implement

### ✅ Sửa 1: Truncate candidates về 20 trong `qwen_reranker.py`

```python
# LLM chỉ hỗ trợ tối đa 20 candidates (A-T)
MAX_CANDIDATES = 20
original_count = len(candidates)
if original_count > MAX_CANDIDATES:
    import warnings
    warnings.warn(
        f"Truncating {original_count} candidates to {MAX_CANDIDATES} "
        f"(LLM reranker limit). Consider using fewer candidates from retrieval stage."
    )
    candidates = candidates[:MAX_CANDIDATES]
```

### ✅ Sửa 2: Validation trong `build_prompt_from_candidates()`

```python
MAX_CANDIDATES = len(LETTERS)  # 20

if len(candidate_ids) > MAX_CANDIDATES:
    raise ValueError(
        f"Too many candidates: {len(candidate_ids)} > {MAX_CANDIDATES}. "
        f"LLM reranker only supports up to {MAX_CANDIDATES} candidates (A-T)."
    )
```

### ✅ Sửa 3: Fix mapping probs với candidates

```python
# Tạo mapping từ item_id -> score để tránh index lookup
item_to_score = {item_id: float(probs[i]) for i, item_id in enumerate(candidates)}

scored = []
for item_id in ranked_items[:self.top_k]:
    score = item_to_score.get(item_id, 0.0)
    scored.append((item_id, score))
```

### ✅ Sửa 4: Validation trong `rank_candidates()`

```python
if len(probs) != len(candidate_ids):
    raise ValueError(
        f"Mismatch: {len(candidate_ids)} candidates but {len(probs)} probabilities. "
        f"Each candidate must have exactly one probability."
    )
```

### Giải pháp 2: Mở rộng LETTERS (nếu cần > 20)

```python
# Sử dụng A-Z (26 chữ cái) hoặc thêm số
LETTERS = list(string.ascii_uppercase)  # A-Z (26 chữ cái)
# Hoặc
LETTERS = list(string.ascii_uppercase) + [str(i) for i in range(10)]  # A-Z + 0-9
```

### Giải pháp 3: Batch processing cho > 20 candidates

```python
def rerank(self, user_id: int, candidates: List[int], **kwargs: Any) -> List[Tuple[int, float]]:
    if len(candidates) <= 20:
        # Process normally
        ...
    else:
        # Process in batches of 20
        all_scores = []
        for i in range(0, len(candidates), 20):
            batch = candidates[i:i+20]
            # Process batch
            ...
        # Merge và rerank lại
        ...
```

### Giải pháp 4: Sử dụng scoring approach thay vì classification

Thay vì predict 1 letter, predict score cho mỗi candidate:
- Input: prompt với candidates
- Output: scores cho từng candidate
- Không bị giới hạn bởi số lượng chữ cái

## 📊 Kết luận

**Logic hiện tại ĐÃ HỢP LÝ** sau khi sửa:
1. ✅ Handle được > 20 candidates (tự động truncate + warning)
2. ✅ Mapping probs với candidates đúng
3. ✅ Có validation và error handling đầy đủ
4. ✅ Document rõ ràng về giới hạn 20 candidates

## 💡 Khuyến nghị sử dụng

### Mối quan hệ với Retrieval Stage

**Quan trọng**: Số lượng candidates phụ thuộc vào `retrieval_top_k` từ Stage 1, và thông số này **có thể được điều chỉnh**.

**Các tùy chọn**:

1. **Tùy chọn 1: Điều chỉnh `retrieval_top_k <= 20`** (Khuyến nghị)
   ```python
   retrieval_cfg = RetrievalConfig(method="lrurec", top_k=20)  # ✅ Phù hợp với Qwen
   rerank_cfg = RerankConfig(method="qwen", top_k=10)
   ```
   - ✅ Không mất mát thông tin
   - ✅ Tất cả candidates đều được xem xét
   - ⚠️ Có thể giảm recall@20 của retrieval stage

2. **Tùy chọn 2: Giữ `retrieval_top_k > 20`** (Vẫn hoạt động)
   ```python
   retrieval_cfg = RetrievalConfig(method="lrurec", top_k=200)  # ⚠️ Sẽ truncate
   rerank_cfg = RerankConfig(method="qwen", top_k=10)
   ```
   - ✅ Giữ được recall@20 cao từ retrieval
   - ⚠️ Chỉ 20 candidates đầu tiên được rerank
   - ⚠️ Có warning khi truncate

3. **Tùy chọn 3: Dùng reranker khác cho > 20 candidates**
   ```python
   retrieval_cfg = RetrievalConfig(method="lrurec", top_k=200)
   rerank_cfg = RerankConfig(method="vip5", top_k=50)  # VIP5 không có giới hạn 20
   ```

### Best Practice

- **Khi dùng Qwen reranker**: Set `retrieval_top_k = 20` để tối ưu
- **Khi dùng VIP5/Identity reranker**: Có thể dùng `retrieval_top_k = 200` hoặc lớn hơn
- **Pipeline tự động**: Sẽ truncate nếu cần, nhưng tốt hơn là điều chỉnh config từ đầu

