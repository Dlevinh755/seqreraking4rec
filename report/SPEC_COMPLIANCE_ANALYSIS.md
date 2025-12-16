# Phân tích Tuân thủ Specification

## Tổng quan

Tài liệu này phân tích mức độ tuân thủ của codebase với specification được cung cấp.

---

## ✅ 1. DATA FILTERING & PREPROCESSING

### 1.1 Interaction Filtering

**Spec yêu cầu:**
- Keep only users with ≥ 5 interactions
- Keep only items with ≥ 5 interactions  
- Remove all interactions with rating < 3
- Keep only items with valid text metadata AND valid image
- **⚠️ Filtering must be applied globally before splitting**

**Code hiện tại:**
- ✅ `min_uc` và `min_sc` được áp dụng (default: 5)
- ✅ `min_rating` được lưu trong config (default: 3)
- ⚠️ **VẤN ĐỀ**: Cần kiểm tra xem `min_rating` có được áp dụng TRƯỚC khi split không
- ✅ Text filtering được áp dụng trước khi filter triplets
- ✅ Image filtering được áp dụng sau khi filter triplets

**File**: `dataset/base.py:137-159`, `dataset/beauty.py:66-90`

**Cần sửa:**
```python
# Trong preprocess(), cần đảm bảo:
df = df[df['rating'] >= self.min_rating]  # ✅ Áp dụng TRƯỚC filter_triplets
df = self.filter_triplets(df)  # Sau đó mới filter min_uc/min_sc
```

### 1.2 Text Construction

**Spec yêu cầu:**
- Concatenate title + description
- Normalize: lowercase, remove special characters
- Truncate to max_text_length (configurable, 256-512 tokens)
- Truncation from the end

**Code hiện tại:**
- ⚠️ **CẦN KIỂM TRA**: Text construction có normalize và truncate đúng không
- ⚠️ **CẦN KIỂM TRA**: `max_text_length` có configurable không

**Cần sửa:**
- Thêm text normalization (lowercase, remove special chars)
- Thêm text truncation với configurable `max_text_length`
- Đảm bảo truncation từ cuối (end)

### 1.3 Image Processing

**Spec yêu cầu:**
- Download image (if remote)
- Resize to 224×224
- Save as `images/{item_id}.jpg`

**Code hiện tại:**
- ✅ Images được download và save
- ⚠️ **CẦN KIỂM TRA**: Resize có đúng 224×224 không
- ⚠️ **CẦN KIỂM TRA**: Path có đúng `images/{item_id}.jpg` không

---

## ✅ 2. MULTIMODAL FEATURE EXTRACTION

### 2.1 CLIP Embeddings

**Spec yêu cầu:**
- Extract `clip_image_embedding` và `clip_text_embedding`
- Batch inference on GPU
- Store indexed by item_id

**Code hiện tại:**
- ✅ CLIP embeddings được extract (`dataset/clip_embeddings.py`)
- ✅ Batch inference với `BATCH_SIZE = 128`
- ✅ Store trong `clip_embeddings.pt` với format `[num_items+1, D]`

**Status**: ✅ **TUÂN THỦ**

### 2.2 Image Caption Generation (BLIP)

**Spec yêu cầu:**
- Use BLIP to generate captions
- Batch inference on GPU
- Store `item_caption[item_id]`

**Code hiện tại:**
- ✅ BLIP captions được generate (`dataset/blip2_captions.py`)
- ✅ Batch inference
- ✅ Store trong CSV metadata

**Status**: ✅ **TUÂN THỦ**

### 2.3 Image Semantic Summarization (MLLM)

**Spec yêu cầu:**
- Use unsloth + Qwen3-2B-VL
- Load in 4-bit
- Input: image only (no future interaction data)
- Output: semantic summary (text)
- Batch inference
- Store `item_semantic_summary[item_id]`

**Code hiện tại:**
- ✅ Qwen3-VL được sử dụng
- ✅ 4-bit quantization
- ✅ Batch inference
- ✅ Store trong CSV metadata

**Status**: ✅ **TUÂN THỦ**

---

## ✅ 3. USER SEQUENCE CONSTRUCTION & SPLIT

### 3.1 Sequence Construction

**Spec yêu cầu:**
- Group by user_id
- Sort by timestamp ascending

**Code hiện tại:**
```python
# dataset/base.py:172-173
user2items = user_group.progress_apply(
    lambda d: list(d.sort_values(by=['timestamp', 'sid'])['sid']),
)
```
- ✅ Sort by timestamp ascending

**Status**: ✅ **TUÂN THỦ**

### 3.2 Leave-One-Out Split (STRICT)

**Spec yêu cầu:**
- Last interaction → TEST
- Second last → VALIDATION
- All previous → TRAINING HISTORY
- **⚠️ Split determined ONLY by timestamp, not by model output**

**Code hiện tại:**
```python
# dataset/base.py:180
train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
```
- ✅ Last → test (`items[-1:]`)
- ✅ Second last → val (`items[-2:-1]`)
- ✅ All previous → train (`items[:-2]`)
- ✅ Split chỉ dựa trên timestamp (không có model output)

**Status**: ✅ **TUÂN THỦ**

---

## ✅ 4. RETRIEVAL STAGE (STAGE-1)

### 4.1 Models

**Spec yêu cầu:**
- MMGCN, VBPR, LRURec, BM3

**Code hiện tại:**
- ✅ Tất cả 4 models đều được implement

**Status**: ✅ **TUÂN THỦ**

### 4.2 Negative Sampling (CRITICAL)

**Spec yêu cầu:**
- Sampling ratio: 1 positive : 1 negative
- Negative items must:
  - Never appear in user's interaction history
  - Exclude validation and test items
- Sampling per epoch per user

**Code hiện tại:**

**MMGCN** (`retrieval/methods/mmgcn.py`):
```python
# Line 150-160: Negative sampling
neg_candidates = list(all_items - set(items))  # ✅ Exclude history
neg_item = np.random.choice(neg_candidates)
```
- ✅ 1:1 ratio
- ✅ Exclude history
- ⚠️ **CẦN KIỂM TRA**: Có exclude val/test items không?

**VBPR** (`retrieval/methods/vbpr.py:317`):
```python
neg_candidates = list(all_items - set(items))  # ✅ Exclude history
neg_item = np.random.choice(neg_candidates)
```
- ✅ 1:1 ratio
- ✅ Exclude history
- ⚠️ **CẦN KIỂM TRA**: Có exclude val/test items không?

**BM3** (`retrieval/methods/bm3.py:273`):
```python
neg_candidates = list(all_items - set(items))  # ✅ Exclude history
neg_item = np.random.choice(neg_candidates)
```
- ✅ 1:1 ratio
- ✅ Exclude history
- ⚠️ **CẦN KIỂM TRA**: Có exclude val/test items không?

**Cần sửa:**
```python
# Đảm bảo negatives exclude cả val và test items
all_items = set(range(1, self.num_item + 1))
user_history = set(items)  # train items
val_items = set(val_data.get(user_id, []))
test_items = set(test_data.get(user_id, []))
neg_candidates = list(all_items - user_history - val_items - test_items)
```

### 4.3 Training Procedure

**Spec yêu cầu:**
- Mini-batch training
- Validate after each epoch
- Early stopping if Recall@K does not improve for patience epochs

**Code hiện tại:**
- ✅ Mini-batch training
- ✅ Validate after each epoch
- ✅ Early stopping với patience

**Status**: ✅ **TUÂN THỦ**

### 4.4 Retrieval Evaluation (FULL RANKING)

**Spec yêu cầu:**
- For each user:
  - Compute scores against ALL items
  - Mask: all training history items
  - **Ground truth item must NOT be masked**
- Evaluation: Recall@{5,10,20}, NDCG@{5,10,20}
- Batch computation on GPU REQUIRED

**Code hiện tại:**

**MMGCN** (`retrieval/methods/mmgcn.py:248-317`):
```python
# Compute scores for all items
scores_batch = torch.matmul(batch_user_emb, item_tensor.t())  # ✅ ALL items

# Mask history items
for item in history_items:
    scores_batch[j, item_idx] = -1e9  # ✅ Mask history

# Get top-K
_, top_items_batch = torch.topk(scores_batch, k=k, dim=1)  # ✅ Batch GPU
```
- ✅ Compute against ALL items
- ✅ Mask history items
- ✅ Batch computation on GPU
- ⚠️ **CẦN XÁC NHẬN**: GT item có bị mask không? (Spec yêu cầu KHÔNG mask GT)

**VBPR** (`retrieval/methods/vbpr.py:326-408`):
- ✅ Tương tự MMGCN

**BM3** (`retrieval/methods/bm3.py:282-350`):
- ✅ Tương tự MMGCN

**Cần kiểm tra:**
- Đảm bảo GT items KHÔNG bị mask trong evaluation
- Hiện tại code mask `history_items`, nhưng cần đảm bảo GT không nằm trong history

### 4.5 Candidate Saving (OPTIONAL)

**Spec yêu cầu:**
- If enabled: Save top-K retrieved items for val/test
- K ∈ {20, 50} (configurable)
- Candidate lists MUST be fixed and reused by all reranking methods

**Code hiện tại:**
- ✅ `_build_retrieved_matrices` saves top-K candidates
- ✅ `RETRIEVAL_SAVE_TOP_K = 20` (configurable)
- ✅ Saved to CSV và reused by rerankers

**Status**: ✅ **TUÂN THỦ**

---

## ✅ 5. RERANKING STAGE (STAGE-2)

### 5.1 Reranking Methods

**Spec yêu cầu:**
- VIP5, BERT4Rec, QwenRec (LLM), Qwen-VL-Rec (MLLM)
- All methods must use the same candidate lists

**Code hiện tại:**
- ✅ Tất cả 4 methods đều được implement
- ✅ Sử dụng cùng candidate lists từ retrieval stage

**Status**: ✅ **TUÂN THỦ**

### 5.2 Evaluation Modes

**Spec yêu cầu:**
- **Mode A: Full Pipeline**: Rerank candidates from retrieval stage
- **Mode B: Ground-Truth-Augmented**:
  - Ensure ground truth item is included
  - Sample additional negatives (items user has never interacted with)
  - Number of negatives is configurable
  - **⚠️ Items not interacted with are NEGATIVE, not positive**

**Code hiện tại:**
- ✅ Mode A: `rerank_mode="retrieval"` - sử dụng candidates từ retrieval
- ✅ Mode B: `rerank_mode="ground_truth"` - GT + negatives
- ✅ Negatives exclude history items
- ✅ Number of negatives configurable (`rerank_eval_candidates`)

**Status**: ✅ **TUÂN THỦ**

---

## ✅ 6. QWEN-BASED RERANKING (CORE METHOD)

### 6.1 Prompt Template (STRICT)

**Spec yêu cầu:**
```
Choose exactly one item from [1, 2, 3, ...].

User history:
- Item description 1
- Item description 2
- ...

Candidates:
1: description
2: description
3: description

Answer:
```

**Code hiện tại:**

**Qwen3VLReranker** (`rerank/methods/qwen3vl_reranker.py:1113-1124`):
```python
prompt = f"""You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
{history_str}

Candidate items:
{cand_str}

Answer with only one number (1-{num_candidates}).
""".strip()
```
- ✅ "Choose exactly ONE item"
- ✅ User history format
- ✅ Candidates format (1: description, 2: description, ...)
- ✅ Answer format

**Status**: ✅ **TUÂN THỦ** (gần đúng, chỉ khác wording nhỏ)

### 6.2 Training Target & Inference

**Spec yêu cầu:**
- Output MUST be exactly one token
- Candidate labels: single-token (numbers ≤ 9) or special tokens
- During inference:
  - Extract logits of next token
  - Select logits corresponding to candidate tokens
  - Apply softmax
  - Use probabilities to rerank items

**Code hiện tại:**
- ✅ Output exactly one token (number)
- ✅ Candidate labels: numbers (1, 2, 3, ...)
- ✅ Extract logits, apply softmax, rerank

**Status**: ✅ **TUÂN THỦ**

### 6.3 Training Modes

**Spec yêu cầu:**
1. Only Text
2. Text + Raw Image
3. Caption + Text
4. Semantic Summary + Text
5. Semantic Summary + Text (Small Model) - Qwen3-0.6B

**Code hiện tại:**
- ✅ Mode 1: Text only (`qwen3vl_mode` không có, nhưng có thể dùng QwenReranker)
- ✅ Mode 2: `qwen3vl_mode="raw_image"` - Text + Raw Image
- ✅ Mode 3: `qwen3vl_mode="caption"` - Caption + Text
- ✅ Mode 4: `qwen3vl_mode="semantic_summary"` - Semantic Summary + Text
- ✅ Mode 5: `qwen3vl_mode="semantic_summary_small"` - Qwen3-0.6B

**Status**: ✅ **TUÂN THỦ**

---

## ✅ 7. RERANK TRAINING & EVALUATION

**Spec yêu cầu:**
- Validate after each epoch
- Early stopping on validation Recall@K
- Load best checkpoint
- Evaluate on test set
- Metrics: Recall@{5,10,20}, NDCG@{5,10,20}

**Code hiện tại:**
- ✅ Validate after each epoch
- ✅ Early stopping
- ✅ Load best checkpoint
- ✅ Evaluate on test set
- ✅ Metrics: Recall@{5,10,20}, NDCG@{5,10,20}

**Status**: ✅ **TUÂN THỦ**

---

## ✅ 8. REPRODUCIBILITY & LOGGING

**Spec yêu cầu:**
- Fix and log random seed
- Save: configs, checkpoints, candidate lists, evaluation results
- No silent default values

**Code hiện tại:**
- ✅ Random seed fixed (`arg.seed = 42`)
- ✅ Configs saved
- ✅ Checkpoints saved
- ✅ Candidate lists saved (CSV)
- ✅ Evaluation results saved (JSON)
- ⚠️ **CẦN KIỂM TRA**: Tất cả default values có được log không?

**Status**: ⚠️ **CẦN KIỂM TRA**

---

## 🔴 TÓM TẮT CÁC VẤN ĐỀ CẦN SỬA

### Critical (Phải sửa ngay):

1. **Negative Sampling**: Đảm bảo negatives exclude cả val và test items
   - File: `retrieval/methods/mmgcn.py`, `retrieval/methods/vbpr.py`, `retrieval/methods/bm3.py`
   - Fix: Thêm val/test items vào exclusion set

2. **Rating Filtering**: Đảm bảo `min_rating` được áp dụng TRƯỚC khi split
   - File: `dataset/base.py`, `dataset/beauty.py`, etc.
   - Fix: Filter `df[df['rating'] >= min_rating]` trước `filter_triplets()`

3. **Text Normalization & Truncation**: Thêm text normalization và truncation
   - File: `dataset/base.py` hoặc dataset-specific files
   - Fix: Thêm normalize (lowercase, remove special chars) và truncate với configurable `max_text_length`

### Important (Nên sửa):

4. **Image Resize**: Đảm bảo resize đúng 224×224
   - File: Image download/preprocessing code
   - Fix: Kiểm tra và sửa resize logic

5. **Default Values Logging**: Log tất cả default values
   - File: `config.py`, training scripts
   - Fix: Print/log tất cả config values khi start training

6. **GT Item Masking**: Đảm bảo GT items KHÔNG bị mask trong evaluation
   - File: `retrieval/methods/*.py` evaluation code
   - Fix: Kiểm tra logic masking, đảm bảo GT không nằm trong history

---

## ✅ ĐIỂM MẠNH

1. ✅ Leave-one-out split được implement đúng
2. ✅ CLIP, BLIP, Semantic Summary extraction đầy đủ
3. ✅ Full ranking evaluation với batch GPU computation
4. ✅ Qwen prompt template gần đúng spec
5. ✅ Tất cả training modes được support
6. ✅ Candidate lists được save và reuse
7. ✅ History masking trong rerankers đã được fix

---

## 📋 CHECKLIST SỬA LỖI

- [ ] Fix negative sampling để exclude val/test items
- [ ] Fix rating filtering để áp dụng trước split
- [ ] Thêm text normalization (lowercase, remove special chars)
- [ ] Thêm text truncation với configurable `max_text_length`
- [ ] Kiểm tra image resize (224×224)
- [ ] Log tất cả default values
- [ ] Kiểm tra GT item masking trong evaluation
- [ ] Verify prompt template matches spec exactly

---

## 📝 NOTES

- Codebase đã tuân thủ phần lớn specification
- Các vấn đề chính là về negative sampling và data filtering order
- Cần test kỹ sau khi sửa để đảm bảo không break existing functionality

