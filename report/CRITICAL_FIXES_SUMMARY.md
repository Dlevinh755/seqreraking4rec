# Tóm tắt các Critical Fixes đã thực hiện

## ✅ Đã hoàn thành

### 1. Negative Sampling - Exclude Val và Test Items

**Vấn đề**: Negative sampling chỉ exclude history items, không exclude val và test items như spec yêu cầu.

**Đã sửa**:
- ✅ `retrieval/methods/mmgcn.py`: Thêm logic exclude val và test items trong negative sampling
- ✅ `retrieval/methods/vbpr.py`: Cập nhật `_prepare_training_samples()` để nhận `val_data` và `test_data`, exclude chúng khỏi negative candidates
- ✅ `retrieval/methods/bm3.py`: Cập nhật `_prepare_training_samples()` để nhận `val_data` và `test_data`, exclude chúng khỏi negative candidates
- ✅ `scripts/train_retrieval.py`: Thêm `test_data` vào `fit_kwargs` để truyền vào các retrieval methods

**Code changes**:
```python
# Trước:
neg_candidates = list(all_items - set(items))  # Chỉ exclude history

# Sau:
user_train_items = set(items)  # Training history
user_val_items = set(val_data.get(user_id, []))
user_test_items = set(test_data.get(user_id, []))
excluded_items = user_train_items | user_val_items | user_test_items
neg_candidates = list(all_items - excluded_items)  # Exclude cả val và test
```

---

### 2. Rating Filtering - Áp dụng TRƯỚC khi Split

**Vấn đề**: `min_rating` có trong config nhưng không được áp dụng trước khi split như spec yêu cầu.

**Đã sửa**:
- ✅ `dataset/beauty.py`: Thêm rating filter TRƯỚC `filter_triplets()`
- ✅ `dataset/games.py`: Thêm rating filter TRƯỚC `filter_triplets()`
- ✅ `dataset/ml_100k.py`: Thêm rating filter TRƯỚC `filter_triplets()`

**Code changes**:
```python
# Thêm vào đầu preprocess(), sau load_ratings_df():
# ✅ CRITICAL FIX: Filter by min_rating FIRST (before any other filtering)
if self.min_rating > 0:
    initial_count = len(df)
    df = df[df['rating'] >= self.min_rating]
    print(f'Ratings after min_rating filter (rating >= {self.min_rating}): {len(df)}/{initial_count}')

# Sau đó mới filter text và triplets
```

---

### 3. Text Normalization & Truncation

**Vấn đề**: Text metadata không được normalize (lowercase, remove special chars) và truncate theo spec.

**Đã sửa**:
- ✅ `dataset/utils.py`: Thêm 3 helper functions:
  - `normalize_text()`: Lowercase + remove special characters
  - `truncate_text()`: Truncate từ cuối (end)
  - `process_item_text()`: Combine tất cả (concatenate, normalize, truncate)
- ✅ `config.py`: Thêm `--max_text_length` argument (default: 512, range: 256-512)
- ✅ `dataset/beauty.py`: Áp dụng `process_item_text()` khi load metadata
- ✅ `dataset/games.py`: Áp dụng `process_item_text()` khi load metadata
- ✅ `dataset/ml_100k.py`: Áp dụng `process_item_text()` khi load metadata

**Code changes**:
```python
# Trước:
text = f"{title} {description}".strip()

# Sau:
from dataset.utils import process_item_text
max_text_length = getattr(self.args, 'max_text_length', 512)
text = process_item_text(title, description, max_length=max_text_length)
```

**Text processing pipeline**:
1. Concatenate `title + description`
2. Normalize: lowercase + remove special chars (keep only alphanumeric + spaces)
3. Truncate from end to `max_text_length` characters

---

## 📋 Checklist

- [x] Fix negative sampling để exclude val và test items (MMGCN, VBPR, BM3)
- [x] Fix rating filtering để áp dụng trước split (beauty, games, ml_100k)
- [x] Thêm text normalization (lowercase, remove special chars)
- [x] Thêm text truncation với configurable `max_text_length`
- [x] Đảm bảo truncation từ cuối (end)
- [x] Thêm `test_data` vào `fit_kwargs` trong `train_retrieval.py`

---

## 🔍 Cần kiểm tra thêm (Important, không critical)

1. **Image Resize**: Đảm bảo resize đúng 224×224
2. **Default Values Logging**: Log tất cả config values khi start training
3. **GT Item Masking**: Đảm bảo GT items KHÔNG bị mask trong evaluation (hiện tại đã đúng vì GT không nằm trong history)

---

## 📝 Notes

- Tất cả các critical fixes đã được implement
- Code đã pass linter checks
- Cần test lại để đảm bảo không break existing functionality
- Text normalization có thể ảnh hưởng đến CLIP embeddings (nếu embeddings đã được extract trước khi normalize), cần re-extract CLIP embeddings sau khi normalize text

---

## ⚠️ Breaking Changes

1. **Text Normalization**: Text metadata sẽ được normalize, có thể ảnh hưởng đến:
   - CLIP text embeddings (cần re-extract nếu đã extract trước)
   - Semantic summaries (nếu đã generate trước)
   - Captions (nếu đã generate trước)

2. **Rating Filtering**: Rating filtering được áp dụng sớm hơn, có thể thay đổi số lượng users/items trong dataset.

3. **Negative Sampling**: Negative sampling strict hơn (exclude cả val/test), có thể ảnh hưởng đến training dynamics.

**Khuyến nghị**: Re-run `data_prepare.py` để regenerate dataset với các fixes mới.

