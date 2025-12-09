# Data Preprocessing Pipeline - Complete Guide

## 📚 MỤC LỤC

1. [Tổng quan](#tổng-quan)
2. [Cài đặt](#cài-đặt)
3. [Sử dụng cơ bản](#sử-dụng-cơ-bản)
4. [Tính năng lọc dữ liệu](#tính-năng-lọc-dữ-liệu)
5. [Download images](#download-images)
6. [Quy trình preprocessing](#quy-trình-preprocessing)
7. [Cấu trúc dữ liệu](#cấu-trúc-dữ-liệu)
8. [Utilities](#utilities)

---

## 🎯 TỔNG QUAN

Pipeline preprocessing cho recommendation system với khả năng:
- ✅ Lọc items theo text và image availability
- ✅ Download và verify images tự động
- ✅ Parallel processing (20 threads) để tăng tốc
- ✅ Tối ưu thứ tự lọc để tiết kiệm thời gian

### Datasets hỗ trợ:
- **Beauty**: Amazon Beauty products (text + image)
- **Games**: Video Games (text + image)
- **ML-100k**: MovieLens (chỉ text)

---

## 🔧 CÀI ĐẶT

### Yêu cầu:
```bash
python >= 3.8
pytorch-lightning
pandas
numpy
tqdm
Pillow
pyyaml
```

### Cài đặt dependencies:
```bash
pip install torch pytorch-lightning pandas numpy tqdm Pillow pyyaml
```

---

## 🚀 SỬ DỤNG CƠ BẢN

### 1. Preprocessing đơn giản (không lọc):
```bash
python data_prepare.py
```

### 2. Với filtering text:
```bash
python data_prepare.py --use_text
```

### 3. Với filtering text + download images:
```bash
python data_prepare.py --use_text --use_image
```

### 4. Tùy chỉnh dataset:
```bash
python data_prepare.py --dataset_code games --min_uc 10 --min_sc 10
```

### Các arguments:
- `--dataset_code`: beauty, games, ml-100k (default: beauty)
- `--min_rating`: Minimum rating để giữ lại (default: 3)
- `--min_uc`: Minimum ratings per user (default: 5)
- `--min_sc`: Minimum ratings per item (default: 5)
- `--use_text`: Lọc items không có text
- `--use_image`: Lọc items không có image + download images
- `--seed`: Random seed (default: 42)

---

## 🎨 TÍNH NĂNG LỌC DỮ LIỆU

### Metadata Structure:

```python
{
    'text': 'Title and description combined',  # None nếu không có
    'image': 'http://image-url.com/...',       # None nếu không có
    'image_path': 'data/.../images/item_1.jpg', # Local path (sau khi download)
    'title': 'Product title'
}
```

### Filtering Logic:

#### 1. Text Filtering (`--use_text`)
- Giữ items có text (title + description) hợp lệ
- Loại bỏ items có text null hoặc rỗng

#### 2. Image Filtering (`--use_image`)
- Verify URL accessible
- Download image và verify format (PIL)
- Chỉ giữ items download thành công
- **CHÚ Ý**: MovieLens không có images!

### Kết quả Filtering (Beauty dataset):

| Bước | Số items | Ghi chú |
|------|----------|---------|
| Total metadata | 259,204 | |
| After text filter | 258,992 | -212 items |
| After triplet filter | 12,101 | **-246,891 items!** |
| After image download | 11,800 | -300 items failed |

---

## 📥 DOWNLOAD IMAGES

### Tại sao nên download images?

✅ **Tốc độ training**: Đọc local disk >> download internet  
✅ **Ổn định**: Không phụ thuộc vào internet hoặc URL  
✅ **Reproducibility**: Đảm bảo cùng dữ liệu mỗi lần  
✅ **Tiết kiệm bandwidth**: Chỉ download 1 lần  

### Cách download được thực hiện:

#### VỪA CHECK VỪA DOWNLOAD (1 request/image):
```python
download_and_verify_images_batch()
├─ Download image từ URL
├─ Verify bằng PIL (format, integrity)
├─ Lưu vào local disk
└─ Return valid items
```

#### Parallel Processing:
- **20 threads** chạy đồng thời
- ~17 images/giây (tùy network)
- 12,000 images → **~12 phút**

#### Đặt tên file:
```
item_{item_id}_{url_hash}.{ext}
Ví dụ: item_1_aeb6393c.jpg
```

### Cấu trúc lưu trữ:

```
data/
└── preprocessed/
    └── beauty_min_rating3-min_uc5-min_sc5/
        ├── dataset.pkl              # Metadata + mappings
        └── images/                  # Downloaded images
            ├── item_1_aeb6393c.jpg
            ├── item_2_9179de12.jpg
            └── ... (11,800 files)
```

---

## ⚡ QUY TRÌNH PREPROCESSING (TỐI ƯU)

### Thứ tự các bước:

```
1. Load metadata (~259k items)
   ↓
2. Lọc TEXT (nhanh - < 1s)
   ↓ ~258k items
3. Lọc TRIPLETS (min_uc, min_sc)
   ↓ ~12k items ⭐ GIẢM 95%!
4. Densify Index (tạo mapping mới)
   ↓
5. DOWNLOAD IMAGES (chỉ 12k thay vì 259k!)
   ↓ ~11.8k items
6. Lọc lại nếu download failed
   ↓
7. Split train/val/test & Lưu dataset
```

### Tại sao thứ tự này tối ưu?

❌ **Cách CŨ** (KHÔNG tối ưu):
```
Download 259k images → Lọc triplets → Còn 12k
→ ĐÃ DOWNLOAD THỪA 247k IMAGES! (lãng phí 95%)
Time: 5-10 giờ
```

✅ **Cách MỚI** (Tối ưu):
```
Lọc triplets → Còn 12k → Download 12k images
→ TIẾT KIỆM 95% THỜI GIAN!
Time: 10-20 phút
```

### Chi tiết từng bước:

#### Bước 1: Load Metadata
```python
meta_raw = self.load_meta_dict()
# → 259,204 items với text + image URL
```

#### Bước 2: Lọc Text (nhanh)
```python
if self.args.use_text:
    valid_text_items = {id for id, meta in meta_raw.items() 
                       if meta['text']}
# → 258,992 items
```

#### Bước 3: Lọc Triplets (quan trọng!)
```python
df = self.filter_triplets(df)
# Lọc users có < min_uc ratings
# Lọc items có < min_sc ratings
# → 12,101 items (GIẢM 95%!)
```

#### Bước 4: Densify Index
```python
df, umap, smap = self.densify_index(df)
remaining_items = set(smap.keys())
# → Xác định chính xác items nào cần download
```

#### Bước 5: Download Images
```python
downloaded_images, valid_items = download_and_verify_images_batch(
    items_to_download,  # CHỈ 12k items!
    image_folder,
    max_workers=20
)
# → 11,800 images (300 failed)
```

---

## 📊 CẤU TRÚC DỮ LIỆU

### Dataset pickle file:

```python
dataset = {
    'train': {
        1: [321, 4001, 4344, 8730],  # user_id → list of item_ids
        2: [4293, 8184, 6173],
        ...
    },
    'val': {
        1: [8785],  # Second-to-last item
        2: [2993],
        ...
    },
    'test': {
        1: [11063],  # Last item
        2: [2802],
        ...
    },
    'meta': {
        1: {  # NEW item_id (after densify)
            'text': 'Product description...',
            'image': 'http://original-url.com/...',
            'image_path': 'data/.../images/item_1_abc.jpg',
            'title': 'Product title'
        },
        ...
    },
    'umap': {
        'A1BKSLDI2V3D5K': 1,  # original_user_id → new_user_id
        ...
    },
    'smap': {
        'B00ABC123': 1,  # original_item_id → new_item_id
        ...
    }
}
```

### Sử dụng trong training:

```python
import pickle
from PIL import Image
from torchvision import transforms

# Load dataset
with open('data/preprocessed/.../dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Get training data
user_items = dataset['train'][user_id]

# Load image
for item_id in user_items:
    img_path = dataset['meta'][item_id]['image_path']
    img = Image.open(img_path).convert('RGB')
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)
```

---

## 🛠️ UTILITIES

Chi tiết đầy đủ tại: **`tools/README.md`**

### 1. Xem cấu trúc dataset:
```bash
python tools/inspect_pickle.py
```

Output:
- Số users, items
- Phân tích metadata (text/image availability)
- Sample data

### 2. Kiểm tra filtering results:
```bash
python tools/test_filtering.py
```

Với arguments:
```bash
python tools/test_filtering.py --use_text --use_image
```

### 3. Kiểm tra downloaded images:
```bash
python tools/test_download_images.py
```

Output:
- Số images downloaded
- Tổng size
- Sample images với paths

### 4. Xóa preprocessed data cũ:
```bash
python tools/clean_preprocessed.py
```

Hữu ích khi:
- Thay đổi settings (min_uc, min_sc, etc.)
- Muốn re-preprocess từ đầu

---

## 📈 HIỆU NĂNG & THỐNG KÊ

### Beauty Dataset (Thực tế):

| Metric | Value |
|--------|-------|
| Total items in metadata | 259,204 |
| After text filter | 258,992 |
| After triplet filter | 12,101 |
| Downloaded images | 11,800 |
| Final users | 22,332 |
| Total size (images) | ~80 MB |
| Download time | 10-15 phút |

### So sánh hiệu suất:

| Phương pháp | Items cần download | Thời gian |
|-------------|-------------------|-----------|
| Download trước | 259,204 | 5-10 giờ |
| **Download sau (Tối ưu)** | **12,101** | **10-20 phút** |
| **Tiết kiệm** | **95.3%** | **96%** |

---

## 🔍 TROUBLESHOOTING

### Lỗi: "ModuleNotFoundError: No module named 'PIL'"
```bash
pip install Pillow
```

### Lỗi: Download quá chậm
- Giảm `max_workers` xuống 10
- Tăng `timeout` lên 30s

### Dataset quá nhỏ sau filtering
- Giảm `min_uc` và `min_sc`
- Bỏ `--use_image` nếu không cần thiết

### Images bị corrupt
- Hàm `download_image()` đã verify bằng PIL
- Nếu vẫn có vấn đề, xóa folder images và download lại

---

## 📝 NOTES

### Best Practices:

1. **Luôn lọc triplets TRƯỚC khi download images**
   - Tiết kiệm 95% thời gian và bandwidth

2. **Sử dụng `--use_text` nếu cần text data**
   - Loại bỏ items không có description

3. **Chỉ dùng `--use_image` khi thực sự cần**
   - Download mất thời gian
   - MovieLens không có images

4. **Giữ lại URL gốc trong metadata**
   - Backup nếu cần download lại
   - Debug khi có vấn đề

5. **Pre-resize images nếu dataset lớn**
   - Faster training
   - Consistent input size

---

## 📞 SUPPORT

Nếu có vấn đề:
1. Kiểm tra logs khi chạy `data_prepare.py`
2. Dùng utility scripts để debug
3. Xem các file .md để hiểu chi tiết

---

**Version**: 2.0  
**Last Updated**: December 2025  
**Author**: Data Preprocessing Pipeline Team
