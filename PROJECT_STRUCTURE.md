# Project Structure

```
inprocessing/
│
├── README.md                    # 📚 Tài liệu đầy đủ
│
├── config.py                    # ⚙️ Cấu hình arguments
├── data_prepare.py              # 🚀 Script chính để preprocessing
│
├── datasets/                    # 📦 Module datasets
│   ├── __init__.py
│   ├── base.py                  # Abstract base class
│   ├── utils.py                 # Download & utility functions
│   ├── beauty.py                # Amazon Beauty dataset
│   ├── games.py                 # Video Games dataset
│   └── ml_100k.py               # MovieLens dataset
│
├── llm4rec/                     # 🤖 Module cho model
│   └── retrieval.py
│
├── tools/                       # 🛠️ Scripts tiện ích
│   ├── README.md                # Hướng dẫn sử dụng tools
│   ├── clean_preprocessed.py   # Xóa data cũ
│   ├── inspect_pickle.py        # Xem cấu trúc dataset
│   ├── test_filtering.py        # Test filtering
│   └── test_download_images.py  # Test download results
│
├── data/                        # 💾 Dữ liệu (tự tạo khi chạy)
│   ├── beauty/                  # Raw data
│   └── preprocessed/            # Preprocessed data
│       └── beauty_min_rating3-min_uc5-min_sc5/
│           ├── dataset.pkl      # Dataset đã xử lý
│           └── images/          # Downloaded images
│
└── venv/                        # 🐍 Virtual environment
```

## Core Files (QUAN TRỌNG):

### 1. config.py
- Định nghĩa arguments
- Default settings

### 2. data_prepare.py
- Script chính để chạy preprocessing
- Gọi dataset_factory()
- Load và lưu dataset

### 3. datasets/
- **base.py**: Abstract class với logic chung
- **utils.py**: Download, verify images, utility functions
- **beauty.py, games.py, ml_100k.py**: Implementation cho từng dataset

## Utility Scripts:

Xem chi tiết tại: `tools/README.md`

### clean_preprocessed.py
Xóa folder preprocessed để tạo lại từ đầu

```bash
python tools/clean_preprocessed.py
```

### inspect_pickle.py
Xem cấu trúc và thống kê của dataset

```bash
python tools/inspect_pickle.py
```

### test_filtering.py
Kiểm tra kết quả filtering

```bash
python tools/test_filtering.py --use_text --use_image
```

### test_download_images.py
Kiểm tra images đã download

```bash
python tools/test_download_images.py
```

## Workflow:

```
1. python data_prepare.py --use_text --use_image
   ↓
2. python tools/inspect_pickle.py
   (Kiểm tra kết quả)
   ↓
3. Sử dụng dataset cho training
```

## Dọn dẹp đã thực hiện:

### ✅ Đã xóa:
- ❌ test_image_checker.py (test cũ)
- ❌ test_image_validation.py (test cũ)
- ❌ test_image_download.py (test đơn giản)
- ❌ demo_download_images.py (không cần thiết)
- ❌ test_images/ (folder test)
- ❌ 4 file .md riêng lẻ (đã gộp vào README.md)

### ✅ Giữ lại:
- ✓ Core files (config, data_prepare, datasets/)
- ✓ Utility scripts (4 files hữu ích)
- ✓ README.md (tài liệu tổng hợp)

## Tổng kết:

**Trước**: 20+ files (nhiều trùng lặp)  
**Sau**: 12 files chính (gọn gàng, có tổ chức)  
**Tools**: 4 files trong folder riêng biệt

→ Dễ maintain và sử dụng hơn!
