# GPU Utilization Analysis - Semantic Summary Generation

## 🔍 Vấn đề phát hiện

Khi chạy semantic summary generation trên Kaggle T4 GPU, GPU utilization thấp do:

### 1. **Sequential Processing (Vấn đề chính)**
- **Line 207-208**: Code process từng image một trong loop:
  ```python
  # Process each image individually (VL models typically process one at a time)
  for idx, (item_id, img) in enumerate(zip(batch_ids, batch_images)):
  ```
- GPU phải chờ CPU xử lý xong mỗi image trước khi process tiếp
- GPU idle time giữa các images

### 2. **CPU Bottleneck**
- **Image Loading** (line 184): `Image.open(path).convert("RGB")` - chạy trên CPU
- **Image Resizing** (line 196): `img.resize()` - chạy trên CPU  
- **Preprocessing** (line 224-230): `processor.apply_chat_template()` - chạy trên CPU
- Tất cả chạy tuần tự, không parallel

### 3. **I/O Bottleneck**
- Đọc file từ disk cho mỗi image trong loop
- Không có prefetching hoặc parallel I/O

### 4. **Không có Batch Processing**
- Mỗi image được process riêng lẻ
- Không tận dụng được batch processing của GPU
- So sánh với BLIP2: BLIP2 process batch images cùng lúc

## 📊 So sánh với BLIP2

**BLIP2 (batch processing)**:
```python
# Process batch images cùng lúc
inputs = processor(images=batch_images, return_tensors="pt").to(device)
generated_ids = model.generate(**inputs, ...)
```

**Qwen3-VL (sequential processing)**:
```python
# Process từng image một
for img in batch_images:
    inputs = processor.apply_chat_template(messages, ...)  # 1 image
    generated_ids = model.generate(**inputs, ...)
```

## 🚀 Giải pháp đề xuất

### 1. **Pre-load và Preprocess Images (Parallel)**
- Sử dụng ThreadPoolExecutor để load và resize images song song
- Pre-load batch tiếp theo trong khi GPU đang process batch hiện tại

### 2. **Overlap I/O với GPU Computation**
- Load batch tiếp theo trong background thread
- GPU process batch hiện tại trong khi CPU load batch tiếp theo

### 3. **Tăng Batch Size**
- Tăng `--semantic_summary_batch_size` từ 4 lên 8, 16, 32
- Giảm overhead của sequential processing

### 4. **Thử Batch Processing (nếu Qwen3-VL support)**
- Qwen3-VL có thể không support batch cho multimodal inputs
- Nhưng có thể thử với list of messages

### 5. **Sử dụng DataLoader với num_workers**
- Parallelize image loading với DataLoader
- Prefetch next batch trong background

## ⚠️ Lưu ý

- Qwen3-VL có thể không support true batch processing cho multimodal inputs
- Nhưng vẫn có thể optimize bằng cách:
  - Parallel I/O
  - Pre-loading
  - Overlapping computation và I/O

