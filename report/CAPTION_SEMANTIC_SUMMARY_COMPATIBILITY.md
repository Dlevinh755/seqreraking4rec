# Caption và Semantic Summary - Compatibility Check

## ✅ Kết quả kiểm tra

**Có thể chạy cả caption generation và semantic summary generation trong cùng 1 lần chạy.**

## 📋 Chi tiết

### 1. Code Structure (`data_prepare.py`)

```python
# Line 21: Generate captions (nếu --generate_caption được set)
captions = maybe_generate_blip2_captions(dataset, data, args)

# Line 25: Generate semantic summaries (nếu --generate_semantic_summary được set)
semantic_summaries = maybe_generate_semantic_summaries(dataset, data, args)
```

**✅ Không có conflict**: Cả hai functions được gọi độc lập và không phụ thuộc vào nhau.

### 2. Dependencies

#### Caption Generation (`maybe_generate_blip2_captions`)
- **Requires**: `--use_image` flag
- **Requires**: `--generate_caption` flag
- **Model**: BLIP/BLIP2
- **Output**: `blip2_captions.pt` (cache) + CSV column `item_caption`

#### Semantic Summary Generation (`maybe_generate_semantic_summaries`)
- **Requires**: `--use_image` flag
- **Requires**: `--generate_semantic_summary` flag
- **Model**: Qwen3-VL
- **Output**: `qwen3vl_semantic_summaries.pt` (cache) + CSV column `item_semantic_summary`

**✅ Không có conflict**: Cả hai đều cần `--use_image`, nhưng không conflict với nhau.

### 3. CSV Export (`data_prepare.py`)

```python
# Lines 87-88: Cả hai đều được lưu vào CSV
"item_caption": caption or "",
"item_semantic_summary": semantic_summary or "",
```

**✅ Hỗ trợ đầy đủ**: CSV có cả hai columns, mỗi column được populate độc lập.

### 4. Cache Files

- **Captions**: `data/preprocessed/{dataset}/blip2_captions.pt`
- **Semantic Summaries**: `data/preprocessed/{dataset}/qwen3vl_semantic_summaries.pt`

**✅ Không conflict**: Mỗi function có cache file riêng.

## 🚀 Cách sử dụng

### Chạy cả hai cùng lúc:

```bash
python data_prepare.py \
    --dataset_code beauty \
    --use_image \
    --generate_caption \
    --generate_semantic_summary
```

### Kết quả:
- ✅ Captions được generate và lưu vào `item_caption` column
- ✅ Semantic summaries được generate và lưu vào `item_semantic_summary` column
- ✅ Cả hai được lưu vào cùng 1 CSV file: `dataset_single_export.csv`

## ⚠️ Lưu ý

1. **Memory Usage**: 
   - Cả hai models sẽ được load vào memory cùng lúc
   - BLIP/BLIP2: ~1-2GB
   - Qwen3-VL: ~4-8GB (tùy quantization)
   - **Total**: ~5-10GB GPU memory

2. **Time**: 
   - Cả hai sẽ chạy tuần tự (không parallel)
   - Caption generation: ~X phút
   - Semantic summary generation: ~Y phút
   - **Total**: X + Y phút

3. **GPU**: 
   - Cả hai đều cần GPU để chạy nhanh
   - Có thể chạy trên CPU nhưng sẽ rất chậm

## 💡 Optimization Tips

1. **Nếu memory không đủ**:
   - Chạy từng cái một (bỏ flag của cái không cần)
   - Sử dụng `--use_quantization` để giảm memory cho Qwen3-VL

2. **Nếu muốn nhanh hơn**:
   - Tăng `--semantic_summary_batch_size` (nếu GPU memory cho phép)
   - Sử dụng `--use_torch_compile` cho cả hai

3. **Nếu đã có cache**:
   - Nếu `blip2_captions.pt` đã tồn tại, caption generation sẽ skip
   - Nếu `qwen3vl_semantic_summaries.pt` đã tồn tại, semantic summary generation sẽ skip

## ✅ Kết luận

**Có thể chạy cả hai cùng lúc một cách an toàn và hiệu quả.**

Code đã được thiết kế để hỗ trợ cả hai operations độc lập, không có conflict hay dependency issues.

