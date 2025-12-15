# Sequential Reranking for Recommendation

Hệ thống recommendation hai giai đoạn (Two-Stage): Retrieval (Stage 1) + Reranking (Stage 2).

## 📋 Tổng quan

Project này implement một pipeline recommendation hai giai đoạn:
- **Stage 1 (Retrieval)**: Generate candidates từ toàn bộ item pool
- **Stage 2 (Reranking)**: Re-rank candidates từ Stage 1 để tạo final recommendations

### Features
- ✅ 4 Retrieval methods: LRURec, MMGCN, VBPR, BM3
- ✅ 5 Rerank methods: Qwen, Qwen3-VL (4 modes), VIP5, BERT4Rec
- ✅ Multimodal support: Images, text, captions, semantic summaries
- ✅ Training và evaluation độc lập cho từng stage
- ✅ Evaluation metrics: Recall@K, NDCG@K, Hit@K tại @5, @10, @20
- ✅ Image preprocessing: Tự động resize về 448px
- ✅ Per-epoch validation với early stopping

## 📦 Cài đặt

### 1. Cài đặt dependencies

```bash
# Cài đặt các thư viện cơ bản
pip install -r requirements.txt

# Cài đặt transformers từ source (cần cho Qwen3-VL)
pip install git+https://github.com/huggingface/transformers
```

### 2. Cấu hình dataset

Chỉnh sửa `config.py` hoặc sử dụng command-line arguments để cấu hình dataset và hyperparameters.

## 🚀 Chạy Project

### Bước 1: Prepare Data

```bash
# Basic data preparation
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20

# Với image filtering và CLIP embeddings
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image \
    --use_text

# Với caption generation (BLIP2)
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image \
    --generate_caption

# Với semantic summary generation (Qwen3-VL)
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image \
    --generate_semantic_summary
```

### Bước 2: Train Retrieval (Stage 1)

#### 2.1. Neural LRURec
**Requirements**: Không cần gì đặc biệt, chỉ cần dataset đã được prepare.

```bash
python scripts/train_retrieval.py --retrieval_method lrurec
```

#### 2.2. MMGCN (Multimodal Graph Convolutional Network)
**Requirements**: 
- Dataset với images hoặc text (chạy `data_prepare.py` với `--use_image` hoặc `--use_text`)
- CLIP embeddings sẽ được tự động extract

```bash
# Prepare data với images/text trước
python data_prepare.py \
    --dataset_code beauty \
    --use_image \
    --use_text

# Train MMGCN
python scripts/train_retrieval.py --retrieval_method mmgcn
```

#### 2.3. VBPR (Visual Bayesian Personalized Ranking)
**Requirements**: 
- Dataset với images (chạy `data_prepare.py` với `--use_image`)
- CLIP image embeddings sẽ được tự động extract

```bash
# Prepare data với images trước
python data_prepare.py \
    --dataset_code beauty \
    --use_image

# Train VBPR
python scripts/train_retrieval.py --retrieval_method vbpr
```

#### 2.4. BM3 (Bootstrap Latent Representations for Multi-modal Recommendation)
**Requirements**: 
- Dataset với images và text (chạy `data_prepare.py` với `--use_image` và `--use_text`)
- CLIP embeddings sẽ được tự động extract

```bash
# Prepare data với images và text trước
python data_prepare.py \
    --dataset_code beauty \
    --use_image \
    --use_text

# Train BM3
python scripts/train_retrieval.py --retrieval_method bm3
```

### Bước 3: Train Rerank (Stage 2) - Standalone

#### 3.1. Qwen LLM Reranker

**Requirements**: Dataset với text data (`item_text` column)

**Ground Truth Mode** (không cần retrieval model):
```bash
python scripts/train_rerank_standalone.py \
    --rerank_method qwen \
    --mode ground_truth \
    --rerank_top_k 50
```

**Retrieval Mode** (cần retrieval model đã train):
```bash
python scripts/train_rerank_standalone.py \
    --rerank_method qwen \
    --mode retrieval \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_top_k 50
```

#### 3.2. Qwen3-VL Reranker

**Requirements**: 
- `raw_image` mode: Dataset với images
- `caption` mode: Dataset với images + captions (chạy `data_prepare.py` với `--generate_caption`)
- `semantic_summary` mode: Dataset với images + semantic summaries (chạy `data_prepare.py` với `--generate_semantic_summary`)
- `semantic_summary_small` mode: Dataset với images + semantic summaries

**3.2.1. Raw Image Mode**
```bash
# Set mode trong config.py hoặc command line
# config.py: --qwen3vl_mode raw_image

python scripts/train_rerank_standalone.py \
    --rerank_method qwen3vl \
    --mode ground_truth \
    --rerank_top_k 50
```

**3.2.2. Caption Mode**
```bash
# Prepare data với captions trước
python data_prepare.py \
    --dataset_code beauty \
    --use_image \
    --generate_caption

# Set mode trong config.py
# config.py: --qwen3vl_mode caption

python scripts/train_rerank_standalone.py \
    --rerank_method qwen3vl \
    --mode ground_truth \
    --rerank_top_k 50
```

**3.2.3. Semantic Summary Mode**
```bash
# Prepare data với semantic summaries trước
python data_prepare.py \
    --dataset_code beauty \
    --use_image \
    --generate_semantic_summary

# Set mode trong config.py
# config.py: --qwen3vl_mode semantic_summary

python scripts/train_rerank_standalone.py \
    --rerank_method qwen3vl \
    --mode ground_truth \
    --rerank_top_k 50
```

**3.2.4. Semantic Summary Small Mode**
```bash
# Prepare data với semantic summaries trước (giống semantic_summary mode)
python data_prepare.py \
    --dataset_code beauty \
    --use_image \
    --generate_semantic_summary

# Set mode trong config.py
# config.py: --qwen3vl_mode semantic_summary_small

python scripts/train_rerank_standalone.py \
    --rerank_method qwen3vl \
    --mode ground_truth \
    --rerank_top_k 50
```

**Retrieval Mode** (cho tất cả Qwen3-VL modes):
```bash
# Set qwen3vl_mode trong config.py trước
python scripts/train_rerank_standalone.py \
    --rerank_method qwen3vl \
    --mode retrieval \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_top_k 50
```

#### 3.3. VIP5 Reranker

**Requirements**: Dataset với images + CLIP embeddings

**Ground Truth Mode**:
```bash
python scripts/train_rerank_standalone.py \
    --rerank_method vip5 \
    --mode ground_truth \
    --rerank_top_k 50
```

**Retrieval Mode**:
```bash
python scripts/train_rerank_standalone.py \
    --rerank_method vip5 \
    --mode retrieval \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_top_k 50
```

#### 3.4. BERT4Rec Reranker

**Requirements**: Dataset với sequential data (không cần images/text)

**Ground Truth Mode**:
```bash
python scripts/train_rerank_standalone.py \
    --rerank_method bert4rec \
    --mode ground_truth \
    --rerank_top_k 50
```

**Retrieval Mode**:
```bash
python scripts/train_rerank_standalone.py \
    --rerank_method bert4rec \
    --mode retrieval \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_top_k 50
```

### Bước 4: Train Pipeline (Stage 1 + Stage 2) - End-to-End

#### 4.1. Qwen LLM Reranker

**Retrieval Mode**:
```bash
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen \
    --rerank_top_k 50 \
    --rerank_mode retrieval
```

**Ground Truth Mode**:
```bash
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen \
    --rerank_top_k 50 \
    --rerank_mode ground_truth
```

#### 4.2. Qwen3-VL Reranker

**4.2.1. Raw Image Mode**
```bash
# Set mode trong config.py: --qwen3vl_mode raw_image
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --rerank_top_k 50 \
    --rerank_mode retrieval
```

**4.2.2. Caption Mode**
```bash
# Prepare data với captions trước
python data_prepare.py \
    --dataset_code beauty \
    --use_image \
    --generate_caption

# Set mode trong config.py: --qwen3vl_mode caption
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --rerank_top_k 50 \
    --rerank_mode retrieval
```

**4.2.3. Semantic Summary Mode**
```bash
# Prepare data với semantic summaries trước
python data_prepare.py \
    --dataset_code beauty \
    --use_image \
    --generate_semantic_summary

# Set mode trong config.py: --qwen3vl_mode semantic_summary
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --rerank_top_k 50 \
    --rerank_mode retrieval
```

**4.2.4. Semantic Summary Small Mode**
```bash
# Prepare data với semantic summaries trước
python data_prepare.py \
    --dataset_code beauty \
    --use_image \
    --generate_semantic_summary

# Set mode trong config.py: --qwen3vl_mode semantic_summary_small
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --rerank_top_k 50 \
    --rerank_mode retrieval
```

#### 4.3. VIP5 Reranker

```bash
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method vip5 \
    --rerank_top_k 50 \
    --rerank_mode retrieval
```

#### 4.4. BERT4Rec Reranker

```bash
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method bert4rec \
    --rerank_top_k 50 \
    --rerank_mode retrieval
```

### Bước 5: Offline Evaluation

Tất cả evaluation tự động tính metrics cho @5, @10, @20 với Recall, NDCG, và Hit Rate trên cả **val** và **test** sets.

```bash
# Evaluate retrieval only
python evaluation/offline_eval.py \
    --mode retrieval \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --split val  # or --split test

# Evaluate full pipeline với Qwen reranker
python evaluation/offline_eval.py \
    --mode full \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen \
    --rerank_top_k 50 \
    --rerank_mode retrieval \
    --split test

# Evaluate full pipeline với Qwen3-VL (raw_image mode)
python evaluation/offline_eval.py \
    --mode full \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --qwen3vl_mode raw_image \
    --rerank_top_k 50 \
    --rerank_mode retrieval \
    --split val

# Evaluate full pipeline với Qwen3-VL (caption mode)
python evaluation/offline_eval.py \
    --mode full \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --qwen3vl_mode caption \
    --rerank_top_k 50 \
    --rerank_mode retrieval \
    --split test

# Evaluate full pipeline với Qwen3-VL (semantic_summary mode)
python evaluation/offline_eval.py \
    --mode full \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --qwen3vl_mode semantic_summary \
    --rerank_top_k 50 \
    --rerank_mode retrieval \
    --split val

# Evaluate full pipeline với Qwen3-VL (semantic_summary_small mode)
python evaluation/offline_eval.py \
    --mode full \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --qwen3vl_mode semantic_summary_small \
    --rerank_top_k 50 \
    --rerank_mode retrieval \
    --split test

# Evaluate rerank only (ground truth + negatives)
python evaluation/offline_eval.py \
    --mode rerank_only \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen \
    --rerank_top_k 50 \
    --split val
```

**Output format**: Tất cả metrics được hiển thị dạng bảng với @5, @10, @20:
```
Metric       @5        @10        @20
Recall     0.1234    0.2345    0.3456
Ndcg       0.0567    0.0890    0.1234
Hit        0.4500    0.6700    0.8900
```

## 📝 Các Methods Available

### Retrieval Methods (Stage 1)

| Method | Description | Requirements | Training Command |
|--------|-------------|--------------|------------------|
| `lrurec` | Neural LRU-based sequential recommender | Dataset cơ bản | `python scripts/train_retrieval.py --retrieval_method lrurec` |
| `mmgcn` | Multimodal Graph Convolutional Network | Images/text + CLIP embeddings | `python scripts/train_retrieval.py --retrieval_method mmgcn` |
| `vbpr` | Visual Bayesian Personalized Ranking | Images + CLIP image embeddings | `python scripts/train_retrieval.py --retrieval_method vbpr` |
| `bm3` | Bootstrap Latent Representations for Multi-modal Recommendation | Images + text + CLIP embeddings | `python scripts/train_retrieval.py --retrieval_method bm3` |

### Rerank Methods (Stage 2)

| Method | Description | Requirements | Training Command (Standalone) |
|--------|-------------|--------------|-------------------------------|
| `qwen` | Qwen LLM-based reranker (text-only) | Text data | `python scripts/train_rerank_standalone.py --rerank_method qwen --mode ground_truth` |
| `qwen3vl` | Qwen3-VL reranker với 4 modes | Tùy mode (xem bảng dưới) | `python scripts/train_rerank_standalone.py --rerank_method qwen3vl --mode ground_truth` |
| `vip5` | VIP5 multimodal T5-based reranker | Images + CLIP embeddings | `python scripts/train_rerank_standalone.py --rerank_method vip5 --mode ground_truth` |
| `bert4rec` | BERT4Rec sequential reranker | Sequential data | `python scripts/train_rerank_standalone.py --rerank_method bert4rec --mode ground_truth` |

### Qwen3-VL Modes

| Mode | Description | Requirements | Data Preparation |
|------|-------------|--------------|-----------------|
| `raw_image` | Use raw images directly in prompt | Images | `--use_image` |
| `caption` | Use image captions | Images + captions | `--use_image --generate_caption` |
| `semantic_summary` | Use semantic summaries with Qwen3-VL | Images + semantic summaries | `--use_image --generate_semantic_summary` |
| `semantic_summary_small` | Use semantic summaries with smaller model | Images + semantic summaries | `--use_image --generate_semantic_summary` |

### Rerank Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `retrieval` | Use candidates from Stage 1 | Full pipeline evaluation |
| `ground_truth` | Use ground truth + 19 random negatives | Rerank quality evaluation (independent of retrieval) |

### Training Modes

| Mode | Script | Description | Requirements |
|------|--------|-------------|--------------|
| **End-to-end** | `train_pipeline.py` | Train cả retrieval và rerank cùng lúc | Cả hai stages |
| **Standalone rerank** | `train_rerank_standalone.py` | Train rerank riêng lẻ | Rerank method + (optional) retrieval model |
| **Standalone retrieval** | `train_retrieval.py` | Train retrieval riêng lẻ | Retrieval method |

## ⚙️ Configuration

Các hyperparameters có thể điều chỉnh trong `config.py`:

### Retrieval Hyperparameters
- `--retrieval_epochs`: Số epochs cho retrieval training (default: 10)
- `--retrieval_lr`: Learning rate cho retrieval methods (default: 1e-3)
- `--batch_size_retrieval`: Batch size cho retrieval training (default: 128)
- `--retrieval_patience`: Early stopping patience (default: 5)

### Rerank Hyperparameters
- `--rerank_epochs`: Số epochs cho rerank training (default: 10)
- `--rerank_lr`: Learning rate cho rerank methods (default: 1e-4)
- `--rerank_batch_size`: Batch size cho rerank training (default: 32)
- `--rerank_patience`: Early stopping patience (default: 5)

### Reranker-Specific Config
- `--qwen_max_candidates`: Max candidates cho Qwen reranker (None = dùng tất cả từ retrieval)
- `--qwen3vl_mode`: Prompt mode cho Qwen3-VL reranker (raw_image, caption, semantic_summary, semantic_summary_small)

### Performance Optimization Config
- `--semantic_summary_batch_size`: Batch size cho semantic summary generation (default: 4, có thể tăng lên 8, 16, 32 nếu GPU memory cho phép)
- `--use_quantization`: Sử dụng 4-bit quantization cho models (tiết kiệm memory, tăng tốc)
- `--use_torch_compile`: Sử dụng torch.compile() để compile models (tăng tốc inference, cần PyTorch 2.0+)

## 📊 Output

### Preprocessed Data
`data/preprocessed/{dataset_code}_min_rating{min_rating}-min_uc{min_uc}-min_sc{min_sc}/`
- `dataset_single_export.csv`: Dataset với captions và semantic summaries
- `clip_embeddings.pt`: CLIP embeddings (nếu có)
- `blip2_captions.pt`: BLIP2 captions cache (nếu có)
- `qwen3vl_semantic_summaries.pt`: Qwen3-VL semantic summaries cache (nếu có)

### Retrieval Results
`experiments/retrieval/{method}/{dataset_code}/seed{seed}/`
- `retrieved.csv`: Retrieved candidates
- `retrieved_metrics.json`: Evaluation metrics với @5, @10, @20

### Evaluation Results
Tất cả evaluation tự động tính và hiển thị metrics cho @5, @10, @20:
- **Recall@K**: Tỷ lệ relevant items được retrieve trong top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain tại K
- **Hit@K**: Tỷ lệ users có ít nhất 1 relevant item trong top-K

Output format:
```
Metric       @5        @10        @20
Recall     0.1234    0.2345    0.3456
Ndcg       0.0567    0.0890    0.1234
Hit        0.4500    0.6700    0.8900
```

## 💡 Tips

1. **Training độc lập**: 
   - Sử dụng `train_rerank_standalone.py` để train rerank riêng lẻ, không cần train retrieval
   - Ground truth mode không cần retrieval model
   - Retrieval mode cần load retrieval model đã train sẵn

2. **Qwen reranker**: 
   - Số lượng candidates tự động điều chỉnh theo `retrieval_top_k`
   - Có thể giới hạn bằng `--qwen_max_candidates` trong config.py

3. **CLIP embeddings**: 
   - Cần chạy `data_prepare.py` với `--use_image` hoặc `--use_text` trước khi train MMGCN/VBPR/BM3

4. **Caption/Semantic Summary**: 
   - Cần chạy `data_prepare.py` với `--generate_caption` hoặc `--generate_semantic_summary` để generate
   - Captions cần cho Qwen3-VL `caption` mode
   - Semantic summaries cần cho Qwen3-VL `semantic_summary` và `semantic_summary_small` modes

5. **Image Resize**: 
   - Tất cả images được tự động resize về max 448px trên cạnh dài hơn (giữ nguyên aspect ratio)
   - Giúp tiết kiệm memory và tăng tốc xử lý
   - Áp dụng cho cả training và inference

6. **Ground truth mode**: 
   - Dùng để đánh giá rerank quality độc lập với retrieval quality
   - Tạo candidates = [ground_truth] + 19 random negatives

7. **Evaluation metrics**: 
   - Tất cả evaluation tự động tính @5, @10, @20
   - Metrics: Recall, NDCG, Hit Rate
   - Có thể evaluate trên cả val và test sets (dùng `--split val` hoặc `--split test`)

8. **Qwen3-VL Training**: 
   - Tất cả 4 modes đều hỗ trợ training: `raw_image`, `caption`, `semantic_summary`, `semantic_summary_small`
   - `raw_image` mode sử dụng raw images trực tiếp cho cả training và inference
   - Training sử dụng per-epoch validation với early stopping
   - Xem chi tiết trong `QWEN3VL_TRAINING_REPORT.md`

## ⚡ Performance Optimization

### Tăng tốc Semantic Summary Generation

```bash
# Tăng batch size (nếu GPU memory cho phép)
python data_prepare.py \
    --dataset_code beauty \
    --use_image \
    --generate_semantic_summary \
    --semantic_summary_batch_size 8  # Tăng từ 4 lên 8

# Sử dụng quantization để tiết kiệm memory
python data_prepare.py \
    --dataset_code beauty \
    --use_image \
    --generate_semantic_summary \
    --use_quantization  # 4-bit quantization

# Sử dụng torch.compile() để tăng tốc (PyTorch 2.0+)
python data_prepare.py \
    --dataset_code beauty \
    --use_image \
    --generate_semantic_summary \
    --use_torch_compile
```

### Tăng tốc LLM Inference

```bash
# Sử dụng torch.compile() cho LLM inference
python scripts/train_pipeline.py \
    --rerank_method qwen \
    --use_torch_compile  # Compile model để tăng tốc

# Sử dụng quantization (đã có sẵn trong Unsloth)
# Model đã được load với 4-bit quantization mặc định
```

**Expected Speedup**:
- `--semantic_summary_batch_size 8`: 2-4x faster
- `--use_quantization`: 1.5-2x faster, -50% memory
- `--use_torch_compile`: 1.2-1.5x faster

Xem chi tiết trong `OPTIMIZATION_GUIDE.md`.

## 🔧 Troubleshooting

- **Qwen3-VL không load được**: Cần cài transformers từ source:
  ```bash
  pip install git+https://github.com/huggingface/transformers
  ```

- **CLIP embeddings không tìm thấy**: Chạy `data_prepare.py` với `--use_image` hoặc `--use_text` trước.

- **Out of memory**: 
  - Giảm `--batch_size_retrieval` hoặc `--rerank_batch_size` trong `config.py`
  - Với Qwen3-VL `raw_image` mode: giảm batch size xuống 4-8
  - Images đã được tự động resize về 448px để tiết kiệm memory
  - Sử dụng `--use_quantization` để giảm memory usage

- **Qwen3-VL training chậm**: 
  - Sử dụng `semantic_summary_small` mode (nhẹ hơn, nhanh hơn)
  - Giảm batch size hoặc số lượng training samples
  - Sử dụng GPU với đủ memory (recommended: 12GB+ cho VL modes)
  - Sử dụng `--use_torch_compile` để tăng tốc

- **Semantic summary generation chậm / GPU utilization thấp**:
  - **Vấn đề**: Code process từng image một, gây CPU bottleneck và GPU idle time
  - **Giải pháp đã implement**:
    - ✅ Parallel image loading (ThreadPoolExecutor) để giảm I/O bottleneck
    - ✅ Pre-loading next batch trong background (overlap I/O với GPU computation)
    - ✅ **Batch processing thử nghiệm** (nếu Qwen3-VL support, sẽ tự động fallback nếu không)
    - ✅ Giảm `max_new_tokens` từ 128 xuống 64 (có thể config)
    - ✅ Pre-load all images option (nhanh hơn nhưng tốn RAM)
  - **Khuyến nghị để tăng tốc**:
    ```bash
    # Option 1: Tăng batch size và giảm max tokens
    python data_prepare.py \
        --dataset_code beauty \
        --use_image \
        --generate_semantic_summary \
        --semantic_summary_batch_size 8 \
        --semantic_summary_max_tokens 64
    
    # Option 2: Pre-load all images (nhanh nhất nhưng tốn RAM)
    python data_prepare.py \
        --dataset_code beauty \
        --use_image \
        --generate_semantic_summary \
        --semantic_summary_batch_size 8 \
        --semantic_summary_max_tokens 64 \
        --preload_all_images
    
    # Option 3: Kết hợp tất cả optimizations
    python data_prepare.py \
        --dataset_code beauty \
        --use_image \
        --generate_semantic_summary \
        --semantic_summary_batch_size 16 \
        --semantic_summary_max_tokens 64 \
        --preload_all_images \
        --use_quantization \
        --use_torch_compile
    ```
  - **Lưu ý**: 
    - Qwen3-VL có thể không support true batch processing cho multimodal inputs
    - Code sẽ tự động thử batch processing, nếu fail sẽ fallback về sequential
    - `--preload_all_images` tốn RAM nhưng loại bỏ hoàn toàn I/O bottleneck
    - Giảm `--semantic_summary_max_tokens` từ 64 xuống 32-48 nếu cần tốc độ hơn nữa (nhưng có thể giảm chất lượng)

- **LLM inference chậm**:
  - Sử dụng `--use_torch_compile` để compile model
  - Model đã được load với 4-bit quantization mặc định (Unsloth)
  - Có thể batch multiple prompts nếu cần (xem OPTIMIZATION_GUIDE.md)

- **Evaluation không chạy được**: 
  - Kiểm tra xem đã train model chưa
  - Đảm bảo dataset đã được prepare với đúng flags (--use_image, --generate_caption, etc.)
  - Kiểm tra `--qwen3vl_mode` có đúng với mode đã train không

