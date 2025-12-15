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

```bash
# Neural LRURec
python scripts/train_retrieval.py --retrieval_method lrurec

# MMGCN (requires CLIP embeddings)
python scripts/train_retrieval.py --retrieval_method mmgcn

# VBPR (requires CLIP image embeddings)
python scripts/train_retrieval.py --retrieval_method vbpr

# BM3 (requires CLIP embeddings)
python scripts/train_retrieval.py --retrieval_method bm3
```

### Bước 3: Train Rerank (Stage 2) - Standalone

```bash
# Train rerank riêng lẻ - Ground truth mode (không cần retrieval)
python scripts/train_rerank_standalone.py \
    --rerank_method bert4rec \
    --mode ground_truth \
    --rerank_top_k 50

# Train rerank với retrieval đã train sẵn
python scripts/train_rerank_standalone.py \
    --rerank_method qwen \
    --mode retrieval \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_top_k 50

# Train Qwen3-VL reranker (raw_image mode)
# Note: qwen3vl_mode được lấy từ config.py (--qwen3vl_mode raw_image)
python scripts/train_rerank_standalone.py \
    --rerank_method qwen3vl \
    --mode ground_truth \
    --rerank_top_k 50
# Set qwen3vl_mode trong config.py trước khi chạy: --qwen3vl_mode raw_image
```

### Bước 4: Train Pipeline (Stage 1 + Stage 2) - End-to-End

```bash
# Full pipeline với Qwen reranker
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 20 \
    --rerank_method qwen \
    --rerank_top_k 10 \
    --rerank_mode retrieval

# Full pipeline với Qwen3-VL reranker (raw_image mode)
# Note: qwen3vl_mode được lấy từ config.py (--qwen3vl_mode raw_image)
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --rerank_top_k 50 \
    --rerank_mode retrieval

# Full pipeline với Qwen3-VL reranker (caption mode)
# Note: Set qwen3vl_mode trong config.py hoặc dùng --qwen3vl_mode trong command line
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --rerank_top_k 50 \
    --rerank_mode retrieval
# Set qwen3vl_mode trong config.py: --qwen3vl_mode caption

# Full pipeline với Qwen3-VL reranker (semantic_summary mode)
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --rerank_top_k 50 \
    --rerank_mode retrieval
# Set qwen3vl_mode trong config.py: --qwen3vl_mode semantic_summary

# Full pipeline với Qwen3-VL reranker (semantic_summary_small mode)
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --rerank_top_k 50 \
    --rerank_mode retrieval
# Set qwen3vl_mode trong config.py: --qwen3vl_mode semantic_summary_small

# Full pipeline với VIP5 reranker
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method vip5 \
    --rerank_top_k 50 \
    --rerank_mode retrieval

# Full pipeline với BERT4Rec reranker
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method bert4rec \
    --rerank_top_k 50 \
    --rerank_mode retrieval

# Ground truth mode (đánh giá rerank quality)
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen \
    --rerank_top_k 10 \
    --rerank_mode ground_truth
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
- `lrurec`: Neural LRU-based sequential recommender
- `mmgcn`: Multimodal Graph Convolutional Network (requires CLIP embeddings)
- `vbpr`: Visual Bayesian Personalized Ranking (requires CLIP image embeddings)
- `bm3`: Bootstrap Latent Representations for Multi-modal Recommendation (requires CLIP embeddings)

### Rerank Methods (Stage 2)
- `qwen`: Qwen LLM-based reranker (text-only)
- `qwen3vl`: Qwen3-VL reranker with 4 prompt modes:
  - `raw_image`: Use raw images directly in prompt
  - `caption`: Use image captions
  - `semantic_summary`: Use semantic summaries with Qwen3-VL
  - `semantic_summary_small`: Use semantic summaries with smaller model (Qwen3-0.6B)
- `vip5`: VIP5 multimodal T5-based reranker
- `bert4rec`: BERT4Rec sequential reranker

### Rerank Modes
- `retrieval`: Use candidates from Stage 1 (default)
- `ground_truth`: Use ground truth + 19 random negatives (for rerank quality evaluation)

### Training Modes
- **End-to-end**: Train cả retrieval và rerank cùng lúc (`train_pipeline.py`)
- **Standalone rerank**: Train rerank riêng lẻ, không cần train retrieval (`train_rerank_standalone.py`)
  - `ground_truth` mode: Không cần retrieval model
  - `retrieval` mode: Cần load retrieval model đã train sẵn

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

- **Semantic summary generation chậm**:
  - Tăng `--semantic_summary_batch_size` nếu GPU memory cho phép (8, 16, 32)
  - Sử dụng `--use_quantization` để giảm memory và tăng tốc
  - Sử dụng `--use_torch_compile` để compile model

- **LLM inference chậm**:
  - Sử dụng `--use_torch_compile` để compile model
  - Model đã được load với 4-bit quantization mặc định (Unsloth)
  - Có thể batch multiple prompts nếu cần (xem OPTIMIZATION_GUIDE.md)

- **Evaluation không chạy được**: 
  - Kiểm tra xem đã train model chưa
  - Đảm bảo dataset đã được prepare với đúng flags (--use_image, --generate_caption, etc.)
  - Kiểm tra `--qwen3vl_mode` có đúng với mode đã train không

