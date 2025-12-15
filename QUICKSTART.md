# Quick Start Guide

Hướng dẫn nhanh để chạy project Sequential Reranking for Recommendation.

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

### Bước 3: Train Pipeline (Stage 1 + Stage 2)

```bash
# Full pipeline với Qwen reranker
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 20 \
    --rerank_method qwen \
    --rerank_top_k 10 \
    --rerank_mode retrieval

# Full pipeline với Qwen3-VL reranker (raw_image mode)
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --qwen3vl_mode raw_image \
    --rerank_top_k 50 \
    --rerank_mode retrieval

# Full pipeline với Qwen3-VL reranker (caption mode)
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --qwen3vl_mode caption \
    --rerank_top_k 50 \
    --rerank_mode retrieval

# Full pipeline với Qwen3-VL reranker (semantic_summary mode)
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --qwen3vl_mode semantic_summary \
    --rerank_top_k 50 \
    --rerank_mode retrieval

# Full pipeline với Qwen3-VL reranker (semantic_summary_small mode)
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen3vl \
    --qwen3vl_mode semantic_summary_small \
    --rerank_top_k 50 \
    --rerank_mode retrieval

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

### Bước 4: Offline Evaluation

```bash
# Evaluate retrieval only
python evaluation/offline_eval.py \
    --mode retrieval \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --K 10

# Evaluate full pipeline
python evaluation/offline_eval.py \
    --mode full \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen \
    --rerank_top_k 50 \
    --rerank_mode retrieval \
    --K 10

# Evaluate rerank only (ground truth + negatives)
python evaluation/offline_eval.py \
    --mode rerank_only \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen \
    --rerank_top_k 50 \
    --K 10
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

## ⚙️ Configuration

Các hyperparameters có thể điều chỉnh trong `config.py`:

- `--retrieval_epochs`: Số epochs cho retrieval training (default: 10)
- `--retrieval_lr`: Learning rate cho retrieval methods (default: 1e-3)
- `--rerank_epochs`: Số epochs cho rerank training (default: 10)
- `--rerank_lr`: Learning rate cho rerank methods (default: 1e-4)
- `--rerank_batch_size`: Batch size cho rerank training (default: 32)
- `--rerank_patience`: Early stopping patience (default: 5)
- `--qwen_max_candidates`: Max candidates cho Qwen reranker (None = dùng tất cả từ retrieval)
- `--qwen3vl_mode`: Prompt mode cho Qwen3-VL reranker (raw_image, caption, semantic_summary, semantic_summary_small)

## 📊 Output

- **Preprocessed data**: `data/preprocessed/{dataset_code}_min_rating{min_rating}-min_uc{min_uc}-min_sc{min_sc}/`
  - `dataset_single_export.csv`: Dataset với captions và semantic summaries
  - `clip_embeddings.pt`: CLIP embeddings (nếu có)
  - `blip2_captions.pt`: BLIP2 captions cache (nếu có)
  - `qwen3vl_semantic_summaries.pt`: Qwen3-VL semantic summaries cache (nếu có)

- **Retrieval results**: `experiments/retrieval/{method}/{dataset_code}/seed{seed}/`
  - `retrieved.csv`: Retrieved candidates
  - `retrieved_metrics.json`: Evaluation metrics

- **Pipeline results**: Inline trong console output

## 💡 Tips

1. **Qwen reranker**: Số lượng candidates tự động điều chỉnh theo `retrieval_top_k`. Có thể giới hạn bằng `--qwen_max_candidates` trong config.py.

2. **CLIP embeddings**: Cần chạy `data_prepare.py` với `--use_image` hoặc `--use_text` trước khi train MMGCN/VBPR/BM3.

3. **Caption/Semantic Summary**: Cần chạy `data_prepare.py` với `--generate_caption` hoặc `--generate_semantic_summary` để generate.

4. **Ground truth mode**: Dùng để đánh giá rerank quality độc lập với retrieval quality.

## 🔧 Troubleshooting

- **Qwen3-VL không load được**: Cần cài transformers từ source:
  ```bash
  pip install git+https://github.com/huggingface/transformers
  ```

- **CLIP embeddings không tìm thấy**: Chạy `data_prepare.py` với `--use_image` hoặc `--use_text` trước.

- **Out of memory**: Giảm `--batch_size_retrieval` hoặc `--rerank_batch_size` trong `config.py`.

