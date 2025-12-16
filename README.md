# Sequential Reranking for Recommendation

Hệ thống recommendation hai giai đoạn (Two-Stage): **Retrieval (Stage 1)** + **Reranking (Stage 2)** với hỗ trợ multimodal (images, text, captions).

## 📋 Tổng quan

Pipeline recommendation hai giai đoạn:
- **Stage 1 (Retrieval)**: Generate candidates từ toàn bộ item pool
- **Stage 2 (Reranking)**: Re-rank candidates từ Stage 1 để tạo final recommendations

### ✨ Tính năng chính

- ✅ **4 Retrieval methods**: LRURec, MMGCN, VBPR, BM3
- ✅ **5 Rerank methods**: Qwen (LLM), Qwen3-VL (MLLM, 4 modes), VIP5, BERT4Rec
- ✅ **Multimodal support**: Images, text, captions (BLIP2), semantic summaries (Qwen3-VL)
- ✅ **Training độc lập**: Có thể train từng stage riêng hoặc end-to-end
- ✅ **Evaluation metrics**: Recall@K, NDCG@K, Hit@K tại @5, @10, @20
- ✅ **Early stopping**: Per-epoch validation với early stopping

## 📦 Cài đặt

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt transformers từ source (cần cho Qwen3-VL)
pip install git+https://github.com/huggingface/transformers
```

## 🚀 Quick Start

### Bước 1: Prepare Data

```bash
# Basic (chỉ ratings)
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20

# Với images và text (cho MMGCN, VBPR, BM3)
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image \
    --use_text

# Với captions (cho Qwen3-VL caption mode)
python data_prepare.py \
    --dataset_code beauty \
    --min_rating 3 \
    --min_uc 20 \
    --min_sc 20 \
    --use_image \
    --generate_caption

# Với semantic summaries (cho Qwen3-VL semantic_summary mode)
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
# LRURec (không cần images/text)
python scripts/train_retrieval.py --retrieval_method lrurec

# MMGCN (cần images/text)
python scripts/train_retrieval.py --retrieval_method mmgcn

# VBPR (cần images)
python scripts/train_retrieval.py --retrieval_method vbpr

# BM3 (cần images + text)
python scripts/train_retrieval.py --retrieval_method bm3
```

### Bước 3: Train Rerank (Stage 2)

#### Standalone (không cần retrieval model)

```bash
# Qwen LLM (text-only)
python scripts/train_rerank_standalone.py \
    --rerank_method qwen \
    --mode ground_truth

# Qwen3-VL (multimodal, 4 modes)
python scripts/train_rerank_standalone.py \
    --rerank_method qwen3vl \
    --mode ground_truth \
    --qwen3vl_mode raw_image  # hoặc: caption, semantic_summary, semantic_summary_small

# VIP5 (multimodal T5)
python scripts/train_rerank_standalone.py \
    --rerank_method vip5 \
    --mode ground_truth

# BERT4Rec (sequential)
python scripts/train_rerank_standalone.py \
    --rerank_method bert4rec \
    --mode ground_truth
```

#### Với Retrieval Model (Full Pipeline)

```bash
python scripts/train_rerank_standalone.py \
    --rerank_method qwen \
    --mode retrieval \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_top_k 50
```

### Bước 4: Train End-to-End (Stage 1 + Stage 2)

```bash
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method qwen \
    --rerank_top_k 50 \
    --rerank_mode retrieval
```

## 📊 Methods Available

### Retrieval Methods (Stage 1)

| Method | Description | Requirements | Command |
|--------|-------------|-------------|---------|
| `lrurec` | Neural LRU-based sequential | Dataset cơ bản | `--retrieval_method lrurec` |
| `mmgcn` | Multimodal Graph Convolutional Network | Images/text + CLIP | `--retrieval_method mmgcn` |
| `vbpr` | Visual Bayesian Personalized Ranking | Images + CLIP | `--retrieval_method vbpr` |
| `bm3` | Bootstrap Latent Representations | Images + text + CLIP | `--retrieval_method bm3` |

### Rerank Methods (Stage 2)

| Method | Description | Requirements | Command |
|--------|-------------|-------------|---------|
| `qwen` | Qwen LLM (text-only) | Text data | `--rerank_method qwen` |
| `qwen3vl` | Qwen3-VL (multimodal, 4 modes) | Tùy mode | `--rerank_method qwen3vl --qwen3vl_mode <mode>` |
| `vip5` | VIP5 multimodal T5 | Images + CLIP | `--rerank_method vip5` |
| `bert4rec` | BERT4Rec sequential | Sequential data | `--rerank_method bert4rec` |

### Qwen3-VL Modes

| Mode | Description | Data Preparation |
|------|-------------|------------------|
| `raw_image` | Raw images trong prompt | `--use_image` |
| `caption` | Image captions (BLIP2) | `--use_image --generate_caption` |
| `semantic_summary` | Semantic summaries (Qwen3-VL) | `--use_image --generate_semantic_summary` |
| `semantic_summary_small` | Semantic summaries (smaller model) | `--use_image --generate_semantic_summary` |

### Rerank Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `retrieval` | Dùng candidates từ Stage 1 | Full pipeline evaluation |
| `ground_truth` | GT + 19 random negatives | Rerank quality evaluation (independent) |

## ⚙️ Configuration

Các hyperparameters có thể điều chỉnh trong `config.py` hoặc command-line:

### Retrieval
- `--retrieval_epochs`: Số epochs (default: 100)
- `--retrieval_lr`: Learning rate (default: 1e-4)
- `--batch_size_retrieval`: Batch size (default: 512)
- `--retrieval_patience`: Early stopping patience (default: 10)

### Rerank
- `--rerank_epochs`: Số epochs (default: 10)
- `--rerank_lr`: Learning rate (default: 1e-4)
- `--rerank_batch_size`: Batch size (default: 32)
- `--rerank_patience`: Early stopping patience (default: 5)

### Qwen3-VL
- `--qwen3vl_mode`: Prompt mode (`raw_image`, `caption`, `semantic_summary`, `semantic_summary_small`)
- `--semantic_summary_batch_size`: Batch size cho summary generation (default: 4)

### Performance
- `--use_quantization`: 4-bit quantization (tiết kiệm memory)
- `--use_torch_compile`: torch.compile() optimization (tăng tốc)

## 📁 Output Structure

```
data/preprocessed/{dataset_code}_min_rating{min_rating}-min_uc{min_uc}-min_sc{min_sc}/
├── dataset_single_export.csv      # Dataset với metadata
├── clip_embeddings.pt              # CLIP embeddings (nếu có)
├── blip2_captions.pt               # BLIP2 captions cache (nếu có)
└── qwen3vl_semantic_summaries.pt   # Semantic summaries cache (nếu có)

experiments/
├── retrieval/{method}/{dataset_code}/seed{seed}/
│   ├── retrieved.csv               # Retrieved candidates
│   └── retrieved_metrics.json     # Evaluation metrics
└── rerank/{method}/{dataset_code}/seed{seed}/
    ├── model.pt                    # Trained model
    └── metrics.json                 # Evaluation metrics
```

## 📈 Evaluation Metrics

Tất cả evaluation tự động tính **Recall@K**, **NDCG@K**, **Hit@K** tại **@5, @10, @20**:

```
Metric       @5        @10        @20
Recall     0.1234    0.2345    0.3456
Ndcg       0.0567    0.0890    0.1234
Hit        0.4500    0.6700    0.8900
```

## 💡 Important Notes

1. **Data Preparation Order**:
   - Chạy `data_prepare.py` với đúng flags trước khi train
   - MMGCN/VBPR/BM3 cần `--use_image` hoặc `--use_text`
   - Qwen3-VL caption mode cần `--generate_caption`
   - Qwen3-VL semantic_summary mode cần `--generate_semantic_summary`

2. **Training Modes**:
   - **Standalone**: Train từng stage riêng (`train_retrieval.py`, `train_rerank_standalone.py`)
   - **End-to-end**: Train cả 2 stages (`train_pipeline.py`)

3. **Ground Truth Mode**:
   - Dùng để đánh giá rerank quality độc lập với retrieval
   - Tạo candidates = [ground_truth] + 19 random negatives
   - Không cần retrieval model

4. **CLIP Embeddings**:
   - Tự động extract khi chạy `data_prepare.py` với `--use_image` hoặc `--use_text`
   - Cần cho MMGCN, VBPR, BM3, VIP5

5. **Image Processing**:
   - Tự động resize về 224×224 (giữ aspect ratio)
   - Tiết kiệm memory và tăng tốc xử lý

## 🔧 Troubleshooting

- **Qwen3-VL không load**: Cài transformers từ source: `pip install git+https://github.com/huggingface/transformers`
- **Out of memory**: Giảm batch size trong `config.py` hoặc dùng `--use_quantization`
- **CLIP embeddings không tìm thấy**: Chạy `data_prepare.py` với `--use_image` hoặc `--use_text`

## 📚 Cấu trúc Project

```
seqreraking4rec/
├── config.py                    # Main configuration
├── data_prepare.py              # Data preprocessing
├── dataset/                     # Dataset modules
│   ├── base.py                  # Base dataset class
│   ├── beauty.py                # Amazon Beauty
│   ├── games.py                 # Video Games
│   └── ml_100k.py               # MovieLens
├── retrieval/                   # Stage 1: Retrieval
│   ├── base.py                  # BaseRetriever interface
│   ├── methods/                  # Retrieval methods
│   └── models/                  # PyTorch models
├── rerank/                      # Stage 2: Reranking
│   ├── base.py                  # BaseReranker interface
│   ├── methods/                  # Rerank methods
│   └── models/                  # PyTorch models
└── scripts/                     # Training scripts
    ├── train_retrieval.py        # Train retrieval only
    ├── train_rerank_standalone.py # Train rerank only
    └── train_pipeline.py         # Train end-to-end
```

## 📝 License

[Add your license here]
