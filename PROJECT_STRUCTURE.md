# Project Structure (Updated)

## 📁 Cấu trúc Thư mục Hiện tại

```
seqreraking4rec/
│
├── config.py                    # ⚙️ Main configuration (argparse)
├── data_prepare.py              # 🚀 Data preprocessing
│
├── dataset/                     # 📦 Dataset modules
│   ├── base.py                  # Abstract base class
│   ├── beauty.py                # Amazon Beauty dataset
│   ├── games.py                 # Video Games dataset
│   ├── ml_100k.py               # MovieLens dataset
│   ├── clip_embeddings.py       # CLIP embedding extraction
│   └── utils.py                 # Utility functions
│
├── retrieval/                   # 🔍 Stage 1: Retrieval
│   ├── base.py                  # BaseRetriever interface
│   ├── registry.py              # Method registry
│   ├── models/                  # PyTorch models
│   │   ├── neural_lru.py       # NeuralLRURec
│   │   └── mmgcn.py            # MMGCN
│   ├── methods/                 # BaseRetriever wrappers
│   │   ├── lrurec.py           # LRURecRetriever
│   │   └── mmgcn.py            # MMGCNRetriever
│   └── train_lrurec.py          # ⚠️ DEPRECATED (use scripts/train_retrieval.py)
│
├── rerank/                       # 🎯 Stage 2: Reranking
│   ├── base.py                  # BaseReranker interface
│   ├── registry.py              # Method registry
│   ├── models/                  # LLM models
│   │   └── llm.py               # LLMModel (Qwen)
│   ├── methods/                 # BaseReranker wrappers
│   │   ├── identity.py         # IdentityReranker
│   │   ├── random_reranker.py  # RandomReranker
│   │   └── qwen_reranker.py    # QwenReranker
│   └── train_qwen.py            # ⚠️ DEPRECATED (use scripts/train_rerank.py)
│
├── pipelines/                    # 🔗 Pipeline Integration
│   ├── base.py                  # TwoStagePipeline + Config
│   └── __init__.py
│
├── evaluation/                   # 📊 Evaluation
│   ├── metrics.py               # Metric functions
│   └── offline_eval.py          # Offline evaluation
│
├── scripts/                      # 🚀 Training & Inference Scripts ⭐ MỚI
│   ├── train_retrieval.py      # Train Stage 1
│   ├── train_rerank.py          # Train Stage 2
│   ├── train_pipeline.py        # Train end-to-end
│   └── README.md                # Scripts documentation
│
├── tools/                        # 🛠️ Utility Scripts
│   ├── clean_preprocessed.py   # Clean preprocessed data
│   ├── inspect_pickle.py        # Inspect dataset
│   ├── test_filtering.py        # Test filtering
│   ├── test_download_images.py  # Test image download
│   └── README.md                # Tools documentation
│
├── notebooks/                    # 📓 Jupyter Notebooks ⭐ MỚI
│   └── check.ipynb
│
├── experiments/                  # 📁 Experiment Results
│   ├── retrieval/
│   ├── rerank/
│   └── pipeline/
│
├── data/                         # 💾 Data
│   ├── beauty/                  # Raw data
│   └── preprocessed/            # Preprocessed data
│
└── docs/                         # 📚 Documentation
    ├── ARCHITECTURE.md          # Architecture overview ⭐ MỚI
    ├── REFACTORING_SUMMARY.md   # Refactoring summary
    └── STRUCTURE_EVALUATION.md  # Structure evaluation
```

---

## 🎯 Core Components

### 1. **Data Layer**
- `dataset/`: Dataset implementations
- `data_prepare.py`: Preprocessing pipeline

### 2. **Model Layer**
- `retrieval/models/`: PyTorch retrieval models
- `rerank/models/`: LLM reranking models

### 3. **Method Layer**
- `retrieval/methods/`: Retrieval wrappers (implement BaseRetriever)
- `rerank/methods/`: Reranking wrappers (implement BaseReranker)

### 4. **Pipeline Layer**
- `pipelines/`: Two-stage pipeline integration

### 5. **Scripts Layer**
- `scripts/`: Training and inference scripts

---

## 🚀 Workflow

### 1. Preprocess Data
```bash
python data_prepare.py --use_text --use_image
```

### 2. Train Models
```bash
# Stage 1 only
python scripts/train_retrieval.py

# Stage 2 only
python scripts/train_rerank.py

# End-to-end
python scripts/train_pipeline.py --retrieval_method lrurec --rerank_method qwen
```

### 3. Evaluate
```bash
python evaluation/offline_eval.py
```

---

## 📝 Key Changes from Previous Structure

### ✅ Improvements:
1. **Created `scripts/` folder** - All training scripts centralized
2. **Created `pipelines/` module** - Two-stage pipeline integration
3. **Created `notebooks/` folder** - Jupyter notebooks organized
4. **Removed `retrieval/config.py`** - Config consolidated in `config.py`
5. **Removed `rerank/prompt.py`** - Empty file removed
6. **Updated documentation** - ARCHITECTURE.md, scripts/README.md

### ⚠️ Deprecated:
- `retrieval/train_lrurec.py` → Use `scripts/train_retrieval.py`
- `rerank/train_qwen.py` → Use `scripts/train_rerank.py`

---

## 📚 Documentation Files

- `README.md`: Main project documentation
- `ARCHITECTURE.md`: Architecture overview ⭐ NEW
- `PROJECT_STRUCTURE.md`: This file (updated)
- `REFACTORING_SUMMARY.md`: Refactoring summary
- `STRUCTURE_EVALUATION.md`: Structure evaluation
- `scripts/README.md`: Scripts documentation ⭐ NEW
- `tools/README.md`: Tools documentation

---

## 🔧 Adding New Methods

### Add New Retriever:
1. Create `retrieval/models/your_model.py` (PyTorch model)
2. Create `retrieval/methods/your_method.py` (BaseRetriever wrapper)
3. Register in `retrieval/registry.py`

### Add New Reranker:
1. Create `rerank/models/your_model.py` (LLM/model)
2. Create `rerank/methods/your_method.py` (BaseReranker wrapper)
3. Register in `rerank/registry.py`

---

**Last Updated**: 2025-01-27  
**Status**: ✅ Current structure
