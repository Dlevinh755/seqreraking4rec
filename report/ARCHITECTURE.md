# Architecture Overview

## 📐 Cấu trúc Tổng quan

Project được tổ chức theo **two-stage recommendation pipeline**:

```
┌─────────────────┐
│   Stage 1:      │
│   Retrieval     │  →  Top-K candidates
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Stage 2:      │
│   Reranking     │  →  Final recommendations
└─────────────────┘
```

---

## 📁 Cấu trúc Thư mục

```
seqreraking4rec/
│
├── config.py                    # ⚙️ Main configuration (argparse)
├── data_prepare.py              # 🚀 Data preprocessing script
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
│   │   ├── neural_lru.py       # NeuralLRURec model
│   │   └── mmgcn.py             # MMGCN model
│   └── methods/                 # BaseRetriever wrappers
│       ├── lrurec.py            # LRURecRetriever
│       └── mmgcn.py             # MMGCNRetriever
│
├── rerank/                       # 🎯 Stage 2: Reranking
│   ├── base.py                  # BaseReranker interface
│   ├── registry.py              # Method registry
│   ├── models/                  # LLM models
│   │   └── llm.py               # LLMModel (Qwen)
│   └── methods/                 # BaseReranker wrappers
│       ├── identity.py          # IdentityReranker
│       ├── random_reranker.py   # RandomReranker
│       └── qwen_reranker.py     # QwenReranker
│
├── pipelines/                    # 🔗 Pipeline Integration
│   ├── base.py                  # TwoStagePipeline + Config
│   └── __init__.py
│
├── evaluation/                   # 📊 Evaluation
│   ├── metrics.py               # Metric functions (Recall@K, NDCG@K)
│   └── offline_eval.py          # Offline evaluation script
│
├── scripts/                      # 🚀 Training & Inference Scripts
│   ├── train_retrieval.py      # Train Stage 1
│   ├── train_rerank.py          # Train Stage 2
│   └── train_pipeline.py        # Train end-to-end
│
├── tools/                        # 🛠️ Utility Scripts
│   ├── clean_preprocessed.py   # Clean preprocessed data
│   ├── inspect_pickle.py        # Inspect dataset
│   ├── test_filtering.py        # Test filtering
│   └── test_download_images.py  # Test image download
│
├── notebooks/                    # 📓 Jupyter Notebooks
│   └── check.ipynb
│
├── experiments/                  # 📁 Experiment Results
│   ├── retrieval/
│   ├── rerank/
│   └── pipeline/
│
└── data/                         # 💾 Data
    ├── beauty/                  # Raw data
    └── preprocessed/            # Preprocessed data
```

---

## 🔧 Core Components

### 1. **Base Interfaces**

#### `BaseRetriever` (`retrieval/base.py`)
```python
class BaseRetriever(ABC):
    def fit(train_data: Dict[int, List[int]], **kwargs)
    def retrieve(user_id: int, exclude_items: Set[int]) -> List[int]
```

#### `BaseReranker` (`rerank/base.py`)
```python
class BaseReranker(ABC):
    def fit(train_data: Dict[int, List[int]], **kwargs)
    def rerank(user_id: int, candidates: List[int]) -> List[Tuple[int, float]]
```

### 2. **Registry Pattern**

Methods được đăng ký trong registry để dễ dàng thay đổi:

```python
# retrieval/registry.py
RETRIEVER_REGISTRY = {
    "lrurec": LRURecRetriever,
    "mmgcn": MMGCNRetriever,
}

# rerank/registry.py
RERANKER_REGISTRY = {
    "identity": IdentityReranker,
    "random": RandomReranker,
    "qwen": QwenReranker,
}
```

### 3. **TwoStagePipeline** (`pipelines/base.py`)

```python
pipeline = TwoStagePipeline(
    PipelineConfig(
        retrieval=RetrievalConfig(method="lrurec", top_k=200),
        rerank=RerankConfig(method="qwen", top_k=50)
    )
)

pipeline.fit(train_data)
recommendations = pipeline.recommend(user_id=1)
```

---

## 🔄 Data Flow

### Training Flow:
```
1. data_prepare.py
   ↓
   dataset.pkl (train/val/test splits)
   ↓
2. scripts/train_retrieval.py
   ↓
   Trained retriever + retrieved candidates
   ↓
3. scripts/train_rerank.py
   ↓
   Trained reranker
   ↓
4. evaluation/offline_eval.py
   ↓
   Final metrics
```

### Inference Flow:
```
User ID
   ↓
TwoStagePipeline.recommend()
   ↓
Stage 1: retriever.retrieve() → [candidate_ids]
   ↓
Stage 2: reranker.rerank() → [(item_id, score)]
   ↓
Final recommendations
```

---

## 🎯 Design Principles

### 1. **Separation of Concerns**
- **Models** (`models/`): PyTorch nn.Module implementations
- **Methods** (`methods/`): Interface wrappers (BaseRetriever/BaseReranker)
- **Scripts** (`scripts/`): Training and inference scripts

### 2. **Registry Pattern**
- Easy to add new methods
- Change methods via config, not code

### 3. **Modularity**
- Each stage is independent
- Can run Stage 1 only (retrieval-only mode)
- Can combine any retrieval + rerank method

### 4. **Extensibility**
- Add new retriever: Implement `BaseRetriever` → Register
- Add new reranker: Implement `BaseReranker` → Register

---

## 📚 Key Files

### Configuration
- `config.py`: Main configuration (argparse arguments)

### Data Processing
- `data_prepare.py`: Preprocess datasets
- `dataset/`: Dataset implementations

### Models
- `retrieval/models/`: Neural LRU, MMGCN
- `rerank/models/`: Qwen LLM

### Methods
- `retrieval/methods/`: Retrieval wrappers
- `rerank/methods/`: Reranking wrappers

### Pipeline
- `pipelines/base.py`: TwoStagePipeline implementation

### Scripts
- `scripts/train_retrieval.py`: Train Stage 1
- `scripts/train_rerank.py`: Train Stage 2
- `scripts/train_pipeline.py`: Train end-to-end

### Evaluation
- `evaluation/metrics.py`: Metric functions
- `evaluation/offline_eval.py`: Evaluation script

---

## 🚀 Quick Start

### 1. Preprocess Data
```bash
python data_prepare.py --use_text --use_image
```

### 2. Train Retrieval
```bash
python scripts/train_retrieval.py
```

### 3. Train Rerank
```bash
python scripts/train_rerank.py
```

### 4. Train End-to-End
```bash
python scripts/train_pipeline.py \
    --retrieval_method lrurec \
    --rerank_method qwen
```

---

## 📝 Notes

- All models implement standard interfaces for easy swapping
- Results are saved to `experiments/` directory
- Configuration is centralized in `config.py`
- Documentation is kept up-to-date in this file and `README.md`

