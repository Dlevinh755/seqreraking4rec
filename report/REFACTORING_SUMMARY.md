# Tóm tắt Tối ưu hóa và Tổ chức lại Models

## 📋 Tổng quan

Đã thực hiện tối ưu hóa và tổ chức lại cấu trúc models trong project để:
- ✅ Tách biệt rõ ràng giữa **Models** (PyTorch nn.Module) và **Methods** (BaseRetriever/BaseReranker wrappers)
- ✅ Đảm bảo tất cả models đều implement đúng interface
- ✅ Nhất quán cấu trúc thư mục
- ✅ Loại bỏ code trùng lặp

---

## 🔄 Thay đổi Cấu trúc

### 1. Đổi tên thư mục
- **Trước**: `rerank/model/` 
- **Sau**: `rerank/models/` (nhất quán với `retrieval/models/`)

### 2. Cấu trúc mới

```
retrieval/
├── models/              # PyTorch nn.Module implementations
│   ├── neural_lru.py   # NeuralLRURec model
│   └── mmgcn.py        # MMGCN Net model
├── methods/             # BaseRetriever wrappers
│   ├── lrurec.py       # LRURecRetriever (wrapper cho NeuralLRURec)
│   └── mmgcn.py        # MMGCNRetriever (wrapper cho Net) ⭐ MỚI

rerank/
├── models/              # PyTorch/LLM model implementations
│   └── llm.py          # LLMModel (Qwen)
├── methods/             # BaseReranker wrappers
│   ├── identity.py     # IdentityReranker
│   ├── random_reranker.py
│   └── qwen_reranker.py # QwenReranker (wrapper cho LLMModel) ⭐ MỚI
```

---

## ✨ Các Wrapper mới được tạo

### 1. `QwenReranker` (`rerank/methods/qwen_reranker.py`)

**Mục đích**: Wrapper cho `LLMModel` để implement `BaseReranker` interface.

**Tính năng**:
- ✅ Implement `BaseReranker.fit()` và `BaseReranker.rerank()`
- ✅ Hỗ trợ training LLM model hoặc chỉ load pre-trained
- ✅ Sử dụng `build_prompt_from_candidates()` và `rank_candidates()` từ `rerank.models.llm`

**Usage**:
```python
from rerank.registry import get_reranker_class

RerankerCls = get_reranker_class("qwen")
reranker = RerankerCls(
    top_k=50,
    model_name="Qwen/Qwen3-0.6B",
    max_history=10
)
reranker.fit(
    train_data,
    item_id2text=item_id2text,
    user_history=user_history,
    train_data_for_llm=train_data_for_llm  # Optional
)
ranked = reranker.rerank(user_id=1, candidates=[1, 2, 3, ...])
```

### 2. `MMGCNRetriever` (`retrieval/methods/mmgcn.py`)

**Mục đích**: Wrapper cho `Net` model để implement `BaseRetriever` interface.

**Tính năng**:
- ✅ Implement `BaseRetriever.fit()` và `BaseRetriever.retrieve()`
- ✅ Hỗ trợ training với BPR loss
- ✅ Sử dụng visual và text features từ CLIP embeddings

**Usage**:
```python
from retrieval.registry import get_retriever_class

RetrieverCls = get_retriever_class("mmgcn")
retriever = RetrieverCls(
    top_k=50,
    dim_x=64,
    num_epochs=10
)
retriever.fit(
    train_data,
    num_user=num_users,
    num_item=num_items,
    v_feat=visual_features,  # CLIP image embeddings
    t_feat=text_features,     # CLIP text embeddings
    edge_index=edge_index,     # Graph edges
    val_data=val_data          # Optional for early stopping
)
candidates = retriever.retrieve(user_id=1, exclude_items={2, 3})
```

---

## 📝 Cập nhật Registry

### `retrieval/registry.py`
```python
RETRIEVER_REGISTRY = {
    "lrurec": LRURecRetriever,
    "mmgcn": MMGCNRetriever,  # ⭐ MỚI
}
```

### `rerank/registry.py`
```python
RERANKER_REGISTRY = {
    "identity": IdentityReranker,
    "random": RandomReranker,
    "qwen": QwenReranker,  # ⭐ MỚI
}
```

---

## 🔧 Cập nhật Imports

### `rerank/train_qwen.py`
- **Trước**: `from .model.llm import LLMModel`
- **Sau**: `from .models.llm import LLMModel`

---

## 🧹 Tối ưu Code

### Loại bỏ code trùng lặp:
1. ✅ Loại bỏ `LETTERS` trùng lặp trong `qwen_reranker.py` (đã có trong `llm.py`)
2. ✅ Loại bỏ imports không cần thiết (`ast`, `string`)

### Tạo `__init__.py`:
- ✅ `rerank/models/__init__.py` - Export `LLMModel`, `build_prompt_from_candidates`, `rank_candidates`

---

## 📊 Kết quả

### Trước khi tối ưu:
- ❌ Models và Methods lẫn lộn
- ❌ `LLMModel` không implement `BaseReranker`
- ❌ `MMGCN Net` không có wrapper
- ❌ Cấu trúc thư mục không nhất quán (`model/` vs `models/`)

### Sau khi tối ưu:
- ✅ Tách biệt rõ ràng: `models/` (nn.Module) vs `methods/` (wrappers)
- ✅ Tất cả models đều có wrapper implement đúng interface
- ✅ Cấu trúc nhất quán: `retrieval/models/` và `rerank/models/`
- ✅ Registry đầy đủ với tất cả methods
- ✅ Code sạch, không trùng lặp

---

## 🚀 Sử dụng

### Retrieval (Stage 1):
```python
from retrieval.registry import get_retriever_class

# LRURec
retriever = get_retriever_class("lrurec")(top_k=50)
retriever.fit(train_data, item_count=num_items, val_data=val_data)

# MMGCN
retriever = get_retriever_class("mmgcn")(top_k=50)
retriever.fit(train_data, num_user=..., num_item=..., v_feat=..., t_feat=..., edge_index=...)
```

### Reranking (Stage 2):
```python
from rerank.registry import get_reranker_class

# Qwen
reranker = get_reranker_class("qwen")(top_k=50)
reranker.fit(train_data, item_id2text=..., user_history=..., train_data_for_llm=...)
ranked = reranker.rerank(user_id=1, candidates=[...])
```

---

## 📌 Lưu ý

1. **MMGCNRetriever** cần CLIP embeddings (visual + text features) từ `dataset/clip_embeddings.py`
2. **QwenReranker** cần `item_id2text` mapping và `user_history` để build prompts
3. Tất cả wrappers đều tuân theo interface `BaseRetriever`/`BaseReranker` để dễ dàng thay thế

---

**Date**: 2025-01-27  
**Status**: ✅ Hoàn thành

