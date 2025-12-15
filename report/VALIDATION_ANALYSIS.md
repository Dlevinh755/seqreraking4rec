# Phân tích Validation trong Training

## Tổng quan

Kiểm tra xem các methods có thực hiện validation sau mỗi epoch không và có early stopping không.

## Retrieval Methods (Stage 1)

### ✅ Tất cả retrieval methods đều có validation

#### 1. **LRURecRetriever** (`retrieval/methods/lrurec.py`)
- ✅ **Có validation** sau mỗi epoch (dòng 158-180)
- ✅ **Early stopping** dựa trên `patience`
- ✅ **Best model tracking**: Lưu best state dựa trên `val_recall`
- **Validation metrics**: Recall@K, NDCG@K
- **Condition**: `if val_data is not None and len(val_data) > 0`

#### 2. **MMGCNRetriever** (`retrieval/methods/mmgcn.py`)
- ✅ **Có validation** sau mỗi epoch (dòng 157-179)
- ✅ **Early stopping** dựa trên `patience`
- ✅ **Best model tracking**: Lưu best state dựa trên `val_recall`
- **Validation metrics**: Recall@K
- **Condition**: `if val_data is not None`

#### 3. **VBPRRetriever** (`retrieval/methods/vbpr.py`)
- ✅ **Có validation** sau mỗi epoch (dòng 177-195)
- ✅ **Early stopping** dựa trên `patience`
- ✅ **Best model tracking**: Lưu best state dựa trên `val_recall`
- **Validation metrics**: Recall@K
- **Condition**: `if val_data is not None`

#### 4. **BM3Retriever** (`retrieval/methods/bm3.py`)
- ✅ **Có validation** sau mỗi epoch (dòng 199-217)
- ✅ **Early stopping** dựa trên `patience`
- ✅ **Best model tracking**: Lưu best state dựa trên `val_recall`
- **Validation metrics**: Recall@K
- **Condition**: `if val_data is not None`

## Rerank Methods (Stage 2)

### ✅ BERT4Rec có validation

#### 1. **BERT4RecReranker** (`rerank/methods/bert4rec_reranker.py`)
- ✅ **Có validation** sau mỗi epoch (dòng 199-215)
- ✅ **Early stopping** dựa trên `patience`
- ✅ **Best model tracking**: Lưu best state dựa trên `val_recall`
- **Validation metrics**: Recall@K
- **Condition**: `if val_data is not None`

### ❌ Các rerankers khác KHÔNG có validation (vì không train)

#### 2. **VIP5Reranker** (`rerank/methods/vip5_reranker.py`)
- ❌ **Không có validation** - Chỉ load pretrained model
- ❌ **Không có training loop** - Model được load từ checkpoint
- **Lý do**: VIP5 là pretrained model, không train trong pipeline này

#### 3. **QwenReranker** (`rerank/methods/qwen_reranker.py`)
- ❌ **Không có validation** trong `fit()` method
- ⚠️ **Có thể train LLM** nếu có `train_data_for_llm`, nhưng không có validation loop
- **Lý do**: LLM training (nếu có) được thực hiện bên ngoài, không có validation trong `fit()`

#### 4. **Qwen3VLReranker** (`rerank/methods/qwen3vl_reranker.py`)
- ❌ **Không có validation** - Chỉ load pretrained model
- ❌ **Không có training loop** - Model được load từ pretrained weights
- **Lý do**: Qwen3-VL là pretrained model, không train trong pipeline này

## Validation Flow

### Retrieval Training
```python
for epoch in range(num_epochs):
    # Training loop
    ...
    
    # Validation (nếu có val_data)
    if val_data is not None:
        val_recall = self._evaluate_split(val_data, k=min(10, self.top_k))
        
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if patience and epochs_no_improve >= patience:
            break

# Load best model
if best_state is not None:
    model.load_state_dict(best_state)
```

### Rerank Training (BERT4Rec)
```python
for epoch in range(num_epochs):
    # Training loop
    ...
    
    # Validation (nếu có val_data)
    if val_data is not None:
        val_recall = self._evaluate_split(val_data, k=min(10, self.top_k))
        
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if patience and epochs_no_improve >= patience:
            break

# Load best model
if best_state is not None:
    model.load_state_dict(best_state)
```

## Validation Metrics

Tất cả validation đều sử dụng:
- **Recall@K**: K = min(10, top_k)
- **Best model selection**: Dựa trên `val_recall` cao nhất
- **Early stopping**: Dựa trên `patience` epochs không cải thiện

## Kết luận

### ✅ Đã có validation:
1. **Tất cả retrieval methods** (LRURec, MMGCN, VBPR, BM3)
2. **BERT4Rec reranker** (rerank method duy nhất có training)

### ❌ Không có validation (nhưng hợp lý):
1. **VIP5Reranker**: Pretrained model, không train
2. **QwenReranker**: LLM pretrained, training (nếu có) ở ngoài
3. **Qwen3VLReranker**: Pretrained model, không train

### 📝 Ghi chú:
- Validation được truyền qua `val_data` trong `fit_kwargs`
- Tất cả methods có training đều có early stopping
- Best model được lưu và load sau training
- Validation metrics được in ra console sau mỗi epoch

