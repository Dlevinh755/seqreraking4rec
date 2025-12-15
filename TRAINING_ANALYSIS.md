# Phân tích Training Process

## Vấn đề hiện tại

### 1. **train_pipeline.py** - Training bị dính liền
- Trong `pipelines/base.py`, method `fit()`:
  ```python
  def fit(self, train_data, **kwargs):
      # Fit Stage 1
      self.retriever.fit(train_data, **retriever_kwargs)  # Dòng 117
      
      # Fit Stage 2 (if enabled)
      if self.reranker is not None:
          self.reranker.fit(train_data, **reranker_kwargs)  # Dòng 122
  ```
- **VấN ĐỀ**: Rerank training bị dính liền với retrieval - không thể train rerank riêng mà không train retrieval trước

### 2. **train_rerank.py** - Chỉ cho Qwen LLM
- Script riêng nhưng chỉ hỗ trợ Qwen LLM
- Không sử dụng BaseReranker interface
- Không áp dụng cho các rerankers khác (VIP5, BERT4Rec, Qwen3-VL)

### 3. **evaluation/offline_eval.py** - Chỉ cho evaluation
- `run_rerank_only()` fit cả retriever và reranker, nhưng chỉ để evaluate
- Không phải script training riêng

## Giải pháp đề xuất

Cần tạo script mới `scripts/train_rerank_standalone.py` để:
- Train rerank riêng lẻ, không cần train retrieval
- Hỗ trợ tất cả rerankers (Qwen, Qwen3-VL, VIP5, BERT4Rec)
- Có thể load retrieval model đã train sẵn (nếu cần candidates)
- Hoặc sử dụng ground_truth mode (không cần retrieval)

