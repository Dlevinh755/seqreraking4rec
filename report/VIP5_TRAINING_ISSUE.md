# Vấn đề VIP5Reranker không Training

## Tại sao VIP5Reranker không training?

### 1. **Thiết kế ban đầu**
VIP5Reranker hiện tại được implement để:
- **Load pretrained checkpoint** (nếu có `checkpoint_path`)
- **Hoặc khởi tạo từ T5 backbone** (nếu không có checkpoint)
- **KHÔNG có training loop** trong `fit()` method

### 2. **Code hiện tại** (`rerank/methods/vip5_reranker.py`)

```python
def fit(self, train_data, **kwargs):
    # ... Load CLIP embeddings ...
    # ... Initialize tokenizer ...
    
    if self.checkpoint_path and Path(self.checkpoint_path).exists():
        # Load pretrained checkpoint
        self.model = VIP5(config)
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        # Initialize from T5 backbone (NO TRAINING!)
        self.model = VIP5(config)
        pretrained = T5ForConditionalGeneration.from_pretrained(self.backbone)
        # Copy compatible weights
        self.model.shared.load_state_dict(pretrained.shared.state_dict())
        self.model.decoder.load_state_dict(pretrained.decoder.state_dict(), strict=False)
    
    self.model.eval()  # Set to eval mode, no training
    self.is_fitted = True
```

**Vấn đề**: Model chỉ được khởi tạo từ T5 pretrained weights, **KHÔNG được fine-tune** trên dataset hiện tại.

## Hậu quả nếu không training

### ❌ **Model sẽ KHÔNG hoạt động tốt** trên data hiện tại vì:

1. **T5 backbone không hiểu recommendation tasks**
   - T5 được train trên text generation tasks (translation, summarization)
   - Không hiểu về user-item interactions, sequential patterns
   - Không biết cách encode visual features cho recommendation

2. **Không có domain adaptation**
   - Model không học được patterns đặc thù của dataset (beauty, toys, etc.)
   - Không adapt với item vocabulary và user behavior của dataset

3. **Visual features không được integrate đúng cách**
   - VIP5 cần train để học cách kết hợp visual và textual features
   - T5 backbone không biết cách xử lý visual features cho recommendation

4. **Reranking logic không được optimize**
   - Model không học được cách rerank candidates dựa trên user history
   - Chỉ dùng encoder output norm làm score (rất đơn giản, không tối ưu)

## VIP5 Training Requirements

### 1. **Multiple Tasks**
VIP5 được thiết kế để train với 3 loại tasks:

- **Sequential tasks** (A-1 đến A-8):
  - A-1: "Given the following purchase history of user_{} : \n {} \n predict next possible item to be purchased by the user ?"
  - A-2: "Given the following purchase history of user_{} : \n {} \n predict the next item that user_{} will interact with ?"
  - ... (8 templates)

- **Direct tasks** (B-1 đến B-7):
  - B-1: "Will user_{} interact with item_{} ?"
  - ... (7 templates)

- **Explanation tasks** (C-1 đến C-11):
  - C-1: "Generate an explanation for why user_{} interacts with item_{} ?"
  - ... (11 templates)

### 2. **Data Format**
VIP5 cần data format đặc biệt:
- Sequential data: `user_id item1 item2 item3 ...`
- Explanation data: User reviews với ratings và explanations
- Visual features: CLIP embeddings cho mỗi item
- Text features: Item titles, descriptions

### 3. **Training Process**
- Multi-task learning với loss weights
- Parameter-efficient fine-tuning (adapters)
- Validation sau mỗi epoch (sau epoch 4)
- Early stopping dựa trên validation loss

## Giải pháp

### Option 1: **Thêm Training cho VIP5Reranker** (Recommended)

Cần implement training loop trong `VIP5Reranker.fit()`:

1. **Prepare training data** từ `train_data`:
   - Convert user-item sequences thành VIP5 task format
   - Generate multiple task templates (sequential, direct)
   - Prepare visual and text features

2. **Training loop**:
   - Multi-task loss (sequential, direct)
   - Optimizer và learning rate scheduler
   - Validation sau mỗi epoch
   - Early stopping
   - Best model tracking

3. **Integration**:
   - Sử dụng `VIP5Tuning` class từ `retrieval/vip5_temp/src/model.py`
   - Implement `train_step()` và `valid_step()`
   - Support adapters cho parameter-efficient training

### Option 2: **Sử dụng Pretrained Checkpoint**

Nếu có pretrained checkpoint trên dataset tương tự:
- Load checkpoint trong `fit()`
- Model sẽ hoạt động tốt hơn (nhưng vẫn không tối ưu cho dataset hiện tại)

### Option 3: **Skip VIP5 nếu không có training**

Nếu không thể train VIP5:
- **Không nên sử dụng VIP5Reranker** trong evaluation
- Model sẽ cho kết quả rất kém (chỉ dùng T5 backbone)
- Không fair comparison với các methods khác

## Recommendation

**Nên thêm training cho VIP5Reranker** để:
1. ✅ Model được fine-tune trên dataset hiện tại
2. ✅ Fair comparison với các rerank methods khác (BERT4Rec, Qwen, etc.)
3. ✅ Model hoạt động tốt hơn nhiều so với chỉ dùng T5 backbone
4. ✅ Tận dụng được visual và textual features đúng cách

**Nếu không thể train**, nên:
- ⚠️ Loại VIP5 khỏi comparison
- ⚠️ Hoặc chỉ dùng nếu có pretrained checkpoint tốt
- ⚠️ Document rõ ràng rằng VIP5 không được train

## Implementation Plan

1. **Tạo VIP5 training data preparation**:
   - Convert `train_data` (Dict[int, List[int]]) thành VIP5 format
   - Generate task templates (sequential, direct)
   - Prepare batches với visual/text features

2. **Implement training loop**:
   - Multi-task loss calculation
   - Optimizer setup
   - Validation và early stopping
   - Best model saving

3. **Integration với pipeline**:
   - Support `num_epochs`, `batch_size`, `lr`, `patience` từ config
   - Validation với `val_data`
   - Save/load checkpoints

4. **Testing**:
   - Verify training loss decreases
   - Verify validation metrics improve
   - Compare với pretrained checkpoint (nếu có)

