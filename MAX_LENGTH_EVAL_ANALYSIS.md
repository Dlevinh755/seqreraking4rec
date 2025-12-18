# Phân tích Max Length trong Evaluation

## ✅ Các model đã sử dụng max_length từ config

### 1. **Qwen3VL Model** ✅
- **Location**: `rerank/models/qwen3vl.py`
- **Sử dụng**: `qwen_max_seq_length` từ config
- **Các methods**:
  - `_predict_probs_raw_image()`: Line 424-430, sử dụng `qwen_max_seq_length` từ config
  - `_predict_probs_caption()`: Line 523-533, sử dụng `qwen_max_seq_length` từ config
  - `_predict_probs_semantic_summary_vl()`: Line 614-624, sử dụng `qwen_max_seq_length` từ config
- **Cách sử dụng**: 
  ```python
  base_max_len = getattr(arg, 'qwen_max_seq_length', 2048)
  max_len = base_max_len * 2 if self.mode == "raw_image" else base_max_len
  inputs = self.processor.apply_chat_template(..., max_length=max_len)
  ```

### 2. **Qwen Reranker (Unified)** ✅
- **Location**: `rerank/methods/qwen_reranker_unified.py`
- **Sử dụng**: `qwen_max_seq_length` từ config qua `_get_max_seq_length()`
- **Các chỗ sử dụng**:
  - Token analysis: Line 736, 858, 1448
  - Training tokenization: Line 763, 897
- **Cách sử dụng**:
  ```python
  max_length = _get_max_seq_length()  # From config
  tokenized = self.qwen3vl_model.tokenizer(..., max_length=max_length)
  ```

---

## ❌ Các model CHƯA sử dụng max_length từ config

### 1. **LLM Model (Text-only)** ❌

**Location**: `rerank/models/llm.py`

**Vấn đề**:
- `predict_probs()` method (line 266) gọi `self.tokenizer(prompt, return_tensors="pt")` **KHÔNG có max_length parameter**
- Prompt có thể bị truncate theo default của tokenizer (thường là 2048 hoặc model max), không theo config

**Code hiện tại**:
```python
def predict_probs(self, prompt, num_candidates=None):
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    # ❌ KHÔNG có max_length parameter!
```

**Cần sửa**:
```python
def predict_probs(self, prompt, num_candidates=None):
    # Get max_length from config
    try:
        from config import arg
        max_length = getattr(arg, 'qwen_max_seq_length', 2048)
    except ImportError:
        max_length = 2048
    
    inputs = self.tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,  # ✅ Truncate if too long
        max_length=max_length,  # ✅ Use from config
    ).to(self.model.device)
```

### 2. **VIP5 Reranker** ❌

**Location**: `rerank/methods/vip5_reranker.py`

**Vấn đề**:
- Sử dụng `self.max_text_length` (default: 128) nhưng **KHÔNG lấy từ config**
- Không có config parameter cho VIP5 max_text_length

**Code hiện tại**:
```python
def __init__(self, ..., max_text_length: int = 128, ...):
    self.max_text_length = max_text_length  # ❌ Hardcoded default, không từ config
```

**Cần sửa**:
1. Thêm config parameter `vip5_max_text_length` vào `config.py`
2. Sửa `__init__()` để lấy từ config nếu không được cung cấp

---

## 📊 Tóm tắt

| Model | Method | Sử dụng max_length từ config? | Status |
|-------|--------|-------------------------------|--------|
| **Qwen3VL** | `_predict_probs_raw_image()` | ✅ Có | OK |
| **Qwen3VL** | `_predict_probs_caption()` | ✅ Có | OK |
| **Qwen3VL** | `_predict_probs_semantic_summary_vl()` | ✅ Có | OK |
| **Qwen Reranker** | Token analysis | ✅ Có | OK |
| **Qwen Reranker** | Training tokenization | ✅ Có | OK |
| **LLM Model** | `predict_probs()` | ❌ Không | **CẦN SỬA** |
| **VIP5 Reranker** | `__init__()` | ❌ Không | **CẦN SỬA** |

---

## 🔧 Đề xuất sửa

### Priority 1: Sửa LLM Model `predict_probs()`

**File**: `rerank/models/llm.py`

**Sửa**:
```python
def predict_probs(self, prompt, num_candidates=None):
    """Predict probabilities for candidates using numbers (1, 2, 3, ...)."""
    # Get max_length from config
    try:
        from config import arg
        max_length = getattr(arg, 'qwen_max_seq_length', 2048)
    except ImportError:
        max_length = 2048  # Default fallback
    
    inputs = self.tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,  # ✅ Truncate if too long
        max_length=max_length,  # ✅ Use from config
    ).to(self.model.device)
    
    # ... rest of the code
```

### Priority 2: Thêm config cho VIP5 và sửa `__init__()`

**File**: `config.py`

**Thêm**:
```python
parser.add_argument('--vip5_max_text_length', type=int, default=128,
                    help='Maximum text sequence length for VIP5 (default: 128, increase for longer prompts)')
```

**File**: `rerank/methods/vip5_reranker.py`

**Sửa**:
```python
def __init__(
    self,
    ...,
    max_text_length: Optional[int] = None,  # ✅ Optional, lấy từ config nếu None
    ...
):
    if max_text_length is None:
        try:
            from config import arg
            max_text_length = getattr(arg, 'vip5_max_text_length', 128)
        except ImportError:
            max_text_length = 128  # Default fallback
    
    self.max_text_length = max_text_length
```

---

## ✅ Kết luận

**Đã sử dụng max_length từ config**:
- ✅ Qwen3VL Model (tất cả modes)
- ✅ Qwen Reranker (tokenization, analysis)

**Chưa sử dụng max_length từ config**:
- ❌ LLM Model `predict_probs()` - **CẦN SỬA**
- ❌ VIP5 Reranker `max_text_length` - **CẦN SỬA**

**Khuyến nghị**: Sửa 2 vấn đề trên để đảm bảo tất cả models đều sử dụng max_length từ config khi eval.

