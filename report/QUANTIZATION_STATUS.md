# 4-bit Quantization Status Report

## 📊 Tổng quan

Báo cáo này kiểm tra xem các models đã được load với 4-bit quantization chưa.

## ✅ Models đã có 4-bit quantization (Mặc định) - TẤT CẢ UNSLOTH MODELS

### 1. LLM Model (`rerank/models/llm.py`)
- **Status**: ✅ **Đã có 4-bit mặc định** (Unsloth)
- **Location**: Line 110
- **Code**:
  ```python
  print(f"Loading LLM model with 4-bit quantization: {self.model_name}")
  self.model, self.tokenizer = FastLanguageModel.from_pretrained(
      model_name = self.model_name,
      max_seq_length = 2048,
      dtype = torch.float16,
      load_in_4bit = True,  # ✅ 4-bit mặc định cho tất cả Unsloth models
  )
  ```
- **Model**: `Qwen/Qwen3-0.6B` (via Unsloth)
- **Usage**: Qwen reranker inference
- **Note**: Tất cả models load bằng Unsloth đều có 4-bit mặc định

### 2. Qwen3-VL Text Model - `semantic_summary_small` mode (`rerank/models/qwen3vl.py`)
- **Status**: ✅ **Đã có 4-bit mặc định** (Unsloth)
- **Location**: Line 129
- **Code**:
  ```python
  print(f"Loading Qwen text model with 4-bit quantization: {self.model_name}")
  self.model, self.tokenizer = FastLanguageModel.from_pretrained(
      model_name=self.model_name,  # "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
      max_seq_length=2048,
      dtype=torch.float16,
      load_in_4bit=True,  # ✅ 4-bit mặc định cho tất cả Unsloth models
  )
  ```
- **Model**: `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`
- **Usage**: Qwen3-VL reranker với `semantic_summary_small` mode
- **Note**: Tất cả models load bằng Unsloth đều có 4-bit mặc định

## ❌ Models chưa có 4-bit quantization mặc định

### 1. Qwen3-VL Model cho Semantic Summary Generation (`dataset/qwen3vl_semantic_summary.py`)
- **Status**: ❌ **Chưa có 4-bit mặc định** (chỉ khi `--use_quantization` flag được set)
- **Location**: Lines 67-87
- **Current Code**:
  ```python
  quantization_config = None
  if use_quantization and device.type == "cuda":  # ❌ Chỉ khi flag được set
      try:
          from transformers import BitsAndBytesConfig
          quantization_config = BitsAndBytesConfig(
              load_in_4bit=True,
              ...
          )
  ```
- **Model**: `unsloth/Qwen3-VL-2B-Instruct`
- **Usage**: Generate semantic summaries trong `data_prepare.py`
- **Note**: Cần set `--use_quantization` flag để enable 4-bit

### 2. Qwen3-VL Model cho Reranking - VL modes (`rerank/models/qwen3vl.py`)
- **Status**: ❌ **Chưa có 4-bit mặc định**
- **Location**: Lines 96-119
- **Current Code**:
  ```python
  self.model = Qwen3VLForConditionalGeneration.from_pretrained(
      self.model_name,  # "unsloth/Qwen3-VL-2B-Instruct"
      dtype="auto" if self.device.type == "cuda" else torch.float32,
      device_map="auto" if self.device.type == "cuda" else None,
      trust_remote_code=True,
      # ❌ Không có quantization_config
  )
  ```
- **Models**: 
  - `unsloth/Qwen3-VL-2B-Instruct` (cho `raw_image`, `caption`, `semantic_summary` modes)
- **Usage**: Qwen3-VL reranker với các VL modes

## 🔧 Khuyến nghị

### Option 1: Enable 4-bit mặc định cho Semantic Summary Generation
- **Lý do**: Semantic summary generation không cần độ chính xác cao, 4-bit sẽ tiết kiệm memory đáng kể
- **Action**: Set `use_quantization=True` mặc định trong `_load_qwen3vl_model()`

### Option 2: Giữ nguyên (Optional với flag)
- **Lý do**: Cho phép user lựa chọn giữa speed/memory và accuracy
- **Action**: Giữ nguyên, user cần set `--use_quantization` flag

### Option 3: Enable 4-bit cho Qwen3-VL Reranker
- **Lý do**: Reranking cần độ chính xác cao hơn, nhưng 4-bit vẫn có thể acceptable
- **Action**: Thêm quantization support cho VL modes trong `rerank/models/qwen3vl.py`

## 📝 Summary

| Model | Location | 4-bit Status | Notes |
|-------|----------|---------------|-------|
| **LLM (Qwen reranker)** | `rerank/models/llm.py` | ✅ **Mặc định** | **Unsloth - 4-bit enabled** |
| **Qwen3-VL Text (small)** | `rerank/models/qwen3vl.py` | ✅ **Mặc định** | **Unsloth - 4-bit enabled** |
| Qwen3-VL (semantic summary) | `dataset/qwen3vl_semantic_summary.py` | ⚠️ Optional | Cần `--use_quantization` flag (không dùng Unsloth) |
| Qwen3-VL (reranker VL modes) | `rerank/models/qwen3vl.py` | ❌ Chưa có | Không dùng Unsloth (dùng transformers) |

## ✅ Kết luận

**TẤT CẢ MODELS LOAD BẰNG UNSLOTH ĐÃ CÓ 4-BIT QUANTIZATION MẶC ĐỊNH**

- ✅ `rerank/models/llm.py`: FastLanguageModel.from_pretrained() với `load_in_4bit=True`
- ✅ `rerank/models/qwen3vl.py`: FastLanguageModel.from_pretrained() với `load_in_4bit=True`

Các models không dùng Unsloth (Qwen3-VL với transformers) có thể enable 4-bit qua `--use_quantization` flag.

