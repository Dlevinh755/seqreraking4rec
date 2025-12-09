# Project Report: Two-Stage Recommendation Pipeline

## 1. Tổng quan

Dự án được tổ chức xoay quanh **pipeline 2 stage**:

1. **Stage 1 – Retrieval (Candidate Generation)**
   - Mục tiêu: Lấy ra **top-K items tiềm năng** cho mỗi user với **tốc độ cao**.
   - Ví dụ phương pháp: `LRURec`, `MMGCN`, `VBPR`, ...

2. **Stage 2 – Rerank (Re-Scoring)**
   - Mục tiêu: Đánh giá lại các candidates từ Stage 1 một cách **chính xác hơn**.
   - Ví dụ phương pháp: `VIP4`, `BERT4Rec`, `GPT4Rec`, ...

Hai stage được thiết kế **module hóa**, có thể **thay thế / kết hợp nhiều phương pháp** chỉ bằng cách đổi config, không cần sửa code core.

---

## 2. Cấu trúc thư mục chính

```text
inprocessing/
├── config.py                # Cấu hình chung
├── data_prepare.py          # Chuẩn bị & preprocess dữ liệu (KHÔNG chứa logic pipeline)
├── datasets/                # Định nghĩa dataset & preprocessing
│
├── retrieval/               # Stage 1: Candidate Generation
│   ├── base.py              # BaseRetriever: interface chung
│   ├── __init__.py          # Export các retrievers (nếu cần)
│   ├── registry.py          # Map tên → class retriever
│   └── methods/             # Các thuật toán retrieval
│       ├── lrurec.py        # LRURecRetriever (stub)
│       ├── mmgcn.py         # MMGCNRetriever (stub)
│       └── vbpr.py          # VBPRRetriever (stub)
│
├── rerank/                  # Stage 2: Reranking
│   ├── base.py              # BaseReranker: interface chung
│   ├── registry.py          # Map tên → class reranker
│   └── methods/
│       ├── identity.py      # IdentityReranker: giữ nguyên thứ tự
│       └── random_reranker.py # RandomReranker: shuffle candidates
│
├── pipelines/               # Nơi lắp ráp retrieval + rerank
│   ├── base.py              # TwoStagePipeline + config dataclasses
│   ├── run_pipeline.py      # Entry point demo chạy pipeline
│   └── configs/             # (Dành cho YAML configs trong tương lai)
│
├── tools/                   # Scripts hỗ trợ (không thuộc pipeline chính)
│   ├── clean_preprocessed.py
│   ├── inspect_pickle.py
│   ├── test_filtering.py
│   └── test_download_images.py
│
└── PROJECT_REPORT.md        # (file này) Mô tả cấu trúc & hoạt động
```

**Lưu ý quan trọng:**
- `data_prepare.py` chỉ phụ trách **chuẩn bị dữ liệu** (download, preprocess, lưu `.pkl`).
- Logic pipeline 2-stage được tách riêng trong `retrieval/`, `rerank/`, `pipelines/` để dễ bảo trì và mở rộng.

---

## 3. Stage 1 – Retrieval (Candidate Generation)

### 3.1. Interface chung: `BaseRetriever`

File: `retrieval/base.py`

```python
class BaseRetriever(ABC):
    def __init__(self, top_k: int = 50):
        self.top_k = top_k
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_data: Dict[int, List[int]], **kwargs):
        """Train model trên dict {user_id: [item_ids]}"""

    @abstractmethod
    def retrieve(self, user_id: int, exclude_items: Set[int] = None) -> List[int]:
        """Trả về top-K item_ids cho 1 user."""
```

Tất cả các thuật toán retrieval (LRURec, MMGCN, VBPR, ...) **phải kế thừa** class này và implement `fit()` + `retrieve()`.

### 3.2. Các phương pháp ví dụ (stub)

- `retrieval/methods/lrurec.py` → `LRURecRetriever`
- `retrieval/methods/mmgcn.py` → `MMGCNRetriever`
- `retrieval/methods/vbpr.py` → `VBPRRetriever`

Hiện tại đây là **stub**: chỉ lưu user_history và lấy candidates từ lịch sử. Sau này bạn có thể thay thế bằng implementation thật mà **không cần đổi pipeline**.

### 3.3. Registry cho retrieval

File: `retrieval/registry.py`

```python
RETRIEVER_REGISTRY = {
    "lrurec": LRURecRetriever,
    "mmgcn": MMGCNRetriever,
    "vbpr": VBPRRetriever,
}

get_retriever_class(name: str) -> Type[BaseRetriever]
```

→ Cho phép chọn thuật toán retrieval bằng **tên string** (config/CLI), không cần import thủ công.

---

## 4. Stage 2 – Rerank (Re-Scoring)

### 4.1. Interface chung: `BaseReranker`

File: `rerank/base.py`

```python
class BaseReranker(ABC):
    def __init__(self, top_k: int = 50):
        self.top_k = top_k
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_data: Dict[int, List[int]], **kwargs):
        ...

    @abstractmethod
    def rerank(self, user_id: int, candidates: List[int], **kwargs) -> List[Tuple[int, float]]:
        ...
```

Reranker nhận **danh sách candidates từ Stage 1** và trả về `(item_id, score)` đã sắp xếp.

### 4.2. Các reranker ví dụ

- `IdentityReranker` (`rerank/methods/identity.py`): giữ nguyên thứ tự, hữu ích để debug.
- `RandomReranker` (`rerank/methods/random_reranker.py`): shuffle, dùng làm baseline.

### 4.3. Registry cho rerankers

File: `rerank/registry.py`

```python
RERANKER_REGISTRY = {
    "identity": IdentityReranker,
    "random": RandomReranker,
}
```

Sau này có thể thêm:
- `"vip4": VIP4Reranker`
- `"bert4rec": BERT4RecReranker`
- `"gpt4rec": GPT4RecReranker`

---

## 5. Pipelines – Kết nối Stage 1 + Stage 2

### 5.1. Cấu hình pipeline bằng dataclass

File: `pipelines/base.py`

```python
@dataclass
class RetrievalConfig:
    method: str = "lrurec"
    top_k: int = 200

@dataclass
class RerankConfig:
    method: str = "identity"
    top_k: int = 50

@dataclass
class PipelineConfig:
    retrieval: RetrievalConfig
    rerank: RerankConfig
```

→ Dễ dàng map từ YAML / argparse sang object cấu hình.

### 5.2. `TwoStagePipeline` (hỗ trợ cả retrieval-only)

```python
class TwoStagePipeline:
    def __init__(self, cfg: PipelineConfig):
        # Stage 1 luôn bắt buộc
        RetrieverCls = get_retriever_class(cfg.retrieval.method)
        self.retriever = RetrieverCls(top_k=cfg.retrieval.top_k)

        # Stage 2 có thể tắt (retrieval-only)
        rerank_method = (cfg.rerank.method or "").lower()
        if rerank_method in ("", "none"):
            self.reranker = None
        else:
            RerankerCls = get_reranker_class(cfg.rerank.method)
            self.reranker = RerankerCls(top_k=cfg.rerank.top_k)

    def fit(self, train_data: Dict[int, List[int]]) -> None:
        self.retriever.fit(train_data)
        if self.reranker is not None:
            self.reranker.fit(train_data)

    def recommend(self, user_id: int) -> List[int]:
        candidates = self.retriever.retrieve(user_id)
        if not candidates:
            return []
        if self.reranker is None:
            return candidates
        scored = self.reranker.rerank(user_id, candidates)
        return [item_id for item_id, _ in scored]
```

→ Stage 1 và Stage 2 được **hoàn toàn tách biệt**, chỉ giao tiếp qua một API đơn giản:
- Stage 1: `retrieve(user_id) -> List[item_id]`
- Stage 2 (tuỳ chọn): `rerank(user_id, candidates) -> List[(item_id, score)]`

Quan trọng: **một pipeline đầy đủ có thể chỉ dùng Stage 1** (retrieval-only). Trong trường hợp đó:
- Stage 2 bị tắt (`rerank_method = "none"`)
- Hàm `recommend()` trả về trực tiếp danh sách từ `retrieve()`.

### 5.3. Entry point demo: `pipelines/run_pipeline.py`

- Đọc arguments từ CLI (retrieval_method, rerank_method, top_k, ...)
- Tạo `PipelineConfig` → `TwoStagePipeline`
- Dùng **toy train_data** `{user_id: [item_ids]}` để demo

Ví dụ chạy:

```bash
python -m pipelines.run_pipeline \
  --retrieval_method lrurec \
  --retrieval_top_k 200 \
    --rerank_method identity \
  --rerank_top_k 50
```

Nếu muốn **chỉ chạy Stage 1** (retrieval-only):

```bash
python -m pipelines.run_pipeline \
    --retrieval_method lrurec \
    --retrieval_top_k 200 \
    --rerank_method none
```

Sau này bạn có thể:
- Thay `train_data` toy bằng dữ liệu thật load từ `dataset.pkl`
- Thêm evaluation (NDCG, Recall@K, ...)
- Hỗ trợ đọc YAML config trong `pipelines/configs/`.

---

## 6. Cách mở rộng thêm phương pháp mới

### 6.1. Thêm retrieval mới (ví dụ: `MyCoolGCN`)

1. Tạo file: `retrieval/methods/my_cool_gcn.py`
2. Kế thừa `BaseRetriever` và implement `fit()` + `retrieve()`
3. Đăng ký trong `retrieval/registry.py`:

```python
from retrieval.methods.my_cool_gcn import MyCoolGCNRetriever

RETRIEVER_REGISTRY["my_cool_gcn"] = MyCoolGCNRetriever
```

4. Dùng trong pipeline:

```bash
python -m pipelines.run_pipeline --retrieval_method my_cool_gcn
```

### 6.2. Thêm reranker mới (ví dụ: `VIP4`)

1. Tạo file: `rerank/methods/vip4.py`
2. Kế thừa `BaseReranker`
3. Đăng ký trong `rerank/registry.py`:

```python
from rerank.methods.vip4 import VIP4Reranker
RERANKER_REGISTRY["vip4"] = VIP4Reranker
```

4. Dùng trong pipeline:

```bash
python -m pipelines.run_pipeline --rerank_method vip4
```

---

## 7. Quan hệ giữa `data_prepare.py` và pipeline

- `data_prepare.py`:
  - Download raw data
  - Lọc users/items
  - Densify id
  - Split train/val/test
  - Lưu vào `dataset.pkl`

- **Pipeline (retrieval/rerank/pipelines)**:
  - Giả định rằng đã có dữ liệu ở dạng `{user_id: [item_ids]}`
  - Chỉ tập trung vào **modeling & inference**, không dính đến việc download/preprocess.

→ Điều này đảm bảo **phân tách rõ ràng**:
- **Data layer**: `datasets/`, `data_prepare.py`
- **Model layer**: `retrieval/`, `rerank/`, `pipelines/`

---

## 8. Kết luận

Thiết kế hiện tại cho phép bạn:

- Dễ dàng **thêm / thay đổi** thuật toán ở từng stage.
- Kết hợp nhiều phương pháp retrieval/rerank chỉ bằng **config/CLI**.
- Giữ cho `data_prepare.py` **sạch**, chỉ dành cho chuẩn bị dữ liệu.
- Mở rộng sang các mô hình mạnh hơn (VIP4, BERT4Rec, GPT4Rec, ...) mà không cần thay đổi cấu trúc pipeline.

Trong các bước tiếp theo, bạn có thể:
- Kết nối `TwoStagePipeline` với dữ liệu thật từ `dataset.pkl`.
- Tạo nhiều cấu hình YAML cho các combo như `MMGCN + BERT4Rec`, `VBPR + GPT4Rec`, ...

---

### 8.1. Huấn luyện & đánh giá Stage 1 (LRURec nhẹ)

Để có một baseline **nhẹ, không cần train nặng**, project bổ sung script:

- `retrieval/train_lrurec.py`:
  - Đọc cấu hình từ `config.py` (ví dụ: `dataset_code`, `min_uc`, `min_sc`, `seed`).
  - Gọi `dataset_factory(arg).load_dataset()` để load `dataset.pkl` đã được chuẩn hóa.
  - Khởi tạo `LRURecRetriever` (heuristic): chỉ lưu lịch sử người dùng và recommend lại các item gần nhất theo thứ tự mới → cũ.
  - Chạy `fit(train)` (rất nhẹ, không có backprop hay optimizer).
  - Đánh giá trên `val` và `test` bằng `Recall@10`, `NDCG@10` và in kết quả ra màn hình.

Mục tiêu của script này là:

- Cung cấp **Stage 1 pipeline hoàn chỉnh** (load dataset → fit → eval) mà **không đòi hỏi GPU mạnh**.
- Làm mẫu cho các retriever khác: thay `LRURecRetriever` bằng model mới nhưng giữ nguyên khung train/eval.
- Về sau có thể mở rộng để xuất thêm file `retrieved.pkl` (pool ứng viên cho Stage 2) vào thư mục `experiments/retrieval/...` nhưng vẫn giữ độ nhẹ.

---

### 8.2. Module đánh giá offline: `evaluation/offline_eval.py`

Project hiện có module `evaluation/` với các thành phần chính:

- `evaluation/metrics.py`:
  - Cài đặt các hàm metric cơ bản như `recall_at_k`, `ndcg_at_k` trên từng user.

- `evaluation/offline_eval.py`:
  - Đọc `dataset.pkl` giống như `data_prepare.py` và `train_lrurec.py`.
  - Kết nối với `TwoStagePipeline` để chạy **offline evaluation** theo 3 mode rõ ràng:

1. `mode = "retrieval"` (Stage 1 only):
    - Chỉ dùng retriever (ví dụ: `LRURecRetriever`).
    - Đo hiệu năng **thuần retrieval** (Recall@K, NDCG@K) mà không có reranker.

2. `mode = "full"` (Stage 1 + Stage 2):
    - Chạy đầy đủ pipeline: retriever → reranker.
    - Đây là metric **end-to-end thực tế**, phản ánh đúng hệ thống deploy.

3. `mode = "rerank_only"` (Stage 2 oracle-in-pool):
    - Dùng retriever để sinh một pool ứng viên cho mỗi user.
    - **Tiêm thêm ground-truth item** của user vào pool để chắc chắn nó luôn có mặt.
    - Reranker (ví dụ: `IdentityReranker`) chỉ việc sắp xếp lại pool này.
    - Đo Recall@K, NDCG@K trong bối cảnh **oracle-in-pool** → đây là **upper bound** cho khả năng rerank, không phải metric end-to-end.

Lưu ý quan trọng:

- Với `mode = "rerank_only"`, việc luôn tiêm ground-truth vào pool + pool nhỏ có thể làm Recall@K nhìn rất cao (ví dụ ~0.8), kể cả với reranker đơn giản. Điều này **không sai**, nhưng cần hiểu đây chỉ là **giới hạn trên** cho Stage 2, chứ không phản ánh toàn hệ thống.
- Khi so sánh mô hình, nên nhìn đồng thời:
  - `retrieval` → chất lượng Stage 1.
  - `rerank_only` → tiềm năng Stage 2 nếu pool đủ tốt.
  - `full` → chất lượng pipeline thực tế.

---

### 8.3. Vai trò của `LlamaRec/` trong project

Thư mục `LlamaRec/` được giữ lại như **một project mẫu/full pipeline nặng**:

- Có retriever neural LRURec riêng, dataloader riêng, trainer riêng.
- Có LLM ranker (LoRA, quantization) với trainer và dataloader phức tạp.

Tuy nhiên:

- Máy hiện tại **không đủ mạnh** để train full pipeline `LlamaRec` một cách thực tế.
- Vì vậy, `LlamaRec/` chỉ đóng vai trò **tham khảo kiến trúc, format data (`retrieved.pkl`) và cách log metric**, không được import chéo sang core project (`retrieval/`, `rerank/`, `evaluation/`).
- Trong core project, mọi thứ (metrics, trainer, pipeline) đều được **tự cài đặt lại nhẹ nhàng**, tránh phụ thuộc vào `LlamaRec`.

---

### 8.4. Kế hoạch tiếp theo (thực tế, nhẹ)

Dựa trên scaffolding hiện có, các bước tiếp theo khả thi và nhẹ bao gồm:

1. **Hoàn thiện xuất `retrieved.pkl` cho Stage 1 nhẹ**:
    - Mở rộng `retrieval/train_lrurec.py` để, ngoài in metric, còn ghi ra `retrieved.pkl` (prob hoặc score cho từng ứng viên + nhãn) vào `experiments/retrieval/<method>/<dataset>/seed<seed>/`.
    - Format có thể học theo `LlamaRec`, nhưng giữ mức compute nhỏ.

2. **Viết một trainer Stage 2 dummy/lightweight** (tùy chọn):
    - Ví dụ: script `rerank/train_identity.py` hoặc `rerank/train_random.py`.
    - Đọc `dataset.pkl` + `retrieved.pkl`, xây pool ứng viên per-user, chạy reranker đơn giản, đo metric và log ra file.

3. **Thêm script tổng hợp kết quả experiment**:
    - Tạo một script nhỏ (ví dụ: `tools/summarize_experiments.py`) đọc các log/JSON/CSV trong `experiments/`.
    - Ghi một bảng `experiments/summary.csv` với các cột như: `stage1_model`, `stage2_model`, `dataset`, `K`, `recall@K`, `ndcg@K`, `seed`, `notes`.

Các bước trên đều **không đòi hỏi GPU mạnh**, nhưng giúp project tiến rất gần tới một framework experiment hoàn chỉnh cho 2-stage recommendation.
