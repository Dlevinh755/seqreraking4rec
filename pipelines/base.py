"""Base pipeline definitions for composing retrieval (Stage 1) and rerank (Stage 2)."""

from dataclasses import dataclass
from typing import Any, Dict, List

from retrieval.registry import get_retriever_class
from rerank.registry import get_reranker_class


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


class TwoStagePipeline:
    """Hai stage: Stage 1 retrieval → Stage 2 rerank.

    Đây chỉ là khung đơn giản, dùng dict {user_id: [item_ids]} làm dữ liệu.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        # Stage 1 luôn bắt buộc
        RetrieverCls = get_retriever_class(cfg.retrieval.method)
        self.retriever = RetrieverCls(top_k=cfg.retrieval.top_k)

        # Stage 2 có thể tắt hoàn toàn (retrieval-only pipeline)
        rerank_method = (cfg.rerank.method or "").lower()
        if rerank_method in ("", "none"):
            self.reranker = None
        else:
            RerankerCls = get_reranker_class(cfg.rerank.method)
            self.reranker = RerankerCls(top_k=cfg.rerank.top_k)

    def fit(self, train_data: Dict[int, List[int]]) -> None:
        """Fit retriever và (nếu có) reranker.

        Nếu reranker bị tắt (method = "none"), chỉ train Stage 1.
        """
        self.retriever.fit(train_data)
        if self.reranker is not None:
            self.reranker.fit(train_data)

    def recommend(self, user_id: int) -> List[int]:
        """Chạy pipeline: có thể chỉ Stage 1 hoặc cả hai stage.

        - Nếu không có reranker → trả về candidates từ Stage 1.
        - Nếu có reranker → retrieve → rerank.
        """

        # Stage 1: lấy candidates nhanh (bắt buộc)
        candidates = self.retriever.retrieve(user_id)
        if not candidates:
            return []

        # Nếu không cấu hình Stage 2 → trả thẳng kết quả retrieval
        if self.reranker is None:
            return candidates

        # Stage 2: rerank chính xác hơn
        scored = self.reranker.rerank(user_id, candidates)
        # Trả về chỉ item_id
        return [item_id for item_id, _ in scored]
