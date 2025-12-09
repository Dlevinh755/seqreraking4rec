from typing import Dict, List, Tuple, Any

from rerank.base import BaseReranker


class IdentityReranker(BaseReranker):
    """Reranker đơn giản: giữ nguyên thứ tự candidates.

    Hữu ích để debug pipeline hoặc làm baseline.
    """

    def fit(self, train_data: Dict[int, List[int]], **kwargs: Any) -> None:
        # Identity không cần train
        self.is_fitted = True

    def rerank(self, user_id: int, candidates: List[int], **kwargs: Any) -> List[Tuple[int, float]]:
        self._validate_fitted()
        # Gán score giảm dần theo vị trí hiện tại
        scored = [(item_id, float(len(candidates) - idx)) for idx, item_id in enumerate(candidates)]
        return scored[: self.top_k]
