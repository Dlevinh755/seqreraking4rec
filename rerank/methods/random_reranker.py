import random
from typing import Dict, List, Tuple, Any

from rerank.base import BaseReranker


class RandomReranker(BaseReranker):
    """Reranker ngẫu nhiên - dùng để so sánh baseline.
    """

    def fit(self, train_data: Dict[int, List[int]], **kwargs: Any) -> None:
        # Không cần train
        self.is_fitted = True

    def rerank(self, user_id: int, candidates: List[int], **kwargs: Any) -> List[Tuple[int, float]]:
        self._validate_fitted()
        shuffled = list(candidates)
        random.shuffle(shuffled)
        # Score = vị trí ngược lại (chỉ để có số)
        scored = [(item_id, float(len(shuffled) - idx)) for idx, item_id in enumerate(shuffled)]
        return scored[: self.top_k]
