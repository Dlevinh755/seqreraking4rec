from typing import Dict, List, Set

from retrieval.base import BaseRetriever


class LRURecRetriever(BaseRetriever):
    """Stub LRU-based retriever (placeholder for real implementation)."""

    def __init__(self, top_k: int = 50):
        super().__init__(top_k=top_k)
        self.user_history: Dict[int, List[int]] = {}

    def fit(self, train_data: Dict[int, List[int]], **kwargs):
        self.user_history = train_data
        self.is_fitted = True

    def retrieve(self, user_id: int, exclude_items: Set[int] = None) -> List[int]:
        self._validate_fitted()
        exclude_items = exclude_items or set()
        history = self.user_history.get(user_id, [])
        # LRU: ưu tiên items xuất hiện gần đây, lọc items đã exclude
        candidates = [i for i in reversed(history) if i not in exclude_items]
        return candidates[: self.top_k]
