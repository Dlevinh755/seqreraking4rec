from typing import Dict, List, Set

from retrieval.base import BaseRetriever


class MMGCNRetriever(BaseRetriever):
    """Stub MMGCN retriever (placeholder, only interface)."""

    def __init__(self, top_k: int = 50):
        super().__init__(top_k=top_k)
        self.user_history: Dict[int, List[int]] = {}

    def fit(self, train_data: Dict[int, List[int]], **kwargs):
        # TODO: implement real MMGCN training
        self.user_history = train_data
        self.is_fitted = True

    def retrieve(self, user_id: int, exclude_items: Set[int] = None) -> List[int]:
        self._validate_fitted()
        # TODO: replace bằng scores từ MMGCN
        exclude_items = exclude_items or set()
        history = self.user_history.get(user_id, [])
        candidates = [i for i in history if i not in exclude_items]
        return candidates[: self.top_k]
