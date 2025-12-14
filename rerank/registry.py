"""Registry cho các rerankers (Stage 2)."""

from typing import Dict, Type

from rerank.base import BaseReranker
from rerank.methods.identity import IdentityReranker
from rerank.methods.random_reranker import RandomReranker
from rerank.methods.vip5 import VIP5Reranker


RERANKER_REGISTRY: Dict[str, Type[BaseReranker]] = {
    "identity": IdentityReranker,
    "random": RandomReranker,
    "vip5": VIP5Reranker,
    # Sau này có thể thêm: "vip4", "bert4rec", "gpt4rec", ...
}


def get_reranker_class(name: str) -> Type[BaseReranker]:
    name = name.lower()
    if name not in RERANKER_REGISTRY:
        raise KeyError(f"Unknown reranker: {name}. Available: {list(RERANKER_REGISTRY.keys())}")
    return RERANKER_REGISTRY[name]
