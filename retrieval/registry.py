"""Registry cho các retrieval methods (Stage 1)."""

from typing import Dict, Type

from retrieval.base import BaseRetriever
from retrieval.methods.lrurec import LRURecRetriever
from retrieval.methods.mmgcn import MMGCNRetriever
from retrieval.methods.vbpr import VBPRRetriever


RETRIEVER_REGISTRY: Dict[str, Type[BaseRetriever]] = {
    "lrurec": LRURecRetriever,
    "mmgcn": MMGCNRetriever,
    "vbpr": VBPRRetriever,
}


def get_retriever_class(name: str) -> Type[BaseRetriever]:
    name = name.lower()
    if name not in RETRIEVER_REGISTRY:
        raise KeyError(f"Unknown retriever: {name}. Available: {list(RETRIEVER_REGISTRY.keys())}")
    return RETRIEVER_REGISTRY[name]
