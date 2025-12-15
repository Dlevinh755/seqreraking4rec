"""Base classes for two-stage recommendation pipeline.

This module defines the configuration dataclasses and TwoStagePipeline class
that combines retrieval (Stage 1) and reranking (Stage 2) methods.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RetrievalConfig:
    """Configuration for Stage 1 retrieval.
    
    Note:
        - `top_k` determines how many candidates are passed to Stage 2
        - If using Qwen reranker, recommend setting `top_k <= 20` (LLM limit)
        - Default 200 is suitable for other rerankers (VIP5, identity, etc.)
    """
    method: str = "lrurec"
    top_k: int = 200


@dataclass
class RerankConfig:
    """Configuration for Stage 2 reranking.
    
    Note:
        - `top_k` is the final number of recommendations returned
        - For Qwen reranker: input candidates are limited to 20 (A-T letters)
        - If retrieval stage returns > 20 candidates, Qwen will truncate to 20
    """
    method: str = "identity"
    top_k: int = 50


@dataclass
class PipelineConfig:
    """Configuration for two-stage pipeline."""
    retrieval: RetrievalConfig
    rerank: RerankConfig


class TwoStagePipeline:
    """Two-stage recommendation pipeline.
    
    Combines retrieval (Stage 1) and reranking (Stage 2) methods.
    Stage 2 is optional - can run retrieval-only mode.
    """
    
    def __init__(self, cfg: PipelineConfig):
        """
        Args:
            cfg: Pipeline configuration
        """
        # Stage 1: Always required
        from retrieval.registry import get_retriever_class
        
        RetrieverCls = get_retriever_class(cfg.retrieval.method)
        self.retriever = RetrieverCls(top_k=cfg.retrieval.top_k)
        
        # Stage 2: Optional (can be disabled for retrieval-only)
        rerank_method = (cfg.rerank.method or "").lower()
        if rerank_method in ("", "none", "identity"):
            self.reranker = None
        else:
            from rerank.registry import get_reranker_class
            
            RerankerCls = get_reranker_class(cfg.rerank.method)
            self.reranker = RerankerCls(top_k=cfg.rerank.top_k)
        
        self.cfg = cfg
    
    def fit(self, train_data: Dict[int, List[int]], **kwargs) -> None:
        """Train both stages on training data.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            **kwargs: Additional arguments passed to fit methods
        """
        # Fit Stage 1
        retriever_kwargs = kwargs.get("retriever_kwargs", {})
        self.retriever.fit(train_data, **retriever_kwargs)
        
        # Fit Stage 2 (if enabled)
        if self.reranker is not None:
            reranker_kwargs = kwargs.get("reranker_kwargs", {})
            self.reranker.fit(train_data, **reranker_kwargs)
    
    def recommend(
        self,
        user_id: int,
        exclude_items: Optional[List[int]] = None
    ) -> List[int]:
        """Generate recommendations for a user.
        
        Args:
            user_id: ID of the user
            exclude_items: Items to exclude from recommendations
            
        Returns:
            List[int]: Recommended item IDs (sorted by score descending)
        """
        # Stage 1: Retrieve candidates
        exclude_set = set(exclude_items) if exclude_items else set()
        candidates = self.retriever.retrieve(user_id, exclude_items=exclude_set)
        
        if not candidates:
            return []
        
        # Stage 2: Rerank (if enabled)
        if self.reranker is None:
            return candidates
        
        scored = self.reranker.rerank(user_id, candidates)
        return [item_id for item_id, _ in scored]
    
    def recommend_batch(
        self,
        user_ids: List[int],
        exclude_items: Optional[Dict[int, List[int]]] = None
    ) -> Dict[int, List[int]]:
        """Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            exclude_items: Dict {user_id: [item_ids]} to exclude
            
        Returns:
            Dict[int, List[int]]: {user_id: [recommended_item_ids]}
        """
        if exclude_items is None:
            exclude_items = {}
        
        results = {}
        for user_id in user_ids:
            exclude = exclude_items.get(user_id, [])
            results[user_id] = self.recommend(user_id, exclude_items=exclude)
        
        return results

