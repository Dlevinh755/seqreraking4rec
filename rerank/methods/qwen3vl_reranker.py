"""Qwen3-VL-based reranker with multiple prompt modes."""

from typing import Dict, List, Tuple, Any, Optional

from rerank.base import BaseReranker
from rerank.models.qwen3vl import Qwen3VLModel
from rerank.models.llm import rank_candidates


class Qwen3VLReranker(BaseReranker):
    """Reranker sử dụng Qwen3-VL với nhiều prompt modes.
    
    **4 Prompt Modes**:
    1. `raw_image`: Sử dụng trực tiếp raw image truyền vào prompt
    2. `caption`: Sử dụng image caption
    3. `semantic_summary`: Sử dụng image semantic summary với Qwen3-VL
    4. `semantic_summary_small`: Sử dụng semantic summary với model nhỏ hơn (Qwen3-0.6B)
    
    **Requirements**:
    - Mode `raw_image`: Requires images in item_meta
    - Mode `caption`: Requires captions in item_meta (run data_prepare.py with --generate_caption)
    - Mode `semantic_summary`: Requires semantic summaries in item_meta (run data_prepare.py with --generate_semantic_summary)
    - Mode `semantic_summary_small`: Requires semantic summaries in item_meta
    """
    
    def __init__(
        self,
        top_k: int = 50,
        mode: str = "raw_image",
        model_name: Optional[str] = None,
        max_history: int = 10,
        max_candidates: Optional[int] = None,
    ) -> None:
        """
        Args:
            top_k: Số lượng items trả về sau rerank
            mode: Prompt mode - "raw_image", "caption", "semantic_summary", "semantic_summary_small"
            model_name: Model name (auto-selected based on mode if None)
            max_history: Số lượng items trong history để dùng cho prompt
            max_candidates: Maximum number of candidates to process (None = no limit)
        """
        super().__init__(top_k=top_k)
        self.mode = mode.lower()
        self.model_name = model_name
        self.max_history = max_history
        self.max_candidates = max_candidates
        self.qwen3vl_model: Optional[Qwen3VLModel] = None
        
        # Data structures cần thiết cho rerank
        self.user_history: Dict[int, List[str]] = {}  # user_id -> [item_texts/images]
        self.item_meta: Dict[int, Dict[str, Any]] = {}  # item_id -> {image_path, caption, semantic_summary, text}
    
    def fit(
        self,
        train_data: Dict[int, List[int]],
        **kwargs: Any
    ) -> None:
        """Fit reranker.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            **kwargs: Additional arguments:
                - item_meta: Dict[int, Dict] - mapping item_id -> {image_path, caption, semantic_summary, text}
                - user_history: Dict[int, List[str]] - user history texts/images
        """
        # Load item_meta và user_history từ kwargs
        self.item_meta = kwargs.get("item_meta", {})
        self.user_history = kwargs.get("user_history", {})
        
        # Load Qwen3-VL model
        self.qwen3vl_model = Qwen3VLModel(
            mode=self.mode,
            model_name=self.model_name
        )
        
        self.is_fitted = True
    
    def rerank(
        self,
        user_id: int,
        candidates: List[int],
        **kwargs: Any
    ) -> List[Tuple[int, float]]:
        """Rerank candidates cho một user.
        
        Args:
            user_id: ID của user
            candidates: List các item IDs cần rerank
            **kwargs: Additional arguments (không dùng ở đây)
            
        Returns:
            List[Tuple[int, float]]: [(item_id, score)] đã sort giảm dần
        """
        self._validate_fitted()
        
        if self.qwen3vl_model is None:
            raise RuntimeError("Qwen3-VL model chưa được load. Gọi fit() trước!")
        
        if not candidates:
            return []
        
        # Apply max_candidates limit if set
        if self.max_candidates is not None and len(candidates) > self.max_candidates:
            candidates = candidates[:self.max_candidates]
        
        # Lấy user history
        history = self.user_history.get(user_id, [])
        history = history[-self.max_history:]  # Chỉ lấy max_history items gần nhất
        
        # Predict probabilities
        num_candidates = len(candidates)
        probs = self.qwen3vl_model.predict_probs(
            user_history=history,
            candidates=candidates,
            item_meta=self.item_meta,
            num_candidates=num_candidates
        )
        
        # Validate: probs phải có đúng số lượng với candidates
        if len(probs) != len(candidates):
            raise ValueError(
                f"Mismatch: {len(candidates)} candidates but {len(probs)} probabilities. "
                f"This should not happen - check predict_probs() implementation."
            )
        
        # Rank candidates theo probabilities
        ranked_items = rank_candidates(probs, candidates)
        
        # Convert thành (item_id, score) tuples
        item_to_score = {item_id: float(probs[i]) for i, item_id in enumerate(candidates)}
        
        scored = []
        for item_id in ranked_items[:self.top_k]:
            score = item_to_score.get(item_id, 0.0)
            scored.append((item_id, score))
        
        return scored

