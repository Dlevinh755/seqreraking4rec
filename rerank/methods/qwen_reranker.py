"""Qwen-based reranker using LLM for candidate reranking."""

from typing import Dict, List, Tuple, Any, Optional

from rerank.base import BaseReranker
from rerank.models.llm import LLMModel, build_prompt_from_candidates, rank_candidates


class QwenReranker(BaseReranker):
    """Reranker sử dụng Qwen LLM để rerank candidates.
    
    Wrapper cho LLMModel để implement BaseReranker interface.
    
    ⚠️ **Giới hạn quan trọng**: LLM reranker chỉ hỗ trợ tối đa 20 candidates (A-T).
    
    **Mối quan hệ với Retrieval Stage**:
    - Số lượng candidates phụ thuộc vào `retrieval_top_k` từ Stage 1
    - Nếu `retrieval_top_k > 20`, sẽ tự động truncate về 20 candidates đầu tiên
    - **Khuyến nghị**: Set `retrieval_top_k <= 20` khi dùng Qwen reranker để tránh mất mát thông tin
    
    **Ví dụ**:
    ```python
    # Tốt: retrieval_top_k = 20
    retrieval_cfg = RetrievalConfig(method="lrurec", top_k=20)
    rerank_cfg = RerankConfig(method="qwen", top_k=10)
    
    # Cũng OK nhưng sẽ truncate: retrieval_top_k = 200
    retrieval_cfg = RetrievalConfig(method="lrurec", top_k=200)
    rerank_cfg = RerankConfig(method="qwen", top_k=10)  # Chỉ dùng 20 đầu tiên
    ```
    """

    def __init__(
        self,
        top_k: int = 50,
        model_name: Optional[str] = None,
        max_history: int = 10,
    ) -> None:
        """
        Args:
            top_k: Số lượng items trả về sau rerank (không liên quan đến giới hạn 20)
            model_name: Tên model (default: "Qwen/Qwen3-0.6B")
            max_history: Số lượng items trong history để dùng cho prompt
            
        Note:
            - `top_k` ở đây là số items cuối cùng trả về (có thể < 20)
            - Giới hạn 20 candidates áp dụng cho input từ retrieval stage
            - Nếu retrieval trả về > 20 candidates, sẽ truncate về 20 đầu tiên
        """
        super().__init__(top_k=top_k)
        self.model_name = model_name or "Qwen/Qwen3-0.6B"
        self.max_history = max_history
        self.llm_model: Optional[LLMModel] = None
        
        # Data structures cần thiết cho rerank
        self.user_history: Dict[int, List[str]] = {}  # user_id -> [item_texts]
        self.item_id2text: Dict[int, str] = {}  # item_id -> item_text

    def fit(
        self,
        train_data: Dict[int, List[int]],
        **kwargs: Any
    ) -> None:
        """Fit reranker.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            **kwargs: Additional arguments:
                - item_id2text: Dict[int, str] - mapping item_id -> text
                - user_history: Dict[int, List[str]] - user history texts
                - train_data_for_llm: List[dict] - training data cho LLM (optional)
        """
        # Load item_id2text và user_history từ kwargs
        self.item_id2text = kwargs.get("item_id2text", {})
        self.user_history = kwargs.get("user_history", {})
        
        # Nếu có train_data_for_llm, train LLM model
        train_data_for_llm = kwargs.get("train_data_for_llm")
        if train_data_for_llm is not None:
            self.llm_model = LLMModel(
                train_data=train_data_for_llm,
                model_name=self.model_name
            )
            self.llm_model.load_model()
            self.llm_model.train()
        else:
            # Chỉ load model, không train
            self.llm_model = LLMModel(
                train_data=None,
                model_name=self.model_name
            )
            self.llm_model.load_model()
        
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
            
        Note:
            - LLM reranker chỉ hỗ trợ tối đa 20 candidates (A-T)
            - Nếu có > 20 candidates, sẽ truncate về 20 đầu tiên và có warning
            - **Khuyến nghị**: Điều chỉnh `retrieval_top_k <= 20` trong pipeline config
              để tránh mất mát thông tin từ retrieval stage
        """
        self._validate_fitted()
        
        if self.llm_model is None:
            raise RuntimeError("LLM model chưa được load. Gọi fit() trước!")
        
        if not candidates:
            return []
        
        # LLM chỉ hỗ trợ tối đa 20 candidates (A-T)
        MAX_CANDIDATES = 20
        original_count = len(candidates)
        if original_count > MAX_CANDIDATES:
            import warnings
            warnings.warn(
                f"Truncating {original_count} candidates to {MAX_CANDIDATES} "
                f"(LLM reranker limit). Consider using fewer candidates from retrieval stage."
            )
            candidates = candidates[:MAX_CANDIDATES]
        
        # Lấy user history
        history = self.user_history.get(user_id, [])
        history = history[-self.max_history:]  # Chỉ lấy max_history items gần nhất
        
        # Build prompt
        prompt = build_prompt_from_candidates(
            history,
            candidates,
            self.item_id2text
        )
        
        # Predict probabilities
        probs = self.llm_model.predict_probs(prompt)
        
        # Validate: probs phải có đúng số lượng với candidates
        if len(probs) != len(candidates):
            raise ValueError(
                f"Mismatch: {len(candidates)} candidates but {len(probs)} probabilities. "
                f"This should not happen - check predict_probs() implementation."
            )
        
        # Rank candidates theo probabilities
        ranked_items = rank_candidates(probs, candidates)
        
        # Convert thành (item_id, score) tuples
        # Score = probability từ model
        # Tạo mapping từ item_id -> score để tránh index lookup
        item_to_score = {item_id: float(probs[i]) for i, item_id in enumerate(candidates)}
        
        scored = []
        for item_id in ranked_items[:self.top_k]:
            score = item_to_score.get(item_id, 0.0)
            scored.append((item_id, score))
        
        return scored

