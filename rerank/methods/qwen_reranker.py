"""Qwen-based reranker using LLM for candidate reranking."""

from typing import Dict, List, Tuple, Any, Optional

from rerank.base import BaseReranker
from rerank.models.llm import LLMModel, build_prompt_from_candidates, rank_candidates


class QwenReranker(BaseReranker):
    """Reranker sử dụng Qwen LLM để rerank candidates.
    
    Wrapper cho LLMModel để implement BaseReranker interface.
    
    **Tính năng**:
    - Số lượng candidates tự động điều chỉnh theo input từ retrieval stage
    - Sử dụng số (1, 2, 3, ...) thay vì chữ cái để hỗ trợ nhiều candidates
    - Max candidates có thể được cấu hình qua `--qwen_max_candidates` trong config.py
    
    **Mối quan hệ với Retrieval Stage**:
    - Số lượng candidates phụ thuộc vào `retrieval_top_k` từ Stage 1
    - Nếu `qwen_max_candidates` được set, sẽ truncate nếu cần
    - Nếu `qwen_max_candidates` là None, sử dụng tất cả candidates từ retrieval
    
    **Ví dụ**:
    ```python
    # Sử dụng tất cả candidates từ retrieval
    retrieval_cfg = RetrievalConfig(method="lrurec", top_k=200)
    rerank_cfg = RerankConfig(method="qwen", top_k=50)
    # → Qwen sẽ xử lý 200 candidates
    
    # Giới hạn số candidates
    # Set --qwen_max_candidates 50 trong config.py
    # → Qwen sẽ chỉ xử lý 50 candidates đầu tiên
    ```
    """

    def __init__(
        self,
        top_k: int = 50,
        model_name: Optional[str] = None,
        max_history: int = 5,
        max_candidates: Optional[int] = None,
    ) -> None:
        """
        Args:
            top_k: Số lượng items trả về sau rerank
            model_name: Tên model (default: "Qwen/Qwen3-0.6B")
            max_history: Số lượng items trong history để dùng cho prompt (default: 5, chỉ giữ 5 items cuối cùng)
            max_candidates: Maximum number of candidates to process (None = no limit, uses all from retrieval)
            
        Note:
            - `top_k` là số items cuối cùng trả về sau rerank
            - `max_candidates` giới hạn số candidates được xử lý (None = tự động theo retrieval_top_k)
            - Nếu `max_candidates` được set, sẽ truncate candidates nếu cần
        """
        super().__init__(top_k=top_k)
        self.model_name = model_name or "Qwen/Qwen3-0.6B"
        self.max_history = max_history
        self.max_candidates = max_candidates
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
        
        # Get optimization settings
        from config import arg
        use_torch_compile = getattr(arg, 'use_torch_compile', False)
        
        # Nếu có train_data_for_llm, train LLM model
        train_data_for_llm = kwargs.get("train_data_for_llm")
        if train_data_for_llm is not None:
            # Don't compile model when training - PEFT doesn't support compiled models
            self.llm_model = LLMModel(
                train_data=train_data_for_llm,
                model_name=self.model_name
            )
            self.llm_model.load_model(use_torch_compile=False)  # Disable compile for training
            self.llm_model.train()
        else:
            # Chỉ load model, không train - có thể compile để tăng tốc inference
            self.llm_model = LLMModel(
                train_data=None,
                model_name=self.model_name
            )
            self.llm_model.load_model(use_torch_compile=use_torch_compile)
        
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
            - Số lượng candidates tự động điều chỉnh theo input từ retrieval stage
            - Nếu `max_candidates` được set, sẽ truncate nếu cần
            - Prompt format sử dụng số (1, 2, 3, ...) thay vì chữ cái để hỗ trợ nhiều candidates
        """
        self._validate_fitted()
        
        if self.llm_model is None:
            raise RuntimeError("LLM model chưa được load. Gọi fit() trước!")
        
        if not candidates:
            return []
        
        # Apply max_candidates limit if set
        original_count = len(candidates)
        if self.max_candidates is not None and original_count > self.max_candidates:
            candidates = candidates[:self.max_candidates]
        
        # Lấy user history - chỉ giữ lại max_history (5) items cuối cùng nếu quá dài
        history = self.user_history.get(user_id, [])
        history = history[-self.max_history:]  # Chỉ lấy max_history items cuối cùng (default: 5)
        
        # Build prompt (sử dụng số thay vì chữ cái)
        prompt = build_prompt_from_candidates(
            history,
            candidates,
            self.item_id2text,
            max_candidates=self.max_candidates
        )
        
        # Predict probabilities
        num_candidates = len(candidates)
        probs = self.llm_model.predict_probs(prompt, num_candidates=num_candidates)
        
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

