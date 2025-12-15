"""Model components for reranking stage.

Currently includes:
- LLMModel: Qwen-based LLM model for reranking candidates.
- VIP5: VIP5 multimodal model from original repository.
"""

from .llm import LLMModel, build_prompt_from_candidates, rank_candidates

# VIP5 imports - from original implementation
try:
    from .vip5_modeling import VIP5, JointEncoder, VisualEmbedding, VIP5Seq2SeqLMOutput
    from .vip5_utils import prepare_vip5_input, build_rerank_prompt, calculate_whole_word_ids
    VIP5_AVAILABLE = True
except ImportError:
    VIP5_AVAILABLE = False
    VIP5 = None
    JointEncoder = None
    VisualEmbedding = None
    VIP5Seq2SeqLMOutput = None
    prepare_vip5_input = None
    build_rerank_prompt = None
    calculate_whole_word_ids = None

__all__ = [
    "LLMModel",
    "build_prompt_from_candidates",
    "rank_candidates",
]

if VIP5_AVAILABLE:
    __all__.extend([
        "VIP5",
        "JointEncoder",
        "VisualEmbedding",
        "VIP5Seq2SeqLMOutput",
        "prepare_vip5_input",
        "build_rerank_prompt",
        "calculate_whole_word_ids",
    ])

