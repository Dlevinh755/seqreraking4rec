"""Model components for retrieval stage (independent from LlamaRec).

Currently includes:
- NeuralLRURec: lightweight neural LRU-style sequential recommender.
"""

from .neural_lru import NeuralLRUConfig, NeuralLRURec
