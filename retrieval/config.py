"""
Configuration cho retrieval methods
"""

class RetrievalConfig:
    """Config chung cho retrieval"""
    
    # Số lượng candidates trả về
    TOP_K = 50
    
    # Random seed
    SEED = 42
    
    # Collaborative Filtering
    CF_N_FACTORS = 64  # Số latent factors
    CF_N_EPOCHS = 20
    CF_LR = 0.01
    
    # Content-based
    CONTENT_EMBEDDING_DIM = 128
    
    # Neural retrieval
    NEURAL_HIDDEN_DIM = 256
    NEURAL_DROPOUT = 0.2
    NEURAL_BATCH_SIZE = 256
    
    @classmethod
    def from_dict(cls, config_dict):
        """Load config từ dictionary"""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
        return cls
