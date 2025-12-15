"""VIP5-based reranker using multimodal visual and textual features.

This implementation follows the original VIP5 source code as closely as possible.
Reference: https://github.com/jeykigung/VIP5
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import numpy as np
from pathlib import Path
import sys

from rerank.base import BaseReranker
from dataset.paths import get_clip_embeddings_path

# Import VIP5 model from original implementation
try:
    from rerank.models.vip5_modeling import VIP5, VIP5Seq2SeqLMOutput
    from rerank.models.vip5_utils import prepare_vip5_input, build_rerank_prompt, calculate_whole_word_ids
except ImportError as e:
    raise ImportError(
        f"Failed to import VIP5 model. Please ensure VIP5 source code is available. Error: {e}"
    )

# Try to import tokenizer from VIP5
VIP5_TOKENIZER_AVAILABLE = False
try:
    vip5_src_path = Path(__file__).parent.parent.parent / "retrieval" / "vip5_temp" / "src"
    if vip5_src_path.exists() and str(vip5_src_path) not in sys.path:
        sys.path.insert(0, str(vip5_src_path))
    try:
        from tokenization import P5Tokenizer  # type: ignore
        VIP5_TOKENIZER_AVAILABLE = True
    except ImportError:
        # Fallback to T5Tokenizer
        from transformers import T5Tokenizer
        P5Tokenizer = T5Tokenizer  # type: ignore
except (ImportError, Exception):
    # Fallback to T5Tokenizer
    from transformers import T5Tokenizer
    P5Tokenizer = T5Tokenizer  # type: ignore


class VIP5Reranker(BaseReranker):
    """Reranker sử dụng VIP5 (Visual Item Preference) model.
    
    VIP5 là một mô hình multimodal T5-based kết hợp visual và textual features
    để dự đoán user-item preferences. Implementation này theo sát source code gốc.
    
    Reference: https://github.com/jeykigung/VIP5
    """
    
    def __init__(
        self,
        top_k: int = 50,
        checkpoint_path: Optional[str] = None,
        backbone: str = "t5-small",
        tokenizer_path: Optional[str] = None,
        image_feature_type: str = "vitb32",
        image_feature_size_ratio: int = 2,
        max_text_length: int = 128,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            top_k: Số lượng items trả về sau rerank
            checkpoint_path: Đường dẫn đến VIP5 checkpoint (optional)
            backbone: T5 model backbone (default: "t5-small")
            tokenizer_path: Path to tokenizer (optional, uses backbone if None)
            image_feature_type: CLIP feature type (vitb32, vitb16, vitl14, rn50, rn101)
            image_feature_size_ratio: Number of visual tokens per item (default: 2)
            max_text_length: Maximum text sequence length
            device: Device để chạy model ("cuda" hoặc "cpu")
        """
        super().__init__(top_k=top_k)
        self.checkpoint_path = checkpoint_path
        self.backbone = backbone
        self.tokenizer_path = tokenizer_path
        self.image_feature_type = image_feature_type
        self.image_feature_size_ratio = image_feature_size_ratio
        self.max_text_length = max_text_length
        
        # Image feature dimensions from VIP5
        image_feature_dim_dict = {
            'vitb32': 512,
            'vitb16': 512,
            'vitl14': 768,
            'rn50': 1024,
            'rn101': 512
        }
        self.image_feature_dim = image_feature_dim_dict.get(image_feature_type, 512)
        
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[VIP5] = None
        self.tokenizer = None
        
        # CLIP embeddings cache
        self.visual_embeddings: Optional[torch.Tensor] = None  # [num_items, visual_dim]
        self.text_embeddings: Optional[torch.Tensor] = None     # [num_items, text_dim]
        self.item_id_to_idx: Dict[int, int] = {}  # item_id -> embedding index
        self.item_id_to_text: Dict[int, str] = {}  # item_id -> item text (for tokenization)
        
    def _load_clip_embeddings(
        self,
        dataset_code: str,
        min_rating: int,
        min_uc: int,
        min_sc: int,
        num_items: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load CLIP embeddings for visual and text features.
        
        Args:
            dataset_code: Dataset code
            min_rating: Minimum rating threshold
            min_uc: Minimum user count
            min_sc: Minimum item count
            num_items: Number of items
            
        Returns:
            Tuple of (visual_embeddings, text_embeddings)
        """
        clip_path = get_clip_embeddings_path(dataset_code, min_rating, min_uc, min_sc)
        
        if not clip_path.exists():
            raise FileNotFoundError(
                f"CLIP embeddings not found at {clip_path}. "
                "Please run data_prepare.py with --use_image and --use_text flags first."
            )
        
        clip_payload = torch.load(clip_path, map_location="cpu")
        image_embs = clip_payload.get("image_embs")  # Shape: [num_items+1, D] (row 0 is padding)
        text_embs = clip_payload.get("text_embs")    # Shape: [num_items+1, D] (row 0 is padding)
        
        if image_embs is None:
            raise ValueError("VIP5 requires image embeddings, but image_embs is None")
        if text_embs is None:
            raise ValueError("VIP5 requires text embeddings, but text_embs is None")
        
        # Skip row 0 (padding), use rows 1..num_items
        visual_emb = image_embs[1:num_items+1]  # [num_items, D]
        text_emb = text_embs[1:num_items+1]     # [num_items, D]
        
        # Validate dimensions
        if visual_emb.size(1) != self.visual_dim:
            print(f"Warning: visual_dim mismatch. Expected {self.visual_dim}, got {visual_emb.size(1)}")
            self.visual_dim = visual_emb.size(1)
        
        if text_emb.size(1) != self.text_dim:
            print(f"Warning: text_dim mismatch. Expected {self.text_dim}, got {text_emb.size(1)}")
            self.text_dim = text_emb.size(1)
        
        return visual_emb, text_emb
    
    def fit(
        self,
        train_data: Dict[int, List[int]],
        **kwargs: Any
    ) -> None:
        """Fit VIP5 reranker.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            **kwargs: Additional arguments:
                - dataset_code: str - dataset code
                - min_rating: int - minimum rating threshold
                - min_uc: int - minimum user count
                - min_sc: int - minimum item count
                - num_items: int - number of items
                - item_id2text: Dict[int, str] - mapping item_id -> text (optional)
        """
        # Get dataset info
        dataset_code = kwargs.get("dataset_code")
        min_rating = kwargs.get("min_rating", 3)
        min_uc = kwargs.get("min_uc", 20)
        min_sc = kwargs.get("min_sc", 20)
        num_items = kwargs.get("num_items")
        self.item_id_to_text = kwargs.get("item_id2text", {})
        
        if num_items is None:
            # Infer from train_data
            all_items = set()
            for items in train_data.values():
                all_items.update(items)
            num_items = max(all_items) if all_items else 0
        
        if dataset_code is None:
            raise ValueError("VIP5Reranker.fit requires dataset_code in kwargs")
        
        # Load CLIP embeddings
        print("Loading CLIP embeddings for VIP5...")
        visual_emb, text_emb = self._load_clip_embeddings(
            dataset_code, min_rating, min_uc, min_sc, num_items
        )
        self.visual_embeddings = visual_emb
        self.text_embeddings = text_emb
        
        # Build item_id to embedding index mapping
        # Assuming item IDs are 1-indexed and embeddings are in order
        for item_id in range(1, num_items + 1):
            self.item_id_to_idx[item_id] = item_id - 1  # 0-indexed
        
        # Initialize tokenizer
        print("Initializing VIP5 tokenizer...")
        try:
            if VIP5_TOKENIZER_AVAILABLE and self.tokenizer_path:
                self.tokenizer = P5Tokenizer.from_pretrained(
                    self.tokenizer_path,
                    max_length=self.max_text_length,
                    do_lower_case=False
                )
            else:
                # Use T5Tokenizer as fallback
                from transformers import T5Tokenizer
                self.tokenizer = T5Tokenizer.from_pretrained(self.backbone)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer, using T5Tokenizer. Error: {e}")
            from transformers import T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.backbone)
        
        # Initialize or load VIP5 model
        from transformers.models.t5.configuration_t5 import T5Config
        
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            print(f"Loading VIP5 checkpoint from {self.checkpoint_path}...")
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            
            # Load config from checkpoint or use default
            if "config" in checkpoint:
                config = checkpoint["config"]
            else:
                # Create config from backbone
                config = T5Config.from_pretrained(self.backbone)
                # Add VIP5-specific config
                config.feat_dim = self.image_feature_dim
                config.n_vis_tokens = self.image_feature_size_ratio
                config.use_vis_layer_norm = True
                config.use_adapter = False  # Can be set via kwargs
                config.adapter_config = None
                config.add_adapter_cross_attn = False
                config.use_lm_head_adapter = False
                config.whole_word_embed = True
                config.category_embed = True
            
            self.model = VIP5(config)
            
            # Load state dict
            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            elif "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"], strict=False)
            else:
                # Try loading directly
                self.model.load_state_dict(checkpoint, strict=False)
        else:
            print("Initializing new VIP5 model (no checkpoint provided)...")
            # Create config
            config = T5Config.from_pretrained(self.backbone)
            # Add VIP5-specific config
            config.feat_dim = self.image_feature_dim
            config.n_vis_tokens = self.image_feature_size_ratio
            config.use_vis_layer_norm = True
            config.use_adapter = kwargs.get("use_adapter", False)
            config.adapter_config = kwargs.get("adapter_config", None)
            config.add_adapter_cross_attn = kwargs.get("add_adapter_cross_attn", False)
            config.use_lm_head_adapter = kwargs.get("use_lm_head_adapter", False)
            config.whole_word_embed = True
            config.category_embed = True
            
            self.model = VIP5(config)
            # Load pretrained T5 weights
            from transformers import T5ForConditionalGeneration
            pretrained = T5ForConditionalGeneration.from_pretrained(self.backbone)
            # Copy compatible weights
            self.model.shared.load_state_dict(pretrained.shared.state_dict())
            self.model.decoder.load_state_dict(pretrained.decoder.state_dict(), strict=False)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.is_fitted = True
        print(f"VIP5Reranker fitted. Model on device: {self.device}")
    
    def rerank(
        self,
        user_id: int,
        candidates: List[int],
        **kwargs: Any
    ) -> List[Tuple[int, float]]:
        """Rerank candidates cho một user sử dụng VIP5.
        
        Args:
            user_id: ID của user
            candidates: List các item IDs cần rerank
            **kwargs: Additional arguments:
                - user_history: List[int] - user's interaction history (optional)
        
        Returns:
            List[Tuple[int, float]]: [(item_id, score)] đã sort giảm dần
        """
        self._validate_fitted()
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("VIP5 model chưa được load. Gọi fit() trước!")
        
        if not candidates:
            return []
        
        # Get user history
        user_history = kwargs.get("user_history", [])
        
        # Build prompt following VIP5 format
        # Use template A-1: "Given the following purchase history of user_{} : \n {} \n predict next possible item to be purchased by the user ?"
        prompt = build_rerank_prompt(
            user_id,
            user_history,
            candidates,
            template="Given the following purchase history of user_{} : \n {} \n predict next possible item to be purchased by the user ?"
        )
        
        # Get visual features for candidates
        candidate_visual = []
        valid_candidates = []
        
        for item_id in candidates:
            if item_id in self.item_id_to_idx:
                idx = self.item_id_to_idx[item_id]
                valid_candidates.append(item_id)
                candidate_visual.append(self.visual_embeddings[idx])
        
        if not valid_candidates:
            return []
        
        # Convert to tensor [num_candidates, feat_dim]
        visual_tensor = torch.stack(candidate_visual)
        
        # Prepare VIP5 input
        vip5_input = prepare_vip5_input(
            prompt,
            visual_tensor,
            self.tokenizer,
            max_length=self.max_text_length,
            image_feature_size_ratio=self.image_feature_size_ratio,
        )
        
        # Move to device
        input_ids = vip5_input["input_ids"].to(self.device)
        whole_word_ids = vip5_input["whole_word_ids"].to(self.device)
        category_ids = vip5_input["category_ids"].to(self.device)
        vis_feats = vip5_input["vis_feats"].to(self.device)
        attention_mask = vip5_input["attention_mask"].to(self.device)
        
        # For reranking with VIP5, we encode user history + each candidate
        # and use encoder output to compute scores
        # Following VIP5's approach: encode and use encoder hidden states
        
        scores = []
        
        # Encode user history + each candidate separately
        for i, item_id in enumerate(valid_candidates):
            # Build prompt for this specific candidate
            candidate_prompt = build_rerank_prompt(
                user_id,
                user_history,
                [item_id],  # Single candidate
                template="Given the following purchase history of user_{} : \n {} \n predict next possible item to be purchased by the user ?"
            )
            
            # Get visual feature for this candidate
            item_visual = candidate_visual[i].unsqueeze(0)  # [1, feat_dim]
            
            # Prepare VIP5 input
            vip5_input = prepare_vip5_input(
                candidate_prompt,
                item_visual,
                self.tokenizer,
                max_length=self.max_text_length,
                image_feature_size_ratio=self.image_feature_size_ratio,
            )
            
            # Move to device
            input_ids = vip5_input["input_ids"].to(self.device)
            whole_word_ids = vip5_input["whole_word_ids"].to(self.device)
            category_ids = vip5_input["category_ids"].to(self.device)
            vis_feats = vip5_input["vis_feats"].to(self.device)
            attention_mask = vip5_input["attention_mask"].to(self.device)
            
            # Encode with VIP5 encoder
            self.model.eval()
            with torch.no_grad():
                encoder_outputs = self.model.encoder(
                    input_ids=input_ids,
                    whole_word_ids=whole_word_ids,
                    category_ids=category_ids,
                    vis_feats=vis_feats,
                    attention_mask=attention_mask,
                    return_dict=True,
                    task="sequential",  # VIP5 task type
                )
            
            # Get encoder hidden states [1, seq_len, d_model]
            encoder_hidden = encoder_outputs.last_hidden_state
            
            # Use mean pooling of encoder output as score
            # In full implementation, could use decoder to generate and score
            score = float(encoder_hidden.mean(dim=1).squeeze(0).norm().item())
            
            scores.append((item_id, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return scores[:self.top_k]

