"""VIP5-based reranker using multimodal visual and textual features.

This implementation follows the original VIP5 source code as closely as possible.
Reference: https://github.com/jeykigung/VIP5
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import numpy as np
from pathlib import Path
import sys
import random
from copy import deepcopy

from rerank.base import BaseReranker
from dataset.paths import get_clip_embeddings_path
from evaluation.metrics import recall_at_k

# Import VIP5 model from original implementation
try:
    from rerank.models.vip5_modeling import VIP5, VIP5Seq2SeqLMOutput
    from rerank.models.vip5_utils import prepare_vip5_input, build_rerank_prompt, calculate_whole_word_ids
    VIP5_AVAILABLE = True
    VIP5_IMPORT_ERROR = None
except ImportError as e:
    VIP5_AVAILABLE = False
    # Store error for later use
    VIP5_IMPORT_ERROR = str(e)
    # Create dummy classes to allow import (will raise error when used)
    VIP5 = None
    VIP5Seq2SeqLMOutput = None
    def prepare_vip5_input(*args, **kwargs):
        raise ImportError(f"VIP5 not available: {VIP5_IMPORT_ERROR}")
    def build_rerank_prompt(*args, **kwargs):
        raise ImportError(f"VIP5 not available: {VIP5_IMPORT_ERROR}")
    def calculate_whole_word_ids(*args, **kwargs):
        raise ImportError(f"VIP5 not available: {VIP5_IMPORT_ERROR}")

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
        batch_size: int = 32,
        num_epochs: int = 10,
        lr: float = 1e-4,
        patience: Optional[int] = None,
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
            batch_size: Batch size cho training (default: 32)
            num_epochs: Số epochs (default: 10)
            lr: Learning rate (default: 1e-4)
            patience: Early stopping patience (None = no early stopping)
            device: Device để chạy model ("cuda" hoặc "cpu")
        """
        super().__init__(top_k=top_k)
        self.checkpoint_path = checkpoint_path
        self.backbone = backbone
        self.tokenizer_path = tokenizer_path
        self.image_feature_type = image_feature_type
        self.image_feature_size_ratio = image_feature_size_ratio
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.patience = patience
        
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
        if not VIP5_AVAILABLE:
            raise ImportError(
                f"VIP5 model is not available. Please ensure VIP5 source code is available. "
                f"Error: {VIP5_IMPORT_ERROR}"
            )
        # Type annotation: VIP5 is imported from vip5_modeling
        self.model: Optional[Any] = None  # Will be VIP5 instance after fit()
        self.tokenizer = None
        
        # CLIP embeddings cache
        self.visual_embeddings: Optional[torch.Tensor] = None  # [num_items, visual_dim]
        self.text_embeddings: Optional[torch.Tensor] = None     # [num_items, text_dim]
        self.item_id_to_idx: Dict[int, int] = {}  # item_id -> embedding index
        self.item_id_to_text: Dict[int, str] = {}  # item_id -> item text (for tokenization)
        
        # Store user history for training
        self.user_history: Dict[int, List[int]] = {}  # user_id -> [item_ids]
        
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
        
        return visual_emb, text_emb
    
    def fit(
        self,
        train_data: Dict[int, List[int]],
        **kwargs: Any
    ) -> None:
        """Fit VIP5 reranker with training.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            **kwargs: Additional arguments:
                - dataset_code: str - dataset code
                - min_rating: int - minimum rating threshold
                - min_uc: int - minimum user count
                - min_sc: int - minimum item count
                - num_items: int - number of items
                - item_id2text: Dict[int, str] - mapping item_id -> text (optional)
                - val_data: Dict[int, List[int]] - validation data for early stopping
                - num_epochs: int - override default num_epochs (optional)
                - batch_size: int - override default batch_size (optional)
                - lr: float - override default lr (optional)
                - patience: int - override default patience (optional)
        """
        # Override hyperparameters from kwargs if provided
        if "num_epochs" in kwargs:
            self.num_epochs = kwargs["num_epochs"]
        if "batch_size" in kwargs:
            self.batch_size = kwargs["batch_size"]
        if "lr" in kwargs:
            self.lr = kwargs["lr"]
        if "patience" in kwargs:
            self.patience = kwargs["patience"]
        
        # Get dataset info
        dataset_code = kwargs.get("dataset_code")
        min_rating = kwargs.get("min_rating", 3)
        min_uc = kwargs.get("min_uc", 20)
        min_sc = kwargs.get("min_sc", 20)
        num_items = kwargs.get("num_items")
        self.item_id_to_text = kwargs.get("item_id2text", {})
        val_data = kwargs.get("val_data")
        
        if num_items is None:
            # Infer from train_data
            all_items = set()
            for items in train_data.values():
                all_items.update(items)
            num_items = max(all_items) if all_items else 0
        
        if dataset_code is None:
            raise ValueError("VIP5Reranker.fit requires dataset_code in kwargs")
        
        # Store user history
        self.user_history = train_data
        
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
                config.use_adapter = kwargs.get("use_adapter", False)
                config.adapter_config = kwargs.get("adapter_config", None)
                config.add_adapter_cross_attn = kwargs.get("add_adapter_cross_attn", False)
                config.use_lm_head_adapter = kwargs.get("use_lm_head_adapter", False)
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
        
        # Training
        print("Preparing training samples...")
        train_samples = self._prepare_training_samples(train_data)
        
        if len(train_samples) == 0:
            print("Warning: No training samples generated. Skipping training.")
            self.model.eval()
            self.is_fitted = True
            return
        
        print(f"Generated {len(train_samples)} training samples")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Training loop
        best_state = None
        best_val_recall = -1.0
        epochs_no_improve = 0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            # Shuffle training samples
            random.shuffle(train_samples)
            
            # Training batches
            for i in range(0, len(train_samples), self.batch_size):
                batch = train_samples[i:i + self.batch_size]
                if len(batch) == 0:
                    continue
                
                # Prepare batch
                batch_dict = self._prepare_batch(batch)
                
                # Move to device
                input_ids = batch_dict["input_ids"].to(self.device)
                whole_word_ids = batch_dict["whole_word_ids"].to(self.device)
                category_ids = batch_dict["category_ids"].to(self.device)
                vis_feats = batch_dict["vis_feats"].to(self.device)
                target_ids = batch_dict["target_ids"].to(self.device)
                loss_weights = batch_dict["loss_weights"].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(
                    input_ids=input_ids,
                    whole_word_ids=whole_word_ids,
                    category_ids=category_ids,
                    vis_feats=vis_feats,
                    labels=target_ids,
                    return_dict=True,
                    task="sequential",
                    reduce_loss=False  # Get per-token loss
                )
                
                # Compute loss
                loss = output["loss"]  # Shape: [batch_size * seq_len] (per-token loss)
                
                # Reshape loss to [batch_size, seq_len] and compute per-sample loss
                batch_size = target_ids.size(0)
                seq_len = target_ids.size(1)
                
                if loss.dim() > 0 and loss.numel() > 0:
                    # Reshape loss from [batch_size * seq_len] to [batch_size, seq_len]
                    loss = loss.view(batch_size, seq_len)
                    
                    # Mask out padding tokens (where target_ids == -100)
                    target_mask = (target_ids != -100).float()  # [batch_size, seq_len]
                    
                    # Compute per-sample loss: sum over sequence, divide by number of valid tokens
                    per_sample_loss = (loss * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp(min=1.0)  # [batch_size]
                    
                    # Apply loss weights (length-aware normalization)
                    loss = (per_sample_loss * loss_weights).mean()
                else:
                    # Scalar loss (shouldn't happen with reduce_loss=False, but handle it)
                    loss = loss * loss_weights.mean() if loss.dim() == 0 else loss.mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(1, num_batches)
            
            # Validation
            if val_data is not None:
                val_recall = self._evaluate_split(val_data, k=min(10, self.top_k))
                
                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    best_state = deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                print(f"[VIP5Reranker] Epoch {epoch+1}/{self.num_epochs} - "
                      f"loss: {avg_loss:.4f}, val_Recall@{min(10, self.top_k)}: {val_recall:.4f}")
                
                if self.patience and epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"[VIP5Reranker] Epoch {epoch+1}/{self.num_epochs} - loss: {avg_loss:.4f}")
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self.model.eval()
        self.is_fitted = True
        print(f"VIP5Reranker training completed. Model on device: {self.device}")
    
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
    
    def _prepare_training_samples(
        self,
        train_data: Dict[int, List[int]]
    ) -> List[Dict]:
        """Prepare training samples for VIP5 sequential task (A-1).
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            
        Returns:
            List of training samples with keys: user_id, history, target_item, source_text, target_text
        """
        samples = []
        
        # Sequential task template A-1
        template = "Given the following purchase history of user_{} : \n {} \n predict next possible item to be purchased by the user ?"
        
        for user_id, items in train_data.items():
            if len(items) < 2:
                continue  # Need at least 2 items for history and target
            
            # For training, randomly select a split point
            # History: items[0:end_pos], Target: items[end_pos]
            # Similar to VIP5 original implementation
            if len(items) > 6:
                end_candidates = list(range(max(2, len(items) - 6), len(items) - 1))
                end_pos = random.choice(end_candidates)
                start_candidates = list(range(1, min(4, end_pos)))
                start_pos = random.choice(start_candidates) if start_candidates else 1
            else:
                start_pos = 1
                end_pos = len(items) - 1
            
            purchase_history = items[start_pos:end_pos+1]
            target_item = items[end_pos+1] if end_pos+1 < len(items) else items[-1]
            
            # Format history: item_1, item_2, ...
            # Use visual token placeholders
            history_str = ", ".join([f"item_{item_id}" for item_id in purchase_history])
            
            # Build source text with visual token placeholders
            # Format: "item_1 <extra_id_0> <extra_id_0>, item_2 <extra_id_0> <extra_id_0>, ..."
            visual_token_placeholder = " <extra_id_0>" * self.image_feature_size_ratio
            history_with_visual = visual_token_placeholder.join([f"item_{item_id}" for item_id in purchase_history]) + visual_token_placeholder
            
            source_text = template.format(user_id, history_with_visual)
            target_text = f"item_{target_item}"
            
            samples.append({
                "user_id": user_id,
                "history": purchase_history,
                "target_item": target_item,
                "source_text": source_text,
                "target_text": target_text,
            })
        
        return samples
    
    def _prepare_batch(
        self,
        batch: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch for VIP5 training.
        
        Args:
            batch: List of training samples
            
        Returns:
            Dict with keys: input_ids, whole_word_ids, category_ids, vis_feats, target_ids, loss_weights
        """
        batch_size = len(batch)
        
        # Tokenize source texts
        source_texts = [s["source_text"] for s in batch]
        target_texts = [s["target_text"] for s in batch]
        
        # Tokenize inputs
        tokenized_inputs = self.tokenizer(
            source_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        input_ids = tokenized_inputs["input_ids"]  # [B, L]
        
        # Tokenize targets
        tokenized_targets = self.tokenizer(
            target_texts,
            padding="max_length",
            truncation=True,
            max_length=64,  # gen_max_length from VIP5
            return_tensors="pt"
        )
        target_ids = tokenized_targets["input_ids"]  # [B, L_t]
        
        # Calculate whole_word_ids
        whole_word_ids_list = []
        for source_text in source_texts:
            tokenized_text = self.tokenizer.tokenize(source_text)
            input_ids_text = self.tokenizer.encode(source_text, add_special_tokens=False)
            whole_word_ids = calculate_whole_word_ids(tokenized_text, input_ids_text, self.tokenizer)
            # Pad to max_length
            if len(whole_word_ids) < self.max_text_length:
                whole_word_ids = whole_word_ids + [0] * (self.max_text_length - len(whole_word_ids))
            else:
                whole_word_ids = whole_word_ids[:self.max_text_length]
            whole_word_ids_list.append(whole_word_ids)
        whole_word_ids = torch.LongTensor(whole_word_ids_list)  # [B, L]
        
        # Calculate category_ids (1 for <extra_id_0>, 0 for others)
        if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
            extra_id_0_token_id = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
        else:
            extra_id_0_token_id = self.tokenizer.encode('<extra_id_0>', add_special_tokens=False)[0]
        
        category_ids = (input_ids == extra_id_0_token_id).long()  # [B, L]
        
        # Prepare visual features
        max_vis_tokens = category_ids.sum(dim=1).max().item()
        vis_feats_list = []
        
        for sample in batch:
            history = sample["history"]
            # Get visual features for history items
            history_visual = []
            for item_id in history:
                if item_id in self.item_id_to_idx:
                    idx = self.item_id_to_idx[item_id]
                    history_visual.append(self.visual_embeddings[idx])
            
            if len(history_visual) == 0:
                # No valid items, use zeros
                vis_feats = torch.zeros(max_vis_tokens, self.image_feature_dim)
            else:
                # Stack and repeat for image_feature_size_ratio
                history_visual_tensor = torch.stack(history_visual)  # [num_items, feat_dim]
                # Repeat each item image_feature_size_ratio times
                vis_feats = history_visual_tensor.repeat_interleave(self.image_feature_size_ratio, dim=0)  # [num_items * ratio, feat_dim]
                
                # Pad or truncate to max_vis_tokens
                if vis_feats.size(0) < max_vis_tokens:
                    padding = torch.zeros(max_vis_tokens - vis_feats.size(0), self.image_feature_dim)
                    vis_feats = torch.cat([vis_feats, padding], dim=0)
                else:
                    vis_feats = vis_feats[:max_vis_tokens]
            
            vis_feats_list.append(vis_feats)
        
        vis_feats = torch.stack(vis_feats_list)  # [B, max_vis_tokens, feat_dim]
        
        # Loss weights (length-aware normalization)
        target_lengths = (target_ids != self.tokenizer.pad_token_id).sum(dim=1).float()
        loss_weights = 1.0 / target_lengths.clamp(min=1.0)  # [B]
        
        # Mask target_ids: set padding to -100 (ignore in loss)
        target_mask = (target_ids != self.tokenizer.pad_token_id)
        target_ids_masked = target_ids.clone()
        target_ids_masked[~target_mask] = -100
        
        return {
            "input_ids": input_ids,
            "whole_word_ids": whole_word_ids,
            "category_ids": category_ids,
            "vis_feats": vis_feats,
            "target_ids": target_ids_masked,
            "loss_weights": loss_weights,
        }
    
    def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> float:
        """Compute average Recall@K for validation.
        
        Args:
            split: Dict {user_id: [gt_item_ids]} - validation/test split
            k: Top-K for recall calculation
            
        Returns:
            Average Recall@K
        """
        if self.model is None or self.tokenizer is None:
            return 0.0
        
        self.model.eval()
        recalls = []
        
        with torch.no_grad():
            for user_id, gt_items in split.items():
                if user_id not in self.user_history:
                    continue
                
                # Get user history
                history = self.user_history[user_id]
                if len(history) == 0:
                    continue
                
                # Get all items as candidates (for evaluation)
                all_items = list(self.item_id_to_idx.keys())
                
                if not all_items:
                    continue
                
                # Limit candidates for efficiency (during training)
                max_eval_candidates = min(100, len(all_items))
                candidates = random.sample(all_items, max_eval_candidates) if len(all_items) > max_eval_candidates else all_items
                
                # Ensure at least one ground truth is in candidates
                if not any(item in candidates for item in gt_items):
                    # Add one ground truth item
                    candidates[0] = gt_items[0]
                
                # Rerank candidates using internal logic (bypass validation check)
                # Encode user history + each candidate separately (same as rerank method)
                scores = []
                
                for item_id in candidates:
                    if item_id not in self.item_id_to_idx:
                        continue
                    
                    # Build prompt for this specific candidate
                    candidate_prompt = build_rerank_prompt(
                        user_id,
                        history,
                        [item_id],  # Single candidate
                        template="Given the following purchase history of user_{} : \n {} \n predict next possible item to be purchased by the user ?"
                    )
                    
                    # Get visual feature for this candidate
                    idx = self.item_id_to_idx[item_id]
                    item_visual = self.visual_embeddings[idx].unsqueeze(0)  # [1, feat_dim]
                    
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
                    encoder_outputs = self.model.encoder(
                        input_ids=input_ids,
                        whole_word_ids=whole_word_ids,
                        category_ids=category_ids,
                        vis_feats=vis_feats,
                        attention_mask=attention_mask,
                        return_dict=True,
                        task="sequential",
                    )
                    
                    # Get encoder hidden states [1, seq_len, d_model]
                    encoder_hidden = encoder_outputs.last_hidden_state
                    
                    # Use mean pooling of encoder output as score
                    score = float(encoder_hidden.mean(dim=1).squeeze(0).norm().item())
                    
                    scores.append((item_id, score))
                
                # Sort by score descending
                scores.sort(key=lambda x: x[1], reverse=True)
                
                # Get top-K item IDs
                top_k_items = [item_id for item_id, _ in scores[:k]]
                
                # Compute recall
                hits = len(set(top_k_items) & set(gt_items))
                if len(gt_items) > 0:
                    recalls.append(hits / min(k, len(gt_items)))
        
        return float(np.mean(recalls)) if recalls else 0.0

