"""Qwen3-VL-based reranker with multiple prompt modes."""

from typing import Dict, List, Tuple, Any, Optional
import random
import numpy as np
import torch
import os
from copy import deepcopy
from PIL import Image

from rerank.base import BaseReranker
from rerank.models.qwen3vl import Qwen3VLModel, resize_image_for_qwen3vl
from rerank.models.llm import rank_candidates
from evaluation.metrics import recall_at_k


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
        batch_size: int = 32,
        num_epochs: int = 10,
        lr: float = 1e-4,
        patience: Optional[int] = None,
    ) -> None:
        """
        Args:
            top_k: Số lượng items trả về sau rerank
            mode: Prompt mode - "raw_image", "caption", "semantic_summary", "semantic_summary_small"
            model_name: Model name (auto-selected based on mode if None)
            max_history: Số lượng items trong history để dùng cho prompt
            max_candidates: Maximum number of candidates to process (None = no limit)
            batch_size: Batch size cho training (default: 32)
            num_epochs: Số epochs (default: 10)
            lr: Learning rate (default: 1e-4)
            patience: Early stopping patience (None = no early stopping)
        """
        super().__init__(top_k=top_k)
        self.mode = mode.lower()
        self.model_name = model_name
        self.max_history = max_history
        self.max_candidates = max_candidates
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.patience = patience
        self.qwen3vl_model: Optional[Qwen3VLModel] = None
        
        # Data structures cần thiết cho rerank
        self.user_history: Dict[int, List[str]] = {}  # user_id -> [item_texts/images]
        self.item_meta: Dict[int, Dict[str, Any]] = {}  # item_id -> {image_path, caption, semantic_summary, text}
        self.train_user_history: Dict[int, List[int]] = {}  # user_id -> [item_ids] for training
    
    def fit(
        self,
        train_data: Dict[int, List[int]],
        **kwargs: Any
    ) -> None:
        """Fit reranker with training.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            **kwargs: Additional arguments:
                - item_meta: Dict[int, Dict] - mapping item_id -> {image_path, caption, semantic_summary, text}
                - user_history: Dict[int, List[str]] - user history texts/images
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
        
        # Load item_meta và user_history từ kwargs
        self.item_meta = kwargs.get("item_meta", {})
        self.user_history = kwargs.get("user_history", {})
        val_data = kwargs.get("val_data")
        
        # Store training data
        self.train_user_history = train_data
        
        # Load Qwen3-VL model
        self.qwen3vl_model = Qwen3VLModel(
            mode=self.mode,
            model_name=self.model_name
        )
        
        # Training support cho tất cả modes
        # semantic_summary_small: text model với tokenizer (Unsloth)
        # raw_image, caption, semantic_summary: VL model với processor (transformers Trainer)
        if self.mode in ["raw_image", "caption", "semantic_summary", "semantic_summary_small"]:
            # Prepare training samples
            print("Preparing training samples...")
            train_samples = self._prepare_training_samples(train_data)
            
            if len(train_samples) == 0:
                print("Warning: No training samples generated. Skipping training.")
                self.is_fitted = True
                return
            
            print(f"Generated {len(train_samples)} training samples")
            
            # Train model
            self._train_model(train_samples, val_data)
        else:
            # raw_image mode: không train, chỉ load pretrained
            print(f"Training not supported for mode '{self.mode}'. Using pretrained model only.")
        
        self.is_fitted = True
    
    def _rerank_internal(
        self,
        user_id: int,
        candidates: List[int],
        user_history: Optional[List[int]] = None,
        **kwargs: Any
    ) -> List[Tuple[int, float]]:
        """Internal rerank method without validation check (for use during training).
        
        Args:
            user_id: ID của user
            candidates: List các item IDs cần rerank
            user_history: Optional user history (if not provided, uses self.user_history)
            **kwargs: Additional arguments (không dùng ở đây)
            
        Returns:
            List[Tuple[int, float]]: [(item_id, score)] đã sort giảm dần
        """
        if self.qwen3vl_model is None:
            raise RuntimeError("Qwen3-VL model chưa được load. Gọi fit() trước!")
        
        if not candidates:
            return []
        
        # Apply max_candidates limit if set
        if self.max_candidates is not None and len(candidates) > self.max_candidates:
            candidates = candidates[:self.max_candidates]
        
        # Lấy user history
        if user_history is None:
            history = self.user_history.get(user_id, [])
        else:
            history = user_history
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
            **kwargs: Additional arguments:
                - user_history: List[int] - user's interaction history (optional)
            
        Returns:
            List[Tuple[int, float]]: [(item_id, score)] đã sort giảm dần
        """
        self._validate_fitted()
        
        # Extract user_history from kwargs if provided
        user_history = kwargs.get("user_history")
        
        return self._rerank_internal(user_id, candidates, user_history=user_history, **kwargs)
    
    def _prepare_training_samples(
        self,
        train_data: Dict[int, List[int]]
    ) -> List[Dict]:
        """Prepare training samples for Qwen3-VL.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            
        Returns:
            List of training samples with keys: user_id, history, candidates, target_item
        """
        samples = []
        all_items = set()
        for items in train_data.values():
            all_items.update(items)
        
        for user_id, items in train_data.items():
            if len(items) < 2:
                continue  # Need at least 2 items for history and target
            
            # For training, randomly select a split point
            # History: items[0:end_pos], Target: items[end_pos]
            if len(items) > 3:
                end_pos = random.randint(1, len(items) - 1)
            else:
                end_pos = len(items) - 1
            
            history = items[:end_pos]
            target_item = items[end_pos]
            
            # Generate candidates: target + random negatives
            # Similar to ground_truth mode: 1 positive + 19 negatives
            user_items_set = set(items)
            negative_candidates = [item for item in all_items if item not in user_items_set]
            
            num_negatives = min(19, len(negative_candidates))
            if num_negatives > 0:
                negatives = random.sample(negative_candidates, num_negatives)
            else:
                negatives = []
            
            candidates = [target_item] + negatives
            random.shuffle(candidates)  # Shuffle so target is not always first
            
            # Find target index in candidates
            target_idx = candidates.index(target_item)
            
            samples.append({
                "user_id": user_id,
                "history": history,
                "candidates": candidates,
                "target_item": target_item,
                "target_idx": target_idx,  # 0-indexed (will be converted to 1-indexed in prompt)
            })
        
        return samples
    
    def _train_model(
        self,
        train_samples: List[Dict],
        val_data: Optional[Dict[int, List[int]]] = None
    ) -> None:
        """Train Qwen3-VL model with Unsloth.
        
        Args:
            train_samples: List of training samples
            val_data: Optional validation data
        """
        if self.qwen3vl_model is None:
            raise RuntimeError("Qwen3VLModel not initialized")
        
        # Check model type and train accordingly
        if self.mode == "semantic_summary_small":
            # Text model with tokenizer (Unsloth)
            self._train_text_model(train_samples, val_data)
        else:
            # VL model with processor (raw_image, caption, semantic_summary)
            self._train_vl_model(train_samples, val_data)
    
    def _train_text_model(
        self,
        train_samples: List[Dict],
        val_data: Optional[Dict[int, List[int]]] = None
    ) -> None:
        """Train text model (semantic_summary_small mode) with Unsloth.
        
        Args:
            train_samples: List of training samples
            val_data: Optional validation data
        """
        if not hasattr(self.qwen3vl_model, 'model') or not hasattr(self.qwen3vl_model, 'tokenizer'):
            print("Warning: Text model does not support training. Using pretrained model only.")
            return
        
        from datasets import Dataset
        from transformers import TrainingArguments, Trainer
        
        # Prepare training data
        training_data = []
        for sample in train_samples:
            prompt = self._build_training_prompt(sample)
            target = str(sample["target_idx"] + 1)  # 1-indexed
            
            training_data.append({
                "instruction": prompt,
                "input": "",
                "output": target,
            })
        
        hf_train_dataset = Dataset.from_list(training_data)
        
        # Tokenize
        def tokenize_function(examples):
            texts = []
            for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
                text = f"{inst}\n{out}" if not inp else f"{inst}\n{inp}\n{out}"
                texts.append(text)
            
            tokenized = self.qwen3vl_model.tokenizer(
                texts,
                truncation=True,
                max_length=2048,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        hf_train_dataset = hf_train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=hf_train_dataset.column_names,
        )
        
        # Training arguments - set num_train_epochs=1 because we'll call trainer.train() in a loop
        training_args = TrainingArguments(
            output_dir="./qwen3vl_rerank",
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=self.lr,
            num_train_epochs=1,  # ✅ Set to 1 - we'll train 1 epoch per loop iteration
            logging_steps=50,
            save_steps=500,
            report_to="none",
            fp16=True,
            optim="adamw_torch",
        )
        
        trainer = Trainer(
            model=self.qwen3vl_model.model,
            args=training_args,
            train_dataset=hf_train_dataset,
            tokenizer=self.qwen3vl_model.tokenizer,
        )
        
        # Training loop
        self._run_training_loop(trainer, val_data)
    
    def _train_vl_model(
        self,
        train_samples: List[Dict],
        val_data: Optional[Dict[int, List[int]]] = None
    ) -> None:
        """Train VL model (raw_image, caption, semantic_summary modes) with Unsloth.
        
        Uses Unsloth's SFTTrainer and UnslothVisionDataCollator for proper vision model training.
        
        Args:
            train_samples: List of training samples
            val_data: Optional validation data
        """
        if not hasattr(self.qwen3vl_model, 'model') or not hasattr(self.qwen3vl_model, 'processor'):
            print("Warning: VL model does not support training. Using pretrained model only.")
            return
        
        from datasets import Dataset
        
        # Try to use Unsloth's training API for vision models
        try:
            from unsloth.trainer import UnslothVisionDataCollator
            from trl import SFTTrainer, SFTConfig
            from rerank.models.qwen3vl import FAST_VISION_MODEL_AVAILABLE
            USE_UNSLOTH_TRAINER = FAST_VISION_MODEL_AVAILABLE
        except ImportError:
            print("Warning: Unsloth trainer not available. Using standard transformers Trainer.")
            USE_UNSLOTH_TRAINER = False
            from transformers import TrainingArguments, Trainer
        
        # Prepare training data in Unsloth format (messages format)
        # Note: For raw_image mode, we store item_ids and load images dynamically
        # because PIL Images cannot be serialized in HuggingFace Dataset
        training_data = []
        for sample in train_samples:
            target = str(sample["target_idx"] + 1)  # 1-indexed
            
            if self.mode == "raw_image":
                # For raw_image, store item_ids and load images in collate_fn
                # UnslothVisionDataCollator will handle the image loading
                training_data.append({
                    "user_id": sample["user_id"],
                    "history": sample["history"],
                    "candidates": sample["candidates"],
                    "target": target,
                })
            else:
                # For caption and semantic_summary, create text-only messages
                prompt = self._build_training_prompt(sample)
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target}
                ]
                # Store both messages (for Unsloth) and prompt/target (for fallback)
                training_data.append({
                    "messages": messages,
                    "prompt": prompt,
                    "target": target,
                })
        
        hf_train_dataset = Dataset.from_list(training_data)
        
        # Custom data collator for VL model
        def collate_fn(batch):
            """Custom collate function for Qwen3-VL training.
            
            Handles both text-only prompts (caption, semantic_summary) and 
            multimodal prompts with images (raw_image mode).
            """
            processor = self.qwen3vl_model.processor
            
            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []
            
            for item in batch:
                target = item["target"]
                
                # Check if we have messages (raw_image mode) or prompt (text-only mode)
                if "candidates" in item:
                    # raw_image mode: need to build messages with images from item_ids
                    # Load images fresh in collate_fn to avoid serialization issues
                    sample = {
                        "user_id": item.get("user_id"),
                        "history": item.get("history", []),
                        "candidates": item.get("candidates", []),
                    }
                    prompt_data = self._build_training_prompt_with_images(sample)
                    messages = prompt_data["messages"]
                elif "messages" in item:
                    # raw_image mode: messages already built (shouldn't happen with new approach)
                    messages = item["messages"]
                    # Ensure images are PIL Images, not dicts
                    for msg in messages:
                        if isinstance(msg.get("content"), list):
                            for content_item in msg["content"]:
                                if isinstance(content_item, dict) and content_item.get("type") == "image":
                                    img = content_item.get("image")
                                    if not isinstance(img, Image.Image):
                                        # Try to convert from dict/string if needed
                                        if isinstance(img, dict):
                                            # Image was serialized, skip this item
                                            content_item["image"] = None
                                        elif isinstance(img, str):
                                            # Image path, load it
                                            try:
                                                content_item["image"] = Image.open(img).convert("RGB")
                                                content_item["image"] = resize_image_for_qwen3vl(content_item["image"], max_size=448)
                                            except:
                                                content_item["image"] = None
                else:
                    # text-only mode: create message from prompt
                    prompt = item["prompt"]
                    messages = [{"role": "user", "content": prompt}]
                
                # Apply chat template (handles both text and multimodal inputs)
                # Note: Keep tensors on CPU - Trainer will handle device placement and pin memory
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                
                # Ensure inputs are on CPU (in case processor returns GPU tensors)
                def ensure_cpu(obj):
                    """Recursively ensure tensors are on CPU."""
                    if isinstance(obj, torch.Tensor):
                        return obj.cpu()
                    elif isinstance(obj, dict):
                        return {k: ensure_cpu(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(ensure_cpu(item) for item in obj)
                    else:
                        return obj
                
                inputs = ensure_cpu(inputs)
                
                input_ids = inputs["input_ids"].squeeze(0)  # [seq_len]
                attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
                if attention_mask.dim() > 1:
                    attention_mask = attention_mask.squeeze(0)
                
                # Tokenize target
                target_tokens = processor.tokenizer.encode(
                    target,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).squeeze(0)  # [target_len]
                
                # Create full sequence: input_ids + target_tokens
                full_input_ids = torch.cat([input_ids, target_tokens])
                full_attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(len(target_tokens), dtype=attention_mask.dtype)
                ])
                
                # Create labels: -100 for input, target tokens for output
                labels = torch.cat([
                    torch.full_like(input_ids, -100),
                    target_tokens
                ])
                
                batch_input_ids.append(full_input_ids)
                batch_attention_mask.append(full_attention_mask)
                batch_labels.append(labels)
            
            # Pad to max length
            max_len = max(ids.size(0) for ids in batch_input_ids)
            pad_token_id = processor.tokenizer.pad_token_id
            
            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []
            
            for input_ids, attn_mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
                pad_len = max_len - input_ids.size(0)
                if pad_len > 0:
                    input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)])
                    attn_mask = torch.cat([attn_mask, torch.zeros((pad_len,), dtype=attn_mask.dtype)])
                    labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])
                
                padded_input_ids.append(input_ids)
                padded_attention_mask.append(attn_mask)
                padded_labels.append(labels)
            
            # Return tensors on CPU - Trainer will handle device placement and pin memory
            return {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(padded_attention_mask),
                "labels": torch.stack(padded_labels),
            }
        
        # Use Unsloth's training API if available
        # Check if model is actually a vision model (required for UnslothVisionDataCollator)
        # UnslothVisionDataCollator only works with vision models loaded via FastVisionModel
        is_vision_model = False
        try:
            from unsloth import FastVisionModel
            # Check if model is wrapped in FastVisionModel or is a Qwen3VL model
            model_to_check = self.qwen3vl_model.model
            if hasattr(model_to_check, 'model'):  # PEFT models wrap the base model
                model_to_check = model_to_check.model
            if hasattr(model_to_check, '__class__'):
                model_class_name = model_to_check.__class__.__name__
                # Check if it's a Qwen3VL/Qwen2VL model (which FastVisionModel wraps)
                if 'Qwen3VL' in model_class_name or 'Qwen2VL' in model_class_name:
                    is_vision_model = True
        except Exception as e:
            print(f"Warning: Could not check vision model status: {e}")
        
        # Only use UnslothVisionDataCollator if model is actually a vision model
        # For text-only modes with non-vision models, fallback to standard Trainer
        if USE_UNSLOTH_TRAINER and is_vision_model:
            print("Using Unsloth SFTTrainer for vision model training...")
            
            # Enable training mode for FastVisionModel
            try:
                from rerank.models.qwen3vl import FastVisionModel
                if hasattr(FastVisionModel, 'for_training'):
                    FastVisionModel.for_training(self.qwen3vl_model.model)
                    print("Enabled training mode for FastVisionModel")
            except Exception as e:
                print(f"Warning: Could not enable training mode: {e}")
            
            # Custom data collator that handles raw_image mode (loads images from item_ids)
            # For text-only modes, UnslothVisionDataCollator should work with vision models
            if self.mode == "raw_image":
                # For raw_image mode, we need to build messages with images from item_ids
                # Track if we've already warned about UnslothVisionDataCollator failure
                _unsloth_warned = [False]  # Use list to allow modification in nested function
                
                def custom_collate_fn(batch):
                    """Custom collate function that loads images and uses UnslothVisionDataCollator."""
                    # Build messages with images for each item in batch
                    messages_batch = []
                    batch_with_target = []
                    for item in batch:
                        sample = {
                            "user_id": item.get("user_id"),
                            "history": item.get("history", []),
                            "candidates": item.get("candidates", []),
                        }
                        prompt_data = self._build_training_prompt_with_images(sample)
                        messages = prompt_data["messages"]
                        # Add assistant response
                        target = item.get("target", "")
                        messages.append({"role": "assistant", "content": target})
                        messages_batch.append({"messages": messages})
                        # Also keep original item format for fallback
                        batch_with_target.append({
                            "user_id": item.get("user_id"),
                            "history": item.get("history", []),
                            "candidates": item.get("candidates", []),
                            "target": target,
                        })
                    
                    # Use UnslothVisionDataCollator to process the messages
                    try:
                        data_collator = UnslothVisionDataCollator(
                            self.qwen3vl_model.model,
                            self.qwen3vl_model.processor.tokenizer
                        )
                        return data_collator(messages_batch)
                    except (TypeError, ValueError) as e:
                        # If UnslothVisionDataCollator fails, fallback to standard collate
                        # Only warn once to avoid spam
                        if not _unsloth_warned[0]:
                            print(f"Warning: UnslothVisionDataCollator failed: {e}")
                            print("Falling back to standard collate function...")
                            _unsloth_warned[0] = True
                        # Use batch_with_target which has the correct format for collate_fn
                        return collate_fn(batch_with_target)
            else:
                # For text-only modes, try UnslothVisionDataCollator (only works with vision models)
                # But also prepare a fallback collate_fn in case it fails
                # Track if we've already warned about UnslothVisionDataCollator failure
                _text_unsloth_warned = [False]  # Use list to allow modification in nested function
                
                def text_only_collate_fn(batch):
                    """Custom collate function for text-only modes with fallback."""
                    # Try UnslothVisionDataCollator first
                    try:
                        data_collator = UnslothVisionDataCollator(
                            self.qwen3vl_model.model,
                            self.qwen3vl_model.processor.tokenizer
                        )
                        # Extract messages from batch
                        messages_batch = [{"messages": item.get("messages", [])} for item in batch]
                        return data_collator(messages_batch)
                    except (TypeError, ValueError) as e:
                        # Fallback to standard collate_fn
                        # Only warn once to avoid spam
                        if not _text_unsloth_warned[0]:
                            print(f"Warning: UnslothVisionDataCollator failed for text-only mode: {e}")
                            print("Falling back to standard collate function...")
                            _text_unsloth_warned[0] = True
                        # Use standard collate_fn which expects prompt and target
                        return collate_fn(batch)
                
                try:
                    # Test if UnslothVisionDataCollator works
                    test_collator = UnslothVisionDataCollator(
                        self.qwen3vl_model.model,
                        self.qwen3vl_model.processor.tokenizer
                    )
                    # If it works, use text_only_collate_fn which will use it
                    data_collator = text_only_collate_fn
                except (TypeError, ValueError) as e:
                    print(f"Warning: UnslothVisionDataCollator failed for text-only mode: {e}")
                    print("Falling back to standard Trainer...")
                    USE_UNSLOTH_TRAINER = False  # Force fallback
                    data_collator = None
            
            if USE_UNSLOTH_TRAINER:
                # Training config with vision-specific settings
                # Set num_train_epochs=1 because we'll call trainer.train() in a loop
                training_args = SFTConfig(
                    per_device_train_batch_size=self.batch_size,
                    gradient_accumulation_steps=1,
                    warmup_steps=5,
                    num_train_epochs=1,  # ✅ Set to 1 - we'll train 1 epoch per loop iteration
                    learning_rate=self.lr,
                    logging_steps=50,
                    optim="adamw_8bit",  # Use 8-bit optimizer for memory efficiency
                    weight_decay=0.001,
                    lr_scheduler_type="linear",
                    seed=3407,
                    output_dir="./qwen3vl_rerank_vl",
                    report_to="none",
                    
                    # REQUIRED for vision finetuning:
                    remove_unused_columns=False,
                    dataset_text_field="",
                    dataset_kwargs={"skip_prepare_dataset": True},
                    max_length=2048,
                )
                
                # Use SFTTrainer with UnslothVisionDataCollator
                if self.mode == "raw_image":
                    # For raw_image mode, use custom collate_fn
                    trainer = SFTTrainer(
                        model=self.qwen3vl_model.model,
                        tokenizer=self.qwen3vl_model.processor.tokenizer,
                        data_collator=custom_collate_fn,  # Custom collate function for raw_image
                        train_dataset=hf_train_dataset,
                        args=training_args,
                    )
                else:
                    # For text-only modes, use UnslothVisionDataCollator directly
                    trainer = SFTTrainer(
                        model=self.qwen3vl_model.model,
                        tokenizer=self.qwen3vl_model.processor.tokenizer,
                        data_collator=data_collator,  # Must use UnslothVisionDataCollator!
                        train_dataset=hf_train_dataset,
                        args=training_args,
                    )
            else:
                # Fallback to standard Trainer
                USE_UNSLOTH_TRAINER = False
        else:
            # Model is not a vision model or Unsloth trainer not available
            if USE_UNSLOTH_TRAINER and not is_vision_model:
                print("Model is not a vision model, using standard Trainer...")
            USE_UNSLOTH_TRAINER = False
        
        if not USE_UNSLOTH_TRAINER:
            # Fallback: Use standard transformers Trainer with custom collate_fn
            print("Using standard transformers Trainer (fallback)...")
            
            # Tokenize dataset (simplified - we'll use collate_fn for full processing)
            # For raw_image mode, we already have messages, no need to map
            # For text-only modes, we need to keep prompts
            if self.mode != "raw_image":
                def prepare_dataset(examples):
                    # Store prompts and targets for collate_fn
                    result = {
                        "target": examples.get("target", []),
                    }
                    if "prompt" in examples:
                        result["prompt"] = examples["prompt"]
                    return result
                
                hf_train_dataset = hf_train_dataset.map(
                    prepare_dataset,
                    batched=True,
                )
            
            # Training arguments - set num_train_epochs=1 because we'll call trainer.train() in a loop
            training_args = TrainingArguments(
                output_dir="./qwen3vl_rerank_vl",
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=1,
                learning_rate=self.lr,
                num_train_epochs=1,  # ✅ Set to 1 - we'll train 1 epoch per loop iteration
                logging_steps=50,
                save_steps=500,
                report_to="none",
                fp16=True,
                optim="adamw_torch",
                dataloader_pin_memory=True,  # Enable pin memory for faster GPU transfer
            )
            
            # Custom trainer with our collate function
            class CustomTrainer(Trainer):
                def get_train_dataloader(self):
                    from torch.utils.data import DataLoader
                    return DataLoader(
                        self.train_dataset,
                        batch_size=self.args.per_device_train_batch_size,
                        collate_fn=collate_fn,
                        shuffle=True,
                    )
            
            trainer = CustomTrainer(
                model=self.qwen3vl_model.model,
                args=training_args,
                train_dataset=hf_train_dataset,
                tokenizer=self.qwen3vl_model.processor.tokenizer,
            )
        
        # Set model to train mode
        self.qwen3vl_model.model.train()
        
        # Training loop
        self._run_training_loop(trainer, val_data)
    
    def _run_training_loop(
        self,
        trainer,
        val_data: Optional[Dict[int, List[int]]] = None
    ) -> None:
        """Run training loop with validation.
        
        Args:
            trainer: Trainer instance
            val_data: Optional validation data
        """
        best_val_recall = -1.0
        epochs_no_improve = 0
        best_model_state = None
        
        for epoch in range(self.num_epochs):
            # Train
            print(f"[Qwen3VLReranker] Training epoch {epoch+1}/{self.num_epochs}...")
            trainer.train()
            
            # Validation
            if val_data is not None:
                val_recall = self._evaluate_split(val_data, k=min(10, self.top_k))
                
                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    best_model_state = deepcopy(self.qwen3vl_model.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                print(f"[Qwen3VLReranker] Epoch {epoch+1}/{self.num_epochs} - "
                      f"val_Recall@{min(10, self.top_k)}: {val_recall:.4f}")
                
                if self.patience and epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.qwen3vl_model.model.load_state_dict(best_model_state)
            print(f"Loaded best model with val_Recall@{min(10, self.top_k)}: {best_val_recall:.4f}")
        
        # Set back to eval mode
        self.qwen3vl_model.model.eval()
    
    def _build_training_prompt_with_images(self, sample: Dict) -> Dict:
        """Build training prompt with raw images for raw_image mode.
        
        NOTE: This is the ONLY method that loads and uses images directly.
        Other modes (caption, semantic_summary) use _build_training_prompt() which only uses text.
        
        Args:
            sample: Training sample
            
        Returns:
            Dict with messages (containing images) and target
        """
        history = sample["history"]
        candidates = sample["candidates"]
        
        # Build history text (text-only, consistent with inference)
        history_texts = []
        for item_id in history[-self.max_history:]:
            meta = self.item_meta.get(item_id, {})
            text = meta.get("text", f"item_{item_id}")
            history_texts.append(text)
        
        history_text = "\n".join([f"- {h}" for h in history_texts]) if history_texts else "No previous interactions."
        
        # Build candidate images and texts
        candidate_images = []
        candidate_texts = []
        for item_id in candidates:
            meta = self.item_meta.get(item_id, {})
            text = meta.get("text", f"item_{item_id}")
            image_path = meta.get("image_path") or meta.get("image")
            
            if image_path and os.path.isfile(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    # Resize image for Qwen3-VL (max 448px on longer side)
                    img = resize_image_for_qwen3vl(img, max_size=448)
                    candidate_images.append(img)
                    candidate_texts.append(text)
                except Exception:
                    candidate_images.append(None)
                    candidate_texts.append(text)
            else:
                candidate_images.append(None)
                candidate_texts.append(text)
        
        # Build content with images (consistent with inference format)
        content = [
            {"type": "text", "text": f"""You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
{history_text}

Candidate items:"""}
        ]
        
        # Add candidate images and text labels
        num_candidates = len(candidates)
        for i, (text, img) in enumerate(zip(candidate_texts, candidate_images)):
            if img is not None:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": f"{i+1}. {text}"})
        
        content.append({
            "type": "text",
            "text": f"\nAnswer with only one number (1-{num_candidates})."
        })
        
        messages = [{"role": "user", "content": content}]
        
        return {"messages": messages}
    
    def _build_training_prompt_text_only(self, sample: Dict) -> Dict:
        """Build text-only training prompt for caption and semantic_summary modes.
        
        Args:
            sample: Training sample
            
        Returns:
            Dict with prompt
        """
        prompt = self._build_training_prompt(sample)
        return {"prompt": prompt}
    
    def _build_training_prompt(self, sample: Dict) -> str:
        """Build training prompt from sample.
        
        NOTE: This method does NOT load images. It only uses text, caption, or semantic_summary.
        Only raw_image mode uses _build_training_prompt_with_images() which loads images directly.
        
        Args:
            sample: Training sample with history, candidates, target_item
            
        Returns:
            Formatted prompt string
        """
        user_id = sample["user_id"]
        history = sample["history"]
        candidates = sample["candidates"]
        
        # Get history texts based on mode
        history_texts = []
        for item_id in history[-self.max_history:]:
            meta = self.item_meta.get(item_id, {})
            if self.mode == "caption":
                caption = meta.get("caption", "")
                text = meta.get("text", f"item_{item_id}")
                if caption:
                    history_texts.append(f"{text} (Image: {caption})")
                else:
                    history_texts.append(text)
            elif self.mode in ["semantic_summary", "semantic_summary_small"]:
                semantic_summary = meta.get("semantic_summary", "")
                text = meta.get("text", f"item_{item_id}")
                if semantic_summary:
                    history_texts.append(f"{text} (Semantic: {semantic_summary})")
                else:
                    history_texts.append(text)
            else:
                text = meta.get("text", f"item_{item_id}")
                history_texts.append(text)
        
        # Get candidate texts
        candidate_texts = []
        for item_id in candidates:
            meta = self.item_meta.get(item_id, {})
            if self.mode == "caption":
                caption = meta.get("caption", "")
                text = meta.get("text", f"item_{item_id}")
                if caption:
                    candidate_texts.append(f"{text} (Image: {caption})")
                else:
                    candidate_texts.append(text)
            elif self.mode in ["semantic_summary", "semantic_summary_small"]:
                semantic_summary = meta.get("semantic_summary", "")
                text = meta.get("text", f"item_{item_id}")
                if semantic_summary:
                    candidate_texts.append(f"{text} (Semantic: {semantic_summary})")
                else:
                    candidate_texts.append(text)
            else:
                text = meta.get("text", f"item_{item_id}")
                candidate_texts.append(text)
        
        # Build prompt
        history_str = "\n".join([f"- {h}" for h in history_texts]) if history_texts else "No previous interactions."
        cand_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidate_texts)])
        num_candidates = len(candidates)
        
        prompt = f"""You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
{history_str}

Candidate items:
{cand_str}

Answer with only one number (1-{num_candidates}).
""".strip()
        
        return prompt
    
    def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> float:
        """Compute average Recall@K for validation.
        
        Args:
            split: Dict {user_id: [gt_item_ids]} - validation/test split
            k: Top-K for recall calculation
            
        Returns:
            Average Recall@K
        """
        if self.qwen3vl_model is None:
            return 0.0
        
        recalls = []
        
        for user_id, gt_items in split.items():
            if user_id not in self.train_user_history:
                continue
            
            # Get user history
            history = self.train_user_history[user_id]
            if len(history) == 0:
                continue
            
            # Get all items as candidates (for evaluation)
            all_items = set()
            for items in self.train_user_history.values():
                all_items.update(items)
            all_items = list(all_items)
            
            if not all_items:
                # Fallback: use all items from item_meta
                all_items = list(self.item_meta.keys())
            
            if not all_items:
                continue
            
            # ✅ FIX: Exclude history items from candidate pool (avoid recommending already purchased items)
            history_set = set(history)
            candidate_pool = [item for item in all_items if item not in history_set]
            
            if not candidate_pool:
                continue  # No candidates available after excluding history
            
            # Limit candidates for efficiency
            # Get eval_candidates from config (default: 20)
            try:
                from config import arg
                max_eval_candidates = getattr(arg, 'rerank_eval_candidates', 20)
            except ImportError:
                max_eval_candidates = 20
            max_eval_candidates = min(max_eval_candidates, len(candidate_pool))
            candidates = random.sample(candidate_pool, max_eval_candidates) if len(candidate_pool) > max_eval_candidates else candidate_pool
            
            # Ensure at least one ground truth is in candidates
            if not any(item in candidates for item in gt_items):
                # Add one ground truth item
                candidates[0] = gt_items[0]
            
            # ✅ Shuffle candidates to avoid bias (GT item should not always be first)
            random.shuffle(candidates)
            
            # Rerank (use _rerank_internal to bypass validation check during training)
            reranked = self._rerank_internal(user_id, candidates, user_history=history)
            
            # Get top-K item IDs
            top_k_items = [item_id for item_id, _ in reranked[:k]]
            
            # Compute recall
            hits = len(set(top_k_items) & set(gt_items))
            if len(gt_items) > 0:
                recalls.append(hits / min(k, len(gt_items)))
        
        return float(np.mean(recalls)) if recalls else 0.0

