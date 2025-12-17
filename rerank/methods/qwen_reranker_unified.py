"""Unified Qwen reranker supporting text-only and multimodal modes."""

from typing import Dict, List, Tuple, Any, Optional
import random
import numpy as np
import torch
from copy import deepcopy

from rerank.base import BaseReranker
from rerank.models.llm import LLMModel, build_prompt_from_candidates, rank_candidates
from rerank.models.qwen3vl import Qwen3VLModel
from evaluation.metrics import recall_at_k


def _truncate_item_text(text: str, max_chars: int = 200) -> str:
    """Truncate item text metadata to prevent it from being too long."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def _count_prompt_tokens(
    prompts: List[str],
    tokenizer,
    include_target: bool = False,
    targets: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Count tokens in prompts and return statistics.
    
    Args:
        prompts: List of prompt strings
        tokenizer: Tokenizer to use for counting
        include_target: Whether to include target tokens in count
        targets: List of target strings (required if include_target=True)
    
    Returns:
        Dict with statistics: min, max, mean, median, percentiles, token_counts list, etc.
    """
    token_counts = []
    
    for i, prompt in enumerate(prompts):
        # Tokenize prompt
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_count = len(prompt_tokens)
        
        # Add target tokens if requested
        if include_target and targets and i < len(targets):
            target = targets[i]
            target_tokens = tokenizer.encode(target, add_special_tokens=False)
            prompt_count += len(target_tokens)
        
        token_counts.append(prompt_count)
    
    if not token_counts:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "total": 0,
            "count": 0,
            "token_counts": [],
        }
    
    token_counts_arr = np.array(token_counts)
    
    return {
        "min": int(np.min(token_counts_arr)),
        "max": int(np.max(token_counts_arr)),
        "mean": float(np.mean(token_counts_arr)),
        "median": float(np.median(token_counts_arr)),
        "p50": float(np.percentile(token_counts_arr, 50)),
        "p75": float(np.percentile(token_counts_arr, 75)),
        "p90": float(np.percentile(token_counts_arr, 90)),
        "p95": float(np.percentile(token_counts_arr, 95)),
        "p99": float(np.percentile(token_counts_arr, 99)),
        "total": int(np.sum(token_counts_arr)),
        "count": len(token_counts_arr),
        "token_counts": token_counts,  # Keep original list for checking exceeding
    }


# Model name mappings
MODEL_MAPPING = {
    "qwen3-0.6b": "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    "qwen3-2bvl": "unsloth/Qwen3-VL-2B-Instruct",
    "qwen3-1.6b": "unsloth/Qwen3-1.6B-unsloth-bnb-4bit",  # Note: Adjust model path if different
}


class QwenReranker(BaseReranker):
    """Unified Qwen reranker supporting text-only and multimodal modes.
    
    **3 Prompt Modes**:
    1. `text_only`: Chỉ sử dụng description (text)
    2. `caption`: Sử dụng image caption
    3. `semantic_summary`: Sử dụng image semantic summary
    
    **3 Model Options**:
    1. `qwen3-0.6b`: Text-only model (for text_only mode)
    2. `qwen3-2bvl`: Vision-Language model (for caption/semantic_summary modes)
    3. `qwen3-1.6b`: Text model (for text_only or semantic_summary mode)
    
    **Requirements**:
    - Mode `text_only`: Requires text in item_meta or item_id2text
    - Mode `caption`: Requires captions in item_meta (run data_prepare.py with --generate_caption)
    - Mode `semantic_summary`: Requires semantic summaries in item_meta (run data_prepare.py with --generate_semantic_summary)
    """
    
    def __init__(
        self,
        top_k: int = 50,
        mode: str = "text_only",
        model: str = "qwen3-0.6b",
        max_history: int = 5,
        max_candidates: Optional[int] = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        lr: float = 1e-4,
        patience: Optional[int] = None,
    ) -> None:
        """
        Args:
            top_k: Số lượng items trả về sau rerank
            mode: Prompt mode - "text_only", "caption", "semantic_summary"
            model: Model name - "qwen3-0.6b", "qwen3-2bvl", "qwen3-1.6b"
            max_history: Số lượng items trong history để dùng cho prompt (default: 5)
            max_candidates: Maximum number of candidates to process (None = no limit)
            batch_size: Batch size cho training (default: 32)
            num_epochs: Số epochs (default: 10)
            lr: Learning rate (default: 1e-4)
            patience: Early stopping patience (None = no early stopping)
        """
        super().__init__(top_k=top_k)
        self.mode = mode.lower()
        self.model_name = model.lower()
        self.max_history = max_history
        self.max_candidates = max_candidates
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.patience = patience
        
        # Validate mode
        if self.mode not in ["text_only", "caption", "semantic_summary"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of: text_only, caption, semantic_summary"
            )
        
        # Validate model
        if self.model_name not in MODEL_MAPPING:
            raise ValueError(
                f"Invalid model: {model}. Must be one of: {list(MODEL_MAPPING.keys())}"
            )
        
        # Validate mode-model compatibility
        if self.mode == "text_only" and self.model_name == "qwen3-2bvl":
            raise ValueError(
                "text_only mode requires text model (qwen3-0.6b or qwen3-1.6b), not qwen3-2bvl"
            )
        if self.mode in ["caption", "semantic_summary"] and self.model_name not in ["qwen3-2bvl", "qwen3-1.6b"]:
            raise ValueError(
                f"{self.mode} mode requires VL model (qwen3-2bvl) or compatible model (qwen3-1.6b)"
            )
        
        # Model instances
        self.llm_model: Optional[LLMModel] = None
        self.qwen3vl_model: Optional[Qwen3VLModel] = None
        
        # Data structures
        self.user_history: Dict[int, List[str]] = {}  # user_id -> [item_texts]
        self.item_id2text: Dict[int, str] = {}  # item_id -> item_text (for text_only mode)
        self.item_meta: Dict[int, Dict[str, Any]] = {}  # item_id -> {caption, semantic_summary, text} (for multimodal modes)
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
                - item_id2text: Dict[int, str] - mapping item_id -> text (for text_only mode)
                - item_meta: Dict[int, Dict] - mapping item_id -> {caption, semantic_summary, text} (for multimodal modes)
                - user_history: Dict[int, List[str]] - user history texts
                - val_data: Dict[int, List[int]] - validation data for early stopping
                - train_data_for_llm: List[dict] - training data cho LLM (for text_only mode, optional)
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
        
        # Load data structures
        self.item_id2text = kwargs.get("item_id2text", {})
        self.item_meta = kwargs.get("item_meta", {})
        self.user_history = kwargs.get("user_history", {})
        val_data = kwargs.get("val_data")
        
        # Store training data
        self.train_user_history = train_data
        
        # Get optimization settings
        from config import arg
        use_torch_compile = getattr(arg, 'use_torch_compile', False)
        
        # Load model based on mode
        if self.mode == "text_only":
            # Use LLMModel for text-only mode
            model_path = MODEL_MAPPING[self.model_name]
            train_data_for_llm = kwargs.get("train_data_for_llm")
            
            if train_data_for_llm is not None:
                self.llm_model = LLMModel(
                    train_data=train_data_for_llm,
                    model_name=model_path
                )
                self.llm_model.load_model(use_torch_compile=False)
                self.llm_model.train()
            else:
                self.llm_model = LLMModel(
                    train_data=None,
                    model_name=model_path
                )
                self.llm_model.load_model(use_torch_compile=use_torch_compile)
        else:
            # Use Qwen3VLModel for multimodal modes
            # Map mode to Qwen3VLModel mode
            # For semantic_summary with text models (qwen3-0.6b, qwen3-1.6b), use semantic_summary_small
            # For semantic_summary with VL model (qwen3-2bvl), use semantic_summary
            if self.mode == "semantic_summary" and self.model_name in ["qwen3-0.6b", "qwen3-1.6b"]:
                qwen3vl_mode = "semantic_summary_small"
            else:
                qwen3vl_mode = self.mode
            
            model_path = MODEL_MAPPING[self.model_name]
            
            self.qwen3vl_model = Qwen3VLModel(
                mode=qwen3vl_mode,
                model_name=model_path
            )
            
            # Training support for multimodal modes
            if self.mode in ["caption", "semantic_summary"]:
                train_samples = self._prepare_training_samples(train_data)
                if len(train_samples) == 0:
                    print("Warning: No training samples generated. Skipping training.")
                    self.is_fitted = True
                    return
                
                print(f"Generated {len(train_samples)} training samples")
                self._train_model(train_samples, val_data)
        
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
            **kwargs: Additional arguments:
                - user_history: List[int] - user's interaction history (optional)
            
        Returns:
            List[Tuple[int, float]]: [(item_id, score)] đã sort giảm dần
        """
        self._validate_fitted()
        
        if not candidates:
            return []
        
        # Apply max_candidates limit if set
        if self.max_candidates is not None and len(candidates) > self.max_candidates:
            candidates = candidates[:self.max_candidates]
        
        # Get user history
        user_history = kwargs.get("user_history")
        if user_history is None:
            history = self.user_history.get(user_id, [])
        else:
            history = user_history
        history = history[-self.max_history:]  # Truncate to max_history
        
        # Predict probabilities based on mode
        num_candidates = len(candidates)
        if self.mode == "text_only":
            # Use LLMModel for text-only mode
            if self.llm_model is None:
                raise RuntimeError("LLM model chưa được load. Gọi fit() trước!")
            
            # Build prompt using helper function
            prompt = build_prompt_from_candidates(
                history,
                candidates,
                self.item_id2text,
                max_candidates=self.max_candidates
            )
            probs = self.llm_model.predict_probs(prompt, num_candidates=num_candidates)
        else:
            # Use Qwen3VLModel for multimodal modes
            if self.qwen3vl_model is None:
                raise RuntimeError("Qwen3-VL model chưa được load. Gọi fit() trước!")
            
            probs = self.qwen3vl_model.predict_probs(
                user_history=history,
                candidates=candidates,
                item_meta=self.item_meta,
                num_candidates=num_candidates
            )
        
        # Validate probabilities
        if len(probs) != len(candidates):
            raise ValueError(
                f"Mismatch: {len(candidates)} candidates but {len(probs)} probabilities."
            )
        
        # Rank candidates
        ranked_items = rank_candidates(probs, candidates)
        
        # Convert to (item_id, score) tuples
        item_to_score = {item_id: float(probs[i]) for i, item_id in enumerate(candidates)}
        scored = []
        for item_id in ranked_items[:self.top_k]:
            score = item_to_score.get(item_id, 0.0)
            scored.append((item_id, score))
        
        return scored
    
    def _prepare_training_samples(
        self,
        train_data: Dict[int, List[int]]
    ) -> List[Dict]:
        """Prepare training samples for multimodal modes."""
        samples = []
        all_items = set()
        for items in train_data.values():
            all_items.update(items)
        
        for user_id, items in train_data.items():
            if len(items) < 2:
                continue
            
            if len(items) > 3:
                end_pos = random.randint(1, len(items) - 1)
            else:
                end_pos = len(items) - 1
            
            history = items[:end_pos]
            target_item = items[end_pos]
            
            user_items_set = set(items)
            negative_candidates = [item for item in all_items if item not in user_items_set]
            
            num_negatives = min(19, len(negative_candidates))
            if num_negatives > 0:
                negatives = random.sample(negative_candidates, num_negatives)
            else:
                negatives = []
            
            candidates = [target_item] + negatives
            random.shuffle(candidates)
            target_idx = candidates.index(target_item)
            
            samples.append({
                "user_id": user_id,
                "history": history,
                "candidates": candidates,
                "target_item": target_item,
                "target_idx": target_idx,
            })
        
        return samples
    
    def _train_model(
        self,
        train_samples: List[Dict],
        val_data: Optional[Dict[int, List[int]]] = None
    ) -> None:
        """Train model based on mode."""
        if self.qwen3vl_model is None:
            raise RuntimeError("Qwen3VLModel not initialized")
        
        # Check if text model (semantic_summary with small model) or VL model
        if self.mode == "semantic_summary" and self.model_name in ["qwen3-0.6b", "qwen3-1.6b"]:
            self._train_text_model(train_samples, val_data)
        else:
            self._train_vl_model(train_samples, val_data)
    
    def _train_text_model(
        self,
        train_samples: List[Dict],
        val_data: Optional[Dict[int, List[int]]] = None
    ) -> None:
        """Train text model with Unsloth."""
        if not hasattr(self.qwen3vl_model, 'model') or not hasattr(self.qwen3vl_model, 'tokenizer'):
            print("Warning: Text model does not support training. Using pretrained model only.")
            return
        
        from datasets import Dataset
        from transformers import TrainingArguments, Trainer
        
        training_data = []
        prompts = []
        targets = []
        for sample in train_samples:
            prompt = self._build_training_prompt(sample)
            target = str(sample["target_idx"] + 1)  # 1-indexed
            
            prompts.append(prompt)
            targets.append(target)
            
            training_data.append({
                "instruction": prompt,
                "input": "",
                "output": target,
            })
        
        # ✅ Count tokens before training
        print("\n[QwenReranker] Analyzing prompt token counts...")
        stats = _count_prompt_tokens(
            prompts,
            self.qwen3vl_model.tokenizer,
            include_target=True,
            targets=targets
        )
        print(f"  Total samples: {stats['count']}")
        print(f"  Token statistics (prompt + target):")
        print(f"    Min: {stats['min']} tokens")
        print(f"    Max: {stats['max']} tokens")
        print(f"    Mean: {stats['mean']:.1f} tokens")
        print(f"    Median: {stats['median']:.1f} tokens")
        print(f"    Percentiles:")
        print(f"      P50: {stats['p50']:.1f} tokens")
        print(f"      P75: {stats['p75']:.1f} tokens")
        print(f"      P90: {stats['p90']:.1f} tokens")
        print(f"      P95: {stats['p95']:.1f} tokens")
        print(f"      P99: {stats['p99']:.1f} tokens")
        print(f"    Total tokens: {stats['total']:,}")
        
        # Check if any prompts exceed max_length
        max_length = 2048  # From tokenize_function
        num_exceeding = sum(1 for count in stats.get('token_counts', []) if count > max_length)
        if num_exceeding > 0:
            print(f"  ⚠️  WARNING: {num_exceeding}/{stats['count']} prompts exceed max_length={max_length} and will be truncated!")
        else:
            print(f"  ✅ All prompts fit within max_length={max_length}")
        print()
        
        hf_train_dataset = Dataset.from_list(training_data)
        
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
        
        training_args = TrainingArguments(
            output_dir="./qwen_rerank",
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=self.lr,
            num_train_epochs=1,
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
        
        self._run_training_loop(trainer, val_data)
    
    def _train_vl_model(
        self,
        train_samples: List[Dict],
        val_data: Optional[Dict[int, List[int]]] = None
    ) -> None:
        """Train VL model (caption, semantic_summary modes)."""
        if not hasattr(self.qwen3vl_model, 'model') or not hasattr(self.qwen3vl_model, 'processor'):
            print("Warning: VL model does not support training. Using pretrained model only.")
            return
        
        from datasets import Dataset
        from transformers import TrainingArguments, Trainer
        
        # Prepare training data
        training_data = []
        prompts = []
        targets = []
        for sample in train_samples:
            target = str(sample["target_idx"] + 1)  # 1-indexed
            prompt = self._build_training_prompt(sample)
            
            prompts.append(prompt)
            targets.append(target)
            
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target}
            ]
            training_data.append({
                "messages": messages,
                "prompt": prompt,
                "target": target,
            })
        
        # ✅ Count tokens before training
        print("\n[QwenReranker] Analyzing prompt token counts...")
        stats = _count_prompt_tokens(
            prompts,
            self.qwen3vl_model.processor.tokenizer,
            include_target=True,
            targets=targets
        )
        print(f"  Total samples: {stats['count']}")
        print(f"  Token statistics (prompt + target):")
        print(f"    Min: {stats['min']} tokens")
        print(f"    Max: {stats['max']} tokens")
        print(f"    Mean: {stats['mean']:.1f} tokens")
        print(f"    Median: {stats['median']:.1f} tokens")
        print(f"    Percentiles:")
        print(f"      P50: {stats['p50']:.1f} tokens")
        print(f"      P75: {stats['p75']:.1f} tokens")
        print(f"      P90: {stats['p90']:.1f} tokens")
        print(f"      P95: {stats['p95']:.1f} tokens")
        print(f"      P99: {stats['p99']:.1f} tokens")
        print(f"    Total tokens: {stats['total']:,}")
        
        # Check if any prompts exceed max_length
        max_length = 2048  # From collate_fn
        num_exceeding = sum(1 for count in stats.get('token_counts', []) if count > max_length)
        if num_exceeding > 0:
            print(f"  ⚠️  WARNING: {num_exceeding}/{stats['count']} prompts exceed max_length={max_length} and will be truncated!")
        else:
            print(f"  ✅ All prompts fit within max_length={max_length}")
        print()
        
        hf_train_dataset = Dataset.from_list(training_data)
        
        # Custom data collator for VL model
        def collate_fn(batch):
            """Custom collate function for Qwen3-VL training."""
            processor = self.qwen3vl_model.processor
            
            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []
            
            for item in batch:
                target = item["target"]
                prompt = item["prompt"]
                
                inputs = processor.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=2048,
                )
                
                def ensure_cpu(obj):
                    if isinstance(obj, torch.Tensor):
                        return obj.cpu()
                    elif isinstance(obj, dict):
                        return {k: ensure_cpu(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(ensure_cpu(item) for item in obj)
                    else:
                        return obj
                
                inputs = ensure_cpu(inputs)
                
                input_ids = inputs["input_ids"].squeeze(0)
                attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
                if attention_mask.dim() > 1:
                    attention_mask = attention_mask.squeeze(0)
                
                target_tokens = processor.tokenizer.encode(
                    target,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).squeeze(0).cpu()
                
                full_input_ids = torch.cat([input_ids, target_tokens])
                full_attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(len(target_tokens), dtype=attention_mask.dtype)
                ])
                
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
            
            return {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(padded_attention_mask),
                "labels": torch.stack(padded_labels),
            }
        
        # Prepare dataset
        def prepare_dataset(examples):
            result = {"target": examples.get("target", [])}
            if "prompt" in examples:
                result["prompt"] = examples["prompt"]
            return result
        
        hf_train_dataset = hf_train_dataset.map(
            prepare_dataset,
            batched=True,
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./qwen_rerank_vl",
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=self.lr,
            num_train_epochs=1,
            logging_steps=50,
            save_steps=500,
            report_to="none",
            fp16=True,
            optim="adamw_torch",
            dataloader_pin_memory=True,
        )
        
        # Custom trainer with collate function
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
    
    def _build_training_prompt(self, sample: Dict) -> str:
        """Build training prompt from sample."""
        history = sample["history"]
        candidates = sample["candidates"]
        
        # Get history texts based on mode
        history_texts = []
        for item_id in history[-self.max_history:]:
            meta = self.item_meta.get(item_id, {})
            if self.mode == "caption":
                caption = meta.get("caption", "")
                text = meta.get("text", f"item_{item_id}")
                text = _truncate_item_text(text, max_chars=200)
                if caption:
                    history_texts.append(f"{text} (Image: {caption})")
                else:
                    history_texts.append(text)
            elif self.mode == "semantic_summary":
                semantic_summary = meta.get("semantic_summary", "")
                text = meta.get("text", f"item_{item_id}")
                text = _truncate_item_text(text, max_chars=200)
                if semantic_summary:
                    history_texts.append(f"{text} (Semantic: {semantic_summary})")
                else:
                    history_texts.append(text)
            else:
                text = meta.get("text", f"item_{item_id}")
                text = _truncate_item_text(text, max_chars=200)
                history_texts.append(text)
        
        # Get candidate texts
        candidate_texts = []
        for item_id in candidates:
            meta = self.item_meta.get(item_id, {})
            if self.mode == "caption":
                caption = meta.get("caption", "")
                text = meta.get("text", f"item_{item_id}")
                text = _truncate_item_text(text, max_chars=200)
                if caption:
                    candidate_texts.append(f"{text} (Image: {caption})")
                else:
                    candidate_texts.append(text)
            elif self.mode == "semantic_summary":
                semantic_summary = meta.get("semantic_summary", "")
                text = meta.get("text", f"item_{item_id}")
                text = _truncate_item_text(text, max_chars=200)
                if semantic_summary:
                    candidate_texts.append(f"{text} (Semantic: {semantic_summary})")
                else:
                    candidate_texts.append(text)
            else:
                text = meta.get("text", f"item_{item_id}")
                text = _truncate_item_text(text, max_chars=200)
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
    
    def _run_training_loop(
        self,
        trainer,
        val_data: Optional[Dict[int, List[int]]] = None
    ) -> None:
        """Run training loop with validation."""
        best_val_recall = -1.0
        epochs_no_improve = 0
        best_model_state = None
        
        for epoch in range(self.num_epochs):
            print(f"[QwenReranker] Training epoch {epoch+1}/{self.num_epochs}...")
            trainer.train()
            
            if val_data is not None:
                val_recall = self._evaluate_split(val_data, k=min(10, self.top_k))
                
                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    best_model_state = deepcopy(trainer.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                print(f"[QwenReranker] Epoch {epoch+1}/{self.num_epochs} - "
                      f"val_Recall@{min(10, self.top_k)}: {val_recall:.4f}")
                
                if self.patience and epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        if best_model_state is not None:
            trainer.model.load_state_dict(best_model_state)
            print(f"Loaded best model with val_Recall@{min(10, self.top_k)}: {best_val_recall:.4f}")
        
        trainer.model.eval()
    
    def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> float:
        """Compute average Recall@K for validation."""
        recalls = []
        
        for user_id, gt_items in split.items():
            if user_id not in self.train_user_history:
                continue
            
            history = self.train_user_history[user_id]
            if len(history) == 0:
                continue
            
            # Load pre-generated candidates
            try:
                from evaluation.utils import load_rerank_candidates
                from config import arg
                
                all_candidates = load_rerank_candidates(
                    dataset_code=getattr(arg, 'dataset', 'beauty'),
                    min_rating=getattr(arg, 'min_rating', 0),
                    min_uc=getattr(arg, 'min_uc', 5),
                    min_sc=getattr(arg, 'min_sc', 5),
                )
                
                if user_id in all_candidates.get("val", {}):
                    candidates = all_candidates["val"][user_id]
                elif user_id in all_candidates.get("test", {}):
                    candidates = all_candidates["test"][user_id]
                else:
                    continue
            except Exception:
                continue
            
            random.shuffle(candidates)
            reranked = self.rerank(user_id, candidates, user_history=history)
            top_k_items = [item_id for item_id, _ in reranked[:k]]
            
            hits = len(set(top_k_items) & set(gt_items))
            if len(gt_items) > 0:
                recalls.append(hits / len(gt_items))
        
        return float(np.mean(recalls)) if recalls else 0.0

