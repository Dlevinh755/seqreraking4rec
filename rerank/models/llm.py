
##%%writefile /kaggle/working/rerank/models/llm.py
from unsloth import FastLanguageModel
import torch
from transformers import Trainer, TrainingArguments
from transformers import TrainingArguments
import torch.nn.functional as F
import string
import pandas as pd
import ast
import numpy as np
import os

# Legacy: Keep for backward compatibility, but now we use numbers
# Use both uppercase and lowercase for up to 52 candidates (A-Z, a-z)
LETTERS = list(string.ascii_uppercase) + list(string.ascii_lowercase)  # A-Z, a-z (52 letters)


def build_prompt_from_candidates(user_history, candidate_ids, item_id2text, max_candidates=None):
    """Build prompt for LLM reranking.
    
    Args:
        user_history: List of item texts in user history
        candidate_ids: List of candidate item IDs
        item_id2text: Mapping from item_id to item text
        max_candidates: Maximum number of candidates (None = no limit, uses all)
        
    Returns:
        Formatted prompt string with candidate labels (A, B, C, ... or a, b, c, ...)
    """
    if max_candidates is not None and len(candidate_ids) > max_candidates:
        candidate_ids = candidate_ids[:max_candidates]
    
    history_text = "\n".join([f"- {h}" for h in user_history])

    candidates = [item_id2text.get(cid, f"item_{cid}") for cid in candidate_ids]
    num_candidates = len(candidates)
    
    # Use letters (A-Z, a-z) for up to 52 candidates (LlamaRec style)
    if num_candidates > len(LETTERS):
        raise ValueError(
            f"Too many candidates ({num_candidates}). Maximum supported: {len(LETTERS)} candidates "
            f"(using letters A-Z, a-z). Consider reducing max_candidates or using number labels."
        )
    
    # Use letters instead of numbers (LlamaRec style)
    cand_text = "\n".join(
        [f"{LETTERS[i]}. {c}" for i, c in enumerate(candidates)]
    )

    # Answer format with letters
    if num_candidates <= 26:
        answer_format = f"Answer with only one letter (A-{LETTERS[num_candidates-1]})."
    else:
        answer_format = f"Answer with only one letter (A-Z, a-{LETTERS[num_candidates-1]})."

    prompt = f"""
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
{history_text}

Candidate items:
{cand_text}

{answer_format}
""".strip()

    return prompt

def rank_candidates(probs, candidate_ids):
    """Rank candidates by probabilities.
    
    Args:
        probs: Array of probabilities (one per candidate)
        candidate_ids: List of candidate item IDs
        
    Returns:
        List of candidate IDs sorted by probability (descending)
        
    Raises:
        ValueError: If len(probs) != len(candidate_ids)
    """
    if len(probs) != len(candidate_ids):
        raise ValueError(
            f"Mismatch: {len(candidate_ids)} candidates but {len(probs)} probabilities. "
            f"Each candidate must have exactly one probability."
        )
    
    ranked = sorted(
        zip(candidate_ids, probs),
        key=lambda x: x[1],
        reverse=True
    )
    return [cid for cid, _ in ranked]

def recall_at_k(ranked_items, gt_item, k):
    return int(gt_item in ranked_items[:k])


import math

def ndcg_at_k(ranked_items, gt_item, k):
    if gt_item not in ranked_items[:k]:
        return 0.0
    rank = ranked_items.index(gt_item) + 1
    return 1.0 / math.log2(rank + 1)

class LLMModel:
    def __init__(self, train_data=None, model_name=None):
            # Priority: Use Unsloth models by default for better performance and 4-bit quantization
            self.model_name = model_name or "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
            self.train_data = train_data

    def load_model(self, use_torch_compile=False, max_seq_length=None):
        """Load LLM model with 4-bit quantization (default for Unsloth models).
        
        Args:
            use_torch_compile: Whether to use torch.compile() for faster inference
            max_seq_length: Maximum sequence length (None = get from config, default: 2048)
        
        Note:
            All Unsloth models are loaded with 4-bit quantization by default
            to reduce memory usage while maintaining performance.
        """
        # Get max_seq_length from config if not provided
        if max_seq_length is None:
            try:
                from config import arg
                max_seq_length = getattr(arg, 'qwen_max_seq_length', 2048)
            except ImportError:
                max_seq_length = 2048  # Default fallback
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"Loading LLM model with 4-bit quantization: {self.model_name}")
        print(f"  Max sequence length: {max_seq_length}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = max_seq_length,
            dtype = torch.float16,
            load_in_4bit = True,  # 4-bit quantization enabled by default for all Unsloth models
            device_map={'': local_rank},
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 8,
            target_modules = ["q_proj","k_proj","v_proj","o_proj"],
            lora_alpha = 16,
            lora_dropout = 0.05,
            bias = "none",
            use_gradient_checkpointing = True,
        )
        
        # Compile model if requested (PyTorch 2.0+)
        if use_torch_compile and hasattr(torch, 'compile'):
            try:
                print("Compiling LLM model with torch.compile() for faster inference...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("LLM model compiled successfully!")
            except Exception as e:
                print(f"Warning: torch.compile() failed: {e}. Continuing without compilation.")
    def train(self, batch_size=None):

        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
        from unsloth.chat_templates import train_on_responses_only

        hf_train_dataset = Dataset.from_list(self.train_data)
        
        # ✅ Format messages to text using chat template (like notebook Cell 7)
        def formatting_prompts_func(examples):
            """Format conversations to text using chat template.
            
            When batched=True: examples["messages"] is a list of message lists
            When batched=False: examples["messages"] is a single message list
            """
            messages_list = examples["messages"]
            
            # Check if batched (list of lists) or single (single list)
            if isinstance(messages_list, list) and len(messages_list) > 0:
                # Check if first element is a list (batched) or dict (single message list)
                if isinstance(messages_list[0], list):
                    # Batched: list of message lists
                    texts = []
                    for messages in messages_list:
                        text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        texts.append(text)
                    return {"text": texts}
                elif isinstance(messages_list[0], dict):
                    # Single: messages_list is already a list of message dicts
                    text = self.tokenizer.apply_chat_template(
                        messages_list,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    return {"text": text}
            
            raise ValueError(f"Invalid messages format: {type(messages_list)}")
        
        # ✅ Map to format as text (like notebook Cell 7) - use batched=True for efficiency
        hf_train_dataset = hf_train_dataset.map(
            formatting_prompts_func,
            batched=True,  # Process in batches for efficiency (like notebook)
        )

        # Get num_epochs, batch_size, and learning_rate from config if available
        try:
            from config import arg
            num_epochs = getattr(arg, 'rerank_epochs', 2)
            # Use batch_size parameter if provided, otherwise get from config
            if batch_size is None:
                batch_size = getattr(arg, 'rerank_batch_size', 16)
            # Get learning rate from config (default: 1e-4, was hardcoded to 2e-5)
            learning_rate = getattr(arg, 'rerank_lr', 1e-4)
        except ImportError:
            num_epochs = 1
            if batch_size is None:
                batch_size = 16  # Default fallback
            learning_rate = 1e-4  # Default fallback
        
        # ✅ Use SFTConfig and SFTTrainer (like notebook Cell 8)
        training_args = SFTConfig(
            dataset_text_field="text",  # Field name in dataset
            output_dir="./qwen_rerank",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,  # ✅ Use from config (default: 1e-4)
            num_train_epochs=num_epochs,
            logging_steps=10,
            save_steps=500,
            report_to="none",
            fp16=True,
            optim="adamw_8bit",
            ddp_find_unused_parameters = False,
            dataloader_pin_memory = False,
            load_best_model_at_end=False, 
            lr_scheduler_type = "cosine",
            warmup_steps = 20,
            weight_decay = 0.1,
        )
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=hf_train_dataset,
            args=training_args,
        )
        
        # ✅ Use train_on_responses_only to automatically mask prompt tokens (like notebook)
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        
        # ✅ Actually train the model!
        print(f"[LLMModel] Starting training for {num_epochs} epochs...")
        print(f"[LLMModel] Training config: lr={learning_rate}, batch_size={batch_size}, epochs={num_epochs}")
        print(f"[LLMModel] Dataset size: {len(hf_train_dataset)} samples")
        print(f"[LLMModel] Total steps: {len(hf_train_dataset) // (batch_size * 2) * num_epochs}")
        
        trainer.train()
        
        # ✅ Log final training loss
        print(f"[LLMModel] Training completed!")
        print(f"[LLMModel] Check training logs above for loss progression")



    def predict_prob(self, prompt, num_candidates=None):
        """Predict probabilities for candidates (legacy method, uses letters).
        
        Args:
            prompt: Input prompt
            num_candidates: Number of candidates (for backward compatibility)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1]

        # Use letters for backward compatibility
        max_letters = num_candidates if num_candidates and num_candidates <= 20 else 20
        letters = list(string.ascii_uppercase[:max_letters])
        token_ids = self.tokenizer.convert_tokens_to_ids(letters)
        probs = F.softmax(logits[:, token_ids], dim=-1)

        return dict(zip(letters, probs[0].tolist()))
    
    def predict_probs(self, prompt, num_candidates=None):
        """Predict probabilities for candidates using letters (A, B, C, ... or a, b, c, ...) - LlamaRec style.
        
        Args:
            prompt: Input prompt (plain text, will be converted to chat template format)
            num_candidates: Number of candidates (if None, infers from prompt)
            
        Returns:
            numpy array of probabilities [num_candidates]
        """
        # Get max_length from config
        try:
            from config import arg
            max_length = getattr(arg, 'qwen_max_seq_length', 2048)
        except ImportError:
            max_length = 2048  # Default fallback
        
        # ✅ Convert plain text prompt to chat template format (consistency with training)
        # Training uses apply_chat_template, so inference should too
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template with generation prompt (adds <|im_start|>assistant\n at the end)
        # This ensures consistency with training format
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # ✅ Add <|im_start|>assistant\n for generation
        )
        
        # Tokenize the chat template formatted text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,  # ✅ Truncate if too long
            max_length=max_length,  # ✅ Use from config
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[:, -1]  # [vocab_size]
        
        # Infer num_candidates from original prompt (before chat template) if not provided
        # Note: We use the original prompt for inference, not the chat template formatted text
        # because the chat template adds special tokens that might interfere with parsing
        if num_candidates is None:
            # Count "Candidate items:" section in original prompt
            if "Candidate items:" in prompt:
                candidates_section = prompt.split("Candidate items:")[1].split("Answer")[0]
                num_candidates = candidates_section.strip().count("\n") + 1
            else:
                # Fallback: try to infer from answer format
                if "Answer with only one letter" in prompt:
                    # Extract letter range like "A-Z" or "A-Z, a-z"
                    import re
                    # Try to match patterns like "A-Z" or "A-Z, a-z"
                    match = re.search(r'\(([A-Z])-([A-Za-z])\)', prompt)
                    if match:
                        start_letter = match.group(1)
                        end_letter = match.group(2)
                        start_idx = LETTERS.index(start_letter) if start_letter in LETTERS else 0
                        end_idx = LETTERS.index(end_letter) if end_letter in LETTERS else len(LETTERS) - 1
                        num_candidates = end_idx - start_idx + 1
                    else:
                        num_candidates = min(20, len(LETTERS))  # Default fallback
                else:
                    num_candidates = min(20, len(LETTERS))  # Default fallback
        
        # Validate num_candidates
        if num_candidates > len(LETTERS):
            print(f"[WARNING] num_candidates ({num_candidates}) exceeds max letters ({len(LETTERS)}). Truncating to {len(LETTERS)}.")
            num_candidates = len(LETTERS)
        
        # Get token IDs for letters A-Z, a-z (LlamaRec style)
        letter_tokens = []
        for i in range(num_candidates):
            letter = LETTERS[i]
            
            # Strategy 1: Try direct letter token
            token_id = self.tokenizer.convert_tokens_to_ids(letter)
            if token_id != self.tokenizer.unk_token_id:
                letter_tokens.append((i, letter, token_id))
                continue
            
            # Strategy 2: Try with space prefix (common in BPE tokenizers)
            token_id = self.tokenizer.convert_tokens_to_ids(" " + letter)
            if token_id != self.tokenizer.unk_token_id:
                letter_tokens.append((i, letter, token_id))
                continue
            
            # Strategy 3: Try encoding and taking first token
            encoded = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(encoded) > 0 and encoded[0] != self.tokenizer.unk_token_id:
                letter_tokens.append((i, letter, encoded[0]))
                continue
        
        # Debug: Check if we found letter tokens
        if len(letter_tokens) < num_candidates:
            print(f"[WARNING] Only found {len(letter_tokens)}/{num_candidates} letter tokens!")
            print(f"  Found letters: {[l for _, l, _ in letter_tokens[:10]]}...")
            # If we found very few tokens, this is a problem
            if len(letter_tokens) == 0:
                print(f"[ERROR] No letter tokens found! Falling back to uniform distribution.")
                print(f"[ERROR] This will cause recall = random! Check tokenizer compatibility.")
                return np.ones(num_candidates) / num_candidates
        
        # Extract probabilities for letter tokens
        token_ids = [tid for _, _, tid in letter_tokens]
        probs = F.softmax(logits[:, token_ids], dim=-1)
        
        # Map back to candidate indices (0-indexed)
        prob_array = np.zeros(num_candidates)
        for idx, (cand_idx, letter, token_id) in enumerate(letter_tokens):
            if cand_idx < num_candidates:
                prob_array[cand_idx] = probs[0, idx].item()
        
        # Normalize probabilities (in case some letters weren't found)
        prob_sum = prob_array.sum()
        if prob_sum > 0:
            prob_array = prob_array / prob_sum
        else:
            # Fallback: uniform distribution if all probabilities are zero
            print(f"[WARNING] All probabilities are zero, using uniform distribution")
            print(f"[WARNING] This will cause recall = random! Model may not have learned anything.")
            prob_array = np.ones(num_candidates) / num_candidates
        
        # ✅ Debug: Check if probabilities are uniform (model chưa học được gì)
        prob_std = np.std(prob_array)
        expected_uniform_std = 0.0  # Uniform distribution has std = 0
        if prob_std < 0.01:  # Nearly uniform
            print(f"[WARNING] Probabilities are nearly uniform (std={prob_std:.4f})!")
            print(f"[WARNING] Model may not have learned anything. Check training loss.")
        
        return prob_array
    def evaluate(self,df, user2history, item_id2text):
        recalls = {1: [], 5: [], 10: []}
        ndcgs = {5: [], 10: []}

        for _, row in df.iterrows():
            user_id = row["user_index"]
            gt_item = row["label"]

            candidate_ids = ast.literal_eval(row["candidate_ids"])
            history = user2history[user_id][-10:]

            prompt = build_prompt_from_candidates(history, candidate_ids,item_id2text)
            probs = self.predict_probs(prompt)
            ranked_items = rank_candidates(probs, candidate_ids)

            for k in recalls:
                recalls[k].append(recall_at_k(ranked_items, gt_item, k))

            for k in ndcgs:
                ndcgs[k].append(ndcg_at_k(ranked_items, gt_item, k))

        return {
            **{f"Recall@{k}": sum(v)/len(v) for k, v in recalls.items()},
            **{f"NDCG@{k}": sum(v)/len(v) for k, v in ndcgs.items()}
        }




