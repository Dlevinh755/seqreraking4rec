from unsloth import FastLanguageModel
import torch
from transformers import Trainer, TrainingArguments
from transformers import TrainingArguments
import torch.nn.functional as F
import string
import pandas as pd
import ast
import numpy as np

# Legacy: Keep for backward compatibility, but now we use numbers
LETTERS = list(string.ascii_uppercase[:20])  # A-T (for backward compatibility)


def build_prompt_from_candidates(user_history, candidate_ids, item_id2text, max_candidates=None):
    """Build prompt for LLM reranking.
    
    Args:
        user_history: List of item texts in user history
        candidate_ids: List of candidate item IDs
        item_id2text: Mapping from item_id to item text
        max_candidates: Maximum number of candidates (None = no limit, uses all)
        
    Returns:
        Formatted prompt string with candidate labels (1, 2, 3, ...)
    """
    if max_candidates is not None and len(candidate_ids) > max_candidates:
        candidate_ids = candidate_ids[:max_candidates]
    
    history_text = "\n".join([f"- {h}" for h in user_history])

    candidates = [item_id2text.get(cid, f"item_{cid}") for cid in candidate_ids]
    # Use numbers instead of letters for flexibility
    cand_text = "\n".join(
        [f"{i+1}. {c}" for i, c in enumerate(candidates)]
    )

    num_candidates = len(candidates)
    answer_format = f"Answer with only one number (1-{num_candidates})." if num_candidates > 1 else "Answer with only one number (1)."

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

    def load_model(self, use_torch_compile=False):
        """Load LLM model with 4-bit quantization (default for Unsloth models).
        
        Args:
            use_torch_compile: Whether to use torch.compile() for faster inference
        
        Note:
            All Unsloth models are loaded with 4-bit quantization by default
            to reduce memory usage while maintaining performance.
        """
        print(f"Loading LLM model with 4-bit quantization: {self.model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = 2048,
            dtype = torch.float16,
            load_in_4bit = True,  # 4-bit quantization enabled by default for all Unsloth models
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
    def train(self):

        from datasets import Dataset

        hf_train_dataset = Dataset.from_list(self.train_data)
        hf_train_dataset = hf_train_dataset.map(
            self.tokenize_response_only,
            remove_columns=hf_train_dataset.column_names,
        )

        # Get num_epochs from config if available
        try:
            from config import arg
            num_epochs = getattr(arg, 'rerank_epochs', 2)
        except ImportError:
            num_epochs = 2
        
        training_args = TrainingArguments(
            output_dir="./qwen_rerank",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            num_train_epochs=num_epochs,
            logging_steps=50,
            save_steps=500,
            report_to="none",
            # bf16=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=hf_train_dataset,
            tokenizer=self.tokenizer,
        )
        
        # ✅ Actually train the model!
        print(f"[LLMModel] Starting training for {num_epochs} epochs...")
        trainer.train()
        print(f"[LLMModel] Training completed!")



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
        """Predict probabilities for candidates using numbers (1, 2, 3, ...).
        
        Args:
            prompt: Input prompt
            num_candidates: Number of candidates (if None, infers from prompt)
            
        Returns:
            numpy array of probabilities [num_candidates]
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[:, -1]  # [vocab_size]
        
        # Infer num_candidates from prompt if not provided
        if num_candidates is None:
            # Count "Candidate items:" section in prompt
            if "Candidate items:" in prompt:
                candidates_section = prompt.split("Candidate items:")[1].split("Answer")[0]
                num_candidates = candidates_section.strip().count("\n") + 1
            else:
                # Fallback: try to infer from answer format
                if "Answer with only one number" in prompt:
                    # Extract range like "1-50"
                    import re
                    match = re.search(r'\((\d+)-(\d+)\)', prompt)
                    if match:
                        num_candidates = int(match.group(2))
                    else:
                        num_candidates = 20  # Default fallback
                else:
                    num_candidates = 20  # Default fallback
        
        # Get token IDs for numbers 1 to num_candidates
        # Convert numbers to string tokens
        number_tokens = []
        for i in range(1, num_candidates + 1):
            # Try to get token ID for number as string
            num_str = str(i)
            token_id = self.tokenizer.convert_tokens_to_ids(num_str)
            if token_id != self.tokenizer.unk_token_id:
                number_tokens.append((i, token_id))
        
        if not number_tokens:
            # Fallback: use letters if numbers don't work
            max_letters = min(num_candidates, 20)
            letters = list(string.ascii_uppercase[:max_letters])
            token_ids = self.tokenizer.convert_tokens_to_ids(letters)
            probs = F.softmax(logits[:, token_ids], dim=-1)
            # Pad to num_candidates if needed
            probs_np = probs[0].cpu().numpy()
            if len(probs_np) < num_candidates:
                # Pad with zeros
                padded = np.zeros(num_candidates)
                padded[:len(probs_np)] = probs_np
                return padded
            return probs_np[:num_candidates]
        
        # Extract probabilities for number tokens
        token_ids = [tid for _, tid in number_tokens]
        probs = F.softmax(logits[:, token_ids], dim=-1)
        
        # Map back to candidate indices (1-indexed)
        prob_array = np.zeros(num_candidates)
        for idx, (cand_num, token_id) in enumerate(number_tokens):
            if cand_num <= num_candidates:
                prob_array[cand_num - 1] = probs[0, idx].item()
        
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
    def tokenize_response_only(self, example):
        """Tokenize training example for Unsloth.
        
        Uses apply_chat_template and lets Unsloth handle label masking automatically.
        This is the recommended approach for Unsloth compatibility.
        """
        messages = example["messages"]
        
        # ✅ Use apply_chat_template for proper formatting (Unsloth-compatible)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize the full text
        # ✅ Pad to max_length to ensure all sequences have same length
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=2048,
            padding="max_length",  # Pad to max_length for consistent batch shapes
            return_tensors=None,  # Return lists, not tensors (for Dataset.map)
        )
        
        # ✅ Get input_ids and ensure it's a flat list
        input_ids = tokenized["input_ids"]
        
        # Handle different return formats from tokenizer
        if isinstance(input_ids, list):
            if len(input_ids) > 0 and isinstance(input_ids[0], list):
                # Flatten nested list
                input_ids = [item for sublist in input_ids for item in sublist]
            # Ensure all elements are ints
            input_ids = [int(x) for x in input_ids]
        else:
            # Convert tensor or other type to list
            input_ids = [int(x) for x in list(input_ids)]
        
        # ✅ Create labels: copy input_ids (Unsloth will handle masking)
        # For Unsloth, we create labels as a copy of input_ids
        # The Trainer will automatically mask prompt tokens based on the chat template
        labels = input_ids.copy()
        
        # ✅ Find where assistant response starts to mask prompt tokens
        # Tokenize prompt (without assistant response) to find the split point
        prompt_messages = messages[:-1] if len(messages) > 1 else []
        
        if prompt_messages:
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Tokenize prompt with same settings
            prompt_tokenized = self.tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=True,
                max_length=2048,
                padding=False,  # Don't pad when calculating length
            )
            
            prompt_input_ids = prompt_tokenized["input_ids"]
            # Flatten if nested
            if isinstance(prompt_input_ids, list) and len(prompt_input_ids) > 0:
                if isinstance(prompt_input_ids[0], list):
                    prompt_input_ids = [item for sublist in prompt_input_ids for item in sublist]
                prompt_input_ids = [int(x) for x in prompt_input_ids]
            else:
                prompt_input_ids = [int(x) for x in list(prompt_input_ids)]
            
            # Find matching position in input_ids
            prompt_len = min(len(prompt_input_ids), len(input_ids))
            
            # Verify match at the beginning
            if prompt_len > 0:
                matches = sum(1 for i in range(min(prompt_len, len(input_ids))) 
                            if i < len(prompt_input_ids) and prompt_input_ids[i] == input_ids[i])
                if matches < prompt_len * 0.7:  # Less than 70% match
                    # Fallback: use conservative estimate
                    prompt_len = max(0, len(input_ids) - 20)
        else:
            prompt_len = max(0, len(input_ids) - 10)
        
        # ✅ Mask prompt tokens in labels (set to -100 to ignore in loss)
        # Only mask if we have a valid prompt_len
        if 0 < prompt_len < len(labels):
            for i in range(prompt_len):
                labels[i] = -100
        
        # ✅ Ensure labels is a flat list of ints (critical for Unsloth)
        # Convert all elements to int, ensuring no nested structures
        final_labels = []
        for x in labels:
            if isinstance(x, (list, tuple)):
                final_labels.extend([int(item) for item in x])
            elif isinstance(x, torch.Tensor):
                final_labels.append(int(x.item()))
            else:
                try:
                    final_labels.append(int(x))
                except (ValueError, TypeError):
                    final_labels.append(-100)  # Fallback for invalid values
        
        # ✅ Final validation: ensure labels has same length as input_ids
        if len(final_labels) != len(input_ids):
            if len(final_labels) < len(input_ids):
                final_labels.extend([-100] * (len(input_ids) - len(final_labels)))
            else:
                final_labels = final_labels[:len(input_ids)]
        
        # ✅ Ensure input_ids is also properly formatted
        final_input_ids = []
        for x in input_ids:
            if isinstance(x, (list, tuple)):
                final_input_ids.extend([int(item) for item in x])
            elif isinstance(x, torch.Tensor):
                final_input_ids.append(int(x.item()))
            else:
                try:
                    final_input_ids.append(int(x))
                except (ValueError, TypeError):
                    final_input_ids.append(self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id else 0)
        
        tokenized["input_ids"] = final_input_ids
        tokenized["labels"] = final_labels

        return tokenized




