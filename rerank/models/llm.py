
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

        # Get num_epochs and batch_size from config if available
        try:
            from config import arg
            num_epochs = getattr(arg, 'rerank_epochs', 2)
            # Use batch_size parameter if provided, otherwise get from config
            if batch_size is None:
                batch_size = getattr(arg, 'rerank_batch_size', 16)
        except ImportError:
            num_epochs = 1
            if batch_size is None:
                batch_size = 16  # Default fallback
        
        # ✅ Use SFTConfig and SFTTrainer (like notebook Cell 8)
        training_args = SFTConfig(
            dataset_text_field="text",  # Field name in dataset
            output_dir="./qwen_rerank",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            num_train_epochs=num_epochs,
            logging_steps=50,
            save_steps=500,
            report_to="none",
            fp16=True,
            optim="adamw_8bit",
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




