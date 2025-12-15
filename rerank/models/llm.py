from unsloth import FastLanguageModel
import torch
from transformers import Trainer, TrainingArguments
from transformers import TrainingArguments
import torch.nn.functional as F
import string
import pandas as pd
LETTERS = list(string.ascii_uppercase[:20])  # A-T
import ast


def build_prompt_from_candidates(user_history, candidate_ids, item_id2text):
    """Build prompt for LLM reranking.
    
    Args:
        user_history: List of item texts in user history
        candidate_ids: List of candidate item IDs (max 20)
        item_id2text: Mapping from item_id to item text
        
    Returns:
        Formatted prompt string
        
    Raises:
        ValueError: If len(candidate_ids) > 20
    """
    MAX_CANDIDATES = len(LETTERS)  # 20
    
    if len(candidate_ids) > MAX_CANDIDATES:
        raise ValueError(
            f"Too many candidates: {len(candidate_ids)} > {MAX_CANDIDATES}. "
            f"LLM reranker only supports up to {MAX_CANDIDATES} candidates (A-T)."
        )
    
    history_text = "\n".join([f"- {h}" for h in user_history])

    candidates = [item_id2text.get(cid, f"item_{cid}") for cid in candidate_ids]
    cand_text = "\n".join(
        [f"{LETTERS[i]}. {c}" for i, c in enumerate(candidates)]
    )

    prompt = f"""
You are a recommendation ranking assistant.

Choose exactly ONE item the user is most likely to interact with next.

User history:
{history_text}

Candidate items:
{cand_text}

Answer with only one letter (A-T).
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
            self.model_name = model_name or "Qwen/Qwen3-0.6B"
            self.train_data = train_data

    def load_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = 2048,
            dtype = torch.float16,
            load_in_4bit = True,
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
    def train(self):

        from datasets import Dataset

        hf_train_dataset = Dataset.from_list(self.train_data)
        hf_train_dataset = hf_train_dataset.map(
            self.tokenize_response_only,
            remove_columns=hf_train_dataset.column_names,
        )

        training_args = TrainingArguments(
        output_dir="./qwen_rerank",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=2,
        logging_steps=50,
        save_steps=500,
        report_to="none",
      #  bf16=True,
    )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=hf_train_dataset,
            tokenizer=self.tokenizer,
        )



    def predict_prob(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1]

        token_ids = self.tokenizer.convert_tokens_to_ids(LETTERS)
        probs = F.softmax(logits[:, token_ids], dim=-1)

        return dict(zip(LETTERS, probs[0].tolist()))
    def predict_probs(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[:, -1]
        token_ids = self.tokenizer.convert_tokens_to_ids(LETTERS)
        probs = F.softmax(logits[:, token_ids], dim=-1)

        return probs[0].cpu().numpy()
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
        messages = example["messages"]

        prompt = ""
        for m in messages[:-1]:
            prompt += f"{m['role'].upper()}: {m['content']}\n"

        response = messages[-1]["content"]

        full_text = prompt + response

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=2048,
        )

        labels = tokenized["input_ids"].copy()

        prompt_len = len(
            self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )

        labels[:prompt_len] = [-100] * prompt_len
        tokenized["labels"] = labels

        return tokenized




