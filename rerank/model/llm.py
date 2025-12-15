from unsloth import FastLanguageModel
import torch
from unsloth import SFTTrainer
from transformers import TrainingArguments
import torch.nn.functional as F
import string
LETTERS = list(string.ascii_uppercase[:20])  # A-T
import ast


def build_prompt_from_candidates(user_history, candidate_ids, item_id2text):
    history_text = "\n".join([f"- {h}" for h in user_history])

    candidates = [item_id2text[cid] for cid in candidate_ids]
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
    def __init__(self, train_data=None):
            self.model_name = "Qwen/Qwen3-0.6B"
            self.train_data = train_data

    def load_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = 1024,
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


        self.trainer = SFTTrainer(
        model=self.model,
        tokenizer=self.tokenizer,
        train_dataset= self.train_data,
        dataset_text_field="messages",
        max_seq_length=2048,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            num_train_epochs=2,
            logging_steps=50,
            save_steps=500,
            output_dir="./qwen_rec_lora",
            report_to="none"
        ),
        train_on_prompt=False   # ⭐ RESPONSE-ONLY
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



