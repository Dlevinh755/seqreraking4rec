import pandas as pd
import pandas as pd
import random
import string
from collections import defaultdict
from .model.llm import LLMModel


LETTERS = list(string.ascii_uppercase[:20])  # A-T

import torch.nn.functional as F




def build_training_samples(user2items, all_items, max_history=10):
    samples = []

    for user, interactions in user2items.items():
        if len(interactions) < 3:
            continue

        # sort nếu có timestamp
        interactions = interactions[-(max_history + 1):]

        pos_item = interactions[-1]
        history_items = interactions[:-1]

        pos_id = pos_item["item_new_id"]
        pos_text = pos_item["item_text"]

        # sample negatives
        neg_items = random.sample(
            [i for i in all_items if i["item_new_id"] != pos_id],
            19
        )

        candidates = [(pos_id, pos_text)] + [
            (i["item_new_id"], i["item_text"]) for i in neg_items
        ]
        random.shuffle(candidates)

        label_idx = [i for i, (cid, _) in enumerate(candidates) if cid == pos_id][0]
        label = LETTERS[label_idx]

        samples.append({
            "user_id": user,
            "history": [i["item_text"] for i in history_items],
            "candidates": [c[1] for c in candidates],
            "label": label
        })

    return samples

def build_prompt(sample):
    history_text = "\n".join(
        [f"- {h}" for h in sample["history"]]
    )

    cand_text = "\n".join(
        [f"{LETTERS[i]}. {c}" for i, c in enumerate(sample["candidates"])]
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


def to_unsloth_format(samples):
    data = []

    for s in samples:
        data.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a recommendation ranking assistant."
                },
                {
                    "role": "user",
                    "content": build_prompt(s)
                },
                {
                    "role": "assistant",
                    "content": s["label"]   # ONLY label token
                }
            ]
        })

    return data







def main():
    df = pd.read_csv("interactions.csv")

    df_train = df[df["split"] == "train"]
    user2items = defaultdict(list)

    for _, row in df_train.iterrows():
        user2items[row["user_id"]].append(row)
        all_items = (
        df_train[["item_new_id", "item_text"]]
        .drop_duplicates()
        .to_dict("records")
        )

    train_samples = build_training_samples(user2items, all_items)
    train_data = to_unsloth_format(train_samples)
    model = LLMModel(train_data=train_data, model_name="qwen/Qwen-7B-Chat")
    model.load_model()
    model.trainer.train()


    import pandas as pd
    import ast

    df_val = pd.read_csv("candidates_val.csv")
    df_test = pd.read_csv("candidates_test.csv")

    # item_id -> text
    item_df = pd.read_csv("interactions.csv")[["item_new_id", "item_text"]].drop_duplicates()
    item_id2text = dict(zip(item_df.item_new_id, item_df.item_text))
    from collections import defaultdict

    df_inter = pd.read_csv("interactions.csv")

    user2history = defaultdict(list)

    for _, row in df_inter.iterrows():
        user2history[row["user_id"]].append(row["item_text"])

    val_metrics = model.evaluate(df_val, user2history, item_id2text)
    test_metrics = model.evaluate(df_test, user2history, item_id2text )

    print("VAL:", val_metrics)
    print("TEST:", test_metrics)



        







if __name__ == "__main__":
    main()