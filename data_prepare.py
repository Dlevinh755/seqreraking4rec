from datasets import *
from datasets.clip_embeddings import maybe_extract_clip_embeddings
from config import *
from pytorch_lightning import seed_everything
import torch
import json
import pandas as pd
from pathlib import Path


def main(args):
    seed_everything(args.seed)
    dataset = dataset_factory(args)
    data = dataset.load_dataset()
    maybe_extract_clip_embeddings(dataset, data, args)

    # Export a single CSV with columns:
    # {Item_id, user_id, item_new_id, item_text, item_image_embedding,
    #  item_text_embedding, item_image_path, split}
    preproc_folder = Path(dataset._get_preprocessed_folder_path())
    clip_path = preproc_folder.joinpath("clip_embeddings.pt")

    clip_payload = None
    if clip_path.exists():
        clip_payload = torch.load(clip_path)

    image_embs = None
    text_embs = None
    if clip_payload:
        image_embs = clip_payload.get("image_embs")
        text_embs = clip_payload.get("text_embs")

    # Build inverse smap to recover original item ids
    smap = data.get("smap", {})
    inv_smap = {new: orig for orig, new in smap.items()}

    meta = data.get("meta", {})

    rows = []
    def add_rows_for_split(split_name, split_dict):
        for user, items in split_dict.items():
            for item in items:
                orig_item = inv_smap.get(item, None)
                info = meta.get(item, {})
                text = info.get("text") if info else None
                image_path = info.get("image_path") or info.get("image") if info else None

                img_emb = None
                txt_emb = None
                if image_embs is not None and 0 <= item < image_embs.size(0):
                    emb = image_embs[item]
                    if emb is not None:
                        img_emb = emb.tolist()
                if text_embs is not None and 0 <= item < text_embs.size(0):
                    emb = text_embs[item]
                    if emb is not None:
                        txt_emb = emb.tolist()

                rows.append({
                    "Item_id": orig_item,
                    "user_id": int(user),
                    "item_new_id": int(item),
                    "item_text": text,
                    "item_image_embedding": json.dumps(img_emb) if img_emb is not None else "",
                    "item_text_embedding": json.dumps(txt_emb) if txt_emb is not None else "",
                    "item_image_path": image_path or "",
                    "split": split_name,
                })

    add_rows_for_split("train", data.get("train", {}))
    add_rows_for_split("val", data.get("val", {}))
    add_rows_for_split("test", data.get("test", {}))

    if rows:
        df = pd.DataFrame(rows)
        out_csv = preproc_folder.joinpath("dataset_single_export.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved single CSV export to: {out_csv}")

if __name__ == "__main__":
    main(arg)