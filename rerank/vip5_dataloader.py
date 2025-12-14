from __future__ import annotations

def vip5_collate_fn_with_clip(clip_embs: torch.Tensor, device: str = "cpu"):
    """Collate function cho VIP5: batch hóa input_ids, target_ids, attention_mask, và ghép vis_feats từ CLIP.
    - clip_embs: tensor [num_items+1, feat_dim], index = item_id (0 là padding).
    """
    def collate(batch):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
        target_ids = [torch.tensor(x["target_ids"], dtype=torch.long) for x in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=-100)
        whole_word_ids = torch.zeros_like(input_ids)
        category_ids = torch.zeros_like(input_ids)

        # Ghép vis_feats cho toàn bộ candidates: các sample phải có cùng số candidates (k)
        candidates_list = [x.get("candidates", []) for x in batch]
        # determine k (max length)
        k = max(len(c) for c in candidates_list) if candidates_list else 0
        feat_dim = clip_embs.size(1)
        # prepare tensor [B, k, feat_dim]
        vis_feats = torch.zeros((len(batch), k, feat_dim), dtype=clip_embs.dtype)
        for i, cands in enumerate(candidates_list):
            for j, item_id in enumerate(cands):
                if 0 <= item_id < clip_embs.size(0):
                    vis_feats[i, j] = clip_embs[item_id]
                else:
                    vis_feats[i, j] = 0.0

        loss_weights = torch.tensor([x.get("loss_weights", 1.0) for x in batch], dtype=torch.float)
        # labels for evaluation (label_id created in sample)
        label_ids = torch.tensor([x.get("label_id", -1) for x in batch], dtype=torch.long)
        task = [x.get("task", "rec") for x in batch]
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "target_ids": target_ids.to(device),
            "whole_word_ids": whole_word_ids.to(device),
            "category_ids": category_ids.to(device),
            "vis_feats": vis_feats.to(device),
            "loss_weights": loss_weights.to(device),
            "label_id": label_ids.to(device),
            "task": task,
            "candidates": candidates_list,
        }
    return collate
"""Dataloader & Dataset cho VIP5 reranker dựa trên retrieved.pkl.

Thiết kế tương tự LLMDataloader trong LlamaRec nhưng đơn giản hơn:
- Đọc dataset tuần tự (train/val/test, meta) từ datasets.dataset_factory.
- Đọc retrieved.pkl từ Stage 1 (val_probs, val_labels, test_probs, test_labels).
- Xây dựng subset user/candidates cho val/test như LLMDataloader.
- Sinh các training sample cho VIP5 bằng build_vip5_training_sample.

Phần này chỉ xử lý text; ghép visual features (CLIP) có thể bổ sung sau.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from config import arg, EXPERIMENT_ROOT
from datasets import dataset_factory
from rerank.methods.vip5 import Args as VIP5Args, build_vip5_training_sample, VIP5Trainer


@dataclass
class VIP5SampleConfig:
    """Cấu hình đơn giản cho việc xây dựng sample VIP5 từ retrieved.pkl."""

    negative_sample_size: int = 19  # 19 negative + 1 positive
    max_title_len: int = 32


class VIP5TrainDataset(Dataset):
    """Dataset train cho VIP5 sử dụng train sequences + negative sampling.

    - Giống LLMTrainDataset: sinh nhiều prefix từ train[user] và negative
      sampling ngẫu nhiên trong toàn bộ item space.
    - Chỉ sử dụng text (meta title) để xây prompt input cho VIP5.
    """

    def __init__(
        self,
        vip5_args: VIP5Args,
        sample_cfg: VIP5SampleConfig,
        train: Dict[int, List[int]],
        text_dict: Dict[int, Dict[str, Any]],
        tokenizer,
        num_items: int,
    ) -> None:
        self.vip5_args = vip5_args
        self.sample_cfg = sample_cfg
        self.train = train
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.num_items = num_items

        self.all_seqs: List[List[int]] = []
        for u in sorted(train.keys()):
            seq = train[u]
            for i in range(2, len(seq) + 1):
                self.all_seqs.append(seq[:i])

    def __len__(self) -> int:
        return len(self.all_seqs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        tokens = self.all_seqs[index]
        answer = tokens[-1]
        history = tokens[:-1]

        # Negative sampling giống LLMTrainDataset nhưng không cần rng riêng
        seq = history[- self.vip5_args.max_text_length :]
        candidates: List[int] = [answer]

        # Lấy ngẫu nhiên nhiều id, tránh trùng lặp với history/answer
        while len(candidates) < self.sample_cfg.negative_sample_size + 1:
            item_id = int(torch.randint(1, self.num_items + 1, (1,)).item())
            if item_id in history or item_id == answer:
                continue
            candidates.append(item_id)

        # Không shuffle: để nhãn luôn tương ứng vị trí cố định A/B/C...
        sample = build_vip5_training_sample(
            args=self.vip5_args,
            seq=seq,
            candidates=candidates,
            label=answer,
            text_dict=self.text_dict,
            tokenizer=self.tokenizer,
            task="rec",
            max_title_len=self.sample_cfg.max_title_len,
        )
        return sample


class VIP5EvalDataset(Dataset):
    """Dataset eval (val/test) dùng candidates từ retrieved.pkl.

    - users: danh sách user id được chọn (có ground-truth trong top-k retrieval).
    - candidates_per_user: list[list[item_id]] top-k candidates per user.
    - split_dict: train/val/test dict để lấy lịch sử và label.
    - mode: 'val' hoặc 'test' để quyết định cách ghép history.
    """

    def __init__(
        self,
        vip5_args: VIP5Args,
        sample_cfg: VIP5SampleConfig,
        users: List[int],
        candidates_per_user: List[List[int]],
        train: Dict[int, List[int]],
        val: Dict[int, List[int]],
        test: Dict[int, List[int]],
        text_dict: Dict[int, Dict[str, Any]],
        tokenizer,
        mode: str = "val",
    ) -> None:
        assert mode in ("val", "test")
        self.vip5_args = vip5_args
        self.sample_cfg = sample_cfg
        self.users = users
        self.candidates_per_user = candidates_per_user
        self.train = train
        self.val = val
        self.test = test
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        user = self.users[index]
        cands = self.candidates_per_user[index]

        if self.mode == "val":
            history = self.train[user]
            label = self.val[user][0]
        else:
            history = self.train[user] + self.val[user]
            label = self.test[user][0]

        zero_flag = getattr(self.vip5_args, 'zero_metrics_if_missing', False)
        if label not in cands:
            if zero_flag:
                label = -1
            else:
                raise ValueError("Label item_id must exist in candidates for VIP5 sample.")

        seq = history[- self.vip5_args.max_text_length :]

        sample = build_vip5_training_sample(
            args=self.vip5_args,
            seq=seq,
            candidates=cands,
            label=label,
            text_dict=self.text_dict,
            tokenizer=self.tokenizer,
            task="rec",
            max_title_len=self.sample_cfg.max_title_len,
        )
        return sample


def _build_eval_users_and_candidates(
    probs: List[List[float]], labels: List[int], k: int
) -> Tuple[List[int], List[List[int]]]:
    """Chọn user & candidates giống LLMDataloader.

    - probs: list[list[score]] kích thước [num_users, num_items+1].
    - labels: list[item_id] ground-truth cho từng user.
    - k: số candidate (1 pos + k-1 neg).
    """
    if not probs or not labels:
        return [], []

    scores_tensor = torch.tensor(probs)
    labels_tensor = torch.tensor(labels).view(-1)

    topk_indices = torch.topk(scores_tensor, k, dim=1).indices  # [U, k]
    valid_mask = (labels_tensor.unsqueeze(1) == topk_indices).any(dim=1)

    users: List[int] = []
    candidates_per_user: List[List[int]] = []
    all_topk = topk_indices.tolist()

    for u_idx, valid in enumerate(valid_mask.tolist()):
        if not valid:
            continue
        user_id = u_idx + 1  # user indexing bắt đầu từ 1 trong dataset
        users.append(user_id)
        candidates_per_user.append(all_topk[u_idx])

    return users, candidates_per_user


def build_vip5_dataloaders_from_retrieved(
    vip5_args: VIP5Args,
    retrieved_payload: Dict[str, Any],
    batch_size: int = 8,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, VIP5Trainer]:
    """Xây dựng train/val/test DataLoader cho VIP5 từ retrieved.pkl + dataset.

    Trả về (train_loader, val_loader, test_loader, trainer_stub).
    Trainer_stub chỉ chứa tokenizer/model=None, để lấy tokenizer đã init.
    """

    # 1) Load dataset tuần tự (train/val/test + meta)
    dataset = dataset_factory(arg)
    data = dataset.load_dataset()
    train: Dict[int, List[int]] = data["train"]
    val: Dict[int, List[int]] = data["val"]
    test: Dict[int, List[int]] = data["test"]
    text_dict: Dict[int, Dict[str, Any]] = data["meta"]
    num_items: int = len(data["smap"])

    # 2) Chuẩn bị tokenizer & một VIP5Trainer stub để dùng tokenizer chung
    trainer_stub = VIP5Trainer(vip5_args, train_loader=None, val_loader=None, test_loader=None, train=False)
    tokenizer = trainer_stub.create_tokenizer()

    sample_cfg = VIP5SampleConfig()

    # 3) Train dataset (negative sampling từ toàn bộ item)
    train_dataset = VIP5TrainDataset(
        vip5_args=vip5_args,
        sample_cfg=sample_cfg,
        train=train,
        text_dict=text_dict,
        tokenizer=tokenizer,
        num_items=num_items,
    )

    # 4) Val/Test users từ retrieved.pkl (val_probs/val_labels/test_probs/test_labels)
    val_probs = retrieved_payload.get("val_probs", [])
    val_labels = retrieved_payload.get("val_labels", [])
    test_probs = retrieved_payload.get("test_probs", [])
    test_labels = retrieved_payload.get("test_labels", [])

    k = sample_cfg.negative_sample_size + 1
    val_users, val_cands = _build_eval_users_and_candidates(val_probs, val_labels, k)
    test_users, test_cands = _build_eval_users_and_candidates(test_probs, test_labels, k)

    val_dataset = VIP5EvalDataset(
        vip5_args=vip5_args,
        sample_cfg=sample_cfg,
        users=val_users,
        candidates_per_user=val_cands,
        train=train,
        val=val,
        test=test,
        text_dict=text_dict,
        tokenizer=tokenizer,
        mode="val",
    )

    test_dataset = VIP5EvalDataset(
        vip5_args=vip5_args,
        sample_cfg=sample_cfg,
        users=test_users,
        candidates_per_user=test_cands,
        train=train,
        val=val,
        test=test,
        text_dict=text_dict,
        tokenizer=tokenizer,
        mode="test",
    )

    # 5) Load CLIP embedding (visual features) nếu có
    preprocessed_folder = dataset._get_preprocessed_folder_path()
    clip_path = preprocessed_folder / "clip_embeddings.pt"
    if clip_path.is_file():
        payload = torch.load(clip_path, map_location="cpu")
        clip_embs = payload["image_embs"]
    else:
        clip_embs = torch.zeros((num_items + 1, 512))  # fallback dummy

    collate_fn = vip5_collate_fn_with_clip(clip_embs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, trainer_stub
