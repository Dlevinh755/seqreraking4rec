from typing import Dict, List, Set

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from evaluation.metrics import recall_at_k, ndcg_at_k
from retrieval.base import BaseRetriever
from retrieval.models.vbpr import VBPR


class _VPBRTrainDataset(Dataset):

    def __init__(
        self,
        train_data: Dict[int, List[int]],
        user_count: int,
        item_count: int,
        user_item_dict: Dict[int, Set[int]],
    ) -> None:
        self.user_count = user_count
        self.item_count = item_count
        self.user_item_dict = user_item_dict

        self.user_pos_pairs: List[tuple[int, int]] = []
        for u, items in train_data.items():
            for item in items:
                self.user_pos_pairs.append((u, item))

        self.all_items = np.arange(1, item_count + 1)

    def __len__(self) -> int:
        return len(self.user_pos_pairs)

    def __getitem__(self, idx: int):
        user_id, pos_item_id = self.user_pos_pairs[idx]

        pos_items = self.user_item_dict.get(user_id, set())
        # Negative sampling: random item không thuộc history của user.
        while True:
            neg_item_id = int(np.random.choice(self.all_items))
            if neg_item_id not in pos_items:
                break

        # Chuyển về index 0-based cho VBPR:
        # user: [0 .. user_count-1], item: [0 .. item_count-1]
        u_idx = user_id - 1
        pos_idx = pos_item_id - 1
        neg_idx = neg_item_id - 1

        # user_tensor: [2] (u, u) để tương thích loss hiện tại
        user_tensor = torch.tensor([u_idx, u_idx], dtype=torch.long)
        # item_tensor: [2] (pos, neg)
        item_tensor = torch.tensor([pos_idx, neg_idx], dtype=torch.long)
        return user_tensor, item_tensor


class VBPRRetriever(BaseRetriever):

    def __init__(
        self,
        top_k: int = 50,
        batch_size: int = 256,
        num_epochs: int = 10,
        lr: float = 1e-3,
        reg_weight: float = 1e-4,
        num_workers: int = 4,
        dim_x: int = 64,
        aggr_mode: str = "add",
        concate: bool = False,
        num_layer: int = 2,
        has_id: bool = True,
    ) -> None:
        super().__init__(top_k=top_k)
        self.user_history: Dict[int, List[int]] = {}

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.reg_weight = reg_weight
        self.num_workers = num_workers
        self.dim_x = dim_x
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: VBPR | None = None
        self.user_count: int | None = None
        self.item_count: int | None = None

    def _batch_retrieve(self, users: List[int]) -> Dict[int, List[int]]:

        self._validate_fitted()
        if self.model is None or self.user_count is None or self.item_count is None:
            raise RuntimeError("VBPRRetriever model not initialized")
        if not users:
            return {}

        device = next(self.model.parameters()).device

        # Precompute cache cho phần visual (không phụ thuộc user)
        visual_rating_space, opinion_visual_appearance = self.model.generate_cache(
            grad_enabled=False
        )  # [n_items, dim_theta], [n_items, 1]

        k = min(self.top_k, self.item_count)
        batch_size = max(1, self.batch_size * 4)
        results: Dict[int, List[int]] = {}

        all_item_indices = torch.arange(self.item_count, device=device, dtype=torch.long)
        gamma_items = self.model.gamma_items(all_item_indices)  # [n_items, dim_gamma]
        beta_items = self.model.beta_items(all_item_indices).squeeze(-1)  # [n_items]

        # opinion_visual_appearance: [n_items, 1] -> [n_items]
        ova = opinion_visual_appearance.squeeze(-1)

        for start in range(0, len(users), batch_size):
            batch_users = users[start : start + batch_size]
            # user index 0-based
            u_idx = torch.tensor([u - 1 for u in batch_users], device=device, dtype=torch.long)

            gamma_u = self.model.gamma_users(u_idx)  # [B, dim_gamma]
            theta_u = self.model.theta_users(u_idx)  # [B, dim_theta]

            # latent part: gamma_u * gamma_items
            latent_scores = torch.matmul(gamma_u, gamma_items.t())  # [B, n_items]

            # visual part: theta_u * visual_rating_space
            visual_scores = torch.matmul(theta_u, visual_rating_space.t())  # [B, n_items]

            # bias: beta_items + opinion_visual_appearance (broadcast)
            bias_scores = beta_items.unsqueeze(0) + ova.unsqueeze(0)  # [1, n_items]

            scores = latent_scores + visual_scores + bias_scores  # [B, n_items]

            # Mask history (train interactions)
            for row, u in enumerate(batch_users):
                history = self.user_history.get(u, [])
                for item in history:
                    if 1 <= item <= self.item_count:
                        scores[row, item - 1] = -1e9

            _, top_idx = torch.topk(scores, k, dim=1)
            top_idx = top_idx.cpu()

            for row, u in enumerate(batch_users):
                # +1 để chuyển về item_id 1-based
                results[u] = (top_idx[row] + 1).tolist()

        return results

    def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> Dict[str, float]:

        users = sorted(split.keys())
        # Chỉ giữ user có ground-truth
        eval_users = [u for u in users if split.get(u)]
        if not eval_users:
            return {"recall": 0.0, "ndcg": 0.0, "num_users": 0}

        # Top-K candidates cho tất cả user trong split
        user_to_cands = self._batch_retrieve(eval_users)

        recalls, ndcgs = [], []
        for u in eval_users:
            gt_items = split.get(u, [])
            recs = user_to_cands.get(u, [])
            if not recs or not gt_items:
                continue
            r = recall_at_k(recs, gt_items, k)
            n = ndcg_at_k(recs, gt_items, k)
            recalls.append(r)
            ndcgs.append(n)

        if not recalls:
            return {"recall": 0.0, "ndcg": 0.0, "num_users": 0}

        return {
            "recall": float(sum(recalls) / len(recalls)),
            "ndcg": float(sum(ndcgs) / len(ndcgs)),
            "num_users": len(recalls),
        }

    def fit(self, train_data: Dict[int, List[int]], **kwargs) -> None:
        item_count = kwargs.get("item_count")
        user_count = kwargs.get("user_count")
        v_feat = kwargs.get("v_feat")
        user_item_dict: Dict[int, Set[int]] | None = kwargs.get("user_item_dict")
        val_data: Dict[int, List[int]] | None = kwargs.get("val_data")
        metric_k: int = int(kwargs.get("metric_k", 10))

        if item_count is None or user_count is None:
            raise ValueError("VBPRRetriever.fit requires 'item_count' and 'user_count'")
        if v_feat is None or user_item_dict is None:
            raise ValueError("VBPRRetriever.fit requires 'v_feat' and 'user_item_dict'")

        self.item_count = int(item_count)
        self.user_count = int(user_count)
        self.user_history = train_data

        if isinstance(v_feat, torch.Tensor):
            v_feat_np = v_feat.cpu().numpy()
        else:
            v_feat_np = np.asarray(v_feat)

        model = VBPR(
            n_users=user_count,
            n_items=item_count,
            features=torch.tensor(v_feat_np, dtype=torch.float32),
            dim_gamma=20,
            dim_theta=10,
        )

        model.to(self.device)

        train_dataset = _VPBRTrainDataset(
            train_data=train_data,
            user_count=self.user_count,
            item_count=self.item_count,
            user_item_dict=user_item_dict,
        )

        if len(train_dataset) == 0:
            self.model = model
            self.is_fitted = True
            return

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        best_state_dict = None
        best_val_recall = -1.0

        model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            seen = 0

            for user_tensor, item_tensor in train_loader:
                user_tensor = user_tensor.to(self.device)
                item_tensor = item_tensor.to(self.device)

                optimizer.zero_grad()
                loss, reg_loss, base_loss, _, _ = model.loss(user_tensor, item_tensor, reg_weight=self.reg_weight)
                loss.backward()
                optimizer.step()

                bsz = user_tensor.size(0)
                seen += bsz
                total_loss += float(loss.item()) * bsz

            avg_loss = total_loss / max(1, seen)
            cur_epoch = epoch + 1
            print(f"[VBPRRetriever] Epoch {cur_epoch}/{self.num_epochs} - loss: {avg_loss:.4f}")

            # Validation định kỳ nếu có val_data: 3 epoch 1 lần (và luôn eval epoch cuối)
            if val_data is not None and (cur_epoch % 2 == 0 or cur_epoch == self.num_epochs):
                with torch.no_grad():
                    model.eval()

                    # tạm thời gán model hiện tại vào retriever để dùng `retrieve`
                    prev_model = self.model
                    prev_flag = self.is_fitted
                    self.model = model
                    self.is_fitted = True

                    val_metrics = self._evaluate_split(val_data, metric_k)
                    print(
                        f"[VBPRRetriever]   Val users: {val_metrics['num_users']}, "
                        f"Recall@{metric_k}: {val_metrics['recall']:.4f}, "
                        f"NDCG@{metric_k}: {val_metrics['ndcg']:.4f}"
                    )

                    # Lưu best model theo Recall@K
                    if val_metrics["recall"] > best_val_recall:
                        best_val_recall = val_metrics["recall"]
                        best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

                    # restore trạng thái cũ
                    self.model = prev_model
                    self.is_fitted = prev_flag

                model.train()

        # Sau training: dùng best model nếu có validation, ngược lại dùng model cuối
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        with torch.no_grad():
            model.eval()

        self.model = model
        self.is_fitted = True

    def retrieve(self, user_id: int, exclude_items: Set[int] = None) -> List[int]:
        self._validate_fitted()
        if self.model is None or self.user_count is None or self.item_count is None:
            raise RuntimeError("VBPRRetriever model not initialized")

        exclude_items = exclude_items or set()
        history = self.user_history.get(user_id, [])
        blocked = set(history) | set(exclude_items)

        u_idx = user_id - 1
        if not (0 <= u_idx < self.user_count):
            return []

        # Tận dụng hàm batch cho 1 user để dùng lại code
        with torch.no_grad():
            cands_dict = self._batch_retrieve([user_id])

        return cands_dict.get(user_id, [])
