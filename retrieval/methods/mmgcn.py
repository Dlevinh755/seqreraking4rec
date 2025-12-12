from typing import Dict, List, Set

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from evaluation.metrics import recall_at_k, ndcg_at_k
from retrieval.base import BaseRetriever
from retrieval.models.mmgcn import Net


class _MMGCNTrainDataset(Dataset):
    """Pairwise (user, pos, neg) sampler cho MMGCN.

    Trả về:
    - user_tensor: LongTensor [2]  -> [u_idx, u_idx]
    - item_tensor: LongTensor [2]  -> [pos_item_node, neg_item_node]
    """

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

        # Chuyển sang index node trong MMGCN:
        # - user nodes: [0 .. user_count-1]
        # - item nodes: [user_count .. user_count+item_count-1]
        u_idx = user_id - 1
        pos_node_idx = self.user_count + (pos_item_id - 1)
        neg_node_idx = self.user_count + (neg_item_id - 1)

        user_tensor = torch.tensor([u_idx, u_idx], dtype=torch.long)
        item_tensor = torch.tensor([pos_node_idx, neg_node_idx], dtype=torch.long)
        return user_tensor, item_tensor


class MMGCNRetriever(BaseRetriever):
    """MMGCN-based retriever bọc quanh model `Net`.

    - Dùng train_edge, v_feat, t_feat, user_item_dict đã chuẩn bị ở script.
    - Train bằng pairwise loss trong `Net.loss`.
    - `retrieve` dùng embedding cuối (`Net.result`) để tính top-K items.
    """

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

        self.model: Net | None = None
        self.user_count: int | None = None
        self.item_count: int | None = None

    def _batch_retrieve(self, users: List[int]) -> Dict[int, List[int]]:
        """Trả về top-K candidates cho nhiều user cùng lúc, tận dụng GPU.

        Dùng embedding đã tính sẵn trong `self.model.result`:
        - user_embs: [num_user, dim]
        - item_embs: [num_item, dim]
        """

        self._validate_fitted()
        if self.model is None or self.user_count is None or self.item_count is None:
            raise RuntimeError("MMGCNRetriever model not initialized")

        if not users:
            return {}

        device = next(self.model.parameters()).device
        user_embs = self.model.result[: self.user_count].to(device)
        item_embs = self.model.result[self.user_count : self.user_count + self.item_count].to(device)

        k = min(self.top_k, self.item_count)
        batch_size = max(1, self.batch_size * 4)
        results: Dict[int, List[int]] = {}

        for start in range(0, len(users), batch_size):
            batch_users = users[start : start + batch_size]
            # Chuyển user_id (1-based) sang index 0-based
            u_idx = torch.tensor([u - 1 for u in batch_users], device=device, dtype=torch.long)
            u_vecs = user_embs[u_idx]  # [B, dim]

            scores = torch.matmul(u_vecs, item_embs.t())  # [B, num_item]

            # Mask history (train interactions) giống hàm retrieve()
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
        """Đánh giá Recall@k / NDCG@k trên 1 split bằng batch scoring.

        Thay vì gọi `retrieve` từng user (rất chậm), hàm này:
        - Lấy embedding user/item từ `self.model.result`.
        - Tính score cho nhiều user cùng lúc trên GPU.
        - Gọi lại `recall_at_k`, `ndcg_at_k` cho từng user (nhưng phần nặng là matmul)."""

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
        """Train MMGCN model.

        Required kwargs:
        - item_count, user_count
        - train_edge: edge_index [2, E]
        - v_feat, t_feat
        - user_item_dict: Dict[int, Set[int]]

        Optional kwargs for validation mỗi epoch:
        - val_data: Dict[int, List[int]] (validation interactions)
        - metric_k: int (cutoff cho Recall@K, NDCG@K), default = 10
        """

        item_count = kwargs.get("item_count")
        user_count = kwargs.get("user_count")
        edge_index = kwargs.get("train_edge")
        v_feat = kwargs.get("v_feat")
        t_feat = kwargs.get("t_feat")
        user_item_dict: Dict[int, Set[int]] | None = kwargs.get("user_item_dict")
        val_data: Dict[int, List[int]] | None = kwargs.get("val_data")
        metric_k: int = int(kwargs.get("metric_k", 10))

        if item_count is None or user_count is None:
            raise ValueError("MMGCNRetriever.fit requires 'item_count' and 'user_count'")
        if edge_index is None or v_feat is None or t_feat is None or user_item_dict is None:
            raise ValueError("MMGCNRetriever.fit requires 'train_edge', 'v_feat', 't_feat', 'user_item_dict'")

        self.item_count = int(item_count)
        self.user_count = int(user_count)
        self.user_history = train_data

        # Chuẩn hoá về numpy cho Net
        if isinstance(edge_index, torch.Tensor):
            edge_np = edge_index.cpu().numpy()
        else:
            edge_np = np.asarray(edge_index)

        if isinstance(v_feat, torch.Tensor):
            v_feat_np = v_feat.cpu().numpy()
        else:
            v_feat_np = np.asarray(v_feat)

        if isinstance(t_feat, torch.Tensor):
            t_feat_np = t_feat.cpu().numpy()
        else:
            t_feat_np = np.asarray(t_feat)

        # Chỉ dùng 2 modality: visual (v_feat) và text (t_feat).
        words_tensor = None

        model = Net(
            v_feat=v_feat_np,
            t_feat=t_feat_np,
            words_tensor=words_tensor,
            edge_index=edge_np,
            batch_size=self.batch_size,
            num_user=self.user_count,
            num_item=self.item_count,
            aggr_mode=self.aggr_mode,
            concate=self.concate,
            num_layer=self.num_layer,
            has_id=self.has_id,
            user_item_dict=user_item_dict,
            reg_weight=self.reg_weight,
            dim_x=self.dim_x,
        )

        model.to(self.device)

        train_dataset = _MMGCNTrainDataset(
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
                loss, reg_loss, base_loss, _, _ = model.loss(user_tensor, item_tensor)
                loss.backward()
                optimizer.step()

                bsz = user_tensor.size(0)
                seen += bsz
                total_loss += float(loss.item()) * bsz

            avg_loss = total_loss / max(1, seen)
            cur_epoch = epoch + 1
            print(f"[MMGCNRetriever] Epoch {cur_epoch}/{self.num_epochs} - loss: {avg_loss:.4f}")

            # Validation định kỳ nếu có val_data: 3 epoch 1 lần (và luôn eval epoch cuối)
            if val_data is not None and (cur_epoch % 2 == 0 or cur_epoch == self.num_epochs):
                with torch.no_grad():
                    model.eval()
                    _ = model.forward()

                    # tạm thời gán model hiện tại vào retriever để dùng `retrieve`
                    prev_model = self.model
                    prev_flag = self.is_fitted
                    self.model = model
                    self.is_fitted = True

                    val_metrics = self._evaluate_split(val_data, metric_k)
                    print(
                        f"[MMGCNRetriever]   Val users: {val_metrics['num_users']}, "
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
            _ = model.forward()

        self.model = model
        self.is_fitted = True

    def retrieve(self, user_id: int, exclude_items: Set[int] = None) -> List[int]:
        self._validate_fitted()
        if self.model is None or self.user_count is None or self.item_count is None:
            raise RuntimeError("MMGCNRetriever model not initialized")

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
