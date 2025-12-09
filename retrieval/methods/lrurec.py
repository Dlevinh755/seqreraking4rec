from typing import Dict, List, Set

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from retrieval.base import BaseRetriever
from retrieval.models.neural_lru import NeuralLRUConfig, NeuralLRURec


class _NeuralLRUTrainDataset(Dataset):
    """Simple sequence dataset for training NeuralLRURec.

    For each user sequence, we create (tokens, labels):
    - `tokens`: history without the last item, padded/truncated to max_len.
    - `labels`: full history, padded/truncated to max_len.

    This works with a standard CrossEntropyLoss over all positions.
    """

    def __init__(self, user2seq: Dict[int, List[int]], max_len: int) -> None:
        self.max_len = max_len
        self.samples: List[tuple[List[int], List[int]]] = []

        for seq in user2seq.values():
            if len(seq) < 2:
                continue
            labels = seq[-self.max_len :]
            tokens = seq[:-1][-self.max_len :]

            pad_t = self.max_len - len(tokens)
            pad_l = self.max_len - len(labels)
            tokens = [0] * pad_t + tokens
            labels = [0] * pad_l + labels
            self.samples.append((tokens, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        tokens, labels = self.samples[idx]
        return torch.LongTensor(tokens), torch.LongTensor(labels)


class LRURecRetriever(BaseRetriever):
    """Neural LRURec-based retriever using an internal implementation.

    - Does NOT import or depend on `LlamaRec/`.
    - `fit` runs a small number of training epochs on user histories.
    - `retrieve` feeds a user's sequence and returns top-K items from the
      predicted scores at the last position.
    """

    def __init__(
        self,
        top_k: int = 50,
        max_len: int = 50,
        hidden_units: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        batch_size: int = 128,
        num_epochs: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(top_k=top_k)
        self.user_history: Dict[int, List[int]] = {}
        self.max_len = max_len
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.model: NeuralLRURec | None = None
        self.item_count: int | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_data: Dict[int, List[int]], **kwargs) -> None:
        """Train NeuralLRURec on user→sequence data.

        Expected kwargs:
        - item_count: total number of items (len(smap)).
        """

        item_count = kwargs.get("item_count")
        if item_count is None:
            raise ValueError("LRURecRetriever.fit requires 'item_count' in kwargs")
        self.item_count = int(item_count)
        self.user_history = train_data

        cfg = NeuralLRUConfig(
            num_items=self.item_count,
            hidden_units=self.hidden_units,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
        )

        model = NeuralLRURec(cfg).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        train_dataset = _NeuralLRUTrainDataset(train_data, max_len=self.max_len)
        if len(train_dataset) == 0:
            self.model = model
            self.is_fitted = True
            return

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            seen = 0
            for tokens, labels in train_loader:
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                scores = model(tokens)  # [B, L, num_items+1]
                logits = scores.view(-1, scores.size(-1))
                labels_flat = labels.view(-1)
                loss = criterion(logits, labels_flat)
                loss.backward()
                optimizer.step()

                batch_size = tokens.size(0)
                seen += batch_size
                total_loss += float(loss.item()) * batch_size

            avg_loss = total_loss / max(1, seen)
            print(f"[LRURecRetriever] Epoch {epoch+1}/{self.num_epochs} - loss: {avg_loss:.4f}")

        self.model = model
        self.is_fitted = True

    def retrieve(self, user_id: int, exclude_items: Set[int] = None) -> List[int]:
        self._validate_fitted()
        if self.model is None or self.item_count is None:
            raise RuntimeError("LRURecRetriever model not initialized")

        exclude_items = exclude_items or set()
        seq = self.user_history.get(user_id, [])
        if not seq:
            return []

        # Build input sequence for this user.
        seq_tokens = seq[-self.max_len :]
        pad_len = self.max_len - len(seq_tokens)
        seq_tokens = [0] * pad_len + seq_tokens

        with torch.no_grad():
            self.model.eval()
            inputs = torch.LongTensor([seq_tokens]).to(self.device)
            scores = self.model(inputs)[:, -1, :]  # [1, num_items+1]
            scores = scores.squeeze(0)

        # Mask out history and padding index 0.
        for item in seq:
            if 0 < item <= self.item_count:
                scores[item] = -1e9
        scores[0] = -1e9

        for item in exclude_items:
            if 0 < item <= self.item_count:
                scores[item] = -1e9

        k = min(self.top_k, self.item_count)
        _, indices = torch.topk(scores, k=k)
        return [int(i.item()) for i in indices]
