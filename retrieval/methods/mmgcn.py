"""MMGCN-based retriever using multimodal graph convolutional network."""

from typing import Dict, List, Set, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np

from retrieval.base import BaseRetriever
from retrieval.models.mmgcn import Net


class MMGCNRetriever(BaseRetriever):
    """Retriever sử dụng MMGCN (Multimodal Graph Convolutional Network).
    
    Wrapper cho Net model để implement BaseRetriever interface.
    MMGCN sử dụng visual và text features từ CLIP embeddings.
    """

    def __init__(
        self,
        top_k: int = 50,
        dim_x: int = 64,
        aggr_mode: str = "add",
        concate: bool = True,
        num_layer: int = 3,
        has_id: bool = True,
        reg_weight: float = 1e-4,
        batch_size: int = 128,
        num_epochs: int = 10,
        lr: float = 1e-3,
        patience: Optional[int] = None,
    ) -> None:
        """
        Args:
            top_k: Số lượng candidates trả về
            dim_x: Embedding dimension
            aggr_mode: Aggregation mode ("add", "mean", "max")
            concate: Có concat features không
            num_layer: Số GCN layers
            has_id: Có dùng ID embedding không
            reg_weight: Regularization weight
            batch_size: Batch size cho training
            num_epochs: Số epochs
            lr: Learning rate
            patience: Early stopping patience
        """
        super().__init__(top_k=top_k)
        self.dim_x = dim_x
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id
        self.reg_weight = reg_weight
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.patience = patience
        
        self.model: Optional[Net] = None
        self.num_user: Optional[int] = None
        self.num_item: Optional[int] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        train_data: Dict[int, List[int]],
        **kwargs: Any
    ) -> None:
        """Train MMGCN model.
        
        Args:
            train_data: Dict {user_id: [item_ids]} - training interactions
            **kwargs: Additional arguments:
                - num_user: int - số users
                - num_item: int - số items
                - v_feat: np.ndarray - visual features [num_items, D]
                - t_feat: np.ndarray - text features [num_items, D]
                - edge_index: np.ndarray - graph edges [2, E]
        """
        self.num_user = kwargs.get("num_user")
        self.num_item = kwargs.get("num_item")
        v_feat = kwargs.get("v_feat")
        t_feat = kwargs.get("t_feat")
        edge_index = kwargs.get("edge_index")
        
        if any(x is None for x in [self.num_user, self.num_item, v_feat, t_feat, edge_index]):
            raise ValueError(
                "MMGCNRetriever.fit requires: num_user, num_item, v_feat, t_feat, edge_index"
            )
        
        # Build user-item dict for training
        user_item_dict = {u: items for u, items in train_data.items()}
        
        # Initialize model
        words_tensor = None  # Not used in current implementation
        self.model = Net(
            v_feat=v_feat,
            t_feat=t_feat,
            words_tensor=words_tensor,
            edge_index=edge_index,
            batch_size=self.batch_size,
            num_user=self.num_user,
            num_item=self.num_item,
            aggr_mode=self.aggr_mode,
            concate=self.concate,
            num_layer=self.num_layer,
            has_id=self.has_id,
            user_item_dict=user_item_dict,
            reg_weight=self.reg_weight,
            dim_x=self.dim_x,
        ).to(self.device)
        
        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_state = None
        best_val_recall = -1.0
        epochs_no_improve = 0
        
        val_data = kwargs.get("val_data")
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            # Sample training batches
            for user_id, items in train_data.items():
                if len(items) < 2:
                    continue
                
                # Sample positive and negative
                pos_item = items[-1]
                neg_item = np.random.choice([
                    i for i in range(1, self.num_item + 1) 
                    if i not in items
                ])
                
                user_tensor = torch.tensor([user_id, user_id], dtype=torch.long)
                item_tensor = torch.tensor([pos_item, neg_item], dtype=torch.long)
                
                optimizer.zero_grad()
                loss, _, _, _, _ = self.model.loss(
                    user_tensor.to(self.device),
                    item_tensor.to(self.device)
                )
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches >= len(train_data) // self.batch_size:
                    break
            
            avg_loss = total_loss / max(1, num_batches)
            
            # Validation
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    self.model.forward()  # Update self.result
                
                val_recall = self._evaluate_split(val_data, k=min(10, self.top_k))
                
                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    best_state = self.model.state_dict().copy()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                print(f"[MMGCNRetriever] Epoch {epoch+1}/{self.num_epochs} - "
                      f"loss: {avg_loss:.4f}, val_Recall@{min(10, self.top_k)}: {val_recall:.4f}")
                
                if self.patience and epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"[MMGCNRetriever] Epoch {epoch+1}/{self.num_epochs} - loss: {avg_loss:.4f}")
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self.is_fitted = True

    def _evaluate_split(self, split: Dict[int, List[int]], k: int) -> float:
        """Compute average Recall@K for validation."""
        if self.model is None:
            return 0.0
        
        self.model.eval()
        with torch.no_grad():
            self.model.forward()  # Update self.result
        
        user_tensor = self.model.result[:self.num_user]
        item_tensor = self.model.result[self.num_user:]
        
        recalls = []
        for user_id, gt_items in split.items():
            if user_id >= self.num_user:
                continue
            
            scores = torch.matmul(user_tensor[user_id:user_id+1], item_tensor.t())
            _, top_items = torch.topk(scores, k=k)
            top_items = top_items[0].cpu().numpy() + self.num_user  # Adjust for indexing
            
            hits = len(set(top_items) & set(gt_items))
            if len(gt_items) > 0:
                recalls.append(hits / min(k, len(gt_items)))
        
        return float(np.mean(recalls)) if recalls else 0.0

    def retrieve(
        self,
        user_id: int,
        exclude_items: Set[int] = None
    ) -> List[int]:
        """Retrieve top-K candidates cho một user.
        
        Args:
            user_id: ID của user
            exclude_items: Set các items cần loại trừ
            
        Returns:
            List[int]: Top-K item IDs
        """
        self._validate_fitted()
        
        if self.model is None or self.num_user is None or self.num_item is None:
            raise RuntimeError("MMGCNRetriever model not initialized")
        
        if user_id >= self.num_user:
            return []
        
        exclude_items = exclude_items or set()
        
        # Get user and item embeddings
        self.model.eval()
        with torch.no_grad():
            self.model.forward()  # Update self.result
        
        user_emb = self.model.result[user_id:user_id+1]
        item_emb = self.model.result[self.num_user:]
        
        # Compute scores
        scores = torch.matmul(user_emb, item_emb.t()).squeeze(0)
        
        # Mask excluded items
        for item in exclude_items:
            if 1 <= item <= self.num_item:
                item_idx = item - 1  # Convert to 0-indexed
                scores[item_idx] = -1e9
        
        # Get top-K
        k = min(self.top_k, self.num_item)
        _, indices = torch.topk(scores, k=k)
        
        # Convert back to 1-indexed item IDs
        return [int(idx.item() + 1) for idx in indices]

