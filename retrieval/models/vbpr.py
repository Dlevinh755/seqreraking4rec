"""This module contains a VBPR implementation in PyTorch."""

from typing import Optional, Tuple, Union, cast

import torch
from torch import nn


class VBPR(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        features: torch.Tensor,
        dim_gamma: int,
        dim_theta: int,
    ):
        super().__init__()

        # Image features
        self.features = nn.Embedding.from_pretrained(
            features, freeze=True
        )  # type: ignore

        # Latent factors (gamma)
        self.gamma_users = nn.Embedding(n_users, dim_gamma)
        self.gamma_items = nn.Embedding(n_items, dim_gamma)

        # Visual factors (theta)
        self.theta_users = nn.Embedding(n_users, dim_theta)
        self.embedding = nn.Embedding(features.size(1), dim_theta)

        # Biases (beta)
        # self.beta_users = nn.Embedding(n_users, 1)
        self.beta_items = nn.Embedding(n_items, 1)
        self.visual_bias = nn.Embedding(features.size(1), 1)

        # Random weight initialization
        self.reset_parameters()

    def forward(
        self, ui: torch.Tensor, pi: torch.Tensor, ni: torch.Tensor
    ) -> torch.Tensor:
        # User
        ui_latent_factors = self.gamma_users(ui)  # Latent factors of user u
        ui_visual_factors = self.theta_users(ui)  # Visual factors of user u
        
        # Items
        pi_bias = self.beta_items(pi)  # Pos. item bias
        ni_bias = self.beta_items(ni)  # Neg. item bias
        
        pi_latent_factors = self.gamma_items(pi)  # Pos. item visual factors
        ni_latent_factors = self.gamma_items(ni)  # Neg. item visual factors
        pi_features = self.features(pi)  # Pos. item visual features
        ni_features = self.features(ni)  # Neg. item visual features

        # Precompute differences
        diff_features = pi_features - ni_features
        diff_latent_factors = pi_latent_factors - ni_latent_factors

        # x_uij
        x_uij = (
            pi_bias
            - ni_bias
            + (ui_latent_factors * diff_latent_factors).sum(dim=1).unsqueeze(-1)
            + (ui_visual_factors * diff_features.mm(self.embedding.weight))
            .sum(dim=1)
            .unsqueeze(-1)
            + diff_features.mm(self.visual_bias.weight)
        )

        return cast(torch.Tensor, x_uij.squeeze())




    def reset_parameters(self) -> None:
        """Resets network weights.

        Restart network weights using a Xavier uniform distribution.
        """
        # Latent factors (gamma)
        nn.init.xavier_uniform_(self.gamma_users.weight)
        nn.init.xavier_uniform_(self.gamma_items.weight)

        # Visual factors (theta)
        nn.init.xavier_uniform_(self.theta_users.weight)
        nn.init.xavier_uniform_(self.embedding.weight)

        # Biases (beta)
        nn.init.xavier_uniform_(self.beta_items.weight)
        nn.init.xavier_uniform_(self.visual_bias.weight)

    def generate_cache(
        self, grad_enabled: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precalculate intermediate values before calculating recommendations"""
        with torch.set_grad_enabled(grad_enabled):
            i_features = self.features.weight  # Items visual features
            visual_rating_space = i_features.mm(self.embedding.weight)
            opinion_visual_appearance = i_features.mm(self.visual_bias.weight)
        return visual_rating_space, opinion_visual_appearance
    
    def loss(self, user_tensor: torch.Tensor, item_tensor: torch.Tensor, reg_weight: float = 1e-4):
        """BPR loss cho VBPR.

        Parameters
        ----------
        user_tensor : Tensor
            Tensor có shape [B, 2] hoặc [B], trong đó cột đầu là user index (0-based).
            Trong code hiện tại, _VPBRTrainDataset trả về [u, u] nên ta dùng cột 0.
        item_tensor : Tensor
            Tensor có shape [B, 2], trong đó [:, 0] là pos item, [:, 1] là neg item
            (đều là index 0-based của item).
        reg_weight : float
            Hệ số regularization L2.
        """
        # B: batch size
        if user_tensor.dim() == 1:
            ui = user_tensor.long()
        else:
            ui = user_tensor[:, 0].long()

        if item_tensor.dim() == 1 or item_tensor.size(1) < 2:
            raise ValueError("item_tensor phải có shape [B, 2] với (pos_item, neg_item)")

        pi = item_tensor[:, 0].long()
        ni = item_tensor[:, 1].long()

        # x_uij = x_ui_pos - x_ui_neg từ forward
        x_uij = self.forward(ui, pi, ni)  # [B]

        # BPR loss ổn định số: -log(sigmoid(x)) = softplus(-x)
        base_loss = torch.nn.functional.softplus(-x_uij).mean()

        # L2 regularization trên một số tham số chính
        gamma_u = self.gamma_users(ui)
        gamma_pi = self.gamma_items(pi)
        gamma_ni = self.gamma_items(ni)
        theta_u = self.theta_users(ui)
        beta_pi = self.beta_items(pi)
        beta_ni = self.beta_items(ni)

        reg_embedding_loss = (
            gamma_u.pow(2).mean()
            + gamma_pi.pow(2).mean()
            + gamma_ni.pow(2).mean()
            + theta_u.pow(2).mean()
            + self.embedding.weight.pow(2).mean()
            + beta_pi.pow(2).mean()
            + beta_ni.pow(2).mean()
            + self.visual_bias.weight.pow(2).mean()
        )

        reg_loss = reg_weight * reg_embedding_loss
        loss = base_loss + reg_loss

        return loss, reg_loss, base_loss, reg_embedding_loss, reg_embedding_loss
    
    def full_accuracy(self, val_data, step=2000, topk=10):
        user_tensor = self.result[:self.num_user]
        item_tensor = self.result[self.num_user:]

        start_index = 0
        end_index = self.num_user if step==None else step

        all_index_of_rank_list = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

            for row, col in self.user_item_dict.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col))-self.num_user
                    score_matrix[row][col] = 1e-5

            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()+self.num_user), dim=0)
            start_index = end_index
            
            if end_index+step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        length = 0        
        precision = recall = ndcg = 0.0

        for data in val_data:
            user = data[0]
            pos_items = set(data[1:])
            num_pos = len(pos_items)
            if num_pos == 0:
                continue
            length += 1
            items_list = all_index_of_rank_list[user].tolist()

            items = set(items_list)

            num_hit = len(pos_items.intersection(items))
            
            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)

            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(num_pos, topk)):
                max_ndcg_score += 1 / math.log2(i+2)
            if max_ndcg_score == 0:
                continue
                
            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / math.log2(i+2)

            ndcg += ndcg_score/max_ndcg_score

        return precision/length, recall/length, ndcg/length