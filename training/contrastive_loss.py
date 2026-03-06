"""
RouteMoA Training — Dual Contrastive Loss

Implements the paper's dual contrastive loss for training the SLM scorer:
- Part 1 (Query→Model): Pull queries toward models that answered correctly
- Part 2 (Query→Query): Group semantically similar queries together
- Combined: L = L_query_model + α * L_query_query
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualContrastiveLoss(nn.Module):
    """
    Dual contrastive loss as described in the RouteMoA paper.
    
    L = L_query_model + α * L_query_query
    
    L_query_model: Pulls query embeddings toward good-model embeddings
                   and pushes away from bad-model embeddings.
    L_query_query: Groups semantically similar queries (same cluster) together.
    """

    def __init__(self, temperature: float = 0.07, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def query_model_loss(
        self,
        query_embeddings: torch.Tensor,
        model_embeddings: torch.Tensor,
        quality_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive loss between queries and model embeddings.

        Args:
            query_embeddings: (batch, hidden_dim) — encoded queries
            model_embeddings: (num_models, hidden_dim) — learnable model embeddings
            quality_scores: (batch, num_models) — ground truth quality scores [0,1]

        Returns:
            Scalar loss
        """
        # Normalize embeddings
        q_norm = F.normalize(query_embeddings, dim=-1)
        m_norm = F.normalize(model_embeddings, dim=-1)

        # Similarity matrix: (batch, num_models)
        sim = q_norm @ m_norm.T / self.temperature

        # Positive models: quality > 0.5, Negative: quality <= 0.5
        positives = (quality_scores > 0.5).float()
        negatives = (quality_scores <= 0.5).float()

        # For each query, compute InfoNCE-style loss
        # Pull toward positive models, push away from negatives
        exp_sim = torch.exp(sim)

        # Positive attraction
        pos_sim = (exp_sim * positives).sum(dim=-1)
        # Total (positive + negative)
        total_sim = exp_sim.sum(dim=-1)

        # Avoid log(0)
        loss = -torch.log(pos_sim / (total_sim + 1e-8) + 1e-8)

        return loss.mean()

    def query_query_loss(
        self,
        query_embeddings: torch.Tensor,
        cluster_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive loss between queries in the same semantic cluster.

        Args:
            query_embeddings: (batch, hidden_dim)
            cluster_labels: (batch,) — integer cluster IDs from K-means

        Returns:
            Scalar loss
        """
        q_norm = F.normalize(query_embeddings, dim=-1)

        # Pairwise similarity: (batch, batch)
        sim = q_norm @ q_norm.T / self.temperature

        # Create mask: 1 if same cluster, 0 otherwise
        labels_eq = (cluster_labels.unsqueeze(0) == cluster_labels.unsqueeze(1)).float()
        # Remove self-similarity
        mask = labels_eq - torch.eye(labels_eq.size(0), device=labels_eq.device)

        # If no positive pairs, return 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=query_embeddings.device)

        exp_sim = torch.exp(sim)
        # Mask out self-similarity for denominator
        self_mask = 1.0 - torch.eye(sim.size(0), device=sim.device)
        denominator = (exp_sim * self_mask).sum(dim=-1)
        numerator = (exp_sim * mask).sum(dim=-1)

        # Per-query loss
        loss = -torch.log(numerator / (denominator + 1e-8) + 1e-8)

        # Only average over queries that have at least one positive pair
        has_positive = mask.sum(dim=-1) > 0
        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=query_embeddings.device)

        return loss[has_positive].mean()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        model_embeddings: torch.Tensor,
        quality_scores: torch.Tensor,
        cluster_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Combined dual contrastive loss.

        Returns:
            (total_loss, {"l_qm": float, "l_qq": float})
        """
        l_qm = self.query_model_loss(query_embeddings, model_embeddings, quality_scores)
        l_qq = self.query_query_loss(query_embeddings, cluster_labels)

        total = l_qm + self.alpha * l_qq

        return total, {
            "l_query_model": l_qm.item(),
            "l_query_query": l_qq.item(),
            "total": total.item(),
        }
