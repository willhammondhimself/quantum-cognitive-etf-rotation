"""
Loss functions for training.

Includes MSE, ranking loss, and combined losses for QCML model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard MSE loss."""
    return F.mse_loss(predictions, targets)


def ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.01,
    n_pairs: Optional[int] = None
) -> torch.Tensor:
    """
    Pairwise ranking loss.

    Encourages the model to correctly rank samples by their target values.
    For each pair (i, j) where target_i > target_j, we want pred_i > pred_j.

    Parameters
    ----------
    predictions : Tensor of shape (n,)
        Model predictions.
    targets : Tensor of shape (n,)
        True labels.
    margin : float
        Margin for hinge loss.
    n_pairs : int, optional
        Number of pairs to sample. If None, use all pairs.

    Returns
    -------
    loss : scalar Tensor
    """
    n = len(predictions)
    if n < 2:
        return torch.tensor(0.0, device=predictions.device)

    # Create all pairs
    pred_i = predictions.unsqueeze(1)  # (n, 1)
    pred_j = predictions.unsqueeze(0)  # (1, n)
    target_i = targets.unsqueeze(1)    # (n, 1)
    target_j = targets.unsqueeze(0)    # (1, n)

    # Mask for pairs where target_i > target_j
    positive_pairs = (target_i > target_j).float()

    # Hinge loss: max(0, margin - (pred_i - pred_j)) for positive pairs
    diff = pred_i - pred_j
    hinge = F.relu(margin - diff)

    # Apply mask and compute mean
    loss = (positive_pairs * hinge).sum()
    n_positive = positive_pairs.sum()

    if n_positive > 0:
        loss = loss / n_positive
    else:
        loss = torch.tensor(0.0, device=predictions.device)

    return loss


def sampled_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_pairs: int = 10,
    margin: float = 0.01
) -> torch.Tensor:
    """
    Sampled pairwise ranking loss.

    More efficient than full ranking loss for large batches.
    Randomly samples pairs and computes hinge loss.

    Parameters
    ----------
    predictions : Tensor of shape (n,)
    targets : Tensor of shape (n,)
    n_pairs : int
        Number of pairs to sample.
    margin : float
        Margin for hinge loss.

    Returns
    -------
    loss : scalar Tensor
    """
    n = len(predictions)
    if n < 2:
        return torch.tensor(0.0, device=predictions.device)

    # Sample random pairs
    n_pairs = min(n_pairs, n * (n - 1) // 2)

    idx_i = torch.randint(0, n, (n_pairs,), device=predictions.device)
    idx_j = torch.randint(0, n, (n_pairs,), device=predictions.device)

    # Ensure i != j
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    if len(idx_i) == 0:
        return torch.tensor(0.0, device=predictions.device)

    pred_i = predictions[idx_i]
    pred_j = predictions[idx_j]
    target_i = targets[idx_i]
    target_j = targets[idx_j]

    # Swap to ensure target_i > target_j
    swap_mask = target_i < target_j
    pred_i_new = torch.where(swap_mask, pred_j, pred_i)
    pred_j_new = torch.where(swap_mask, pred_i, pred_j)

    # Hinge loss
    diff = pred_i_new - pred_j_new
    hinge = F.relu(margin - diff)

    return hinge.mean()


def combined_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mse_weight: float = 0.7,
    ranking_weight: float = 0.3,
    ranking_margin: float = 0.01,
    ranking_pairs: int = 10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined MSE and ranking loss.

    Parameters
    ----------
    predictions : Tensor
    targets : Tensor
    mse_weight : float
        Weight for MSE loss.
    ranking_weight : float
        Weight for ranking loss.
    ranking_margin : float
        Margin for ranking hinge loss.
    ranking_pairs : int
        Number of pairs to sample for ranking.

    Returns
    -------
    total_loss : combined loss
    mse : MSE component
    ranking : ranking loss component
    """
    mse = mse_loss(predictions, targets)
    ranking = sampled_ranking_loss(predictions, targets, ranking_pairs, ranking_margin)

    total = mse_weight * mse + ranking_weight * ranking

    return total, mse, ranking


class CombinedLoss(nn.Module):
    """
    Module wrapper for combined loss.

    Useful for cleaner training loops.
    """

    def __init__(
        self,
        mse_weight: float = 0.7,
        ranking_weight: float = 0.3,
        ranking_margin: float = 0.01,
        ranking_pairs: int = 10
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.ranking_weight = ranking_weight
        self.ranking_margin = ranking_margin
        self.ranking_pairs = ranking_pairs

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss.

        Returns total loss and dict with component values.
        """
        total, mse, ranking = combined_loss(
            predictions,
            targets,
            self.mse_weight,
            self.ranking_weight,
            self.ranking_margin,
            self.ranking_pairs
        )

        components = {
            "mse": mse.item(),
            "ranking": ranking.item(),
            "total": total.item()
        }

        return total, components
