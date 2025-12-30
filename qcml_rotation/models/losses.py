"""
Advanced loss functions for volatility prediction (Phase 5.4).

Includes:
- AsymmetricVolLoss: Penalize under-prediction of high volatility
- QuantileLoss: Multi-quantile distribution prediction
- CombinedVolLoss: Weighted combination of losses
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricVolLoss(nn.Module):
    """
    Asymmetric loss that penalizes under-prediction of high volatility.

    Motivation: Under-predicting volatility during turbulent periods leads to:
    - Insufficient hedging
    - Risk limit breaches
    - Larger drawdowns

    Over-predicting volatility is safer (conservative positioning).
    """

    def __init__(self, alpha: float = 2.0, threshold_pct: float = 0.7):
        """
        Parameters
        ----------
        alpha : float
            Penalty multiplier for under-predicting high volatility.
            alpha=2.0 means 2x penalty for under-prediction.
        threshold_pct : float
            Percentile above which volatility is considered "high".
            0.7 = top 30% of samples.
        """
        super().__init__()
        self.alpha = alpha
        self.threshold_pct = threshold_pct

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss.

        Parameters
        ----------
        pred : torch.Tensor (batch, 1)
            Predicted volatility
        target : torch.Tensor (batch, 1)
            Actual realized volatility

        Returns
        -------
        loss : torch.Tensor (scalar)
        """
        errors = pred - target

        # Identify high volatility samples
        threshold = torch.quantile(target, self.threshold_pct)
        high_vol_mask = target > threshold
        under_pred_mask = errors < 0

        # Standard squared errors
        base_loss = errors ** 2

        # Apply extra penalty for under-predicting high volatility
        penalty_mask = high_vol_mask & under_pred_mask
        base_loss[penalty_mask] *= self.alpha

        return base_loss.mean()


class QuantileLoss(nn.Module):
    """
    Multi-quantile loss for distribution prediction.

    Instead of predicting a single point estimate, predict multiple quantiles
    to capture uncertainty and tail risk.
    """

    def __init__(self, quantiles: List[float] = None):
        """
        Parameters
        ----------
        quantiles : List[float]
            Quantiles to predict. Default: [0.1, 0.25, 0.5, 0.75, 0.9]
        """
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.quantiles = quantiles

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.

        Parameters
        ----------
        pred : torch.Tensor (batch, n_quantiles)
            Predicted quantiles
        target : torch.Tensor (batch, 1)
            Actual realized volatility

        Returns
        -------
        loss : torch.Tensor (scalar)
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - pred[:, i:i+1]
            # Pinball loss: asymmetric penalty based on quantile
            losses.append(torch.max(q * errors, (q - 1) * errors))

        return torch.cat(losses, dim=1).mean()


class CombinedVolLoss(nn.Module):
    """
    Combined loss: MSE + Asymmetric + Ranking.

    Balances:
    - MSE: Overall accuracy
    - Asymmetric: Safety (conservative high-vol predictions)
    - Ranking: Relative ordering (important for portfolio allocation)
    """

    def __init__(self,
                 mse_weight: float = 0.5,
                 asym_weight: float = 0.3,
                 rank_weight: float = 0.2,
                 alpha: float = 2.0):
        """
        Parameters
        ----------
        mse_weight : float
            Weight for MSE loss
        asym_weight : float
            Weight for asymmetric loss
        rank_weight : float
            Weight for ranking loss
        alpha : float
            Asymmetric loss penalty multiplier
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.asym_weight = asym_weight
        self.rank_weight = rank_weight
        self.asym_loss = AsymmetricVolLoss(alpha=alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.

        Parameters
        ----------
        pred : torch.Tensor (batch, 1)
            Predicted volatility
        target : torch.Tensor (batch, 1)
            Actual realized volatility

        Returns
        -------
        loss : torch.Tensor (scalar)
        """
        # MSE loss
        mse = F.mse_loss(pred, target)

        # Asymmetric loss
        asym = self.asym_loss(pred, target)

        # Ranking loss (Spearman-like)
        pred_ranks = pred.argsort().argsort().float()
        target_ranks = target.argsort().argsort().float()
        rank_loss = F.mse_loss(pred_ranks, target_ranks)

        # Weighted combination
        total_loss = (
            self.mse_weight * mse +
            self.asym_weight * asym +
            self.rank_weight * rank_loss
        )

        return total_loss


class UncertaintyAwareLoss(nn.Module):
    """
    Gaussian negative log-likelihood with uncertainty.

    Model predicts both mean and variance. Higher uncertainty in volatile
    periods leads to wider distributions and lower penalties for errors.
    """

    def __init__(self, min_std: float = 0.01):
        """
        Parameters
        ----------
        min_std : float
            Minimum allowed standard deviation to prevent collapse
        """
        super().__init__()
        self.min_std = min_std

    def forward(self,
                pred_mean: torch.Tensor,
                pred_log_std: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian NLL.

        Parameters
        ----------
        pred_mean : torch.Tensor (batch, 1)
            Predicted mean volatility
        pred_log_std : torch.Tensor (batch, 1)
            Predicted log standard deviation
        target : torch.Tensor (batch, 1)
            Actual realized volatility

        Returns
        -------
        loss : torch.Tensor (scalar)
        """
        std = torch.exp(pred_log_std).clamp(min=self.min_std)
        variance = std ** 2

        # Gaussian negative log-likelihood
        nll = 0.5 * (torch.log(variance) + ((target - pred_mean) ** 2) / variance)

        return nll.mean()
