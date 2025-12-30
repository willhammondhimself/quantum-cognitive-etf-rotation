"""
Stacking ensemble models for volatility prediction (Phase 5.3).

Includes:
- StackingEnsemble: Learned meta-weights based on features
- RegimeConditionalStacking: Regime-aware model weighting
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble with learned meta-weights.

    The meta-learner learns to weight base models based on input features,
    allowing it to dynamically adjust the ensemble based on market conditions.
    """

    def __init__(self,
                 base_models: List[nn.Module],
                 meta_features: int = 0,
                 hidden_dim: int = 32,
                 dropout: float = 0.1):
        """
        Parameters
        ----------
        base_models : List[nn.Module]
            List of trained base models (SimplifiedQCML, MoE, TemporalQCML, etc.)
        meta_features : int
            Number of additional meta-features (e.g., VIX, regime indicators)
            Default 0 = only use base predictions
        hidden_dim : int
            Hidden dimension for meta-learner
        dropout : float
            Dropout probability
        """
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        n_models = len(base_models)

        # Meta-learner: learns optimal combination based on features
        meta_input_dim = n_models + meta_features
        self.meta_learner = nn.Sequential(
            nn.Linear(meta_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_models),
            nn.Softmax(dim=-1)  # Weights sum to 1
        )

    def forward(self,
                x: torch.Tensor,
                meta_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through stacking ensemble.

        Parameters
        ----------
        x : torch.Tensor (batch, n_features)
            Input features
        meta_features : torch.Tensor (batch, meta_features), optional
            Additional meta-features for weighting

        Returns
        -------
        prediction : torch.Tensor (batch, 1)
            Weighted ensemble prediction
        """
        # Get base predictions
        base_preds = torch.stack([m(x) for m in self.base_models], dim=-1)
        # (batch, 1, n_models)

        # Meta-learner input
        if meta_features is not None:
            meta_input = torch.cat([base_preds.squeeze(1), meta_features], dim=-1)
        else:
            meta_input = base_preds.squeeze(1)

        # Learn weights
        weights = self.meta_learner(meta_input)  # (batch, n_models)

        # Weighted combination
        weighted_pred = (base_preds.squeeze(1) * weights).sum(dim=-1, keepdim=True)

        return weighted_pred

    def get_weights(self,
                    x: torch.Tensor,
                    meta_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get ensemble weights for interpretability.

        Returns
        -------
        weights : torch.Tensor (batch, n_models)
        """
        with torch.no_grad():
            base_preds = torch.stack([m(x) for m in self.base_models], dim=-1)

            if meta_features is not None:
                meta_input = torch.cat([base_preds.squeeze(1), meta_features], dim=-1)
            else:
                meta_input = base_preds.squeeze(1)

            weights = self.meta_learner(meta_input)

        return weights


class RegimeConditionalStacking(nn.Module):
    """
    Regime-conditional stacking ensemble.

    Uses regime probabilities from HMM/rule-based detector to weight models.
    Different regimes favor different model architectures:
    - Low vol: SimplifiedQCML performs well
    - Crisis: MoE with specialized experts performs better
    - Temporal patterns: TemporalQCML captures clustering
    """

    def __init__(self,
                 base_models: List[nn.Module],
                 n_regimes: int = 4,
                 learnable_weights: bool = True):
        """
        Parameters
        ----------
        base_models : List[nn.Module]
            List of trained base models
        n_regimes : int
            Number of volatility regimes (default 4: low, normal, high, crisis)
        learnable_weights : bool
            If True, learn regime-model weights from data
            If False, use equal weighting per regime
        """
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.n_regimes = n_regimes
        n_models = len(base_models)

        if learnable_weights:
            # Per-regime weights for each model
            # Initialize with equal weights
            self.regime_weights = nn.Parameter(
                torch.ones(n_regimes, n_models) / n_models
            )
        else:
            # Fixed equal weights
            self.register_buffer(
                'regime_weights',
                torch.ones(n_regimes, n_models) / n_models
            )

    def forward(self,
                x: torch.Tensor,
                regime_probs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with regime-conditional weighting.

        Parameters
        ----------
        x : torch.Tensor (batch, n_features)
            Input features
        regime_probs : torch.Tensor (batch, n_regimes)
            Regime probabilities from detector (should sum to 1 per sample)

        Returns
        -------
        prediction : torch.Tensor (batch, 1)
            Regime-weighted ensemble prediction
        """
        # Get base predictions
        base_preds = torch.stack([m(x) for m in self.base_models], dim=-1)
        # (batch, 1, n_models)

        # Compute regime-weighted model weights
        # regime_probs: (batch, n_regimes)
        # regime_weights: (n_regimes, n_models)
        # weights: (batch, n_models)
        weights = torch.matmul(regime_probs, self.regime_weights)

        # Apply softmax to ensure weights sum to 1
        weights = F.softmax(weights, dim=-1)

        # Weighted combination
        weighted_pred = (base_preds.squeeze(1) * weights).sum(dim=-1, keepdim=True)

        return weighted_pred

    def get_regime_weights(self) -> torch.Tensor:
        """
        Get regime-model weight matrix for interpretability.

        Returns
        -------
        weights : torch.Tensor (n_regimes, n_models)
        """
        return self.regime_weights.detach()


class AdaptiveEnsemble(nn.Module):
    """
    Adaptive ensemble that learns to select or blend models based on input.

    Uses gating network to decide whether to:
    1. Select a single best model
    2. Blend multiple models
    """

    def __init__(self,
                 base_models: List[nn.Module],
                 gating_hidden: int = 64,
                 use_hard_selection: bool = False):
        """
        Parameters
        ----------
        base_models : List[nn.Module]
            List of base models
        gating_hidden : int
            Hidden dimension for gating network
        use_hard_selection : bool
            If True, select single best model (argmax)
            If False, use soft weighting
        """
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.use_hard_selection = use_hard_selection
        n_models = len(base_models)

        # Gating network
        # Takes base predictions as input and outputs model weights
        self.gating = nn.Sequential(
            nn.Linear(n_models, gating_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gating_hidden, n_models)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive gating.

        Parameters
        ----------
        x : torch.Tensor (batch, n_features)

        Returns
        -------
        prediction : torch.Tensor (batch, 1)
        """
        # Get base predictions
        base_preds = torch.stack([m(x) for m in self.base_models], dim=-1)
        # (batch, 1, n_models)

        # Gating network takes base predictions as input
        gate_input = base_preds.squeeze(1)  # (batch, n_models)
        gate_logits = self.gating(gate_input)  # (batch, n_models)

        if self.use_hard_selection:
            # Hard selection (Gumbel-Softmax during training, argmax during inference)
            if self.training:
                # Gumbel-Softmax for differentiability
                weights = F.gumbel_softmax(gate_logits, tau=1.0, hard=True)
            else:
                # Argmax during inference
                weights = F.one_hot(gate_logits.argmax(dim=-1), num_classes=len(self.base_models)).float()
        else:
            # Soft weighting
            weights = F.softmax(gate_logits, dim=-1)

        # Weighted combination
        weighted_pred = (base_preds.squeeze(1) * weights).sum(dim=-1, keepdim=True)

        return weighted_pred
