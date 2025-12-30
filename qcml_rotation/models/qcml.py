"""
Quantum-Cognitive Market Learning (QCML) Model.

Implements a quantum-inspired model that represents market states as
amplitude vectors in a complex Hilbert space. Predictions are made
via expectation values of a learned Hermitian observable.

Key concepts:
- Features are mapped to complex amplitude vectors |ψ⟩ ∈ C^d
- The state is normalized: ||ψ|| = 1
- A Hermitian matrix W represents the "observable"
- Prediction: y = Re(⟨ψ|W|ψ⟩)

The interference/ranking loss encourages correct relative ordering
of predictions within each week, mimicking quantum interference
between competing "concept states".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class QCMLConfig:
    """Configuration for QCML model."""
    hilbert_dim: int = 16           # Dimension of complex Hilbert space
    encoder_hidden: int = 32        # Hidden layer size in encoder
    use_complex: bool = True        # Use complex embeddings (vs real-only ablation)
    lr: float = 0.001
    epochs: int = 150
    batch_size: int = 64
    ranking_weight: float = 0.3     # Weight for ranking loss (1 - this = MSE weight)
    ranking_margin: float = 0.01    # Margin for pairwise hinge loss
    pairs_per_week: int = 10        # Number of pairs to sample per week
    early_stopping_patience: int = 15
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5


class HilbertSpaceEncoder(nn.Module):
    """
    Maps real feature vectors to complex amplitude vectors in Hilbert space.

    Architecture: F -> hidden -> 2*d (representing real and imaginary parts)
    Then normalize to unit vector in C^d.
    """

    def __init__(
        self,
        input_dim: int,
        hilbert_dim: int = 16,
        hidden_dim: int = 32,
        use_complex: bool = True
    ):
        super().__init__()

        self.hilbert_dim = hilbert_dim
        self.use_complex = use_complex

        # Output dimension: 2*d for complex (real + imag), d for real-only
        output_dim = 2 * hilbert_dim if use_complex else hilbert_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode features to normalized state vector.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)
            Input features.

        Returns
        -------
        psi : Tensor of shape (batch, 2*hilbert_dim) or (batch, hilbert_dim)
            Normalized state vector. If complex, first half is real, second is imag.
        """
        z = self.encoder(x)

        # Normalize to unit vector
        if self.use_complex:
            # Compute complex norm: sqrt(sum of |z_i|^2)
            # z = [real_1, ..., real_d, imag_1, ..., imag_d]
            d = self.hilbert_dim
            real = z[:, :d]
            imag = z[:, d:]
            norm = torch.sqrt((real ** 2 + imag ** 2).sum(dim=1, keepdim=True) + 1e-8)
            psi = z / norm
        else:
            # Real-only: standard L2 normalization
            norm = torch.norm(z, dim=1, keepdim=True) + 1e-8
            psi = z / norm

        return psi


class HermitianObservable(nn.Module):
    """
    Learnable Hermitian matrix W for computing expectation values.

    For a Hermitian matrix: W = W^†
    We parameterize it as W = A + A^† where A is a general complex matrix.

    For real symmetric case: W = A + A^T
    """

    def __init__(self, dim: int, use_complex: bool = True):
        super().__init__()

        self.dim = dim
        self.use_complex = use_complex

        if use_complex:
            # A is a complex matrix, stored as two real matrices
            self.A_real = nn.Parameter(torch.randn(dim, dim) * 0.1)
            self.A_imag = nn.Parameter(torch.randn(dim, dim) * 0.1)
        else:
            # A is a real matrix
            self.A = nn.Parameter(torch.randn(dim, dim) * 0.1)

    def get_hermitian(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Construct the Hermitian matrix W = A + A^†.

        Returns
        -------
        W_real : Tensor of shape (dim, dim)
        W_imag : Tensor of shape (dim, dim) or None if real-only
        """
        if self.use_complex:
            # W = A + A^†
            # (A + A^†)_real = A_real + A_real^T
            # (A + A^†)_imag = A_imag - A_imag^T
            W_real = self.A_real + self.A_real.T
            W_imag = self.A_imag - self.A_imag.T
            return W_real, W_imag
        else:
            # W = A + A^T (symmetric)
            W = self.A + self.A.T
            return W, None

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Compute expectation value ⟨ψ|W|ψ⟩.

        Parameters
        ----------
        psi : Tensor of shape (batch, 2*dim) or (batch, dim)
            Normalized state vector.

        Returns
        -------
        expectation : Tensor of shape (batch,)
            Real part of expectation value.
        """
        W_real, W_imag = self.get_hermitian()

        if self.use_complex:
            d = self.dim
            psi_real = psi[:, :d]      # (batch, d)
            psi_imag = psi[:, d:]      # (batch, d)

            # ⟨ψ|W|ψ⟩ for complex case
            # W|ψ⟩ = (W_real + i*W_imag)(ψ_real + i*ψ_imag)
            #      = (W_real*ψ_real - W_imag*ψ_imag) + i*(W_real*ψ_imag + W_imag*ψ_real)

            Wpsi_real = torch.matmul(psi_real, W_real) - torch.matmul(psi_imag, W_imag)
            Wpsi_imag = torch.matmul(psi_real, W_imag) + torch.matmul(psi_imag, W_real)

            # ⟨ψ|W|ψ⟩ = ψ^†(W|ψ⟩) = (ψ_real - i*ψ_imag)·(Wpsi_real + i*Wpsi_imag)
            #         = ψ_real·Wpsi_real + ψ_imag·Wpsi_imag + i*(ψ_real·Wpsi_imag - ψ_imag·Wpsi_real)
            # Take real part:
            expectation = (psi_real * Wpsi_real).sum(dim=1) + (psi_imag * Wpsi_imag).sum(dim=1)

        else:
            # Real case: ⟨ψ|W|ψ⟩ = ψ^T W ψ
            Wpsi = torch.matmul(psi, W_real)  # (batch, d)
            expectation = (psi * Wpsi).sum(dim=1)

        return expectation


class QCML(nn.Module):
    """
    Quantum-Cognitive Market Learning model.

    Combines the Hilbert space encoder and Hermitian observable
    to produce predictions from features.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[QCMLConfig] = None
    ):
        super().__init__()

        if config is None:
            config = QCMLConfig()

        self.config = config

        self.encoder = HilbertSpaceEncoder(
            input_dim=input_dim,
            hilbert_dim=config.hilbert_dim,
            hidden_dim=config.encoder_hidden,
            use_complex=config.use_complex
        )

        self.observable = HermitianObservable(
            dim=config.hilbert_dim,
            use_complex=config.use_complex
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: features -> prediction.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)

        Returns
        -------
        y_hat : Tensor of shape (batch,)
            Predicted excess returns.
        """
        psi = self.encoder(x)
        y_hat = self.observable(psi)
        return y_hat

    def get_state_vectors(self, x: torch.Tensor) -> torch.Tensor:
        """Get the encoded state vectors (useful for analysis)."""
        return self.encoder(x)

    def get_observable_matrix(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get the Hermitian observable matrix."""
        return self.observable.get_hermitian()


class QCMLWithRanking(nn.Module):
    """
    QCML with built-in ranking loss computation.

    This wrapper handles both individual predictions and
    week-grouped ranking loss computation.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[QCMLConfig] = None
    ):
        super().__init__()

        if config is None:
            config = QCMLConfig()

        self.config = config
        self.qcml = QCML(input_dim, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.qcml(x)

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        week_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined MSE + ranking loss.

        Parameters
        ----------
        features : Tensor of shape (batch, n_features)
        labels : Tensor of shape (batch,)
        week_indices : Tensor of shape (batch,), optional
            Week index for each sample. If provided, ranking loss
            is computed within each week.

        Returns
        -------
        total_loss : Tensor
        components : dict with loss breakdown
        """
        predictions = self.forward(features)

        # MSE loss
        mse = F.mse_loss(predictions, labels)

        # Ranking loss
        if week_indices is not None and self.config.ranking_weight > 0:
            ranking = self._compute_ranking_loss(predictions, labels, week_indices)
        else:
            ranking = self._compute_global_ranking_loss(predictions, labels)

        # Combine losses
        mse_weight = 1.0 - self.config.ranking_weight
        total = mse_weight * mse + self.config.ranking_weight * ranking

        components = {
            "total": total.item(),
            "mse": mse.item(),
            "ranking": ranking.item()
        }

        return total, components

    def _compute_ranking_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        week_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute ranking loss within each week."""
        unique_weeks = week_indices.unique()
        total_loss = 0.0
        n_pairs = 0

        for week in unique_weeks:
            mask = week_indices == week
            week_preds = predictions[mask]
            week_labels = labels[mask]

            if len(week_preds) < 2:
                continue

            # Sample pairs within this week
            loss, pairs = self._pairwise_ranking_loss(week_preds, week_labels)
            total_loss += loss * pairs
            n_pairs += pairs

        if n_pairs > 0:
            return total_loss / n_pairs
        return torch.tensor(0.0, device=predictions.device)

    def _compute_global_ranking_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute ranking loss over entire batch (fallback)."""
        if len(predictions) < 2:
            return torch.tensor(0.0, device=predictions.device)

        loss, _ = self._pairwise_ranking_loss(predictions, labels)
        return loss

    def _pairwise_ranking_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute pairwise hinge ranking loss.

        For pairs where label_i > label_j, we want pred_i > pred_j.
        Loss: max(0, margin - (pred_i - pred_j))
        """
        n = len(predictions)
        margin = self.config.ranking_margin
        n_pairs_to_sample = min(self.config.pairs_per_week, n * (n - 1) // 2)

        if n_pairs_to_sample <= 0:
            return torch.tensor(0.0, device=predictions.device), 0

        # Sample random pairs
        idx_i = torch.randint(0, n, (n_pairs_to_sample,), device=predictions.device)
        idx_j = torch.randint(0, n, (n_pairs_to_sample,), device=predictions.device)

        # Ensure i != j
        valid = idx_i != idx_j
        idx_i = idx_i[valid]
        idx_j = idx_j[valid]

        if len(idx_i) == 0:
            return torch.tensor(0.0, device=predictions.device), 0

        pred_i = predictions[idx_i]
        pred_j = predictions[idx_j]
        label_i = labels[idx_i]
        label_j = labels[idx_j]

        # Determine which should be ranked higher
        # If label_i > label_j, we want pred_i - pred_j > margin
        # If label_j > label_i, we want pred_j - pred_i > margin

        diff_label = label_i - label_j
        diff_pred = pred_i - pred_j

        # For positive diff_label, encourage positive diff_pred
        # For negative diff_label, encourage negative diff_pred
        # This is equivalent to: encourage sign(diff_pred) == sign(diff_label)

        # Use signed margin approach:
        # loss = max(0, margin - sign(diff_label) * diff_pred)
        target_sign = torch.sign(diff_label)
        # Handle ties (diff_label == 0) by ignoring them
        non_tie = diff_label.abs() > 1e-8

        if non_tie.sum() == 0:
            return torch.tensor(0.0, device=predictions.device), 0

        target_sign = target_sign[non_tie]
        diff_pred = diff_pred[non_tie]

        hinge = F.relu(margin - target_sign * diff_pred)
        loss = hinge.mean()

        return loss, len(target_sign)


def create_qcml_model(
    input_dim: int,
    config: Optional[QCMLConfig] = None,
    device: Optional[torch.device] = None
) -> QCMLWithRanking:
    """
    Factory function to create QCML model.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    config : QCMLConfig, optional
        Model configuration.
    device : torch.device, optional
        Device to place model on.

    Returns
    -------
    model : QCMLWithRanking
    """
    model = QCMLWithRanking(input_dim, config)

    if device is not None:
        model = model.to(device)

    return model


# =============================================================================
# SimplifiedQCML - Stripped down version that actually works
# =============================================================================

@dataclass
class SimplifiedQCMLConfig:
    """Configuration for Simplified QCML model.

    Key differences from original QCML:
    - NO unit normalization (preserves magnitude information)
    - Real numbers only (no complex overhead)
    - Larger dimensions (hidden=64, embed=32)
    - Dropout for regularization
    - Optional batch normalization
    """
    # Architecture
    hidden_dim: int = 64            # Hidden layer size (was 32)
    embed_dim: int = 32             # Embedding dimension (was 16)
    n_hidden_layers: int = 2        # Number of hidden layers
    dropout: float = 0.1           # Dropout rate
    use_batch_norm: bool = True    # Use batch normalization

    # Training
    lr: float = 0.001
    weight_decay: float = 1e-4      # L2 regularization
    epochs: int = 150
    batch_size: int = 64

    # Loss configuration
    ranking_weight: float = 0.3     # Weight for ranking loss
    ranking_margin: float = 0.01    # Margin for pairwise hinge loss
    pairs_per_week: int = 10        # Pairs to sample per week

    # Early stopping
    early_stopping_patience: int = 15
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5


class SimplifiedEncoder(nn.Module):
    """
    Simplified encoder WITHOUT unit normalization.

    Key insight: The original QCML's unit normalization destroys magnitude
    information. If input features have meaningful magnitudes (they do -
    momentum, volatility, etc.), normalizing to unit sphere loses this.

    This encoder preserves magnitude through:
    1. No normalization at output
    2. Batch normalization for training stability (not information loss)
    3. Larger dimensions to capture more structure
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Embedding layer (NO normalization here!)
        layers.append(nn.Linear(hidden_dim, embed_dim))

        self.encoder = nn.Sequential(*layers)

        # Initialize with smaller weights to prevent explosion
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode features to embedding space (NO normalization).

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)

        Returns
        -------
        embedding : Tensor of shape (batch, embed_dim)
            Unnormalized embedding preserving magnitude information.
        """
        return self.encoder(x)


class SimplifiedPredictor(nn.Module):
    """
    Simple prediction head without Hermitian constraints.

    The original QCML's Hermitian observable adds mathematical
    elegance but may constrain the function class unnecessarily.
    This uses a simple linear layer + bias.
    """

    def __init__(self, embed_dim: int, n_outputs: int = 1):
        super().__init__()

        self.n_outputs = n_outputs

        # Simple linear projection to output
        self.predictor = nn.Linear(embed_dim, n_outputs)

        # Initialize near zero for stable training
        nn.init.xavier_uniform_(self.predictor.weight, gain=0.1)
        nn.init.zeros_(self.predictor.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict from embedding.

        Parameters
        ----------
        embedding : Tensor of shape (batch, embed_dim)

        Returns
        -------
        prediction : Tensor of shape (batch,) or (batch, n_outputs)
        """
        out = self.predictor(embedding)
        if self.n_outputs == 1:
            return out.squeeze(-1)
        return out


class SimplifiedQCML(nn.Module):
    """
    Simplified QCML without the problematic components.

    What's removed:
    1. Unit normalization - destroys magnitude info
    2. Complex numbers - adds overhead without benefit
    3. Hermitian constraint - may limit function class

    What's added:
    1. Batch normalization - stable training
    2. Dropout - regularization
    3. Larger dimensions - more capacity
    4. Weight decay in optimizer - L2 regularization

    If this doesn't show predictive power, then the problem is
    fundamental (features, targets, or market efficiency), not
    the architecture.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[SimplifiedQCMLConfig] = None
    ):
        super().__init__()

        if config is None:
            config = SimplifiedQCMLConfig()

        self.config = config

        self.encoder = SimplifiedEncoder(
            input_dim=input_dim,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            n_hidden_layers=config.n_hidden_layers,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm
        )

        self.predictor = SimplifiedPredictor(
            embed_dim=config.embed_dim,
            n_outputs=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: features -> prediction.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)

        Returns
        -------
        y_hat : Tensor of shape (batch,)
        """
        embedding = self.encoder(x)
        return self.predictor(embedding)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for analysis."""
        return self.encoder(x)


class SimplifiedQCMLWithRanking(nn.Module):
    """
    Simplified QCML with ranking loss support.

    Mirrors QCMLWithRanking interface for easy substitution.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[SimplifiedQCMLConfig] = None
    ):
        super().__init__()

        if config is None:
            config = SimplifiedQCMLConfig()

        self.config = config
        self.model = SimplifiedQCML(input_dim, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.model(x)

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        week_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined MSE + ranking loss.

        Same interface as QCMLWithRanking for compatibility.
        """
        predictions = self.forward(features)

        # MSE loss
        mse = F.mse_loss(predictions, labels)

        # Ranking loss
        if week_indices is not None and self.config.ranking_weight > 0:
            ranking = self._compute_ranking_loss(predictions, labels, week_indices)
        else:
            ranking = self._compute_global_ranking_loss(predictions, labels)

        # Combine losses
        mse_weight = 1.0 - self.config.ranking_weight
        total = mse_weight * mse + self.config.ranking_weight * ranking

        components = {
            "total": total.item(),
            "mse": mse.item(),
            "ranking": ranking.item()
        }

        return total, components

    def _compute_ranking_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        week_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute ranking loss within each week."""
        unique_weeks = week_indices.unique()
        total_loss = 0.0
        n_pairs = 0

        for week in unique_weeks:
            mask = week_indices == week
            week_preds = predictions[mask]
            week_labels = labels[mask]

            if len(week_preds) < 2:
                continue

            loss, pairs = self._pairwise_ranking_loss(week_preds, week_labels)
            total_loss += loss * pairs
            n_pairs += pairs

        if n_pairs > 0:
            return total_loss / n_pairs
        return torch.tensor(0.0, device=predictions.device)

    def _compute_global_ranking_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute ranking loss over entire batch."""
        if len(predictions) < 2:
            return torch.tensor(0.0, device=predictions.device)

        loss, _ = self._pairwise_ranking_loss(predictions, labels)
        return loss

    def _pairwise_ranking_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Compute pairwise hinge ranking loss."""
        n = len(predictions)
        margin = self.config.ranking_margin
        n_pairs_to_sample = min(self.config.pairs_per_week, n * (n - 1) // 2)

        if n_pairs_to_sample <= 0:
            return torch.tensor(0.0, device=predictions.device), 0

        idx_i = torch.randint(0, n, (n_pairs_to_sample,), device=predictions.device)
        idx_j = torch.randint(0, n, (n_pairs_to_sample,), device=predictions.device)

        valid = idx_i != idx_j
        idx_i = idx_i[valid]
        idx_j = idx_j[valid]

        if len(idx_i) == 0:
            return torch.tensor(0.0, device=predictions.device), 0

        pred_i = predictions[idx_i]
        pred_j = predictions[idx_j]
        label_i = labels[idx_i]
        label_j = labels[idx_j]

        diff_label = label_i - label_j
        diff_pred = pred_i - pred_j

        target_sign = torch.sign(diff_label)
        non_tie = diff_label.abs() > 1e-8

        if non_tie.sum() == 0:
            return torch.tensor(0.0, device=predictions.device), 0

        target_sign = target_sign[non_tie]
        diff_pred = diff_pred[non_tie]

        hinge = F.relu(margin - target_sign * diff_pred)
        loss = hinge.mean()

        return loss, len(target_sign)


def create_simplified_qcml_model(
    input_dim: int,
    config: Optional[SimplifiedQCMLConfig] = None,
    device: Optional[torch.device] = None
) -> SimplifiedQCMLWithRanking:
    """
    Factory function to create Simplified QCML model.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    config : SimplifiedQCMLConfig, optional
        Model configuration.
    device : torch.device, optional
        Device to place model on.

    Returns
    -------
    model : SimplifiedQCMLWithRanking
    """
    model = SimplifiedQCMLWithRanking(input_dim, config)

    if device is not None:
        model = model.to(device)

    return model


# =============================================================================
# RankNet Loss and Ranking-focused QCML
# =============================================================================

class RankNetLoss(nn.Module):
    """
    RankNet loss for learning-to-rank.

    Uses pairwise cross-entropy to learn correct ranking order.
    More robust than raw MSE for financial prediction where we care
    about relative ordering, not exact values.

    Reference: Burges et al. "Learning to Rank using Gradient Descent"
    """

    def __init__(self, sigma: float = 1.0):
        """
        Parameters
        ----------
        sigma : float
            Scaling factor for score differences.
        """
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        week_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute RankNet loss.

        Parameters
        ----------
        scores : Tensor of shape (batch,)
            Model predictions.
        labels : Tensor of shape (batch,)
            True labels.
        week_indices : Tensor of shape (batch,), optional
            Week indices for within-week ranking.

        Returns
        -------
        loss : Tensor
        """
        if week_indices is not None:
            return self._compute_weekly_loss(scores, labels, week_indices)
        return self._compute_global_loss(scores, labels)

    def _compute_global_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute RankNet loss over all pairs."""
        n = len(scores)
        if n < 2:
            return torch.tensor(0.0, device=scores.device)

        # Create pairwise differences
        # For efficiency, sample pairs instead of all n^2
        n_pairs = min(n * 10, n * (n - 1) // 2)

        idx_i = torch.randint(0, n, (n_pairs,), device=scores.device)
        idx_j = torch.randint(0, n, (n_pairs,), device=scores.device)

        # Filter out same indices
        valid = idx_i != idx_j
        idx_i = idx_i[valid]
        idx_j = idx_j[valid]

        if len(idx_i) == 0:
            return torch.tensor(0.0, device=scores.device)

        s_i = scores[idx_i]
        s_j = scores[idx_j]
        y_i = labels[idx_i]
        y_j = labels[idx_j]

        # Target: P(i > j) based on labels
        # If y_i > y_j, target = 1
        # If y_i < y_j, target = 0
        # If y_i == y_j, target = 0.5
        diff = y_i - y_j
        target = (torch.sign(diff) + 1) / 2  # Map to [0, 1]

        # RankNet probability
        pred = torch.sigmoid(self.sigma * (s_i - s_j))

        # Binary cross-entropy
        eps = 1e-7
        loss = -target * torch.log(pred + eps) - (1 - target) * torch.log(1 - pred + eps)

        return loss.mean()

    def _compute_weekly_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        week_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute RankNet loss within each week."""
        unique_weeks = week_indices.unique()
        total_loss = 0.0
        n_pairs = 0

        for week in unique_weeks:
            mask = week_indices == week
            week_scores = scores[mask]
            week_labels = labels[mask]

            if len(week_scores) < 2:
                continue

            loss = self._compute_global_loss(week_scores, week_labels)
            n_week = len(week_scores)
            total_loss += loss * n_week
            n_pairs += n_week

        if n_pairs > 0:
            return total_loss / n_pairs
        return torch.tensor(0.0, device=scores.device)


class ListNetLoss(nn.Module):
    """
    ListNet loss - listwise learning-to-rank.

    Uses top-one probability distribution matching.
    More efficient than RankNet for small lists (like 10-15 ETFs per week).

    Reference: Cao et al. "Learning to Rank: From Pairwise Approach to Listwise Approach"
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        week_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ListNet loss within each week.

        Parameters
        ----------
        scores : Tensor of shape (batch,)
            Model predictions.
        labels : Tensor of shape (batch,)
            True labels.
        week_indices : Tensor of shape (batch,)
            Week indices.

        Returns
        -------
        loss : Tensor
        """
        unique_weeks = week_indices.unique()
        total_loss = 0.0
        n_weeks = 0

        for week in unique_weeks:
            mask = week_indices == week
            week_scores = scores[mask]
            week_labels = labels[mask]

            if len(week_scores) < 2:
                continue

            # Top-one probabilities
            p_true = F.softmax(week_labels / self.temperature, dim=0)
            p_pred = F.softmax(week_scores / self.temperature, dim=0)

            # Cross-entropy between distributions
            loss = -torch.sum(p_true * torch.log(p_pred + 1e-8))

            total_loss += loss
            n_weeks += 1

        if n_weeks > 0:
            return total_loss / n_weeks
        return torch.tensor(0.0, device=scores.device)


@dataclass
class RankingQCMLConfig:
    """Configuration for Ranking-focused QCML.

    Key difference: Uses ranking loss instead of MSE for training.
    """
    # Architecture
    hidden_dim: int = 64
    embed_dim: int = 32
    n_hidden_layers: int = 2
    dropout: float = 0.2          # Higher dropout
    use_batch_norm: bool = True

    # Training
    lr: float = 0.001
    weight_decay: float = 1e-3    # Stronger L2
    epochs: int = 150
    batch_size: int = 64

    # Loss configuration
    mse_weight: float = 0.3       # Small MSE component
    ranking_weight: float = 0.7   # Primary ranking objective
    ranking_type: str = 'ranknet'  # 'ranknet' or 'listnet'
    ranknet_sigma: float = 1.0
    listnet_temp: float = 1.0

    # Early stopping
    early_stopping_patience: int = 20
    lr_scheduler_patience: int = 7
    lr_scheduler_factor: float = 0.5


class RankingQCML(nn.Module):
    """
    Ranking-focused QCML model.

    Key insight from experiments:
    - Rank prediction works better than return prediction (XGBoost Sharpe went from -0.11 to +0.35)
    - We don't need to predict exact returns, just relative ordering

    This model uses RankNet/ListNet loss to learn correct ranking.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[RankingQCMLConfig] = None
    ):
        super().__init__()

        if config is None:
            config = RankingQCMLConfig()

        self.config = config

        # Use the simplified encoder (no normalization)
        self.encoder = SimplifiedEncoder(
            input_dim=input_dim,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            n_hidden_layers=config.n_hidden_layers,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm
        )

        self.predictor = SimplifiedPredictor(
            embed_dim=config.embed_dim,
            n_outputs=1
        )

        # Loss functions
        if config.ranking_type == 'ranknet':
            self.ranking_loss = RankNetLoss(sigma=config.ranknet_sigma)
        else:
            self.ranking_loss = ListNetLoss(temperature=config.listnet_temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        embedding = self.encoder(x)
        return self.predictor(embedding)

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        week_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined ranking + MSE loss.
        """
        predictions = self.forward(features)

        # MSE component (for magnitude calibration)
        mse = F.mse_loss(predictions, labels)

        # Ranking component (for correct ordering)
        ranking = self.ranking_loss(predictions, labels, week_indices)

        # Combined loss
        total = (
            self.config.mse_weight * mse +
            self.config.ranking_weight * ranking
        )

        components = {
            "total": total.item(),
            "mse": mse.item(),
            "ranking": ranking.item()
        }

        return total, components

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for analysis."""
        return self.encoder(x)


def create_ranking_qcml_model(
    input_dim: int,
    config: Optional[RankingQCMLConfig] = None,
    device: Optional[torch.device] = None
) -> RankingQCML:
    """
    Factory function to create Ranking QCML model.
    """
    model = RankingQCML(input_dim, config)

    if device is not None:
        model = model.to(device)

    return model


# =============================================================================
# QuantumEnhancedQCML - Research Paper Implementation
# =============================================================================

@dataclass
class QuantumEnhancedConfig:
    """Configuration for Quantum-Enhanced QCML model.

    Key features for research paper:
    1. Amplitude-phase encoding (r, θ) preserving magnitude
    2. Multiple observables (4 Hermitian matrices)
    3. Dual-pathway interference (momentum vs volatility)
    4. Adaptive Hilbert dimension
    """
    # Architecture
    hilbert_dim: int = 32           # Dimension of Hilbert space
    n_observables: int = 4          # Number of Hermitian observables
    hidden_dim: int = 64            # Hidden layer size
    n_hidden_layers: int = 2        # Number of hidden layers
    dropout: float = 0.1            # Dropout rate
    use_batch_norm: bool = True     # Use batch normalization

    # Dual-pathway configuration
    use_dual_pathway: bool = True   # Enable interference pathways
    momentum_features: Optional[List[int]] = None  # Indices of momentum features
    volatility_features: Optional[List[int]] = None  # Indices of volatility features
    interference_alpha: float = 0.5  # Initial interference weight (learnable)

    # Training
    lr: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 150
    batch_size: int = 64

    # Loss configuration
    mse_weight: float = 0.3
    ranking_weight: float = 0.7
    ranking_type: str = 'ranknet'

    # Early stopping
    early_stopping_patience: int = 20
    lr_scheduler_patience: int = 7
    lr_scheduler_factor: float = 0.5


class AmplitudePhaseEncoder(nn.Module):
    """
    Encodes features into amplitude-phase representation.

    Maps input features x to complex state |ψ⟩ = r·exp(iθ) where:
    - Amplitude r ∈ (0, 1) captures magnitude information
    - Phase θ ∈ (-π, π) captures directional information

    This preserves magnitude (unlike unit normalization) while providing
    the rich structure of complex representations for interference.

    Mathematical formulation:
        r = σ(W_r · h + b_r)     # Sigmoid for bounded amplitude
        θ = π · tanh(W_θ · h + b_θ)  # Full phase range
        |ψ⟩ = [r₁·cos(θ₁), r₁·sin(θ₁), ..., r_d·cos(θ_d), r_d·sin(θ_d)]
    """

    def __init__(
        self,
        input_dim: int,
        hilbert_dim: int = 32,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.hilbert_dim = hilbert_dim

        # Build feature extraction layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.feature_extractor = nn.Sequential(*layers)

        # Separate heads for amplitude and phase
        self.amplitude_head = nn.Linear(hidden_dim, hilbert_dim)
        self.phase_head = nn.Linear(hidden_dim, hilbert_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode features to amplitude-phase representation.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)

        Returns
        -------
        psi : Tensor of shape (batch, 2*hilbert_dim)
            Complex state in real representation [real, imag]
        amplitude : Tensor of shape (batch, hilbert_dim)
            Amplitude r ∈ (0, 1)
        phase : Tensor of shape (batch, hilbert_dim)
            Phase θ ∈ (-π, π)
        """
        # Extract features
        h = self.feature_extractor(x)

        # Compute amplitude (bounded in (0, 1))
        r = torch.sigmoid(self.amplitude_head(h))

        # Compute phase (bounded in (-π, π))
        theta = np.pi * torch.tanh(self.phase_head(h))

        # Convert to Cartesian form: ψ = r·exp(iθ) = r·cos(θ) + i·r·sin(θ)
        psi_real = r * torch.cos(theta)  # (batch, hilbert_dim)
        psi_imag = r * torch.sin(theta)  # (batch, hilbert_dim)

        # Concatenate real and imaginary parts
        psi = torch.cat([psi_real, psi_imag], dim=1)  # (batch, 2*hilbert_dim)

        return psi, r, theta

    def get_amplitude_phase(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get amplitude and phase separately for analysis."""
        _, r, theta = self.forward(x)
        return r, theta


class MultiObservable(nn.Module):
    """
    Multiple Hermitian observables for diverse measurements.

    Instead of a single observable W, we use n_observables different
    Hermitian matrices, each potentially capturing different aspects:
    - W₁: Short-term momentum patterns
    - W₂: Long-term mean-reversion
    - W₃: Volatility characteristics
    - W₄: Cross-sectional relative strength

    The final prediction is a learnable combination:
        y = Σᵢ αᵢ ⟨ψ|Wᵢ|ψ⟩

    This is analogous to measuring multiple observables in quantum mechanics
    and combining the results.
    """

    def __init__(self, dim: int, n_observables: int = 4):
        super().__init__()

        self.dim = dim
        self.n_observables = n_observables

        # Parameterize each Hermitian matrix as W = A + A†
        # Store real and imaginary parts separately
        self.A_real = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim) * 0.1)
            for _ in range(n_observables)
        ])
        self.A_imag = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim) * 0.1)
            for _ in range(n_observables)
        ])

        # Learnable combination weights
        self.combination_weights = nn.Parameter(torch.ones(n_observables) / n_observables)

    def get_hermitian(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct Hermitian matrix W = A + A† for observable idx.

        Returns
        -------
        W_real : Tensor of shape (dim, dim)
        W_imag : Tensor of shape (dim, dim)
        """
        A_real = self.A_real[idx]
        A_imag = self.A_imag[idx]

        # W = A + A†
        # W_real = A_real + A_real^T
        # W_imag = A_imag - A_imag^T (for Hermiticity)
        W_real = A_real + A_real.T
        W_imag = A_imag - A_imag.T

        return W_real, W_imag

    def compute_expectation(
        self,
        psi: torch.Tensor,
        W_real: torch.Tensor,
        W_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute expectation value ⟨ψ|W|ψ⟩.

        Parameters
        ----------
        psi : Tensor of shape (batch, 2*dim)
            Complex state vector [real, imag]
        W_real, W_imag : Tensors of shape (dim, dim)
            Hermitian observable

        Returns
        -------
        expectation : Tensor of shape (batch,)
            Real part of expectation value
        """
        d = self.dim
        psi_real = psi[:, :d]  # (batch, dim)
        psi_imag = psi[:, d:]  # (batch, dim)

        # W|ψ⟩ = (W_real + iW_imag)(ψ_real + iψ_imag)
        #      = (W_real·ψ_real - W_imag·ψ_imag) + i(W_real·ψ_imag + W_imag·ψ_real)
        Wpsi_real = torch.matmul(psi_real, W_real) - torch.matmul(psi_imag, W_imag)
        Wpsi_imag = torch.matmul(psi_real, W_imag) + torch.matmul(psi_imag, W_real)

        # ⟨ψ|W|ψ⟩ = ψ†(W|ψ⟩)
        # Real part: ψ_real·Wpsi_real + ψ_imag·Wpsi_imag
        expectation = (psi_real * Wpsi_real).sum(dim=1) + (psi_imag * Wpsi_imag).sum(dim=1)

        return expectation

    def forward(self, psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute combined expectation from all observables.

        Parameters
        ----------
        psi : Tensor of shape (batch, 2*dim)

        Returns
        -------
        prediction : Tensor of shape (batch,)
            Weighted combination of expectation values
        individual_expectations : Tensor of shape (batch, n_observables)
            Each observable's expectation value (for analysis)
        """
        expectations = []

        for i in range(self.n_observables):
            W_real, W_imag = self.get_hermitian(i)
            exp_i = self.compute_expectation(psi, W_real, W_imag)
            expectations.append(exp_i)

        # Stack expectations: (batch, n_observables)
        expectations = torch.stack(expectations, dim=1)

        # Weighted combination with softmax-normalized weights
        weights = F.softmax(self.combination_weights, dim=0)
        prediction = (expectations * weights).sum(dim=1)

        return prediction, expectations

    def get_eigenspectrum(self) -> List[np.ndarray]:
        """
        Compute eigenvalues of each observable for analysis.

        Returns list of eigenvalue arrays for paper visualization.
        """
        eigenvalues = []

        with torch.no_grad():
            for i in range(self.n_observables):
                W_real, W_imag = self.get_hermitian(i)

                # Construct full complex matrix
                W = W_real.cpu().numpy() + 1j * W_imag.cpu().numpy()

                # Eigenvalues of Hermitian matrix are real
                eigs = np.linalg.eigvalsh(W)
                eigenvalues.append(eigs)

        return eigenvalues


class DualPathwayInterference(nn.Module):
    """
    Dual-pathway interference module for quantum-inspired feature processing.

    Creates two parallel encoding pathways:
    - Pathway A: Momentum/trend features → |ψ_A⟩
    - Pathway B: Volatility/risk features → |ψ_B⟩

    The pathways interfere:
        |ψ_total⟩ = α|ψ_A⟩ + β|ψ_B⟩

    where α, β are learnable complex coefficients.

    This is inspired by quantum interference where different paths
    to the same outcome can constructively or destructively interfere.
    """

    def __init__(
        self,
        input_dim: int,
        hilbert_dim: int = 32,
        hidden_dim: int = 64,
        momentum_indices: Optional[List[int]] = None,
        volatility_indices: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.hilbert_dim = hilbert_dim

        # Default: split features in half if not specified
        if momentum_indices is None and volatility_indices is None:
            momentum_indices = list(range(input_dim // 2))
            volatility_indices = list(range(input_dim // 2, input_dim))

        self.momentum_indices = momentum_indices
        self.volatility_indices = volatility_indices

        momentum_dim = len(momentum_indices)
        volatility_dim = len(volatility_indices)

        # Pathway A: Momentum encoder
        self.pathway_A = AmplitudePhaseEncoder(
            input_dim=momentum_dim,
            hilbert_dim=hilbert_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

        # Pathway B: Volatility encoder
        self.pathway_B = AmplitudePhaseEncoder(
            input_dim=volatility_dim,
            hilbert_dim=hilbert_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

        # Learnable interference coefficients (complex)
        # α = α_r + i·α_i, β = β_r + i·β_i
        self.alpha_real = nn.Parameter(torch.tensor(0.7))
        self.alpha_imag = nn.Parameter(torch.tensor(0.0))
        self.beta_real = nn.Parameter(torch.tensor(0.7))
        self.beta_imag = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process features through dual pathways with interference.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)

        Returns
        -------
        psi_total : Tensor of shape (batch, 2*hilbert_dim)
            Interfered state vector
        pathway_info : dict
            Contains pathway states and amplitudes for analysis
        """
        # Extract feature subsets
        x_momentum = x[:, self.momentum_indices]
        x_volatility = x[:, self.volatility_indices]

        # Encode through separate pathways
        psi_A, r_A, theta_A = self.pathway_A(x_momentum)
        psi_B, r_B, theta_B = self.pathway_B(x_volatility)

        # Complex multiplication for interference
        # |ψ_total⟩ = α|ψ_A⟩ + β|ψ_B⟩
        d = self.hilbert_dim

        # Split into real/imag parts
        psi_A_real = psi_A[:, :d]
        psi_A_imag = psi_A[:, d:]
        psi_B_real = psi_B[:, :d]
        psi_B_imag = psi_B[:, d:]

        # α|ψ_A⟩ = (α_r + iα_i)(ψ_A_r + iψ_A_i)
        alpha_psi_A_real = self.alpha_real * psi_A_real - self.alpha_imag * psi_A_imag
        alpha_psi_A_imag = self.alpha_real * psi_A_imag + self.alpha_imag * psi_A_real

        # β|ψ_B⟩
        beta_psi_B_real = self.beta_real * psi_B_real - self.beta_imag * psi_B_imag
        beta_psi_B_imag = self.beta_real * psi_B_imag + self.beta_imag * psi_B_real

        # Sum for interference
        psi_total_real = alpha_psi_A_real + beta_psi_B_real
        psi_total_imag = alpha_psi_A_imag + beta_psi_B_imag

        psi_total = torch.cat([psi_total_real, psi_total_imag], dim=1)

        pathway_info = {
            'psi_A': psi_A,
            'psi_B': psi_B,
            'r_A': r_A,
            'r_B': r_B,
            'theta_A': theta_A,
            'theta_B': theta_B,
            'alpha': torch.complex(self.alpha_real, self.alpha_imag),
            'beta': torch.complex(self.beta_real, self.beta_imag)
        }

        return psi_total, pathway_info


class QuantumEnhancedQCML(nn.Module):
    """
    Quantum-Enhanced Cognitive Market Learning model.

    Combines all quantum-inspired components for research paper:
    1. Amplitude-Phase Encoding - preserves magnitude in (r, θ) representation
    2. Multiple Observables - 4 Hermitian matrices for diverse measurements
    3. Dual-Pathway Interference - momentum vs volatility feature paths
    4. Comprehensive logging for paper visualizations

    This is designed for academic research, not necessarily trading performance.
    The goal is to explore whether quantum-inspired architectures provide
    interesting mathematical structure for financial time series.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[QuantumEnhancedConfig] = None
    ):
        super().__init__()

        if config is None:
            config = QuantumEnhancedConfig()

        self.config = config
        self.input_dim = input_dim

        if config.use_dual_pathway:
            # Use dual-pathway interference
            self.encoder = DualPathwayInterference(
                input_dim=input_dim,
                hilbert_dim=config.hilbert_dim,
                hidden_dim=config.hidden_dim,
                momentum_indices=config.momentum_features,
                volatility_indices=config.volatility_features,
                dropout=config.dropout,
                use_batch_norm=config.use_batch_norm
            )
        else:
            # Single pathway
            self.encoder = AmplitudePhaseEncoder(
                input_dim=input_dim,
                hilbert_dim=config.hilbert_dim,
                hidden_dim=config.hidden_dim,
                n_hidden_layers=config.n_hidden_layers,
                dropout=config.dropout,
                use_batch_norm=config.use_batch_norm
            )

        # Multiple observables
        self.observable = MultiObservable(
            dim=config.hilbert_dim,
            n_observables=config.n_observables
        )

        # Loss functions
        if config.ranking_type == 'ranknet':
            self.ranking_loss = RankNetLoss(sigma=1.0)
        else:
            self.ranking_loss = ListNetLoss(temperature=1.0)

        # Storage for analysis
        self._last_pathway_info = None
        self._last_expectations = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: features -> prediction.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)

        Returns
        -------
        prediction : Tensor of shape (batch,)
        """
        # Encode to Hilbert space
        if self.config.use_dual_pathway:
            psi, pathway_info = self.encoder(x)
            self._last_pathway_info = pathway_info
        else:
            psi, r, theta = self.encoder(x)
            self._last_pathway_info = {'r': r, 'theta': theta}

        # Measure with multiple observables
        prediction, expectations = self.observable(psi)
        self._last_expectations = expectations

        return prediction

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        week_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined ranking + MSE loss.
        """
        predictions = self.forward(features)

        # MSE component
        mse = F.mse_loss(predictions, labels)

        # Ranking component
        ranking = self.ranking_loss(predictions, labels, week_indices)

        # Combined loss
        total = (
            self.config.mse_weight * mse +
            self.config.ranking_weight * ranking
        )

        components = {
            "total": total.item(),
            "mse": mse.item(),
            "ranking": ranking.item()
        }

        return total, components

    def get_state_analysis(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Get detailed state analysis for paper visualizations.

        Returns
        -------
        analysis : dict with:
            - 'amplitude': Amplitude values r
            - 'phase': Phase values θ
            - 'expectations': Individual observable expectations
            - 'eigenspectra': Eigenvalues of observables
            - 'combination_weights': Observable combination weights
        """
        with torch.no_grad():
            _ = self.forward(x)

            analysis = {}

            # Pathway information
            if self._last_pathway_info is not None:
                for key, val in self._last_pathway_info.items():
                    if isinstance(val, torch.Tensor):
                        analysis[key] = val.cpu().numpy()

            # Observable expectations
            if self._last_expectations is not None:
                analysis['expectations'] = self._last_expectations.cpu().numpy()

            # Eigenspectra
            analysis['eigenspectra'] = self.observable.get_eigenspectrum()

            # Combination weights
            weights = F.softmax(self.observable.combination_weights, dim=0)
            analysis['combination_weights'] = weights.cpu().numpy()

            return analysis

    def get_interference_strength(self) -> float:
        """
        Compute interference strength metric for analysis.

        Returns the magnitude of the interference term relative to
        the individual pathway contributions.
        """
        if not self.config.use_dual_pathway:
            return 0.0

        with torch.no_grad():
            alpha = torch.complex(
                self.encoder.alpha_real,
                self.encoder.alpha_imag
            )
            beta = torch.complex(
                self.encoder.beta_real,
                self.encoder.beta_imag
            )

            # Interference strength: 2*|α||β| / (|α|² + |β|²)
            alpha_mag = torch.abs(alpha)
            beta_mag = torch.abs(beta)

            interference = 2 * alpha_mag * beta_mag / (alpha_mag**2 + beta_mag**2 + 1e-8)

            return interference.item()


def create_quantum_enhanced_model(
    input_dim: int,
    config: Optional[QuantumEnhancedConfig] = None,
    device: Optional[torch.device] = None
) -> QuantumEnhancedQCML:
    """
    Factory function to create Quantum-Enhanced QCML model.
    """
    model = QuantumEnhancedQCML(input_dim, config)

    if device is not None:
        model = model.to(device)

    return model


# ============================================================================
# Phase 5.2: Cross-Feature Attention Architecture
# ============================================================================


class CrossFeatureAttention(nn.Module):
    """
    Cross-feature attention layer (Phase 5.2).

    Learns interactions between features through multi-head attention.
    Key insight: Not all features are equally important in all market regimes.
    Attention allows the model to dynamically weight feature importance.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1):
        """
        Parameters
        ----------
        embed_dim : int
            Embedding dimension (must be divisible by n_heads)
        n_heads : int
            Number of attention heads
        dropout : float
            Dropout probability
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-feature attention with residual connection.

        Parameters
        ----------
        x : torch.Tensor (batch, n_features, embed_dim)
            Feature embeddings

        Returns
        -------
        out : torch.Tensor (batch, n_features, embed_dim)
            Attention-enhanced features
        """
        # Self-attention across features
        attn_out, _ = self.attention(x, x, x)

        # Residual connection + layer norm
        out = self.norm(x + self.dropout(attn_out))

        return out


class EnhancedSimplifiedQCML(nn.Module):
    """
    SimplifiedQCML with cross-feature attention (Phase 5.2).

    Architecture:
    1. Per-feature embedding (Linear projection)
    2. Cross-feature attention (learns feature interactions)
    3. SimplifiedEncoder (processes attended features)
    4. Predictor (final output)

    Expected improvement: +0.02-0.04 correlation
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[SimplifiedQCMLConfig] = None,
        n_attention_heads: int = 4,
        use_attention: bool = True
    ):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features
        config : SimplifiedQCMLConfig, optional
            Model configuration
        n_attention_heads : int
            Number of attention heads
        use_attention : bool
            If False, skip attention (for ablation study)
        """
        super().__init__()

        if config is None:
            config = SimplifiedQCMLConfig()

        self.config = config
        self.use_attention = use_attention
        self.input_dim = input_dim

        # Per-feature embedding
        self.feature_embed = nn.Linear(1, config.embed_dim)

        # Cross-feature attention
        if use_attention:
            self.cross_attention = CrossFeatureAttention(
                config.embed_dim,
                n_heads=n_attention_heads,
                dropout=config.dropout
            )

        # Encoder processes flattened attended features
        self.encoder = SimplifiedEncoder(
            input_dim=input_dim * config.embed_dim,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            n_hidden_layers=config.n_hidden_layers,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm
        )

        # Predictor
        self.predictor = SimplifiedPredictor(
            embed_dim=config.embed_dim,
            n_outputs=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cross-feature attention.

        Parameters
        ----------
        x : torch.Tensor (batch, n_features)
            Input features

        Returns
        -------
        y_hat : torch.Tensor (batch, 1)
            Predicted volatility
        """
        # Embed each feature separately
        # x: (batch, n_features) -> (batch, n_features, 1) -> (batch, n_features, embed_dim)
        x = x.unsqueeze(-1)  # (batch, n_features, 1)
        x = self.feature_embed(x)  # (batch, n_features, embed_dim)

        # Apply cross-feature attention
        if self.use_attention:
            x = self.cross_attention(x)  # (batch, n_features, embed_dim)

        # Flatten for encoder
        x = x.flatten(1)  # (batch, n_features * embed_dim)

        # Encode and predict
        embedding = self.encoder(x)
        prediction = self.predictor(embedding)

        return prediction

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get attention weights for interpretability.

        Returns None if attention is disabled.
        """
        if not self.use_attention:
            return None

        with torch.no_grad():
            x = x.unsqueeze(-1)
            x = self.feature_embed(x)
            _, attn_weights = self.cross_attention.attention(x, x, x)

        return attn_weights
