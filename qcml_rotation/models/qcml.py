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
