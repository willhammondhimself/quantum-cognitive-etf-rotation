"""
Regime-aware models for volatility prediction.

Implements Mixture of Experts (MoE) architecture where each expert
specializes in a different volatility regime.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts model."""
    input_dim: int = 59  # Number of input features
    n_experts: int = 4   # Number of expert networks
    hidden_dim: int = 64  # Hidden dimension per expert
    embed_dim: int = 32   # Embedding dimension
    n_hidden_layers: int = 2
    dropout: float = 0.15

    # Training
    lr: float = 0.001
    weight_decay: float = 1e-3
    epochs: int = 150
    batch_size: int = 64
    early_stopping_patience: int = 20

    # Loss weights
    mse_weight: float = 0.7
    diversity_weight: float = 0.1
    load_balance_weight: float = 0.1
    ranking_weight: float = 0.1


class ExpertNetwork(nn.Module):
    """
    Single expert network for one volatility regime.

    Each expert learns regime-specific patterns.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        n_hidden_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Embedding layer
        layers.append(nn.Linear(hidden_dim, embed_dim))

        self.encoder = nn.Sequential(*layers)

        # Prediction head
        self.predictor = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        embedding : torch.Tensor
            Expert embedding (batch, embed_dim)
        prediction : torch.Tensor
            Expert prediction (batch, 1)
        """
        embedding = self.encoder(x)
        prediction = self.predictor(embedding)
        return embedding, prediction


class GatingNetwork(nn.Module):
    """
    Gating network that routes inputs to appropriate experts.

    Learns to recognize regime patterns and assign expert weights.
    """

    def __init__(
        self,
        input_dim: int,
        n_experts: int = 4,
        hidden_dim: int = 32,
        temperature: float = 1.0
    ):
        super().__init__()

        self.temperature = temperature

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute expert weights.

        Returns
        -------
        weights : torch.Tensor
            Softmax weights for each expert (batch, n_experts)
        """
        logits = self.gate(x) / self.temperature
        return F.softmax(logits, dim=-1)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts model for regime-aware volatility prediction.

    Architecture:
    - Multiple expert networks, each specializing in a regime
    - Gating network that routes inputs to appropriate experts
    - Weighted combination of expert outputs

    The gating network learns to recognize regime patterns and
    route inputs to the appropriate expert.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        # Create experts
        self.experts = nn.ModuleList([
            ExpertNetwork(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                embed_dim=config.embed_dim,
                n_hidden_layers=config.n_hidden_layers,
                dropout=config.dropout
            )
            for _ in range(config.n_experts)
        ])

        # Gating network
        self.gate = GatingNetwork(
            input_dim=config.input_dim,
            n_experts=config.n_experts,
            hidden_dim=config.hidden_dim // 2
        )

    def forward(
        self,
        x: torch.Tensor,
        return_expert_outputs: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through MoE.

        Parameters
        ----------
        x : torch.Tensor
            Input features (batch, input_dim)
        return_expert_outputs : bool
            If True, also return individual expert outputs

        Returns
        -------
        prediction : torch.Tensor
            Combined prediction (batch, 1)
        expert_outputs : torch.Tensor (optional)
            Individual expert outputs (batch, n_experts)
        gate_weights : torch.Tensor (optional)
            Gating weights (batch, n_experts)
        """
        batch_size = x.shape[0]

        # Get gating weights
        gate_weights = self.gate(x)  # (batch, n_experts)

        # Get expert outputs
        expert_predictions = []
        expert_embeddings = []

        for expert in self.experts:
            embed, pred = expert(x)
            expert_embeddings.append(embed)
            expert_predictions.append(pred)

        # Stack expert predictions
        expert_outputs = torch.cat(expert_predictions, dim=1)  # (batch, n_experts)

        # Weighted combination
        prediction = (expert_outputs * gate_weights).sum(dim=1, keepdim=True)

        if return_expert_outputs:
            return prediction, expert_outputs, gate_weights
        return prediction

    def get_expert_usage(self, x: torch.Tensor) -> torch.Tensor:
        """Get average expert usage (for monitoring load balance)."""
        gate_weights = self.gate(x)
        return gate_weights.mean(dim=0)


class MoELoss(nn.Module):
    """
    Combined loss for Mixture of Experts training.

    Components:
    - MSE: Standard regression loss
    - Diversity: Encourages different experts to make different predictions
    - Load Balance: Encourages balanced expert usage
    - Ranking: Pairwise ranking loss for ordinal prediction
    """

    def __init__(
        self,
        mse_weight: float = 0.7,
        diversity_weight: float = 0.1,
        load_balance_weight: float = 0.1,
        ranking_weight: float = 0.1
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.diversity_weight = diversity_weight
        self.load_balance_weight = load_balance_weight
        self.ranking_weight = ranking_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        expert_outputs: torch.Tensor,
        gate_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Parameters
        ----------
        prediction : torch.Tensor
            Combined prediction (batch, 1)
        target : torch.Tensor
            Target values (batch, 1)
        expert_outputs : torch.Tensor
            Individual expert outputs (batch, n_experts)
        gate_weights : torch.Tensor
            Gating weights (batch, n_experts)

        Returns
        -------
        loss : torch.Tensor
            Combined loss
        components : dict
            Individual loss components for logging
        """
        # MSE loss
        mse_loss = F.mse_loss(prediction, target)

        # Diversity loss (encourage different expert predictions)
        # Computed as negative of average pairwise variance
        n_experts = expert_outputs.shape[1]
        diversity = 0
        for i in range(n_experts):
            for j in range(i + 1, n_experts):
                diversity += (expert_outputs[:, i] - expert_outputs[:, j]).pow(2).mean()
        diversity_loss = -diversity / (n_experts * (n_experts - 1) / 2)

        # Load balance loss (encourage balanced expert usage)
        # Penalize deviation from uniform usage
        expert_usage = gate_weights.mean(dim=0)
        uniform = torch.ones_like(expert_usage) / n_experts
        load_balance_loss = F.kl_div(
            expert_usage.log(),
            uniform,
            reduction='batchmean'
        )

        # Ranking loss (pairwise)
        ranking_loss = self._ranking_loss(prediction, target)

        # Combine losses
        total_loss = (
            self.mse_weight * mse_loss +
            self.diversity_weight * diversity_loss +
            self.load_balance_weight * load_balance_loss +
            self.ranking_weight * ranking_loss
        )

        components = {
            'mse': mse_loss.item(),
            'diversity': diversity_loss.item(),
            'load_balance': load_balance_loss.item(),
            'ranking': ranking_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, components

    def _ranking_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        margin: float = 0.01,
        n_pairs: int = 10
    ) -> torch.Tensor:
        """Pairwise ranking loss."""
        batch_size = prediction.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=prediction.device)

        # Sample random pairs
        n_pairs = min(n_pairs, batch_size * (batch_size - 1) // 2)
        idx1 = torch.randint(0, batch_size, (n_pairs,), device=prediction.device)
        idx2 = torch.randint(0, batch_size, (n_pairs,), device=prediction.device)

        # Avoid same indices
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]

        if len(idx1) == 0:
            return torch.tensor(0.0, device=prediction.device)

        # Compute pairwise losses
        pred_diff = prediction[idx1] - prediction[idx2]
        target_diff = target[idx1] - target[idx2]
        target_sign = torch.sign(target_diff)

        # Hinge loss
        loss = F.relu(margin - target_sign * pred_diff)
        return loss.mean()


class RegimeConditionedEncoder(nn.Module):
    """
    Simple regime-conditioned encoder as an alternative to MoE.

    Uses regime features to scale/shift internal representations.
    Less complex than MoE but still regime-aware.
    """

    def __init__(
        self,
        input_dim: int,
        n_regimes: int = 4,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_regimes = n_regimes

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )

        # Regime-specific scale and shift (FiLM-style conditioning)
        self.regime_scale = nn.Embedding(n_regimes, embed_dim)
        self.regime_shift = nn.Embedding(n_regimes, embed_dim)

        # Initialize scale to 1, shift to 0
        nn.init.ones_(self.regime_scale.weight)
        nn.init.zeros_(self.regime_shift.weight)

        # Predictor
        self.predictor = nn.Linear(embed_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        regime: Optional[torch.Tensor] = None,
        regime_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features (batch, input_dim)
        regime : torch.Tensor, optional
            Hard regime labels (batch,)
        regime_probs : torch.Tensor, optional
            Soft regime probabilities (batch, n_regimes)

        Returns
        -------
        prediction : torch.Tensor
            Volatility prediction (batch, 1)
        """
        # Encode
        embed = self.encoder(x)

        # Apply regime conditioning
        if regime is not None:
            # Hard conditioning
            scale = self.regime_scale(regime)
            shift = self.regime_shift(regime)
        elif regime_probs is not None:
            # Soft conditioning (weighted sum)
            scale = (regime_probs.unsqueeze(-1) *
                     self.regime_scale.weight.unsqueeze(0)).sum(dim=1)
            shift = (regime_probs.unsqueeze(-1) *
                     self.regime_shift.weight.unsqueeze(0)).sum(dim=1)
        else:
            # No conditioning (default to regime 1 = normal)
            scale = torch.ones_like(embed)
            shift = torch.zeros_like(embed)

        # Apply FiLM: out = scale * embed + shift
        conditioned = scale * embed + shift

        # Predict
        return self.predictor(conditioned)


class MoETrainer:
    """Trainer for Mixture of Experts model."""

    def __init__(
        self,
        config: MoEConfig,
        device: str = 'cpu'
    ):
        self.config = config
        self.device = device
        self.model = MixtureOfExperts(config).to(device)
        self.criterion = MoELoss(
            mse_weight=config.mse_weight,
            diversity_weight=config.diversity_weight,
            load_balance_weight=config.load_balance_weight,
            ranking_weight=config.ranking_weight
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_components = {'mse': 0, 'diversity': 0, 'load_balance': 0, 'ranking': 0}
        n_batches = 0

        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            pred, expert_outputs, gate_weights = self.model(X, return_expert_outputs=True)

            # Compute loss
            loss, components = self.criterion(pred, y, expert_outputs, gate_weights)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in components.items():
                if k in total_components:
                    total_components[k] += v
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in total_components.items()}

        return avg_loss, avg_components

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)

                pred, expert_outputs, gate_weights = self.model(X, return_expert_outputs=True)
                loss, _ = self.criterion(pred, y, expert_outputs, gate_weights)

                total_loss += loss.item()
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())

        avg_loss = total_loss / len(val_loader)

        # Compute correlation
        preds = torch.cat(all_preds).numpy().ravel()
        targets = torch.cat(all_targets).numpy().ravel()
        correlation = np.corrcoef(preds, targets)[0, 1]

        return avg_loss, correlation

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        verbose: bool = True
    ) -> Dict[str, list]:
        """Full training loop."""
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_corr': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Train
            train_loss, components = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)

            # Validate
            val_loss, val_corr = self.validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_corr'].append(val_corr)

            # LR scheduling
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict()
            else:
                patience_counter += 1

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, val_corr={val_corr:.4f}")

            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        self.model.load_state_dict(best_state)

        return history

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            pred = self.model(X)
        return pred.cpu().numpy().ravel()

    def get_expert_usage(self, X: torch.Tensor) -> np.ndarray:
        """Get expert usage statistics."""
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            usage = self.model.get_expert_usage(X)
        return usage.cpu().numpy()
