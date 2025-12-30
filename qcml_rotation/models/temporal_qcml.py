"""
Temporal QCML model for volatility prediction.

Uses GRU + self-attention to capture volatility clustering and
temporal dependencies. Outputs multi-horizon predictions with
uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List


@dataclass
class TemporalQCMLConfig:
    """Configuration for Temporal QCML model."""
    input_dim: int = 59       # Number of input features
    seq_len: int = 12         # Sequence length (weeks of history)
    hidden_dim: int = 64      # GRU hidden dimension
    embed_dim: int = 32       # Embedding dimension
    n_gru_layers: int = 2     # Number of GRU layers
    n_attention_heads: int = 4  # Number of attention heads
    dropout: float = 0.15

    # Multi-horizon prediction
    horizons: Tuple[int, ...] = (5, 10, 20)  # Prediction horizons (days)
    horizon_weights: Tuple[float, ...] = (0.5, 0.25, 0.25)  # Loss weights

    # Uncertainty quantification
    output_uncertainty: bool = True  # Output mean + variance

    # Loss settings
    ranking_weight: float = 0.2

    # Training
    lr: float = 0.001
    weight_decay: float = 1e-3
    epochs: int = 150
    batch_size: int = 32
    early_stopping_patience: int = 20


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.

    Adds position information to embeddings using sin/cos functions.
    """

    def __init__(self, embed_dim: int, max_len: int = 100):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, embed_dim)

        Returns
        -------
        torch.Tensor
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class TemporalEncoder(nn.Module):
    """
    Temporal encoder using GRU + positional encoding.

    Captures volatility clustering and mean-reversion patterns.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # Project input features
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

        # GRU (unidirectional for causal modeling)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode temporal sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input features (batch, seq_len, input_dim)

        Returns
        -------
        output : torch.Tensor
            GRU outputs (batch, seq_len, hidden_dim)
        hidden : torch.Tensor
            Final hidden state (n_layers, batch, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project each timestep
        # Reshape for BatchNorm: (batch * seq_len, input_dim)
        x_flat = x.reshape(-1, x.shape[-1])
        x_proj = self.input_proj(x_flat)
        x_proj = x_proj.reshape(batch_size, seq_len, -1)

        # Add positional encoding
        x_proj = self.pos_encoding(x_proj)

        # GRU
        output, hidden = self.gru(x_proj)

        return output, hidden


class TemporalSelfAttention(nn.Module):
    """
    Self-attention over temporal sequence.

    Learns which past timesteps are most relevant for current prediction.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, embed_dim)
        return_weights : bool
            Whether to return attention weights

        Returns
        -------
        output : torch.Tensor
            Attention output (batch, seq_len, embed_dim)
        weights : torch.Tensor, optional
            Attention weights (batch, seq_len, seq_len)
        """
        # Self-attention with residual connection
        attn_out, weights = self.attention(x, x, x, need_weights=return_weights)
        output = self.norm(x + self.dropout(attn_out))

        if return_weights:
            return output, weights
        return output, None


class GaussianHead(nn.Module):
    """
    Prediction head that outputs mean and variance.

    For uncertainty quantification in predictions.
    """

    def __init__(self, input_dim: int, min_variance: float = 1e-6):
        super().__init__()
        self.min_variance = min_variance

        self.mean_head = nn.Linear(input_dim, 1)
        self.logvar_head = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and variance.

        Parameters
        ----------
        x : torch.Tensor
            Input features (batch, input_dim)

        Returns
        -------
        mean : torch.Tensor
            Predicted mean (batch, 1)
        variance : torch.Tensor
            Predicted variance (batch, 1)
        """
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        variance = F.softplus(logvar) + self.min_variance
        return mean, variance


class MultiHorizonHead(nn.Module):
    """
    Multi-horizon prediction head.

    Predicts volatility at multiple horizons (5d, 10d, 20d).
    """

    def __init__(
        self,
        input_dim: int,
        horizons: Tuple[int, ...] = (5, 10, 20),
        output_uncertainty: bool = True
    ):
        super().__init__()
        self.horizons = horizons
        self.output_uncertainty = output_uncertainty

        # Create head for each horizon
        self.heads = nn.ModuleDict()
        for h in horizons:
            if output_uncertainty:
                self.heads[f"h{h}"] = GaussianHead(input_dim)
            else:
                self.heads[f"h{h}"] = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Predict at all horizons.

        Returns
        -------
        predictions : dict
            {horizon: (mean, variance)} for each horizon
        """
        predictions = {}
        for h in self.horizons:
            if self.output_uncertainty:
                mean, var = self.heads[f"h{h}"](x)
                predictions[h] = (mean, var)
            else:
                pred = self.heads[f"h{h}"](x)
                predictions[h] = (pred, None)
        return predictions


class TemporalQCML(nn.Module):
    """
    Temporal QCML model for volatility prediction.

    Architecture:
    1. Feature projection per timestep
    2. Positional encoding
    3. GRU for temporal modeling
    4. Self-attention for learning relevant past
    5. Multi-horizon prediction heads with uncertainty
    """

    def __init__(self, config: TemporalQCMLConfig):
        super().__init__()
        self.config = config

        # Temporal encoder (GRU)
        self.temporal_encoder = TemporalEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_gru_layers,
            dropout=config.dropout
        )

        # Self-attention
        self.attention = TemporalSelfAttention(
            embed_dim=config.hidden_dim,
            n_heads=config.n_attention_heads,
            dropout=config.dropout
        )

        # Projection to embedding
        self.embed_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # Multi-horizon prediction heads
        self.prediction_head = MultiHorizonHead(
            input_dim=config.embed_dim,
            horizons=config.horizons,
            output_uncertainty=config.output_uncertainty
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence (batch, seq_len, input_dim)
        return_attention : bool
            Whether to return attention weights

        Returns
        -------
        predictions : dict
            {horizon: (mean, variance)} for each horizon
        attention_weights : torch.Tensor, optional
            Attention weights if requested
        """
        # Temporal encoding
        gru_output, _ = self.temporal_encoder(x)  # (batch, seq_len, hidden_dim)

        # Self-attention
        attn_output, attn_weights = self.attention(gru_output, return_weights=return_attention)

        # Take last timestep for prediction
        last_hidden = attn_output[:, -1, :]  # (batch, hidden_dim)

        # Project to embedding
        embed = self.embed_proj(last_hidden)  # (batch, embed_dim)

        # Multi-horizon predictions
        predictions = self.prediction_head(embed)

        if return_attention:
            return predictions, attn_weights
        return predictions

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get temporal embedding."""
        gru_output, _ = self.temporal_encoder(x)
        attn_output, _ = self.attention(gru_output)
        last_hidden = attn_output[:, -1, :]
        embed = self.embed_proj(last_hidden)
        return embed


class GaussianNLLLoss(nn.Module):
    """
    Gaussian negative log-likelihood loss.

    For training models with uncertainty output.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL.

        Parameters
        ----------
        mean : torch.Tensor
            Predicted mean
        variance : torch.Tensor
            Predicted variance
        target : torch.Tensor
            Target values

        Returns
        -------
        loss : torch.Tensor
            Gaussian NLL loss
        """
        # NLL = 0.5 * (log(variance) + (target - mean)^2 / variance)
        loss = 0.5 * (torch.log(variance) + (target - mean).pow(2) / variance)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class MultiHorizonLoss(nn.Module):
    """
    Combined loss for multi-horizon prediction.

    Combines Gaussian NLL across horizons with ranking loss.
    """

    def __init__(
        self,
        horizons: Tuple[int, ...] = (5, 10, 20),
        horizon_weights: Tuple[float, ...] = (0.5, 0.25, 0.25),
        ranking_weight: float = 0.2,
        use_uncertainty: bool = True
    ):
        super().__init__()
        self.horizons = horizons
        self.horizon_weights = horizon_weights
        self.ranking_weight = ranking_weight
        self.use_uncertainty = use_uncertainty

        self.nll = GaussianNLLLoss()

    def forward(
        self,
        predictions: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]],
        targets: Dict[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-horizon loss.

        Parameters
        ----------
        predictions : dict
            {horizon: (mean, variance)} for each horizon
        targets : dict
            {horizon: target} for each horizon

        Returns
        -------
        loss : torch.Tensor
            Combined loss
        components : dict
            Individual loss components
        """
        total_loss = 0
        components = {}

        # Horizon-specific losses
        for i, h in enumerate(self.horizons):
            mean, variance = predictions[h]
            target = targets[h]

            if self.use_uncertainty and variance is not None:
                h_loss = self.nll(mean, variance, target)
            else:
                h_loss = F.mse_loss(mean, target)

            weighted_loss = self.horizon_weights[i] * h_loss
            total_loss = total_loss + weighted_loss
            components[f'loss_h{h}'] = h_loss.item()

        # Ranking loss (on primary horizon)
        if self.ranking_weight > 0:
            primary_mean = predictions[self.horizons[0]][0]
            primary_target = targets[self.horizons[0]]
            ranking_loss = self._ranking_loss(primary_mean, primary_target)
            total_loss = total_loss + self.ranking_weight * ranking_loss
            components['ranking'] = ranking_loss.item()

        components['total'] = total_loss.item()
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

        n_pairs = min(n_pairs, batch_size * (batch_size - 1) // 2)
        idx1 = torch.randint(0, batch_size, (n_pairs,), device=prediction.device)
        idx2 = torch.randint(0, batch_size, (n_pairs,), device=prediction.device)

        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]

        if len(idx1) == 0:
            return torch.tensor(0.0, device=prediction.device)

        pred_diff = prediction[idx1] - prediction[idx2]
        target_diff = target[idx1] - target[idx2]
        target_sign = torch.sign(target_diff)

        loss = F.relu(margin - target_sign * pred_diff)
        return loss.mean()


class TemporalQCMLTrainer:
    """Trainer for Temporal QCML model."""

    def __init__(
        self,
        config: TemporalQCMLConfig,
        device: str = 'cpu'
    ):
        self.config = config
        self.device = device
        self.model = TemporalQCML(config).to(device)
        self.criterion = MultiHorizonLoss(
            horizons=config.horizons,
            horizon_weights=config.horizon_weights,
            ranking_weight=config.ranking_weight,
            use_uncertainty=config.output_uncertainty
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
            patience=10
        )

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            X = batch['features'].to(self.device)
            targets = {h: batch[f'target_h{h}'].to(self.device) for h in self.config.horizons}

            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(X)

            # Compute loss
            loss, components = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        return avg_loss, components

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
            for batch in val_loader:
                X = batch['features'].to(self.device)
                targets = {h: batch[f'target_h{h}'].to(self.device) for h in self.config.horizons}

                predictions = self.model(X)
                loss, _ = self.criterion(predictions, targets)

                total_loss += loss.item()

                # Collect primary horizon predictions
                h0 = self.config.horizons[0]
                all_preds.append(predictions[h0][0].cpu())
                all_targets.append(targets[h0].cpu())

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
        best_state = None

        for epoch in range(self.config.epochs):
            # Train
            train_loss, _ = self.train_epoch(train_loader)
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
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def predict(
        self,
        X: torch.Tensor,
        horizon: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions.

        Parameters
        ----------
        X : torch.Tensor
            Input sequences (batch, seq_len, input_dim)
        horizon : int, optional
            Specific horizon to return. Default is primary horizon.

        Returns
        -------
        mean : np.ndarray
            Predicted means
        std : np.ndarray, optional
            Predicted standard deviations (if uncertainty enabled)
        """
        if horizon is None:
            horizon = self.config.horizons[0]

        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            predictions = self.model(X)
            mean, variance = predictions[horizon]

        mean_np = mean.cpu().numpy().ravel()

        if variance is not None:
            std_np = torch.sqrt(variance).cpu().numpy().ravel()
            return mean_np, std_np

        return mean_np, None
