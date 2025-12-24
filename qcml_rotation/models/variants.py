"""
Model variants for testing different prediction targets and loss functions.

Includes:
1. Ranking-focused models (ListNet-style, pairwise)
2. Classification models (top-k classification)
3. Ensemble methods
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class ModelVariantConfig:
    """Configuration for model variants."""
    input_dim: int
    hidden_dims: List[int] = None
    dropout: float = 0.2
    # Ranking loss settings
    use_ranking_loss: bool = True
    ranking_weight: float = 0.5
    # Classification settings
    n_classes: int = 3  # Bottom, middle, top tercile
    # Ensemble settings
    n_models: int = 5

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


class RankingMLP(nn.Module):
    """
    MLP with combined regression and ranking loss.

    Uses ListNet-style ranking loss combined with MSE.
    """

    def __init__(self, config: ModelVariantConfig):
        super().__init__()
        self.config = config

        layers = []
        prev_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        group_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined MSE and ranking loss.

        Parameters
        ----------
        predictions : tensor of shape (n_samples,)
        targets : tensor of shape (n_samples,)
        group_indices : tensor mapping samples to groups (dates)
            If provided, ranking loss is computed within groups
        """
        # MSE loss
        mse_loss = F.mse_loss(predictions, targets)

        if not self.config.use_ranking_loss:
            return mse_loss

        # Ranking loss (ListNet-style cross-entropy on softmax)
        if group_indices is None:
            # Treat all samples as one group
            pred_probs = F.softmax(predictions, dim=0)
            target_probs = F.softmax(targets, dim=0)
            ranking_loss = -torch.sum(target_probs * torch.log(pred_probs + 1e-10))
        else:
            # Compute ranking loss per group
            unique_groups = torch.unique(group_indices)
            group_losses = []

            for group in unique_groups:
                mask = group_indices == group
                if mask.sum() < 2:
                    continue

                pred_group = predictions[mask]
                target_group = targets[mask]

                pred_probs = F.softmax(pred_group, dim=0)
                target_probs = F.softmax(target_group, dim=0)

                group_loss = -torch.sum(target_probs * torch.log(pred_probs + 1e-10))
                group_losses.append(group_loss)

            if group_losses:
                ranking_loss = torch.stack(group_losses).mean()
            else:
                ranking_loss = torch.tensor(0.0, device=predictions.device)

        # Combine losses
        total_loss = (1 - self.config.ranking_weight) * mse_loss + \
                     self.config.ranking_weight * ranking_loss

        return total_loss


class PairwiseRankingMLP(nn.Module):
    """
    MLP trained with pairwise ranking loss (similar to LambdaRank).

    For each pair (i, j), if y_i > y_j, we want pred_i > pred_j.
    """

    def __init__(self, config: ModelVariantConfig):
        super().__init__()
        self.config = config

        layers = []
        prev_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_pairs: int = 100
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss.

        Samples random pairs and computes cross-entropy on their ordering.
        """
        n = len(predictions)
        if n < 2:
            return F.mse_loss(predictions, targets)

        # Sample random pairs
        n_pairs = min(n_pairs, n * (n - 1) // 2)
        idx1 = torch.randint(0, n, (n_pairs,), device=predictions.device)
        idx2 = torch.randint(0, n, (n_pairs,), device=predictions.device)

        # Ensure different indices
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]

        if len(idx1) == 0:
            return F.mse_loss(predictions, targets)

        # Pairwise differences
        pred_diff = predictions[idx1] - predictions[idx2]
        target_diff = targets[idx1] - targets[idx2]

        # Binary labels: 1 if target_i > target_j, else 0
        labels = (target_diff > 0).float()

        # Binary cross-entropy on sigmoid of prediction difference
        pairwise_loss = F.binary_cross_entropy_with_logits(pred_diff, labels)

        # Combine with small MSE component for scale
        mse_loss = F.mse_loss(predictions, targets)

        return 0.8 * pairwise_loss + 0.2 * mse_loss


class TopKClassifier(nn.Module):
    """
    Classifies ETFs into top-k, middle, or bottom-k.

    Uses ordinal regression for better ranking.
    """

    def __init__(self, config: ModelVariantConfig):
        super().__init__()
        self.config = config
        self.n_classes = config.n_classes

        layers = []
        prev_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits."""
        features = self.feature_extractor(x)
        return self.classifier(features)

    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Returns scores for ranking (probability of top class)."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        # Score = P(top) - P(bottom) for ranking
        return probs[:, -1] - probs[:, 0]

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        group_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute classification loss.

        Converts continuous targets to class labels based on quantiles.
        """
        # Convert targets to class labels within each group
        if group_indices is None:
            labels = self._to_class_labels(targets)
        else:
            labels = torch.zeros_like(targets, dtype=torch.long)
            for group in torch.unique(group_indices):
                mask = group_indices == group
                labels[mask] = self._to_class_labels(targets[mask])

        return F.cross_entropy(predictions, labels)

    def _to_class_labels(self, values: torch.Tensor) -> torch.Tensor:
        """Convert continuous values to class labels using quantiles."""
        n = len(values)
        ranks = values.argsort().argsort()  # Get ranks 0 to n-1

        labels = torch.zeros(n, dtype=torch.long, device=values.device)

        if self.n_classes == 3:
            # Bottom third, middle third, top third
            labels[ranks < n // 3] = 0
            labels[(ranks >= n // 3) & (ranks < 2 * n // 3)] = 1
            labels[ranks >= 2 * n // 3] = 2
        elif self.n_classes == 5:
            # Quintiles
            for i in range(5):
                labels[(ranks >= i * n // 5) & (ranks < (i + 1) * n // 5)] = i

        return labels


class EnsembleModel:
    """
    Ensemble of models with different random initializations.

    Averages predictions for more robust output.
    """

    def __init__(
        self,
        model_class: type,
        config: ModelVariantConfig,
        n_models: int = 5
    ):
        self.config = config
        self.n_models = n_models
        self.models = []

        for i in range(n_models):
            torch.manual_seed(42 + i)
            model = model_class(config)
            self.models.append(model)

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self

    def train(self, mode: bool = True):
        for model in self.models:
            model.train(mode)

    def eval(self):
        self.train(False)

    def parameters(self):
        """Yield all parameters from all models."""
        for model in self.models:
            yield from model.parameters()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Average predictions across ensemble."""
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        return torch.stack(predictions).mean(dim=0)

    def compute_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute average loss across ensemble."""
        losses = []
        for model in self.models:
            pred = model(x)
            if hasattr(model, 'compute_loss'):
                loss = model.compute_loss(pred, targets, **kwargs)
            else:
                loss = F.mse_loss(pred, targets)
            losses.append(loss)

        return torch.stack(losses).mean()


def train_model_variant(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    date_indices: Optional[np.ndarray] = None,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 64,
    early_stopping_patience: int = 10,
    device: str = 'cpu'
) -> Dict:
    """
    Train a model variant with early stopping.

    Returns training history and best model state.
    """
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)

    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)

    if date_indices is not None:
        date_indices_t = torch.LongTensor(date_indices).to(device)
    else:
        date_indices_t = None

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    n_samples = len(X_train)

    for epoch in range(epochs):
        model.train()

        # Shuffle data
        perm = torch.randperm(n_samples)

        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i:i + batch_size]

            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()

            if hasattr(model, 'compute_loss'):
                preds = model(X_batch)
                if date_indices_t is not None:
                    loss = model.compute_loss(preds, y_batch, date_indices_t[batch_idx])
                else:
                    loss = model.compute_loss(preds, y_batch)
            else:
                preds = model(X_batch)
                loss = F.mse_loss(preds, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        history['train_loss'].append(avg_train_loss)

        # Validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t)
                if hasattr(model, 'compute_loss'):
                    val_loss = model.compute_loss(val_preds, y_val_t).item()
                else:
                    val_loss = F.mse_loss(val_preds, y_val_t).item()

            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'history': history,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1
    }
