"""
Simple MLP baseline model.

Direct prediction from features to excess returns using a 2-layer
feedforward network with dropout regularization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class MLPConfig:
    """Configuration for MLP model."""
    hidden_dims: List[int] = None
    dropout: float = 0.2
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 64
    early_stopping_patience: int = 15
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


class MLP(nn.Module):
    """
    Multi-layer perceptron for regression.

    Architecture: input -> hidden1 -> ReLU -> dropout -> hidden2 -> ReLU -> dropout -> output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.2
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPPredictor:
    """
    MLP wrapper with training utilities.

    Handles training loop, validation, early stopping, and LR scheduling.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[MLPConfig] = None,
        device: Optional[torch.device] = None
    ):
        if config is None:
            config = MLPConfig()

        self.config = config
        self.device = device or torch.device("cpu")

        self.model = MLP(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout
        ).to(self.device)

        self.is_fitted = False
        self.best_state = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, List[float]]:
        """
        Train MLP model.

        Parameters
        ----------
        X_train : array of shape (n_samples, n_features)
        y_train : array of shape (n_samples,)
        X_val : array, optional
        y_val : array, optional

        Returns
        -------
        history : dict with training and validation loss per epoch
        """
        config = self.config

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience
        )
        criterion = nn.MSELoss()

        if X_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(config.epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_x)

            train_loss /= len(X_train_t)
            history["train_loss"].append(train_loss)

            # Validation
            val_loss = None
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(X_val_t)
                    val_loss = criterion(pred, y_val_t).item()
                    history["val_loss"].append(val_loss)

                scheduler.step(val_loss)

                # Early stopping and checkpointing
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            if (epoch + 1) % 20 == 0:
                lr = optimizer.param_groups[0]['lr']
                msg = f"Epoch {epoch+1}: train_loss={train_loss:.6f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.6f}"
                msg += f", lr={lr:.6f}"
                print(msg)

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        self.is_fitted = True

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict excess returns."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred = self.model(X_t)

        return pred.cpu().numpy().squeeze()

    def save(self, path: str) -> None:
        """Save model state."""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config
        }, path)

    def load(self, path: str) -> None:
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.is_fitted = True
