"""
Autoencoder + Linear Head baseline model.

Learns a compressed representation via reconstruction, then uses the
bottleneck features for prediction with a simple linear model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class AutoencoderConfig:
    """Configuration for Autoencoder model."""
    hidden_dim: int = 32
    bottleneck_dim: int = 8
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 64
    early_stopping_patience: int = 10


class Autoencoder(nn.Module):
    """
    Simple autoencoder for feature compression.

    Architecture: input -> hidden -> bottleneck -> hidden -> output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        bottleneck_dim: int = 8
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns both reconstruction and bottleneck representation.
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get bottleneck representation only."""
        return self.encoder(x)


class AutoencoderPredictor:
    """
    Complete pipeline: Autoencoder + Linear predictor.

    Training happens in two stages:
    1. Train autoencoder on reconstruction loss
    2. Freeze encoder, train linear head on prediction task
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[AutoencoderConfig] = None,
        device: Optional[torch.device] = None
    ):
        if config is None:
            config = AutoencoderConfig()

        self.config = config
        self.device = device or torch.device("cpu")

        # Build models
        self.autoencoder = Autoencoder(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            bottleneck_dim=config.bottleneck_dim
        ).to(self.device)

        self.linear_head = nn.Linear(config.bottleneck_dim, 1).to(self.device)

        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, List[float]]:
        """
        Train autoencoder then linear head.

        Returns training history.
        """
        config = self.config

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        if X_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Stage 1: Train autoencoder
        print("Stage 1: Training autoencoder...")
        ae_history = self._train_autoencoder(X_train_t, X_val_t if X_val is not None else None)

        # Stage 2: Train linear head
        print("Stage 2: Training linear head...")
        pred_history = self._train_linear_head(X_train_t, y_train_t, X_val_t, y_val_t)

        self.is_fitted = True

        return {"autoencoder": ae_history, "predictor": pred_history}

    def _train_autoencoder(
        self,
        X_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None
    ) -> Dict[str, List[float]]:
        """Train autoencoder with reconstruction loss."""
        config = self.config

        dataset = TensorDataset(X_train)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        optimizer = optim.Adam(self.autoencoder.parameters(), lr=config.lr)
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(config.epochs):
            # Training
            self.autoencoder.train()
            train_loss = 0.0

            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()
                x_recon, _ = self.autoencoder(batch_x)
                loss = criterion(x_recon, batch_x)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_x)

            train_loss /= len(X_train)
            history["train_loss"].append(train_loss)

            # Validation
            val_loss = None
            if X_val is not None:
                self.autoencoder.eval()
                with torch.no_grad():
                    x_recon, _ = self.autoencoder(X_val)
                    val_loss = criterion(x_recon, X_val).item()
                    history["val_loss"].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break

            if (epoch + 1) % 20 == 0:
                msg = f"  Epoch {epoch+1}: train_loss={train_loss:.6f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.6f}"
                print(msg)

        return history

    def _train_linear_head(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None
    ) -> Dict[str, List[float]]:
        """Train linear head on frozen encoder output."""
        config = self.config

        # Freeze autoencoder
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        # Get encoded features
        with torch.no_grad():
            z_train = self.autoencoder.encode(X_train.to(self.device))
            if X_val is not None:
                z_val = self.autoencoder.encode(X_val)

        dataset = TensorDataset(z_train, y_train.to(self.device))
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        optimizer = optim.Adam(self.linear_head.parameters(), lr=config.lr)
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(config.epochs):
            # Training
            self.linear_head.train()
            train_loss = 0.0

            for batch_z, batch_y in loader:
                optimizer.zero_grad()
                pred = self.linear_head(batch_z)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_z)

            train_loss /= len(z_train)
            history["train_loss"].append(train_loss)

            # Validation
            val_loss = None
            if X_val is not None and y_val is not None:
                self.linear_head.eval()
                with torch.no_grad():
                    pred = self.linear_head(z_val)
                    val_loss = criterion(pred, y_val).item()
                    history["val_loss"].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break

            if (epoch + 1) % 20 == 0:
                msg = f"  Epoch {epoch+1}: train_loss={train_loss:.6f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.6f}"
                print(msg)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict excess returns."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.autoencoder.eval()
        self.linear_head.eval()

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            z = self.autoencoder.encode(X_t)
            pred = self.linear_head(z)

        return pred.cpu().numpy().squeeze()

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Get bottleneck embeddings for input features."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.autoencoder.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            z = self.autoencoder.encode(X_t)

        return z.cpu().numpy()
