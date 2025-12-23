"""
Training utilities for PyTorch models.

Includes generic trainer, early stopping, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
import copy


@dataclass
class TrainerConfig:
    """Configuration for training."""
    epochs: int = 100
    lr: float = 0.001
    batch_size: int = 64
    early_stopping_patience: int = 15
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    clip_grad_norm: Optional[float] = 1.0
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True


class EarlyStopping:
    """
    Early stopping handler.

    Tracks validation loss and signals when to stop training.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Parameters
        ----------
        patience : int
            Number of epochs to wait for improvement.
        min_delta : float
            Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Parameters
        ----------
        val_loss : current validation loss

        Returns
        -------
        should_stop : bool
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False


class Trainer:
    """
    Generic PyTorch trainer.

    Handles training loop, validation, early stopping, LR scheduling,
    gradient clipping, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        config: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Parameters
        ----------
        model : nn.Module
            Model to train.
        criterion : nn.Module
            Loss function.
        config : TrainerConfig
            Training configuration.
        device : torch.device
            Device to train on.
        """
        if config is None:
            config = TrainerConfig()

        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = device or torch.device("cpu")

        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience
        )
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)

        self.best_model_state = None
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            # Unpack batch - expecting (features, labels, ...)
            features, labels = batch[0], batch[1]
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(features)
            if outputs.dim() > 1:
                outputs = outputs.squeeze(-1)

            loss = self.criterion(outputs, labels)
            loss.backward()

            if self.config.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_grad_norm
                )

            self.optimizer.step()

            total_loss += loss.item() * len(features)
            n_samples += len(features)

        return total_loss / n_samples

    def validate(self, val_loader: DataLoader) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                features, labels = batch[0], batch[1]
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                if outputs.dim() > 1:
                    outputs = outputs.squeeze(-1)

                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * len(features)
                n_samples += len(features)

        return total_loss / n_samples

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader, optional
        verbose : bool
            Print progress.

        Returns
        -------
        history : dict with train_loss and val_loss lists
        """
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)

                self.scheduler.step(val_loss)

                # Check early stopping
                if self.early_stopping(val_loss):
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

                # Save best model
                if val_loss <= self.early_stopping.best_loss:
                    self.best_model_state = copy.deepcopy(self.model.state_dict())

            if verbose and (epoch + 1) % 10 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                msg = f"Epoch {epoch+1}/{self.config.epochs}: train_loss={train_loss:.6f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.6f}"
                msg += f", lr={lr:.2e}"
                print(msg)

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Generate predictions for a dataset.

        Returns numpy array of predictions.
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                features = batch[0].to(self.device)
                outputs = self.model(features)
                if outputs.dim() > 1:
                    outputs = outputs.squeeze(-1)
                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions)

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.history = checkpoint['history']


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: Optional[TrainerConfig] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for training a model.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    criterion : loss function
    config : TrainerConfig
    device : torch.device
    verbose : bool

    Returns
    -------
    result : dict with 'model', 'history', 'trainer'
    """
    trainer = Trainer(model, criterion, config, device)
    history = trainer.fit(train_loader, val_loader, verbose)

    return {
        "model": model,
        "history": history,
        "trainer": trainer
    }
