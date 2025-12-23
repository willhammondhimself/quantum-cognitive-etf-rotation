"""
Specialized training loop for QCML model.

Handles week-based batching for proper ranking loss computation
and supports both standard and ablation training modes.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import copy

from ..data.dataset import ETFDataset, WeeklyBatchSampler
from ..models.qcml import QCMLWithRanking, QCMLConfig


@dataclass
class QCMLTrainerConfig:
    """Training configuration for QCML."""
    epochs: int = 150
    lr: float = 0.001
    early_stopping_patience: int = 15
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    clip_grad_norm: float = 1.0
    use_weekly_batching: bool = True  # If True, batch by week for ranking loss
    log_interval: int = 10


class QCMLTrainer:
    """
    Trainer for QCML model with support for ranking loss.

    Handles:
    - Week-based batching for proper ranking loss
    - Early stopping and LR scheduling
    - Model checkpointing
    - Ablation modes (no ranking, real-only)
    """

    def __init__(
        self,
        model: QCMLWithRanking,
        train_dataset: ETFDataset,
        val_dataset: ETFDataset,
        config: Optional[QCMLTrainerConfig] = None,
        device: Optional[torch.device] = None
    ):
        if config is None:
            config = QCMLTrainerConfig()

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
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

        self.best_model_state = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self.history = {
            "train_loss": [], "train_mse": [], "train_ranking": [],
            "val_loss": [], "val_mse": [], "val_ranking": []
        }

    def train(self, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Full training loop.

        Returns training history.
        """
        config = self.config

        for epoch in range(config.epochs):
            # Training
            train_metrics = self._train_epoch()
            self.history["train_loss"].append(train_metrics["total"])
            self.history["train_mse"].append(train_metrics["mse"])
            self.history["train_ranking"].append(train_metrics["ranking"])

            # Validation
            val_metrics = self._validate()
            self.history["val_loss"].append(val_metrics["total"])
            self.history["val_mse"].append(val_metrics["mse"])
            self.history["val_ranking"].append(val_metrics["ranking"])

            # LR scheduling
            self.scheduler.step(val_metrics["total"])

            # Early stopping check
            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= config.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Logging
            if verbose and (epoch + 1) % config.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{config.epochs}: "
                      f"train_loss={train_metrics['total']:.6f} "
                      f"(mse={train_metrics['mse']:.6f}, rank={train_metrics['ranking']:.6f}) | "
                      f"val_loss={val_metrics['total']:.6f} | lr={lr:.2e}")

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.history

    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_ranking = 0.0
        n_batches = 0

        if self.config.use_weekly_batching:
            # Use week-based batching
            sampler = WeeklyBatchSampler(self.train_dataset, shuffle=True)

            for week_indices in sampler:
                # Get all samples for this week
                features = self.train_dataset.features[week_indices].to(self.device)
                labels = self.train_dataset.labels[week_indices].to(self.device)
                date_indices = self.train_dataset.date_indices[week_indices].to(self.device)

                self.optimizer.zero_grad()

                loss, components = self.model.compute_loss(
                    features, labels, week_indices=date_indices
                )

                loss.backward()

                if self.config.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.clip_grad_norm
                    )

                self.optimizer.step()

                total_loss += components["total"]
                total_mse += components["mse"]
                total_ranking += components["ranking"]
                n_batches += 1

        else:
            # Standard batch-based training
            loader = DataLoader(
                self.train_dataset,
                batch_size=64,
                shuffle=True
            )

            for batch in loader:
                features, labels, date_idx, ticker_idx = batch
                features = features.to(self.device)
                labels = labels.to(self.device)
                date_idx = date_idx.to(self.device)

                self.optimizer.zero_grad()

                loss, components = self.model.compute_loss(
                    features, labels, week_indices=date_idx
                )

                loss.backward()

                if self.config.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.clip_grad_norm
                    )

                self.optimizer.step()

                total_loss += components["total"]
                total_mse += components["mse"]
                total_ranking += components["ranking"]
                n_batches += 1

        return {
            "total": total_loss / n_batches,
            "mse": total_mse / n_batches,
            "ranking": total_ranking / n_batches
        }

    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_loss = 0.0
        total_mse = 0.0
        total_ranking = 0.0
        n_batches = 0

        with torch.no_grad():
            sampler = WeeklyBatchSampler(self.val_dataset, shuffle=False)

            for week_indices in sampler:
                features = self.val_dataset.features[week_indices].to(self.device)
                labels = self.val_dataset.labels[week_indices].to(self.device)
                date_indices = self.val_dataset.date_indices[week_indices].to(self.device)

                _, components = self.model.compute_loss(
                    features, labels, week_indices=date_indices
                )

                total_loss += components["total"]
                total_mse += components["mse"]
                total_ranking += components["ranking"]
                n_batches += 1

        return {
            "total": total_loss / n_batches,
            "mse": total_mse / n_batches,
            "ranking": total_ranking / n_batches
        }

    def predict(self, dataset: ETFDataset) -> np.ndarray:
        """Generate predictions for a dataset."""
        self.model.eval()

        predictions = []
        with torch.no_grad():
            features = dataset.features.to(self.device)
            preds = self.model(features)
            predictions = preds.cpu().numpy()

        return predictions

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']


def train_qcml(
    train_dataset: ETFDataset,
    val_dataset: ETFDataset,
    model_config: Optional[QCMLConfig] = None,
    trainer_config: Optional[QCMLTrainerConfig] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Tuple[QCMLWithRanking, Dict[str, List[float]]]:
    """
    Convenience function to train QCML model.

    Parameters
    ----------
    train_dataset, val_dataset : ETFDataset
    model_config : QCMLConfig
    trainer_config : QCMLTrainerConfig
    device : torch.device
    verbose : bool

    Returns
    -------
    model : trained QCMLWithRanking model
    history : training history
    """
    from ..models.qcml import create_qcml_model

    if model_config is None:
        model_config = QCMLConfig()

    model = create_qcml_model(
        input_dim=train_dataset.n_features,
        config=model_config,
        device=device
    )

    trainer = QCMLTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_config,
        device=device
    )

    history = trainer.train(verbose=verbose)

    return model, history
