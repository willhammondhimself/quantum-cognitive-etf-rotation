"""
PyTorch Dataset for ETF rotation model.

Handles train/val/test splitting and provides both individual samples
and week-grouped samples (for ranking loss).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class DataSplit:
    """Container for train/val/test data."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_cols: List[str]
    stats: Dict  # normalization stats


class ETFDataset(Dataset):
    """
    PyTorch Dataset for ETF prediction.

    Each sample is (features, label, date_idx, ticker_idx).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "excess_return"
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Data with multi-index (date, ticker). Contains features and labels.
        feature_cols : list
            Names of feature columns.
        label_col : str
            Name of label column.
        """
        self.data = data.reset_index()
        self.feature_cols = feature_cols
        self.label_col = label_col

        # Create mappings for dates and tickers
        self.dates = self.data["date"].unique()
        self.tickers = self.data["ticker"].unique()

        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}

        # Precompute tensors
        self.features = torch.tensor(
            self.data[feature_cols].values,
            dtype=torch.float32
        )
        self.labels = torch.tensor(
            self.data[label_col].values,
            dtype=torch.float32
        )

        # Store date/ticker indices
        self.date_indices = torch.tensor(
            [self.date_to_idx[d] for d in self.data["date"]],
            dtype=torch.long
        )
        self.ticker_indices = torch.tensor(
            [self.ticker_to_idx[t] for t in self.data["ticker"]],
            dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Returns (features, label, date_idx, ticker_idx)."""
        return (
            self.features[idx],
            self.labels[idx],
            self.date_indices[idx].item(),
            self.ticker_indices[idx].item()
        )

    def get_week_samples(self, date_idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Get all samples for a given week (for ranking loss).

        Returns
        -------
        features : Tensor of shape (n_etfs, n_features)
        labels : Tensor of shape (n_etfs,)
        ticker_indices : list of ticker indices
        """
        mask = self.date_indices == date_idx
        features = self.features[mask]
        labels = self.labels[mask]
        ticker_idxs = self.ticker_indices[mask].tolist()

        return features, labels, ticker_idxs

    def get_all_weeks(self) -> List[int]:
        """Get list of unique date indices."""
        return list(range(len(self.dates)))

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def n_tickers(self) -> int:
        return len(self.tickers)


def create_data_splits(
    data: pd.DataFrame,
    feature_cols: List[str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    label_col: str = "excess_return"
) -> DataSplit:
    """
    Create time-based train/val/test splits.

    No shuffling - strict temporal ordering to avoid lookahead bias.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset with multi-index (date, ticker).
    feature_cols : list
        Feature column names.
    train_ratio : float
        Fraction of dates for training.
    val_ratio : float
        Fraction of dates for validation.
    label_col : str
        Label column name.

    Returns
    -------
    DataSplit with normalized train/val/test dataframes.
    """
    # Get unique dates sorted chronologically
    dates = data.index.get_level_values("date").unique().sort_values()
    n_dates = len(dates)

    # Split points
    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    print(f"Split: {len(train_dates)} train / {len(val_dates)} val / {len(test_dates)} test weeks")
    print(f"Train: {train_dates[0].date()} to {train_dates[-1].date()}")
    print(f"Val:   {val_dates[0].date()} to {val_dates[-1].date()}")
    print(f"Test:  {test_dates[0].date()} to {test_dates[-1].date()}")

    # Split data
    train_data = data.loc[data.index.get_level_values("date").isin(train_dates)]
    val_data = data.loc[data.index.get_level_values("date").isin(val_dates)]
    test_data = data.loc[data.index.get_level_values("date").isin(test_dates)]

    # Normalize using training stats
    mean = train_data[feature_cols].mean()
    std = train_data[feature_cols].std().replace(0, 1)

    def normalize(df):
        result = df.copy()
        result[feature_cols] = (df[feature_cols] - mean) / std
        return result

    stats = {"mean": mean, "std": std}

    return DataSplit(
        train=normalize(train_data),
        val=normalize(val_data),
        test=normalize(test_data),
        feature_cols=feature_cols,
        stats=stats
    )


def create_dataloaders(
    split: DataSplit,
    batch_size: int = 64,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from a DataSplit.

    Parameters
    ----------
    split : DataSplit
        Train/val/test data.
    batch_size : int
        Batch size for all loaders.
    shuffle_train : bool
        Whether to shuffle training data (within the temporal split).

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_ds = ETFDataset(split.train, split.feature_cols)
    val_ds = ETFDataset(split.val, split.feature_cols)
    test_ds = ETFDataset(split.test, split.feature_cols)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


class WeeklyBatchSampler:
    """
    Batch sampler that groups samples by week.

    Useful for ranking loss which needs all ETFs from the same week.
    """

    def __init__(self, dataset: ETFDataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.week_indices = self._group_by_week()

    def _group_by_week(self) -> List[List[int]]:
        """Group sample indices by date."""
        groups = {}
        for idx in range(len(self.dataset)):
            date_idx = self.dataset.date_indices[idx].item()
            if date_idx not in groups:
                groups[date_idx] = []
            groups[date_idx].append(idx)

        return list(groups.values())

    def __iter__(self):
        weeks = list(range(len(self.week_indices)))
        if self.shuffle:
            np.random.shuffle(weeks)

        for week_idx in weeks:
            yield self.week_indices[week_idx]

    def __len__(self):
        return len(self.week_indices)
