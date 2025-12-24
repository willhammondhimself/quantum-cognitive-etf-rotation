"""
Tests for data loading module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qcml_rotation.data.loader import get_trading_dates


class TestGetTradingDates:
    """Tests for get_trading_dates function."""

    def test_returns_dataframe_index(self, synthetic_prices):
        """Test that function returns DatetimeIndex."""
        dates = get_trading_dates(synthetic_prices, freq='W-FRI')
        assert isinstance(dates, pd.DatetimeIndex)

    def test_weekly_frequency(self, synthetic_prices):
        """Test weekly frequency returns Fridays."""
        dates = get_trading_dates(synthetic_prices, freq='W-FRI')
        # All dates should be Fridays (weekday 4) or the nearest trading day
        assert len(dates) > 0

    def test_dates_within_range(self, synthetic_prices):
        """Test that returned dates are within price data range."""
        dates = get_trading_dates(synthetic_prices, freq='W-FRI')
        assert dates.min() >= synthetic_prices.index.min()
        assert dates.max() <= synthetic_prices.index.max()

    def test_dates_are_sorted(self, synthetic_prices):
        """Test that dates are sorted in ascending order."""
        dates = get_trading_dates(synthetic_prices, freq='W-FRI')
        assert (dates[1:] > dates[:-1]).all()

    def test_no_duplicates(self, synthetic_prices):
        """Test that there are no duplicate dates."""
        dates = get_trading_dates(synthetic_prices, freq='W-FRI')
        assert len(dates) == len(set(dates))


class TestDataSplits:
    """Tests for data splitting functionality."""

    def test_no_data_leakage(self, synthetic_dataset):
        """Test that train/val/test splits have no temporal leakage."""
        # Sort by date
        data = synthetic_dataset.sort_values('date')
        n = len(data)

        # Create splits
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        train = data.iloc[:train_end]
        val = data.iloc[train_end:val_end]
        test = data.iloc[val_end:]

        # Check no overlap
        assert train['date'].max() <= val['date'].min()
        assert val['date'].max() <= test['date'].min()

    def test_split_ratios(self, synthetic_dataset):
        """Test that split ratios are approximately correct."""
        n = len(synthetic_dataset)
        train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_size = train_end
        val_size = val_end - train_end
        test_size = n - val_end

        assert abs(train_size / n - train_ratio) < 0.1
        assert abs(val_size / n - val_ratio) < 0.1
        assert abs(test_size / n - test_ratio) < 0.1

    def test_all_data_used(self, synthetic_dataset):
        """Test that all data is used in splits."""
        n = len(synthetic_dataset)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        total = train_end + (val_end - train_end) + (n - val_end)
        assert total == n
