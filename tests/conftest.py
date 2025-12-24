"""
Pytest fixtures and synthetic data for testing.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta


@pytest.fixture
def seed():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    return 42


@pytest.fixture
def synthetic_prices():
    """Generate synthetic price data for testing."""
    n_days = 100
    n_tickers = 5
    tickers = ['SPY', 'XLK', 'XLF', 'XLE', 'XLY']

    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_days, freq='B')

    # Generate random walk prices
    np.random.seed(42)
    returns = np.random.randn(n_days, n_tickers) * 0.01
    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    df = pd.DataFrame(prices, index=dates, columns=tickers)
    return df


@pytest.fixture
def synthetic_features():
    """Generate synthetic feature data for testing."""
    n_samples = 200
    n_features = 6

    np.random.seed(42)
    features = np.random.randn(n_samples, n_features) * 0.01
    return features


@pytest.fixture
def synthetic_labels():
    """Generate synthetic label data for testing."""
    n_samples = 200

    np.random.seed(42)
    labels = np.random.randn(n_samples) * 0.01
    return labels


@pytest.fixture
def synthetic_dataset(synthetic_features, synthetic_labels):
    """Generate synthetic dataset with features and labels."""
    n_samples = len(synthetic_labels)
    n_weeks = 20
    samples_per_week = n_samples // n_weeks

    # Create DataFrame
    data = pd.DataFrame(synthetic_features, columns=[
        'ret_1d', 'ret_5d', 'ret_20d', 'vol_20d', 'ret_5d_vs_spy', 'vol_ratio_vs_spy'
    ])
    data['excess_return'] = synthetic_labels

    # Add metadata
    dates = pd.date_range(start='2020-01-01', periods=n_weeks, freq='W-FRI')
    tickers = ['XLK', 'XLF', 'XLE', 'XLY', 'XLP'] * (samples_per_week // 5 + 1)

    date_list = []
    ticker_list = []
    for i, date in enumerate(dates):
        for j in range(min(samples_per_week, n_samples - i * samples_per_week)):
            if len(date_list) < n_samples:
                date_list.append(date)
                ticker_list.append(tickers[j % len(tickers)])

    data['date'] = date_list[:n_samples]
    data['ticker'] = ticker_list[:n_samples]

    return data


@pytest.fixture
def synthetic_predictions():
    """Generate synthetic predictions for testing."""
    n_samples = 100
    np.random.seed(42)
    return np.random.randn(n_samples) * 0.01


@pytest.fixture
def synthetic_returns():
    """Generate synthetic weekly returns for backtest testing."""
    n_weeks = 50
    np.random.seed(42)
    returns = np.random.randn(n_weeks) * 0.02  # ~2% weekly vol
    return returns


@pytest.fixture
def device():
    """Get available device for PyTorch."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@pytest.fixture
def feature_cols():
    """Standard feature column names."""
    return ['ret_1d', 'ret_5d', 'ret_20d', 'vol_20d', 'ret_5d_vs_spy', 'vol_ratio_vs_spy']
