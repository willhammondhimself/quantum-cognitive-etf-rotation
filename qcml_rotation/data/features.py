"""
Feature engineering for ETF rotation model.

Builds features from daily price data at weekly rebalance dates.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature construction."""
    return_windows: List[int] = None  # lookback windows for returns
    vol_window: int = 20              # window for volatility calc
    benchmark: str = "SPY"

    def __post_init__(self):
        if self.return_windows is None:
            self.return_windows = [1, 5, 20]


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from prices."""
    return np.log(prices / prices.shift(1))


def compute_rolling_returns(
    log_returns: pd.DataFrame,
    windows: List[int]
) -> Dict[int, pd.DataFrame]:
    """
    Compute rolling cumulative returns over various windows.

    Returns dict mapping window -> return dataframe.
    """
    result = {}
    for w in windows:
        # Sum of log returns = log of cumulative return
        result[w] = log_returns.rolling(window=w).sum()
    return result


def compute_realized_vol(
    log_returns: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    Compute rolling realized volatility (annualized).

    Uses standard deviation of daily returns * sqrt(252).
    """
    daily_vol = log_returns.rolling(window=window).std()
    annualized = daily_vol * np.sqrt(252)
    return annualized


def build_features(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    config: Optional[FeatureConfig] = None
) -> pd.DataFrame:
    """
    Build feature matrix for all ETFs at all rebalance dates.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices. Columns are tickers.
    rebalance_dates : pd.DatetimeIndex
        Dates at which to compute features.
    config : FeatureConfig, optional
        Feature configuration.

    Returns
    -------
    features : pd.DataFrame
        Multi-indexed by (date, ticker). Columns are feature names.
    """
    if config is None:
        config = FeatureConfig()

    benchmark = config.benchmark
    etf_tickers = [c for c in prices.columns if c != benchmark]

    # Daily log returns
    log_rets = compute_log_returns(prices)

    # Rolling returns for different windows
    rolling_rets = compute_rolling_returns(log_rets, config.return_windows)

    # Realized volatility
    vol = compute_realized_vol(log_rets, config.vol_window)

    # Build feature rows
    rows = []

    for date in rebalance_dates:
        if date not in prices.index:
            continue

        # Get index position
        idx = prices.index.get_loc(date)

        # Skip if not enough history
        max_window = max(config.return_windows + [config.vol_window])
        if idx < max_window:
            continue

        # Benchmark values at this date
        spy_vol = vol.loc[date, benchmark]
        spy_ret_5d = rolling_rets[5].loc[date, benchmark]

        for ticker in etf_tickers:
            row = {
                "date": date,
                "ticker": ticker,
            }

            # Absolute features
            for w in config.return_windows:
                row[f"ret_{w}d"] = rolling_rets[w].loc[date, ticker]

            row[f"vol_{config.vol_window}d"] = vol.loc[date, ticker]

            # Relative features vs benchmark
            row["ret_5d_vs_spy"] = rolling_rets[5].loc[date, ticker] - spy_ret_5d
            row["vol_ratio_vs_spy"] = vol.loc[date, ticker] / spy_vol if spy_vol > 0 else 1.0

            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index(["date", "ticker"])

    return df


def compute_labels(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    benchmark: str = "SPY",
    forward_days: int = 5
) -> pd.DataFrame:
    """
    Compute next-week excess returns (label) for each ETF.

    Label = log return of ETF over next week - log return of SPY over next week.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices.
    rebalance_dates : pd.DatetimeIndex
        Dates at which to compute labels (these are the rebalance points).
    benchmark : str
        Benchmark ticker.
    forward_days : int
        Number of trading days to look ahead (5 = 1 week).

    Returns
    -------
    labels : pd.DataFrame
        Multi-indexed by (date, ticker). Single column 'excess_return'.
    """
    etf_tickers = [c for c in prices.columns if c != benchmark]

    # Forward returns (next week)
    log_rets = compute_log_returns(prices)
    fwd_rets = log_rets.shift(-forward_days).rolling(window=forward_days).sum()
    # This gives the return from t to t+5 (next 5 days)

    rows = []

    for date in rebalance_dates:
        if date not in prices.index:
            continue

        idx = prices.index.get_loc(date)

        # Need forward data available
        if idx + forward_days >= len(prices):
            continue

        spy_fwd = fwd_rets.loc[date, benchmark]

        for ticker in etf_tickers:
            etf_fwd = fwd_rets.loc[date, ticker]
            excess = etf_fwd - spy_fwd

            rows.append({
                "date": date,
                "ticker": ticker,
                "excess_return": excess
            })

    df = pd.DataFrame(rows)
    df = df.set_index(["date", "ticker"])

    return df


def merge_features_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge features and labels into a single dataframe.

    Drops rows where either features or labels are missing.
    """
    merged = features.join(labels, how="inner")
    merged = merged.dropna()
    return merged


def normalize_features(
    train_features: pd.DataFrame,
    val_features: Optional[pd.DataFrame] = None,
    test_features: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]:
    """
    Z-score normalize features using training set statistics.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training features (used to compute mean/std).
    val_features : pd.DataFrame, optional
        Validation features.
    test_features : pd.DataFrame, optional
        Test features.

    Returns
    -------
    train_norm, val_norm, test_norm : normalized dataframes
    stats : dict with 'mean' and 'std' Series for each feature
    """
    # Get only numeric feature columns (exclude 'excess_return' if present)
    feature_cols = [c for c in train_features.columns if c != "excess_return"]

    mean = train_features[feature_cols].mean()
    std = train_features[feature_cols].std()

    # Avoid division by zero
    std = std.replace(0, 1)

    def normalize(df):
        if df is None:
            return None
        result = df.copy()
        result[feature_cols] = (df[feature_cols] - mean) / std
        return result

    stats = {"mean": mean, "std": std}

    return normalize(train_features), normalize(val_features), normalize(test_features), stats


def get_feature_names(config: Optional[FeatureConfig] = None) -> List[str]:
    """Get list of feature column names."""
    if config is None:
        config = FeatureConfig()

    names = []
    for w in config.return_windows:
        names.append(f"ret_{w}d")
    names.append(f"vol_{config.vol_window}d")
    names.append("ret_5d_vs_spy")
    names.append("vol_ratio_vs_spy")

    return names
