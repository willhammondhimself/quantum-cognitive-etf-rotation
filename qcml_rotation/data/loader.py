"""
Data loading utilities for ETF price and volume data.

Downloads from Yahoo Finance via yfinance and caches locally as parquet.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
import warnings


# Default ETF universe
DEFAULT_BENCHMARK = "SPY"
DEFAULT_ETFS = [
    "XLK", "XLF", "XLE", "XLY", "XLP", "XLV", "XLU", "XLI", "XLB",
    "MTUM", "QUAL", "VLUE", "USMV", "IWD", "IWF", "IWM", "IWB"
]


def download_etf_data(
    tickers: Optional[List[str]] = None,
    benchmark: str = DEFAULT_BENCHMARK,
    start_date: str = "2012-01-01",
    end_date: Optional[str] = None,
    cache_dir: str = "data/cache",
    force_refresh: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download daily adjusted close and volume data for ETFs.

    Parameters
    ----------
    tickers : list of str, optional
        ETF tickers to download. Defaults to sector + factor ETFs.
    benchmark : str
        Benchmark ticker (SPY by default).
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date. Defaults to today.
    cache_dir : str
        Directory to cache parquet files.
    force_refresh : bool
        If True, re-download even if cache exists.

    Returns
    -------
    prices : pd.DataFrame
        Adjusted close prices, indexed by date, columns are tickers.
    volume : pd.DataFrame
        Volume data, same structure as prices.
    """
    if tickers is None:
        tickers = DEFAULT_ETFS

    # Make sure benchmark is included
    all_tickers = [benchmark] + [t for t in tickers if t != benchmark]

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    prices_file = cache_path / "prices.parquet"
    volume_file = cache_path / "volume.parquet"

    # Check cache
    if not force_refresh and prices_file.exists() and volume_file.exists():
        prices = pd.read_parquet(prices_file)
        volume = pd.read_parquet(volume_file)

        # Verify all tickers present
        missing = set(all_tickers) - set(prices.columns)
        if not missing:
            print(f"Loaded cached data: {len(prices)} days, {len(prices.columns)} tickers")
            return prices, volume

    print(f"Downloading data for {len(all_tickers)} tickers from {start_date} to {end_date}...")

    # Download with yfinance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = yf.download(
            all_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=True
        )

    # Extract prices and volume
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
        volume = data["Volume"].copy()
    else:
        # Single ticker case
        prices = data[["Close"]].copy()
        prices.columns = [all_tickers[0]]
        volume = data[["Volume"]].copy()
        volume.columns = [all_tickers[0]]

    # Ensure column order
    prices = prices[all_tickers]
    volume = volume[all_tickers]

    # Handle missing data
    prices = _clean_price_data(prices)
    volume = volume.fillna(0)

    # Cache
    prices.to_parquet(prices_file)
    volume.to_parquet(volume_file)

    print(f"Downloaded and cached: {len(prices)} days, {len(prices.columns)} tickers")
    return prices, volume


def _clean_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Clean price data by handling missing values.

    Strategy:
    1. Forward-fill gaps (carry forward last known price)
    2. Backward-fill any remaining NaNs at the start
    3. Drop any rows that still have NaNs (shouldn't happen)
    """
    prices = prices.ffill()
    prices = prices.bfill()

    # Drop rows with any remaining NaNs
    n_before = len(prices)
    prices = prices.dropna()
    n_dropped = n_before - len(prices)

    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with missing data")

    return prices


def load_cached_data(cache_dir: str = "data/cache") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load previously cached price and volume data.

    Raises FileNotFoundError if cache doesn't exist.
    """
    cache_path = Path(cache_dir)
    prices_file = cache_path / "prices.parquet"
    volume_file = cache_path / "volume.parquet"

    if not prices_file.exists() or not volume_file.exists():
        raise FileNotFoundError(
            f"Cache not found in {cache_dir}. Run download_etf_data first."
        )

    prices = pd.read_parquet(prices_file)
    volume = pd.read_parquet(volume_file)

    return prices, volume


def get_trading_dates(
    prices: pd.DataFrame,
    freq: str = "W-FRI"
) -> pd.DatetimeIndex:
    """
    Get rebalance dates (e.g., every Friday).

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with datetime index.
    freq : str
        Pandas frequency string. Default is weekly on Fridays.

    Returns
    -------
    dates : pd.DatetimeIndex
        Rebalance dates that exist in the price data.
    """
    # Get all Fridays in the date range
    all_fridays = pd.date_range(
        start=prices.index.min(),
        end=prices.index.max(),
        freq=freq
    )

    # Keep only dates that exist in our data
    # If a Friday is a holiday, use the previous trading day
    valid_dates = []
    for friday in all_fridays:
        if friday in prices.index:
            valid_dates.append(friday)
        else:
            # Find the closest prior trading day
            mask = prices.index < friday
            if mask.any():
                prior_date = prices.index[mask][-1]
                # Only use if it's within 3 days (same week)
                if (friday - prior_date).days <= 3:
                    valid_dates.append(prior_date)

    return pd.DatetimeIndex(valid_dates)
