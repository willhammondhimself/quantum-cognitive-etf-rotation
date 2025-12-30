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


def download_etf_data_ohlc(
    tickers: Optional[List[str]] = None,
    benchmark: str = DEFAULT_BENCHMARK,
    start_date: str = "2012-01-01",
    end_date: Optional[str] = None,
    cache_dir: str = "data/cache",
    force_refresh: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download daily OHLC and volume data for ETFs.

    OHLC-based volatility estimators are 5-7x more efficient than close-to-close.

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
    open_prices : pd.DataFrame
        Opening prices, indexed by date, columns are tickers.
    high_prices : pd.DataFrame
        High prices, indexed by date, columns are tickers.
    low_prices : pd.DataFrame
        Low prices, indexed by date, columns are tickers.
    close_prices : pd.DataFrame
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

    # Cache files for OHLC
    open_file = cache_path / "open.parquet"
    high_file = cache_path / "high.parquet"
    low_file = cache_path / "low.parquet"
    close_file = cache_path / "close.parquet"
    volume_file = cache_path / "volume_ohlc.parquet"

    # Check cache
    cache_files = [open_file, high_file, low_file, close_file, volume_file]
    if not force_refresh and all(f.exists() for f in cache_files):
        open_prices = pd.read_parquet(open_file)
        high_prices = pd.read_parquet(high_file)
        low_prices = pd.read_parquet(low_file)
        close_prices = pd.read_parquet(close_file)
        volume = pd.read_parquet(volume_file)

        # Verify all tickers present
        missing = set(all_tickers) - set(close_prices.columns)
        if not missing:
            print(f"Loaded cached OHLC data: {len(close_prices)} days, {len(close_prices.columns)} tickers")
            return open_prices, high_prices, low_prices, close_prices, volume

    print(f"Downloading OHLC data for {len(all_tickers)} tickers from {start_date} to {end_date}...")

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

    # Extract OHLC and volume
    if isinstance(data.columns, pd.MultiIndex):
        open_prices = data["Open"].copy()
        high_prices = data["High"].copy()
        low_prices = data["Low"].copy()
        close_prices = data["Close"].copy()
        volume = data["Volume"].copy()
    else:
        # Single ticker case
        open_prices = data[["Open"]].copy()
        open_prices.columns = [all_tickers[0]]
        high_prices = data[["High"]].copy()
        high_prices.columns = [all_tickers[0]]
        low_prices = data[["Low"]].copy()
        low_prices.columns = [all_tickers[0]]
        close_prices = data[["Close"]].copy()
        close_prices.columns = [all_tickers[0]]
        volume = data[["Volume"]].copy()
        volume.columns = [all_tickers[0]]

    # Ensure column order
    open_prices = open_prices[all_tickers]
    high_prices = high_prices[all_tickers]
    low_prices = low_prices[all_tickers]
    close_prices = close_prices[all_tickers]
    volume = volume[all_tickers]

    # Handle missing data
    open_prices = _clean_price_data(open_prices)
    high_prices = _clean_price_data(high_prices)
    low_prices = _clean_price_data(low_prices)
    close_prices = _clean_price_data(close_prices)
    volume = volume.fillna(0)

    # Validate OHLC consistency
    _validate_ohlc(open_prices, high_prices, low_prices, close_prices)

    # Cache
    open_prices.to_parquet(open_file)
    high_prices.to_parquet(high_file)
    low_prices.to_parquet(low_file)
    close_prices.to_parquet(close_file)
    volume.to_parquet(volume_file)

    print(f"Downloaded and cached OHLC: {len(close_prices)} days, {len(close_prices.columns)} tickers")
    return open_prices, high_prices, low_prices, close_prices, volume


def _validate_ohlc(
    open_prices: pd.DataFrame,
    high_prices: pd.DataFrame,
    low_prices: pd.DataFrame,
    close_prices: pd.DataFrame
) -> None:
    """Validate OHLC data consistency (High >= Low, etc.)."""
    # Check High >= Low
    violations = (high_prices < low_prices).sum().sum()
    if violations > 0:
        print(f"Warning: {violations} instances where High < Low (fixing...)")
        # Swap values where violated
        mask = high_prices < low_prices
        high_prices_fixed = high_prices.where(~mask, low_prices)
        low_prices_fixed = low_prices.where(~mask, high_prices)
        high_prices.update(high_prices_fixed)
        low_prices.update(low_prices_fixed)

    # Check High >= Open, Close and Low <= Open, Close
    high_violations = ((high_prices < open_prices) | (high_prices < close_prices)).sum().sum()
    low_violations = ((low_prices > open_prices) | (low_prices > close_prices)).sum().sum()

    if high_violations > 0 or low_violations > 0:
        print(f"Warning: {high_violations + low_violations} OHLC consistency issues (minor)")


def load_cached_ohlc_data(cache_dir: str = "data/cache") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load previously cached OHLC and volume data.

    Raises FileNotFoundError if cache doesn't exist.
    """
    cache_path = Path(cache_dir)
    open_file = cache_path / "open.parquet"
    high_file = cache_path / "high.parquet"
    low_file = cache_path / "low.parquet"
    close_file = cache_path / "close.parquet"
    volume_file = cache_path / "volume_ohlc.parquet"

    cache_files = [open_file, high_file, low_file, close_file, volume_file]
    if not all(f.exists() for f in cache_files):
        raise FileNotFoundError(
            f"OHLC cache not found in {cache_dir}. Run download_etf_data_ohlc first."
        )

    open_prices = pd.read_parquet(open_file)
    high_prices = pd.read_parquet(high_file)
    low_prices = pd.read_parquet(low_file)
    close_prices = pd.read_parquet(close_file)
    volume = pd.read_parquet(volume_file)

    return open_prices, high_prices, low_prices, close_prices, volume


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


def download_vix_data(
    start_date: str = "2012-01-01",
    end_date: Optional[str] = None,
    cache_dir: str = "data/cache",
    force_refresh: bool = False
) -> pd.Series:
    """
    Download VIX index data from Yahoo Finance.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date. Defaults to today.
    cache_dir : str
        Directory to cache parquet file.
    force_refresh : bool
        If True, re-download even if cache exists.

    Returns
    -------
    vix : pd.Series
        VIX closing values, indexed by date.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    vix_file = cache_path / "vix.parquet"

    # Check cache
    if not force_refresh and vix_file.exists():
        vix_df = pd.read_parquet(vix_file)
        print(f"Loaded cached VIX data: {len(vix_df)} days")
        return vix_df["VIX"]

    print(f"Downloading VIX data from {start_date} to {end_date}...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = yf.download(
            "^VIX",
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )

    if data.empty:
        print("Warning: No VIX data downloaded")
        return pd.Series(dtype=float, name="VIX")

    # Handle multi-index columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        vix = data["Close"]["^VIX"].copy()
    else:
        vix = data["Close"].copy()

    vix.name = "VIX"

    # Clean data
    vix = vix.ffill().bfill()

    # Cache
    vix_df = pd.DataFrame({"VIX": vix})
    vix_df.to_parquet(vix_file)

    print(f"Downloaded and cached VIX: {len(vix)} days")
    return vix


def download_vix_term_structure(
    start_date: str = "2012-01-01",
    end_date: Optional[str] = None,
    cache_dir: str = "data/cache",
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Download VIX and VIX3M for term structure analysis.

    The VIX term structure (VIX/VIX3M ratio) is a powerful leading indicator:
    - Ratio > 1 (backwardation): Near-term stress, market expects volatility
    - Ratio < 1 (contango): Normal conditions, market calm

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date. Defaults to today.
    cache_dir : str
        Directory to cache parquet file.
    force_refresh : bool
        If True, re-download even if cache exists.

    Returns
    -------
    vix_term : pd.DataFrame
        DataFrame with VIX, VIX3M, and derived term structure features.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    vix_term_file = cache_path / "vix_term_structure.parquet"

    # Check cache
    if not force_refresh and vix_term_file.exists():
        vix_term = pd.read_parquet(vix_term_file)
        print(f"Loaded cached VIX term structure: {len(vix_term)} days")
        return vix_term

    print(f"Downloading VIX term structure from {start_date} to {end_date}...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Download both VIX and VIX3M
        data = yf.download(
            ["^VIX", "^VIX3M"],
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )

    if data.empty:
        print("Warning: No VIX term structure data downloaded")
        return pd.DataFrame()

    # Extract close prices
    if isinstance(data.columns, pd.MultiIndex):
        vix = data["Close"]["^VIX"].copy()
        vix3m = data["Close"]["^VIX3M"].copy()
    else:
        # Fallback for single ticker case
        vix = data["Close"].copy()
        vix3m = pd.Series(dtype=float, index=vix.index)

    # Clean data
    vix = vix.ffill().bfill()
    vix3m = vix3m.ffill().bfill()

    # Build term structure DataFrame
    vix_term = pd.DataFrame(index=vix.index)
    vix_term["vix"] = vix
    vix_term["vix3m"] = vix3m

    # Term structure ratio (backwardation > 1, contango < 1)
    vix_term["vix_term_structure"] = vix / vix3m

    # Normalized term structure slope
    vix_term["vix_term_slope"] = (vix3m - vix) / vix

    # Term structure percentile (historical context)
    vix_term["vix_term_percentile"] = vix_term["vix_term_structure"].rolling(252).rank(pct=True)

    # Backwardation indicator (stress signal)
    vix_term["vix_backwardation"] = (vix_term["vix_term_structure"] > 1.0).astype(float)

    # VIX momentum features
    vix_term["vix_change_1d"] = vix.pct_change()
    vix_term["vix_change_5d"] = vix.pct_change(5)
    vix_term["vix_ma_20d"] = vix.rolling(20).mean()
    vix_term["vix_zscore"] = (vix - vix.rolling(20).mean()) / vix.rolling(20).std()

    # VIX spike detection
    vix_term["vix_spike"] = (vix.diff().abs() > 3.0).astype(float)  # 3+ point daily change

    # Cache
    vix_term.to_parquet(vix_term_file)

    print(f"Downloaded and cached VIX term structure: {len(vix_term)} days")
    return vix_term


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


def download_treasury_data(
    start_date: str = "2012-01-01",
    end_date: Optional[str] = None,
    cache_dir: str = "data/cache",
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Download Treasury yield data for yield curve analysis.

    Uses Treasury ETFs as proxies:
    - SHY: 1-3 year Treasury (short end)
    - IEI: 3-7 year Treasury (intermediate)
    - IEF: 7-10 year Treasury (long end)
    - TLT: 20+ year Treasury (very long)

    The yield curve slope can be approximated from price ratios.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date. Defaults to today.
    cache_dir : str
        Directory to cache parquet file.
    force_refresh : bool
        If True, re-download even if cache exists.

    Returns
    -------
    treasury_data : pd.DataFrame
        Treasury ETF prices and derived yield curve features.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    treasury_file = cache_path / "treasury.parquet"

    # Check cache
    if not force_refresh and treasury_file.exists():
        treasury_df = pd.read_parquet(treasury_file)
        print(f"Loaded cached Treasury data: {len(treasury_df)} days")
        return treasury_df

    print(f"Downloading Treasury data from {start_date} to {end_date}...")

    treasury_etfs = ["SHY", "IEI", "IEF", "TLT"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = yf.download(
            treasury_etfs,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )

    if data.empty:
        print("Warning: No Treasury data downloaded")
        return pd.DataFrame()

    # Handle multi-index columns
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = treasury_etfs[:1]

    prices = prices.ffill().bfill()

    # Compute yield curve features
    treasury_df = pd.DataFrame(index=prices.index)

    # Store raw prices
    for etf in treasury_etfs:
        if etf in prices.columns:
            treasury_df[etf] = prices[etf]

    # Yield curve slope proxy: TLT/SHY ratio
    # When long-term yields rise faster than short-term, TLT falls more
    if "TLT" in prices.columns and "SHY" in prices.columns:
        treasury_df["yield_curve_slope"] = prices["SHY"] / prices["TLT"]

        # Yield curve steepening/flattening (change in slope)
        treasury_df["yield_curve_change"] = treasury_df["yield_curve_slope"].pct_change(21)

        # Yield curve inversion signal (when short > long)
        # Actually approximated by TLT outperforming SHY
        treasury_df["yield_curve_inverted"] = (
            prices["TLT"].pct_change(63) > prices["SHY"].pct_change(63)
        ).astype(int)

    # Duration risk proxy: TLT volatility
    if "TLT" in prices.columns:
        tlt_returns = prices["TLT"].pct_change()
        treasury_df["duration_risk"] = tlt_returns.rolling(21).std() * np.sqrt(252)

    # Cache
    treasury_df.to_parquet(treasury_file)

    print(f"Downloaded and cached Treasury data: {len(treasury_df)} days")
    return treasury_df


def download_credit_spread_data(
    start_date: str = "2012-01-01",
    end_date: Optional[str] = None,
    cache_dir: str = "data/cache",
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Download credit spread proxy data.

    Uses ETF ratios as proxies for credit spreads:
    - HYG/LQD ratio: High yield vs investment grade (credit risk appetite)
    - JNK/TLT ratio: High yield vs Treasuries (credit spread proxy)

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date. Defaults to today.
    cache_dir : str
        Directory to cache parquet file.
    force_refresh : bool
        If True, re-download even if cache exists.

    Returns
    -------
    credit_data : pd.DataFrame
        Credit spread proxy features.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    credit_file = cache_path / "credit.parquet"

    # Check cache
    if not force_refresh and credit_file.exists():
        credit_df = pd.read_parquet(credit_file)
        print(f"Loaded cached Credit data: {len(credit_df)} days")
        return credit_df

    print(f"Downloading Credit spread data from {start_date} to {end_date}...")

    credit_etfs = ["HYG", "LQD", "JNK", "TLT"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = yf.download(
            credit_etfs,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )

    if data.empty:
        print("Warning: No Credit data downloaded")
        return pd.DataFrame()

    # Handle multi-index columns
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = credit_etfs[:1]

    prices = prices.ffill().bfill()

    # Compute credit spread features
    credit_df = pd.DataFrame(index=prices.index)

    # HYG/LQD ratio: credit risk appetite
    if "HYG" in prices.columns and "LQD" in prices.columns:
        credit_df["hyg_lqd_ratio"] = prices["HYG"] / prices["LQD"]

        # Change in credit risk appetite
        credit_df["credit_appetite_change"] = credit_df["hyg_lqd_ratio"].pct_change(21)

        # Credit stress indicator (when HYG underperforms LQD)
        credit_df["credit_stress"] = (
            prices["HYG"].pct_change(21) < prices["LQD"].pct_change(21)
        ).astype(int)

    # JNK/TLT ratio: credit spread proxy
    if "JNK" in prices.columns and "TLT" in prices.columns:
        credit_df["jnk_tlt_ratio"] = prices["JNK"] / prices["TLT"]

        # Spread widening/tightening
        credit_df["spread_change"] = credit_df["jnk_tlt_ratio"].pct_change(21)

    # High yield momentum
    if "HYG" in prices.columns:
        hyg_returns = prices["HYG"].pct_change()
        credit_df["hyg_momentum"] = prices["HYG"].pct_change(21)
        credit_df["hyg_volatility"] = hyg_returns.rolling(21).std() * np.sqrt(252)

    # Cache
    credit_df.to_parquet(credit_file)

    print(f"Downloaded and cached Credit data: {len(credit_df)} days")
    return credit_df


def download_macro_regime_data(
    start_date: str = "2012-01-01",
    end_date: Optional[str] = None,
    cache_dir: str = "data/cache",
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Download all macro regime data (VIX, Treasury, Credit).

    Convenience function that combines all regime indicators.

    Returns
    -------
    macro_data : pd.DataFrame
        Combined macro regime features.
    """
    # Download individual components
    vix = download_vix_data(start_date, end_date, cache_dir, force_refresh)
    treasury = download_treasury_data(start_date, end_date, cache_dir, force_refresh)
    credit = download_credit_spread_data(start_date, end_date, cache_dir, force_refresh)

    # Combine on common index
    macro_df = pd.DataFrame(index=vix.index if hasattr(vix, 'index') else [])

    if len(vix) > 0:
        macro_df["vix"] = vix

    if len(treasury) > 0:
        for col in treasury.columns:
            if col not in ["SHY", "IEI", "IEF", "TLT"]:  # Skip raw prices
                macro_df[col] = treasury[col]

    if len(credit) > 0:
        for col in credit.columns:
            if col not in ["HYG", "LQD", "JNK", "TLT"]:  # Skip raw prices
                macro_df[col] = credit[col]

    # Forward fill and clean
    macro_df = macro_df.ffill().bfill()

    print(f"Combined macro data: {len(macro_df)} days, {len(macro_df.columns)} features")
    return macro_df
