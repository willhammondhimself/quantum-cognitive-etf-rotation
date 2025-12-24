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
    # Extended feature settings
    include_technical: bool = True    # RSI, Bollinger, ATR
    include_cross_sectional: bool = True  # rank features
    include_regime: bool = True       # VIX, market regime
    rsi_window: int = 14
    bollinger_window: int = 20
    atr_window: int = 14
    momentum_windows: List[int] = None  # longer momentum windows

    def __post_init__(self):
        if self.return_windows is None:
            self.return_windows = [1, 5, 20]
        if self.momentum_windows is None:
            self.momentum_windows = [20, 60]  # 1-month and 3-month momentum


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


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Compute Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    where RS = avg_gain / avg_loss over window

    Returns values between 0-100.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)  # Neutral RSI for undefined cases


def compute_bollinger_pct_b(
    prices: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Compute Bollinger Band %B indicator.

    %B = (Price - Lower Band) / (Upper Band - Lower Band)

    Values:
    - %B > 1: Price above upper band (overbought)
    - %B < 0: Price below lower band (oversold)
    - %B = 0.5: Price at middle band
    """
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()

    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)

    band_width = upper_band - lower_band
    pct_b = (prices - lower_band) / band_width.replace(0, np.nan)

    return pct_b.fillna(0.5)


def compute_atr(
    prices: pd.DataFrame,
    window: int = 14
) -> pd.DataFrame:
    """
    Compute Average True Range (ATR) as percentage of price.

    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    For daily close-only data, we approximate with price range.

    Returns ATR as percentage of price (normalized).
    """
    # Using close-only approximation: |today's return| as proxy for range
    returns = prices.pct_change().abs()
    atr = returns.rolling(window=window).mean()

    return atr


def compute_momentum(
    prices: pd.DataFrame,
    windows: List[int]
) -> Dict[int, pd.DataFrame]:
    """
    Compute price momentum (rate of change) over various windows.

    Momentum = (Price / Price_n_days_ago) - 1
    """
    result = {}
    for w in windows:
        result[w] = prices.pct_change(periods=w)
    return result


# =============================================================================
# CROSS-SECTIONAL FEATURES
# =============================================================================

def compute_cross_sectional_rank(
    values: pd.DataFrame,
    ascending: bool = True
) -> pd.DataFrame:
    """
    Compute cross-sectional percentile rank at each date.

    Ranks each ticker relative to others on the same date.
    Returns values between 0-1.
    """
    # Rank each row (date) across columns (tickers)
    ranks = values.rank(axis=1, pct=True, ascending=ascending)
    return ranks


def compute_cross_sectional_zscore(values: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectional z-score at each date.

    Z-scores each ticker relative to the cross-section mean/std.
    """
    mean = values.mean(axis=1)
    std = values.std(axis=1)

    zscore = values.sub(mean, axis=0).div(std.replace(0, 1), axis=0)
    return zscore


# =============================================================================
# REGIME INDICATORS
# =============================================================================

def compute_market_regime(
    benchmark_returns: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Compute market regime based on rolling returns.

    Returns:
    - 1: Bull market (positive rolling return)
    - 0: Bear market (negative rolling return)
    """
    rolling_ret = benchmark_returns.rolling(window=window).sum()
    regime = (rolling_ret > 0).astype(int)
    return regime


def compute_volatility_regime(
    returns: pd.DataFrame,
    window: int = 20,
    threshold_percentile: float = 50
) -> pd.Series:
    """
    Compute volatility regime based on rolling volatility.

    Returns:
    - 1: High volatility regime
    - 0: Low volatility regime
    """
    # Use benchmark or average volatility
    if isinstance(returns, pd.DataFrame):
        vol = returns.std(axis=1).rolling(window=window).mean()
    else:
        vol = returns.rolling(window=window).std()

    median_vol = vol.expanding().median()
    regime = (vol > median_vol).astype(int)
    return regime


def build_features(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    config: Optional[FeatureConfig] = None,
    vix_data: Optional[pd.Series] = None
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
    vix_data : pd.Series, optional
        VIX index data for regime features.

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

    # ==========================================================================
    # Technical indicators (computed once for all tickers)
    # ==========================================================================
    if config.include_technical:
        rsi = compute_rsi(prices, window=config.rsi_window)
        bollinger_b = compute_bollinger_pct_b(prices, window=config.bollinger_window)
        atr = compute_atr(prices, window=config.atr_window)
        momentum = compute_momentum(prices, config.momentum_windows)

    # ==========================================================================
    # Cross-sectional features (computed for ETFs only, per date)
    # ==========================================================================
    if config.include_cross_sectional:
        # Get 20-day momentum for ranking
        mom_20d = prices[etf_tickers].pct_change(periods=20)
        mom_60d = prices[etf_tickers].pct_change(periods=60)
        vol_20d = log_rets[etf_tickers].rolling(window=20).std()

        # Compute cross-sectional ranks
        rank_mom_20d = compute_cross_sectional_rank(mom_20d, ascending=True)
        rank_mom_60d = compute_cross_sectional_rank(mom_60d, ascending=True)
        rank_vol = compute_cross_sectional_rank(vol_20d, ascending=False)  # Lower vol = higher rank

        # Cross-sectional z-scores
        zscore_mom_20d = compute_cross_sectional_zscore(mom_20d)

    # ==========================================================================
    # Regime indicators
    # ==========================================================================
    if config.include_regime:
        spy_log_rets = log_rets[benchmark]
        market_regime = compute_market_regime(spy_log_rets, window=20)
        vol_regime = compute_volatility_regime(log_rets[etf_tickers], window=20)

    # ==========================================================================
    # Build feature rows
    # ==========================================================================
    rows = []

    # Compute max window needed
    all_windows = config.return_windows + [config.vol_window]
    if config.include_technical:
        all_windows += [config.rsi_window, config.bollinger_window, config.atr_window]
        all_windows += config.momentum_windows
    max_window = max(all_windows) + 5  # Add buffer

    for date in rebalance_dates:
        if date not in prices.index:
            continue

        # Get index position
        idx = prices.index.get_loc(date)

        # Skip if not enough history
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

            # ------------------------------------------------------------------
            # Original features
            # ------------------------------------------------------------------
            for w in config.return_windows:
                row[f"ret_{w}d"] = rolling_rets[w].loc[date, ticker]

            row[f"vol_{config.vol_window}d"] = vol.loc[date, ticker]
            row["ret_5d_vs_spy"] = rolling_rets[5].loc[date, ticker] - spy_ret_5d
            row["vol_ratio_vs_spy"] = vol.loc[date, ticker] / spy_vol if spy_vol > 0 else 1.0

            # ------------------------------------------------------------------
            # Technical indicators
            # ------------------------------------------------------------------
            if config.include_technical:
                row["rsi"] = rsi.loc[date, ticker]
                row["bollinger_b"] = bollinger_b.loc[date, ticker]
                row["atr"] = atr.loc[date, ticker]

                for w in config.momentum_windows:
                    row[f"mom_{w}d"] = momentum[w].loc[date, ticker]

            # ------------------------------------------------------------------
            # Cross-sectional features
            # ------------------------------------------------------------------
            if config.include_cross_sectional:
                row["rank_mom_20d"] = rank_mom_20d.loc[date, ticker]
                row["rank_mom_60d"] = rank_mom_60d.loc[date, ticker]
                row["rank_vol"] = rank_vol.loc[date, ticker]
                row["zscore_mom_20d"] = zscore_mom_20d.loc[date, ticker]

            # ------------------------------------------------------------------
            # Regime indicators
            # ------------------------------------------------------------------
            if config.include_regime:
                row["market_regime"] = market_regime.loc[date]
                row["vol_regime"] = vol_regime.loc[date]

                # VIX features (if available)
                if vix_data is not None and date in vix_data.index:
                    vix_val = vix_data.loc[date]
                    row["vix_level"] = vix_val
                    row["vix_high"] = 1 if vix_val > 20 else 0
                    row["vix_extreme"] = 1 if vix_val > 30 else 0

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


def get_feature_names(config: Optional[FeatureConfig] = None, include_vix: bool = False) -> List[str]:
    """Get list of feature column names."""
    if config is None:
        config = FeatureConfig()

    names = []

    # Original features
    for w in config.return_windows:
        names.append(f"ret_{w}d")
    names.append(f"vol_{config.vol_window}d")
    names.append("ret_5d_vs_spy")
    names.append("vol_ratio_vs_spy")

    # Technical indicators
    if config.include_technical:
        names.append("rsi")
        names.append("bollinger_b")
        names.append("atr")
        for w in config.momentum_windows:
            names.append(f"mom_{w}d")

    # Cross-sectional features
    if config.include_cross_sectional:
        names.append("rank_mom_20d")
        names.append("rank_mom_60d")
        names.append("rank_vol")
        names.append("zscore_mom_20d")

    # Regime indicators
    if config.include_regime:
        names.append("market_regime")
        names.append("vol_regime")
        if include_vix:
            names.append("vix_level")
            names.append("vix_high")
            names.append("vix_extreme")

    return names
