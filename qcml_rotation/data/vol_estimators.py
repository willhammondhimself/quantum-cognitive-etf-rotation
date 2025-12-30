"""
OHLC-based volatility estimators.

These estimators are 5-7x more efficient than close-to-close volatility
because they use intraday price information (High, Low, Open).

References:
- Parkinson (1980): "The Extreme Value Method for Estimating the Variance of the Rate of Return"
- Garman & Klass (1980): "On the Estimation of Security Price Volatilities from Historical Data"
- Rogers & Satchell (1991): "Estimating Variance From High, Low and Closing Prices"
- Yang & Zhang (2000): "Drift Independent Volatility Estimation Based on High, Low, Open, and Close Prices"
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


def compute_parkinson_volatility(
    high: Union[pd.Series, pd.DataFrame],
    low: Union[pd.Series, pd.DataFrame],
    window: int = 5,
    annualize: bool = True,
    trading_days: int = 252
) -> Union[pd.Series, pd.DataFrame]:
    """
    Parkinson volatility estimator using high-low range.

    This estimator is 5.2x more efficient than close-to-close volatility
    because it uses the full daily range information.

    Formula: sigma^2 = (1 / 4*ln(2)) * mean((ln(H/L))^2)

    Parameters
    ----------
    high : pd.Series or pd.DataFrame
        High prices.
    low : pd.Series or pd.DataFrame
        Low prices.
    window : int
        Rolling window size for averaging.
    annualize : bool
        Whether to annualize the volatility.
    trading_days : int
        Number of trading days per year for annualization.

    Returns
    -------
    volatility : pd.Series or pd.DataFrame
        Parkinson volatility estimate.
    """
    log_hl = np.log(high / low)
    factor = 1 / (4 * np.log(2))
    variance = factor * (log_hl ** 2).rolling(window).mean()
    vol = np.sqrt(variance)

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return vol


def compute_garman_klass_volatility(
    open_prices: Union[pd.Series, pd.DataFrame],
    high: Union[pd.Series, pd.DataFrame],
    low: Union[pd.Series, pd.DataFrame],
    close: Union[pd.Series, pd.DataFrame],
    window: int = 5,
    annualize: bool = True,
    trading_days: int = 252
) -> Union[pd.Series, pd.DataFrame]:
    """
    Garman-Klass volatility estimator.

    This estimator is 7.4x more efficient than close-to-close volatility
    and uses all OHLC data. It assumes no drift (mean return = 0).

    Formula: sigma^2 = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2

    Parameters
    ----------
    open_prices : pd.Series or pd.DataFrame
        Opening prices.
    high : pd.Series or pd.DataFrame
        High prices.
    low : pd.Series or pd.DataFrame
        Low prices.
    close : pd.Series or pd.DataFrame
        Closing prices.
    window : int
        Rolling window size for averaging.
    annualize : bool
        Whether to annualize the volatility.
    trading_days : int
        Number of trading days per year for annualization.

    Returns
    -------
    volatility : pd.Series or pd.DataFrame
        Garman-Klass volatility estimate.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_prices)

    # Garman-Klass formula
    variance = (0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2).rolling(window).mean()

    # Handle negative variance (can happen with extreme moves)
    variance = variance.clip(lower=0)
    vol = np.sqrt(variance)

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return vol


def compute_rogers_satchell_volatility(
    open_prices: Union[pd.Series, pd.DataFrame],
    high: Union[pd.Series, pd.DataFrame],
    low: Union[pd.Series, pd.DataFrame],
    close: Union[pd.Series, pd.DataFrame],
    window: int = 5,
    annualize: bool = True,
    trading_days: int = 252
) -> Union[pd.Series, pd.DataFrame]:
    """
    Rogers-Satchell volatility estimator.

    Unlike Garman-Klass, this estimator handles drift (non-zero mean returns).
    Better suited for trending markets.

    Formula: sigma^2 = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)

    Parameters
    ----------
    open_prices : pd.Series or pd.DataFrame
        Opening prices.
    high : pd.Series or pd.DataFrame
        High prices.
    low : pd.Series or pd.DataFrame
        Low prices.
    close : pd.Series or pd.DataFrame
        Closing prices.
    window : int
        Rolling window size for averaging.
    annualize : bool
        Whether to annualize the volatility.
    trading_days : int
        Number of trading days per year for annualization.

    Returns
    -------
    volatility : pd.Series or pd.DataFrame
        Rogers-Satchell volatility estimate.
    """
    log_hc = np.log(high / close)
    log_ho = np.log(high / open_prices)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open_prices)

    variance = (log_hc * log_ho + log_lc * log_lo).rolling(window).mean()

    # Handle negative variance
    variance = variance.clip(lower=0)
    vol = np.sqrt(variance)

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return vol


def compute_yang_zhang_volatility(
    open_prices: Union[pd.Series, pd.DataFrame],
    high: Union[pd.Series, pd.DataFrame],
    low: Union[pd.Series, pd.DataFrame],
    close: Union[pd.Series, pd.DataFrame],
    window: int = 5,
    annualize: bool = True,
    trading_days: int = 252
) -> Union[pd.Series, pd.DataFrame]:
    """
    Yang-Zhang volatility estimator.

    The most comprehensive OHLC estimator. Combines overnight volatility,
    open-to-close volatility, and Rogers-Satchell. Best for handling
    overnight gaps and intraday moves separately.

    Parameters
    ----------
    open_prices : pd.Series or pd.DataFrame
        Opening prices.
    high : pd.Series or pd.DataFrame
        High prices.
    low : pd.Series or pd.DataFrame
        Low prices.
    close : pd.Series or pd.DataFrame
        Closing prices.
    window : int
        Rolling window size for averaging.
    annualize : bool
        Whether to annualize the volatility.
    trading_days : int
        Number of trading days per year for annualization.

    Returns
    -------
    volatility : pd.Series or pd.DataFrame
        Yang-Zhang volatility estimate.
    """
    # Optimal k parameter
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # Overnight volatility (close-to-open)
    log_oc = np.log(open_prices / close.shift(1))
    overnight_var = log_oc.rolling(window).var()

    # Open-to-close volatility
    log_co = np.log(close / open_prices)
    open_close_var = log_co.rolling(window).var()

    # Rogers-Satchell component
    log_hc = np.log(high / close)
    log_ho = np.log(high / open_prices)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open_prices)
    rs_var = (log_hc * log_ho + log_lc * log_lo).rolling(window).mean()

    # Yang-Zhang combined variance
    variance = overnight_var + k * open_close_var + (1 - k) * rs_var

    # Handle negative variance
    variance = variance.clip(lower=0)
    vol = np.sqrt(variance)

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return vol


def compute_close_to_close_volatility(
    close: Union[pd.Series, pd.DataFrame],
    window: int = 5,
    annualize: bool = True,
    trading_days: int = 252
) -> Union[pd.Series, pd.DataFrame]:
    """
    Standard close-to-close volatility for comparison.

    This is the baseline estimator that OHLC estimators improve upon.

    Parameters
    ----------
    close : pd.Series or pd.DataFrame
        Closing prices.
    window : int
        Rolling window size.
    annualize : bool
        Whether to annualize the volatility.
    trading_days : int
        Number of trading days per year for annualization.

    Returns
    -------
    volatility : pd.Series or pd.DataFrame
        Close-to-close volatility estimate.
    """
    returns = np.log(close / close.shift(1))
    vol = returns.rolling(window).std()

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return vol


def compute_all_volatility_estimators(
    open_prices: Union[pd.Series, pd.DataFrame],
    high: Union[pd.Series, pd.DataFrame],
    low: Union[pd.Series, pd.DataFrame],
    close: Union[pd.Series, pd.DataFrame],
    windows: Optional[list] = None,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Compute all volatility estimators at multiple windows.

    Parameters
    ----------
    open_prices : pd.Series or pd.DataFrame
        Opening prices.
    high : pd.Series or pd.DataFrame
        High prices.
    low : pd.Series or pd.DataFrame
        Low prices.
    close : pd.Series or pd.DataFrame
        Closing prices.
    windows : list, optional
        Window sizes to compute. Default is [5, 10, 20].
    annualize : bool
        Whether to annualize the volatility.

    Returns
    -------
    features : pd.DataFrame
        DataFrame with all volatility estimator features.
    """
    if windows is None:
        windows = [5, 10, 20]

    # Handle both Series and DataFrame
    is_series = isinstance(close, pd.Series)
    if is_series:
        index = close.index
    else:
        index = close.index

    features = pd.DataFrame(index=index)

    for window in windows:
        suffix = f"_{window}d"

        # Close-to-close (baseline)
        cc_vol = compute_close_to_close_volatility(close, window, annualize)
        if is_series:
            features[f"vol_cc{suffix}"] = cc_vol
        else:
            for col in cc_vol.columns:
                features[f"{col}_vol_cc{suffix}"] = cc_vol[col]

        # Parkinson
        park_vol = compute_parkinson_volatility(high, low, window, annualize)
        if is_series:
            features[f"vol_parkinson{suffix}"] = park_vol
        else:
            for col in park_vol.columns:
                features[f"{col}_vol_parkinson{suffix}"] = park_vol[col]

        # Garman-Klass
        gk_vol = compute_garman_klass_volatility(open_prices, high, low, close, window, annualize)
        if is_series:
            features[f"vol_gk{suffix}"] = gk_vol
        else:
            for col in gk_vol.columns:
                features[f"{col}_vol_gk{suffix}"] = gk_vol[col]

        # Rogers-Satchell
        rs_vol = compute_rogers_satchell_volatility(open_prices, high, low, close, window, annualize)
        if is_series:
            features[f"vol_rs{suffix}"] = rs_vol
        else:
            for col in rs_vol.columns:
                features[f"{col}_vol_rs{suffix}"] = rs_vol[col]

        # Yang-Zhang
        yz_vol = compute_yang_zhang_volatility(open_prices, high, low, close, window, annualize)
        if is_series:
            features[f"vol_yz{suffix}"] = yz_vol
        else:
            for col in yz_vol.columns:
                features[f"{col}_vol_yz{suffix}"] = yz_vol[col]

    return features


def compute_volatility_features_for_ticker(
    open_prices: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    windows: Optional[list] = None
) -> pd.DataFrame:
    """
    Compute comprehensive volatility features for a single ticker.

    Includes all OHLC estimators plus derived features like:
    - Estimator ratios (efficiency indicators)
    - Volatility term structure
    - Volatility acceleration

    Parameters
    ----------
    open_prices : pd.Series
        Opening prices.
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Closing prices.
    windows : list, optional
        Window sizes. Default is [5, 10, 20, 60].

    Returns
    -------
    features : pd.DataFrame
        All volatility features.
    """
    if windows is None:
        windows = [5, 10, 20, 60]

    features = pd.DataFrame(index=close.index)

    # Compute all estimators at all windows
    for window in windows:
        suffix = f"_{window}d"

        features[f"vol_cc{suffix}"] = compute_close_to_close_volatility(close, window)
        features[f"vol_parkinson{suffix}"] = compute_parkinson_volatility(high, low, window)
        features[f"vol_gk{suffix}"] = compute_garman_klass_volatility(open_prices, high, low, close, window)
        features[f"vol_rs{suffix}"] = compute_rogers_satchell_volatility(open_prices, high, low, close, window)
        features[f"vol_yz{suffix}"] = compute_yang_zhang_volatility(open_prices, high, low, close, window)

    # Estimator ratios (can indicate unusual market conditions)
    # When range-based >> close-to-close, indicates intraday volatility
    features["vol_ratio_park_cc_5d"] = features["vol_parkinson_5d"] / (features["vol_cc_5d"] + 1e-8)
    features["vol_ratio_gk_cc_5d"] = features["vol_gk_5d"] / (features["vol_cc_5d"] + 1e-8)
    features["vol_ratio_yz_cc_5d"] = features["vol_yz_5d"] / (features["vol_cc_5d"] + 1e-8)

    # Volatility term structure (short vs long)
    features["vol_term_5_20"] = features["vol_yz_5d"] / (features["vol_yz_20d"] + 1e-8)
    features["vol_term_5_60"] = features["vol_yz_5d"] / (features["vol_yz_60d"] + 1e-8)
    features["vol_term_20_60"] = features["vol_yz_20d"] / (features["vol_yz_60d"] + 1e-8)

    # Volatility changes (acceleration)
    features["vol_change_5d"] = features["vol_yz_5d"].pct_change(5)
    features["vol_change_10d"] = features["vol_yz_10d"].pct_change(5)

    # Volatility of volatility (clustering)
    features["vol_of_vol"] = features["vol_yz_5d"].rolling(20).std()

    # Overnight vs intraday volatility (from Yang-Zhang components)
    log_oc = np.log(open_prices / close.shift(1))
    log_co = np.log(close / open_prices)
    features["overnight_vol_5d"] = log_oc.rolling(5).std() * np.sqrt(252)
    features["intraday_vol_5d"] = log_co.rolling(5).std() * np.sqrt(252)
    features["overnight_intraday_ratio"] = features["overnight_vol_5d"] / (features["intraday_vol_5d"] + 1e-8)

    # Intraday range features
    intraday_range = (high - low) / close
    features["avg_range_5d"] = intraday_range.rolling(5).mean()
    features["max_range_20d"] = intraday_range.rolling(20).max()

    return features
