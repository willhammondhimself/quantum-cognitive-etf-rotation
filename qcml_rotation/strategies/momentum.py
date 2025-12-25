"""
Momentum-based strategies for ETF rotation.

Key insight: Instead of trying to PREDICT returns, FOLLOW relative strength.
Academic research shows cross-sectional momentum is more robust than
return prediction.

Strategies implemented:
1. Dual Momentum - Combines absolute and relative momentum
2. Trend Following Overlay - Only trade with the trend
3. Relative Strength - Rank-based allocation
4. Risk-Adjusted Momentum - Volatility-scaled positions
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class MomentumConfig:
    """Configuration for momentum strategies."""
    # Momentum lookback periods
    fast_lookback: int = 21  # ~1 month
    slow_lookback: int = 63  # ~3 months
    long_lookback: int = 252  # ~1 year

    # Trend filter
    use_trend_filter: bool = True
    trend_lookback: int = 200  # 200-day MA for trend

    # Volatility targeting
    use_vol_scaling: bool = True
    target_vol: float = 0.15  # 15% annualized target vol
    vol_lookback: int = 21

    # Position sizing
    top_k: int = 3
    bottom_k: int = 3

    # Cash allocation when trend is negative
    cash_threshold: float = 0.0  # Go to cash if below this


class DualMomentumStrategy:
    """
    Gary Antonacci's Dual Momentum strategy adapted for sector rotation.

    Combines:
    1. Absolute Momentum: Only invest when momentum > 0 (trend following)
    2. Relative Momentum: Rank assets and pick the best (cross-sectional)

    Reference: "Dual Momentum Investing" by Gary Antonacci
    """

    def __init__(self, config: Optional[MomentumConfig] = None):
        self.config = config or MomentumConfig()

    def compute_signals(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        benchmark: str = 'SPY'
    ) -> Dict[str, float]:
        """
        Compute dual momentum signals for all assets.

        Returns
        -------
        signals : dict mapping ticker -> signal strength
            Positive = long, negative = short, 0 = avoid
        """
        # Get lookback data
        idx = prices.index.get_loc(current_date)
        if idx < self.config.long_lookback:
            return {}

        lookback_start = idx - self.config.long_lookback

        # Calculate returns over different horizons
        fast_ret = self._compute_return(prices, idx, self.config.fast_lookback)
        slow_ret = self._compute_return(prices, idx, self.config.slow_lookback)
        long_ret = self._compute_return(prices, idx, self.config.long_lookback)

        # Exclude benchmark from signals
        etf_tickers = [c for c in prices.columns if c != benchmark]

        # Step 1: Absolute momentum filter (trend)
        # Only invest if benchmark is in uptrend
        benchmark_trend = long_ret.get(benchmark, 0)

        signals = {}

        for ticker in etf_tickers:
            # Step 2: Relative momentum (cross-sectional rank)
            # Average momentum across horizons
            ticker_fast = fast_ret.get(ticker, 0)
            ticker_slow = slow_ret.get(ticker, 0)
            ticker_long = long_ret.get(ticker, 0)

            # Composite momentum score
            momentum_score = 0.5 * ticker_fast + 0.3 * ticker_slow + 0.2 * ticker_long

            # Apply absolute momentum filter
            if self.config.use_trend_filter:
                ticker_abs = ticker_long  # Asset's own trend
                if ticker_abs < self.config.cash_threshold:
                    # Asset in downtrend - reduce or avoid
                    momentum_score *= 0.5

                if benchmark_trend < self.config.cash_threshold:
                    # Market in downtrend - be more defensive
                    momentum_score *= 0.5

            signals[ticker] = momentum_score

        return signals

    def _compute_return(
        self,
        prices: pd.DataFrame,
        current_idx: int,
        lookback: int
    ) -> Dict[str, float]:
        """Compute returns over lookback period."""
        if current_idx < lookback:
            return {}

        current_prices = prices.iloc[current_idx]
        past_prices = prices.iloc[current_idx - lookback]

        returns = {}
        for col in prices.columns:
            if past_prices[col] > 0:
                returns[col] = (current_prices[col] / past_prices[col]) - 1
            else:
                returns[col] = 0

        return returns

    def generate_weights(
        self,
        signals: Dict[str, float],
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        strategy: str = 'long_short'
    ) -> Dict[str, float]:
        """
        Convert signals to portfolio weights.

        Parameters
        ----------
        signals : dict
            Signal strength for each ticker.
        prices : pd.DataFrame
            Price data for volatility calculation.
        current_date : pd.Timestamp
            Current date for vol lookback.
        strategy : str
            'long_only' or 'long_short'

        Returns
        -------
        weights : dict mapping ticker -> weight
        """
        if not signals:
            return {}

        # Sort by signal strength
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        tickers = [t for t, _ in sorted_signals]
        signal_values = [s for _, s in sorted_signals]

        weights = {t: 0.0 for t in tickers}

        # Long top_k
        long_tickers = tickers[:self.config.top_k]
        for ticker in long_tickers:
            weights[ticker] = 1.0 / self.config.top_k

        # Short bottom_k (if long_short)
        if strategy == 'long_short':
            short_tickers = tickers[-self.config.bottom_k:]
            for ticker in short_tickers:
                if ticker not in long_tickers:  # Avoid overlap
                    weights[ticker] = -1.0 / self.config.bottom_k

        # Apply volatility scaling if enabled
        if self.config.use_vol_scaling:
            weights = self._apply_vol_scaling(weights, prices, current_date)

        return weights

    def _apply_vol_scaling(
        self,
        weights: Dict[str, float],
        prices: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Dict[str, float]:
        """Scale positions inversely to volatility."""
        idx = prices.index.get_loc(current_date)
        if idx < self.config.vol_lookback:
            return weights

        # Compute realized volatility
        returns = prices.pct_change()
        vol_window = returns.iloc[idx - self.config.vol_lookback:idx]

        scaled_weights = {}
        total_abs_weight = 0

        for ticker, weight in weights.items():
            if weight == 0 or ticker not in vol_window.columns:
                scaled_weights[ticker] = weight
                continue

            ticker_vol = vol_window[ticker].std() * np.sqrt(252)

            if ticker_vol > 0:
                vol_scale = self.config.target_vol / ticker_vol
                vol_scale = np.clip(vol_scale, 0.5, 2.0)  # Cap scaling
                scaled_weights[ticker] = weight * vol_scale
            else:
                scaled_weights[ticker] = weight

            total_abs_weight += abs(scaled_weights[ticker])

        # Normalize to maintain leverage
        if total_abs_weight > 0:
            target_leverage = sum(abs(w) for w in weights.values())
            scale = target_leverage / total_abs_weight
            scaled_weights = {t: w * scale for t, w in scaled_weights.items()}

        return scaled_weights


class TrendFollowingOverlay:
    """
    Trend-following filter that modifies ML predictions.

    Key idea: Only trade in the direction of the trend.
    If the ML model predicts positive return but trend is down, skip or reduce.
    """

    def __init__(
        self,
        trend_lookback: int = 200,
        min_trend_strength: float = 0.0
    ):
        """
        Parameters
        ----------
        trend_lookback : int
            Number of days for trend calculation (e.g., 200-day MA).
        min_trend_strength : float
            Minimum trend strength (return) to allow position.
        """
        self.trend_lookback = trend_lookback
        self.min_trend_strength = min_trend_strength

    def compute_trend_filter(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Compute trend filter for each asset.

        Returns
        -------
        filter : dict mapping ticker -> trend multiplier (0 to 1)
            1.0 = strong uptrend, allow full position
            0.5 = weak/no trend, reduce position
            0.0 = strong downtrend, avoid position
        """
        idx = prices.index.get_loc(current_date)
        if idx < self.trend_lookback:
            return {col: 1.0 for col in prices.columns}

        # Current price vs moving average
        current_prices = prices.iloc[idx]
        ma_prices = prices.iloc[idx - self.trend_lookback:idx].mean()

        # Compute trend as % above/below MA
        trend_strength = (current_prices - ma_prices) / ma_prices

        filters = {}
        for col in prices.columns:
            ts = trend_strength[col]

            if ts > self.min_trend_strength:
                # Uptrend - full weight
                filters[col] = 1.0
            elif ts > -self.min_trend_strength:
                # No clear trend - reduce weight
                filters[col] = 0.5
            else:
                # Downtrend - avoid
                filters[col] = 0.0

        return filters

    def apply_filter(
        self,
        predictions: Dict[str, float],
        prices: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Apply trend filter to ML predictions.

        Parameters
        ----------
        predictions : dict
            ML model predictions (ticker -> predicted return).
        prices : pd.DataFrame
            Price data.
        current_date : pd.Timestamp
            Current date.

        Returns
        -------
        filtered_predictions : dict
            Predictions adjusted by trend filter.
        """
        trend_filter = self.compute_trend_filter(prices, current_date)

        filtered = {}
        for ticker, pred in predictions.items():
            if ticker in trend_filter:
                filtered[ticker] = pred * trend_filter[ticker]
            else:
                filtered[ticker] = pred

        return filtered


class RelativeStrengthRanker:
    """
    Rank-based allocation instead of return prediction.

    Key insight: We don't need to predict exact returns,
    just the relative ordering of assets.
    """

    def __init__(
        self,
        lookback_windows: List[int] = None,
        weights: List[float] = None
    ):
        """
        Parameters
        ----------
        lookback_windows : list of int
            Lookback periods for momentum calculation.
            Default: [21, 63, 126] (1, 3, 6 months)
        weights : list of float
            Weights for each lookback period.
            Default: [0.4, 0.35, 0.25]
        """
        self.lookback_windows = lookback_windows or [21, 63, 126]
        self.weights = weights or [0.4, 0.35, 0.25]

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def compute_ranks(
        self,
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        exclude_tickers: List[str] = None
    ) -> pd.Series:
        """
        Compute composite relative strength ranks.

        Returns
        -------
        ranks : pd.Series
            Percentile rank (0-1) for each ticker.
            Higher = stronger relative strength.
        """
        exclude_tickers = exclude_tickers or []
        tickers = [c for c in prices.columns if c not in exclude_tickers]

        idx = prices.index.get_loc(current_date)
        max_lookback = max(self.lookback_windows)

        if idx < max_lookback:
            return pd.Series(0.5, index=tickers)

        # Compute returns for each lookback
        returns_by_window = []
        for window in self.lookback_windows:
            current = prices.iloc[idx][tickers]
            past = prices.iloc[idx - window][tickers]
            ret = (current - past) / past
            returns_by_window.append(ret)

        # Rank each window
        ranks_by_window = []
        for ret in returns_by_window:
            rank = ret.rank(pct=True)
            ranks_by_window.append(rank)

        # Weighted average of ranks
        composite_rank = pd.Series(0.0, index=tickers)
        for rank, weight in zip(ranks_by_window, self.weights):
            composite_rank += rank * weight

        return composite_rank

    def get_top_bottom_k(
        self,
        ranks: pd.Series,
        top_k: int = 3,
        bottom_k: int = 3
    ) -> Tuple[List[str], List[str]]:
        """
        Get top-k and bottom-k tickers by rank.

        Returns
        -------
        top_tickers : list
            Top k tickers (highest rank).
        bottom_tickers : list
            Bottom k tickers (lowest rank).
        """
        sorted_tickers = ranks.sort_values(ascending=False).index.tolist()

        top_tickers = sorted_tickers[:top_k]
        bottom_tickers = sorted_tickers[-bottom_k:]

        return top_tickers, bottom_tickers


class HybridStrategy:
    """
    Combines ML predictions with momentum/trend rules.

    The idea: Use ML to refine systematic momentum signals,
    not replace them entirely.
    """

    def __init__(
        self,
        ml_weight: float = 0.3,
        momentum_weight: float = 0.7,
        use_trend_filter: bool = True,
        trend_lookback: int = 200
    ):
        """
        Parameters
        ----------
        ml_weight : float
            Weight for ML predictions (0-1).
        momentum_weight : float
            Weight for momentum signals (0-1).
        use_trend_filter : bool
            Apply trend filter to final signals.
        trend_lookback : int
            Lookback for trend calculation.
        """
        self.ml_weight = ml_weight
        self.momentum_weight = momentum_weight

        self.ranker = RelativeStrengthRanker()
        self.trend_overlay = TrendFollowingOverlay(trend_lookback) if use_trend_filter else None

    def combine_signals(
        self,
        ml_predictions: Dict[str, float],
        prices: pd.DataFrame,
        current_date: pd.Timestamp,
        benchmark: str = 'SPY'
    ) -> Dict[str, float]:
        """
        Combine ML predictions with momentum signals.

        Parameters
        ----------
        ml_predictions : dict
            ML model predictions (ticker -> predicted return).
        prices : pd.DataFrame
            Price data.
        current_date : pd.Timestamp
            Current date.
        benchmark : str
            Benchmark ticker to exclude.

        Returns
        -------
        combined_signals : dict
            Combined signal strength for each ticker.
        """
        # Get momentum ranks
        ranks = self.ranker.compute_ranks(
            prices, current_date,
            exclude_tickers=[benchmark]
        )

        # Normalize ML predictions to 0-1 scale
        ml_values = np.array(list(ml_predictions.values()))
        if len(ml_values) > 0 and ml_values.std() > 0:
            ml_min, ml_max = ml_values.min(), ml_values.max()
            ml_normalized = {
                t: (v - ml_min) / (ml_max - ml_min) if ml_max > ml_min else 0.5
                for t, v in ml_predictions.items()
            }
        else:
            ml_normalized = {t: 0.5 for t in ml_predictions}

        # Combine signals
        combined = {}
        tickers = set(ml_predictions.keys()) | set(ranks.index)

        for ticker in tickers:
            ml_signal = ml_normalized.get(ticker, 0.5)
            momentum_signal = ranks.get(ticker, 0.5)

            combined_signal = (
                self.ml_weight * ml_signal +
                self.momentum_weight * momentum_signal
            )
            combined[ticker] = combined_signal

        # Apply trend filter
        if self.trend_overlay is not None:
            trend_filter = self.trend_overlay.compute_trend_filter(prices, current_date)
            combined = {
                t: s * trend_filter.get(t, 1.0)
                for t, s in combined.items()
            }

        return combined


def compute_monthly_rebalance_dates(
    prices: pd.DataFrame,
    day_of_month: int = -1
) -> pd.DatetimeIndex:
    """
    Get monthly rebalance dates.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with datetime index.
    day_of_month : int
        Day of month to rebalance (-1 = last trading day).

    Returns
    -------
    dates : pd.DatetimeIndex
        Monthly rebalance dates.
    """
    # Group by year-month
    monthly_groups = prices.groupby([prices.index.year, prices.index.month])

    dates = []
    for (year, month), group in monthly_groups:
        if len(group) == 0:
            continue

        if day_of_month == -1:
            # Last trading day of month
            dates.append(group.index[-1])
        else:
            # Specific day (or closest trading day)
            target = pd.Timestamp(year=year, month=month, day=min(day_of_month, 28))
            available = group.index[group.index >= target]
            if len(available) > 0:
                dates.append(available[0])
            elif len(group) > 0:
                dates.append(group.index[-1])

    return pd.DatetimeIndex(dates)
