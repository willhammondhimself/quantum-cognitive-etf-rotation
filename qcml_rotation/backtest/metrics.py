"""
Performance metrics for backtesting.

Includes standard risk-adjusted return metrics used in portfolio analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for backtest performance metrics."""
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    hit_rate: float
    avg_weekly_return: float
    win_rate: float
    avg_turnover: float
    n_trades: int


def sharpe_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 52
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Parameters
    ----------
    returns : array-like
        Period returns (e.g., weekly).
    risk_free_rate : float
        Annual risk-free rate.
    periods_per_year : int
        Number of periods per year (52 for weekly).

    Returns
    -------
    sharpe : float
        Annualized Sharpe ratio.
    """
    returns = np.asarray(returns)

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    # Convert annual rf to per-period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - rf_per_period
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)

    if std_excess == 0:
        return 0.0

    # Annualize
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)

    return float(sharpe)


def sortino_ratio(
    returns: Union[np.ndarray, pd.Series],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 52
) -> float:
    """
    Calculate annualized Sortino ratio (uses downside deviation).

    Parameters
    ----------
    returns : array-like
        Period returns.
    risk_free_rate : float
        Annual risk-free rate.
    periods_per_year : int
        Number of periods per year.

    Returns
    -------
    sortino : float
        Annualized Sortino ratio.
    """
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period

    # Downside deviation: std of negative excess returns only
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0

    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0:
        return 0.0

    mean_excess = np.mean(excess_returns)
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)

    return float(sortino)


def max_drawdown(equity_curve: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Parameters
    ----------
    equity_curve : array-like
        Cumulative portfolio value over time.

    Returns
    -------
    max_dd : float
        Maximum drawdown as a positive decimal (e.g., 0.20 = 20% drawdown).
    """
    equity = np.asarray(equity_curve)

    if len(equity) == 0:
        return 0.0

    # Running maximum
    running_max = np.maximum.accumulate(equity)

    # Drawdown at each point
    drawdown = (running_max - equity) / running_max

    # Handle potential division by zero
    drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)

    return float(np.max(drawdown))


def calmar_ratio(
    returns: Union[np.ndarray, pd.Series],
    equity_curve: Union[np.ndarray, pd.Series],
    periods_per_year: int = 52
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Parameters
    ----------
    returns : array-like
        Period returns.
    equity_curve : array-like
        Cumulative portfolio value.
    periods_per_year : int
        Number of periods per year.

    Returns
    -------
    calmar : float
        Calmar ratio.
    """
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    # Annualized return
    mean_return = np.mean(returns)
    ann_return = (1 + mean_return) ** periods_per_year - 1

    # Max drawdown
    max_dd = max_drawdown(equity_curve)

    if max_dd == 0:
        return np.inf if ann_return > 0 else 0.0

    return float(ann_return / max_dd)


def hit_rate(
    predictions: np.ndarray,
    actuals: np.ndarray,
    top_k: int = 3
) -> float:
    """
    Calculate hit rate: fraction of top-K picks that outperformed.

    Parameters
    ----------
    predictions : array of shape (n_weeks, n_etfs)
        Predicted excess returns.
    actuals : array of shape (n_weeks, n_etfs)
        Actual excess returns.
    top_k : int
        Number of top picks per week.

    Returns
    -------
    hit_rate : float
        Fraction of correct predictions.
    """
    n_weeks = predictions.shape[0]
    hits = 0
    total = 0

    for week in range(n_weeks):
        pred_ranks = np.argsort(predictions[week])[::-1]  # descending
        top_picks = pred_ranks[:top_k]

        for pick in top_picks:
            if actuals[week, pick] > 0:  # Beat SPY (excess return > 0)
                hits += 1
            total += 1

    return hits / total if total > 0 else 0.0


def compute_returns_from_equity(equity_curve: np.ndarray) -> np.ndarray:
    """Compute period returns from equity curve."""
    equity = np.asarray(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    return returns


def compute_equity_from_returns(
    returns: np.ndarray,
    initial_value: float = 1.0
) -> np.ndarray:
    """Compute equity curve from returns."""
    returns = np.asarray(returns)
    equity = initial_value * np.cumprod(1 + returns)
    equity = np.insert(equity, 0, initial_value)
    return equity


def compute_all_metrics(
    returns: np.ndarray,
    equity_curve: Optional[np.ndarray] = None,
    predictions: Optional[np.ndarray] = None,
    actuals: Optional[np.ndarray] = None,
    turnovers: Optional[np.ndarray] = None,
    top_k: int = 3,
    periods_per_year: int = 52,
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """
    Compute all performance metrics.

    Parameters
    ----------
    returns : array
        Period returns.
    equity_curve : array, optional
        Cumulative portfolio value. Computed from returns if not provided.
    predictions : array, optional
        Predicted values for hit rate calculation.
    actuals : array, optional
        Actual values for hit rate calculation.
    turnovers : array, optional
        Turnover per period.
    top_k : int
        Number of top picks for hit rate.
    periods_per_year : int
        Number of periods per year.
    risk_free_rate : float
        Annual risk-free rate.

    Returns
    -------
    metrics : PerformanceMetrics
    """
    returns = np.asarray(returns)

    if equity_curve is None:
        equity_curve = compute_equity_from_returns(returns)

    # Basic return stats
    total_ret = equity_curve[-1] / equity_curve[0] - 1
    mean_ret = np.mean(returns)
    ann_ret = (1 + mean_ret) ** periods_per_year - 1
    ann_vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year)

    # Risk-adjusted
    sharpe = sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = sortino_ratio(returns, risk_free_rate, periods_per_year)
    max_dd = max_drawdown(equity_curve)
    calmar = calmar_ratio(returns, equity_curve, periods_per_year)

    # Hit rate
    hr = 0.0
    if predictions is not None and actuals is not None:
        hr = hit_rate(predictions, actuals, top_k)

    # Win rate
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0.0

    # Turnover
    avg_turnover = np.mean(turnovers) if turnovers is not None else 0.0
    n_trades = len(turnovers) if turnovers is not None else 0

    return PerformanceMetrics(
        total_return=float(total_ret),
        annualized_return=float(ann_ret),
        annualized_volatility=float(ann_vol),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown=float(max_dd),
        calmar_ratio=float(calmar),
        hit_rate=float(hr),
        avg_weekly_return=float(mean_ret),
        win_rate=float(win_rate),
        avg_turnover=float(avg_turnover),
        n_trades=n_trades
    )


def metrics_to_dict(metrics: PerformanceMetrics) -> Dict:
    """Convert PerformanceMetrics to dictionary."""
    return {
        "total_return": metrics.total_return,
        "annualized_return": metrics.annualized_return,
        "annualized_volatility": metrics.annualized_volatility,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "max_drawdown": metrics.max_drawdown,
        "calmar_ratio": metrics.calmar_ratio,
        "hit_rate": metrics.hit_rate,
        "avg_weekly_return": metrics.avg_weekly_return,
        "win_rate": metrics.win_rate,
        "avg_turnover": metrics.avg_turnover,
        "n_trades": metrics.n_trades
    }
