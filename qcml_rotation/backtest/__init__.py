"""Backtesting engine and performance metrics."""

from .engine import BacktestEngine, BacktestResult, compare_results, results_to_dict
from .portfolio import PortfolioConstructor, Portfolio, PortfolioConfig
from .metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, hit_rate,
    compute_all_metrics, PerformanceMetrics, metrics_to_dict
)
