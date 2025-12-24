"""Backtesting engine and performance metrics."""

from .engine import BacktestEngine, BacktestResult, compare_results, results_to_dict
from .portfolio import PortfolioConstructor, Portfolio, PortfolioConfig
from .metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, hit_rate,
    compute_all_metrics, PerformanceMetrics, metrics_to_dict,
    # Significance testing
    bootstrap_sharpe_ci, sharpe_p_value, permutation_test,
    compute_significance, SignificanceResult, significance_to_dict
)
from .walk_forward import (
    WalkForwardValidator, WalkForwardResult, WalkForwardFold,
    walk_forward_result_to_dict, compare_walk_forward_results
)
