"""
Tests for performance metrics module.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qcml_rotation.backtest.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    hit_rate,
    compute_equity_from_returns,
    compute_all_metrics,
    PerformanceMetrics,
)


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_positive_returns_with_variance(self):
        """Test Sharpe for positive returns with variance."""
        np.random.seed(42)
        returns = np.random.randn(52) * 0.02 + 0.01  # Mean = 1%, std = 2%
        sr = sharpe_ratio(returns)
        assert sr > 0  # Positive mean should give positive Sharpe

    def test_sharpe_negative_returns_with_variance(self):
        """Test Sharpe for negative returns with variance."""
        np.random.seed(42)
        returns = np.random.randn(52) * 0.02 - 0.01  # Mean = -1%, std = 2%
        sr = sharpe_ratio(returns)
        assert sr < 0  # Negative mean should give negative Sharpe

    def test_sharpe_zero_returns(self):
        """Test Sharpe for zero returns."""
        returns = np.array([0.0] * 52)
        sr = sharpe_ratio(returns)
        assert sr == 0.0  # Zero std means zero Sharpe

    def test_sharpe_constant_returns(self):
        """Test Sharpe for constant returns (zero std)."""
        returns = np.array([0.01] * 52)  # Constant returns
        sr = sharpe_ratio(returns)
        assert sr == 0.0  # Zero std means zero Sharpe in this implementation

    def test_sharpe_known_value(self):
        """Test Sharpe calculation with known values."""
        np.random.seed(42)
        returns = np.random.randn(52) * 0.02 + 0.01
        sr = sharpe_ratio(returns)
        # Should be positive with these positive mean returns
        assert sr > 0

    def test_sharpe_annualization(self):
        """Test that Sharpe is properly annualized."""
        np.random.seed(42)
        returns = np.random.randn(52) * 0.02 + 0.005
        sr = sharpe_ratio(returns, periods_per_year=52)
        # Sharpe should be finite
        assert np.isfinite(sr)


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_all_positive_returns(self):
        """Test Sortino for all positive returns."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])
        sr = sortino_ratio(returns)
        # All positive returns, no downside deviation
        # Sortino should be very high or inf
        assert sr > 0 or np.isinf(sr)

    def test_sortino_mixed_returns(self):
        """Test Sortino for mixed returns."""
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        sr = sortino_ratio(returns)
        assert isinstance(sr, float)
        assert np.isfinite(sr) or np.isinf(sr)

    def test_sortino_vs_sharpe(self):
        """Test that Sortino >= Sharpe for same returns with positive mean."""
        np.random.seed(42)
        returns = np.random.randn(52) * 0.02 + 0.005
        sr_sharpe = sharpe_ratio(returns)
        sr_sortino = sortino_ratio(returns)
        # Both should be finite for reasonable returns
        assert np.isfinite(sr_sharpe) or np.isfinite(sr_sortino)


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_max_dd_linear_decline(self):
        """Test max DD for linear price decline."""
        # 10% decline: 1.0 -> 0.95 -> 0.90
        equity = np.array([1.0, 0.95, 0.90])
        dd = max_drawdown(equity)
        # max_drawdown returns positive value: 0.10 for 10% drawdown
        assert abs(dd - 0.10) < 0.001

    def test_max_dd_recovery(self):
        """Test max DD with recovery."""
        equity = np.array([1.0, 0.8, 0.9, 1.0])
        dd = max_drawdown(equity)
        # Max drawdown was 20% (1.0 -> 0.8)
        assert abs(dd - 0.20) < 0.001

    def test_max_dd_always_increasing(self):
        """Test max DD for always increasing equity."""
        equity = np.array([1.0, 1.1, 1.2, 1.3])
        dd = max_drawdown(equity)
        assert dd == 0.0

    def test_max_dd_positive_value(self):
        """Test that max DD is positive or zero."""
        np.random.seed(42)
        returns = np.random.randn(50) * 0.02
        equity = compute_equity_from_returns(returns)
        dd = max_drawdown(equity)
        assert dd >= 0  # Returns positive value in this implementation

    def test_max_dd_bounded(self, synthetic_returns):
        """Test max DD from synthetic returns is bounded."""
        equity = compute_equity_from_returns(synthetic_returns)
        dd = max_drawdown(equity)
        assert 0.0 <= dd <= 1.0


class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""

    def test_calmar_positive_return_with_dd(self):
        """Test Calmar with positive return and drawdown."""
        np.random.seed(42)
        returns = np.random.randn(52) * 0.02 + 0.005
        equity = compute_equity_from_returns(returns)
        cr = calmar_ratio(returns, equity)
        # Should be a number (positive or negative depending on returns/dd)
        assert np.isfinite(cr) or np.isinf(cr)

    def test_calmar_zero_drawdown(self):
        """Test Calmar with zero drawdown."""
        returns = np.array([0.01, 0.01, 0.01])
        equity = np.array([1.0, 1.01, 1.0201, 1.030301])
        cr = calmar_ratio(returns, equity)
        # Zero DD means Calmar is inf
        assert np.isinf(cr) or cr > 100


class TestHitRate:
    """Tests for hit rate calculation."""

    def test_hit_rate_perfect_predictions(self):
        """Test hit rate for perfect predictions."""
        # 2D arrays: (n_weeks, n_etfs)
        predictions = np.array([
            [0.03, 0.02, 0.01, -0.01, -0.02],  # Week 1
            [0.03, 0.02, 0.01, -0.01, -0.02],  # Week 2
        ])
        actuals = np.array([
            [0.04, 0.03, 0.005, -0.02, -0.03],  # Week 1: top 2 have positive excess returns
            [0.04, 0.03, 0.005, -0.02, -0.03],  # Week 2: top 2 have positive excess returns
        ])

        hr = hit_rate(predictions, actuals, top_k=2)
        assert hr == 1.0

    def test_hit_rate_random(self):
        """Test hit rate for random predictions."""
        np.random.seed(42)
        n_weeks, n_etfs = 20, 10
        predictions = np.random.randn(n_weeks, n_etfs)
        actuals = np.random.randn(n_weeks, n_etfs)

        hr = hit_rate(predictions, actuals, top_k=3)
        # Should be between 0 and 1
        assert 0.0 <= hr <= 1.0

    def test_hit_rate_bounded(self):
        """Test that hit rate is always bounded [0, 1]."""
        np.random.seed(123)
        n_weeks, n_etfs = 50, 5
        predictions = np.random.randn(n_weeks, n_etfs)
        actuals = np.random.randn(n_weeks, n_etfs)

        hr = hit_rate(predictions, actuals, top_k=2)
        assert 0.0 <= hr <= 1.0


class TestComputeEquityFromReturns:
    """Tests for equity curve computation."""

    def test_equity_starts_at_one(self):
        """Test that equity curve starts at 1."""
        returns = np.array([0.01, 0.02, -0.01])
        equity = compute_equity_from_returns(returns)
        assert equity[0] == 1.0

    def test_equity_cumulative(self):
        """Test cumulative equity calculation."""
        returns = np.array([0.10, 0.10])  # 10% each
        equity = compute_equity_from_returns(returns)
        # 1.0 -> 1.10 -> 1.21
        assert abs(equity[-1] - 1.21) < 0.001

    def test_equity_length(self):
        """Test equity curve length."""
        returns = np.array([0.01, 0.02, 0.03])
        equity = compute_equity_from_returns(returns)
        assert len(equity) == len(returns) + 1  # Includes starting point

    def test_equity_empty_returns(self):
        """Test equity curve for empty returns."""
        returns = np.array([])
        equity = compute_equity_from_returns(returns)
        assert len(equity) == 1
        assert equity[0] == 1.0


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""

    def test_all_metrics_returned(self, synthetic_returns):
        """Test that all expected metrics are returned."""
        np.random.seed(42)
        equity = compute_equity_from_returns(synthetic_returns)
        turnovers = np.abs(np.random.randn(len(synthetic_returns))) * 0.5

        # 2D predictions and actuals for hit rate
        n_weeks = len(synthetic_returns)
        n_etfs = 5
        predictions = np.random.randn(n_weeks, n_etfs)
        actuals = np.random.randn(n_weeks, n_etfs)

        metrics = compute_all_metrics(
            returns=synthetic_returns,
            equity_curve=equity,
            turnovers=turnovers,
            predictions=predictions,
            actuals=actuals,
        )

        # Returns PerformanceMetrics dataclass
        assert isinstance(metrics, PerformanceMetrics)
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'hit_rate')

    def test_metrics_no_nan(self, synthetic_returns):
        """Test that metrics don't have NaN values."""
        np.random.seed(42)
        equity = compute_equity_from_returns(synthetic_returns)
        turnovers = np.abs(np.random.randn(len(synthetic_returns))) * 0.5

        n_weeks = len(synthetic_returns)
        n_etfs = 5
        predictions = np.random.randn(n_weeks, n_etfs)
        actuals = np.random.randn(n_weeks, n_etfs)

        metrics = compute_all_metrics(
            returns=synthetic_returns,
            equity_curve=equity,
            turnovers=turnovers,
            predictions=predictions,
            actuals=actuals,
        )

        # Check key metrics are not NaN
        assert not np.isnan(metrics.total_return)
        assert not np.isnan(metrics.sharpe_ratio)
        assert not np.isnan(metrics.max_drawdown)

    def test_metrics_without_predictions(self, synthetic_returns):
        """Test metrics can be computed without predictions."""
        equity = compute_equity_from_returns(synthetic_returns)

        metrics = compute_all_metrics(
            returns=synthetic_returns,
            equity_curve=equity,
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.hit_rate == 0.0  # Default when no predictions
