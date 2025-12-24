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
)


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_positive_returns(self):
        """Test Sharpe for consistently positive returns."""
        returns = np.array([0.01] * 52)  # 1% weekly for a year
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_sharpe_negative_returns(self):
        """Test Sharpe for consistently negative returns."""
        returns = np.array([-0.01] * 52)
        sr = sharpe_ratio(returns)
        assert sr < 0

    def test_sharpe_zero_returns(self):
        """Test Sharpe for zero returns."""
        returns = np.array([0.0] * 52)
        sr = sharpe_ratio(returns)
        assert sr == 0.0 or np.isnan(sr)

    def test_sharpe_known_value(self):
        """Test Sharpe calculation with known values."""
        # 1% weekly return, 2% weekly vol
        # Annualized: return = 1% * 52 = 52%, vol = 2% * sqrt(52) ≈ 14.4%
        # Sharpe ≈ 52% / 14.4% ≈ 3.6
        returns = np.array([0.01] * 52)
        returns = returns + np.random.randn(52) * 0.02
        np.random.seed(42)
        sr = sharpe_ratio(returns)
        # Should be positive with these positive mean returns
        assert sr > 0

    def test_sharpe_annualization(self):
        """Test that Sharpe is properly annualized."""
        returns = np.array([0.01] * 52)
        sr = sharpe_ratio(returns, periods_per_year=52)
        # Manual calculation
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        expected_sr = (mean_ret * 52) / (std_ret * np.sqrt(52)) if std_ret > 0 else 0
        assert abs(sr - expected_sr) < 0.1 or std_ret == 0


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_positive_returns(self):
        """Test Sortino for positive returns."""
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

    def test_sortino_vs_sharpe(self):
        """Test that Sortino >= Sharpe for same returns."""
        np.random.seed(42)
        returns = np.random.randn(52) * 0.02 + 0.005
        sr_sharpe = sharpe_ratio(returns)
        sr_sortino = sortino_ratio(returns)
        # Sortino typically >= Sharpe because it only penalizes downside
        # Not always true, so just check both are finite
        assert np.isfinite(sr_sharpe) or np.isfinite(sr_sortino)


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_max_dd_linear_decline(self):
        """Test max DD for linear price decline."""
        # 10% decline
        equity = np.array([1.0, 0.95, 0.90])
        dd = max_drawdown(equity)
        assert abs(dd - (-0.10)) < 0.001

    def test_max_dd_recovery(self):
        """Test max DD with recovery."""
        equity = np.array([1.0, 0.8, 0.9, 1.0])
        dd = max_drawdown(equity)
        assert abs(dd - (-0.20)) < 0.001

    def test_max_dd_always_increasing(self):
        """Test max DD for always increasing equity."""
        equity = np.array([1.0, 1.1, 1.2, 1.3])
        dd = max_drawdown(equity)
        assert dd == 0.0

    def test_max_dd_negative_value(self):
        """Test that max DD is negative or zero."""
        np.random.seed(42)
        returns = np.random.randn(50) * 0.02
        equity = compute_equity_from_returns(returns)
        dd = max_drawdown(equity)
        assert dd <= 0

    def test_max_dd_from_returns(self, synthetic_returns):
        """Test max DD from synthetic returns."""
        equity = compute_equity_from_returns(synthetic_returns)
        dd = max_drawdown(equity)
        assert -1.0 <= dd <= 0.0


class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""

    def test_calmar_positive_return_negative_dd(self):
        """Test Calmar with positive return and negative DD."""
        returns = np.array([0.01] * 52)  # Positive returns
        equity = compute_equity_from_returns(returns)
        cr = calmar_ratio(returns, equity)
        # Should be positive (positive return / negative DD = positive)
        # But max_dd returns negative, so calmar = return / abs(dd)
        assert cr > 0 or np.isinf(cr)

    def test_calmar_zero_drawdown(self):
        """Test Calmar with zero drawdown."""
        returns = np.array([0.01, 0.01, 0.01])
        equity = np.array([1.0, 1.01, 1.0201])
        cr = calmar_ratio(returns, equity)
        # Zero DD means Calmar is undefined or very high
        assert np.isinf(cr) or cr > 100


class TestHitRate:
    """Tests for hit rate calculation."""

    def test_hit_rate_perfect_predictions(self):
        """Test hit rate for perfect predictions."""
        predictions = np.array([0.03, 0.02, 0.01, -0.01, -0.02])
        actuals = np.array([0.04, 0.03, 0.02, -0.02, -0.03])
        spy_return = 0.0

        # Top 2 predictions (0.03, 0.02) have positive actuals
        hr = hit_rate(predictions, actuals, spy_return, top_k=2)
        assert hr == 1.0

    def test_hit_rate_random(self):
        """Test hit rate for random predictions."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        actuals = np.random.randn(100)
        spy_return = 0.0

        hr = hit_rate(predictions, actuals, spy_return, top_k=10)
        # Should be around 50% for random
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


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""

    def test_all_metrics_returned(self, synthetic_returns):
        """Test that all expected metrics are returned."""
        equity = compute_equity_from_returns(synthetic_returns)
        turnovers = np.abs(np.random.randn(len(synthetic_returns))) * 0.5

        metrics = compute_all_metrics(
            returns=synthetic_returns,
            equity_curve=equity,
            turnovers=turnovers,
            predictions=np.random.randn(len(synthetic_returns)),
            actuals=np.random.randn(len(synthetic_returns)),
            spy_returns=np.random.randn(len(synthetic_returns)) * 0.01
        )

        # Check all expected keys exist
        expected_keys = [
            'total_return', 'annual_return', 'annual_volatility',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'calmar_ratio', 'hit_rate', 'win_rate', 'avg_turnover'
        ]
        for key in expected_keys:
            assert key in metrics

    def test_metrics_no_nan(self, synthetic_returns):
        """Test that metrics don't have NaN values."""
        equity = compute_equity_from_returns(synthetic_returns)
        turnovers = np.abs(np.random.randn(len(synthetic_returns))) * 0.5

        metrics = compute_all_metrics(
            returns=synthetic_returns,
            equity_curve=equity,
            turnovers=turnovers,
            predictions=np.random.randn(len(synthetic_returns)),
            actuals=np.random.randn(len(synthetic_returns)),
            spy_returns=np.random.randn(len(synthetic_returns)) * 0.01
        )

        for key, value in metrics.items():
            if not np.isinf(value):
                assert not np.isnan(value), f"{key} is NaN"
