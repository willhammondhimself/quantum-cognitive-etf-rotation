"""
Tests for backtest engine module.
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qcml_rotation.backtest.engine import BacktestEngine, BacktestResult
from qcml_rotation.backtest.portfolio import PortfolioConfig, Portfolio, Position
from qcml_rotation.backtest.metrics import PerformanceMetrics


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_result_creation(self):
        """Test BacktestResult can be created."""
        mock_metrics = PerformanceMetrics(
            total_return=0.03,
            annualized_return=0.15,
            annualized_volatility=0.10,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=-0.05,
            calmar_ratio=3.0,
            hit_rate=0.55,
            avg_weekly_return=0.005,
            win_rate=0.6,
            avg_turnover=0.5,
            n_trades=100
        )

        result = BacktestResult(
            model_name='test_model',
            returns=np.array([0.01, 0.02]),
            equity_curve=np.array([1.0, 1.01, 1.0302]),
            turnovers=np.array([1.0, 0.5]),
            portfolios=[],
            metrics=mock_metrics,
            predictions=np.array([[0.01, 0.02], [0.01, 0.02]]),
            actuals=np.array([[0.01, 0.015], [0.02, 0.01]]),
            dates=[pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-08')]
        )

        assert len(result.returns) == 2
        assert len(result.equity_curve) == 3
        assert result.metrics.sharpe_ratio == 1.5
        assert result.model_name == 'test_model'


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    @pytest.fixture
    def test_data(self):
        """Create test data for backtest."""
        np.random.seed(42)

        # Create 100 days of synthetic prices
        dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
        tickers = ['SPY', 'XLK', 'XLF', 'XLE', 'XLY', 'XLP']

        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(np.random.randn(100, len(tickers)) * 0.01, axis=0)),
            index=dates,
            columns=tickers
        )

        # Create weekly test dates
        test_dates = pd.date_range(start='2020-02-01', periods=10, freq='W-FRI')
        test_dates = [d for d in test_dates if d <= dates[-1]]

        return {
            'prices': prices,
            'test_dates': test_dates,
            'tickers': [t for t in tickers if t != 'SPY'],
        }

    @pytest.fixture
    def mock_model(self):
        """Create mock model with predict method."""
        class MockModel:
            def predict(self, X):
                np.random.seed(42)
                return np.random.randn(X.shape[0]) * 0.01
        return MockModel()

    def test_engine_initialization(self, test_data):
        """Test engine can be initialized."""
        config = PortfolioConfig(top_k=2, strategy='long_short')
        engine = BacktestEngine(
            prices=test_data['prices'],
            test_dates=test_data['test_dates'],
            tickers=test_data['tickers'],
            portfolio_config=config
        )
        assert engine.portfolio_config.top_k == 2

    def test_engine_with_default_config(self, test_data):
        """Test engine with default portfolio config."""
        engine = BacktestEngine(
            prices=test_data['prices'],
            test_dates=test_data['test_dates'],
            tickers=test_data['tickers']
        )
        assert engine.portfolio_config is not None

    def test_engine_tickers_stored(self, test_data):
        """Test engine stores tickers correctly."""
        engine = BacktestEngine(
            prices=test_data['prices'],
            test_dates=test_data['test_dates'],
            tickers=test_data['tickers']
        )
        assert engine.tickers == test_data['tickers']


class TestPortfolioIntegration:
    """Tests for portfolio integration with engine."""

    def test_portfolio_dates_sorted(self):
        """Test that test dates are sorted."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
        tickers = ['SPY', 'XLK', 'XLF']

        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(np.random.randn(100, len(tickers)) * 0.01, axis=0)),
            index=dates,
            columns=tickers
        )

        # Unsorted dates
        test_dates = [dates[30], dates[20], dates[40]]

        engine = BacktestEngine(
            prices=prices,
            test_dates=test_dates,
            tickers=['XLK', 'XLF']
        )

        # Engine should sort dates
        assert engine.test_dates == sorted(test_dates)


class TestCompareResults:
    """Tests for results comparison functionality."""

    def test_comparison_dataframe(self):
        """Test that comparison creates valid DataFrame."""
        mock_metrics_a = PerformanceMetrics(
            total_return=0.0302,
            annualized_return=0.15,
            annualized_volatility=0.10,
            sharpe_ratio=1.0,
            sortino_ratio=1.5,
            max_drawdown=-0.05,
            calmar_ratio=3.0,
            hit_rate=0.55,
            avg_weekly_return=0.005,
            win_rate=0.6,
            avg_turnover=0.5,
            n_trades=50
        )

        mock_metrics_b = PerformanceMetrics(
            total_return=0.0302,
            annualized_return=0.12,
            annualized_volatility=0.15,
            sharpe_ratio=0.8,
            sortino_ratio=1.2,
            max_drawdown=-0.08,
            calmar_ratio=1.5,
            hit_rate=0.50,
            avg_weekly_return=0.004,
            win_rate=0.55,
            avg_turnover=0.6,
            n_trades=60
        )

        results = {
            'model_a': BacktestResult(
                model_name='model_a',
                returns=np.array([0.01, 0.02]),
                equity_curve=np.array([1.0, 1.01, 1.0302]),
                turnovers=np.array([1.0, 0.5]),
                portfolios=[],
                metrics=mock_metrics_a,
                predictions=np.array([[0.01], [0.02]]),
                actuals=np.array([[0.01], [0.015]]),
                dates=[pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-08')]
            ),
            'model_b': BacktestResult(
                model_name='model_b',
                returns=np.array([0.02, 0.01]),
                equity_curve=np.array([1.0, 1.02, 1.0302]),
                turnovers=np.array([1.0, 0.3]),
                portfolios=[],
                metrics=mock_metrics_b,
                predictions=np.array([[0.02], [0.01]]),
                actuals=np.array([[0.015], [0.02]]),
                dates=[pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-08')]
            ),
        }

        # Create comparison manually
        from qcml_rotation.backtest.metrics import metrics_to_dict
        comparison = pd.DataFrame([
            {'model': name, **metrics_to_dict(r.metrics)}
            for name, r in results.items()
        ]).set_index('model')

        assert len(comparison) == 2
        assert 'model_a' in comparison.index
        assert 'model_b' in comparison.index
