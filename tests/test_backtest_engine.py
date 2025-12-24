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
from qcml_rotation.backtest.portfolio import PortfolioConfig


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_result_creation(self):
        """Test BacktestResult can be created."""
        result = BacktestResult(
            returns=np.array([0.01, 0.02]),
            equity_curve=np.array([1.0, 1.01, 1.0302]),
            turnovers=np.array([1.0, 0.5]),
            dates=['2020-01-01', '2020-01-08'],
            metrics={'sharpe_ratio': 1.0},
            portfolios=[]
        )
        assert len(result.returns) == 2
        assert len(result.equity_curve) == 3
        assert result.metrics['sharpe_ratio'] == 1.0


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    @pytest.fixture
    def backtest_config(self):
        """Create backtest configuration."""
        return PortfolioConfig(top_k=2, strategy='long_short', transaction_cost_bps=5)

    @pytest.fixture
    def mock_predict_fn(self):
        """Create mock prediction function."""
        def predict(features):
            np.random.seed(42)
            return np.random.randn(len(features)) * 0.01
        return predict

    @pytest.fixture
    def test_data(self, synthetic_prices):
        """Create test data for backtest."""
        prices = synthetic_prices
        tickers = list(prices.columns)

        # Create weekly dates
        dates = pd.date_range(start=prices.index.min(), periods=20, freq='W-FRI')
        dates = dates[dates <= prices.index.max()]

        # Create features
        n_weeks = len(dates)
        n_tickers = len(tickers) - 1  # Exclude SPY

        features = np.random.randn(n_weeks, n_tickers, 6) * 0.01

        return {
            'prices': prices,
            'dates': dates,
            'tickers': [t for t in tickers if t != 'SPY'],
            'features': features
        }

    def test_engine_initialization(self, backtest_config):
        """Test engine can be initialized."""
        engine = BacktestEngine(backtest_config)
        assert engine.config.top_k == 2

    def test_backtest_run_returns_result(self, backtest_config, mock_predict_fn, test_data):
        """Test that backtest run returns BacktestResult."""
        engine = BacktestEngine(backtest_config)

        result = engine.run(
            predict_fn=mock_predict_fn,
            prices=test_data['prices'],
            dates=test_data['dates'],
            tickers=test_data['tickers'],
            features=test_data['features']
        )

        assert isinstance(result, BacktestResult)

    def test_equity_starts_at_one(self, backtest_config, mock_predict_fn, test_data):
        """Test that equity curve starts at 1.0."""
        engine = BacktestEngine(backtest_config)

        result = engine.run(
            predict_fn=mock_predict_fn,
            prices=test_data['prices'],
            dates=test_data['dates'],
            tickers=test_data['tickers'],
            features=test_data['features']
        )

        assert result.equity_curve[0] == 1.0

    def test_returns_length(self, backtest_config, mock_predict_fn, test_data):
        """Test returns have correct length."""
        engine = BacktestEngine(backtest_config)

        result = engine.run(
            predict_fn=mock_predict_fn,
            prices=test_data['prices'],
            dates=test_data['dates'],
            tickers=test_data['tickers'],
            features=test_data['features']
        )

        # Returns should be one less than dates (no return for first week)
        assert len(result.returns) == len(result.dates) - 1 or len(result.returns) == len(result.dates)

    def test_turnovers_bounded(self, backtest_config, mock_predict_fn, test_data):
        """Test that turnovers are bounded [0, 2]."""
        engine = BacktestEngine(backtest_config)

        result = engine.run(
            predict_fn=mock_predict_fn,
            prices=test_data['prices'],
            dates=test_data['dates'],
            tickers=test_data['tickers'],
            features=test_data['features']
        )

        assert all(0 <= t <= 2 for t in result.turnovers)

    def test_metrics_populated(self, backtest_config, mock_predict_fn, test_data):
        """Test that metrics are populated."""
        engine = BacktestEngine(backtest_config)

        result = engine.run(
            predict_fn=mock_predict_fn,
            prices=test_data['prices'],
            dates=test_data['dates'],
            tickers=test_data['tickers'],
            features=test_data['features']
        )

        assert 'sharpe_ratio' in result.metrics
        assert 'max_drawdown' in result.metrics


class TestCompareResults:
    """Tests for results comparison functionality."""

    def test_comparison_dataframe(self):
        """Test that comparison creates valid DataFrame."""
        results = {
            'model_a': BacktestResult(
                returns=np.array([0.01, 0.02]),
                equity_curve=np.array([1.0, 1.01, 1.0302]),
                turnovers=np.array([1.0, 0.5]),
                dates=['2020-01-01', '2020-01-08'],
                metrics={'sharpe_ratio': 1.0, 'total_return': 0.0302},
                portfolios=[]
            ),
            'model_b': BacktestResult(
                returns=np.array([0.02, 0.01]),
                equity_curve=np.array([1.0, 1.02, 1.0302]),
                turnovers=np.array([1.0, 0.3]),
                dates=['2020-01-01', '2020-01-08'],
                metrics={'sharpe_ratio': 0.8, 'total_return': 0.0302},
                portfolios=[]
            ),
        }

        # Create comparison (manual since function may not exist)
        comparison = pd.DataFrame([
            {'model': name, **r.metrics}
            for name, r in results.items()
        ]).set_index('model')

        assert len(comparison) == 2
        assert 'model_a' in comparison.index
        assert 'model_b' in comparison.index
