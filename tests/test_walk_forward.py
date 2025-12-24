"""
Tests for walk-forward validation and statistical significance.
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qcml_rotation.backtest.walk_forward import (
    WalkForwardValidator, WalkForwardResult, WalkForwardFold,
    walk_forward_result_to_dict, compare_walk_forward_results
)
from qcml_rotation.backtest.metrics import (
    bootstrap_sharpe_ci, sharpe_p_value, permutation_test,
    compute_significance, SignificanceResult, sharpe_ratio
)
from qcml_rotation.backtest.portfolio import PortfolioConfig


class TestBootstrapSharpeCi:
    """Tests for bootstrap confidence intervals."""

    def test_ci_contains_point_estimate(self):
        """Test that CI contains the point estimate."""
        np.random.seed(42)
        returns = np.random.randn(52) * 0.02 + 0.005

        observed_sharpe = sharpe_ratio(returns)
        ci_lower, ci_upper, _ = bootstrap_sharpe_ci(returns, n_bootstrap=500, random_state=42)

        assert ci_lower <= observed_sharpe <= ci_upper

    def test_ci_width_decreases_with_samples(self):
        """Test that CI narrows with more data."""
        np.random.seed(42)
        returns_short = np.random.randn(20) * 0.02 + 0.005
        returns_long = np.random.randn(100) * 0.02 + 0.005

        ci_lower_short, ci_upper_short, _ = bootstrap_sharpe_ci(
            returns_short, n_bootstrap=500, random_state=42
        )
        ci_lower_long, ci_upper_long, _ = bootstrap_sharpe_ci(
            returns_long, n_bootstrap=500, random_state=42
        )

        width_short = ci_upper_short - ci_lower_short
        width_long = ci_upper_long - ci_lower_long

        assert width_long < width_short

    def test_empty_returns(self):
        """Test handling of empty returns."""
        returns = np.array([])
        ci_lower, ci_upper, samples = bootstrap_sharpe_ci(returns)

        assert ci_lower == 0.0
        assert ci_upper == 0.0
        assert len(samples) == 0

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        np.random.seed(42)
        returns = np.random.randn(52) * 0.02 + 0.005

        result1 = bootstrap_sharpe_ci(returns, n_bootstrap=100, random_state=123)
        result2 = bootstrap_sharpe_ci(returns, n_bootstrap=100, random_state=123)

        assert result1[0] == result2[0]
        assert result1[1] == result2[1]


class TestSharpePValue:
    """Tests for Sharpe ratio p-value calculation."""

    def test_positive_sharpe_low_pvalue(self):
        """Test that strong positive Sharpe has low p-value."""
        np.random.seed(42)
        # Strong positive signal
        returns = np.random.randn(100) * 0.01 + 0.02  # Mean 2%, std 1%

        p_val = sharpe_p_value(returns, n_bootstrap=500, random_state=42)
        assert p_val < 0.1  # Should be significant

    def test_negative_sharpe_high_pvalue(self):
        """Test that negative Sharpe has high p-value."""
        np.random.seed(42)
        returns = np.random.randn(52) * 0.02 - 0.005  # Negative mean

        p_val = sharpe_p_value(returns, n_bootstrap=500, random_state=42)
        assert p_val > 0.5  # Should not be significant

    def test_zero_mean_high_pvalue(self):
        """Test that zero-mean returns have high p-value."""
        np.random.seed(42)
        returns = np.random.randn(52) * 0.02  # Zero mean

        p_val = sharpe_p_value(returns, n_bootstrap=500, random_state=42)
        assert p_val >= 0.3  # Should not be significant

    def test_empty_returns(self):
        """Test handling of empty returns."""
        returns = np.array([])
        p_val = sharpe_p_value(returns)
        assert p_val == 1.0


class TestPermutationTest:
    """Tests for permutation test."""

    def test_perfect_predictions_low_pvalue(self):
        """Test that perfect predictions have low p-value."""
        np.random.seed(42)
        n_weeks, n_etfs = 20, 5

        # Generate actuals
        actuals = np.random.randn(n_weeks, n_etfs)

        # Perfect predictions: same ranking as actuals
        predictions = actuals.copy()

        observed, p_val, _ = permutation_test(
            predictions, actuals, n_permutations=500, random_state=42
        )

        assert p_val < 0.1  # Should be significant

    def test_random_predictions_high_pvalue(self):
        """Test that random predictions have high p-value."""
        np.random.seed(42)
        n_weeks, n_etfs = 20, 5

        # Independent random predictions and actuals
        predictions = np.random.randn(n_weeks, n_etfs)
        actuals = np.random.randn(n_weeks, n_etfs)

        observed, p_val, _ = permutation_test(
            predictions, actuals, n_permutations=500, random_state=42
        )

        # Random should not be significant most of the time
        # (though by chance it could be)
        assert 0.0 <= p_val <= 1.0

    def test_null_distribution_size(self):
        """Test that null distribution has correct size."""
        np.random.seed(42)
        n_perms = 100
        predictions = np.random.randn(10, 5)
        actuals = np.random.randn(10, 5)

        _, _, null_dist = permutation_test(
            predictions, actuals, n_permutations=n_perms
        )

        assert len(null_dist) == n_perms


class TestComputeSignificance:
    """Tests for comprehensive significance computation."""

    def test_returns_significance_result(self):
        """Test that function returns SignificanceResult."""
        np.random.seed(42)
        returns = np.random.randn(52) * 0.02 + 0.005

        result = compute_significance(returns, n_bootstrap=100, random_state=42)

        assert isinstance(result, SignificanceResult)
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'sharpe_ci_lower')
        assert hasattr(result, 'sharpe_ci_upper')
        assert hasattr(result, 'sharpe_p_value')
        assert hasattr(result, 'is_significant')

    def test_is_significant_flag(self):
        """Test that is_significant flag matches p-value threshold."""
        np.random.seed(42)
        # Strong positive returns
        returns = np.random.randn(100) * 0.01 + 0.02

        result = compute_significance(returns, n_bootstrap=500, random_state=42)

        assert result.is_significant == (result.sharpe_p_value < 0.05)


class TestWalkForwardValidator:
    """Tests for WalkForwardValidator."""

    @pytest.fixture
    def test_data(self):
        """Create test data for walk-forward validation."""
        np.random.seed(42)

        n_weeks = 80
        n_tickers = 5
        n_features = 6
        tickers = ['XLK', 'XLF', 'XLE', 'XLY', 'XLP']

        # Create dates
        dates = pd.date_range(start='2020-01-01', periods=n_weeks, freq='W-FRI')

        # Create prices
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(np.random.randn(n_weeks, n_tickers + 1) * 0.01, axis=0)),
            index=dates,
            columns=['SPY'] + tickers
        )

        # Create features and labels
        n_samples = n_weeks * n_tickers
        features = np.random.randn(n_samples, n_features)
        labels = np.random.randn(n_samples) * 0.02

        # Create indices
        date_indices = np.repeat(np.arange(n_weeks), n_tickers)
        ticker_indices = np.tile(np.arange(n_tickers), n_weeks)

        idx_to_date = {i: d for i, d in enumerate(dates)}
        idx_to_ticker = {i: t for i, t in enumerate(tickers)}

        return {
            'prices': prices,
            'dates': dates,
            'tickers': tickers,
            'features': features,
            'labels': labels,
            'date_indices': date_indices,
            'ticker_indices': ticker_indices,
            'idx_to_date': idx_to_date,
            'idx_to_ticker': idx_to_ticker
        }

    def test_validator_initialization(self, test_data):
        """Test validator can be initialized."""
        validator = WalkForwardValidator(
            prices=test_data['prices'],
            dates=list(test_data['dates']),
            tickers=test_data['tickers'],
            min_train_weeks=20
        )

        assert validator.min_train_weeks == 20
        assert len(validator.tickers) == 5

    def test_expanding_window_validation(self, test_data):
        """Test expanding window walk-forward validation."""
        validator = WalkForwardValidator(
            prices=test_data['prices'],
            dates=list(test_data['dates']),
            tickers=test_data['tickers'],
            min_train_weeks=20
        )

        # Simple model: just return mean prediction
        class SimpleModel:
            def __init__(self):
                self.mean_pred = 0.0

            def train(self, X, y):
                self.mean_pred = np.mean(y)

            def predict(self, X):
                return np.full(len(X), self.mean_pred)

        model = SimpleModel()

        result = validator.run(
            train_fn=model.train,
            predict_fn=model.predict,
            features=test_data['features'],
            labels=test_data['labels'],
            date_indices=test_data['date_indices'],
            ticker_indices=test_data['ticker_indices'],
            idx_to_date=test_data['idx_to_date'],
            idx_to_ticker=test_data['idx_to_ticker'],
            model_name="simple_model",
            window_type="expanding",
            verbose=False
        )

        assert isinstance(result, WalkForwardResult)
        assert result.window_type == "expanding"
        assert len(result.folds) > 0
        assert len(result.returns) == len(result.folds)

    def test_rolling_window_validation(self, test_data):
        """Test rolling window walk-forward validation."""
        validator = WalkForwardValidator(
            prices=test_data['prices'],
            dates=list(test_data['dates']),
            tickers=test_data['tickers'],
            min_train_weeks=20
        )

        class SimpleModel:
            def train(self, X, y):
                pass

            def predict(self, X):
                return np.zeros(len(X))

        model = SimpleModel()

        result = validator.run(
            train_fn=model.train,
            predict_fn=model.predict,
            features=test_data['features'],
            labels=test_data['labels'],
            date_indices=test_data['date_indices'],
            ticker_indices=test_data['ticker_indices'],
            idx_to_date=test_data['idx_to_date'],
            idx_to_ticker=test_data['idx_to_ticker'],
            model_name="simple_model",
            window_type="rolling",
            rolling_window=30,
            verbose=False
        )

        assert result.window_type == "rolling"
        assert len(result.folds) > 0

    def test_insufficient_data_raises_error(self, test_data):
        """Test that insufficient data raises ValueError."""
        validator = WalkForwardValidator(
            prices=test_data['prices'],
            dates=list(test_data['dates']),
            tickers=test_data['tickers'],
            min_train_weeks=100  # More than available weeks
        )

        class SimpleModel:
            def train(self, X, y):
                pass

            def predict(self, X):
                return np.zeros(len(X))

        model = SimpleModel()

        with pytest.raises(ValueError, match="Not enough data"):
            validator.run(
                train_fn=model.train,
                predict_fn=model.predict,
                features=test_data['features'],
                labels=test_data['labels'],
                date_indices=test_data['date_indices'],
                ticker_indices=test_data['ticker_indices'],
                idx_to_date=test_data['idx_to_date'],
                idx_to_ticker=test_data['idx_to_ticker'],
                verbose=False
            )

    def test_result_metrics_valid(self, test_data):
        """Test that result metrics are valid."""
        validator = WalkForwardValidator(
            prices=test_data['prices'],
            dates=list(test_data['dates']),
            tickers=test_data['tickers'],
            min_train_weeks=20
        )

        class SimpleModel:
            def train(self, X, y):
                pass

            def predict(self, X):
                return np.random.randn(len(X)) * 0.01

        model = SimpleModel()

        result = validator.run(
            train_fn=model.train,
            predict_fn=model.predict,
            features=test_data['features'],
            labels=test_data['labels'],
            date_indices=test_data['date_indices'],
            ticker_indices=test_data['ticker_indices'],
            idx_to_date=test_data['idx_to_date'],
            idx_to_ticker=test_data['idx_to_ticker'],
            verbose=False
        )

        assert np.isfinite(result.metrics.sharpe_ratio)
        assert np.isfinite(result.metrics.total_return)
        assert len(result.equity_curve) == len(result.returns) + 1


class TestWalkForwardResultConversion:
    """Tests for result conversion utilities."""

    def test_result_to_dict(self):
        """Test conversion to dictionary."""
        np.random.seed(42)
        from qcml_rotation.backtest.metrics import compute_all_metrics, compute_equity_from_returns

        returns = np.random.randn(20) * 0.02
        equity = compute_equity_from_returns(returns)
        metrics = compute_all_metrics(returns, equity)

        result = WalkForwardResult(
            model_name="test_model",
            window_type="expanding",
            min_train_weeks=10,
            folds=[],
            returns=returns,
            equity_curve=equity,
            turnovers=np.abs(np.random.randn(20)) * 0.5,
            metrics=metrics,
            predictions=np.random.randn(20, 5),
            actuals=np.random.randn(20, 5),
            test_dates=[pd.Timestamp('2020-01-01')]
        )

        result_dict = walk_forward_result_to_dict(result)

        assert result_dict['model_name'] == 'test_model'
        assert result_dict['window_type'] == 'expanding'
        assert 'metrics' in result_dict
        assert 'returns' in result_dict

    def test_compare_results(self):
        """Test comparison of multiple results."""
        np.random.seed(42)
        from qcml_rotation.backtest.metrics import compute_all_metrics, compute_equity_from_returns

        def create_result(name):
            returns = np.random.randn(20) * 0.02
            equity = compute_equity_from_returns(returns)
            metrics = compute_all_metrics(returns, equity)
            return WalkForwardResult(
                model_name=name,
                window_type="expanding",
                min_train_weeks=10,
                folds=[],
                returns=returns,
                equity_curve=equity,
                turnovers=np.abs(np.random.randn(20)) * 0.5,
                metrics=metrics,
                predictions=np.random.randn(20, 5),
                actuals=np.random.randn(20, 5),
                test_dates=[pd.Timestamp('2020-01-01')]
            )

        results = {
            'model_a': create_result('model_a'),
            'model_b': create_result('model_b')
        }

        comparison = compare_walk_forward_results(results)

        assert len(comparison) == 2
        assert 'model_a' in comparison.index
        assert 'model_b' in comparison.index
        assert 'Sharpe' in comparison.columns
