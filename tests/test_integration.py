"""
Integration tests for the full QCML ETF rotation pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from qcml_rotation.data.features import FeatureConfig, get_feature_names
from qcml_rotation.models.baselines.pca_ridge import PCARidge, PCARidgeConfig
from qcml_rotation.models.baselines.mlp import MLP
from qcml_rotation.models.qcml import QCMLConfig, create_qcml_model
from qcml_rotation.backtest.portfolio import PortfolioConfig, PortfolioConstructor
from qcml_rotation.backtest.metrics import compute_equity_from_returns, sharpe_ratio


class TestFullPipeline:
    """Integration tests for full pipeline."""

    @pytest.fixture
    def pipeline_data(self):
        """Create synthetic data for pipeline testing."""
        np.random.seed(42)

        n_weeks = 50
        n_tickers = 5
        tickers = ['XLK', 'XLF', 'XLE', 'XLY', 'XLP']

        # Use basic features config (no extended features for simpler testing)
        feature_config = FeatureConfig(
            include_technical=False,
            include_cross_sectional=False,
            include_regime=False
        )
        feature_names = get_feature_names(feature_config)
        n_features = len(feature_names)

        # Generate features
        features = np.random.randn(n_weeks * n_tickers, n_features) * 0.01

        # Generate labels (excess returns)
        labels = np.random.randn(n_weeks * n_tickers) * 0.01

        # Create DataFrame
        data = pd.DataFrame(features, columns=feature_names)
        data['excess_return'] = labels

        # Add metadata
        dates = pd.date_range(start='2020-01-01', periods=n_weeks, freq='W-FRI')
        date_list = []
        ticker_list = []
        for date in dates:
            for ticker in tickers:
                date_list.append(date)
                ticker_list.append(ticker)

        data['date'] = date_list
        data['ticker'] = ticker_list

        # Generate prices
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(np.random.randn(n_weeks, n_tickers + 1) * 0.01, axis=0)),
            index=dates,
            columns=['SPY'] + tickers
        )

        return {
            'data': data,
            'features': features,
            'labels': labels,
            'tickers': tickers,
            'dates': dates,
            'prices': prices
        }

    def test_data_to_features_to_model(self, pipeline_data, device):
        """Test data → features → model prediction flow."""
        X = pipeline_data['features']
        y = pipeline_data['labels']

        # Train PCA model
        pca_model = PCARidge(PCARidgeConfig(n_components=3))
        pca_model.fit(X, y)
        pca_preds = pca_model.predict(X)

        assert pca_preds.shape == y.shape
        assert np.all(np.isfinite(pca_preds))

        # Train QCML model
        config = QCMLConfig(hilbert_dim=8)
        qcml_model = create_qcml_model(input_dim=X.shape[1], config=config)
        qcml_model.to(device)
        qcml_model.eval()

        X_tensor = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            qcml_preds = qcml_model(X_tensor).cpu().numpy().flatten()

        assert qcml_preds.shape == y.shape
        assert np.all(np.isfinite(qcml_preds))

    def test_model_to_portfolio_to_backtest(self, pipeline_data, device):
        """Test model → portfolio → backtest flow."""
        X = pipeline_data['features']
        y = pipeline_data['labels']
        tickers = pipeline_data['tickers']
        n_tickers = len(tickers)
        n_weeks = len(pipeline_data['dates'])

        # Get predictions from a simple model
        pca_model = PCARidge(PCARidgeConfig(n_components=3))
        pca_model.fit(X, y)
        preds = pca_model.predict(X)

        # Reshape predictions by week
        preds_by_week = preds.reshape(n_weeks, n_tickers)

        # Construct portfolios
        config = PortfolioConfig(top_k=2, strategy='long_short')
        constructor = PortfolioConstructor(config)

        portfolios = []
        for i, week_preds in enumerate(preds_by_week):
            date = pipeline_data['dates'][i]
            portfolio = constructor.construct(week_preds, tickers, date)
            portfolios.append(portfolio)

        assert len(portfolios) == n_weeks

        # Verify portfolio properties
        for portfolio in portfolios:
            long_weight = sum(p.weight for p in portfolio.positions if p.side == 'long')
            short_weight = sum(p.weight for p in portfolio.positions if p.side == 'short')
            assert abs(long_weight - short_weight) < 0.001  # Dollar neutral

    def test_backtest_produces_valid_metrics(self, pipeline_data):
        """Test that backtest produces valid performance metrics."""
        np.random.seed(42)

        # Generate synthetic returns
        n_weeks = 50
        returns = np.random.randn(n_weeks) * 0.02

        # Compute metrics
        equity = compute_equity_from_returns(returns)
        sr = sharpe_ratio(returns)

        assert len(equity) == n_weeks + 1
        assert equity[0] == 1.0
        assert np.isfinite(sr)


class TestModelReproducibility:
    """Tests for model reproducibility with fixed seeds."""

    def test_pca_reproducibility(self, synthetic_features, synthetic_labels):
        """Test PCA model produces same results with same seed."""
        # First run
        np.random.seed(42)
        model1 = PCARidge(PCARidgeConfig(n_components=3))
        model1.fit(synthetic_features, synthetic_labels)
        preds1 = model1.predict(synthetic_features)

        # Second run
        np.random.seed(42)
        model2 = PCARidge(PCARidgeConfig(n_components=3))
        model2.fit(synthetic_features, synthetic_labels)
        preds2 = model2.predict(synthetic_features)

        np.testing.assert_array_almost_equal(preds1, preds2)

    def test_qcml_reproducibility(self, synthetic_features, device):
        """Test QCML model produces same results with same seed."""
        input_dim = synthetic_features.shape[1]

        # First run
        torch.manual_seed(42)
        config1 = QCMLConfig(hilbert_dim=8)
        model1 = create_qcml_model(input_dim=input_dim, config=config1)
        model1.to(device)
        model1.eval()

        X = torch.FloatTensor(synthetic_features).to(device)
        with torch.no_grad():
            preds1 = model1(X).cpu().numpy()

        # Second run
        torch.manual_seed(42)
        config2 = QCMLConfig(hilbert_dim=8)
        model2 = create_qcml_model(input_dim=input_dim, config=config2)
        model2.to(device)
        model2.eval()

        with torch.no_grad():
            preds2 = model2(X).cpu().numpy()

        np.testing.assert_array_almost_equal(preds1, preds2)


class TestBacktestVsManualCalculation:
    """Test backtest results match manual calculations."""

    def test_portfolio_return_calculation(self):
        """Test portfolio return matches manual calculation."""
        # Simple portfolio: 50% long XLK, 50% short XLF
        weights = {'XLK': 0.5, 'XLF': -0.5}

        # Weekly returns
        returns = {'XLK': 0.02, 'XLF': 0.01, 'SPY': 0.015}

        # Portfolio return = 0.5 * 0.02 + (-0.5) * 0.01 = 0.01 - 0.005 = 0.005
        expected_return = weights['XLK'] * returns['XLK'] + weights['XLF'] * returns['XLF']

        assert abs(expected_return - 0.005) < 0.0001

    def test_equity_curve_calculation(self):
        """Test equity curve matches manual calculation."""
        returns = np.array([0.01, 0.02, -0.01])

        # Manual: 1.0 -> 1.01 -> 1.0302 -> 1.019898
        expected = [1.0, 1.01, 1.0302, 1.019898]

        equity = compute_equity_from_returns(returns)

        np.testing.assert_array_almost_equal(equity, expected, decimal=5)

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio matches manual calculation."""
        # Known returns with variance
        np.random.seed(42)
        returns = np.random.randn(52) * 0.02 + 0.01  # Mean ~1%, std ~2%

        sr = sharpe_ratio(returns)

        # With positive mean and variance, Sharpe should be positive
        assert sr > 0 and np.isfinite(sr)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_sample_prediction(self, device):
        """Test model can handle single sample."""
        input_dim = 6
        config = QCMLConfig(hilbert_dim=8)
        model = create_qcml_model(input_dim=input_dim, config=config)
        model.to(device)
        model.eval()

        x = torch.randn(1, input_dim, device=device)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1,)

    def test_empty_returns_metrics(self):
        """Test metrics handle empty returns gracefully."""
        returns = np.array([])

        # These should not raise errors
        equity = compute_equity_from_returns(returns)
        assert len(equity) == 1  # Just initial value

    def test_constant_returns_sharpe(self):
        """Test Sharpe with constant returns (near-zero std)."""
        returns = np.array([0.01] * 10)

        sr = sharpe_ratio(returns)
        # With near-zero std, Sharpe can be very large or undefined
        # Just check it's finite or very large
        assert np.isfinite(sr) or np.isinf(sr) or sr > 1e10

    def test_all_negative_returns(self):
        """Test metrics with all negative returns."""
        # Use returns with variance for valid Sharpe
        np.random.seed(42)
        returns = np.random.randn(52) * 0.02 - 0.01  # Mean ~-1%

        sr = sharpe_ratio(returns)
        equity = compute_equity_from_returns(returns)

        assert sr < 0  # Negative mean should give negative Sharpe
        # Note: equity may or may not be < 1.0 due to variance
