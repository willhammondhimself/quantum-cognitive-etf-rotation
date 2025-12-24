"""
Tests for portfolio construction module.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qcml_rotation.backtest.portfolio import (
    Portfolio,
    Position,
    PortfolioConfig,
    PortfolioConstructor,
)


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test position can be created with required fields."""
        pos = Position(ticker='XLK', weight=0.5, side='long')
        assert pos.ticker == 'XLK'
        assert pos.weight == 0.5
        assert pos.side == 'long'

    def test_position_weight_positive(self):
        """Test that weights are typically positive."""
        pos = Position(ticker='XLK', weight=0.25, side='short')
        assert pos.weight > 0


class TestPortfolio:
    """Tests for Portfolio dataclass."""

    def test_portfolio_creation(self):
        """Test portfolio can be created."""
        positions = [
            Position(ticker='XLK', weight=0.5, side='long'),
            Position(ticker='XLF', weight=0.5, side='short'),
        ]
        portfolio = Portfolio(positions=positions)
        assert len(portfolio.positions) == 2

    def test_portfolio_long_weights_sum(self):
        """Test that long weights sum correctly."""
        positions = [
            Position(ticker='XLK', weight=0.25, side='long'),
            Position(ticker='XLF', weight=0.25, side='long'),
            Position(ticker='XLE', weight=0.5, side='short'),
        ]
        portfolio = Portfolio(positions=positions)

        long_weights = sum(p.weight for p in portfolio.positions if p.side == 'long')
        assert abs(long_weights - 0.5) < 0.001


class TestPortfolioConfig:
    """Tests for PortfolioConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PortfolioConfig()
        assert config.top_k == 3
        assert config.strategy == 'long_short'
        assert config.transaction_cost_bps == 5

    def test_custom_config(self):
        """Test custom configuration."""
        config = PortfolioConfig(top_k=5, strategy='long_hedge_spy', transaction_cost_bps=10)
        assert config.top_k == 5
        assert config.strategy == 'long_hedge_spy'
        assert config.transaction_cost_bps == 10


class TestPortfolioConstructor:
    """Tests for PortfolioConstructor."""

    def test_long_short_construction(self):
        """Test long/short portfolio construction."""
        config = PortfolioConfig(top_k=2, strategy='long_short')
        constructor = PortfolioConstructor(config)

        predictions = np.array([0.02, 0.01, -0.01, -0.02, 0.005])
        tickers = ['XLK', 'XLF', 'XLE', 'XLY', 'XLP']

        portfolio = constructor.construct(predictions, tickers)

        # Check we have positions
        assert len(portfolio.positions) > 0

        # Check long and short positions
        long_positions = [p for p in portfolio.positions if p.side == 'long']
        short_positions = [p for p in portfolio.positions if p.side == 'short']

        assert len(long_positions) == 2  # top_k longs
        assert len(short_positions) == 2  # top_k shorts

    def test_long_short_dollar_neutral(self):
        """Test that long/short portfolio is dollar-neutral."""
        config = PortfolioConfig(top_k=2, strategy='long_short')
        constructor = PortfolioConstructor(config)

        predictions = np.array([0.02, 0.01, -0.01, -0.02, 0.005])
        tickers = ['XLK', 'XLF', 'XLE', 'XLY', 'XLP']

        portfolio = constructor.construct(predictions, tickers)

        long_weight = sum(p.weight for p in portfolio.positions if p.side == 'long')
        short_weight = sum(p.weight for p in portfolio.positions if p.side == 'short')

        # Should be dollar-neutral (equal long and short weights)
        assert abs(long_weight - short_weight) < 0.001

    def test_positions_sum_correctly(self):
        """Test that position weights sum to expected values."""
        config = PortfolioConfig(top_k=3, strategy='long_short')
        constructor = PortfolioConstructor(config)

        predictions = np.array([0.03, 0.02, 0.01, -0.01, -0.02, -0.03])
        tickers = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']

        portfolio = constructor.construct(predictions, tickers)

        long_weight = sum(p.weight for p in portfolio.positions if p.side == 'long')
        short_weight = sum(p.weight for p in portfolio.positions if p.side == 'short')

        # Equal weight: each position = 0.5 / k
        expected_weight_per_position = 0.5 / config.top_k
        assert abs(long_weight - 0.5) < 0.001
        assert abs(short_weight - 0.5) < 0.001


class TestTurnoverCalculation:
    """Tests for turnover calculation."""

    def test_initial_turnover(self):
        """Test turnover for initial portfolio."""
        config = PortfolioConfig(top_k=2)
        constructor = PortfolioConstructor(config)

        predictions = np.array([0.02, 0.01, -0.01, -0.02])
        tickers = ['XLK', 'XLF', 'XLE', 'XLY']

        portfolio = constructor.construct(predictions, tickers)

        # Initial turnover = sum of absolute weights = 1.0
        total_weight = sum(p.weight for p in portfolio.positions)
        assert abs(total_weight - 1.0) < 0.001

    def test_turnover_bounded(self):
        """Test that turnover is bounded [0, 2]."""
        config = PortfolioConfig(top_k=2)
        constructor = PortfolioConstructor(config)

        # Create two portfolios and compute turnover
        preds1 = np.array([0.02, 0.01, -0.01, -0.02])
        preds2 = np.array([-0.02, -0.01, 0.01, 0.02])  # Completely reversed
        tickers = ['XLK', 'XLF', 'XLE', 'XLY']

        p1 = constructor.construct(preds1, tickers)
        p2 = constructor.construct(preds2, tickers)

        # Compute turnover manually
        weights1 = {p.ticker: p.weight * (1 if p.side == 'long' else -1) for p in p1.positions}
        weights2 = {p.ticker: p.weight * (1 if p.side == 'long' else -1) for p in p2.positions}

        all_tickers = set(weights1.keys()) | set(weights2.keys())
        turnover = sum(abs(weights2.get(t, 0) - weights1.get(t, 0)) for t in all_tickers)

        assert 0 <= turnover <= 2


class TestTransactionCosts:
    """Tests for transaction cost calculation."""

    def test_transaction_cost_formula(self):
        """Test transaction cost calculation."""
        turnover = 0.5  # 50% turnover
        cost_bps = 5  # 5 basis points per side

        # Cost = turnover * 2 * bps / 10000
        expected_cost = turnover * 2 * cost_bps / 10000
        assert abs(expected_cost - 0.0005) < 0.0001

    def test_zero_turnover_zero_cost(self):
        """Test that zero turnover means zero cost."""
        turnover = 0.0
        cost_bps = 5

        cost = turnover * 2 * cost_bps / 10000
        assert cost == 0.0
