"""
Portfolio construction logic for ETF rotation strategy.

Supports long/short and long-only with SPY hedge strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Position:
    """Single position in portfolio."""
    ticker: str
    weight: float
    side: str  # 'long' or 'short'


@dataclass
class Portfolio:
    """Portfolio state at a point in time."""
    positions: List[Position]
    date: pd.Timestamp

    def get_weights(self) -> Dict[str, float]:
        """Get ticker -> weight mapping (signed for long/short)."""
        return {
            p.ticker: p.weight if p.side == 'long' else -p.weight
            for p in self.positions
        }

    def get_long_tickers(self) -> List[str]:
        return [p.ticker for p in self.positions if p.side == 'long']

    def get_short_tickers(self) -> List[str]:
        return [p.ticker for p in self.positions if p.side == 'short']


@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction."""
    top_k: int = 3                  # Number of top ETFs to go long
    bottom_k: int = 3               # Number of bottom ETFs to go short
    strategy: str = "long_short"    # 'long_short' or 'long_hedge_spy'
    transaction_cost_bps: float = 5 # Cost per side in basis points
    equal_weight: bool = True       # Equal weight or prediction-weighted


class PortfolioConstructor:
    """
    Constructs portfolios from model predictions.

    Strategies:
    - long_short: Long top K, short bottom K (dollar-neutral)
    - long_hedge_spy: Long top K, short SPY for hedge
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        if config is None:
            config = PortfolioConfig()
        self.config = config

    def construct(
        self,
        predictions: np.ndarray,
        tickers: List[str],
        date: pd.Timestamp
    ) -> Portfolio:
        """
        Construct portfolio from predictions.

        Parameters
        ----------
        predictions : array of shape (n_etfs,)
            Predicted excess returns for each ETF.
        tickers : list
            ETF tickers (same order as predictions).
        date : Timestamp
            Date of portfolio construction.

        Returns
        -------
        portfolio : Portfolio
        """
        if self.config.strategy == "long_short":
            return self._construct_long_short(predictions, tickers, date)
        elif self.config.strategy == "long_hedge_spy":
            return self._construct_long_hedge_spy(predictions, tickers, date)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def _construct_long_short(
        self,
        predictions: np.ndarray,
        tickers: List[str],
        date: pd.Timestamp
    ) -> Portfolio:
        """Dollar-neutral long/short portfolio."""
        n = len(predictions)
        top_k = min(self.config.top_k, n // 2)
        bottom_k = min(self.config.bottom_k, n // 2)

        # Rank by prediction (descending)
        ranks = np.argsort(predictions)[::-1]

        positions = []

        # Long positions
        long_weight = 0.5 / top_k if top_k > 0 else 0
        for i in range(top_k):
            idx = ranks[i]
            positions.append(Position(
                ticker=tickers[idx],
                weight=long_weight,
                side='long'
            ))

        # Short positions
        short_weight = 0.5 / bottom_k if bottom_k > 0 else 0
        for i in range(bottom_k):
            idx = ranks[-(i + 1)]
            positions.append(Position(
                ticker=tickers[idx],
                weight=short_weight,
                side='short'
            ))

        return Portfolio(positions=positions, date=date)

    def _construct_long_hedge_spy(
        self,
        predictions: np.ndarray,
        tickers: List[str],
        date: pd.Timestamp
    ) -> Portfolio:
        """Long top K ETFs, hedge with SPY."""
        n = len(predictions)
        top_k = min(self.config.top_k, n)

        ranks = np.argsort(predictions)[::-1]

        positions = []

        # Long positions (equal weight summing to 1.0)
        long_weight = 1.0 / top_k if top_k > 0 else 0
        for i in range(top_k):
            idx = ranks[i]
            positions.append(Position(
                ticker=tickers[idx],
                weight=long_weight,
                side='long'
            ))

        # Short SPY as hedge
        positions.append(Position(
            ticker='SPY',
            weight=1.0,
            side='short'
        ))

        return Portfolio(positions=positions, date=date)

    def compute_turnover(
        self,
        old_portfolio: Optional[Portfolio],
        new_portfolio: Portfolio
    ) -> float:
        """
        Compute turnover between two portfolios.

        Turnover = sum of absolute weight changes / 2
        """
        if old_portfolio is None:
            # Initial portfolio: turnover = sum of absolute weights
            return sum(abs(p.weight) for p in new_portfolio.positions)

        old_weights = old_portfolio.get_weights()
        new_weights = new_portfolio.get_weights()

        all_tickers = set(old_weights.keys()) | set(new_weights.keys())

        total_change = 0.0
        for ticker in all_tickers:
            old_w = old_weights.get(ticker, 0.0)
            new_w = new_weights.get(ticker, 0.0)
            total_change += abs(new_w - old_w)

        return total_change / 2

    def compute_transaction_costs(self, turnover: float) -> float:
        """
        Compute transaction costs from turnover.

        Cost = turnover * cost_bps * 2 (buy + sell)
        """
        cost_decimal = self.config.transaction_cost_bps / 10000
        return turnover * cost_decimal * 2

    def compute_portfolio_return(
        self,
        portfolio: Portfolio,
        returns: Dict[str, float]
    ) -> float:
        """
        Compute portfolio return given individual asset returns.

        Parameters
        ----------
        portfolio : Portfolio
        returns : dict
            Ticker -> period return.

        Returns
        -------
        portfolio_return : float
        """
        total_return = 0.0

        for position in portfolio.positions:
            ticker = position.ticker
            asset_return = returns.get(ticker, 0.0)

            if position.side == 'long':
                total_return += position.weight * asset_return
            else:  # short
                total_return -= position.weight * asset_return

        return total_return


def compute_weekly_returns(
    prices: pd.DataFrame,
    dates: List[pd.Timestamp]
) -> Dict[pd.Timestamp, Dict[str, float]]:
    """
    Compute weekly returns for each ETF.

    Parameters
    ----------
    prices : DataFrame
        Daily prices with tickers as columns.
    dates : list
        Rebalance dates.

    Returns
    -------
    returns : dict
        date -> {ticker -> return}
    """
    returns = {}

    for i in range(len(dates) - 1):
        start_date = dates[i]
        end_date = dates[i + 1]

        # Find actual dates in price data
        if start_date not in prices.index:
            start_idx = prices.index.get_indexer([start_date], method='ffill')[0]
            start_date = prices.index[start_idx]

        if end_date not in prices.index:
            end_idx = prices.index.get_indexer([end_date], method='ffill')[0]
            end_date = prices.index[end_idx]

        start_prices = prices.loc[start_date]
        end_prices = prices.loc[end_date]

        period_returns = {}
        for ticker in prices.columns:
            if start_prices[ticker] > 0:
                ret = (end_prices[ticker] / start_prices[ticker]) - 1
                period_returns[ticker] = ret
            else:
                period_returns[ticker] = 0.0

        returns[dates[i]] = period_returns

    return returns
