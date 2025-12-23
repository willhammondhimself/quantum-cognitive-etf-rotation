"""
Backtesting engine for ETF rotation strategies.

Runs simulations using trained models and computes performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field

from .portfolio import PortfolioConstructor, Portfolio, PortfolioConfig, compute_weekly_returns
from .metrics import (
    compute_all_metrics, PerformanceMetrics, metrics_to_dict,
    compute_equity_from_returns
)


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    model_name: str
    returns: np.ndarray
    equity_curve: np.ndarray
    turnovers: np.ndarray
    portfolios: List[Portfolio]
    metrics: PerformanceMetrics
    predictions: np.ndarray  # (n_weeks, n_etfs)
    actuals: np.ndarray      # (n_weeks, n_etfs)
    dates: List[pd.Timestamp]


class BacktestEngine:
    """
    Engine for backtesting ETF rotation models.

    Usage:
        engine = BacktestEngine(prices, test_dates, tickers)
        result = engine.run(model, "my_model")
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        test_dates: List[pd.Timestamp],
        tickers: List[str],
        portfolio_config: Optional[PortfolioConfig] = None,
        benchmark: str = "SPY"
    ):
        """
        Parameters
        ----------
        prices : DataFrame
            Daily prices with tickers as columns.
        test_dates : list
            Rebalance dates in the test period.
        tickers : list
            ETF tickers (excluding benchmark).
        portfolio_config : PortfolioConfig, optional
        benchmark : str
            Benchmark ticker.
        """
        self.prices = prices
        self.test_dates = sorted(test_dates)
        self.tickers = tickers
        self.benchmark = benchmark

        if portfolio_config is None:
            portfolio_config = PortfolioConfig()
        self.portfolio_config = portfolio_config
        self.constructor = PortfolioConstructor(portfolio_config)

        # Precompute weekly returns for all assets
        all_tickers = [benchmark] + list(tickers)
        self.weekly_returns = compute_weekly_returns(
            prices[all_tickers],
            self.test_dates
        )

    def run(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        model_name: str,
        features: np.ndarray,
        labels: np.ndarray,
        date_indices: np.ndarray,
        ticker_indices: np.ndarray,
        idx_to_ticker: Dict[int, str]
    ) -> BacktestResult:
        """
        Run backtest with a prediction function.

        Parameters
        ----------
        predict_fn : callable
            Function that takes features and returns predictions.
        model_name : str
            Name for this model.
        features : array of shape (n_samples, n_features)
            Features for test set.
        labels : array of shape (n_samples,)
            Actual excess returns for test set.
        date_indices : array of shape (n_samples,)
            Date index for each sample.
        ticker_indices : array of shape (n_samples,)
            Ticker index for each sample.
        idx_to_ticker : dict
            Mapping from ticker index to ticker string.

        Returns
        -------
        result : BacktestResult
        """
        # Get predictions
        predictions = predict_fn(features)

        # Organize by week
        unique_dates = np.unique(date_indices)
        n_weeks = len(unique_dates)
        n_tickers = len(self.tickers)

        # Arrays to store weekly predictions and actuals
        pred_matrix = np.zeros((n_weeks, n_tickers))
        actual_matrix = np.zeros((n_weeks, n_tickers))

        ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}

        for week_idx, date_idx in enumerate(unique_dates):
            mask = date_indices == date_idx

            week_preds = predictions[mask]
            week_labels = labels[mask]
            week_tickers = ticker_indices[mask]

            for i, ticker_idx in enumerate(week_tickers):
                ticker = idx_to_ticker[ticker_idx]
                if ticker in ticker_to_idx:
                    col_idx = ticker_to_idx[ticker]
                    pred_matrix[week_idx, col_idx] = week_preds[i]
                    actual_matrix[week_idx, col_idx] = week_labels[i]

        # Run simulation
        returns = []
        turnovers = []
        portfolios = []
        prev_portfolio = None

        valid_dates = [d for d in self.test_dates if d in self.weekly_returns]

        for week_idx in range(min(n_weeks, len(valid_dates) - 1)):
            date = valid_dates[week_idx]
            week_preds = pred_matrix[week_idx]

            # Construct portfolio
            portfolio = self.constructor.construct(week_preds, self.tickers, date)
            portfolios.append(portfolio)

            # Compute turnover and costs
            turnover = self.constructor.compute_turnover(prev_portfolio, portfolio)
            turnovers.append(turnover)
            costs = self.constructor.compute_transaction_costs(turnover)

            # Get returns for this week
            week_returns = self.weekly_returns.get(date, {})

            # Compute portfolio return
            gross_return = self.constructor.compute_portfolio_return(portfolio, week_returns)
            net_return = gross_return - costs

            returns.append(net_return)
            prev_portfolio = portfolio

        # Convert to arrays
        returns = np.array(returns)
        turnovers = np.array(turnovers)
        equity_curve = compute_equity_from_returns(returns)

        # Compute metrics
        metrics = compute_all_metrics(
            returns=returns,
            equity_curve=equity_curve,
            predictions=pred_matrix,
            actuals=actual_matrix,
            turnovers=turnovers,
            top_k=self.portfolio_config.top_k
        )

        return BacktestResult(
            model_name=model_name,
            returns=returns,
            equity_curve=equity_curve,
            turnovers=turnovers,
            portfolios=portfolios,
            metrics=metrics,
            predictions=pred_matrix,
            actuals=actual_matrix,
            dates=valid_dates[:len(returns)]
        )

    def run_multiple(
        self,
        models: Dict[str, Tuple[Callable, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]],
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest for multiple models.

        Parameters
        ----------
        models : dict
            model_name -> (predict_fn, features, labels, date_indices, ticker_indices, idx_to_ticker)

        Returns
        -------
        results : dict
            model_name -> BacktestResult
        """
        results = {}

        for name, (predict_fn, features, labels, date_idx, ticker_idx, idx_to_ticker) in models.items():
            print(f"Running backtest for {name}...")
            result = self.run(predict_fn, name, features, labels, date_idx, ticker_idx, idx_to_ticker)
            results[name] = result
            print(f"  Sharpe: {result.metrics.sharpe_ratio:.3f}, "
                  f"Return: {result.metrics.total_return:.2%}")

        return results


def compare_results(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """
    Create comparison table of backtest results.

    Parameters
    ----------
    results : dict
        model_name -> BacktestResult

    Returns
    -------
    comparison : DataFrame
    """
    rows = []

    for name, result in results.items():
        m = result.metrics
        rows.append({
            "Model": name,
            "Total Return": f"{m.total_return:.2%}",
            "Ann. Return": f"{m.annualized_return:.2%}",
            "Ann. Vol": f"{m.annualized_volatility:.2%}",
            "Sharpe": f"{m.sharpe_ratio:.3f}",
            "Sortino": f"{m.sortino_ratio:.3f}",
            "Max DD": f"{m.max_drawdown:.2%}",
            "Calmar": f"{m.calmar_ratio:.3f}",
            "Hit Rate": f"{m.hit_rate:.2%}",
            "Win Rate": f"{m.win_rate:.2%}",
            "Avg Turnover": f"{m.avg_turnover:.2%}"
        })

    df = pd.DataFrame(rows)
    df = df.set_index("Model")

    return df


def results_to_dict(results: Dict[str, BacktestResult]) -> Dict[str, Any]:
    """Convert results to JSON-serializable dictionary."""
    output = {}

    for name, result in results.items():
        output[name] = {
            "metrics": metrics_to_dict(result.metrics),
            "returns": result.returns.tolist(),
            "equity_curve": result.equity_curve.tolist(),
            "turnovers": result.turnovers.tolist(),
            "dates": [str(d) for d in result.dates]
        }

    return output
