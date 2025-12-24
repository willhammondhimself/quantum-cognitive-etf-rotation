"""
Walk-forward backtesting for realistic out-of-sample validation.

Provides expanding and rolling window validation to avoid overfitting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Literal
from dataclasses import dataclass

from .metrics import (
    compute_all_metrics, PerformanceMetrics, compute_equity_from_returns,
    sharpe_ratio
)
from .portfolio import PortfolioConstructor, Portfolio, PortfolioConfig, compute_weekly_returns


@dataclass
class WalkForwardFold:
    """Results from a single walk-forward fold."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_date: pd.Timestamp
    prediction: np.ndarray  # (n_tickers,)
    actual: np.ndarray      # (n_tickers,)
    portfolio_return: float
    turnover: float


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward validation."""
    model_name: str
    window_type: str
    min_train_weeks: int
    folds: List[WalkForwardFold]
    returns: np.ndarray
    equity_curve: np.ndarray
    turnovers: np.ndarray
    metrics: PerformanceMetrics
    predictions: np.ndarray  # (n_weeks, n_tickers)
    actuals: np.ndarray      # (n_weeks, n_tickers)
    test_dates: List[pd.Timestamp]


class WalkForwardValidator:
    """
    Walk-forward validation for time-series models.

    Supports:
    - Expanding window: Train on [0:t], test on [t:t+1]
    - Rolling window: Train on [t-N:t], test on [t:t+1]

    Usage:
        validator = WalkForwardValidator(
            prices=prices,
            dates=all_dates,
            tickers=tickers,
            min_train_weeks=52
        )
        result = validator.run(train_fn, predict_fn, features, labels, ...)
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        dates: List[pd.Timestamp],
        tickers: List[str],
        min_train_weeks: int = 52,
        portfolio_config: Optional[PortfolioConfig] = None,
        benchmark: str = "SPY"
    ):
        """
        Parameters
        ----------
        prices : DataFrame
            Daily prices with tickers as columns.
        dates : list
            All rebalance dates (both train and test).
        tickers : list
            ETF tickers (excluding benchmark).
        min_train_weeks : int
            Minimum training window size in weeks.
        portfolio_config : PortfolioConfig, optional
        benchmark : str
            Benchmark ticker.
        """
        self.prices = prices
        self.all_dates = sorted(dates)
        self.tickers = tickers
        self.min_train_weeks = min_train_weeks
        self.benchmark = benchmark

        if portfolio_config is None:
            portfolio_config = PortfolioConfig()
        self.portfolio_config = portfolio_config
        self.constructor = PortfolioConstructor(portfolio_config)

        # Precompute weekly returns
        all_tickers = [benchmark] + list(tickers)
        self.weekly_returns = compute_weekly_returns(
            prices[all_tickers],
            self.all_dates
        )

    def run(
        self,
        train_fn: Callable[[np.ndarray, np.ndarray], None],
        predict_fn: Callable[[np.ndarray], np.ndarray],
        features: np.ndarray,
        labels: np.ndarray,
        date_indices: np.ndarray,
        ticker_indices: np.ndarray,
        idx_to_date: Dict[int, pd.Timestamp],
        idx_to_ticker: Dict[int, str],
        model_name: str = "model",
        window_type: Literal["expanding", "rolling"] = "expanding",
        rolling_window: Optional[int] = None,
        verbose: bool = True
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Parameters
        ----------
        train_fn : callable
            Function that takes (features, labels) and trains the model in-place.
        predict_fn : callable
            Function that takes features and returns predictions.
        features : array of shape (n_samples, n_features)
            All features.
        labels : array of shape (n_samples,)
            All labels.
        date_indices : array of shape (n_samples,)
            Date index for each sample.
        ticker_indices : array of shape (n_samples,)
            Ticker index for each sample.
        idx_to_date : dict
            Mapping from date index to timestamp.
        idx_to_ticker : dict
            Mapping from ticker index to ticker string.
        model_name : str
            Name for this model.
        window_type : str
            "expanding" or "rolling".
        rolling_window : int, optional
            Window size for rolling (required if window_type="rolling").
        verbose : bool
            Print progress.

        Returns
        -------
        result : WalkForwardResult
        """
        if window_type == "rolling" and rolling_window is None:
            rolling_window = self.min_train_weeks

        # Get unique dates sorted
        unique_date_indices = sorted(np.unique(date_indices))
        n_total_weeks = len(unique_date_indices)

        # Determine test fold range
        first_test_idx = self.min_train_weeks
        n_test_weeks = n_total_weeks - first_test_idx

        if n_test_weeks <= 0:
            raise ValueError(
                f"Not enough data: {n_total_weeks} weeks < min_train_weeks ({self.min_train_weeks})"
            )

        if verbose:
            print(f"Walk-forward validation: {window_type} window")
            print(f"  Total weeks: {n_total_weeks}")
            print(f"  Min training: {self.min_train_weeks} weeks")
            print(f"  Test folds: {n_test_weeks}")

        # Build mappings
        date_to_idx = {v: k for k, v in idx_to_date.items()}
        ticker_to_col = {t: i for i, t in enumerate(self.tickers)}

        folds = []
        returns = []
        turnovers = []
        pred_matrix = np.zeros((n_test_weeks, len(self.tickers)))
        actual_matrix = np.zeros((n_test_weeks, len(self.tickers)))
        test_dates = []
        prev_portfolio = None

        for fold_idx in range(n_test_weeks):
            test_week_idx = first_test_idx + fold_idx
            test_date_idx = unique_date_indices[test_week_idx]
            test_date = idx_to_date[test_date_idx]

            # Determine training range
            if window_type == "expanding":
                train_start_idx = 0
            else:  # rolling
                train_start_idx = max(0, test_week_idx - rolling_window)

            train_end_idx = test_week_idx
            train_date_indices_set = set(unique_date_indices[train_start_idx:train_end_idx])

            # Get training data
            train_mask = np.array([d in train_date_indices_set for d in date_indices])
            X_train = features[train_mask]
            y_train = labels[train_mask]

            # Get test data
            test_mask = date_indices == test_date_idx
            X_test = features[test_mask]
            y_test = labels[test_mask]
            test_ticker_indices = ticker_indices[test_mask]

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            # Train model on this fold's training data
            train_fn(X_train, y_train)

            # Get predictions
            preds = predict_fn(X_test)

            # Organize predictions by ticker
            week_preds = np.zeros(len(self.tickers))
            week_actuals = np.zeros(len(self.tickers))

            for i, ticker_idx in enumerate(test_ticker_indices):
                ticker = idx_to_ticker[ticker_idx]
                if ticker in ticker_to_col:
                    col_idx = ticker_to_col[ticker]
                    week_preds[col_idx] = preds[i]
                    week_actuals[col_idx] = y_test[i]

            pred_matrix[fold_idx] = week_preds
            actual_matrix[fold_idx] = week_actuals

            # Construct portfolio and compute return
            portfolio = self.constructor.construct(week_preds, self.tickers, test_date)
            turnover = self.constructor.compute_turnover(prev_portfolio, portfolio)
            costs = self.constructor.compute_transaction_costs(turnover)

            week_returns = self.weekly_returns.get(test_date, {})
            gross_return = self.constructor.compute_portfolio_return(portfolio, week_returns)
            net_return = gross_return - costs

            # Record fold
            train_start_date = idx_to_date[unique_date_indices[train_start_idx]]
            train_end_date = idx_to_date[unique_date_indices[train_end_idx - 1]]

            fold = WalkForwardFold(
                train_start=train_start_date,
                train_end=train_end_date,
                test_date=test_date,
                prediction=week_preds,
                actual=week_actuals,
                portfolio_return=net_return,
                turnover=turnover
            )
            folds.append(fold)
            returns.append(net_return)
            turnovers.append(turnover)
            test_dates.append(test_date)
            prev_portfolio = portfolio

            if verbose and (fold_idx + 1) % 10 == 0:
                print(f"  Completed fold {fold_idx + 1}/{n_test_weeks}")

        # Aggregate results
        returns = np.array(returns)
        turnovers = np.array(turnovers)
        equity_curve = compute_equity_from_returns(returns)

        metrics = compute_all_metrics(
            returns=returns,
            equity_curve=equity_curve,
            predictions=pred_matrix[:len(returns)],
            actuals=actual_matrix[:len(returns)],
            turnovers=turnovers,
            top_k=self.portfolio_config.top_k
        )

        if verbose:
            print(f"  Final Sharpe: {metrics.sharpe_ratio:.3f}")
            print(f"  Total Return: {metrics.total_return:.2%}")

        return WalkForwardResult(
            model_name=model_name,
            window_type=window_type,
            min_train_weeks=self.min_train_weeks,
            folds=folds,
            returns=returns,
            equity_curve=equity_curve,
            turnovers=turnovers,
            metrics=metrics,
            predictions=pred_matrix[:len(returns)],
            actuals=actual_matrix[:len(returns)],
            test_dates=test_dates
        )


def walk_forward_result_to_dict(result: WalkForwardResult) -> Dict:
    """Convert WalkForwardResult to JSON-serializable dictionary."""
    from .metrics import metrics_to_dict

    return {
        "model_name": result.model_name,
        "window_type": result.window_type,
        "min_train_weeks": result.min_train_weeks,
        "n_folds": len(result.folds),
        "test_dates": [str(d) for d in result.test_dates],
        "returns": result.returns.tolist(),
        "equity_curve": result.equity_curve.tolist(),
        "turnovers": result.turnovers.tolist(),
        "metrics": metrics_to_dict(result.metrics)
    }


def compare_walk_forward_results(
    results: Dict[str, WalkForwardResult]
) -> pd.DataFrame:
    """
    Create comparison table of walk-forward results.

    Parameters
    ----------
    results : dict
        model_name -> WalkForwardResult

    Returns
    -------
    comparison : DataFrame
    """
    rows = []

    for name, result in results.items():
        m = result.metrics
        rows.append({
            "Model": name,
            "Window": result.window_type,
            "Test Weeks": len(result.folds),
            "Total Return": f"{m.total_return:.2%}",
            "Sharpe": f"{m.sharpe_ratio:.3f}",
            "Sortino": f"{m.sortino_ratio:.3f}",
            "Max DD": f"{m.max_drawdown:.2%}",
            "Hit Rate": f"{m.hit_rate:.2%}",
            "Win Rate": f"{m.win_rate:.2%}"
        })

    df = pd.DataFrame(rows)
    df = df.set_index("Model")

    return df
