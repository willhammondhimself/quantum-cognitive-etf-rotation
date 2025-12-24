"""
Diagnostic tools for analyzing walk-forward validation results.

Helps identify why models fail and where improvements can be made.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class DiagnosticResult:
    """Container for diagnostic analysis results."""
    # Time period analysis
    yearly_returns: pd.Series
    quarterly_returns: pd.Series
    worst_periods: pd.DataFrame

    # Prediction quality
    rank_correlation: float
    rank_correlation_pvalue: float
    prediction_std: float
    hit_rate_by_period: pd.Series

    # Regime analysis
    regime_returns: Optional[pd.DataFrame] = None


def analyze_time_periods(
    returns: np.ndarray,
    dates: List[pd.Timestamp],
    n_worst: int = 5
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Analyze performance by time period.

    Parameters
    ----------
    returns : array
        Portfolio returns.
    dates : list
        Corresponding dates.
    n_worst : int
        Number of worst periods to identify.

    Returns
    -------
    yearly_returns : Series
        Returns by year.
    quarterly_returns : Series
        Returns by quarter.
    worst_periods : DataFrame
        Worst performing periods.
    """
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'return': returns
    })
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Yearly returns (compounded)
    yearly = df.groupby(df.index.year)['return'].apply(
        lambda x: (1 + x).prod() - 1
    )
    yearly.name = 'return'

    # Quarterly returns
    quarterly = df.groupby(pd.Grouper(freq='Q'))['return'].apply(
        lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0.0
    )
    quarterly.name = 'return'

    # Identify worst periods (rolling 4-week windows)
    df['rolling_return'] = (1 + df['return']).rolling(4).apply(
        lambda x: x.prod() - 1, raw=True
    )
    worst = df.nsmallest(n_worst, 'rolling_return')[['return', 'rolling_return']]
    worst.columns = ['weekly_return', 'rolling_4w_return']

    return yearly, quarterly, worst


def analyze_prediction_quality(
    predictions: np.ndarray,
    actuals: np.ndarray,
    top_k: int = 3
) -> Dict:
    """
    Analyze the quality of predictions.

    Parameters
    ----------
    predictions : array of shape (n_weeks, n_etfs)
        Predicted values.
    actuals : array of shape (n_weeks, n_etfs)
        Actual values.
    top_k : int
        Number of top picks to analyze.

    Returns
    -------
    results : dict
        Prediction quality metrics.
    """
    n_weeks = predictions.shape[0]

    # Flatten for overall correlation
    pred_flat = predictions.flatten()
    actual_flat = actuals.flatten()

    # Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(pred_flat, actual_flat)

    # Spearman rank correlation (per week, then average)
    spearman_corrs = []
    for week in range(n_weeks):
        if np.std(predictions[week]) > 0 and np.std(actuals[week]) > 0:
            corr, _ = stats.spearmanr(predictions[week], actuals[week])
            if np.isfinite(corr):
                spearman_corrs.append(corr)

    avg_spearman = np.mean(spearman_corrs) if spearman_corrs else 0.0

    # Hit rate for top-K picks
    hits = 0
    total = 0
    for week in range(n_weeks):
        top_picks = np.argsort(predictions[week])[-top_k:]
        for pick in top_picks:
            if actuals[week, pick] > 0:
                hits += 1
            total += 1

    hit_rate = hits / total if total > 0 else 0.0

    # Prediction distribution stats
    pred_std = np.std(predictions)
    pred_mean = np.mean(predictions)
    pred_range = np.ptp(predictions)

    # Are predictions differentiated or clustered?
    within_week_std = np.mean([np.std(predictions[w]) for w in range(n_weeks)])

    return {
        'pearson_correlation': pearson_corr,
        'pearson_pvalue': pearson_p,
        'avg_spearman_correlation': avg_spearman,
        'hit_rate_top_k': hit_rate,
        'prediction_mean': pred_mean,
        'prediction_std': pred_std,
        'prediction_range': pred_range,
        'within_week_std': within_week_std,
        'n_weeks': n_weeks
    }


def compute_rank_correlation(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> Tuple[float, float]:
    """
    Compute average Spearman rank correlation across weeks.

    Parameters
    ----------
    predictions : array of shape (n_weeks, n_etfs)
    actuals : array of shape (n_weeks, n_etfs)

    Returns
    -------
    avg_correlation : float
    pvalue : float (from t-test on correlations)
    """
    n_weeks = predictions.shape[0]
    correlations = []

    for week in range(n_weeks):
        if np.std(predictions[week]) > 0 and np.std(actuals[week]) > 0:
            corr, _ = stats.spearmanr(predictions[week], actuals[week])
            if np.isfinite(corr):
                correlations.append(corr)

    if not correlations:
        return 0.0, 1.0

    correlations = np.array(correlations)
    avg_corr = np.mean(correlations)

    # t-test: is mean correlation significantly different from 0?
    if len(correlations) > 1:
        t_stat, pvalue = stats.ttest_1samp(correlations, 0)
    else:
        pvalue = 1.0

    return float(avg_corr), float(pvalue)


def analyze_regime_performance(
    returns: np.ndarray,
    dates: List[pd.Timestamp],
    spy_returns: Optional[np.ndarray] = None,
    vix_levels: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Analyze performance by market regime.

    Parameters
    ----------
    returns : array
        Portfolio returns.
    dates : list
        Corresponding dates.
    spy_returns : array, optional
        SPY returns for regime classification.
    vix_levels : array, optional
        VIX levels for regime classification.

    Returns
    -------
    regime_stats : DataFrame
        Performance by regime.
    """
    df = pd.DataFrame({
        'date': dates,
        'return': returns
    })

    regimes = []

    # Market direction regime (if SPY returns provided)
    if spy_returns is not None:
        df['spy_return'] = spy_returns[:len(returns)]
        df['market_regime'] = np.where(
            df['spy_return'].rolling(4).mean() > 0,
            'bull',
            'bear'
        )
        regimes.append('market_regime')

    # Volatility regime (if VIX provided)
    if vix_levels is not None:
        df['vix'] = vix_levels[:len(returns)]
        df['vol_regime'] = np.where(df['vix'] > 20, 'high_vol', 'low_vol')
        regimes.append('vol_regime')

    if not regimes:
        # Default: use return volatility as proxy
        rolling_vol = pd.Series(returns).rolling(12).std()
        median_vol = rolling_vol.median()
        df['vol_regime'] = np.where(rolling_vol > median_vol, 'high_vol', 'low_vol')
        regimes.append('vol_regime')

    # Compute stats by regime
    stats_list = []
    for regime_col in regimes:
        for regime_val in df[regime_col].dropna().unique():
            mask = df[regime_col] == regime_val
            regime_returns = df.loc[mask, 'return']

            if len(regime_returns) > 0:
                stats_list.append({
                    'regime_type': regime_col,
                    'regime_value': regime_val,
                    'n_weeks': len(regime_returns),
                    'mean_return': regime_returns.mean(),
                    'std_return': regime_returns.std(),
                    'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(52) if regime_returns.std() > 0 else 0,
                    'total_return': (1 + regime_returns).prod() - 1,
                    'win_rate': (regime_returns > 0).mean()
                })

    return pd.DataFrame(stats_list)


def compute_rolling_metrics(
    returns: np.ndarray,
    dates: List[pd.Timestamp],
    window: int = 26
) -> pd.DataFrame:
    """
    Compute rolling performance metrics.

    Parameters
    ----------
    returns : array
        Portfolio returns.
    dates : list
        Corresponding dates.
    window : int
        Rolling window size in weeks.

    Returns
    -------
    rolling_metrics : DataFrame
        Rolling Sharpe, return, volatility.
    """
    df = pd.DataFrame({
        'date': dates,
        'return': returns
    })
    df.set_index('date', inplace=True)

    # Rolling metrics
    df['rolling_return'] = df['return'].rolling(window).mean() * 52  # Annualized
    df['rolling_vol'] = df['return'].rolling(window).std() * np.sqrt(52)  # Annualized
    df['rolling_sharpe'] = df['rolling_return'] / df['rolling_vol']

    # Cumulative return
    df['cumulative'] = (1 + df['return']).cumprod()

    # Drawdown
    df['peak'] = df['cumulative'].cummax()
    df['drawdown'] = (df['cumulative'] - df['peak']) / df['peak']

    return df


def compute_train_test_gap(
    train_errors: List[float],
    test_errors: List[float]
) -> Dict:
    """
    Analyze the gap between training and test errors.

    Parameters
    ----------
    train_errors : list
        Training MSE for each fold.
    test_errors : list
        Test MSE for each fold.

    Returns
    -------
    gap_stats : dict
        Statistics about the train-test gap.
    """
    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)

    gap = test_errors - train_errors
    ratio = test_errors / (train_errors + 1e-10)

    return {
        'avg_train_error': float(np.mean(train_errors)),
        'avg_test_error': float(np.mean(test_errors)),
        'avg_gap': float(np.mean(gap)),
        'gap_std': float(np.std(gap)),
        'avg_ratio': float(np.mean(ratio)),
        'correlation': float(np.corrcoef(train_errors, test_errors)[0, 1]) if len(train_errors) > 1 else 0.0
    }


def create_diagnostic_report(
    returns: np.ndarray,
    dates: List[pd.Timestamp],
    predictions: np.ndarray,
    actuals: np.ndarray,
    top_k: int = 3
) -> DiagnosticResult:
    """
    Create comprehensive diagnostic report.

    Parameters
    ----------
    returns : array
        Portfolio returns.
    dates : list
        Corresponding dates.
    predictions : array
        Model predictions.
    actuals : array
        Actual values.
    top_k : int
        Number of top picks.

    Returns
    -------
    result : DiagnosticResult
    """
    # Time period analysis
    yearly, quarterly, worst = analyze_time_periods(returns, dates)

    # Prediction quality
    pred_quality = analyze_prediction_quality(predictions, actuals, top_k)
    rank_corr, rank_p = compute_rank_correlation(predictions, actuals)

    # Hit rate by year
    df = pd.DataFrame({'date': dates, 'return': returns})
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    # Compute hit rate per year
    n_weeks = predictions.shape[0]
    hits_per_year = {}
    for week in range(min(n_weeks, len(dates))):
        year = pd.Timestamp(dates[week]).year
        if year not in hits_per_year:
            hits_per_year[year] = {'hits': 0, 'total': 0}

        top_picks = np.argsort(predictions[week])[-top_k:]
        for pick in top_picks:
            if actuals[week, pick] > 0:
                hits_per_year[year]['hits'] += 1
            hits_per_year[year]['total'] += 1

    hit_rate_by_year = pd.Series({
        year: v['hits'] / v['total'] if v['total'] > 0 else 0
        for year, v in hits_per_year.items()
    })

    return DiagnosticResult(
        yearly_returns=yearly,
        quarterly_returns=quarterly,
        worst_periods=worst,
        rank_correlation=rank_corr,
        rank_correlation_pvalue=rank_p,
        prediction_std=pred_quality['prediction_std'],
        hit_rate_by_period=hit_rate_by_year
    )
