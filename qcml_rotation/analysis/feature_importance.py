"""
Feature importance analysis for identifying predictive features.

Uses multiple methods to assess feature predictive power:
1. Univariate correlation analysis
2. Permutation importance
3. Forward/backward feature selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


@dataclass
class FeatureImportanceResult:
    """Container for feature importance analysis results."""
    # Univariate analysis
    correlations: pd.DataFrame  # Feature correlations with target
    significant_features: List[str]  # Features with p < 0.05

    # Permutation importance
    permutation_importance: pd.DataFrame

    # Ranking
    overall_ranking: pd.DataFrame  # Combined ranking across methods


def compute_univariate_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Compute univariate correlation and significance for each feature.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    feature_names : list of feature names

    Returns
    -------
    results : DataFrame with correlation, p-value, and significance
    """
    results = []

    for i, name in enumerate(feature_names):
        feature = X[:, i]

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(feature, y)

        # Spearman correlation (rank-based, more robust)
        spearman_r, spearman_p = stats.spearmanr(feature, y)

        # Information coefficient (IC) - correlation with forward returns
        # Using rank correlation as IC approximation
        ic = spearman_r

        results.append({
            'feature': name,
            'pearson_corr': pearson_r,
            'pearson_pval': pearson_p,
            'spearman_corr': spearman_r,
            'spearman_pval': spearman_p,
            'ic': ic,
            'abs_ic': abs(ic),
            'significant': pearson_p < 0.05
        })

    df = pd.DataFrame(results)
    df = df.sort_values('abs_ic', ascending=False)

    return df


def compute_permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_fn: Optional[Callable] = None,
    n_permutations: int = 100,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation importance by measuring prediction degradation.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    feature_names : list of feature names
    model_fn : callable that returns fitted model, defaults to Ridge
    n_permutations : number of permutations per feature
    random_state : random seed

    Returns
    -------
    results : DataFrame with importance scores
    """
    np.random.seed(random_state)

    # Default model
    if model_fn is None:
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        baseline_preds = model.predict(X)
    else:
        model = model_fn(X, y)
        baseline_preds = model.predict(X)

    baseline_mse = mean_squared_error(y, baseline_preds)

    results = []

    for i, name in enumerate(feature_names):
        mse_increases = []

        for _ in range(n_permutations):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])

            perm_preds = model.predict(X_permuted)
            perm_mse = mean_squared_error(y, perm_preds)

            mse_increases.append(perm_mse - baseline_mse)

        mse_increases = np.array(mse_increases)

        results.append({
            'feature': name,
            'importance_mean': np.mean(mse_increases),
            'importance_std': np.std(mse_increases),
            'importance_median': np.median(mse_increases),
            # Significance: is importance significantly > 0?
            't_stat': np.mean(mse_increases) / (np.std(mse_increases) / np.sqrt(n_permutations)) if np.std(mse_increases) > 0 else 0,
            'significant': np.mean(mse_increases) > 2 * np.std(mse_increases)
        })

    df = pd.DataFrame(results)
    df = df.sort_values('importance_mean', ascending=False)

    return df


def forward_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    max_features: Optional[int] = None,
    cv_folds: int = 5
) -> Tuple[List[str], List[float]]:
    """
    Forward feature selection using cross-validated MSE.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    feature_names : list of feature names
    max_features : maximum features to select
    cv_folds : number of cross-validation folds

    Returns
    -------
    selected_features : list of selected feature names in order
    cv_scores : list of CV scores after each addition
    """
    if max_features is None:
        max_features = len(feature_names)

    n_samples = len(y)
    fold_size = n_samples // cv_folds

    available = list(range(len(feature_names)))
    selected = []
    cv_scores = []

    for _ in range(min(max_features, len(available))):
        best_score = float('inf')
        best_feature = None

        for feat_idx in available:
            current_features = selected + [feat_idx]
            X_subset = X[:, current_features]

            # Cross-validation
            fold_mses = []
            for fold in range(cv_folds):
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else n_samples

                val_idx = list(range(val_start, val_end))
                train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))

                X_train, X_val = X_subset[train_idx], X_subset[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)

                fold_mses.append(mean_squared_error(y_val, preds))

            avg_mse = np.mean(fold_mses)

            if avg_mse < best_score:
                best_score = avg_mse
                best_feature = feat_idx

        if best_feature is not None:
            selected.append(best_feature)
            available.remove(best_feature)
            cv_scores.append(best_score)

    selected_names = [feature_names[i] for i in selected]

    return selected_names, cv_scores


def compute_rolling_ic(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    date_indices: np.ndarray,
    window: int = 52
) -> pd.DataFrame:
    """
    Compute rolling Information Coefficient for each feature.

    IC = rank correlation between feature and forward returns

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    feature_names : list of feature names
    date_indices : array mapping samples to dates
    window : rolling window in weeks

    Returns
    -------
    rolling_ic : DataFrame with rolling IC for each feature
    """
    unique_dates = sorted(set(date_indices))
    n_features = len(feature_names)

    results = {name: [] for name in feature_names}
    results['date_idx'] = []

    for i, end_date in enumerate(unique_dates[window:], start=window):
        start_date = unique_dates[i - window]

        # Get samples in window
        mask = (date_indices >= start_date) & (date_indices < end_date)
        X_window = X[mask]
        y_window = y[mask]

        if len(y_window) < 10:  # Need minimum samples
            continue

        results['date_idx'].append(end_date)

        for j, name in enumerate(feature_names):
            if np.std(X_window[:, j]) > 0 and np.std(y_window) > 0:
                ic, _ = stats.spearmanr(X_window[:, j], y_window)
                results[name].append(ic if np.isfinite(ic) else 0)
            else:
                results[name].append(0)

    return pd.DataFrame(results)


def analyze_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_permutations: int = 50,
    random_state: int = 42
) -> FeatureImportanceResult:
    """
    Comprehensive feature importance analysis.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    feature_names : list of feature names
    n_permutations : number of permutations for importance
    random_state : random seed

    Returns
    -------
    result : FeatureImportanceResult
    """
    # Univariate analysis
    univariate = compute_univariate_importance(X, y, feature_names)
    significant = univariate[univariate['significant']]['feature'].tolist()

    # Permutation importance
    perm_importance = compute_permutation_importance(
        X, y, feature_names,
        n_permutations=n_permutations,
        random_state=random_state
    )

    # Create overall ranking
    # Merge univariate and permutation results
    ranking = univariate[['feature', 'abs_ic']].copy()
    ranking = ranking.merge(
        perm_importance[['feature', 'importance_mean']],
        on='feature'
    )

    # Rank by each method
    ranking['ic_rank'] = ranking['abs_ic'].rank(ascending=False)
    ranking['perm_rank'] = ranking['importance_mean'].rank(ascending=False)

    # Average rank
    ranking['avg_rank'] = (ranking['ic_rank'] + ranking['perm_rank']) / 2
    ranking = ranking.sort_values('avg_rank')

    return FeatureImportanceResult(
        correlations=univariate,
        significant_features=significant,
        permutation_importance=perm_importance,
        overall_ranking=ranking
    )


def select_best_features(
    importance_result: FeatureImportanceResult,
    method: str = 'top_n',
    n_features: int = 10,
    ic_threshold: float = 0.02
) -> List[str]:
    """
    Select best features based on importance analysis.

    Parameters
    ----------
    importance_result : FeatureImportanceResult
    method : 'top_n', 'ic_threshold', or 'significant'
    n_features : number of features for 'top_n'
    ic_threshold : minimum IC for 'ic_threshold'

    Returns
    -------
    selected : list of selected feature names
    """
    if method == 'top_n':
        return importance_result.overall_ranking['feature'].head(n_features).tolist()

    elif method == 'ic_threshold':
        df = importance_result.correlations
        return df[df['abs_ic'] >= ic_threshold]['feature'].tolist()

    elif method == 'significant':
        return importance_result.significant_features

    else:
        raise ValueError(f"Unknown method: {method}")
