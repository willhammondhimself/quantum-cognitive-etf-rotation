"""
Gradient boosting models for ETF rotation.

XGBoost and LightGBM are often better than neural networks for:
- Small datasets (limited history)
- Tabular data with mixed feature types
- When interpretability matters

These models also support:
- Built-in feature importance
- Native handling of missing values
- Better regularization for small samples
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field
import warnings

# Try importing gradient boosting libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Install with: pip install lightgbm")


@dataclass
class GradientBoostConfig:
    """Configuration for gradient boosting models."""
    # Common parameters
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    random_state: int = 42

    # Objective: 'regression', 'rank' (for pairwise ranking)
    objective: str = 'regression'

    # Early stopping
    early_stopping_rounds: int = 20
    eval_metric: str = 'rmse'

    # For ranking objective
    n_groups: Optional[int] = None  # Number of ranking groups


class XGBoostModel:
    """
    XGBoost wrapper for ETF return prediction.

    Supports both regression and ranking (pairwise) objectives.
    """

    def __init__(self, config: Optional[GradientBoostConfig] = None):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        self.config = config or GradientBoostConfig()
        self.model = None
        self.feature_importance_ = None

    def _get_params(self) -> Dict:
        """Convert config to XGBoost parameters."""
        params = {
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'min_child_weight': self.config.min_child_weight,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'random_state': self.config.random_state,
            'n_jobs': -1,
            'verbosity': 0,
        }

        if self.config.objective == 'rank':
            params['objective'] = 'rank:pairwise'
            params['eval_metric'] = 'ndcg'
        else:
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'

        return params

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        group: Optional[np.ndarray] = None,
        group_val: Optional[np.ndarray] = None,
    ) -> 'XGBoostModel':
        """
        Fit the XGBoost model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training targets.
        X_val : np.ndarray, optional
            Validation features for early stopping.
        y_val : np.ndarray, optional
            Validation targets.
        group : np.ndarray, optional
            Group sizes for ranking (each group is one date).
        group_val : np.ndarray, optional
            Validation group sizes.
        """
        params = self._get_params()

        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)

        if self.config.objective == 'rank' and group is not None:
            dtrain.set_group(group)

        evals = [(dtrain, 'train')]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            if self.config.objective == 'rank' and group_val is not None:
                dval.set_group(group_val)
            evals.append((dval, 'val'))

        # Train with early stopping
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.n_estimators,
            evals=evals,
            early_stopping_rounds=self.config.early_stopping_rounds if X_val is not None else None,
            verbose_eval=False,
        )

        # Store feature importance
        self.feature_importance_ = self.model.get_score(importance_type='gain')

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.feature_importance_ is None:
            return {}

        importance = {}
        for key, value in self.feature_importance_.items():
            if feature_names is not None:
                # Map feature index to name
                idx = int(key.replace('f', ''))
                if idx < len(feature_names):
                    importance[feature_names[idx]] = value
            else:
                importance[key] = value

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


class LightGBMModel:
    """
    LightGBM wrapper for ETF return prediction.

    Often faster than XGBoost and handles categorical features natively.
    """

    def __init__(self, config: Optional[GradientBoostConfig] = None):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

        self.config = config or GradientBoostConfig()
        self.model = None
        self.feature_importance_ = None

    def _get_params(self) -> Dict:
        """Convert config to LightGBM parameters."""
        params = {
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'min_child_weight': self.config.min_child_weight,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'random_state': self.config.random_state,
            'n_jobs': -1,
            'verbose': -1,
            'force_col_wise': True,
        }

        if self.config.objective == 'rank':
            params['objective'] = 'lambdarank'
            params['metric'] = 'ndcg'
            params['ndcg_eval_at'] = [3, 5]  # Evaluate NDCG@3 and NDCG@5
        else:
            params['objective'] = 'regression'
            params['metric'] = 'rmse'

        return params

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        group: Optional[np.ndarray] = None,
        group_val: Optional[np.ndarray] = None,
    ) -> 'LightGBMModel':
        """
        Fit the LightGBM model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training targets.
        X_val : np.ndarray, optional
            Validation features for early stopping.
        y_val : np.ndarray, optional
            Validation targets.
        group : np.ndarray, optional
            Group sizes for ranking (each group is one date).
        group_val : np.ndarray, optional
            Validation group sizes.
        """
        params = self._get_params()

        # Create datasets
        train_data = lgb.Dataset(X, label=y)
        if self.config.objective == 'rank' and group is not None:
            train_data.set_group(group)

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            if self.config.objective == 'rank' and group_val is not None:
                val_data.set_group(group_val)
            valid_sets.append(val_data)
            valid_names.append('val')

        # Train with early stopping
        callbacks = [lgb.log_evaluation(period=0)]  # Suppress logging
        if X_val is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.config.early_stopping_rounds,
                    verbose=False
                )
            )

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Store feature importance
        self.feature_importance_ = dict(zip(
            range(X.shape[1]),
            self.model.feature_importance(importance_type='gain')
        ))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(X)

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.feature_importance_ is None:
            return {}

        importance = {}
        for idx, value in self.feature_importance_.items():
            if feature_names is not None and idx < len(feature_names):
                importance[feature_names[idx]] = value
            else:
                importance[f'f{idx}'] = value

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


class GradientBoostingEnsemble:
    """
    Ensemble of XGBoost and LightGBM models.

    Combines predictions from multiple models for more robust results.
    """

    def __init__(
        self,
        configs: Optional[List[GradientBoostConfig]] = None,
        use_xgb: bool = True,
        use_lgb: bool = True,
    ):
        """
        Initialize ensemble.

        Parameters
        ----------
        configs : list of GradientBoostConfig, optional
            Configurations for each model. If None, uses default with variations.
        use_xgb : bool
            Include XGBoost models.
        use_lgb : bool
            Include LightGBM models.
        """
        self.models = []
        self.weights = []

        if configs is None:
            # Create diverse configurations
            configs = [
                GradientBoostConfig(max_depth=3, learning_rate=0.05, n_estimators=100),
                GradientBoostConfig(max_depth=4, learning_rate=0.03, n_estimators=150),
                GradientBoostConfig(max_depth=5, learning_rate=0.02, n_estimators=200),
            ]

        for i, config in enumerate(configs):
            config.random_state = 42 + i

            if use_xgb and HAS_XGBOOST:
                self.models.append(('xgb', XGBoostModel(config)))
                self.weights.append(1.0)

            if use_lgb and HAS_LIGHTGBM:
                self.models.append(('lgb', LightGBMModel(config)))
                self.weights.append(1.0)

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'GradientBoostingEnsemble':
        """Fit all models in the ensemble."""
        for name, model in self.models:
            model.fit(X, y, X_val, y_val, **kwargs)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of all model predictions."""
        predictions = []

        for (name, model), weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(weight * pred)

        return np.sum(predictions, axis=0)

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Average feature importance across all models."""
        all_importance = {}

        for name, model in self.models:
            importance = model.get_feature_importance(feature_names)
            for feat, value in importance.items():
                if feat not in all_importance:
                    all_importance[feat] = []
                all_importance[feat].append(value)

        # Average importance
        avg_importance = {
            feat: np.mean(values)
            for feat, values in all_importance.items()
        }

        return dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))


def compute_ranking_groups(
    date_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert date indices to group sizes for ranking.

    For XGBoost/LightGBM ranking objectives, we need group sizes
    rather than group indices.

    Parameters
    ----------
    date_indices : np.ndarray
        Array of date indices for each sample.

    Returns
    -------
    sorted_indices : np.ndarray
        Indices to sort samples by date.
    group_sizes : np.ndarray
        Number of samples in each group (date).
    """
    # Get unique dates and their counts
    unique_dates, counts = np.unique(date_indices, return_counts=True)

    # Sort by date
    sorted_indices = np.argsort(date_indices)

    # Group sizes (already in order after sorting)
    sort_order = np.argsort(unique_dates)
    group_sizes = counts[sort_order]

    return sorted_indices, group_sizes
