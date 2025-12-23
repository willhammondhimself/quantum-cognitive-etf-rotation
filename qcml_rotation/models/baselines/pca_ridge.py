"""
PCA + Ridge Regression baseline model.

Reduces feature dimensionality with PCA, then applies ridge regression.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class PCARidgeConfig:
    """Configuration for PCA + Ridge model."""
    n_components: int = 5
    alpha: float = 1.0


class PCARidge:
    """
    PCA + Ridge Regression model for excess return prediction.

    Two-stage approach:
    1. Fit PCA on training features
    2. Fit ridge regression on principal components
    """

    def __init__(self, config: Optional[PCARidgeConfig] = None):
        if config is None:
            config = PCARidgeConfig()

        self.config = config
        self.pca = PCA(n_components=config.n_components)
        self.ridge = Ridge(alpha=config.alpha)
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Fit PCA and Ridge on training data.

        Parameters
        ----------
        X_train : array of shape (n_samples, n_features)
            Training features.
        y_train : array of shape (n_samples,)
            Training labels.
        X_val : array, optional
            Validation features.
        y_val : array, optional
            Validation labels.

        Returns
        -------
        metrics : dict
            Training and validation MSE.
        """
        # Fit PCA
        X_train_pca = self.pca.fit_transform(X_train)

        # Fit Ridge
        self.ridge.fit(X_train_pca, y_train)
        self.is_fitted = True

        # Compute metrics
        train_pred = self.ridge.predict(X_train_pca)
        train_mse = mean_squared_error(y_train, train_pred)

        metrics = {"train_mse": train_mse}

        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)
            metrics["val_mse"] = val_mse

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict excess returns.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        predictions : array of shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_pca = self.pca.transform(X)
        return self.ridge.predict(X_pca)

    def get_explained_variance(self) -> np.ndarray:
        """Get explained variance ratio for each principal component."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.pca.explained_variance_ratio_

    def get_coefficients(self) -> Tuple[np.ndarray, float]:
        """Get ridge regression coefficients and intercept."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.ridge.coef_, self.ridge.intercept_


def train_pca_ridge(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    feature_cols: list,
    label_col: str = "excess_return",
    config: Optional[PCARidgeConfig] = None
) -> Tuple[PCARidge, Dict[str, float]]:
    """
    Convenience function to train PCA + Ridge from DataFrames.

    Parameters
    ----------
    train_data : pd.DataFrame
        Training data with features and labels.
    val_data : pd.DataFrame
        Validation data.
    feature_cols : list
        Feature column names.
    label_col : str
        Label column name.
    config : PCARidgeConfig, optional
        Model configuration.

    Returns
    -------
    model : fitted PCARidge model
    metrics : training and validation metrics
    """
    X_train = train_data[feature_cols].values
    y_train = train_data[label_col].values
    X_val = val_data[feature_cols].values
    y_val = val_data[label_col].values

    model = PCARidge(config)
    metrics = model.fit(X_train, y_train, X_val, y_val)

    print(f"PCA + Ridge - Train MSE: {metrics['train_mse']:.6f}, Val MSE: {metrics['val_mse']:.6f}")
    print(f"Explained variance: {model.get_explained_variance().sum():.2%}")

    return model, metrics
