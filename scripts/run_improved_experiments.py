#!/usr/bin/env python3
"""
Comprehensive experiments with improved models and strategies.

Key improvements tested:
1. XGBoost/LightGBM vs neural networks
2. Ranking objective vs regression
3. Trend-following overlay
4. Dual momentum strategy
5. Monthly vs weekly rebalancing
6. Macro regime features (VIX, credit, yield curve)
7. Hybrid strategy (ML + systematic momentum)

Run with:
    python scripts/run_improved_experiments.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch

# Data and features
from qcml_rotation.data.loader import (
    download_etf_data, download_vix_data, get_trading_dates,
    download_treasury_data, download_credit_spread_data
)
from qcml_rotation.data.features import (
    build_features, compute_labels, merge_features_labels,
    FeatureConfig, get_feature_names
)
from qcml_rotation.data.dataset import create_data_splits, ETFDataset

# Models
from qcml_rotation.models.baselines.pca_ridge import PCARidge, PCARidgeConfig
from qcml_rotation.models.baselines.mlp import MLP

# Strategies
from qcml_rotation.strategies.momentum import (
    DualMomentumStrategy, MomentumConfig,
    TrendFollowingOverlay, RelativeStrengthRanker,
    HybridStrategy, compute_monthly_rebalance_dates
)

# Backtest
from qcml_rotation.backtest.portfolio import PortfolioConfig
from qcml_rotation.backtest.walk_forward import WalkForwardValidator, compare_walk_forward_results
from qcml_rotation.backtest.metrics import compute_significance, permutation_test

from qcml_rotation.utils.helpers import load_config, set_seed, get_device

# Try importing gradient boosting
try:
    from qcml_rotation.models.gradient_boosting import (
        XGBoostModel, LightGBMModel, GradientBoostConfig,
        GradientBoostingEnsemble, compute_ranking_groups,
        HAS_XGBOOST, HAS_LIGHTGBM
    )
except ImportError:
    HAS_XGBOOST = False
    HAS_LIGHTGBM = False


def prepare_data_with_macro(config: dict, force_refresh: bool = False) -> Tuple:
    """
    Prepare data with extended macro features.

    Returns
    -------
    prices, features_df, labels_df, rebalance_dates, feature_cols
    """
    print("\n" + "="*60)
    print("Preparing Data with Extended Macro Features")
    print("="*60)

    # Download price data
    prices, volume = download_etf_data(
        tickers=config["tickers"]["etfs"],
        benchmark=config["tickers"]["benchmark"],
        start_date=config["data"]["start_date"],
        cache_dir=config["data"]["cache_dir"],
        force_refresh=force_refresh
    )

    # Download macro data
    vix_data = download_vix_data(
        start_date=config["data"]["start_date"],
        cache_dir=config["data"]["cache_dir"],
        force_refresh=force_refresh
    )

    treasury_data = download_treasury_data(
        start_date=config["data"]["start_date"],
        cache_dir=config["data"]["cache_dir"],
        force_refresh=force_refresh
    )

    credit_data = download_credit_spread_data(
        start_date=config["data"]["start_date"],
        cache_dir=config["data"]["cache_dir"],
        force_refresh=force_refresh
    )

    # Get rebalance dates (weekly)
    weekly_dates = get_trading_dates(prices, freq="W-FRI")

    # Get monthly rebalance dates
    monthly_dates = compute_monthly_rebalance_dates(prices)

    print(f"Weekly rebalance dates: {len(weekly_dates)}")
    print(f"Monthly rebalance dates: {len(monthly_dates)}")

    # Build features with extended set
    feature_config = FeatureConfig(
        return_windows=[1, 5, 20],
        vol_window=20,
        benchmark=config["tickers"]["benchmark"],
        include_technical=True,
        include_cross_sectional=True,
        include_regime=True,
        momentum_windows=[20, 60]
    )

    features = build_features(
        prices, weekly_dates, feature_config, vix_data=vix_data
    )
    labels = compute_labels(
        prices, weekly_dates,
        benchmark=config["tickers"]["benchmark"],
        forward_days=5
    )

    data = merge_features_labels(features, labels)

    # Get feature columns
    feature_cols = get_feature_names(feature_config, include_vix=True)

    # Add macro features to each row
    extended_data = add_macro_features(data, treasury_data, credit_data)
    extended_feature_cols = feature_cols + get_macro_feature_names()

    # Filter to columns that exist
    extended_feature_cols = [c for c in extended_feature_cols if c in extended_data.columns]

    print(f"Total features: {len(extended_feature_cols)}")
    print(f"Total samples: {len(extended_data)}")

    return prices, extended_data, weekly_dates, monthly_dates, extended_feature_cols


def add_macro_features(
    data: pd.DataFrame,
    treasury_data: pd.DataFrame,
    credit_data: pd.DataFrame
) -> pd.DataFrame:
    """Add macro features to the dataset."""
    result = data.copy()

    # Get dates from data index
    dates = result.index.get_level_values("date")

    # Add treasury features
    if len(treasury_data) > 0:
        treasury_cols = ["yield_curve_slope", "yield_curve_change",
                         "yield_curve_inverted", "duration_risk"]
        for col in treasury_cols:
            if col in treasury_data.columns:
                # Map to each row by date
                result[col] = dates.map(
                    lambda d: treasury_data[col].get(d, np.nan)
                    if d in treasury_data.index else np.nan
                )

    # Add credit features
    if len(credit_data) > 0:
        credit_cols = ["hyg_lqd_ratio", "credit_appetite_change",
                       "credit_stress", "jnk_tlt_ratio", "spread_change",
                       "hyg_momentum", "hyg_volatility"]
        for col in credit_cols:
            if col in credit_data.columns:
                result[col] = dates.map(
                    lambda d: credit_data[col].get(d, np.nan)
                    if d in credit_data.index else np.nan
                )

    # Forward fill and clean
    result = result.ffill().bfill()
    result = result.dropna()

    return result


def get_macro_feature_names() -> List[str]:
    """Get list of macro feature column names."""
    return [
        # Treasury
        "yield_curve_slope", "yield_curve_change",
        "yield_curve_inverted", "duration_risk",
        # Credit
        "hyg_lqd_ratio", "credit_appetite_change",
        "credit_stress", "jnk_tlt_ratio", "spread_change",
        "hyg_momentum", "hyg_volatility"
    ]


def create_model_factory(
    model_type: str,
    input_dim: int,
    config: dict,
    device: str
):
    """
    Create train_fn and predict_fn for a model type.

    Supports:
    - pca_ridge: PCA + Ridge regression
    - xgboost: XGBoost regressor
    - xgboost_rank: XGBoost with ranking objective
    - lightgbm: LightGBM regressor
    - lightgbm_rank: LightGBM with ranking objective
    - mlp: Neural network
    - ensemble_gb: Ensemble of gradient boosting models
    """
    if model_type == "pca_ridge":
        model = PCARidge(PCARidgeConfig(
            n_components=config.get("pca_components", 5),
            alpha=config.get("ridge_alpha", 1.0)
        ))

        def train_fn(X, y):
            model.fit(X, y)

        def predict_fn(X):
            return model.predict(X)

        return train_fn, predict_fn

    elif model_type == "xgboost" and HAS_XGBOOST:
        gb_config = GradientBoostConfig(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 4),
            learning_rate=config.get("learning_rate", 0.05),
            objective="regression"
        )

        def train_fn(X, y):
            nonlocal model
            model = XGBoostModel(gb_config)
            model.fit(X, y)

        def predict_fn(X):
            return model.predict(X)

        model = None
        return train_fn, predict_fn

    elif model_type == "xgboost_rank" and HAS_XGBOOST:
        gb_config = GradientBoostConfig(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 4),
            learning_rate=config.get("learning_rate", 0.05),
            objective="rank"
        )

        def train_fn(X, y, date_indices=None):
            nonlocal model
            model = XGBoostModel(gb_config)
            if date_indices is not None:
                sorted_idx, groups = compute_ranking_groups(date_indices)
                X_sorted = X[sorted_idx]
                y_sorted = y[sorted_idx]
                model.fit(X_sorted, y_sorted, group=groups)
            else:
                model.fit(X, y)

        def predict_fn(X):
            return model.predict(X)

        model = None
        return train_fn, predict_fn

    elif model_type == "lightgbm" and HAS_LIGHTGBM:
        gb_config = GradientBoostConfig(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 4),
            learning_rate=config.get("learning_rate", 0.05),
            objective="regression"
        )

        def train_fn(X, y):
            nonlocal model
            model = LightGBMModel(gb_config)
            model.fit(X, y)

        def predict_fn(X):
            return model.predict(X)

        model = None
        return train_fn, predict_fn

    elif model_type == "lightgbm_rank" and HAS_LIGHTGBM:
        gb_config = GradientBoostConfig(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 4),
            learning_rate=config.get("learning_rate", 0.05),
            objective="rank"
        )

        def train_fn(X, y, date_indices=None):
            nonlocal model
            model = LightGBMModel(gb_config)
            if date_indices is not None:
                sorted_idx, groups = compute_ranking_groups(date_indices)
                X_sorted = X[sorted_idx]
                y_sorted = y[sorted_idx]
                model.fit(X_sorted, y_sorted, group=groups)
            else:
                model.fit(X, y)

        def predict_fn(X):
            return model.predict(X)

        model = None
        return train_fn, predict_fn

    elif model_type == "ensemble_gb" and (HAS_XGBOOST or HAS_LIGHTGBM):
        def train_fn(X, y):
            nonlocal model
            model = GradientBoostingEnsemble(
                use_xgb=HAS_XGBOOST,
                use_lgb=HAS_LIGHTGBM
            )
            model.fit(X, y)

        def predict_fn(X):
            return model.predict(X)

        model = None
        return train_fn, predict_fn

    elif model_type == "mlp":
        lr = config.get("mlp_lr", 0.001)
        epochs = config.get("mlp_epochs", 50)
        hidden_dims = config.get("mlp_hidden", [64, 32])

        def train_fn(X, y):
            nonlocal model
            model = MLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=config.get("mlp_dropout", 0.2)
            )
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = torch.nn.MSELoss()

            X_t = torch.FloatTensor(X).to(device)
            y_t = torch.FloatTensor(y).to(device)

            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                preds = model(X_t).squeeze()
                loss = criterion(preds, y_t)
                loss.backward()
                optimizer.step()

        def predict_fn(X):
            model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(device)
                return model(X_t).squeeze().cpu().numpy()

        model = MLP(input_dim=input_dim, hidden_dims=hidden_dims)
        model.to(device)
        return train_fn, predict_fn

    else:
        raise ValueError(f"Unknown or unavailable model type: {model_type}")


def run_pure_momentum_strategy(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    benchmark: str,
    config: MomentumConfig,
    strategy: str = "long_short"
) -> Dict:
    """
    Run pure dual momentum strategy (no ML).

    This serves as a baseline to compare against ML approaches.
    """
    print("\nRunning Pure Dual Momentum Strategy...")

    momentum_strategy = DualMomentumStrategy(config)
    tickers = [c for c in prices.columns if c != benchmark]

    returns = []
    dates_used = []

    for i, date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]

        # Get signals
        signals = momentum_strategy.compute_signals(prices, date, benchmark)

        if not signals:
            continue

        # Generate weights
        weights = momentum_strategy.generate_weights(
            signals, prices, date, strategy=strategy
        )

        # Compute portfolio return
        period_return = 0
        for ticker, weight in weights.items():
            if ticker in prices.columns:
                price_start = prices.loc[date, ticker]
                price_end = prices.loc[next_date, ticker]
                if price_start > 0:
                    ticker_return = (price_end / price_start) - 1
                    period_return += weight * ticker_return

        returns.append(period_return)
        dates_used.append(date)

    returns = np.array(returns)

    # Compute metrics
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (52 / len(returns)) - 1 if len(returns) > 0 else 0
    volatility = returns.std() * np.sqrt(52) if len(returns) > 1 else 0
    sharpe = ann_return / volatility if volatility > 0 else 0
    max_dd = compute_max_drawdown(returns)

    return {
        "name": "DualMomentum",
        "returns": returns,
        "dates": dates_used,
        "total_return": total_return,
        "ann_return": ann_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "n_periods": len(returns)
    }


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from returns series."""
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def run_hybrid_strategy(
    prices: pd.DataFrame,
    data: pd.DataFrame,
    feature_cols: List[str],
    rebalance_dates: pd.DatetimeIndex,
    benchmark: str,
    model_type: str,
    model_config: dict,
    hybrid_config: dict,
    device: str
) -> Dict:
    """
    Run hybrid strategy combining ML with momentum.
    """
    print(f"\nRunning Hybrid Strategy with {model_type}...")

    tickers = [c for c in prices.columns if c != benchmark]
    hybrid = HybridStrategy(
        ml_weight=hybrid_config.get("ml_weight", 0.3),
        momentum_weight=hybrid_config.get("momentum_weight", 0.7),
        use_trend_filter=hybrid_config.get("use_trend_filter", True)
    )

    # Get unique dates in data
    unique_dates = data.index.get_level_values("date").unique().sort_values()
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    # Prepare data arrays
    X = data[feature_cols].values
    y = data["excess_return"].values
    sample_dates = data.index.get_level_values("date")
    sample_tickers = data.index.get_level_values("ticker")

    date_indices = np.array([date_to_idx.get(d, -1) for d in sample_dates])

    input_dim = X.shape[1]

    returns = []
    dates_used = []

    # Walk forward with rolling training
    min_train = 52  # 1 year minimum

    for i, date in enumerate(rebalance_dates[min_train:-1]):
        if date not in unique_dates:
            continue

        next_idx = i + min_train + 1
        if next_idx >= len(rebalance_dates):
            break
        next_date = rebalance_dates[next_idx]

        # Training data: all data before current date
        train_mask = sample_dates < date
        X_train = X[train_mask]
        y_train = y[train_mask]

        if len(X_train) < 50:
            continue

        # Train model
        train_fn, predict_fn = create_model_factory(
            model_type, input_dim, model_config, device
        )
        train_fn(X_train, y_train)

        # Get predictions for current date
        current_mask = sample_dates == date
        X_current = X[current_mask]
        tickers_current = sample_tickers[current_mask]

        if len(X_current) == 0:
            continue

        predictions = predict_fn(X_current)
        ml_pred_dict = dict(zip(tickers_current, predictions))

        # Combine with momentum signals
        combined_signals = hybrid.combine_signals(
            ml_pred_dict, prices, date, benchmark
        )

        if not combined_signals:
            continue

        # Select top/bottom k
        sorted_signals = sorted(combined_signals.items(), key=lambda x: x[1], reverse=True)
        top_k = 3
        bottom_k = 3

        weights = {}
        for ticker, _ in sorted_signals[:top_k]:
            weights[ticker] = 1.0 / top_k
        for ticker, _ in sorted_signals[-bottom_k:]:
            if ticker not in weights:
                weights[ticker] = -1.0 / bottom_k

        # Compute return
        period_return = 0
        for ticker, weight in weights.items():
            if ticker in prices.columns and next_date in prices.index:
                price_start = prices.loc[date, ticker]
                price_end = prices.loc[next_date, ticker]
                if price_start > 0:
                    ticker_return = (price_end / price_start) - 1
                    period_return += weight * ticker_return

        returns.append(period_return)
        dates_used.append(date)

    returns = np.array(returns)

    # Compute metrics
    if len(returns) == 0:
        return {"name": f"Hybrid_{model_type}", "sharpe_ratio": 0}

    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (52 / len(returns)) - 1 if len(returns) > 0 else 0
    volatility = returns.std() * np.sqrt(52) if len(returns) > 1 else 0
    sharpe = ann_return / volatility if volatility > 0 else 0
    max_dd = compute_max_drawdown(returns)

    return {
        "name": f"Hybrid_{model_type}",
        "returns": returns,
        "dates": dates_used,
        "total_return": total_return,
        "ann_return": ann_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "n_periods": len(returns)
    }


def main(args):
    """Run comprehensive experiments."""
    print("="*60)
    print("QCML ETF Rotation - Improved Experiments")
    print("="*60)
    print(f"Start time: {datetime.now().isoformat()}")

    # Load config
    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = get_device()
    print(f"Device: {device}")

    # Prepare data with macro features
    prices, data, weekly_dates, monthly_dates, feature_cols = prepare_data_with_macro(
        config, force_refresh=args.force_refresh
    )

    print(f"\nFeatures: {feature_cols}")
    print(f"Data shape: {data.shape}")

    results = {}

    # =========================================================================
    # 1. Pure Momentum Strategy (Baseline)
    # =========================================================================
    momentum_config = MomentumConfig(
        fast_lookback=21,
        slow_lookback=63,
        long_lookback=252,
        use_trend_filter=True,
        use_vol_scaling=True,
        top_k=3,
        bottom_k=3
    )

    momentum_result = run_pure_momentum_strategy(
        prices, weekly_dates,
        benchmark=config["tickers"]["benchmark"],
        config=momentum_config,
        strategy="long_short"
    )
    results["DualMomentum"] = momentum_result
    print(f"\nDual Momentum Sharpe: {momentum_result['sharpe_ratio']:.3f}")

    # =========================================================================
    # 2. Gradient Boosting Models
    # =========================================================================
    model_configs = {
        "xgboost": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
        "xgboost_rank": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
        "lightgbm": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
        "lightgbm_rank": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
        "pca_ridge": {"pca_components": 5, "ridge_alpha": 1.0},
        "mlp": {"mlp_lr": 0.001, "mlp_epochs": 50, "mlp_hidden": [64, 32]},
    }

    # Filter available models
    available_models = {}
    for name, cfg in model_configs.items():
        if "xgboost" in name and not HAS_XGBOOST:
            continue
        if "lightgbm" in name and not HAS_LIGHTGBM:
            continue
        available_models[name] = cfg

    print(f"\nAvailable models: {list(available_models.keys())}")

    # =========================================================================
    # 3. Hybrid Strategies (ML + Momentum)
    # =========================================================================
    hybrid_configs = [
        {"ml_weight": 0.3, "momentum_weight": 0.7, "use_trend_filter": True},
        {"ml_weight": 0.5, "momentum_weight": 0.5, "use_trend_filter": True},
        {"ml_weight": 0.7, "momentum_weight": 0.3, "use_trend_filter": True},
    ]

    for model_name, model_cfg in available_models.items():
        for i, hybrid_cfg in enumerate(hybrid_configs):
            result = run_hybrid_strategy(
                prices, data, feature_cols, weekly_dates,
                benchmark=config["tickers"]["benchmark"],
                model_type=model_name,
                model_config=model_cfg,
                hybrid_config=hybrid_cfg,
                device=device
            )
            key = f"Hybrid_{model_name}_ml{int(hybrid_cfg['ml_weight']*100)}"
            results[key] = result
            print(f"{key} Sharpe: {result.get('sharpe_ratio', 0):.3f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    summary_rows = []
    for name, result in results.items():
        if isinstance(result, dict) and "sharpe_ratio" in result:
            summary_rows.append({
                "Strategy": name,
                "Sharpe": result.get("sharpe_ratio", 0),
                "Ann Return": result.get("ann_return", 0),
                "Volatility": result.get("volatility", 0),
                "Max DD": result.get("max_drawdown", 0),
                "N Periods": result.get("n_periods", 0)
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("Sharpe", ascending=False)
    print(summary_df.to_string(index=False))

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary_df.to_csv(output_dir / "improved_experiments_summary.csv", index=False)

    # Save detailed results
    results_serializable = {}
    for name, result in results.items():
        if isinstance(result, dict):
            results_serializable[name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in result.items()
                if k != "dates"  # Skip datetime objects
            }

    with open(output_dir / "improved_experiments_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")

    # Best strategy
    if len(summary_df) > 0:
        best = summary_df.iloc[0]
        print(f"\nBest Strategy: {best['Strategy']} with Sharpe {best['Sharpe']:.3f}")

    print(f"\nEnd time: {datetime.now().isoformat()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run improved experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/improved_experiments",
        help="Directory to save results"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download of data"
    )

    args = parser.parse_args()
    main(args)
