#!/usr/bin/env python3
"""
Run walk-forward validation with statistical significance testing.

This provides a more realistic out-of-sample performance estimate
by retraining models at each time step.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pickle
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch

from qcml_rotation.data.dataset import ETFDataset
from qcml_rotation.backtest.walk_forward import (
    WalkForwardValidator, compare_walk_forward_results, walk_forward_result_to_dict
)
from qcml_rotation.backtest.metrics import (
    compute_significance, significance_to_dict, permutation_test
)
from qcml_rotation.backtest.portfolio import PortfolioConfig
from qcml_rotation.models.baselines.pca_ridge import PCARidge, PCARidgeConfig
from qcml_rotation.models.baselines.mlp import MLP
from qcml_rotation.models.qcml import QCML, QCMLConfig
from qcml_rotation.utils.helpers import load_config, set_seed, get_device


def create_model_and_fns(model_type: str, input_dim: int, config: dict, device=None):
    """
    Create a fresh model instance and its train/predict functions.

    Returns
    -------
    train_fn : callable
        Function that takes (X, y) and trains the model in-place.
    predict_fn : callable
        Function that takes X and returns predictions.
    """
    if model_type == "pca_ridge":
        model = PCARidge(PCARidgeConfig(
            n_components=config.get("pca_components", 3),
            alpha=config.get("ridge_alpha", 1.0)
        ))

        def train_fn(X, y):
            model.fit(X, y)

        def predict_fn(X):
            return model.predict(X)

        return train_fn, predict_fn

    elif model_type == "mlp":
        lr = config.get("mlp_lr", 0.001)
        epochs = config.get("mlp_epochs", 50)
        hidden_dims = config.get("mlp_hidden", [64, 32])

        def train_fn(X, y):
            nonlocal model
            # Create fresh model each time
            model = MLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=config.get("mlp_dropout", 0.2)
            )
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

        # Initialize model
        model = MLP(input_dim=input_dim, hidden_dims=hidden_dims)
        model.to(device)

        return train_fn, predict_fn

    elif model_type.startswith("qcml"):
        qcml_config = QCMLConfig(
            hilbert_dim=config.get("hilbert_dim", 16),
            encoder_hidden=config.get("encoder_hidden", 32),
            ranking_weight=config.get("ranking_weight", 0.3),
            use_complex="real_only" not in model_type
        )
        lr = config.get("qcml_lr", 0.001)
        epochs = config.get("qcml_epochs", 100)

        def train_fn(X, y):
            nonlocal model
            # Create fresh model each time
            model = QCML(input_dim=input_dim, config=qcml_config)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.MSELoss()

            X_t = torch.FloatTensor(X).to(device)
            y_t = torch.FloatTensor(y).to(device)

            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                preds = model(X_t)
                loss = criterion(preds, y_t)
                loss.backward()
                optimizer.step()

        def predict_fn(X):
            model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(device)
                return model(X_t).cpu().numpy()

        # Initialize model
        model = QCML(input_dim=input_dim, config=qcml_config)
        model.to(device)

        return train_fn, predict_fn

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main(args):
    """Run walk-forward validation."""
    print("=" * 60)
    print("QCML ETF Rotation - Walk-Forward Validation")
    print("=" * 60)

    # Load config
    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    # Load processed data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Run 'python scripts/run_data_prep.py' first.")
        sys.exit(1)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    splits = data["splits"]
    feature_cols = data["feature_cols"]
    prices = data["prices"]
    rebalance_dates = data["rebalance_dates"]

    print(f"\nLoaded data from {data_path}")

    # Combine train and test for walk-forward
    # (walk-forward will handle the splits internally)
    combined_data = pd.concat([splits.train, splits.val, splits.test])
    combined_data = combined_data.sort_index()

    print(f"Total samples: {len(combined_data)}")
    print(f"Total weeks: {len(rebalance_dates)}")

    # Create dataset
    full_dataset = ETFDataset(combined_data, feature_cols)

    # Build index mappings
    unique_dates = combined_data.index.get_level_values("date").unique().sort_values()
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    idx_to_date = {i: d for d, i in date_to_idx.items()}
    idx_to_ticker = {i: t for i, t in enumerate(full_dataset.tickers)}

    # Map date indices
    sample_dates = combined_data.index.get_level_values("date")
    date_indices = np.array([date_to_idx[d] for d in sample_dates])

    # Setup walk-forward validator
    portfolio_config = PortfolioConfig(
        top_k=config["backtest"]["top_k"],
        bottom_k=config["backtest"]["top_k"],
        strategy=config["backtest"]["strategy"],
        transaction_cost_bps=config["backtest"]["transaction_cost_bps"]
    )

    tickers = config["tickers"]["etfs"]

    validator = WalkForwardValidator(
        prices=prices,
        dates=list(unique_dates),
        tickers=tickers,
        min_train_weeks=args.min_train_weeks,
        portfolio_config=portfolio_config,
        benchmark=config["tickers"]["benchmark"]
    )

    # Define models to test
    model_configs = {
        "pca_ridge": {"type": "pca_ridge", "pca_components": 3, "ridge_alpha": 1.0},
        "mlp": {"type": "mlp", "mlp_lr": 0.001, "mlp_epochs": 30, "mlp_hidden": [64, 32]},
        "qcml_full": {"type": "qcml", "hilbert_dim": 16, "qcml_epochs": 50},
        "qcml_real_only": {"type": "qcml_real_only", "hilbert_dim": 16, "qcml_epochs": 50},
    }

    # Prepare data
    X = full_dataset.features.numpy()
    y = full_dataset.labels.numpy()
    ticker_indices = full_dataset.ticker_indices.numpy()
    input_dim = X.shape[1]

    # Run walk-forward for each model
    print("\n" + "=" * 40)
    print(f"Running Walk-Forward Validation")
    print(f"Window type: {args.window_type}")
    print(f"Min training weeks: {args.min_train_weeks}")
    print("=" * 40)

    results = {}

    for name, model_cfg in model_configs.items():
        print(f"\n--- {name} ---")

        model_type = model_cfg["type"]
        train_fn, predict_fn = create_model_and_fns(model_type, input_dim, model_cfg, device)

        result = validator.run(
            train_fn=train_fn,
            predict_fn=predict_fn,
            features=X,
            labels=y,
            date_indices=date_indices,
            ticker_indices=ticker_indices,
            idx_to_date=idx_to_date,
            idx_to_ticker=idx_to_ticker,
            model_name=name,
            window_type=args.window_type,
            rolling_window=args.rolling_window if args.window_type == "rolling" else None,
            verbose=True
        )

        results[name] = result

    # Comparison table
    print("\n" + "=" * 60)
    print("Walk-Forward Comparison")
    print("=" * 60)

    comparison_df = compare_walk_forward_results(results)
    print(comparison_df.to_string())

    # Statistical significance testing
    print("\n" + "=" * 60)
    print("Statistical Significance Testing")
    print("=" * 60)

    significance_results = {}

    for name, result in results.items():
        sig = compute_significance(
            result.returns,
            n_bootstrap=args.n_bootstrap,
            random_state=42
        )

        # Permutation test
        _, perm_p, _ = permutation_test(
            result.predictions,
            result.actuals,
            n_permutations=args.n_bootstrap,
            random_state=42
        )

        significance_results[name] = {
            **significance_to_dict(sig),
            "permutation_p_value": perm_p
        }

        sig_str = "YES" if sig.is_significant else "NO"
        print(f"\n{name}:")
        print(f"  Sharpe: {sig.sharpe_ratio:.3f} (95% CI: [{sig.sharpe_ci_lower:.3f}, {sig.sharpe_ci_upper:.3f}])")
        print(f"  P-value (Sharpe > 0): {sig.sharpe_p_value:.4f}")
        print(f"  P-value (Permutation): {perm_p:.4f}")
        print(f"  Significant at 5%: {sig_str}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compile all results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "window_type": args.window_type,
            "min_train_weeks": args.min_train_weeks,
            "rolling_window": args.rolling_window,
            "n_bootstrap": args.n_bootstrap,
            "top_k": portfolio_config.top_k,
            "strategy": portfolio_config.strategy
        },
        "models": {}
    }

    for name, result in results.items():
        all_results["models"][name] = {
            **walk_forward_result_to_dict(result),
            "significance": significance_results[name]
        }

    with open(output_dir / "walk_forward_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save comparison table
    comparison_df.to_csv(output_dir / "walk_forward_comparison.csv")

    print(f"\nResults saved to {output_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Best model by Sharpe
    sharpe_values = {name: r.metrics.sharpe_ratio for name, r in results.items()}
    best_model = max(sharpe_values, key=sharpe_values.get)
    print(f"Best model by Sharpe: {best_model} ({sharpe_values[best_model]:.3f})")

    # Count significant models
    n_significant = sum(1 for s in significance_results.values() if s["is_significant"])
    print(f"Models with significant alpha (p < 0.05): {n_significant}/{len(results)}")

    # Key finding
    best_sig = significance_results[best_model]
    if best_sig["is_significant"]:
        print(f"\nConclusion: {best_model} shows statistically significant alpha.")
    else:
        print(f"\nConclusion: No model shows statistically significant alpha at 5% level.")
        print("Consider: More data, better features, or different prediction targets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run walk-forward validation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/processed_data.pkl",
        help="Path to processed data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/walk_forward",
        help="Directory to save results"
    )
    parser.add_argument(
        "--window-type",
        type=str,
        choices=["expanding", "rolling"],
        default="expanding",
        help="Type of walk-forward window"
    )
    parser.add_argument(
        "--min-train-weeks",
        type=int,
        default=52,
        help="Minimum training window size (weeks)"
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=104,
        help="Rolling window size (weeks) if window_type=rolling"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap/permutation samples"
    )

    args = parser.parse_args()
    main(args)
