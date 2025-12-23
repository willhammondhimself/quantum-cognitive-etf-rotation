#!/usr/bin/env python3
"""
Run backtest and model comparison.

Loads trained models and runs backtests on the test period,
comparing all models side-by-side.
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

from qcml_rotation.data.dataset import ETFDataset
from qcml_rotation.data.loader import load_cached_data, get_trading_dates
from qcml_rotation.backtest.engine import BacktestEngine, compare_results, results_to_dict
from qcml_rotation.backtest.portfolio import PortfolioConfig
from qcml_rotation.utils.helpers import load_config, set_seed, get_device


def load_models(baseline_path: str, qcml_path: str) -> dict:
    """Load trained models from disk."""
    models = {}

    # Load baselines
    if Path(baseline_path).exists():
        with open(baseline_path, "rb") as f:
            baseline_models = pickle.load(f)
        models.update(baseline_models)
        print(f"Loaded baselines: {list(baseline_models.keys())}")
    else:
        print(f"Warning: Baseline models not found at {baseline_path}")

    # Load QCML models
    if Path(qcml_path).exists():
        with open(qcml_path, "rb") as f:
            qcml_models = pickle.load(f)
        models.update(qcml_models)
        print(f"Loaded QCML models: {list(qcml_models.keys())}")
    else:
        print(f"Warning: QCML models not found at {qcml_path}")

    return models


def create_predict_fn(model, model_type: str, device=None):
    """Create a prediction function for a model."""
    import torch

    if model_type == "pca_ridge":
        return lambda x: model.predict(x)

    elif model_type == "autoencoder":
        return lambda x: model.predict(x)

    elif model_type == "mlp":
        return lambda x: model.predict(x)

    elif model_type.startswith("qcml"):
        def predict_qcml(x):
            model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32)
                if device is not None:
                    x_tensor = x_tensor.to(device)
                preds = model(x_tensor)
                return preds.cpu().numpy()
        return predict_qcml

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main(args):
    """Run backtest comparison."""
    print("=" * 60)
    print("QCML ETF Rotation - Backtest & Model Comparison")
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

    # Create test dataset
    test_dataset = ETFDataset(splits.test, feature_cols)
    print(f"Test samples: {len(test_dataset)}")

    # Get test dates
    test_dates_set = set(splits.test.index.get_level_values("date"))
    test_dates = [d for d in rebalance_dates if d in test_dates_set]
    print(f"Test weeks: {len(test_dates)}")

    # Load models
    models = load_models(args.baseline_models, args.qcml_models)

    if not models:
        print("Error: No models found. Train models first:")
        print("  python scripts/train_baselines.py")
        print("  python scripts/train_qcml.py")
        sys.exit(1)

    # Setup backtest engine
    portfolio_config = PortfolioConfig(
        top_k=config["backtest"]["top_k"],
        bottom_k=config["backtest"]["top_k"],
        strategy=config["backtest"]["strategy"],
        transaction_cost_bps=config["backtest"]["transaction_cost_bps"]
    )

    tickers = config["tickers"]["etfs"]
    benchmark = config["tickers"]["benchmark"]

    engine = BacktestEngine(
        prices=prices,
        test_dates=test_dates,
        tickers=tickers,
        portfolio_config=portfolio_config,
        benchmark=benchmark
    )

    # Prepare data for backtest
    X_test = test_dataset.features.numpy()
    y_test = test_dataset.labels.numpy()
    date_indices = test_dataset.date_indices.numpy()
    ticker_indices = test_dataset.ticker_indices.numpy()

    # Create ticker index mapping
    idx_to_ticker = {i: t for i, t in enumerate(test_dataset.tickers)}

    # Run backtests
    print("\n" + "=" * 40)
    print("Running Backtests")
    print("=" * 40)

    results = {}

    for name, model in models.items():
        # Determine model type
        if name == "pca_ridge":
            model_type = "pca_ridge"
        elif name == "autoencoder":
            model_type = "autoencoder"
        elif name == "mlp":
            model_type = "mlp"
        else:
            model_type = "qcml"

        predict_fn = create_predict_fn(model, model_type, device)

        result = engine.run(
            predict_fn=predict_fn,
            model_name=name,
            features=X_test,
            labels=y_test,
            date_indices=date_indices,
            ticker_indices=ticker_indices,
            idx_to_ticker=idx_to_ticker
        )

        results[name] = result
        print(f"{name}: Sharpe={result.metrics.sharpe_ratio:.3f}, "
              f"Return={result.metrics.total_return:.2%}, "
              f"MaxDD={result.metrics.max_drawdown:.2%}")

    # Comparison table
    print("\n" + "=" * 60)
    print("Model Comparison (Test Period)")
    print("=" * 60)

    comparison_df = compare_results(results)
    print(comparison_df.to_string())

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_dict = results_to_dict(results)
    results_dict["timestamp"] = datetime.now().isoformat()
    results_dict["config"] = {
        "top_k": portfolio_config.top_k,
        "strategy": portfolio_config.strategy,
        "transaction_cost_bps": portfolio_config.transaction_cost_bps
    }

    with open(output_dir / "backtest_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    # Save comparison table
    comparison_df.to_csv(output_dir / "model_comparison.csv")

    print(f"\nResults saved to {output_dir}")

    # Print best model
    sharpe_values = {name: r.metrics.sharpe_ratio for name, r in results.items()}
    best_model = max(sharpe_values, key=sharpe_values.get)
    print(f"\nBest model by Sharpe: {best_model} ({sharpe_values[best_model]:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtest comparison")
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
        "--baseline-models",
        type=str,
        default="outputs/baselines/baseline_models.pkl",
        help="Path to trained baseline models"
    )
    parser.add_argument(
        "--qcml-models",
        type=str,
        default="outputs/qcml/qcml_models.pkl",
        help="Path to trained QCML models"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/backtest",
        help="Directory to save results"
    )

    args = parser.parse_args()
    main(args)
