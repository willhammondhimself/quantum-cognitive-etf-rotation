#!/usr/bin/env python3
"""
Train baseline models.

Trains PCA+Ridge, Autoencoder, and MLP baselines on the processed data.
Saves trained models and evaluation metrics.
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

from qcml_rotation.models.baselines.pca_ridge import PCARidge, PCARidgeConfig
from qcml_rotation.models.baselines.autoencoder import AutoencoderPredictor, AutoencoderConfig
from qcml_rotation.models.baselines.mlp import MLPPredictor, MLPConfig
from qcml_rotation.utils.helpers import load_config, set_seed, get_device


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))

    # Correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    # Directional accuracy (did we predict the sign correctly?)
    sign_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mae),
        "correlation": float(corr) if not np.isnan(corr) else 0.0,
        "sign_accuracy": float(sign_accuracy),
        "r2": float(r2)
    }


def main(args):
    """Train all baseline models."""
    print("=" * 60)
    print("QCML ETF Rotation - Baseline Training")
    print("=" * 60)

    # Load config and data
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

    print(f"\nLoaded data from {data_path}")
    print(f"Train: {len(splits.train)}, Val: {len(splits.val)}, Test: {len(splits.test)}")

    # Extract numpy arrays
    X_train = splits.train[feature_cols].values
    y_train = splits.train["excess_return"].values
    X_val = splits.val[feature_cols].values
    y_val = splits.val["excess_return"].values
    X_test = splits.test[feature_cols].values
    y_test = splits.test["excess_return"].values

    results = {}
    models = {}

    # ===== 1. PCA + Ridge =====
    print("\n" + "-" * 40)
    print("Training PCA + Ridge...")
    print("-" * 40)

    pca_config = PCARidgeConfig(
        n_components=config["models"]["pca_ridge"]["n_components"],
        alpha=config["models"]["pca_ridge"]["alpha"]
    )

    pca_ridge = PCARidge(pca_config)
    pca_ridge.fit(X_train, y_train, X_val, y_val)

    # Evaluate
    train_pred = pca_ridge.predict(X_train)
    val_pred = pca_ridge.predict(X_val)
    test_pred = pca_ridge.predict(X_test)

    results["pca_ridge"] = {
        "train": compute_metrics(y_train, train_pred),
        "val": compute_metrics(y_val, val_pred),
        "test": compute_metrics(y_test, test_pred),
        "explained_variance": pca_ridge.get_explained_variance().tolist()
    }
    models["pca_ridge"] = pca_ridge

    print(f"  Val MSE: {results['pca_ridge']['val']['mse']:.6f}")
    print(f"  Val Correlation: {results['pca_ridge']['val']['correlation']:.4f}")
    print(f"  Test MSE: {results['pca_ridge']['test']['mse']:.6f}")

    # ===== 2. Autoencoder =====
    print("\n" + "-" * 40)
    print("Training Autoencoder + Linear...")
    print("-" * 40)

    ae_config = AutoencoderConfig(
        hidden_dim=config["models"]["autoencoder"]["hidden_dim"],
        bottleneck_dim=config["models"]["autoencoder"]["bottleneck_dim"],
        lr=config["models"]["autoencoder"]["lr"],
        epochs=config["models"]["autoencoder"]["epochs"],
        batch_size=config["models"]["autoencoder"]["batch_size"],
        early_stopping_patience=config["training"]["early_stopping_patience"]
    )

    autoencoder = AutoencoderPredictor(
        input_dim=len(feature_cols),
        config=ae_config,
        device=device
    )
    autoencoder.fit(X_train, y_train, X_val, y_val)

    # Evaluate
    train_pred = autoencoder.predict(X_train)
    val_pred = autoencoder.predict(X_val)
    test_pred = autoencoder.predict(X_test)

    results["autoencoder"] = {
        "train": compute_metrics(y_train, train_pred),
        "val": compute_metrics(y_val, val_pred),
        "test": compute_metrics(y_test, test_pred)
    }
    models["autoencoder"] = autoencoder

    print(f"  Val MSE: {results['autoencoder']['val']['mse']:.6f}")
    print(f"  Val Correlation: {results['autoencoder']['val']['correlation']:.4f}")
    print(f"  Test MSE: {results['autoencoder']['test']['mse']:.6f}")

    # ===== 3. MLP =====
    print("\n" + "-" * 40)
    print("Training MLP...")
    print("-" * 40)

    mlp_config = MLPConfig(
        hidden_dims=config["models"]["mlp"]["hidden_dims"],
        dropout=config["models"]["mlp"]["dropout"],
        lr=config["models"]["mlp"]["lr"],
        epochs=config["models"]["mlp"]["epochs"],
        batch_size=config["models"]["mlp"]["batch_size"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        lr_scheduler_patience=config["training"]["lr_scheduler_patience"],
        lr_scheduler_factor=config["training"]["lr_scheduler_factor"]
    )

    mlp = MLPPredictor(
        input_dim=len(feature_cols),
        config=mlp_config,
        device=device
    )
    mlp.fit(X_train, y_train, X_val, y_val)

    # Evaluate
    train_pred = mlp.predict(X_train)
    val_pred = mlp.predict(X_val)
    test_pred = mlp.predict(X_test)

    results["mlp"] = {
        "train": compute_metrics(y_train, train_pred),
        "val": compute_metrics(y_val, val_pred),
        "test": compute_metrics(y_test, test_pred)
    }
    models["mlp"] = mlp

    print(f"  Val MSE: {results['mlp']['val']['mse']:.6f}")
    print(f"  Val Correlation: {results['mlp']['val']['correlation']:.4f}")
    print(f"  Test MSE: {results['mlp']['test']['mse']:.6f}")

    # ===== Save Results =====
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    results["timestamp"] = datetime.now().isoformat()
    results["config"] = config

    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save models
    with open(output_dir / "baseline_models.pkl", "wb") as f:
        pickle.dump(models, f)

    # Print comparison
    print("\n" + "=" * 60)
    print("Baseline Comparison (Test Set)")
    print("=" * 60)
    print(f"{'Model':<20} {'MSE':<12} {'Corr':<12} {'Sign Acc':<12}")
    print("-" * 60)

    for name in ["pca_ridge", "autoencoder", "mlp"]:
        test_metrics = results[name]["test"]
        print(f"{name:<20} {test_metrics['mse']:<12.6f} "
              f"{test_metrics['correlation']:<12.4f} "
              f"{test_metrics['sign_accuracy']:<12.2%}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline models")
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
        default="outputs/baselines",
        help="Directory to save results"
    )

    args = parser.parse_args()
    main(args)
