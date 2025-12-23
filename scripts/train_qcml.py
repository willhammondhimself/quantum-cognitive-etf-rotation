#!/usr/bin/env python3
"""
Train QCML model.

Trains the Quantum-Cognitive Market Learning model with Hilbert space
embeddings and interference-style ranking loss.

Also supports ablation experiments:
- Remove ranking loss (MSE only)
- Use real-only embeddings (no imaginary part)
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

from qcml_rotation.data.dataset import ETFDataset
from qcml_rotation.models.qcml import QCMLConfig, create_qcml_model
from qcml_rotation.training.qcml_trainer import QCMLTrainer, QCMLTrainerConfig
from qcml_rotation.utils.helpers import load_config, set_seed, get_device


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    sign_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

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


def train_single_model(
    train_dataset: ETFDataset,
    val_dataset: ETFDataset,
    test_dataset: ETFDataset,
    model_config: QCMLConfig,
    trainer_config: QCMLTrainerConfig,
    device,
    name: str = "qcml"
) -> dict:
    """Train a single QCML model variant and evaluate."""
    print(f"\nTraining {name}...")
    print(f"  Hilbert dim: {model_config.hilbert_dim}")
    print(f"  Complex embeddings: {model_config.use_complex}")
    print(f"  Ranking weight: {model_config.ranking_weight}")

    # Create model
    model = create_qcml_model(
        input_dim=train_dataset.n_features,
        config=model_config,
        device=device
    )

    # Train
    trainer = QCMLTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_config,
        device=device
    )

    history = trainer.train(verbose=True)

    # Evaluate
    y_train = train_dataset.labels.numpy()
    y_val = val_dataset.labels.numpy()
    y_test = test_dataset.labels.numpy()

    train_pred = trainer.predict(train_dataset)
    val_pred = trainer.predict(val_dataset)
    test_pred = trainer.predict(test_dataset)

    results = {
        "name": name,
        "config": {
            "hilbert_dim": model_config.hilbert_dim,
            "use_complex": model_config.use_complex,
            "ranking_weight": model_config.ranking_weight
        },
        "train": compute_metrics(y_train, train_pred),
        "val": compute_metrics(y_val, val_pred),
        "test": compute_metrics(y_test, test_pred),
        "history": {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"]
        },
        "n_epochs": len(history["train_loss"])
    }

    print(f"  Val MSE: {results['val']['mse']:.6f}")
    print(f"  Val Correlation: {results['val']['correlation']:.4f}")
    print(f"  Test MSE: {results['test']['mse']:.6f}")
    print(f"  Test Correlation: {results['test']['correlation']:.4f}")

    return results, model


def main(args):
    """Train QCML model and ablations."""
    print("=" * 60)
    print("QCML ETF Rotation - Quantum-Cognitive Model Training")
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

    # Create datasets
    train_dataset = ETFDataset(splits.train, feature_cols)
    val_dataset = ETFDataset(splits.val, feature_cols)
    test_dataset = ETFDataset(splits.test, feature_cols)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Features: {train_dataset.n_features}")

    # Model configurations
    qcml_config = config["models"]["qcml"]

    base_model_config = QCMLConfig(
        hilbert_dim=qcml_config["hilbert_dim"],
        encoder_hidden=qcml_config["encoder_hidden"],
        use_complex=True,
        lr=qcml_config["lr"],
        epochs=qcml_config["epochs"],
        ranking_weight=qcml_config["ranking_weight"],
        ranking_margin=qcml_config["ranking_margin"],
        pairs_per_week=qcml_config["pairs_per_week"],
        early_stopping_patience=config["training"]["early_stopping_patience"]
    )

    trainer_config = QCMLTrainerConfig(
        epochs=qcml_config["epochs"],
        lr=qcml_config["lr"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        lr_scheduler_patience=config["training"]["lr_scheduler_patience"],
        lr_scheduler_factor=config["training"]["lr_scheduler_factor"]
    )

    all_results = {}
    models = {}

    # ===== 1. Full QCML Model =====
    print("\n" + "=" * 40)
    print("1. Full QCML Model (Complex + Ranking)")
    print("=" * 40)

    results, model = train_single_model(
        train_dataset, val_dataset, test_dataset,
        base_model_config, trainer_config, device,
        name="qcml_full"
    )
    all_results["qcml_full"] = results
    models["qcml_full"] = model

    if args.run_ablations:
        # ===== 2. Ablation: No Ranking Loss =====
        print("\n" + "=" * 40)
        print("2. Ablation: No Ranking Loss (MSE Only)")
        print("=" * 40)

        no_ranking_config = QCMLConfig(
            hilbert_dim=qcml_config["hilbert_dim"],
            encoder_hidden=qcml_config["encoder_hidden"],
            use_complex=True,
            lr=qcml_config["lr"],
            epochs=qcml_config["epochs"],
            ranking_weight=0.0,  # No ranking loss
            ranking_margin=qcml_config["ranking_margin"],
            pairs_per_week=qcml_config["pairs_per_week"],
            early_stopping_patience=config["training"]["early_stopping_patience"]
        )

        results, model = train_single_model(
            train_dataset, val_dataset, test_dataset,
            no_ranking_config, trainer_config, device,
            name="qcml_no_ranking"
        )
        all_results["qcml_no_ranking"] = results
        models["qcml_no_ranking"] = model

        # ===== 3. Ablation: Real-Only Embeddings =====
        print("\n" + "=" * 40)
        print("3. Ablation: Real-Only Embeddings")
        print("=" * 40)

        real_only_config = QCMLConfig(
            hilbert_dim=qcml_config["hilbert_dim"],
            encoder_hidden=qcml_config["encoder_hidden"],
            use_complex=False,  # Real only
            lr=qcml_config["lr"],
            epochs=qcml_config["epochs"],
            ranking_weight=qcml_config["ranking_weight"],
            ranking_margin=qcml_config["ranking_margin"],
            pairs_per_week=qcml_config["pairs_per_week"],
            early_stopping_patience=config["training"]["early_stopping_patience"]
        )

        results, model = train_single_model(
            train_dataset, val_dataset, test_dataset,
            real_only_config, trainer_config, device,
            name="qcml_real_only"
        )
        all_results["qcml_real_only"] = results
        models["qcml_real_only"] = model

    # ===== Save Results =====
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    all_results["timestamp"] = datetime.now().isoformat()
    all_results["config"] = config

    with open(output_dir / "qcml_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save models
    with open(output_dir / "qcml_models.pkl", "wb") as f:
        pickle.dump(models, f)

    # Print comparison
    print("\n" + "=" * 60)
    print("QCML Results Summary (Test Set)")
    print("=" * 60)
    print(f"{'Model':<25} {'MSE':<12} {'Corr':<12} {'Sign Acc':<12}")
    print("-" * 60)

    for name, result in all_results.items():
        if name in ["timestamp", "config"]:
            continue
        test_metrics = result["test"]
        print(f"{name:<25} {test_metrics['mse']:<12.6f} "
              f"{test_metrics['correlation']:<12.4f} "
              f"{test_metrics['sign_accuracy']:<12.2%}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QCML model")
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
        default="outputs/qcml",
        help="Directory to save results"
    )
    parser.add_argument(
        "--run-ablations",
        action="store_true",
        help="Run ablation experiments (no ranking, real-only)"
    )

    args = parser.parse_args()
    main(args)
