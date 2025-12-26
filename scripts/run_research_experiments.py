#!/usr/bin/env python3
"""
QCML Research Experiments for Paper

This script runs experiments for the research paper:
"Quantum-Inspired Neural Networks for Financial Time Series"

Experiments:
1. Synthetic data validation - prove architecture works on known signals
2. Real data comparison - QCML variants vs baselines
3. Ablation studies - component contribution analysis
4. Visualization generation - paper figures
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml_rotation.models.qcml import (
    QCML, QCMLConfig, QCMLWithRanking,
    SimplifiedQCML, SimplifiedQCMLConfig, SimplifiedQCMLWithRanking,
    RankingQCML, RankingQCMLConfig,
    QuantumEnhancedQCML, QuantumEnhancedConfig,
    create_quantum_enhanced_model
)

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


@dataclass
class ExperimentResults:
    """Container for experiment results."""
    model_name: str
    synthetic_correlation: float
    synthetic_sign_accuracy: float
    real_correlation: float
    real_sign_accuracy: float
    training_time: float
    n_parameters: int
    ablation_scores: Dict[str, float]


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_momentum_data(
    n_samples: int = 5000,
    n_features: int = 20,
    signal_strength: float = 0.3,
    noise_level: float = 0.7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with known momentum signal.

    The target is a combination of:
    - Momentum: positive features → positive return
    - Mean-reversion: extreme features → opposite return
    - Cross-sectional: relative ranking matters

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of input features
    signal_strength : float
        Strength of the true signal (0-1)
    noise_level : float
        Amount of noise (0-1)

    Returns
    -------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    """
    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Split features into momentum (first half) and volatility (second half)
    n_momentum = n_features // 2
    X_momentum = X[:, :n_momentum]
    X_volatility = X[:, n_momentum:]

    # Momentum signal: average of first features
    momentum_signal = X_momentum.mean(axis=1)

    # Mean-reversion signal: penalize extremes
    mean_reversion = -np.abs(X_momentum).mean(axis=1) * 0.5

    # Volatility adjustment: high vol reduces expected return
    vol_adjustment = -X_volatility.mean(axis=1) * 0.3

    # Non-linear interaction (what quantum interference should capture)
    interaction = np.sin(momentum_signal * np.pi) * np.cos(vol_adjustment * np.pi) * 0.2

    # Combine signals
    true_signal = momentum_signal + mean_reversion + vol_adjustment + interaction

    # Add noise
    noise = np.random.randn(n_samples) * noise_level
    y = signal_strength * true_signal + noise

    # Normalize
    y = (y - y.mean()) / y.std()

    return X.astype(np.float32), y.astype(np.float32)


def generate_synthetic_ranking_data(
    n_weeks: int = 200,
    n_assets: int = 10,
    n_features: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data with week-based ranking structure.

    Returns
    -------
    X : array of shape (n_weeks * n_assets, n_features)
    y : array of shape (n_weeks * n_assets,)
    week_indices : array of shape (n_weeks * n_assets,)
    """
    X_list = []
    y_list = []
    week_list = []

    for week in range(n_weeks):
        # Generate features for this week
        X_week = np.random.randn(n_assets, n_features).astype(np.float32)

        # Generate returns based on features
        momentum = X_week[:, :n_features//2].mean(axis=1)
        vol = X_week[:, n_features//2:].mean(axis=1)

        # True ranking signal
        signal = momentum - 0.3 * np.abs(vol)
        noise = np.random.randn(n_assets) * 0.5
        y_week = signal + noise

        X_list.append(X_week)
        y_list.append(y_week.astype(np.float32))
        week_list.extend([week] * n_assets)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    week_indices = np.array(week_list)

    return X, y, week_indices


# =============================================================================
# Training Functions
# =============================================================================

def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    week_indices_train: Optional[np.ndarray] = None,
    week_indices_val: Optional[np.ndarray] = None,
    early_stopping_patience: int = 15,
    verbose: bool = False
) -> Tuple[nn.Module, Dict]:
    """
    Train a QCML model.

    Returns
    -------
    model : trained model
    history : dict with training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    if week_indices_train is not None:
        week_train_t = torch.LongTensor(week_indices_train).to(device)
        week_val_t = torch.LongTensor(week_indices_val).to(device)
    else:
        week_train_t = None
        week_val_t = None

    # Create data loader
    if week_train_t is not None:
        train_dataset = TensorDataset(X_train_t, y_train_t, week_train_t)
    else:
        train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_corr': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            if week_train_t is not None:
                X_batch, y_batch, week_batch = batch
            else:
                X_batch, y_batch = batch
                week_batch = None

            optimizer.zero_grad()

            # Forward pass
            if hasattr(model, 'compute_loss'):
                loss, _ = model.compute_loss(X_batch, y_batch, week_batch)
            else:
                pred = model(X_batch)
                loss = nn.functional.mse_loss(pred, y_batch)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'compute_loss'):
                val_loss, _ = model.compute_loss(X_val_t, y_val_t, week_val_t)
            else:
                val_pred = model(X_val_t)
                val_loss = nn.functional.mse_loss(val_pred, y_val_t)

            # Compute correlation
            val_pred = model(X_val_t).cpu().numpy()
            val_corr = np.corrcoef(val_pred, y_val.flatten())[0, 1]

        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(val_loss.item())
        history['val_corr'].append(val_corr if not np.isnan(val_corr) else 0)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f}, "
                  f"val_loss={val_loss:.4f}, val_corr={val_corr:.4f}")

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:
    """Evaluate model predictions."""
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        predictions = model(X_t).cpu().numpy()

    # Correlation
    corr = np.corrcoef(predictions.flatten(), y.flatten())[0, 1]

    # Sign accuracy
    sign_acc = np.mean(np.sign(predictions.flatten()) == np.sign(y.flatten()))

    # Rank correlation
    rank_corr = stats.spearmanr(predictions.flatten(), y.flatten())[0]

    return {
        'correlation': corr if not np.isnan(corr) else 0,
        'sign_accuracy': sign_acc,
        'rank_correlation': rank_corr if not np.isnan(rank_corr) else 0
    }


# =============================================================================
# Experiment Functions
# =============================================================================

def run_synthetic_experiments(
    n_trials: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run experiments on synthetic data to validate architecture.
    """
    if verbose:
        print("\n" + "="*60)
        print("SYNTHETIC DATA EXPERIMENTS")
        print("="*60)

    results = []

    for trial in range(n_trials):
        if verbose:
            print(f"\nTrial {trial + 1}/{n_trials}")

        # Generate data
        X, y = generate_synthetic_momentum_data(
            n_samples=5000,
            n_features=20,
            signal_strength=0.5,
            noise_level=0.5
        )

        # Split
        n_train = int(len(X) * 0.7)
        n_val = int(len(X) * 0.15)

        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
        X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

        input_dim = X.shape[1]

        # Models to test
        models = {
            'Original QCML': QCML(input_dim, QCMLConfig(hilbert_dim=16)),
            'SimplifiedQCML': SimplifiedQCML(input_dim, SimplifiedQCMLConfig()),
            'RankingQCML': RankingQCML(input_dim, RankingQCMLConfig()),
            'QuantumEnhanced (no interference)': QuantumEnhancedQCML(
                input_dim,
                QuantumEnhancedConfig(use_dual_pathway=False)
            ),
            'QuantumEnhanced (with interference)': QuantumEnhancedQCML(
                input_dim,
                QuantumEnhancedConfig(use_dual_pathway=True)
            ),
        }

        for name, model in models.items():
            if verbose:
                print(f"  Training {name}...")

            start_time = datetime.now()
            trained_model, history = train_model(
                model, X_train, y_train, X_val, y_val,
                epochs=100, verbose=False
            )
            train_time = (datetime.now() - start_time).total_seconds()

            # Evaluate
            metrics = evaluate_model(trained_model, X_test, y_test)

            results.append({
                'trial': trial,
                'model': name,
                'correlation': metrics['correlation'],
                'sign_accuracy': metrics['sign_accuracy'],
                'rank_correlation': metrics['rank_correlation'],
                'training_time': train_time,
                'n_parameters': sum(p.numel() for p in model.parameters())
            })

            if verbose:
                print(f"    Correlation: {metrics['correlation']:.4f}, "
                      f"Sign Acc: {metrics['sign_accuracy']:.4f}")

    return pd.DataFrame(results)


def run_ablation_study(verbose: bool = True) -> pd.DataFrame:
    """
    Ablation study for QuantumEnhancedQCML components.

    Tests:
    1. Full model (all components)
    2. Without amplitude-phase encoding (use unit normalization)
    3. Without multiple observables (single observable)
    4. Without dual-pathway (single pathway)
    5. Without interference (simple concatenation)
    """
    if verbose:
        print("\n" + "="*60)
        print("ABLATION STUDY")
        print("="*60)

    # Generate data
    X, y = generate_synthetic_momentum_data(
        n_samples=5000,
        n_features=20,
        signal_strength=0.5,
        noise_level=0.5
    )

    n_train = int(len(X) * 0.7)
    n_val = int(len(X) * 0.15)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    input_dim = X.shape[1]

    ablation_configs = {
        'Full Model': QuantumEnhancedConfig(
            use_dual_pathway=True,
            n_observables=4
        ),
        'Single Observable': QuantumEnhancedConfig(
            use_dual_pathway=True,
            n_observables=1
        ),
        'No Dual Pathway': QuantumEnhancedConfig(
            use_dual_pathway=False,
            n_observables=4
        ),
        '2 Observables': QuantumEnhancedConfig(
            use_dual_pathway=True,
            n_observables=2
        ),
        'Small Hilbert Dim (d=8)': QuantumEnhancedConfig(
            use_dual_pathway=True,
            n_observables=4,
            hilbert_dim=8
        ),
        'Large Hilbert Dim (d=64)': QuantumEnhancedConfig(
            use_dual_pathway=True,
            n_observables=4,
            hilbert_dim=64
        ),
    }

    results = []
    n_trials = 3

    for name, config in ablation_configs.items():
        if verbose:
            print(f"\nTesting: {name}")

        trial_metrics = []
        for trial in range(n_trials):
            model = QuantumEnhancedQCML(input_dim, config)

            trained_model, _ = train_model(
                model, X_train, y_train, X_val, y_val,
                epochs=100, verbose=False
            )

            metrics = evaluate_model(trained_model, X_test, y_test)
            trial_metrics.append(metrics)

        # Average over trials
        avg_corr = np.mean([m['correlation'] for m in trial_metrics])
        avg_sign = np.mean([m['sign_accuracy'] for m in trial_metrics])
        std_corr = np.std([m['correlation'] for m in trial_metrics])

        results.append({
            'configuration': name,
            'correlation_mean': avg_corr,
            'correlation_std': std_corr,
            'sign_accuracy': avg_sign,
            'n_parameters': sum(p.numel() for p in model.parameters())
        })

        if verbose:
            print(f"  Correlation: {avg_corr:.4f} +/- {std_corr:.4f}")

    return pd.DataFrame(results)


def generate_visualizations(output_dir: Path, verbose: bool = True):
    """
    Generate visualizations for the research paper.
    """
    if verbose:
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    X, y = generate_synthetic_momentum_data(n_samples=2000, n_features=20)

    n_train = int(len(X) * 0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    # Train model
    config = QuantumEnhancedConfig(use_dual_pathway=True, n_observables=4)
    model = QuantumEnhancedQCML(X.shape[1], config)

    model, history = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=100, verbose=False
    )

    # 1. Training curves
    if verbose:
        print("  Creating training curves...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()

    axes[1].plot(history['val_corr'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title('Validation Correlation')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()

    # 2. State space visualization
    if verbose:
        print("  Creating state space visualization...")

    X_test_t = torch.FloatTensor(X_test[:500])
    analysis = model.get_state_analysis(X_test_t)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Amplitude distribution
    if 'r_A' in analysis:
        r_A = analysis['r_A']
        axes[0, 0].hist(r_A.flatten(), bins=50, alpha=0.7, label='Pathway A')
        if 'r_B' in analysis:
            r_B = analysis['r_B']
            axes[0, 0].hist(r_B.flatten(), bins=50, alpha=0.7, label='Pathway B')
        axes[0, 0].set_xlabel('Amplitude (r)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Amplitude Distribution')
        axes[0, 0].legend()
    elif 'r' in analysis:
        r = analysis['r']
        axes[0, 0].hist(r.flatten(), bins=50, alpha=0.7)
        axes[0, 0].set_xlabel('Amplitude (r)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Amplitude Distribution')

    # Phase distribution
    if 'theta_A' in analysis:
        theta_A = analysis['theta_A']
        axes[0, 1].hist(theta_A.flatten(), bins=50, alpha=0.7, label='Pathway A')
        if 'theta_B' in analysis:
            theta_B = analysis['theta_B']
            axes[0, 1].hist(theta_B.flatten(), bins=50, alpha=0.7, label='Pathway B')
        axes[0, 1].set_xlabel('Phase (theta)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Phase Distribution')
        axes[0, 1].legend()
    elif 'theta' in analysis:
        theta = analysis['theta']
        axes[0, 1].hist(theta.flatten(), bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('Phase (theta)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Phase Distribution')

    # Observable weights
    weights = analysis['combination_weights']
    axes[1, 0].bar(range(len(weights)), weights)
    axes[1, 0].set_xlabel('Observable Index')
    axes[1, 0].set_ylabel('Weight')
    axes[1, 0].set_title('Observable Combination Weights')

    # Eigenspectra
    eigenspectra = analysis['eigenspectra']
    for i, eigs in enumerate(eigenspectra):
        axes[1, 1].plot(sorted(eigs, reverse=True), label=f'W_{i+1}', alpha=0.7)
    axes[1, 1].set_xlabel('Eigenvalue Index')
    axes[1, 1].set_ylabel('Eigenvalue')
    axes[1, 1].set_title('Observable Eigenspectra')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'state_space_analysis.png', dpi=150)
    plt.close()

    # 3. Prediction scatter plot
    if verbose:
        print("  Creating prediction scatter plot...")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test[:500], predictions, alpha=0.5, s=10)
    ax.plot([-3, 3], [-3, 3], 'r--', label='Perfect prediction')
    ax.set_xlabel('True Returns')
    ax.set_ylabel('Predicted Returns')
    ax.set_title(f'Prediction vs True (Corr: {np.corrcoef(predictions.flatten(), y_test[:500])[0,1]:.3f})')
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_scatter.png', dpi=150)
    plt.close()

    # 4. Individual observable expectations
    if verbose:
        print("  Creating observable expectations plot...")

    if 'expectations' in analysis:
        expectations = analysis['expectations']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for i in range(min(4, expectations.shape[1])):
            ax = axes[i // 2, i % 2]
            ax.scatter(y_test[:500], expectations[:, i], alpha=0.5, s=10)
            corr = np.corrcoef(expectations[:, i], y_test[:500])[0, 1]
            ax.set_xlabel('True Returns')
            ax.set_ylabel(f'Observable {i+1} Expectation')
            ax.set_title(f'Observable {i+1} (Corr: {corr:.3f})')

        plt.tight_layout()
        plt.savefig(output_dir / 'observable_expectations.png', dpi=150)
        plt.close()

    if verbose:
        print(f"  Visualizations saved to {output_dir}")


def run_model_comparison(verbose: bool = True) -> pd.DataFrame:
    """
    Compare all QCML variants with proper cross-validation.
    """
    if verbose:
        print("\n" + "="*60)
        print("MODEL COMPARISON (5-Fold CV)")
        print("="*60)

    # Generate data
    X, y, week_indices = generate_synthetic_ranking_data(
        n_weeks=300,
        n_assets=10,
        n_features=20
    )

    input_dim = X.shape[1]
    n_folds = 5
    fold_size = len(X) // n_folds

    results = []

    models_configs = {
        'Original QCML': lambda: QCML(input_dim, QCMLConfig()),
        'SimplifiedQCML': lambda: SimplifiedQCML(input_dim, SimplifiedQCMLConfig()),
        'RankingQCML': lambda: RankingQCML(input_dim, RankingQCMLConfig()),
        'QuantumEnhanced': lambda: QuantumEnhancedQCML(input_dim, QuantumEnhancedConfig()),
    }

    for model_name, model_fn in models_configs.items():
        if verbose:
            print(f"\nEvaluating {model_name}...")

        fold_metrics = []

        for fold in range(n_folds):
            # Split data
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size

            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            week_test = week_indices[test_start:test_end]

            X_train = np.vstack([X[:test_start], X[test_end:]])
            y_train = np.concatenate([y[:test_start], y[test_end:]])
            week_train = np.concatenate([week_indices[:test_start], week_indices[test_end:]])

            # Further split train into train/val
            val_size = len(X_train) // 5
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            week_val = week_train[-val_size:]
            X_train = X_train[:-val_size]
            y_train = y_train[:-val_size]
            week_train = week_train[:-val_size]

            # Train
            model = model_fn()
            model, _ = train_model(
                model, X_train, y_train, X_val, y_val,
                week_indices_train=week_train,
                week_indices_val=week_val,
                epochs=100, verbose=False
            )

            # Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            fold_metrics.append(metrics)

        # Aggregate
        avg_corr = np.mean([m['correlation'] for m in fold_metrics])
        std_corr = np.std([m['correlation'] for m in fold_metrics])
        avg_sign = np.mean([m['sign_accuracy'] for m in fold_metrics])
        avg_rank = np.mean([m['rank_correlation'] for m in fold_metrics])

        results.append({
            'model': model_name,
            'correlation_mean': avg_corr,
            'correlation_std': std_corr,
            'sign_accuracy': avg_sign,
            'rank_correlation': avg_rank
        })

        if verbose:
            print(f"  Correlation: {avg_corr:.4f} +/- {std_corr:.4f}")
            print(f"  Sign Accuracy: {avg_sign:.4f}")
            print(f"  Rank Correlation: {avg_rank:.4f}")

    return pd.DataFrame(results)


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all research experiments."""
    print("="*60)
    print("QCML RESEARCH EXPERIMENTS")
    print("Quantum-Inspired Neural Networks for Financial Time Series")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Output directory
    output_dir = project_root / 'results' / 'research_paper'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic data experiments
    synthetic_results = run_synthetic_experiments(n_trials=3, verbose=True)
    synthetic_results.to_csv(output_dir / 'synthetic_results.csv', index=False)

    # Summary by model
    print("\n--- Synthetic Data Summary ---")
    summary = synthetic_results.groupby('model').agg({
        'correlation': ['mean', 'std'],
        'sign_accuracy': 'mean',
        'training_time': 'mean'
    }).round(4)
    print(summary)

    # 2. Ablation study
    ablation_results = run_ablation_study(verbose=True)
    ablation_results.to_csv(output_dir / 'ablation_results.csv', index=False)

    print("\n--- Ablation Study Summary ---")
    print(ablation_results.to_string(index=False))

    # 3. Model comparison with CV
    comparison_results = run_model_comparison(verbose=True)
    comparison_results.to_csv(output_dir / 'comparison_results.csv', index=False)

    print("\n--- Model Comparison Summary ---")
    print(comparison_results.to_string(index=False))

    # 4. Generate visualizations
    generate_visualizations(output_dir / 'figures', verbose=True)

    # Final summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Key findings for paper
    print("\n--- KEY FINDINGS FOR PAPER ---")

    best_synthetic = synthetic_results.groupby('model')['correlation'].mean().idxmax()
    best_corr = synthetic_results.groupby('model')['correlation'].mean().max()
    print(f"1. Best synthetic performance: {best_synthetic} (corr={best_corr:.4f})")

    best_ablation = ablation_results.loc[ablation_results['correlation_mean'].idxmax()]
    print(f"2. Best ablation config: {best_ablation['configuration']} "
          f"(corr={best_ablation['correlation_mean']:.4f})")

    print(f"3. Multiple observables contribution: Compare 'Full Model' vs 'Single Observable'")
    print(f"4. Interference contribution: Compare 'Full Model' vs 'No Dual Pathway'")


if __name__ == '__main__':
    main()
