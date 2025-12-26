#!/usr/bin/env python3
"""
Benchmark Simplified QCML vs Original QCML and baselines.

This script compares:
1. Original QCML (with unit normalization - expected to fail)
2. Simplified QCML (without normalization - should be better)
3. Momentum baseline

Uses walk-forward validation for realistic OOS performance.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Fix OMP issue
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Data and features
from qcml_rotation.data.loader import download_etf_data, download_vix_data, get_trading_dates
from qcml_rotation.data.features import (
    build_features, compute_labels, merge_features_labels,
    FeatureConfig, get_feature_names
)
from qcml_rotation.models.qcml import (
    QCMLConfig, QCMLWithRanking,
    SimplifiedQCMLConfig, SimplifiedQCMLWithRanking,
    RankingQCMLConfig, RankingQCML
)
from qcml_rotation.utils.helpers import load_config

print("="*70)
print("SIMPLIFIED QCML BENCHMARK")
print("Comparing Original QCML vs Simplified QCML")
print("="*70)


def prepare_data():
    """Load and prepare ETF data with features and labels."""
    print("\n[1] Loading ETF data...")

    config = load_config()

    prices, volume = download_etf_data(
        tickers=config["tickers"]["etfs"],
        benchmark=config["tickers"]["benchmark"],
        start_date=config["data"]["start_date"],
        cache_dir=config["data"]["cache_dir"]
    )

    # Download VIX
    vix_data = download_vix_data(
        start_date=config["data"]["start_date"],
        cache_dir=config["data"]["cache_dir"]
    )

    print(f"    Loaded {len(prices)} days, {len(prices.columns)} tickers")

    # Get weekly rebalance dates
    weekly_dates = get_trading_dates(prices, freq="W-FRI")
    print(f"    Weekly rebalance dates: {len(weekly_dates)}")

    # Build features
    print("\n[2] Computing features...")
    feature_config = FeatureConfig(
        return_windows=[1, 5, 20],
        vol_window=20,
        benchmark=config["tickers"]["benchmark"],
        include_technical=True,
        include_cross_sectional=True,
        include_regime=True,
        momentum_windows=[20, 60]
    )

    features = build_features(prices, weekly_dates, feature_config, vix_data=vix_data)

    # Compute labels
    print("\n[3] Computing labels...")
    labels = compute_labels(
        prices, weekly_dates,
        benchmark=config["tickers"]["benchmark"],
        forward_days=5
    )

    # Merge
    data = merge_features_labels(features, labels)

    # Get feature columns
    feature_cols = get_feature_names(feature_config, include_vix=True)

    # Filter to valid feature columns that exist in data
    feature_cols = [c for c in feature_cols if c in data.columns]

    print(f"    Final dataset: {len(data)} samples, {len(feature_cols)} features")

    return data, feature_cols, config


def train_qcml_model(model, X_train, y_train, epochs=100, lr=0.001, weight_decay=1e-4):
    """Train a QCML model (original or simplified)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = nn.functional.mse_loss(predictions, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 15:
            break

    return model


def evaluate_predictions(y_true, y_pred):
    """Compute prediction metrics."""
    # Handle NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 10:
        return {'correlation': 0, 'sign_accuracy': 0.5, 'mse': 999}

    correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    sign_accuracy = ((y_true > 0) == (y_pred > 0)).mean()
    mse = ((y_true - y_pred) ** 2).mean()

    return {
        'correlation': correlation if not np.isnan(correlation) else 0,
        'sign_accuracy': sign_accuracy,
        'mse': mse
    }


def walk_forward_validation(data, feature_cols, n_folds=5):
    """
    Perform walk-forward validation.

    Uses expanding window: train on all historical data, test on next period.
    """
    print("\n[4] Running Walk-Forward Validation...")

    # Get unique dates
    dates = data.index.get_level_values(0).unique().sort_values()
    n_dates = len(dates)

    # Split into folds
    fold_size = n_dates // (n_folds + 1)
    initial_train_size = fold_size

    results = {
        'original_qcml': {'preds': [], 'labels': [], 'corrs': []},
        'simplified_qcml': {'preds': [], 'labels': [], 'corrs': []},
        'ranking_qcml': {'preds': [], 'labels': [], 'corrs': []},
        'momentum': {'preds': [], 'labels': [], 'corrs': []}
    }

    for fold in range(n_folds):
        # Define train/test periods
        train_end_idx = initial_train_size + fold * fold_size
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + fold_size, n_dates)

        if test_end_idx <= test_start_idx:
            continue

        train_dates = dates[:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]

        print(f"    Fold {fold+1}/{n_folds}: Train {train_dates[0].date()} to {train_dates[-1].date()}, "
              f"Test {test_dates[0].date()} to {test_dates[-1].date()}")

        # Get train/test data
        train_mask = data.index.get_level_values(0).isin(train_dates)
        test_mask = data.index.get_level_values(0).isin(test_dates)

        train_data = data[train_mask]
        test_data = data[test_mask]

        X_train = train_data[feature_cols].values.astype(np.float32)
        y_train = train_data['excess_return'].values.astype(np.float32)

        X_test = test_data[feature_cols].values.astype(np.float32)
        y_test = test_data['excess_return'].values.astype(np.float32)

        # Handle NaN in features
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)

        # Skip if not enough data
        if len(X_train) < 100 or len(X_test) < 20:
            print(f"        Skipping fold {fold+1} - insufficient data")
            continue

        # Standardize features
        train_mean = X_train.mean(axis=0)
        train_std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std

        # Train Original QCML
        config1 = QCMLConfig(hilbert_dim=16, encoder_hidden=32, epochs=100)
        model1 = QCMLWithRanking(len(feature_cols), config1)
        model1 = train_qcml_model(model1, X_train, y_train)

        model1.eval()
        with torch.no_grad():
            pred1 = model1(torch.tensor(X_test, dtype=torch.float32)).numpy()

        # Train Simplified QCML
        config2 = SimplifiedQCMLConfig(hidden_dim=64, embed_dim=32, epochs=100)
        model2 = SimplifiedQCMLWithRanking(len(feature_cols), config2)
        model2 = train_qcml_model(model2, X_train, y_train, weight_decay=config2.weight_decay)

        model2.eval()
        with torch.no_grad():
            pred2 = model2(torch.tensor(X_test, dtype=torch.float32)).numpy()

        # Train Ranking QCML (with RankNet loss)
        config3 = RankingQCMLConfig(hidden_dim=64, embed_dim=32, epochs=100, dropout=0.2)
        model3 = RankingQCML(len(feature_cols), config3)
        # Train with ranking loss
        optimizer = torch.optim.Adam(model3.parameters(), lr=config3.lr, weight_decay=config3.weight_decay)
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)

        model3.train()
        for epoch in range(100):
            optimizer.zero_grad()
            loss, _ = model3.compute_loss(X_train_t, y_train_t, week_indices=None)
            loss.backward()
            optimizer.step()

        model3.eval()
        with torch.no_grad():
            pred3 = model3(torch.tensor(X_test, dtype=torch.float32)).numpy()

        # Simple momentum baseline - use mom_20d feature
        momentum_col_idx = None
        for i, col in enumerate(feature_cols):
            if 'mom_20' in col.lower() or 'momentum_20' in col.lower():
                momentum_col_idx = i
                break
        if momentum_col_idx is None:
            # Try ret_20d
            for i, col in enumerate(feature_cols):
                if 'ret_20' in col.lower():
                    momentum_col_idx = i
                    break
        if momentum_col_idx is None:
            momentum_col_idx = 0

        pred_mom = X_test[:, momentum_col_idx]

        # Compute fold-level correlations
        corr1 = evaluate_predictions(y_test, pred1)['correlation']
        corr2 = evaluate_predictions(y_test, pred2)['correlation']
        corr3 = evaluate_predictions(y_test, pred3)['correlation']
        corr_mom = evaluate_predictions(y_test, pred_mom)['correlation']

        print(f"        Original QCML corr: {corr1:.4f}")
        print(f"        Simplified QCML corr: {corr2:.4f}")
        print(f"        Ranking QCML corr: {corr3:.4f}")
        print(f"        Momentum corr: {corr_mom:.4f}")

        # Store results
        results['original_qcml']['preds'].extend(pred1)
        results['original_qcml']['labels'].extend(y_test)
        results['original_qcml']['corrs'].append(corr1)

        results['simplified_qcml']['preds'].extend(pred2)
        results['simplified_qcml']['labels'].extend(y_test)
        results['simplified_qcml']['corrs'].append(corr2)

        results['ranking_qcml']['preds'].extend(pred3)
        results['ranking_qcml']['labels'].extend(y_test)
        results['ranking_qcml']['corrs'].append(corr3)

        results['momentum']['preds'].extend(pred_mom)
        results['momentum']['labels'].extend(y_test)
        results['momentum']['corrs'].append(corr_mom)

    print(f"    Completed {n_folds} folds")

    return results


def analyze_results(results):
    """Analyze and display benchmark results."""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)

    summary = []

    for model_name, data in results.items():
        preds = np.array(data['preds'])
        labels = np.array(data['labels'])
        fold_corrs = data['corrs']

        if len(preds) == 0:
            continue

        metrics = evaluate_predictions(labels, preds)

        # Compute strategy returns (rank-based allocation)
        strategy_returns = []
        n_per_week = 11  # ~11 ETFs per week
        for i in range(0, len(preds) - n_per_week, n_per_week):
            week_preds = preds[i:i+n_per_week]
            week_labels = labels[i:i+n_per_week]

            if np.isnan(week_preds).any() or np.isnan(week_labels).any():
                continue

            # Top 5 long
            top_idx = np.argsort(week_preds)[-5:]
            week_ret = week_labels[top_idx].mean()
            strategy_returns.append(week_ret)

        strategy_returns = np.array(strategy_returns)

        if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(52)
            total_return = (1 + strategy_returns).prod() - 1
        else:
            sharpe = 0
            total_return = 0

        avg_fold_corr = np.mean(fold_corrs) if fold_corrs else 0

        summary.append({
            'Model': model_name,
            'Avg Fold Corr': avg_fold_corr,
            'Overall Corr': metrics['correlation'],
            'Sign Acc': metrics['sign_accuracy'],
            'Sharpe': sharpe,
            'Total Return': total_return
        })

        print(f"\n{model_name.upper()}:")
        print(f"  Average Fold Correlation: {avg_fold_corr:.4f}")
        print(f"  Overall Correlation:      {metrics['correlation']:.4f}")
        print(f"  Sign Accuracy:            {metrics['sign_accuracy']:.2%}")
        print(f"  Strategy Sharpe:          {sharpe:.2f}")
        print(f"  Total Return:             {total_return:+.1%}")

    # Summary table
    print("\n" + "-"*70)
    print("SUMMARY TABLE")
    print("-"*70)
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    orig_corr = [s['Overall Corr'] for s in summary if s['Model'] == 'original_qcml'][0]
    simp_corr = [s['Overall Corr'] for s in summary if s['Model'] == 'simplified_qcml'][0]
    rank_corr = [s['Overall Corr'] for s in summary if s['Model'] == 'ranking_qcml'][0]

    orig_sharpe = [s['Sharpe'] for s in summary if s['Model'] == 'original_qcml'][0]
    simp_sharpe = [s['Sharpe'] for s in summary if s['Model'] == 'simplified_qcml'][0]
    rank_sharpe = [s['Sharpe'] for s in summary if s['Model'] == 'ranking_qcml'][0]

    # Compare all models
    best_model = 'original_qcml'
    best_corr = orig_corr
    if simp_corr > best_corr:
        best_model = 'simplified_qcml'
        best_corr = simp_corr
    if rank_corr > best_corr:
        best_model = 'ranking_qcml'
        best_corr = rank_corr

    print(f"\nBest model by correlation: {best_model} ({best_corr:.4f})")

    print(f"\nCorrelation comparison:")
    print(f"  Original QCML:   {orig_corr:.4f}")
    print(f"  Simplified QCML: {simp_corr:.4f}")
    print(f"  Ranking QCML:    {rank_corr:.4f}")

    print(f"\nSharpe comparison:")
    print(f"  Original QCML:   {orig_sharpe:.2f}")
    print(f"  Simplified QCML: {simp_sharpe:.2f}")
    print(f"  Ranking QCML:    {rank_sharpe:.2f}")

    if rank_corr > simp_corr:
        print(f"\n✓ Ranking loss IMPROVES correlation ({rank_corr:.4f} vs {simp_corr:.4f})")
    else:
        print(f"\n→ Ranking loss does not improve correlation ({rank_corr:.4f} vs {simp_corr:.4f})")

    if rank_sharpe > simp_sharpe:
        print(f"✓ Ranking loss IMPROVES Sharpe ({rank_sharpe:.2f} vs {simp_sharpe:.2f})")
    else:
        print(f"→ Ranking loss does not improve Sharpe ({rank_sharpe:.2f} vs {simp_sharpe:.2f})")

    # Target check - use best model
    target_corr = 0.02
    target_sign = 0.51
    rank_sign = [s['Sign Acc'] for s in summary if s['Model'] == 'ranking_qcml'][0]

    if rank_corr > target_corr and rank_sign > target_sign:
        print(f"\n✓✓ SUCCESS: RankingQCML meets Phase 1 targets!")
        print(f"   Correlation: {rank_corr:.4f} > {target_corr}")
        print(f"   Sign Accuracy: {rank_sign:.2%} > {target_sign:.0%}")
    else:
        print(f"\n→ RankingQCML does not yet meet Phase 1 targets")
        print(f"   Target: corr > {target_corr}, sign acc > {target_sign:.0%}")
        print(f"   Actual: corr = {rank_corr:.4f}, sign acc = {rank_sign:.2%}")

    return summary


def main():
    """Run the benchmark."""
    start_time = datetime.now()

    # Load data
    data, feature_cols, config = prepare_data()

    # Run walk-forward validation
    results = walk_forward_validation(data, feature_cols, n_folds=5)

    # Analyze results
    summary = analyze_results(results)

    elapsed = datetime.now() - start_time
    print(f"\nTotal runtime: {elapsed}")

    return summary


if __name__ == '__main__':
    main()
