#!/usr/bin/env python3
"""
Run comprehensive experiments to find statistically significant improvements.

Tests:
1. Feature importance and selection
2. Different model variants (ranking loss, pairwise, classification)
3. Different portfolio strategies
4. Rigorous statistical significance testing
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pickle
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Tuple

from qcml_rotation.data.dataset import ETFDataset
from qcml_rotation.analysis.feature_importance import (
    analyze_feature_importance,
    select_best_features,
    compute_univariate_importance
)
from qcml_rotation.models.variants import (
    RankingMLP, PairwiseRankingMLP, TopKClassifier, EnsembleModel,
    ModelVariantConfig, train_model_variant
)
from qcml_rotation.models.baselines.pca_ridge import PCARidge, PCARidgeConfig
from qcml_rotation.backtest.walk_forward import WalkForwardValidator
from qcml_rotation.backtest.metrics import (
    compute_significance, permutation_test, bootstrap_sharpe_ci
)
from qcml_rotation.backtest.portfolio import PortfolioConfig
from qcml_rotation.utils.helpers import load_config, set_seed, get_device


def run_feature_importance_analysis(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: List[str]
) -> Dict:
    """Run comprehensive feature importance analysis."""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    result = analyze_feature_importance(
        X, y, feature_cols,
        n_permutations=50,
        random_state=42
    )

    print("\nTop 10 Features by Information Coefficient (IC):")
    print(result.correlations.head(10).to_string(index=False))

    print(f"\nSignificant features (p < 0.05): {len(result.significant_features)}")
    if result.significant_features:
        print(f"  {result.significant_features}")

    print("\nOverall Feature Ranking:")
    print(result.overall_ranking.head(10).to_string(index=False))

    return {
        'significant_features': result.significant_features,
        'top_10_features': result.overall_ranking['feature'].head(10).tolist(),
        'correlations': result.correlations.to_dict('records')
    }


def create_model_fn(
    model_type: str,
    input_dim: int,
    config: Dict,
    device
) -> Tuple[callable, callable]:
    """
    Create train and predict functions for a model type.

    Returns (train_fn, predict_fn)
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

    elif model_type == "ranking_mlp":
        model_config = ModelVariantConfig(
            input_dim=input_dim,
            hidden_dims=config.get("hidden_dims", [64, 32]),
            dropout=config.get("dropout", 0.2),
            use_ranking_loss=True,
            ranking_weight=config.get("ranking_weight", 0.5)
        )

        def train_fn(X, y):
            nonlocal model
            model = RankingMLP(model_config)
            model.to(device)
            train_model_variant(
                model, X, y,
                epochs=config.get("epochs", 100),
                lr=config.get("lr", 0.001),
                device=device
            )

        def predict_fn(X):
            model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(device)
                return model(X_t).cpu().numpy()

        model = RankingMLP(model_config)
        model.to(device)

        return train_fn, predict_fn

    elif model_type == "pairwise_mlp":
        model_config = ModelVariantConfig(
            input_dim=input_dim,
            hidden_dims=config.get("hidden_dims", [64, 32]),
            dropout=config.get("dropout", 0.2)
        )

        def train_fn(X, y):
            nonlocal model
            model = PairwiseRankingMLP(model_config)
            model.to(device)
            train_model_variant(
                model, X, y,
                epochs=config.get("epochs", 100),
                lr=config.get("lr", 0.001),
                device=device
            )

        def predict_fn(X):
            model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(device)
                return model(X_t).cpu().numpy()

        model = PairwiseRankingMLP(model_config)
        model.to(device)

        return train_fn, predict_fn

    elif model_type == "ensemble":
        model_config = ModelVariantConfig(
            input_dim=input_dim,
            hidden_dims=config.get("hidden_dims", [64, 32]),
            dropout=config.get("dropout", 0.2),
            use_ranking_loss=True,
            ranking_weight=config.get("ranking_weight", 0.3)
        )

        def train_fn(X, y):
            nonlocal ensemble
            ensemble = EnsembleModel(RankingMLP, model_config, n_models=5)
            ensemble.to(device)

            # Train each model in ensemble
            X_t = torch.FloatTensor(X).to(device)
            y_t = torch.FloatTensor(y).to(device)

            for i, model in enumerate(ensemble.models):
                torch.manual_seed(42 + i)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                model.train()
                for _ in range(config.get("epochs", 50)):
                    optimizer.zero_grad()
                    preds = model(X_t)
                    loss = model.compute_loss(preds, y_t)
                    loss.backward()
                    optimizer.step()

        def predict_fn(X):
            X_t = torch.FloatTensor(X).to(device)
            return ensemble.predict(X_t).cpu().numpy()

        ensemble = EnsembleModel(RankingMLP, model_config, n_models=5)
        ensemble.to(device)

        return train_fn, predict_fn

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_walk_forward_experiment(
    validator: WalkForwardValidator,
    model_type: str,
    model_config: Dict,
    features: np.ndarray,
    labels: np.ndarray,
    date_indices: np.ndarray,
    ticker_indices: np.ndarray,
    idx_to_date: Dict,
    idx_to_ticker: Dict,
    device,
    feature_subset: List[int] = None
) -> Dict:
    """
    Run walk-forward validation for a model configuration.

    Returns results dictionary with metrics and significance.
    """
    input_dim = features.shape[1] if feature_subset is None else len(feature_subset)

    if feature_subset is not None:
        features_subset = features[:, feature_subset]
    else:
        features_subset = features

    train_fn, predict_fn = create_model_fn(
        model_type, input_dim, model_config, device
    )

    result = validator.run(
        train_fn=train_fn,
        predict_fn=predict_fn,
        features=features_subset,
        labels=labels,
        date_indices=date_indices,
        ticker_indices=ticker_indices,
        idx_to_date=idx_to_date,
        idx_to_ticker=idx_to_ticker,
        model_name=model_type,
        window_type="expanding",
        verbose=False
    )

    # Statistical significance
    sig = compute_significance(result.returns, n_bootstrap=500, random_state=42)

    # Permutation test
    _, perm_p, _ = permutation_test(
        result.predictions, result.actuals,
        n_permutations=500, random_state=42
    )

    return {
        'model_type': model_type,
        'config': model_config,
        'sharpe': result.metrics.sharpe_ratio,
        'total_return': result.metrics.total_return,
        'max_drawdown': result.metrics.max_drawdown,
        'hit_rate': result.metrics.hit_rate,
        'sharpe_ci_lower': sig.sharpe_ci_lower,
        'sharpe_ci_upper': sig.sharpe_ci_upper,
        'sharpe_pvalue': sig.sharpe_p_value,
        'permutation_pvalue': perm_p,
        'is_significant': sig.is_significant,
        'n_features': input_dim
    }


def main(args):
    """Run comprehensive experiments."""
    print("=" * 70)
    print("QCML ETF Rotation - Comprehensive Experiments")
    print("=" * 70)

    # Load config and data
    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    # Load processed data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    splits = data["splits"]
    feature_cols = data["feature_cols"]
    prices = data["prices"]
    rebalance_dates = data["rebalance_dates"]

    print(f"\nLoaded data: {len(feature_cols)} features")

    # Combine all data for walk-forward
    combined_data = pd.concat([splits.train, splits.val, splits.test])
    combined_data = combined_data.sort_index()

    full_dataset = ETFDataset(combined_data, feature_cols)

    # Build index mappings
    unique_dates = combined_data.index.get_level_values("date").unique().sort_values()
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    idx_to_date = {i: d for d, i in date_to_idx.items()}
    idx_to_ticker = {i: t for i, t in enumerate(full_dataset.tickers)}

    sample_dates = combined_data.index.get_level_values("date")
    date_indices = np.array([date_to_idx[d] for d in sample_dates])

    X = full_dataset.features.numpy()
    y = full_dataset.labels.numpy()
    ticker_indices = full_dataset.ticker_indices.numpy()

    print(f"Total samples: {len(X)}")

    # =========================================================================
    # 1. Feature Importance Analysis
    # =========================================================================
    feature_results = run_feature_importance_analysis(X, y, feature_cols)

    # Get feature subsets
    top_10_indices = [feature_cols.index(f) for f in feature_results['top_10_features']]
    significant_indices = [feature_cols.index(f) for f in feature_results['significant_features']] \
        if feature_results['significant_features'] else None

    # =========================================================================
    # 2. Walk-Forward Experiments
    # =========================================================================
    tickers = config["tickers"]["etfs"]

    # Test different portfolio strategies
    strategies = [
        {'top_k': 3, 'strategy': 'long_short'},
        {'top_k': 5, 'strategy': 'long_short'},
        {'top_k': 3, 'strategy': 'long_only'},
    ]

    all_results = []

    for strat in strategies:
        print(f"\n{'=' * 60}")
        print(f"Testing strategy: top_k={strat['top_k']}, {strat['strategy']}")
        print("=" * 60)

        portfolio_config = PortfolioConfig(
            top_k=strat['top_k'],
            bottom_k=strat['top_k'],
            strategy=strat['strategy'],
            transaction_cost_bps=10
        )

        validator = WalkForwardValidator(
            prices=prices,
            dates=list(unique_dates),
            tickers=tickers,
            min_train_weeks=52,
            portfolio_config=portfolio_config,
            benchmark=config["tickers"]["benchmark"]
        )

        # Model configurations to test
        model_configs = [
            ("pca_ridge", {"pca_components": 5, "ridge_alpha": 1.0}),
            ("pca_ridge", {"pca_components": 3, "ridge_alpha": 10.0}),
            ("ranking_mlp", {"epochs": 50, "ranking_weight": 0.3}),
            ("ranking_mlp", {"epochs": 50, "ranking_weight": 0.7}),
            ("pairwise_mlp", {"epochs": 50}),
            ("ensemble", {"epochs": 30, "ranking_weight": 0.5}),
        ]

        for model_type, model_cfg in model_configs:
            print(f"\n--- {model_type} ({model_cfg}) ---")

            # Test with all features
            result = run_walk_forward_experiment(
                validator, model_type, model_cfg,
                X, y, date_indices, ticker_indices,
                idx_to_date, idx_to_ticker, device,
                feature_subset=None
            )
            result['strategy'] = strat['strategy']
            result['top_k'] = strat['top_k']
            result['feature_set'] = 'all'
            all_results.append(result)

            print(f"  All features: Sharpe={result['sharpe']:.3f}, "
                  f"p={result['sharpe_pvalue']:.3f}")

            # Test with top 10 features
            result_top10 = run_walk_forward_experiment(
                validator, model_type, model_cfg,
                X, y, date_indices, ticker_indices,
                idx_to_date, idx_to_ticker, device,
                feature_subset=top_10_indices
            )
            result_top10['strategy'] = strat['strategy']
            result_top10['top_k'] = strat['top_k']
            result_top10['feature_set'] = 'top_10'
            all_results.append(result_top10)

            print(f"  Top 10 features: Sharpe={result_top10['sharpe']:.3f}, "
                  f"p={result_top10['sharpe_pvalue']:.3f}")

    # =========================================================================
    # 3. Results Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('sharpe', ascending=False)

    # Best results
    print("\nTop 10 Configurations by Sharpe Ratio:")
    display_cols = ['model_type', 'strategy', 'top_k', 'feature_set',
                    'sharpe', 'sharpe_pvalue', 'total_return', 'is_significant']
    print(results_df[display_cols].head(10).to_string(index=False))

    # Significant results
    significant_results = results_df[results_df['is_significant']]
    print(f"\nSignificant configurations (p < 0.05): {len(significant_results)}")

    if len(significant_results) > 0:
        print("\nStatistically Significant Results:")
        print(significant_results[display_cols].to_string(index=False))
    else:
        print("\nNo statistically significant configurations found.")

    # Best by strategy
    print("\nBest Configuration per Strategy:")
    for strategy in results_df['strategy'].unique():
        strat_results = results_df[results_df['strategy'] == strategy]
        best = strat_results.iloc[0]
        print(f"\n{strategy}:")
        print(f"  Model: {best['model_type']}")
        print(f"  Features: {best['feature_set']} ({best['n_features']})")
        print(f"  Sharpe: {best['sharpe']:.3f} (95% CI: [{best['sharpe_ci_lower']:.3f}, {best['sharpe_ci_upper']:.3f}])")
        print(f"  P-value: {best['sharpe_pvalue']:.4f}")
        print(f"  Total Return: {best['total_return']:.2%}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {
        'timestamp': datetime.now().isoformat(),
        'feature_importance': feature_results,
        'experiments': all_results,
        'summary': {
            'n_configurations': len(all_results),
            'n_significant': len(significant_results),
            'best_sharpe': results_df['sharpe'].max(),
            'best_config': results_df.iloc[0].to_dict()
        }
    }

    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    results_df.to_csv(output_dir / "experiment_results.csv", index=False)

    print(f"\nResults saved to {output_dir}")

    # =========================================================================
    # 4. Conclusion
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    best = results_df.iloc[0]
    print(f"""
Based on {len(all_results)} configurations tested:

1. BEST CONFIGURATION:
   - Model: {best['model_type']}
   - Strategy: {best['strategy']} (top_k={best['top_k']})
   - Features: {best['feature_set']}
   - Sharpe Ratio: {best['sharpe']:.3f}
   - 95% Confidence Interval: [{best['sharpe_ci_lower']:.3f}, {best['sharpe_ci_upper']:.3f}]
   - P-value (Sharpe > 0): {best['sharpe_pvalue']:.4f}

2. STATISTICAL SIGNIFICANCE:
   - {len(significant_results)}/{len(all_results)} configurations show p < 0.05
   - Best p-value: {results_df['sharpe_pvalue'].min():.4f}

3. RECOMMENDATION:
""")

    if best['sharpe'] > 0 and best['sharpe_pvalue'] < 0.1:
        print("   Promising results! Consider further validation with:")
        print("   - Out-of-sample testing on more recent data")
        print("   - Transaction cost sensitivity analysis")
        print("   - Regime-specific performance analysis")
    elif best['sharpe'] > -0.2:
        print("   Near-zero performance suggests weak but potentially exploitable signal.")
        print("   Consider:")
        print("   - Alternative data sources (fundamental, sentiment)")
        print("   - Different rebalancing frequencies")
        print("   - Market timing overlay")
    else:
        print("   No profitable signal found with current approach.")
        print("   Consider pivoting to:")
        print("   - Different asset classes")
        print("   - Longer prediction horizons")
        print("   - Alternative trading strategies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive experiments")
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
        default="outputs/experiments",
        help="Directory to save results"
    )

    args = parser.parse_args()
    main(args)
