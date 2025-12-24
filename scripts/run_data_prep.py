#!/usr/bin/env python3
"""
Data preparation script.

Downloads ETF data, builds features, and creates train/val/test splits.
Saves processed data for model training.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pickle
from datetime import datetime

from qcml_rotation.data.loader import download_etf_data, download_vix_data, get_trading_dates
from qcml_rotation.data.features import (
    build_features,
    compute_labels,
    merge_features_labels,
    FeatureConfig,
    get_feature_names
)
from qcml_rotation.data.dataset import create_data_splits
from qcml_rotation.utils.helpers import load_config, set_seed


def main(args):
    """Main data preparation pipeline."""
    print("=" * 60)
    print("QCML ETF Rotation - Data Preparation")
    print("=" * 60)

    # Load config
    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    # Download data
    print("\n[1/5] Downloading ETF data...")
    prices, volume = download_etf_data(
        tickers=config["tickers"]["etfs"],
        benchmark=config["tickers"]["benchmark"],
        start_date=config["data"]["start_date"],
        cache_dir=config["data"]["cache_dir"],
        force_refresh=args.force_refresh
    )

    print(f"  Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"  Tickers: {list(prices.columns)}")

    # Download VIX data
    print("\n[2/5] Downloading VIX data...")
    vix_data = download_vix_data(
        start_date=config["data"]["start_date"],
        cache_dir=config["data"]["cache_dir"],
        force_refresh=args.force_refresh
    )

    # Get rebalance dates
    print("\n[3/5] Computing rebalance dates...")
    rebalance_dates = get_trading_dates(prices, freq="W-FRI")
    print(f"  Found {len(rebalance_dates)} weekly rebalance dates")

    # Build features (with extended feature set)
    print("\n[4/5] Building features...")

    # Get feature settings from config, with defaults for new features
    feature_settings = config.get("features", {})
    use_extended = args.extended_features or feature_settings.get("extended", True)

    feature_config = FeatureConfig(
        return_windows=feature_settings.get("return_windows", [1, 5, 20]),
        vol_window=feature_settings.get("vol_window", 20),
        benchmark=config["tickers"]["benchmark"],
        # Extended feature settings
        include_technical=use_extended,
        include_cross_sectional=use_extended,
        include_regime=use_extended,
        rsi_window=feature_settings.get("rsi_window", 14),
        bollinger_window=feature_settings.get("bollinger_window", 20),
        atr_window=feature_settings.get("atr_window", 14),
        momentum_windows=feature_settings.get("momentum_windows", [20, 60])
    )

    # Build features with VIX data
    features = build_features(
        prices,
        rebalance_dates,
        feature_config,
        vix_data=vix_data if use_extended else None
    )
    labels = compute_labels(
        prices,
        rebalance_dates,
        benchmark=config["tickers"]["benchmark"],
        forward_days=5
    )

    # Merge and clean
    data = merge_features_labels(features, labels)

    # Get feature columns (include VIX if extended features enabled)
    include_vix = use_extended and vix_data is not None and len(vix_data) > 0
    feature_cols = get_feature_names(feature_config, include_vix=include_vix)

    print(f"  Features shape: {data.shape}")
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    # Check for any remaining NaNs
    n_nan = data.isna().sum().sum()
    if n_nan > 0:
        print(f"  Warning: {n_nan} NaN values found, dropping...")
        data = data.dropna()

    # Create splits
    print("\n[5/5] Creating train/val/test splits...")

    splits = create_data_splits(
        data,
        feature_cols,
        train_ratio=config["splits"]["train"],
        val_ratio=config["splits"]["val"],
        label_col="excess_return"
    )

    # Save processed data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "processed_data.pkl"
    with open(output_file, "wb") as f:
        pickle.dump({
            "splits": splits,
            "feature_cols": feature_cols,
            "config": config,
            "prices": prices,
            "rebalance_dates": rebalance_dates,
            "created_at": datetime.now().isoformat()
        }, f)

    print(f"\nSaved processed data to {output_file}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)
    print(f"Train samples: {len(splits.train)}")
    print(f"Val samples:   {len(splits.val)}")
    print(f"Test samples:  {len(splits.test)}")
    print(f"\nLabel statistics (excess returns):")
    print(f"  Train mean: {splits.train['excess_return'].mean():.4f}")
    print(f"  Train std:  {splits.train['excess_return'].std():.4f}")
    print(f"  Val mean:   {splits.val['excess_return'].mean():.4f}")
    print(f"  Test mean:  {splits.test['excess_return'].mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for QCML model training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download of data even if cached"
    )
    parser.add_argument(
        "--extended-features",
        action="store_true",
        default=True,
        help="Use extended feature set (technical indicators, cross-sectional, regime)"
    )
    parser.add_argument(
        "--basic-features",
        action="store_true",
        help="Use only basic features (overrides --extended-features)"
    )

    args = parser.parse_args()

    # Handle basic features flag
    if args.basic_features:
        args.extended_features = False

    main(args)
