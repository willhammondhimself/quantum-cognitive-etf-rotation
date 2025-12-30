#!/usr/bin/env python3
"""
Test script for enhanced volatility features.

Compares correlation improvement from OHLC-based estimators vs close-to-close.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from scipy import stats

# Test imports
print("Testing imports...")
try:
    from qcml_rotation.data.loader import (
        download_etf_data_ohlc,
        download_vix_term_structure
    )
    print("  - loader imports OK")
except ImportError as e:
    print(f"  - loader import FAILED: {e}")
    sys.exit(1)

try:
    from qcml_rotation.data.vol_estimators import (
        compute_parkinson_volatility,
        compute_garman_klass_volatility,
        compute_rogers_satchell_volatility,
        compute_yang_zhang_volatility
    )
    print("  - vol_estimators imports OK")
except ImportError as e:
    print(f"  - vol_estimators import FAILED: {e}")
    sys.exit(1)

try:
    from scripts.run_vol_prediction import (
        compute_enhanced_vol_features,
        prepare_enhanced_vol_data,
        SECTOR_ETFS
    )
    print("  - run_vol_prediction imports OK")
except ImportError as e:
    print(f"  - run_vol_prediction import FAILED: {e}")
    sys.exit(1)

print("\nAll imports successful!")


def test_ohlc_download():
    """Test OHLC data download."""
    print("\n" + "="*60)
    print("TEST 1: OHLC Data Download")
    print("="*60)

    open_p, high_p, low_p, close_p, volume = download_etf_data_ohlc(
        tickers=['XLK', 'XLF', 'SPY'],
        start_date='2020-01-01',
        end_date='2024-01-01',
        force_refresh=True
    )

    print(f"Open prices shape: {open_p.shape}")
    print(f"High prices shape: {high_p.shape}")
    print(f"Low prices shape: {low_p.shape}")
    print(f"Close prices shape: {close_p.shape}")
    print(f"Volume shape: {volume.shape}")

    # Validate OHLC consistency
    n_issues = ((high_p < low_p).sum().sum() +
                (high_p < open_p).sum().sum() +
                (high_p < close_p).sum().sum() +
                (low_p > open_p).sum().sum() +
                (low_p > close_p).sum().sum())
    print(f"OHLC consistency issues: {n_issues}")

    return open_p, high_p, low_p, close_p, volume


def test_vol_estimators(open_p, high_p, low_p, close_p):
    """Test volatility estimators."""
    print("\n" + "="*60)
    print("TEST 2: Volatility Estimators")
    print("="*60)

    ticker = 'XLK'
    returns = close_p[ticker].pct_change()

    # Close-to-close (baseline)
    vol_cc = returns.rolling(20).std() * np.sqrt(252)

    # OHLC estimators
    vol_park = compute_parkinson_volatility(high_p[ticker], low_p[ticker], 20)
    vol_gk = compute_garman_klass_volatility(open_p[ticker], high_p[ticker], low_p[ticker], close_p[ticker], 20)
    vol_yz = compute_yang_zhang_volatility(open_p[ticker], high_p[ticker], low_p[ticker], close_p[ticker], 20)

    print(f"\n{ticker} Volatility Estimator Comparison (20-day window):")
    print("-" * 50)

    # Drop NaN for comparison
    valid = vol_cc.notna() & vol_park.notna() & vol_gk.notna() & vol_yz.notna()

    print(f"Close-to-Close mean: {vol_cc[valid].mean():.4f}")
    print(f"Parkinson mean:      {vol_park[valid].mean():.4f}")
    print(f"Garman-Klass mean:   {vol_gk[valid].mean():.4f}")
    print(f"Yang-Zhang mean:     {vol_yz[valid].mean():.4f}")

    # Correlation between estimators
    print(f"\nCorrelation with Close-to-Close:")
    print(f"  Parkinson:    {vol_cc[valid].corr(vol_park[valid]):.4f}")
    print(f"  Garman-Klass: {vol_cc[valid].corr(vol_gk[valid]):.4f}")
    print(f"  Yang-Zhang:   {vol_cc[valid].corr(vol_yz[valid]):.4f}")


def test_vix_term_structure():
    """Test VIX term structure download."""
    print("\n" + "="*60)
    print("TEST 3: VIX Term Structure")
    print("="*60)

    vix_term = download_vix_term_structure(
        start_date='2020-01-01',
        end_date='2024-01-01',
        force_refresh=True
    )

    print(f"VIX term structure shape: {vix_term.shape}")
    print(f"Columns: {list(vix_term.columns)}")
    print(f"\nSample data:")
    print(vix_term.tail())

    # Check for backwardation events
    backwardation_pct = vix_term['vix_backwardation'].mean() * 100
    print(f"\nBackwardation frequency: {backwardation_pct:.1f}%")

    return vix_term


def test_enhanced_features():
    """Test enhanced feature computation."""
    print("\n" + "="*60)
    print("TEST 4: Enhanced Features")
    print("="*60)

    data, sector_close, all_close, feature_cols = prepare_enhanced_vol_data(
        start_date='2018-01-01',
        end_date='2024-01-01'
    )

    print(f"\nData shape: {data.shape}")
    print(f"Number of features: {len(feature_cols)}")

    # Group features by category
    ohlc_features = [c for c in feature_cols if 'parkinson' in c or 'gk' in c or 'yz' in c or 'rs' in c]
    volume_features = [c for c in feature_cols if 'volume' in c]
    asymmetric_features = [c for c in feature_cols if 'semi' in c or 'skew' in c or 'worst' in c or 'best' in c]
    vix_features = [c for c in feature_cols if 'vix' in c]

    print(f"\nFeature breakdown:")
    print(f"  OHLC volatility: {len(ohlc_features)}")
    print(f"  Volume: {len(volume_features)}")
    print(f"  Asymmetric: {len(asymmetric_features)}")
    print(f"  VIX: {len(vix_features)}")

    return data, feature_cols


def test_predictive_power(data, feature_cols):
    """Test predictive power of features."""
    print("\n" + "="*60)
    print("TEST 5: Feature Predictive Power")
    print("="*60)

    target = data['realized_vol']

    print("\nTop 20 features by correlation with realized volatility:")
    print("-" * 60)

    correlations = {}
    for col in feature_cols:
        if col in data.columns:
            corr = data[col].corr(target)
            if not np.isnan(corr):
                correlations[col] = corr

    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    for i, (feat, corr) in enumerate(sorted_corrs[:20]):
        print(f"{i+1:2}. {feat:35s} {corr:+.4f}")

    # Compare OHLC vs close-to-close
    ohlc_corrs = [(k, v) for k, v in correlations.items() if 'yz' in k or 'gk' in k or 'parkinson' in k]
    cc_corrs = [(k, v) for k, v in correlations.items() if 'vol_cc' in k]

    print(f"\nOHLC estimator avg correlation: {np.mean([abs(c) for _, c in ohlc_corrs]):.4f}")
    print(f"Close-to-close avg correlation: {np.mean([abs(c) for _, c in cc_corrs]):.4f}")


def main():
    """Run all tests."""
    print("="*60)
    print("ENHANCED VOLATILITY FEATURES TEST SUITE")
    print("="*60)

    # Test 1: OHLC download
    open_p, high_p, low_p, close_p, volume = test_ohlc_download()

    # Test 2: Volatility estimators
    test_vol_estimators(open_p, high_p, low_p, close_p)

    # Test 3: VIX term structure
    test_vix_term_structure()

    # Test 4: Enhanced features
    data, feature_cols = test_enhanced_features()

    # Test 5: Predictive power
    test_predictive_power(data, feature_cols)

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()
