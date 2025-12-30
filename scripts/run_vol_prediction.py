#!/usr/bin/env python3
"""
QCML Volatility Prediction Experiment

Use QCML to predict volatility instead of returns.
Volatility is persistent and predictable - this should actually work.

Strategy:
1. Predict next-week realized volatility for each ETF
2. Use predictions for inverse-volatility position sizing
3. Compare vol-adjusted momentum vs simple momentum
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml_rotation.models.qcml import (
    SimplifiedQCML, SimplifiedQCMLConfig,
    RankingQCML, RankingQCMLConfig,
    QuantumEnhancedQCML, QuantumEnhancedConfig
)
from qcml_rotation.data.loader import (
    download_etf_data, download_vix_data,
    download_etf_data_ohlc, download_vix_term_structure
)
from qcml_rotation.data.vol_estimators import (
    compute_parkinson_volatility,
    compute_garman_klass_volatility,
    compute_rogers_satchell_volatility,
    compute_yang_zhang_volatility,
    compute_volatility_features_for_ticker
)

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available, ensemble will use 2 models")

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# ETF universe
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE', 'XLC']
# Include SPY for trend filter
ALL_ETFS = SECTOR_ETFS + ['SPY']


def compute_realized_volatility(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute realized volatility (forward-looking for labels).

    Parameters
    ----------
    prices : DataFrame with ETF prices
    window : int, lookback window for vol calculation

    Returns
    -------
    vol : DataFrame with realized vol (annualized)
    """
    returns = prices.pct_change()
    # Forward-looking realized vol (what we're predicting)
    realized_vol = returns.rolling(window).std().shift(-window) * np.sqrt(252)
    return realized_vol


def compute_vol_features(prices: pd.DataFrame, vix: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Compute features for volatility prediction.

    Features:
    - Past realized vol (5, 10, 20, 60 day)
    - Recent returns
    - Return magnitude (abs)
    - VIX level and change
    - High-low range
    """
    returns = prices.pct_change()

    features_list = []

    for ticker in prices.columns:
        ret = returns[ticker]
        price = prices[ticker]

        df = pd.DataFrame(index=prices.index)

        # Past volatility at different horizons
        df['vol_5d'] = ret.rolling(5).std() * np.sqrt(252)
        df['vol_10d'] = ret.rolling(10).std() * np.sqrt(252)
        df['vol_20d'] = ret.rolling(20).std() * np.sqrt(252)
        df['vol_60d'] = ret.rolling(60).std() * np.sqrt(252)

        # Vol changes
        df['vol_change_5d'] = df['vol_5d'] - df['vol_5d'].shift(5)
        df['vol_ratio_5_20'] = df['vol_5d'] / (df['vol_20d'] + 0.01)

        # Returns
        df['ret_1d'] = ret
        df['ret_5d'] = ret.rolling(5).sum()
        df['ret_20d'] = ret.rolling(20).sum()

        # Absolute returns (magnitude)
        df['abs_ret_1d'] = ret.abs()
        df['abs_ret_5d'] = ret.abs().rolling(5).mean()
        df['abs_ret_20d'] = ret.abs().rolling(20).mean()

        # Squared returns
        df['sq_ret_5d'] = (ret ** 2).rolling(5).mean()

        # Max drawdown recent
        rolling_max = price.rolling(20).max()
        df['drawdown_20d'] = (price - rolling_max) / rolling_max

        # === NEW FEATURES FOR PHASE 2 ===
        # Volatility of volatility (vol clustering signal)
        df['vol_of_vol'] = df['vol_5d'].rolling(20).std()

        # Vol term structure (near vs far - high = vol spike, low = calm)
        df['vol_term_structure'] = df['vol_5d'] / (df['vol_20d'] + 0.01)

        # Large move indicator (jump detection)
        df['large_move'] = (df['abs_ret_1d'] > 2 * df['vol_20d']).astype(float)

        # Rolling large move count (recent jump frequency)
        df['large_move_count_20d'] = df['large_move'].rolling(20).sum()

        df['ticker'] = ticker
        features_list.append(df)

    features = pd.concat(features_list)

    # Cross-sectional vol rank (relative risk within universe on each date)
    # Requires all tickers, so computed after concat
    features = features.reset_index()
    date_col = 'Date' if 'Date' in features.columns else 'index'
    features['vol_rank'] = features.groupby(date_col)['vol_5d'].rank(pct=True)
    features = features.set_index([date_col, 'ticker'])

    # Add VIX if available
    if vix is not None:
        features = features.reset_index()
        features = features.merge(
            vix.to_frame('vix').reset_index(),
            left_on='Date' if 'Date' in features.columns else 'index',
            right_on='Date',
            how='left'
        )
        features['vix_change_5d'] = features['vix'] - features['vix'].shift(5)
        # VIX momentum (% change over 5 days)
        features['vix_momentum'] = features['vix'].pct_change(5)
        # VIX term structure proxy (current vs 20-day MA)
        features['vix_ma_20d'] = features['vix'].rolling(20).mean()
        features['vix_term_structure'] = features['vix'] / (features['vix_ma_20d'] + 0.01)
        features = features.set_index(['Date', 'ticker'] if 'Date' in features.columns else ['index', 'ticker'])

    return features


def compute_enhanced_vol_features(
    open_prices: pd.DataFrame,
    high_prices: pd.DataFrame,
    low_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    volume: pd.DataFrame,
    vix_term: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute enhanced features for volatility prediction using OHLC data.

    This version includes:
    - OHLC-based volatility estimators (5-7x more efficient)
    - Volume features
    - Asymmetric features (leverage effect)
    - VIX term structure features

    Parameters
    ----------
    open_prices : DataFrame with opening prices
    high_prices : DataFrame with high prices
    low_prices : DataFrame with low prices
    close_prices : DataFrame with closing prices
    volume : DataFrame with volume data
    vix_term : DataFrame with VIX term structure (optional)

    Returns
    -------
    features : DataFrame with enhanced features
    """
    returns = close_prices.pct_change()
    features_list = []

    for ticker in close_prices.columns:
        ret = returns[ticker]
        close = close_prices[ticker]
        open_ = open_prices[ticker]
        high = high_prices[ticker]
        low = low_prices[ticker]
        vol = volume[ticker]

        df = pd.DataFrame(index=close_prices.index)

        # ============================================
        # 1. OHLC-BASED VOLATILITY ESTIMATORS
        # ============================================
        # These are 5-7x more efficient than close-to-close

        # Parkinson (uses high-low range, 5.2x efficient)
        df['vol_parkinson_5d'] = compute_parkinson_volatility(high, low, 5)
        df['vol_parkinson_10d'] = compute_parkinson_volatility(high, low, 10)
        df['vol_parkinson_20d'] = compute_parkinson_volatility(high, low, 20)

        # Garman-Klass (uses OHLC, 7.4x efficient)
        df['vol_gk_5d'] = compute_garman_klass_volatility(open_, high, low, close, 5)
        df['vol_gk_10d'] = compute_garman_klass_volatility(open_, high, low, close, 10)
        df['vol_gk_20d'] = compute_garman_klass_volatility(open_, high, low, close, 20)

        # Yang-Zhang (best for overnight gaps)
        df['vol_yz_5d'] = compute_yang_zhang_volatility(open_, high, low, close, 5)
        df['vol_yz_10d'] = compute_yang_zhang_volatility(open_, high, low, close, 10)
        df['vol_yz_20d'] = compute_yang_zhang_volatility(open_, high, low, close, 20)

        # Rogers-Satchell (handles drift)
        df['vol_rs_5d'] = compute_rogers_satchell_volatility(open_, high, low, close, 5)

        # Close-to-close (baseline for comparison)
        df['vol_cc_5d'] = ret.rolling(5).std() * np.sqrt(252)
        df['vol_cc_10d'] = ret.rolling(10).std() * np.sqrt(252)
        df['vol_cc_20d'] = ret.rolling(20).std() * np.sqrt(252)
        df['vol_cc_60d'] = ret.rolling(60).std() * np.sqrt(252)

        # Estimator ratios (unusual when range >> close-to-close)
        df['vol_ratio_gk_cc'] = df['vol_gk_5d'] / (df['vol_cc_5d'] + 1e-8)
        df['vol_ratio_yz_cc'] = df['vol_yz_5d'] / (df['vol_cc_5d'] + 1e-8)

        # Vol term structure
        df['vol_term_5_20'] = df['vol_yz_5d'] / (df['vol_yz_20d'] + 1e-8)
        df['vol_term_5_60'] = df['vol_yz_5d'] / (df['vol_cc_60d'] + 1e-8)

        # Vol changes
        df['vol_change_5d'] = df['vol_yz_5d'] - df['vol_yz_5d'].shift(5)
        df['vol_change_pct_5d'] = df['vol_yz_5d'].pct_change(5)

        # Vol of vol (clustering)
        df['vol_of_vol'] = df['vol_yz_5d'].rolling(20).std()

        # ============================================
        # 2. VOLUME FEATURES
        # ============================================
        # Volume often leads volatility

        # Relative volume (vs 20-day average)
        vol_ma = vol.rolling(20).mean()
        df['volume_ratio'] = vol / (vol_ma + 1)

        # Volume change
        df['volume_change'] = vol.pct_change()
        df['volume_change_5d'] = vol.pct_change(5)

        # High volume indicator (spike detection)
        df['high_volume'] = (vol > 2 * vol_ma).astype(float)
        df['high_volume_count_20d'] = df['high_volume'].rolling(20).sum()

        # Volume-weighted volatility
        vol_norm = vol / vol.rolling(20).sum()
        df['vol_weighted_vol'] = (ret.abs() * vol_norm).rolling(20).sum() * np.sqrt(252)

        # Volume-volatility correlation (lead indicator)
        df['vol_volume_corr'] = ret.abs().rolling(20).corr(vol)

        # ============================================
        # 3. ASYMMETRIC FEATURES (Leverage Effect)
        # ============================================
        # Negative returns lead to higher volatility

        # Downside semi-variance
        neg_ret = ret.where(ret < 0, 0)
        df['downside_semivar'] = neg_ret.rolling(20).std() * np.sqrt(252)

        # Upside semi-variance
        pos_ret = ret.where(ret > 0, 0)
        df['upside_semivar'] = pos_ret.rolling(20).std() * np.sqrt(252)

        # Asymmetry ratio
        df['semivar_ratio'] = df['downside_semivar'] / (df['upside_semivar'] + 1e-8)

        # Negative return count (bad day frequency)
        df['neg_return_pct'] = (ret < 0).rolling(20).sum() / 20

        # Worst day in window
        df['worst_day_20d'] = ret.rolling(20).min()
        df['best_day_20d'] = ret.rolling(20).max()

        # Skewness of returns (asymmetry indicator)
        df['return_skew_20d'] = ret.rolling(20).skew()

        # ============================================
        # 4. RETURN FEATURES
        # ============================================
        df['ret_1d'] = ret
        df['ret_5d'] = ret.rolling(5).sum()
        df['ret_20d'] = ret.rolling(20).sum()
        df['abs_ret_1d'] = ret.abs()
        df['abs_ret_5d'] = ret.abs().rolling(5).mean()

        # Jump detection
        df['large_move'] = (df['abs_ret_1d'] > 2 * df['vol_cc_20d']).astype(float)
        df['large_move_count_20d'] = df['large_move'].rolling(20).sum()

        # Drawdown
        rolling_max = close.rolling(20).max()
        df['drawdown_20d'] = (close - rolling_max) / rolling_max

        # ============================================
        # 5. INTRADAY RANGE FEATURES
        # ============================================
        intraday_range = (high - low) / close
        df['avg_range_5d'] = intraday_range.rolling(5).mean()
        df['avg_range_20d'] = intraday_range.rolling(20).mean()
        df['max_range_20d'] = intraday_range.rolling(20).max()

        # Overnight vs intraday separation
        overnight_ret = np.log(open_ / close.shift(1))
        intraday_ret = np.log(close / open_)
        df['overnight_vol_5d'] = overnight_ret.rolling(5).std() * np.sqrt(252)
        df['intraday_vol_5d'] = intraday_ret.rolling(5).std() * np.sqrt(252)
        df['overnight_intraday_ratio'] = df['overnight_vol_5d'] / (df['intraday_vol_5d'] + 1e-8)

        df['ticker'] = ticker
        features_list.append(df)

    features = pd.concat(features_list)

    # Cross-sectional rank (relative risk within universe)
    features = features.reset_index()
    date_col = 'Date' if 'Date' in features.columns else 'index'
    features['vol_rank'] = features.groupby(date_col)['vol_yz_5d'].rank(pct=True)
    features = features.set_index([date_col, 'ticker'])

    # ============================================
    # 6. VIX TERM STRUCTURE FEATURES
    # ============================================
    if vix_term is not None and len(vix_term) > 0:
        features = features.reset_index()

        # Merge VIX term structure features
        vix_features = vix_term[['vix', 'vix3m', 'vix_term_structure', 'vix_term_slope',
                                  'vix_term_percentile', 'vix_backwardation',
                                  'vix_change_5d', 'vix_zscore', 'vix_spike']].copy()
        vix_features = vix_features.reset_index()
        vix_features.columns = ['Date'] + list(vix_features.columns[1:])

        features = features.merge(
            vix_features,
            left_on='Date' if 'Date' in features.columns else 'index',
            right_on='Date',
            how='left'
        )

        features = features.set_index(['Date', 'ticker'] if 'Date' in features.columns else ['index', 'ticker'])

    return features


def prepare_enhanced_vol_data(
    start_date: str = '2010-01-01',
    end_date: str = '2024-01-01'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Download OHLC data and prepare enhanced features/labels for vol prediction.

    This version uses OHLC-based volatility estimators which are 5-7x more efficient.

    Returns
    -------
    data : DataFrame with features and labels
    sector_prices : DataFrame with sector ETF close prices only
    all_close : DataFrame with all close prices including SPY
    feature_cols : list of feature column names
    """
    print("Downloading OHLC data for enhanced volatility features...")
    open_prices, high_prices, low_prices, close_prices, volume = download_etf_data_ohlc(
        tickers=ALL_ETFS,
        start_date=start_date,
        end_date=end_date,
        force_refresh=True
    )

    print("Downloading VIX term structure...")
    vix_term = download_vix_term_structure(
        start_date=start_date,
        end_date=end_date,
        force_refresh=True
    )

    # Separate sector data
    sector_open = open_prices[SECTOR_ETFS]
    sector_high = high_prices[SECTOR_ETFS]
    sector_low = low_prices[SECTOR_ETFS]
    sector_close = close_prices[SECTOR_ETFS]
    sector_volume = volume[SECTOR_ETFS]

    print("Computing enhanced volatility features...")
    features = compute_enhanced_vol_features(
        sector_open, sector_high, sector_low, sector_close, sector_volume, vix_term
    )

    # Compute labels (forward-looking realized vol)
    print("Computing realized volatility labels...")
    realized_vol = compute_realized_volatility(sector_close, window=5)

    # Reshape labels to match features
    labels_list = []
    for ticker in SECTOR_ETFS:
        df = pd.DataFrame({
            'realized_vol': realized_vol[ticker],
            'ticker': ticker
        }, index=realized_vol.index)
        labels_list.append(df)

    labels = pd.concat(labels_list)
    labels = labels.reset_index()
    date_col = 'Date' if 'Date' in labels.columns else 'index'
    labels = labels.set_index([date_col, 'ticker'])

    # Combine features and labels
    data = features.join(labels, how='inner')

    # Get feature columns (exclude ticker and label)
    feature_cols = [c for c in data.columns if c not in ['ticker', 'realized_vol']]

    # Drop NaN rows
    n_before = len(data)
    data = data.dropna()
    n_dropped = n_before - len(data)
    print(f"Dropped {n_dropped} rows with NaN values ({100*n_dropped/n_before:.1f}%)")

    print(f"Prepared data: {len(data)} samples, {len(feature_cols)} features")
    print(f"Feature categories: OHLC vol estimators, volume, asymmetric, returns, VIX term structure")

    return data, sector_close, close_prices, feature_cols


def prepare_vol_data(
    start_date: str = '2010-01-01',
    end_date: str = '2024-01-01'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Download data and prepare features/labels for vol prediction.

    Returns
    -------
    data : DataFrame with features and labels
    sector_prices : DataFrame with sector ETF prices only
    all_prices : DataFrame with all prices including SPY
    feature_cols : list of feature column names
    """
    print("Downloading ETF data (including SPY for trend filter)...")
    all_prices, _ = download_etf_data(
        tickers=ALL_ETFS,
        start_date=start_date,
        end_date=end_date,
        force_refresh=True
    )

    # Separate sector prices for features/labels
    sector_prices = all_prices[SECTOR_ETFS]

    print("Downloading VIX data...")
    try:
        vix = download_vix_data(start_date, end_date)
    except:
        vix = None
        print("  VIX data not available, continuing without it")

    print("Computing features...")
    features = compute_vol_features(sector_prices, vix)

    print("Computing volatility labels...")
    realized_vol = compute_realized_volatility(sector_prices, window=5)

    # Reshape realized_vol to match features
    vol_labels = realized_vol.stack().to_frame('target_vol')
    vol_labels.index.names = ['Date', 'ticker']

    # Merge
    data = features.join(vol_labels, how='inner')

    # Drop NaN
    data = data.dropna()

    # Get feature columns (exclude ticker and target)
    feature_cols = [c for c in data.columns if c not in ['ticker', 'target_vol']]

    print(f"Data shape: {data.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Date range: {data.index.get_level_values(0).min()} to {data.index.get_level_values(0).max()}")

    return data, sector_prices, all_prices, feature_cols


def train_vol_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'simplified',
    epochs: int = 100,
    verbose: bool = False
) -> nn.Module:
    """
    Train QCML model for volatility prediction.

    Uses log(vol) as target to handle skewness.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = X_train.shape[1]

    # Use log vol as target (more normal distribution)
    y_train_log = np.log(y_train + 0.01)
    y_val_log = np.log(y_val + 0.01)

    # Create model
    if model_type == 'simplified':
        config = SimplifiedQCMLConfig(
            hidden_dim=64,
            embed_dim=32,
            dropout=0.1,
            lr=0.001,
            weight_decay=1e-4
        )
        model = SimplifiedQCML(input_dim, config)
    else:
        config = QuantumEnhancedConfig(
            hilbert_dim=32,
            n_observables=4,
            use_dual_pathway=False,
            dropout=0.1
        )
        model = QuantumEnhancedQCML(input_dim, config)

    model = model.to(device)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train_log).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val_log).to(device)

    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    best_val_loss = float('inf')
    best_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = nn.functional.mse_loss(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = nn.functional.mse_loss(val_pred, y_val_t)

            # Correlation on original scale
            val_pred_exp = torch.exp(val_pred).cpu().numpy()
            val_corr = np.corrcoef(val_pred_exp, y_val)[0, 1]

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f}, "
                  f"val_loss={val_loss:.4f}, val_corr={val_corr:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return model


def train_ranking_vol_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    verbose: bool = False
) -> nn.Module:
    """Train RankingQCML model for volatility prediction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]

    # Use log vol as target
    y_train_log = np.log(y_train + 0.01)
    y_val_log = np.log(y_val + 0.01)

    config = RankingQCMLConfig(
        hidden_dim=64,
        embed_dim=32,
        dropout=0.2,
        lr=0.001,
        weight_decay=1e-3,
        mse_weight=0.5,  # Higher MSE weight for vol prediction
        ranking_weight=0.5
    )
    model = RankingQCML(input_dim, config).to(device)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train_log).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val_log).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    best_val_loss = float('inf')
    best_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss, _ = model.compute_loss(X_train_t, y_train_t, week_indices=None)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = nn.functional.mse_loss(val_pred, y_val_t)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    return model


def train_xgboost_vol_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> 'xgb.XGBRegressor':
    """Train XGBoost model for volatility prediction."""
    if not HAS_XGBOOST:
        return None

    # Use log vol as target
    y_train_log = np.log(y_train + 0.01)
    y_val_log = np.log(y_val + 0.01)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        verbose=False
    )

    return model


class EnsembleVolPredictor:
    """Ensemble of SimplifiedQCML, RankingQCML, and XGBoost for vol prediction."""

    def __init__(
        self,
        simplified_model: nn.Module,
        ranking_model: nn.Module,
        xgb_model=None,
        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
    ):
        self.simplified_model = simplified_model
        self.ranking_model = ranking_model
        self.xgb_model = xgb_model
        self.weights = weights
        self.device = next(simplified_model.parameters()).device

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions."""
        X_t = torch.FloatTensor(X).to(self.device)

        # SimplifiedQCML prediction
        self.simplified_model.eval()
        with torch.no_grad():
            pred_simp_log = self.simplified_model(X_t).cpu().numpy()
        pred_simp = np.exp(pred_simp_log)

        # RankingQCML prediction
        self.ranking_model.eval()
        with torch.no_grad():
            pred_rank_log = self.ranking_model(X_t).cpu().numpy()
        pred_rank = np.exp(pred_rank_log)

        # XGBoost prediction (if available)
        if self.xgb_model is not None and HAS_XGBOOST:
            pred_xgb_log = self.xgb_model.predict(X)
            pred_xgb = np.exp(pred_xgb_log)
            # Use all three models
            w1, w2, w3 = self.weights
            ensemble_pred = w1 * pred_simp + w2 * pred_rank + w3 * pred_xgb
        else:
            # Use only two models (normalized weights)
            w1, w2 = self.weights[0], self.weights[1]
            total = w1 + w2
            ensemble_pred = (w1/total) * pred_simp + (w2/total) * pred_rank

        return ensemble_pred


def train_ensemble_vol_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    verbose: bool = False
) -> EnsembleVolPredictor:
    """Train ensemble of models for volatility prediction."""

    if verbose:
        print("  Training SimplifiedQCML...")
    simplified_model = train_vol_model(
        X_train, y_train, X_val, y_val,
        model_type='simplified', epochs=100, verbose=False
    )

    if verbose:
        print("  Training RankingQCML...")
    ranking_model = train_ranking_vol_model(
        X_train, y_train, X_val, y_val,
        epochs=100, verbose=False
    )

    xgb_model = None
    if HAS_XGBOOST:
        if verbose:
            print("  Training XGBoost...")
        xgb_model = train_xgboost_vol_model(X_train, y_train, X_val, y_val)

    # Create ensemble with weights: 40% QCML, 30% Ranking, 30% XGBoost
    ensemble = EnsembleVolPredictor(
        simplified_model, ranking_model, xgb_model,
        weights=(0.4, 0.3, 0.3)
    )

    return ensemble


def evaluate_ensemble_predictions(
    ensemble: EnsembleVolPredictor,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:
    """Evaluate ensemble volatility predictions."""
    pred = ensemble.predict(X)

    # Correlation
    corr = np.corrcoef(pred, y)[0, 1]

    # Rank correlation
    rank_corr = stats.spearmanr(pred, y)[0]

    # RMSE
    rmse = np.sqrt(np.mean((pred - y) ** 2))

    # Direction accuracy
    mean_vol = y.mean()
    direction_acc = np.mean((pred > mean_vol) == (y > mean_vol))

    return {
        'correlation': corr if not np.isnan(corr) else 0,
        'rank_correlation': rank_corr if not np.isnan(rank_corr) else 0,
        'rmse': rmse,
        'direction_accuracy': direction_acc,
        'predictions': pred
    }


def evaluate_vol_predictions(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:
    """Evaluate volatility predictions."""
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        pred_log = model(X_t).cpu().numpy()
        pred = np.exp(pred_log)  # Convert back from log

    # Correlation
    corr = np.corrcoef(pred, y)[0, 1]

    # Rank correlation
    rank_corr = stats.spearmanr(pred, y)[0]

    # RMSE
    rmse = np.sqrt(np.mean((pred - y) ** 2))

    # Direction accuracy (did vol go up or down)
    # Compare to previous vol (approximated as mean)
    mean_vol = y.mean()
    direction_acc = np.mean((pred > mean_vol) == (y > mean_vol))

    return {
        'correlation': corr if not np.isnan(corr) else 0,
        'rank_correlation': rank_corr if not np.isnan(rank_corr) else 0,
        'rmse': rmse,
        'direction_accuracy': direction_acc,
        'predictions': pred
    }


def run_momentum_backtest(
    prices: pd.DataFrame,
    vol_predictions: Optional[pd.DataFrame] = None,
    top_k: int = 5,
    target_vol: float = 0.15,
    min_weight: float = 0.25,
    max_weight: float = 3.0,
    high_vol_threshold: float = 0.25,
    high_vol_exposure: float = 0.5
) -> pd.DataFrame:
    """
    Run momentum backtest with vol-adjusted sizing and risk management.

    Strategies:
    1. Equal-weight: Simple equal allocation to top-k momentum ETFs
    2. Vol-adjusted: Inverse-vol weighting with wide bounds
    3. Risk-managed: Vol-adjusted + reduce exposure when avg vol is high

    Parameters
    ----------
    prices : DataFrame with ETF prices
    vol_predictions : DataFrame with vol predictions (optional)
    top_k : int, number of ETFs to hold
    target_vol : float, target portfolio volatility
    min_weight : float, minimum position weight multiplier (wider = more differentiation)
    max_weight : float, maximum position weight multiplier
    high_vol_threshold : float, when avg predicted vol exceeds this, reduce exposure
    high_vol_exposure : float, exposure multiplier when vol is high (0.5 = 50% cash)

    Returns
    -------
    results : DataFrame with strategy returns
    """
    returns = prices.pct_change()

    # Momentum signal: 20-day return
    momentum = returns.rolling(20).sum()

    # Weekly rebalance
    rebalance_dates = prices.resample('W-FRI').last().index

    strategy_returns = []
    vol_adjusted_returns = []
    risk_managed_returns = []

    for i, date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]

        # Find nearest trading day
        valid_dates = momentum.index
        if date not in valid_dates:
            prior_dates = valid_dates[valid_dates <= date]
            if len(prior_dates) == 0:
                continue
            date = prior_dates[-1]

        # Get momentum rankings
        mom = momentum.loc[date].dropna()
        if len(mom) < top_k:
            continue

        # Select top k
        top_etfs = mom.nlargest(top_k).index.tolist()

        # Equal weight
        equal_weight = 1.0 / top_k

        # Get returns for holding period
        period_mask = (returns.index > date) & (returns.index <= next_date)
        period_returns = returns.loc[period_mask, top_etfs]

        if len(period_returns) == 0:
            continue

        # Equal-weight returns
        eq_ret = (period_returns * equal_weight).sum(axis=1)
        strategy_returns.extend(eq_ret.values)

        # Vol-adjusted and risk-managed returns
        if vol_predictions is not None and date in vol_predictions.index.get_level_values(0):
            vol_preds = vol_predictions.loc[date]

            # Inverse vol weighting with WIDER bounds
            inv_vol_weights = {}
            vol_values = []
            for etf in top_etfs:
                if etf in vol_preds.index:
                    pred_vol = vol_preds.loc[etf]
                    vol_values.append(pred_vol)
                    # Position size = target_vol / predicted_vol
                    weight = target_vol / (pred_vol + 0.01)
                    weight = np.clip(weight, min_weight, max_weight)  # Wider bounds!
                else:
                    weight = 1.0
                inv_vol_weights[etf] = weight

            # Normalize weights
            total_weight = sum(inv_vol_weights.values())
            inv_vol_weights = {k: v / total_weight for k, v in inv_vol_weights.items()}

            # Vol-adjusted returns (no risk overlay)
            vol_adj_ret = sum(
                period_returns[etf] * inv_vol_weights.get(etf, equal_weight)
                for etf in top_etfs
            )
            vol_adjusted_returns.extend(vol_adj_ret.values)

            # Risk-managed: continuous exposure scaling based on vol level
            # Higher predicted vol → lower exposure (inverse relationship)
            avg_vol = np.mean(vol_values) if vol_values else 0.15
            # Smooth scaling: exposure = target_vol / predicted_vol, bounded [0.3, 1.0]
            exposure = np.clip(target_vol / avg_vol, 0.3, 1.0)

            risk_ret = vol_adj_ret * exposure
            risk_managed_returns.extend(risk_ret.values)
        else:
            vol_adjusted_returns.extend(eq_ret.values)
            risk_managed_returns.extend(eq_ret.values)

    results = pd.DataFrame({
        'equal_weight': strategy_returns,
        'vol_adjusted': vol_adjusted_returns,
        'risk_managed': risk_managed_returns
    })

    return results


def compute_strategy_metrics(returns: pd.Series) -> Dict[str, float]:
    """Compute strategy performance metrics."""
    if len(returns) == 0:
        return {'sharpe': 0, 'annual_return': 0, 'annual_vol': 0, 'max_drawdown': 0}

    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    # Max drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    return {
        'sharpe': sharpe,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'max_drawdown': max_drawdown
    }


def run_practical_backtest(
    prices: pd.DataFrame,
    vol_predictions: Optional[pd.DataFrame] = None,
    top_k: int = 5,
    target_vol: float = 0.15,
    min_exposure: float = 0.4,
    max_exposure: float = 1.2,
    use_trend_filter: bool = True
) -> pd.DataFrame:
    """
    Practical momentum backtest with vol-scaling and trend filter.

    Key differences from risk_managed:
    1. Asymmetric exposure [0.4, 1.2] - allows slight leverage in low vol
    2. Trend filter - go defensive when SPY < 200-day MA
    3. Simpler logic - focused on practical Sharpe improvement

    Parameters
    ----------
    prices : DataFrame with ETF prices
    vol_predictions : DataFrame with vol predictions
    top_k : int, number of ETFs to hold
    target_vol : float, target portfolio volatility
    min_exposure : float, minimum exposure in high vol
    max_exposure : float, maximum exposure in low vol (can be >1)
    use_trend_filter : bool, use SPY 200-day MA filter
    """
    returns = prices.pct_change()
    momentum = returns.rolling(20).sum()

    # Trend filter: SPY 200-day MA
    if 'SPY' in prices.columns and use_trend_filter:
        spy_ma_200 = prices['SPY'].rolling(200).mean()
        trend_bullish = prices['SPY'] > spy_ma_200
    else:
        trend_bullish = pd.Series(True, index=prices.index)

    rebalance_dates = prices.resample('W-FRI').last().index

    baseline_returns = []  # Equal weight
    practical_returns = []  # Vol-scaled + trend filter

    for i, date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]

        # Find nearest trading day
        valid_dates = momentum.index
        if date not in valid_dates:
            prior_dates = valid_dates[valid_dates <= date]
            if len(prior_dates) == 0:
                continue
            date = prior_dates[-1]

        # Get momentum rankings
        mom = momentum.loc[date].dropna()
        if len(mom) < top_k:
            continue

        top_etfs = mom.nlargest(top_k).index.tolist()
        equal_weight = 1.0 / top_k

        # Get returns for holding period
        period_mask = (returns.index > date) & (returns.index <= next_date)
        period_returns = returns.loc[period_mask, top_etfs]

        if len(period_returns) == 0:
            continue

        # Baseline: equal weight
        eq_ret = (period_returns * equal_weight).sum(axis=1)
        baseline_returns.extend(eq_ret.values)

        # Practical strategy
        if vol_predictions is not None and date in vol_predictions.index.get_level_values(0):
            vol_preds = vol_predictions.loc[date]

            # Compute average predicted vol for top ETFs
            vol_values = []
            for etf in top_etfs:
                if etf in vol_preds.index:
                    vol_values.append(vol_preds.loc[etf])

            avg_vol = np.mean(vol_values) if vol_values else 0.15

            # Asymmetric exposure scaling
            # High vol (30%) -> exposure = 0.15/0.30 = 0.5
            # Normal vol (15%) -> exposure = 0.15/0.15 = 1.0
            # Low vol (10%) -> exposure = 0.15/0.10 = 1.5 -> clips to 1.2
            raw_exposure = target_vol / avg_vol
            exposure = np.clip(raw_exposure, min_exposure, max_exposure)

            # Trend filter: reduce exposure further if trend is bearish
            if use_trend_filter and date in trend_bullish.index:
                if not trend_bullish.loc[date]:
                    exposure *= 0.5  # Half exposure in downtrend

            practical_ret = eq_ret * exposure
        else:
            practical_ret = eq_ret

        practical_returns.extend(practical_ret.values)

    return pd.DataFrame({
        'baseline': baseline_returns,
        'practical': practical_returns
    })


def run_walk_forward_vol_prediction(
    data: pd.DataFrame,
    sector_prices: pd.DataFrame,
    all_prices: pd.DataFrame,
    feature_cols: List[str],
    n_folds: int = 5,
    use_ensemble: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Walk-forward validation for vol prediction + momentum backtest.

    Parameters
    ----------
    data : DataFrame with features and labels
    sector_prices : DataFrame with sector ETF prices
    all_prices : DataFrame with all prices (including SPY for trend filter)
    feature_cols : list of feature column names
    n_folds : int, number of walk-forward folds
    use_ensemble : bool, if True use ensemble, else use single SimplifiedQCML
    verbose : bool, print progress
    """
    dates = data.index.get_level_values(0).unique().sort_values()
    n_dates = len(dates)
    fold_size = n_dates // (n_folds + 1)

    all_predictions = []
    vol_metrics = []

    if verbose:
        print("\n" + "="*60)
        print("WALK-FORWARD VOLATILITY PREDICTION")
        print("="*60)
        if use_ensemble:
            if HAS_XGBOOST:
                print("Model: Ensemble (SimplifiedQCML 40% + RankingQCML 30% + XGBoost 30%)")
            else:
                print("Model: Ensemble (SimplifiedQCML 57% + RankingQCML 43%)")
        else:
            print("Model: SimplifiedQCML (single model for best Sharpe)")

    for fold in range(n_folds):
        train_end_idx = fold_size + fold * fold_size
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + fold_size, n_dates)

        train_dates = dates[:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]

        if verbose:
            print(f"\nFold {fold + 1}/{n_folds}")
            print(f"  Train: {train_dates[0].date()} to {train_dates[-1].date()}")
            print(f"  Test:  {test_dates[0].date()} to {test_dates[-1].date()}")

        # Split data
        train_mask = data.index.get_level_values(0).isin(train_dates)
        test_mask = data.index.get_level_values(0).isin(test_dates)

        train_data = data[train_mask]
        test_data = data[test_mask]

        X_train = train_data[feature_cols].values.astype(np.float32)
        y_train = train_data['target_vol'].values.astype(np.float32)

        X_test = test_data[feature_cols].values.astype(np.float32)
        y_test = test_data['target_vol'].values.astype(np.float32)

        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=y_train[~np.isnan(y_train)].mean() if any(~np.isnan(y_train)) else 0.2)

        # Split train into train/val
        val_size = len(X_train) // 5
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]

        # Train model(s)
        if use_ensemble:
            if verbose:
                print("  Training ensemble...")
            ensemble = train_ensemble_vol_models(X_train, y_train, X_val, y_val, verbose=verbose)
            metrics = evaluate_ensemble_predictions(ensemble, X_test, y_test)
        else:
            if verbose:
                print("  Training SimplifiedQCML...")
            model = train_vol_model(X_train, y_train, X_val, y_val, model_type='simplified', verbose=False)
            metrics = evaluate_vol_predictions(model, X_test, y_test)

        vol_metrics.append(metrics)

        if verbose:
            print(f"  Vol Correlation: {metrics['correlation']:.4f}")
            print(f"  Vol Rank Corr:   {metrics['rank_correlation']:.4f}")

        # Store predictions with index
        preds_df = pd.DataFrame({
            'predicted_vol': metrics['predictions'],
            'actual_vol': y_test
        }, index=test_data.index)
        all_predictions.append(preds_df)

    # Combine predictions
    all_preds = pd.concat(all_predictions)

    # Aggregate vol metrics
    avg_corr = np.mean([m['correlation'] for m in vol_metrics])
    avg_rank = np.mean([m['rank_correlation'] for m in vol_metrics])

    if verbose:
        print(f"\n--- Vol Prediction Summary ---")
        print(f"Avg Correlation: {avg_corr:.4f}")
        print(f"Avg Rank Corr:   {avg_rank:.4f}")

    # Run backtest comparison
    if verbose:
        print("\n" + "="*60)
        print("MOMENTUM BACKTEST COMPARISON")
        print("="*60)

    vol_preds_unstacked = all_preds['predicted_vol'].unstack(level=1)

    # Standard strategies
    results_with_vol = run_momentum_backtest(
        sector_prices,
        vol_predictions=vol_preds_unstacked,
        top_k=5,
        target_vol=0.15,
        min_weight=0.25,
        max_weight=3.0,
        high_vol_threshold=0.25,
        high_vol_exposure=0.5
    )

    # Practical strategy with trend filter
    practical_results = run_practical_backtest(
        all_prices,  # Includes SPY for trend filter
        vol_predictions=vol_preds_unstacked,
        top_k=5,
        target_vol=0.15,
        min_exposure=0.4,   # Less conservative
        max_exposure=1.2,   # Allow slight leverage in low vol
        use_trend_filter=True
    )

    # Compute metrics for all strategies
    eq_weight_metrics = compute_strategy_metrics(results_with_vol['equal_weight'])
    vol_adj_metrics = compute_strategy_metrics(results_with_vol['vol_adjusted'])
    risk_mgd_metrics = compute_strategy_metrics(results_with_vol['risk_managed'])
    practical_metrics = compute_strategy_metrics(practical_results['practical'])

    if verbose:
        print(f"\n1. Equal-Weight Momentum (baseline):")
        print(f"   Sharpe:     {eq_weight_metrics['sharpe']:.2f}")
        print(f"   Annual Ret: {eq_weight_metrics['annual_return']:.1%}")
        print(f"   Annual Vol: {eq_weight_metrics['annual_vol']:.1%}")
        print(f"   Max DD:     {eq_weight_metrics['max_drawdown']:.1%}")

        print(f"\n2. Risk-Managed (conservative scaling [0.3-1.0]):")
        print(f"   Sharpe:     {risk_mgd_metrics['sharpe']:.2f}")
        print(f"   Annual Ret: {risk_mgd_metrics['annual_return']:.1%}")
        print(f"   Annual Vol: {risk_mgd_metrics['annual_vol']:.1%}")
        print(f"   Max DD:     {risk_mgd_metrics['max_drawdown']:.1%}")

        print(f"\n3. PRACTICAL (asymmetric [0.4-1.2] + SPY trend filter):")
        print(f"   Sharpe:     {practical_metrics['sharpe']:.2f}")
        print(f"   Annual Ret: {practical_metrics['annual_return']:.1%}")
        print(f"   Annual Vol: {practical_metrics['annual_vol']:.1%}")
        print(f"   Max DD:     {practical_metrics['max_drawdown']:.1%}")

        # Summary
        print(f"\n--- Strategy Comparison ---")
        risk_improvement = risk_mgd_metrics['sharpe'] - eq_weight_metrics['sharpe']
        practical_improvement = practical_metrics['sharpe'] - eq_weight_metrics['sharpe']
        practical_dd_improvement = practical_metrics['max_drawdown'] - eq_weight_metrics['max_drawdown']

        print(f"Risk-Managed Sharpe delta:  {risk_improvement:+.2f}")
        print(f"PRACTICAL Sharpe delta:     {practical_improvement:+.2f}")
        print(f"PRACTICAL Max DD delta:     {practical_dd_improvement:+.1%}")

        # Identify best strategy
        best_sharpe = max(eq_weight_metrics['sharpe'], risk_mgd_metrics['sharpe'], practical_metrics['sharpe'])
        if practical_metrics['sharpe'] == best_sharpe:
            print("\n✓ PRACTICAL strategy has best Sharpe!")
        elif risk_mgd_metrics['sharpe'] == best_sharpe:
            print("\n✓ Risk-Managed strategy has best Sharpe!")

        if practical_dd_improvement > 0.05:
            print("✓ PRACTICAL strategy reduces drawdowns!")

    return {
        'vol_metrics': vol_metrics,
        'avg_correlation': avg_corr,
        'avg_rank_correlation': avg_rank,
        'equal_weight': eq_weight_metrics,
        'vol_adjusted': vol_adj_metrics,
        'risk_managed': risk_mgd_metrics,
        'practical': practical_metrics,
        'predictions': all_preds,
        'returns': results_with_vol,
        'practical_returns': practical_results
    }


def main():
    """Run volatility prediction experiment."""
    print("="*60)
    print("QCML VOLATILITY PREDICTION EXPERIMENT")
    print("Making QCML Actually Useful")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Prepare data
    data, sector_prices, all_prices, feature_cols = prepare_vol_data(
        start_date='2010-01-01',
        end_date='2024-01-01'
    )

    # Run experiment (use single model for better Sharpe)
    results = run_walk_forward_vol_prediction(
        data, sector_prices, all_prices, feature_cols,
        n_folds=5,
        use_ensemble=False,  # Single model = better Sharpe
        verbose=True
    )

    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    print(f"\nVolatility Prediction:")
    print(f"  Correlation: {results['avg_correlation']:.4f}")
    print(f"  (Target: > 0.3 for useful predictions)")

    print(f"\nStrategy Comparison:")
    print(f"  Equal-Weight Sharpe:   {results['equal_weight']['sharpe']:.2f}  (Max DD: {results['equal_weight']['max_drawdown']:.1%})")
    print(f"  Risk-Managed Sharpe:   {results['risk_managed']['sharpe']:.2f}  (Max DD: {results['risk_managed']['max_drawdown']:.1%})")
    print(f"  PRACTICAL Sharpe:      {results['practical']['sharpe']:.2f}  (Max DD: {results['practical']['max_drawdown']:.1%})")

    # Determine winner
    best_sharpe = max(
        results['equal_weight']['sharpe'],
        results['risk_managed']['sharpe'],
        results['practical']['sharpe']
    )
    best_dd = max(  # Less negative is better
        results['equal_weight']['max_drawdown'],
        results['risk_managed']['max_drawdown'],
        results['practical']['max_drawdown']
    )

    if results['avg_correlation'] > 0.3:
        print("\n✓ Vol prediction correlation meets target!")
    else:
        print(f"\n→ Vol prediction correlation below target (0.3)")

    if results['practical']['sharpe'] >= best_sharpe:
        print("✓ PRACTICAL strategy has best Sharpe!")
    elif results['risk_managed']['sharpe'] >= best_sharpe:
        print("✓ Risk-Managed strategy has best Sharpe!")

    if results['practical']['max_drawdown'] >= best_dd:
        print("✓ PRACTICAL strategy has lowest drawdown!")

    # Save results
    output_dir = project_root / 'results' / 'vol_prediction'
    output_dir.mkdir(parents=True, exist_ok=True)

    results['predictions'].to_csv(output_dir / 'vol_predictions.csv')
    results['returns'].to_csv(output_dir / 'strategy_returns.csv')

    print(f"\nResults saved to {output_dir}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
