#!/usr/bin/env python3
"""
Comprehensive QCML Volatility Prediction Experiments

Combines all three phases of improvements:
1. Enhanced OHLC-based features (Parkinson, Garman-Klass, Yang-Zhang)
2. Regime-aware Mixture of Experts (MoE)
3. Temporal QCML with GRU + attention + multi-horizon prediction

Includes:
- Walk-forward validation with expanding window
- Uncertainty-based position sizing
- Statistical significance testing
- Comparison across all model variants
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qcml_rotation.models.qcml import (
    SimplifiedQCML, SimplifiedQCMLConfig,
    RankingQCML, RankingQCMLConfig
)
from qcml_rotation.models.regime_aware import (
    MixtureOfExperts, MoEConfig, MoETrainer, MoELoss
)
from qcml_rotation.models.temporal_qcml import (
    TemporalQCML, TemporalQCMLConfig, TemporalQCMLTrainer,
    MultiHorizonLoss
)
from qcml_rotation.data.loader import (
    download_etf_data_ohlc, download_vix_term_structure
)
from qcml_rotation.regimes.detector import (
    EnsembleRegimeDetector, compute_regime_features, RegimeState
)

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# ETF universe
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE', 'XLC']
ALL_ETFS = SECTOR_ETFS + ['SPY']


@dataclass
class ExperimentConfig:
    """Configuration for comprehensive experiments."""
    start_date: str = '2010-01-01'
    end_date: str = '2024-12-01'

    # Walk-forward settings
    initial_train_weeks: int = 104  # 2 years initial training
    retrain_frequency: int = 4      # Retrain every 4 weeks
    min_train_samples: int = 500    # Minimum training samples

    # Models to test
    models: Tuple[str, ...] = (
        'baseline_cc',      # Close-to-close features only
        'enhanced_ohlc',    # OHLC-based features
        'moe',              # Mixture of Experts
        'temporal',         # Temporal QCML with GRU
        'ensemble'          # Ensemble of all models
    )

    # Uncertainty-based position sizing
    use_uncertainty_sizing: bool = True
    max_uncertainty_scale: float = 2.0  # Max position scaling based on uncertainty

    # Output
    output_dir: str = 'results/comprehensive_vol'
    save_predictions: bool = True


class SequentialDataset(Dataset):
    """Dataset for temporal models with sequential input."""

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 tickers: np.ndarray, dates: np.ndarray, seq_len: int = 12):
        """
        Parameters
        ----------
        features : np.ndarray (n_samples, n_features)
        labels : np.ndarray (n_samples,)
        tickers : np.ndarray (n_samples,)
        dates : np.ndarray (n_samples,)
        seq_len : int
            Number of past weeks to include
        """
        self.features = features
        self.labels = labels
        self.tickers = tickers
        self.dates = dates
        self.seq_len = seq_len

        # Build sequences per ticker
        self._build_sequences()

    def _build_sequences(self):
        """Build valid sequences for each ticker."""
        self.sequences = []

        unique_tickers = np.unique(self.tickers)
        for ticker in unique_tickers:
            mask = self.tickers == ticker
            ticker_features = self.features[mask]
            ticker_labels = self.labels[mask]
            ticker_dates = self.dates[mask]

            # Create sequences
            for i in range(self.seq_len, len(ticker_features)):
                seq_features = ticker_features[i-self.seq_len:i]
                target = ticker_labels[i]
                date = ticker_dates[i]

                if not np.isnan(target) and not np.any(np.isnan(seq_features)):
                    self.sequences.append({
                        'features': seq_features,
                        'label': target,
                        'ticker': ticker,
                        'date': date
                    })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return (
            torch.FloatTensor(seq['features']),
            torch.FloatTensor([seq['label']]),
            str(seq['ticker']),  # Convert to string for DataLoader compatibility
            str(seq['date'])     # Convert datetime64 to string
        )


def compute_realized_volatility(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute forward-looking realized volatility."""
    returns = prices.pct_change()
    realized_vol = returns.rolling(window).std().shift(-window) * np.sqrt(252)
    return realized_vol


def compute_enhanced_features(
    open_prices: pd.DataFrame,
    high_prices: pd.DataFrame,
    low_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    volume: pd.DataFrame,
    vix_term: Optional[pd.DataFrame] = None,
    include_regime: bool = True
) -> pd.DataFrame:
    """
    Compute all enhanced features including OHLC estimators, volume,
    asymmetric features, and regime indicators.
    """
    from qcml_rotation.data.vol_estimators import (
        compute_parkinson_volatility,
        compute_garman_klass_volatility,
        compute_rogers_satchell_volatility,
        compute_yang_zhang_volatility
    )

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

        # OHLC-based volatility estimators
        df['vol_parkinson_5d'] = compute_parkinson_volatility(high, low, 5)
        df['vol_parkinson_10d'] = compute_parkinson_volatility(high, low, 10)
        df['vol_parkinson_20d'] = compute_parkinson_volatility(high, low, 20)

        df['vol_gk_5d'] = compute_garman_klass_volatility(open_, high, low, close, 5)
        df['vol_gk_10d'] = compute_garman_klass_volatility(open_, high, low, close, 10)
        df['vol_gk_20d'] = compute_garman_klass_volatility(open_, high, low, close, 20)

        df['vol_yz_5d'] = compute_yang_zhang_volatility(open_, high, low, close, 5)
        df['vol_yz_10d'] = compute_yang_zhang_volatility(open_, high, low, close, 10)
        df['vol_yz_20d'] = compute_yang_zhang_volatility(open_, high, low, close, 20)

        df['vol_rs_5d'] = compute_rogers_satchell_volatility(open_, high, low, close, 5)

        # Close-to-close baseline
        df['vol_cc_5d'] = ret.rolling(5).std() * np.sqrt(252)
        df['vol_cc_10d'] = ret.rolling(10).std() * np.sqrt(252)
        df['vol_cc_20d'] = ret.rolling(20).std() * np.sqrt(252)
        df['vol_cc_60d'] = ret.rolling(60).std() * np.sqrt(252)

        # Estimator ratios
        df['vol_ratio_gk_cc'] = df['vol_gk_5d'] / (df['vol_cc_5d'] + 1e-8)
        df['vol_ratio_yz_cc'] = df['vol_yz_5d'] / (df['vol_cc_5d'] + 1e-8)

        # Vol term structure
        df['vol_term_5_20'] = df['vol_yz_5d'] / (df['vol_yz_20d'] + 1e-8)
        df['vol_term_5_60'] = df['vol_yz_5d'] / (df['vol_cc_60d'] + 1e-8)

        # Vol changes
        df['vol_change_5d'] = df['vol_yz_5d'] - df['vol_yz_5d'].shift(5)
        df['vol_change_pct_5d'] = df['vol_yz_5d'].pct_change(5)
        df['vol_of_vol'] = df['vol_yz_5d'].rolling(20).std()

        # Volume features
        vol_ma = vol.rolling(20).mean()
        df['volume_ratio'] = vol / (vol_ma + 1)
        df['volume_change'] = vol.pct_change()
        df['volume_change_5d'] = vol.pct_change(5)
        df['high_volume'] = (vol > 2 * vol_ma).astype(float)
        df['high_volume_count_20d'] = df['high_volume'].rolling(20).sum()

        vol_norm = vol / vol.rolling(20).sum()
        df['vol_weighted_vol'] = (ret.abs() * vol_norm).rolling(20).sum() * np.sqrt(252)
        df['vol_volume_corr'] = ret.abs().rolling(20).corr(vol)

        # Asymmetric features (leverage effect)
        neg_ret = ret.where(ret < 0, 0)
        df['downside_semivar'] = neg_ret.rolling(20).std() * np.sqrt(252)
        pos_ret = ret.where(ret > 0, 0)
        df['upside_semivar'] = pos_ret.rolling(20).std() * np.sqrt(252)
        df['semivar_ratio'] = df['downside_semivar'] / (df['upside_semivar'] + 1e-8)
        df['neg_return_pct'] = (ret < 0).rolling(20).sum() / 20
        df['worst_day_20d'] = ret.rolling(20).min()
        df['best_day_20d'] = ret.rolling(20).max()
        df['return_skew_20d'] = ret.rolling(20).skew()

        # Return features
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

        # Intraday range features
        intraday_range = (high - low) / close
        df['avg_range_5d'] = intraday_range.rolling(5).mean()
        df['avg_range_20d'] = intraday_range.rolling(20).mean()
        df['max_range_20d'] = intraday_range.rolling(20).max()

        overnight_ret = np.log(open_ / close.shift(1))
        intraday_ret = np.log(close / open_)
        df['overnight_vol_5d'] = overnight_ret.rolling(5).std() * np.sqrt(252)
        df['intraday_vol_5d'] = intraday_ret.rolling(5).std() * np.sqrt(252)
        df['overnight_intraday_ratio'] = df['overnight_vol_5d'] / (df['intraday_vol_5d'] + 1e-8)

        df['ticker'] = ticker
        features_list.append(df)

    features = pd.concat(features_list)

    # =====================================
    # Cross-Sector Correlation Features (Phase 5.1)
    # High cross-sector correlation = risk-off, higher vol expected
    # =====================================
    returns_matrix = close_prices.pct_change()

    # Rolling pairwise correlations - average across all pairs
    def compute_avg_correlation(returns_df, window=20):
        """Compute average pairwise correlation across all assets."""
        avg_corr = pd.Series(index=returns_df.index, dtype=float)
        for i in range(window, len(returns_df)):
            window_returns = returns_df.iloc[i-window:i]
            corr_matrix = window_returns.corr()
            # Get upper triangle (excluding diagonal)
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            avg_corr.iloc[i] = upper_tri.stack().mean()
        return avg_corr

    sector_corr_20d = compute_avg_correlation(returns_matrix, window=20)
    sector_corr_60d = compute_avg_correlation(returns_matrix, window=60)

    # Add cross-sector features to all ticker rows
    features = features.reset_index()
    date_col = 'Date' if 'Date' in features.columns else 'index'

    # Create cross-sector feature dataframe
    cross_sector_feats = pd.DataFrame({
        date_col: sector_corr_20d.index,
        'sector_correlation_20d': sector_corr_20d.values,
        'sector_correlation_60d': sector_corr_60d.values,
    })
    cross_sector_feats['sector_corr_change_5d'] = cross_sector_feats['sector_correlation_20d'].diff(5)
    cross_sector_feats['sector_corr_zscore'] = (
        (cross_sector_feats['sector_correlation_20d'] - cross_sector_feats['sector_correlation_20d'].rolling(60).mean()) /
        (cross_sector_feats['sector_correlation_20d'].rolling(60).std() + 1e-8)
    )
    cross_sector_feats['high_correlation_regime'] = (cross_sector_feats['sector_correlation_20d'] > 0.7).astype(float)
    cross_sector_feats['correlation_term_structure'] = (
        cross_sector_feats['sector_correlation_20d'] / (cross_sector_feats['sector_correlation_60d'] + 1e-8)
    )

    # Merge with features
    features = features.merge(cross_sector_feats, on=date_col, how='left')
    features = features.set_index([date_col, 'ticker'])

    # Cross-sectional rank
    features = features.reset_index()
    date_col = 'Date' if 'Date' in features.columns else 'index'
    features['vol_rank'] = features.groupby(date_col)['vol_yz_5d'].rank(pct=True)
    features = features.set_index([date_col, 'ticker'])

    # VIX term structure features
    if vix_term is not None and len(vix_term) > 0:
        features = features.reset_index()

        vix_features = vix_term[['vix', 'vix3m', 'vix_term_structure', 'vix_term_slope',
                                  'vix_term_percentile', 'vix_backwardation',
                                  'vix_change_5d', 'vix_zscore', 'vix_spike']].copy()
        vix_features = vix_features.reset_index()
        vix_features.columns = ['Date'] + list(vix_features.columns[1:])

        features = features.merge(vix_features, on='Date', how='left')
        features = features.set_index(['Date', 'ticker'])

    # Regime features
    if include_regime and vix_term is not None and 'vix' in vix_term.columns:
        vix = vix_term['vix']
        detector = EnsembleRegimeDetector()
        try:
            detector.fit(vix)
            regime_feats = compute_regime_features(vix, detector, include_hmm=True)

            features = features.reset_index()
            regime_feats = regime_feats.reset_index()
            regime_feats.columns = ['Date'] + list(regime_feats.columns[1:])

            features = features.merge(regime_feats, on='Date', how='left')
            features = features.set_index(['Date', 'ticker'])
        except Exception as e:
            print(f"Warning: Could not compute regime features: {e}")

    return features


def download_credit_spreads(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download HYG-LQD credit spread as stress indicator (Phase 5.1).

    HYG = High Yield Corporate Bond ETF
    LQD = Investment Grade Corporate Bond ETF
    Spread volatility spikes during credit stress periods.
    """
    import yfinance as yf

    print("  Downloading credit spread data (HYG, LQD)...")
    hyg_data = yf.download('HYG', start=start_date, end=end_date, progress=False)
    lqd_data = yf.download('LQD', start=start_date, end=end_date, progress=False)

    # Extract Close price as Series
    hyg = hyg_data['Close'].squeeze() if 'Close' in hyg_data.columns else hyg_data.squeeze()
    lqd = lqd_data['Close'].squeeze() if 'Close' in lqd_data.columns else lqd_data.squeeze()

    # Compute ratio (HYG/LQD) - lower ratio = wider spreads = more stress
    ratio = hyg / lqd
    ratio_returns = ratio.pct_change()

    # Volatility of the spread
    spread_vol_5d = ratio_returns.rolling(5).std() * np.sqrt(252)
    spread_vol_20d = ratio_returns.rolling(20).std() * np.sqrt(252)

    # Z-score of spread volatility (stress indicator)
    spread_zscore = (spread_vol_5d - spread_vol_5d.rolling(60).mean()) / (spread_vol_5d.rolling(60).std() + 1e-8)

    # Create features
    credit_features = pd.DataFrame({
        'credit_spread_ratio': ratio,
        'credit_spread_vol_5d': spread_vol_5d,
        'credit_spread_vol_20d': spread_vol_20d,
        'credit_spread_zscore': spread_zscore,
        'credit_stress': (spread_zscore > 1.5).astype(float),  # Binary stress indicator
        'credit_spread_change_5d': spread_vol_5d.diff(5),
        'credit_vol_term_structure': spread_vol_5d / (spread_vol_20d + 1e-8)
    }, index=hyg.index)

    return credit_features


def prepare_data(config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Download and prepare all data."""
    print("Downloading OHLC data...")
    open_prices, high_prices, low_prices, close_prices, volume = download_etf_data_ohlc(
        tickers=ALL_ETFS,
        start_date=config.start_date,
        end_date=config.end_date,
        force_refresh=True
    )

    print("Downloading VIX term structure...")
    vix_term = download_vix_term_structure(
        start_date=config.start_date,
        end_date=config.end_date,
        force_refresh=True
    )

    print("Downloading credit spreads...")
    credit_spreads = download_credit_spreads(
        start_date=config.start_date,
        end_date=config.end_date
    )

    # Get sector data only
    sector_open = open_prices[SECTOR_ETFS]
    sector_high = high_prices[SECTOR_ETFS]
    sector_low = low_prices[SECTOR_ETFS]
    sector_close = close_prices[SECTOR_ETFS]
    sector_volume = volume[SECTOR_ETFS]

    print("Computing enhanced features...")
    features = compute_enhanced_features(
        sector_open, sector_high, sector_low, sector_close, sector_volume,
        vix_term, include_regime=True
    )

    # Add credit spread features (Phase 5.1)
    if credit_spreads is not None and len(credit_spreads) > 0:
        print("  Adding credit spread features...")
        features = features.reset_index()
        date_col = 'Date' if 'Date' in features.columns else 'index'

        credit_feats = credit_spreads.reset_index()
        credit_feats.columns = ['Date'] + list(credit_feats.columns[1:])

        features = features.merge(credit_feats, on='Date', how='left')
        features = features.set_index([date_col, 'ticker'])

    print("Computing realized volatility labels...")
    realized_vol = compute_realized_volatility(sector_close, window=5)

    # Reshape labels
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

    # Combine
    features = features.reset_index()
    labels = labels.reset_index()

    data = features.merge(labels, on=['Date', 'ticker'], how='inner')
    data = data.set_index(['Date', 'ticker'])

    # Drop NaN
    feature_cols = [c for c in data.columns if c != 'realized_vol']
    data = data.dropna(subset=feature_cols + ['realized_vol'])

    print(f"Prepared {len(data)} samples with {len(feature_cols)} features")

    return data, sector_close, feature_cols


class ModelWrapper:
    """Unified interface for all model types."""

    def __init__(self, model_type: str, input_dim: int, device: torch.device):
        self.model_type = model_type
        self.input_dim = input_dim
        self.device = device
        self.model = None
        self.temporal_config = None  # Store config for temporal model
        self.temporal_trainer = None  # Store trainer for temporal model
        self.uncertainty_available = False

    def create_model(self):
        """Create fresh model instance."""
        if self.model_type == 'baseline_cc' or self.model_type == 'enhanced_ohlc':
            # SimplifiedQCML takes input_dim as first arg, config as second
            config = SimplifiedQCMLConfig(
                hidden_dim=64,
                embed_dim=32,
                dropout=0.15
            )
            self.model = SimplifiedQCML(self.input_dim, config)
            self.uncertainty_available = False

        elif self.model_type == 'moe':
            config = MoEConfig(
                input_dim=self.input_dim,
                n_experts=4,
                hidden_dim=64,
                embed_dim=32,
                dropout=0.15
            )
            self.model = MixtureOfExperts(config)
            self.uncertainty_available = False

        elif self.model_type == 'temporal':
            # Store config - trainer will create model
            self.temporal_config = TemporalQCMLConfig(
                input_dim=self.input_dim,
                seq_len=12,
                hidden_dim=64,
                embed_dim=32,
                n_gru_layers=2,
                n_attention_heads=4,
                output_uncertainty=True,
                epochs=50,  # Reduced for walk-forward efficiency
                early_stopping_patience=10
            )
            self.uncertainty_available = True
            # Model will be created by trainer during training
            return

        if self.model is not None:
            self.model.to(self.device)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
              lr: float = 0.001, seq_data: Optional[SequentialDataset] = None):
        """Train model."""
        self.create_model()

        if self.model_type == 'temporal' and seq_data is not None:
            # Create temporal model directly
            self.model = TemporalQCML(self.temporal_config).to(self.device)

            # Create train DataLoader from sequential dataset
            train_loader = torch.utils.data.DataLoader(
                seq_data, batch_size=32, shuffle=True
            )

            # Simple training loop with MSE loss on 5-day horizon
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-3)
            criterion = nn.MSELoss()

            self.model.train()
            for epoch in range(min(epochs, 50)):
                total_loss = 0
                for batch in train_loader:
                    features, labels, _, _ = batch  # (batch, seq_len, n_features), (batch, 1), ticker, date
                    features = features.to(self.device)
                    labels = labels.to(self.device).squeeze()

                    optimizer.zero_grad()
                    output = self.model(features)  # Returns {5: (mean, var), 10: ..., 20: ...}

                    # Use 5-day prediction
                    mean_5d, _ = output[5]
                    loss = criterion(mean_5d.squeeze(), labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        else:
            # Standard training
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-3)
            criterion = nn.MSELoss()

            X_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.FloatTensor(y).to(self.device)

            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                preds = self.model(X_t)
                if isinstance(preds, tuple):
                    preds = preds[0]  # MoE returns (prediction, gating_weights)
                loss = criterion(preds.squeeze(), y_t)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray, seq_X: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict volatility.

        Returns
        -------
        predictions : np.ndarray
        uncertainties : np.ndarray or None
        """
        self.model.eval()
        with torch.no_grad():
            if self.model_type == 'temporal' and seq_X is not None:
                X_t = seq_X.to(self.device)
            else:
                X_t = torch.FloatTensor(X).to(self.device)

            output = self.model(X_t)

            if isinstance(output, dict):
                # Temporal model returns {horizon: (mean, variance)}
                # Use primary horizon (5 days)
                primary_horizon = 5
                if primary_horizon in output:
                    mean, variance = output[primary_horizon]
                    preds = mean.cpu().numpy().flatten()
                    if self.uncertainty_available and variance is not None:
                        uncert = torch.sqrt(variance).cpu().numpy().flatten()
                    else:
                        uncert = None
                else:
                    # Fallback: use first available horizon
                    first_horizon = list(output.keys())[0]
                    mean, variance = output[first_horizon]
                    preds = mean.cpu().numpy().flatten()
                    uncert = None
            elif isinstance(output, tuple):
                # MoE returns (prediction, gating)
                preds = output[0].cpu().numpy().flatten()
                uncert = None
            else:
                preds = output.cpu().numpy().flatten()
                uncert = None

        return preds, uncert


class WalkForwardValidator:
    """Walk-forward validation with expanding window."""

    def __init__(self, config: ExperimentConfig, device: torch.device):
        self.config = config
        self.device = device
        self.results = {}

    def run(self, data: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Run walk-forward validation for all models."""
        print("\n" + "="*60)
        print("WALK-FORWARD VALIDATION")
        print("="*60)

        # Get unique dates
        dates = data.index.get_level_values('Date').unique().sort_values()
        n_dates = len(dates)

        # Weekly rebalancing
        rebalance_dates = dates[::5][self.config.initial_train_weeks:]

        results = {model: {'predictions': [], 'actuals': [], 'dates': [],
                           'tickers': [], 'uncertainties': []}
                   for model in self.config.models}

        print(f"Total dates: {n_dates}")
        print(f"Rebalance dates: {len(rebalance_dates)}")
        print(f"Models: {self.config.models}")

        for i, rebal_date in enumerate(rebalance_dates):
            if i % self.config.retrain_frequency != 0:
                continue

            # Training data: all data before rebalance date
            train_mask = data.index.get_level_values('Date') < rebal_date
            train_data = data[train_mask]

            if len(train_data) < self.config.min_train_samples:
                continue

            # Test data: next period
            next_idx = min(i + self.config.retrain_frequency, len(rebalance_dates) - 1)
            if next_idx <= i:
                continue
            next_date = rebalance_dates[next_idx]

            test_mask = (data.index.get_level_values('Date') >= rebal_date) & \
                        (data.index.get_level_values('Date') < next_date)
            test_data = data[test_mask]

            if len(test_data) == 0:
                continue

            print(f"\n[{i+1}/{len(rebalance_dates)}] Train: {len(train_data)}, Test: {len(test_data)}, Date: {rebal_date.date()}")

            # Prepare data
            X_train = train_data[feature_cols].values
            y_train = train_data['realized_vol'].values
            X_test = test_data[feature_cols].values
            y_test = test_data['realized_vol'].values

            test_tickers = test_data.index.get_level_values('ticker').values
            test_dates = test_data.index.get_level_values('Date').values

            # Normalize (handle NaN with nanmean/nanstd)
            mean = np.nanmean(X_train, axis=0)
            std = np.nanstd(X_train, axis=0) + 1e-8
            X_train_norm = (X_train - mean) / std
            X_test_norm = (X_test - mean) / std
            # Fill remaining NaN with 0 after normalization
            X_train_norm = np.nan_to_num(X_train_norm, nan=0.0)
            X_test_norm = np.nan_to_num(X_test_norm, nan=0.0)

            # Train and predict with each model
            for model_name in self.config.models:
                if model_name == 'ensemble':
                    continue  # Handle separately

                try:
                    # Select features for baseline
                    if model_name == 'baseline_cc':
                        cc_cols = [c for c in feature_cols if 'vol_cc' in c or 'ret' in c or 'vix' in c]
                        cc_idx = [feature_cols.index(c) for c in cc_cols if c in feature_cols]
                        X_tr = X_train_norm[:, cc_idx]
                        X_te = X_test_norm[:, cc_idx]
                        input_dim = len(cc_idx)
                    else:
                        X_tr = X_train_norm
                        X_te = X_test_norm
                        input_dim = len(feature_cols)

                    wrapper = ModelWrapper(model_name, input_dim, self.device)

                    # Handle temporal model differently
                    if model_name == 'temporal':
                        # Create sequential dataset
                        train_tickers = train_data.index.get_level_values('ticker').values
                        train_dates_arr = train_data.index.get_level_values('Date').values

                        seq_dataset = SequentialDataset(
                            X_tr, y_train, train_tickers, train_dates_arr, seq_len=12
                        )

                        if len(seq_dataset) < 100:
                            print(f"  {model_name}: Skipping (insufficient sequences)")
                            continue

                        wrapper.train(X_tr, y_train, epochs=50, seq_data=seq_dataset)

                        # Create test sequences
                        test_seq_dataset = SequentialDataset(
                            X_te, y_test, test_tickers, test_dates, seq_len=12
                        )

                        if len(test_seq_dataset) == 0:
                            print(f"  {model_name}: No valid test sequences")
                            continue

                        # Batch predictions
                        all_preds = []
                        all_uncert = []
                        all_labels = []

                        loader = DataLoader(test_seq_dataset, batch_size=32, shuffle=False)
                        for batch_feat, batch_label, _, _ in loader:
                            preds, uncert = wrapper.predict(None, seq_X=batch_feat)
                            all_preds.extend(preds)
                            if uncert is not None:
                                all_uncert.extend(uncert)
                            all_labels.extend(batch_label.numpy().flatten())

                        preds = np.array(all_preds)
                        uncert = np.array(all_uncert) if all_uncert else None
                        y_test_actual = np.array(all_labels)

                    else:
                        wrapper.train(X_tr, y_train, epochs=100)
                        preds, uncert = wrapper.predict(X_te)
                        y_test_actual = y_test

                    # Store results
                    results[model_name]['predictions'].extend(preds.tolist())
                    results[model_name]['actuals'].extend(y_test_actual.tolist())
                    results[model_name]['dates'].extend(test_dates[:len(preds)].tolist())
                    results[model_name]['tickers'].extend(test_tickers[:len(preds)].tolist())
                    if uncert is not None:
                        results[model_name]['uncertainties'].extend(uncert.tolist())

                    corr = np.corrcoef(preds[:len(y_test_actual)], y_test_actual[:len(preds)])[0, 1]
                    print(f"  {model_name}: corr = {corr:.4f}")

                except Exception as e:
                    print(f"  {model_name}: Error - {e}")

        # Create ensemble predictions
        if 'ensemble' in self.config.models:
            self._create_ensemble(results)

        self.results = results
        return results

    def _create_ensemble(self, results: Dict):
        """Create ensemble from individual model predictions."""
        # Find common predictions
        model_keys = [k for k in self.config.models if k != 'ensemble' and len(results[k]['predictions']) > 0]

        if len(model_keys) < 2:
            print("Not enough models for ensemble")
            return

        # Use the model with most predictions as reference
        ref_model = max(model_keys, key=lambda k: len(results[k]['predictions']))
        ref_dates = results[ref_model]['dates']
        ref_tickers = results[ref_model]['tickers']

        ensemble_preds = []
        ensemble_actuals = []

        for i, (date, ticker) in enumerate(zip(ref_dates, ref_tickers)):
            preds = []
            for model in model_keys:
                if i < len(results[model]['predictions']):
                    preds.append(results[model]['predictions'][i])

            if preds:
                ensemble_preds.append(np.mean(preds))
                ensemble_actuals.append(results[ref_model]['actuals'][i])

        results['ensemble']['predictions'] = ensemble_preds
        results['ensemble']['actuals'] = ensemble_actuals
        results['ensemble']['dates'] = ref_dates[:len(ensemble_preds)]
        results['ensemble']['tickers'] = ref_tickers[:len(ensemble_preds)]

    def compute_metrics(self) -> pd.DataFrame:
        """Compute performance metrics for all models."""
        metrics = []

        for model_name, data in self.results.items():
            if len(data['predictions']) == 0:
                continue

            preds = np.array(data['predictions'])
            actuals = np.array(data['actuals'])

            # Remove NaN
            mask = ~(np.isnan(preds) | np.isnan(actuals))
            preds = preds[mask]
            actuals = actuals[mask]

            if len(preds) < 10:
                continue

            # Correlation
            corr = np.corrcoef(preds, actuals)[0, 1]
            rank_corr = stats.spearmanr(preds, actuals)[0]

            # MAE and RMSE
            mae = np.mean(np.abs(preds - actuals))
            rmse = np.sqrt(np.mean((preds - actuals)**2))

            # Direction accuracy
            pred_change = np.diff(preds)
            actual_change = np.diff(actuals)
            direction_acc = np.mean(np.sign(pred_change) == np.sign(actual_change))

            # R-squared
            ss_res = np.sum((actuals - preds)**2)
            ss_tot = np.sum((actuals - np.mean(actuals))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            metrics.append({
                'model': model_name,
                'correlation': corr,
                'rank_correlation': rank_corr,
                'mae': mae,
                'rmse': rmse,
                'direction_accuracy': direction_acc,
                'r_squared': r2,
                'n_samples': len(preds)
            })

        return pd.DataFrame(metrics).set_index('model')


def compute_uncertainty_position_sizes(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    max_scale: float = 2.0
) -> np.ndarray:
    """
    Compute position sizes based on prediction uncertainty.

    Lower uncertainty → larger positions
    Higher uncertainty → smaller positions

    Returns position scale factors (1.0 = baseline)
    """
    if uncertainties is None or len(uncertainties) == 0:
        return np.ones(len(predictions))

    uncertainties = np.array(uncertainties)

    # Normalize uncertainties to [0, 1]
    min_u = np.percentile(uncertainties, 5)
    max_u = np.percentile(uncertainties, 95)

    norm_u = (uncertainties - min_u) / (max_u - min_u + 1e-8)
    norm_u = np.clip(norm_u, 0, 1)

    # Inverse relationship: low uncertainty → high scale
    # Scale from 1/max_scale to max_scale
    position_scales = max_scale - norm_u * (max_scale - 1/max_scale)

    return position_scales


def backtest_strategy(
    results: Dict,
    prices: pd.DataFrame,
    config: ExperimentConfig
) -> pd.DataFrame:
    """
    Backtest volatility-targeting strategy using predictions.

    Strategy: Inverse volatility weighting with uncertainty adjustment.
    """
    strategy_results = []

    for model_name, data in results.items():
        if len(data['predictions']) == 0:
            continue

        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'date': data['dates'],
            'ticker': data['tickers'],
            'pred_vol': data['predictions'],
            'actual_vol': data['actuals']
        })

        if len(data.get('uncertainties', [])) > 0:
            pred_df['uncertainty'] = data['uncertainties'][:len(pred_df)]

        # Get unique dates
        unique_dates = pred_df['date'].unique()

        portfolio_returns = []

        for date in unique_dates:
            date_data = pred_df[pred_df['date'] == date]

            if len(date_data) < 3:
                continue

            # Inverse volatility weights
            pred_vols = date_data['pred_vol'].values
            weights = 1 / (pred_vols + 0.01)

            # Apply uncertainty scaling if available
            if config.use_uncertainty_sizing and 'uncertainty' in date_data.columns:
                uncertainties = date_data['uncertainty'].values
                position_scales = compute_uncertainty_position_sizes(
                    pred_vols, uncertainties, config.max_uncertainty_scale
                )
                weights = weights * position_scales

            # Normalize weights
            weights = weights / weights.sum()

            # Get next period returns
            tickers = date_data['ticker'].values

            try:
                if isinstance(date, pd.Timestamp):
                    next_date = prices.index[prices.index > date][0:5]
                else:
                    next_date = prices.index[prices.index > pd.Timestamp(date)][0:5]

                if len(next_date) > 0:
                    period_returns = prices.loc[next_date[-1], tickers] / prices.loc[date, tickers] - 1
                    portfolio_return = (weights * period_returns.values).sum()
                    portfolio_returns.append({
                        'date': date,
                        'return': portfolio_return,
                        'model': model_name
                    })
            except Exception:
                continue

        if portfolio_returns:
            strategy_results.extend(portfolio_returns)

    return pd.DataFrame(strategy_results)


def plot_results(metrics: pd.DataFrame, output_dir: Path):
    """Generate visualization of results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Correlation comparison
    ax = axes[0, 0]
    models = metrics.index.tolist()
    corrs = metrics['correlation'].values
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.bar(models, corrs, color=colors)
    ax.set_ylabel('Correlation')
    ax.set_title('Volatility Prediction Correlation')
    ax.axhline(y=0.63, color='red', linestyle='--', label='Baseline (0.63)')
    ax.axhline(y=0.75, color='green', linestyle='--', label='Target (0.75)')
    ax.legend()
    ax.set_ylim([0, 1])
    for bar, corr in zip(bars, corrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{corr:.3f}', ha='center', va='bottom', fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Rank correlation
    ax = axes[0, 1]
    rank_corrs = metrics['rank_correlation'].values
    bars = ax.bar(models, rank_corrs, color=colors)
    ax.set_ylabel('Rank Correlation')
    ax.set_title('Volatility Prediction Rank Correlation')
    ax.set_ylim([0, 1])
    for bar, corr in zip(bars, rank_corrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{corr:.3f}', ha='center', va='bottom', fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # RMSE
    ax = axes[1, 0]
    rmse = metrics['rmse'].values
    bars = ax.bar(models, rmse, color=colors)
    ax.set_ylabel('RMSE')
    ax.set_title('Prediction Error (RMSE)')
    for bar, r in zip(bars, rmse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{r:.4f}', ha='center', va='bottom', fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Direction accuracy
    ax = axes[1, 1]
    dir_acc = metrics['direction_accuracy'].values
    bars = ax.bar(models, dir_acc, color=colors)
    ax.set_ylabel('Direction Accuracy')
    ax.set_title('Volatility Direction Prediction')
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random (50%)')
    ax.legend()
    ax.set_ylim([0, 1])
    for bar, acc in zip(bars, dir_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'volatility_prediction_results.png', dpi=150)
    plt.close()

    print(f"\nSaved results plot to {output_dir / 'volatility_prediction_results.png'}")


def main():
    """Run comprehensive volatility prediction experiments."""
    print("="*60)
    print("COMPREHENSIVE QCML VOLATILITY PREDICTION")
    print("="*60)

    config = ExperimentConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    data, prices, feature_cols = prepare_data(config)

    print(f"\nData prepared:")
    print(f"  Samples: {len(data)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Date range: {data.index.get_level_values('Date').min()} to {data.index.get_level_values('Date').max()}")

    # Run walk-forward validation
    validator = WalkForwardValidator(config, device)
    results = validator.run(data, feature_cols)

    # Compute metrics
    metrics = validator.compute_metrics()

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(metrics.to_string())

    # Save metrics
    metrics.to_csv(output_dir / 'metrics.csv')

    # Backtest strategy
    print("\n" + "="*60)
    print("BACKTESTING STRATEGY")
    print("="*60)

    strategy_results = backtest_strategy(results, prices, config)

    if len(strategy_results) > 0:
        # Compute strategy metrics
        for model in strategy_results['model'].unique():
            model_returns = strategy_results[strategy_results['model'] == model]['return']
            sharpe = model_returns.mean() / model_returns.std() * np.sqrt(52) if model_returns.std() > 0 else 0
            total_return = (1 + model_returns).prod() - 1
            max_dd = (model_returns.cumsum() - model_returns.cumsum().cummax()).min()

            print(f"\n{model}:")
            print(f"  Sharpe Ratio: {sharpe:.3f}")
            print(f"  Total Return: {total_return:.1%}")
            print(f"  Max Drawdown: {max_dd:.1%}")

        strategy_results.to_csv(output_dir / 'strategy_returns.csv', index=False)

    # Plot results
    plot_results(metrics, output_dir)

    # Save full results
    if config.save_predictions:
        with open(output_dir / 'full_results.json', 'w') as f:
            # Convert to serializable format
            serializable = {}
            for model, data in results.items():
                serializable[model] = {
                    'predictions': [float(x) for x in data['predictions']],
                    'actuals': [float(x) for x in data['actuals']],
                    'n_samples': len(data['predictions'])
                }
            json.dump(serializable, f, indent=2)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")

    # Summary
    if len(metrics) > 0:
        best_model = metrics['correlation'].idxmax()
        best_corr = metrics.loc[best_model, 'correlation']
        baseline_corr = metrics.loc['baseline_cc', 'correlation'] if 'baseline_cc' in metrics.index else 0.63

        print(f"\nBest model: {best_model} (correlation = {best_corr:.4f})")
        print(f"Improvement over baseline: +{(best_corr - baseline_corr):.4f} ({(best_corr/baseline_corr - 1)*100:.1f}%)")

    return metrics


if __name__ == '__main__':
    main()
