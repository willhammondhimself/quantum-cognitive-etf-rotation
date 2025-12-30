"""
Regime detection for volatility prediction.

Implements HMM and rule-based approaches to detect market volatility regimes.
"""

import numpy as np
import pandas as pd
from enum import IntEnum
from typing import Tuple, Optional, Union
from dataclasses import dataclass


class RegimeState(IntEnum):
    """Market volatility regime states."""
    LOW_VOL = 0      # VIX < 15, calm markets
    NORMAL_VOL = 1   # 15 <= VIX < 22, typical conditions
    HIGH_VOL = 2     # 22 <= VIX < 30, elevated uncertainty
    CRISIS = 3       # VIX >= 30 or rapid vol spike


@dataclass
class RegimeThresholds:
    """Thresholds for rule-based regime classification."""
    low_vol_upper: float = 15.0
    normal_vol_upper: float = 22.0
    high_vol_upper: float = 30.0
    spike_threshold: float = 5.0  # VIX daily change for crisis detection


class RuleBasedRegimeDetector:
    """
    Simple rule-based regime detector using VIX thresholds.

    Fast and interpretable, but doesn't capture regime transitions well.
    """

    def __init__(self, thresholds: Optional[RegimeThresholds] = None):
        self.thresholds = thresholds or RegimeThresholds()

    def detect(self, vix: pd.Series) -> pd.Series:
        """
        Detect regime based on VIX level.

        Parameters
        ----------
        vix : pd.Series
            VIX index values.

        Returns
        -------
        regime : pd.Series
            Regime state for each date.
        """
        regime = pd.Series(index=vix.index, dtype=int)

        # VIX spike detection (crisis trigger)
        vix_change = vix.diff().abs()
        spike_mask = vix_change > self.thresholds.spike_threshold

        # Level-based classification
        regime[:] = RegimeState.NORMAL_VOL  # Default

        regime[vix < self.thresholds.low_vol_upper] = RegimeState.LOW_VOL
        regime[(vix >= self.thresholds.normal_vol_upper) &
               (vix < self.thresholds.high_vol_upper)] = RegimeState.HIGH_VOL
        regime[vix >= self.thresholds.high_vol_upper] = RegimeState.CRISIS

        # Override with spike-based crisis detection
        regime[spike_mask] = RegimeState.CRISIS

        return regime

    def detect_with_probabilities(self, vix: pd.Series) -> pd.DataFrame:
        """
        Detect regime with soft probabilities.

        Uses sigmoid functions to create smooth transitions between regimes.

        Returns
        -------
        probs : pd.DataFrame
            Columns are regime states, values are probabilities.
        """
        probs = pd.DataFrame(index=vix.index)

        # Distance from thresholds (for soft classification)
        t = self.thresholds

        # Low vol probability (decreases as VIX increases)
        probs['prob_low_vol'] = 1 / (1 + np.exp((vix - t.low_vol_upper) / 2))

        # Crisis probability (increases sharply above threshold)
        probs['prob_crisis'] = 1 / (1 + np.exp(-(vix - t.high_vol_upper) / 3))

        # High vol probability (between normal and crisis)
        probs['prob_high_vol'] = (
            1 / (1 + np.exp(-(vix - t.normal_vol_upper) / 3)) -
            probs['prob_crisis']
        ).clip(lower=0)

        # Normal vol probability (remainder)
        probs['prob_normal_vol'] = 1 - probs['prob_low_vol'] - probs['prob_high_vol'] - probs['prob_crisis']
        probs['prob_normal_vol'] = probs['prob_normal_vol'].clip(lower=0)

        # Normalize
        total = probs.sum(axis=1)
        probs = probs.div(total, axis=0)

        return probs


class HMMRegimeDetector:
    """
    Hidden Markov Model based regime detector.

    Learns latent regime states from volatility data and provides
    regime probabilities and transition matrices.
    """

    def __init__(self, n_states: int = 4, n_iter: int = 100, random_state: int = 42):
        """
        Parameters
        ----------
        n_states : int
            Number of hidden states (default 4 for LOW/NORMAL/HIGH/CRISIS).
        n_iter : int
            Maximum iterations for EM algorithm.
        random_state : int
            Random seed for reproducibility.
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self._fitted = False

    def fit(self, vix: pd.Series) -> 'HMMRegimeDetector':
        """
        Fit HMM to VIX data.

        Parameters
        ----------
        vix : pd.Series
            VIX index values.

        Returns
        -------
        self : HMMRegimeDetector
            Fitted detector.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            print("Warning: hmmlearn not installed. Using rule-based fallback.")
            print("Install with: pip install hmmlearn")
            self._fitted = False
            return self

        # Prepare data
        X = vix.values.reshape(-1, 1)

        # Remove NaN
        valid_mask = ~np.isnan(X.ravel())
        X_valid = X[valid_mask]

        # Initialize HMM
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=self.random_state
        )

        # Fit
        self.model.fit(X_valid)
        self._fitted = True

        # Store mapping from HMM states to semantic states
        # Order states by their mean VIX level
        means = self.model.means_.ravel()
        self._state_order = np.argsort(means)

        return self

    def detect(self, vix: pd.Series) -> pd.Series:
        """
        Detect regime using fitted HMM.

        Parameters
        ----------
        vix : pd.Series
            VIX index values.

        Returns
        -------
        regime : pd.Series
            Regime state for each date.
        """
        if not self._fitted:
            # Fallback to rule-based
            return RuleBasedRegimeDetector().detect(vix)

        X = vix.values.reshape(-1, 1)
        valid_mask = ~np.isnan(X.ravel())

        # Predict states
        regime = pd.Series(index=vix.index, dtype=int)
        regime[:] = RegimeState.NORMAL_VOL  # Default for NaN

        if valid_mask.sum() > 0:
            hidden_states = self.model.predict(X[valid_mask])
            # Map to semantic states
            mapped_states = np.array([self._state_order.tolist().index(s)
                                       for s in hidden_states])
            regime.iloc[valid_mask] = mapped_states

        return regime

    def detect_with_probabilities(self, vix: pd.Series) -> pd.DataFrame:
        """
        Get regime probabilities from HMM.

        Returns
        -------
        probs : pd.DataFrame
            Columns are regime states, values are probabilities.
        """
        if not self._fitted:
            return RuleBasedRegimeDetector().detect_with_probabilities(vix)

        X = vix.values.reshape(-1, 1)
        valid_mask = ~np.isnan(X.ravel())

        probs = pd.DataFrame(index=vix.index)
        for i, state_name in enumerate(['prob_low_vol', 'prob_normal_vol',
                                         'prob_high_vol', 'prob_crisis']):
            probs[state_name] = 0.0

        if valid_mask.sum() > 0:
            # Get posterior probabilities
            posteriors = self.model.predict_proba(X[valid_mask])

            # Map to semantic states
            for i, semantic_state in enumerate(self._state_order):
                probs.iloc[valid_mask, i] = posteriors[:, semantic_state]

        return probs

    @property
    def transition_matrix(self) -> Optional[np.ndarray]:
        """Get regime transition matrix if fitted."""
        if not self._fitted:
            return None
        return self.model.transmat_


class EnsembleRegimeDetector:
    """
    Ensemble of rule-based and HMM regime detectors.

    Combines strengths of both approaches:
    - Rule-based: Interpretable, handles VIX spikes well
    - HMM: Learns regime transitions, smoother classifications
    """

    def __init__(
        self,
        rule_weight: float = 0.4,
        hmm_weight: float = 0.6,
        n_states: int = 4
    ):
        """
        Parameters
        ----------
        rule_weight : float
            Weight for rule-based detector (0-1).
        hmm_weight : float
            Weight for HMM detector (0-1).
        n_states : int
            Number of HMM states.
        """
        self.rule_weight = rule_weight
        self.hmm_weight = hmm_weight
        self.rule_detector = RuleBasedRegimeDetector()
        self.hmm_detector = HMMRegimeDetector(n_states=n_states)
        self._fitted = False

    def fit(self, vix: pd.Series) -> 'EnsembleRegimeDetector':
        """Fit ensemble detector."""
        self.hmm_detector.fit(vix)
        self._fitted = True
        return self

    def detect(self, vix: pd.Series) -> pd.Series:
        """Detect regime using ensemble."""
        probs = self.detect_with_probabilities(vix)
        regime = probs.idxmax(axis=1)

        # Map back to RegimeState
        state_map = {
            'prob_low_vol': RegimeState.LOW_VOL,
            'prob_normal_vol': RegimeState.NORMAL_VOL,
            'prob_high_vol': RegimeState.HIGH_VOL,
            'prob_crisis': RegimeState.CRISIS
        }
        return regime.map(state_map)

    def detect_with_probabilities(self, vix: pd.Series) -> pd.DataFrame:
        """Get ensemble regime probabilities."""
        rule_probs = self.rule_detector.detect_with_probabilities(vix)
        hmm_probs = self.hmm_detector.detect_with_probabilities(vix)

        # Weighted combination
        ensemble_probs = (
            self.rule_weight * rule_probs +
            self.hmm_weight * hmm_probs
        )

        # Normalize
        total = ensemble_probs.sum(axis=1)
        ensemble_probs = ensemble_probs.div(total, axis=0)

        return ensemble_probs


def compute_regime_features(
    vix: pd.Series,
    detector: Optional[Union[RuleBasedRegimeDetector, HMMRegimeDetector, EnsembleRegimeDetector]] = None,
    include_hmm: bool = True
) -> pd.DataFrame:
    """
    Compute regime-related features for volatility prediction.

    Parameters
    ----------
    vix : pd.Series
        VIX index values.
    detector : RegimeDetector, optional
        Pre-fitted detector. If None, creates and fits a new one.
    include_hmm : bool
        Whether to include HMM-based features (requires hmmlearn).

    Returns
    -------
    features : pd.DataFrame
        Regime features including:
        - One-hot regime indicators
        - Regime probabilities
        - Regime duration
        - Transition features
    """
    features = pd.DataFrame(index=vix.index)

    # Create detector if not provided
    if detector is None:
        if include_hmm:
            detector = EnsembleRegimeDetector()
            detector.fit(vix)
        else:
            detector = RuleBasedRegimeDetector()

    # Get regime classification
    regime = detector.detect(vix)
    probs = detector.detect_with_probabilities(vix)

    # One-hot regime indicators
    features['regime_low_vol'] = (regime == RegimeState.LOW_VOL).astype(float)
    features['regime_normal_vol'] = (regime == RegimeState.NORMAL_VOL).astype(float)
    features['regime_high_vol'] = (regime == RegimeState.HIGH_VOL).astype(float)
    features['regime_crisis'] = (regime == RegimeState.CRISIS).astype(float)

    # Regime probabilities
    features['prob_low_vol'] = probs['prob_low_vol']
    features['prob_normal_vol'] = probs['prob_normal_vol']
    features['prob_high_vol'] = probs['prob_high_vol']
    features['prob_crisis'] = probs['prob_crisis']

    # Regime entropy (uncertainty)
    probs_arr = probs.values
    probs_arr = np.clip(probs_arr, 1e-10, 1.0)  # Avoid log(0)
    entropy = -np.sum(probs_arr * np.log(probs_arr), axis=1)
    features['regime_entropy'] = entropy

    # Regime duration (days in current regime)
    regime_change = regime != regime.shift(1)
    features['regime_duration'] = regime_change.cumsum().groupby(regime_change.cumsum()).cumcount() + 1

    # Rolling regime change frequency
    features['regime_changes_20d'] = regime_change.rolling(20).sum()

    # Crisis proximity (time since last crisis)
    crisis_mask = regime == RegimeState.CRISIS
    crisis_idx = crisis_mask[crisis_mask].index
    if len(crisis_idx) > 0:
        # For each date, find days since last crisis
        days_since_crisis = pd.Series(index=vix.index, dtype=float)
        last_crisis = None
        for date in vix.index:
            if crisis_mask.get(date, False):
                last_crisis = date
                days_since_crisis[date] = 0
            elif last_crisis is not None:
                days_since_crisis[date] = (date - last_crisis).days
            else:
                days_since_crisis[date] = 999  # No prior crisis
        features['days_since_crisis'] = days_since_crisis

    # VIX acceleration (2nd derivative)
    features['vix_acceleration'] = vix.diff().diff()

    # Regime transition probability (from HMM if available)
    if hasattr(detector, 'hmm_detector') and detector.hmm_detector._fitted:
        trans_matrix = detector.hmm_detector.transition_matrix
        if trans_matrix is not None:
            # Extract crisis entry probability (average prob of transitioning to crisis)
            crisis_idx = RegimeState.CRISIS
            crisis_entry_prob = trans_matrix[:, crisis_idx].mean()
            features['crisis_entry_prob'] = crisis_entry_prob

    return features


def detect_regime_shifts(
    vix: pd.Series,
    lookback: int = 60,
    zscore_threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect regime shift points using statistical change detection.

    Parameters
    ----------
    vix : pd.Series
        VIX index values.
    lookback : int
        Rolling window for baseline statistics.
    zscore_threshold : float
        Z-score threshold for shift detection.

    Returns
    -------
    shifts : pd.DataFrame
        Shift detection features.
    """
    shifts = pd.DataFrame(index=vix.index)

    # Rolling mean and std
    rolling_mean = vix.rolling(lookback).mean()
    rolling_std = vix.rolling(lookback).std()

    # Z-score
    zscore = (vix - rolling_mean) / (rolling_std + 1e-8)
    shifts['vix_zscore'] = zscore

    # Shift indicator
    shifts['regime_shift_up'] = (zscore > zscore_threshold).astype(float)
    shifts['regime_shift_down'] = (zscore < -zscore_threshold).astype(float)

    # CUSUM (cumulative sum of deviations)
    normalized = (vix - rolling_mean) / (rolling_std + 1e-8)
    shifts['cusum_pos'] = np.maximum(0, normalized.cumsum())
    shifts['cusum_neg'] = np.minimum(0, normalized.cumsum())

    return shifts
