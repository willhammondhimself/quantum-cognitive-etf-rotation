"""
Regime detection module for volatility prediction.

Provides HMM-based and rule-based regime classification for:
- LOW_VOL: VIX < 15, calm markets
- NORMAL_VOL: 15 <= VIX < 22, typical conditions
- HIGH_VOL: 22 <= VIX < 30, elevated uncertainty
- CRISIS: VIX >= 30 or rapid vol spike
"""

from .detector import (
    RegimeState,
    RuleBasedRegimeDetector,
    HMMRegimeDetector,
    EnsembleRegimeDetector,
    compute_regime_features
)

__all__ = [
    'RegimeState',
    'RuleBasedRegimeDetector',
    'HMMRegimeDetector',
    'EnsembleRegimeDetector',
    'compute_regime_features'
]
