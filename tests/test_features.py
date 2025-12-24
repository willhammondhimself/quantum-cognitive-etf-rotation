"""
Tests for feature engineering module.
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qcml_rotation.data.features import (
    FeatureConfig,
    get_feature_names,
)


class TestFeatureConfig:
    """Tests for FeatureConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FeatureConfig()
        assert config.return_windows == [1, 5, 20]
        assert config.vol_window == 20
        assert config.benchmark == 'SPY'

    def test_custom_config(self):
        """Test custom configuration."""
        config = FeatureConfig(
            return_windows=[1, 5, 10],
            vol_window=10,
            benchmark='QQQ'
        )
        assert config.return_windows == [1, 5, 10]
        assert config.vol_window == 10
        assert config.benchmark == 'QQQ'


class TestGetFeatureNames:
    """Tests for get_feature_names function."""

    def test_feature_names_count(self):
        """Test correct number of feature names."""
        config = FeatureConfig(return_windows=[1, 5, 20], vol_window=20)
        names = get_feature_names(config)
        # 3 return features + 1 vol + 2 relative features = 6
        assert len(names) == 6

    def test_feature_names_content(self):
        """Test feature names contain expected patterns."""
        config = FeatureConfig(return_windows=[1, 5, 20])
        names = get_feature_names(config)

        assert 'ret_1d' in names
        assert 'ret_5d' in names
        assert 'ret_20d' in names
        assert 'vol_20d' in names


class TestFeatureValues:
    """Tests for feature value computation."""

    def test_features_no_nan_in_valid_range(self, synthetic_features):
        """Test that features have no NaN values."""
        assert not np.isnan(synthetic_features).any()

    def test_features_reasonable_scale(self, synthetic_features):
        """Test that features are in reasonable range."""
        # Returns should typically be < 20% in absolute value
        assert np.abs(synthetic_features).max() < 1.0

    def test_features_shape(self, synthetic_features):
        """Test features have expected shape."""
        assert synthetic_features.ndim == 2
        assert synthetic_features.shape[1] == 6  # 6 features


class TestLabelComputation:
    """Tests for label (excess return) computation."""

    def test_labels_no_nan(self, synthetic_labels):
        """Test that labels have no NaN values."""
        assert not np.isnan(synthetic_labels).any()

    def test_labels_reasonable_scale(self, synthetic_labels):
        """Test that labels are in reasonable range."""
        # Weekly excess returns should typically be < 10%
        assert np.abs(synthetic_labels).max() < 0.5

    def test_labels_centered_near_zero(self, synthetic_labels):
        """Test that labels are centered near zero."""
        assert np.abs(synthetic_labels.mean()) < 0.1
