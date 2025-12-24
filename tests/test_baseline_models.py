"""
Tests for baseline models.
"""

import pytest
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qcml_rotation.models.baselines.pca_ridge import PCARidge, PCARidgeConfig
from qcml_rotation.models.baselines.autoencoder import Autoencoder, AutoencoderConfig
from qcml_rotation.models.baselines.mlp import MLP, MLPConfig


class TestPCARidge:
    """Tests for PCA + Ridge baseline."""

    def test_pca_fit_predict(self, synthetic_features, synthetic_labels):
        """Test PCA predictor can fit and predict."""
        config = PCARidgeConfig(n_components=3)
        model = PCARidge(config)
        model.fit(synthetic_features, synthetic_labels)

        preds = model.predict(synthetic_features)
        assert preds.shape == synthetic_labels.shape

    def test_pca_predictions_finite(self, synthetic_features, synthetic_labels):
        """Test PCA predictions are finite."""
        config = PCARidgeConfig(n_components=3)
        model = PCARidge(config)
        model.fit(synthetic_features, synthetic_labels)

        preds = model.predict(synthetic_features)
        assert np.all(np.isfinite(preds))

    def test_pca_default_config(self, synthetic_features, synthetic_labels):
        """Test PCA with default config."""
        model = PCARidge()  # Uses default config
        model.fit(synthetic_features, synthetic_labels)
        preds = model.predict(synthetic_features)
        assert preds.shape == synthetic_labels.shape


class TestAutoencoder:
    """Tests for Autoencoder model."""

    def test_autoencoder_forward(self, synthetic_features, device):
        """Test autoencoder forward pass."""
        input_dim = synthetic_features.shape[1]
        model = Autoencoder(input_dim=input_dim, hidden_dim=32, bottleneck_dim=8)
        model.to(device)

        x = torch.FloatTensor(synthetic_features).to(device)
        reconstructed, latent = model(x)

        assert reconstructed.shape == x.shape
        assert latent.shape[1] == 8

    def test_autoencoder_latent_shape(self, synthetic_features, device):
        """Test autoencoder latent space dimensions."""
        input_dim = synthetic_features.shape[1]

        for bottleneck in [4, 8, 16]:
            model = Autoencoder(input_dim=input_dim, hidden_dim=32, bottleneck_dim=bottleneck)
            model.to(device)

            x = torch.FloatTensor(synthetic_features).to(device)
            _, latent = model(x)

            assert latent.shape[1] == bottleneck


class TestMLP:
    """Tests for MLP model."""

    def test_mlp_forward(self, synthetic_features, device):
        """Test MLP forward pass."""
        input_dim = synthetic_features.shape[1]
        model = MLP(input_dim=input_dim, hidden_dims=[64, 32], dropout=0.2)
        model.to(device)

        x = torch.FloatTensor(synthetic_features).to(device)
        output = model(x)

        assert output.shape == (len(synthetic_features), 1)

    def test_mlp_predictions_shape(self, synthetic_features, device):
        """Test MLP output shape."""
        input_dim = synthetic_features.shape[1]
        model = MLP(input_dim=input_dim, hidden_dims=[32, 16])
        model.to(device)
        model.eval()

        x = torch.FloatTensor(synthetic_features).to(device)
        with torch.no_grad():
            output = model(x)

        assert output.shape[0] == synthetic_features.shape[0]

    def test_mlp_different_architectures(self, synthetic_features, device):
        """Test MLP with different hidden layer configurations."""
        input_dim = synthetic_features.shape[1]

        architectures = [
            [32],
            [64, 32],
            [128, 64, 32],
        ]

        for hidden_dims in architectures:
            model = MLP(input_dim=input_dim, hidden_dims=hidden_dims)
            model.to(device)

            x = torch.FloatTensor(synthetic_features).to(device)
            output = model(x)

            assert output.shape == (len(synthetic_features), 1)


class TestModelBatchSizeIndependence:
    """Test that models produce consistent results regardless of batch size."""

    def test_pca_batch_independence(self, synthetic_features, synthetic_labels):
        """Test PCA predictions don't depend on batch size."""
        config = PCARidgeConfig(n_components=3)
        model = PCARidge(config)
        model.fit(synthetic_features, synthetic_labels)

        # Full batch
        preds_full = model.predict(synthetic_features)

        # Mini batches
        preds_mini = np.concatenate([
            model.predict(synthetic_features[i:i+10])
            for i in range(0, len(synthetic_features), 10)
        ])

        np.testing.assert_array_almost_equal(preds_full, preds_mini, decimal=5)

    def test_mlp_batch_independence(self, synthetic_features, device):
        """Test MLP predictions don't depend on batch size in eval mode."""
        input_dim = synthetic_features.shape[1]
        model = MLP(input_dim=input_dim, hidden_dims=[32, 16])
        model.to(device)
        model.eval()

        x = torch.FloatTensor(synthetic_features).to(device)

        with torch.no_grad():
            # Full batch
            preds_full = model(x).cpu().numpy()

            # Mini batches
            preds_mini = []
            for i in range(0, len(x), 10):
                batch = x[i:i+10]
                preds_mini.append(model(batch).cpu().numpy())
            preds_mini = np.concatenate(preds_mini)

        np.testing.assert_array_almost_equal(preds_full.flatten(), preds_mini.flatten(), decimal=5)
