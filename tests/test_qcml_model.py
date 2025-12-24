"""
Tests for QCML (Quantum-Cognitive Machine Learning) model.
"""

import pytest
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qcml_rotation.models.qcml import (
    HilbertSpaceEncoder,
    HermitianObservable,
    QCML,
    QCMLConfig,
)


class TestHilbertSpaceEncoder:
    """Tests for Hilbert space encoder."""

    def test_encoder_output_shape(self, synthetic_features, device):
        """Test encoder output shape."""
        input_dim = synthetic_features.shape[1]
        hilbert_dim = 16

        encoder = HilbertSpaceEncoder(
            input_dim=input_dim,
            hilbert_dim=hilbert_dim,
            hidden_dim=32,
            use_complex=True
        )
        encoder.to(device)

        x = torch.FloatTensor(synthetic_features).to(device)
        psi = encoder(x)

        # Complex valued: output dim = 2 * hilbert_dim (real + imag)
        expected_dim = 2 * hilbert_dim
        assert psi.shape == (len(synthetic_features), expected_dim)

    def test_encoder_normalization(self, synthetic_features, device):
        """Test that encoder outputs are normalized (unit norm)."""
        input_dim = synthetic_features.shape[1]
        hilbert_dim = 16

        encoder = HilbertSpaceEncoder(
            input_dim=input_dim,
            hilbert_dim=hilbert_dim,
            hidden_dim=32,
            use_complex=True
        )
        encoder.to(device)

        x = torch.FloatTensor(synthetic_features).to(device)
        psi = encoder(x)

        # Compute norms
        norms = torch.norm(psi, dim=1)

        # All norms should be 1.0 (unit vectors)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_encoder_real_only_mode(self, synthetic_features, device):
        """Test encoder in real-only mode."""
        input_dim = synthetic_features.shape[1]
        hilbert_dim = 16

        encoder = HilbertSpaceEncoder(
            input_dim=input_dim,
            hilbert_dim=hilbert_dim,
            hidden_dim=32,
            use_complex=False
        )
        encoder.to(device)

        x = torch.FloatTensor(synthetic_features).to(device)
        psi = encoder(x)

        # Real only: output dim = hilbert_dim
        assert psi.shape == (len(synthetic_features), hilbert_dim)


class TestHermitianObservable:
    """Tests for Hermitian observable."""

    def test_observable_output_shape(self, device):
        """Test observable output shape."""
        hilbert_dim = 16
        batch_size = 32

        observable = HermitianObservable(dim=hilbert_dim, use_complex=True)
        observable.to(device)

        # Create mock state vectors (normalized)
        psi = torch.randn(batch_size, 2 * hilbert_dim, device=device)
        psi = psi / torch.norm(psi, dim=1, keepdim=True)

        output = observable(psi)
        assert output.shape == (batch_size,)

    def test_observable_real_output(self, device):
        """Test that observable output is real-valued."""
        hilbert_dim = 16
        batch_size = 32

        observable = HermitianObservable(dim=hilbert_dim, use_complex=True)
        observable.to(device)

        psi = torch.randn(batch_size, 2 * hilbert_dim, device=device)
        psi = psi / torch.norm(psi, dim=1, keepdim=True)

        output = observable(psi)

        # Output should be real (no imaginary part)
        assert output.dtype in [torch.float32, torch.float64]


class TestQCML:
    """Tests for full QCML model."""

    def test_qcml_forward(self, synthetic_features, device):
        """Test QCML forward pass."""
        input_dim = synthetic_features.shape[1]
        config = QCMLConfig(hilbert_dim=16)
        model = QCML(input_dim=input_dim, config=config)
        model.to(device)

        x = torch.FloatTensor(synthetic_features).to(device)
        output = model(x)

        assert output.shape == (len(synthetic_features),)

    def test_qcml_predictions_finite(self, synthetic_features, device):
        """Test QCML predictions are finite."""
        input_dim = synthetic_features.shape[1]
        config = QCMLConfig(hilbert_dim=16)
        model = QCML(input_dim=input_dim, config=config)
        model.to(device)
        model.eval()

        x = torch.FloatTensor(synthetic_features).to(device)
        with torch.no_grad():
            output = model(x)

        assert torch.all(torch.isfinite(output))

    def test_qcml_predictions_bounded(self, synthetic_features, device):
        """Test QCML predictions are reasonably bounded."""
        input_dim = synthetic_features.shape[1]
        config = QCMLConfig(hilbert_dim=16)
        model = QCML(input_dim=input_dim, config=config)
        model.to(device)
        model.eval()

        x = torch.FloatTensor(synthetic_features).to(device)
        with torch.no_grad():
            output = model(x)

        # Predictions should be relatively small
        assert torch.abs(output).max() < 10.0


class TestQCMLConfig:
    """Tests for QCML configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QCMLConfig()

        assert config.hilbert_dim == 16
        assert config.encoder_hidden == 32
        assert config.ranking_weight == 0.3

    def test_custom_config(self):
        """Test custom configuration."""
        config = QCMLConfig(
            hilbert_dim=32,
            encoder_hidden=64,
            ranking_weight=0.5,
            use_complex=False
        )

        assert config.hilbert_dim == 32
        assert config.ranking_weight == 0.5
        assert config.use_complex == False


class TestQCMLAblations:
    """Tests for QCML ablation variants."""

    def test_real_only_ablation(self, synthetic_features, device):
        """Test QCML with real-only embeddings."""
        input_dim = synthetic_features.shape[1]
        config = QCMLConfig(hilbert_dim=16, use_complex=False)
        model = QCML(input_dim=input_dim, config=config)
        model.to(device)

        x = torch.FloatTensor(synthetic_features).to(device)
        output = model(x)

        assert output.shape == (len(synthetic_features),)

    def test_complex_vs_real_different(self, synthetic_features, device):
        """Test that complex and real-only models have different architectures."""
        input_dim = synthetic_features.shape[1]

        config_complex = QCMLConfig(hilbert_dim=16, use_complex=True)
        config_real = QCMLConfig(hilbert_dim=16, use_complex=False)

        model_complex = QCML(input_dim=input_dim, config=config_complex)
        model_real = QCML(input_dim=input_dim, config=config_real)

        # Check encoder output dimensions differ
        assert model_complex.encoder.use_complex != model_real.encoder.use_complex
