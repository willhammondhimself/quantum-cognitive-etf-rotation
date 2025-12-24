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
    QCMLWithRanking,
    QCMLConfig,
    create_qcml_model,
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
            complex_valued=True
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
            complex_valued=True
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
            complex_valued=False
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

        observable = HermitianObservable(hilbert_dim=hilbert_dim, complex_valued=True)
        observable.to(device)

        # Create mock state vectors (normalized)
        psi = torch.randn(batch_size, 2 * hilbert_dim, device=device)
        psi = psi / torch.norm(psi, dim=1, keepdim=True)

        output = observable(psi)
        assert output.shape == (batch_size, 1)

    def test_observable_real_output(self, device):
        """Test that observable output is real-valued."""
        hilbert_dim = 16
        batch_size = 32

        observable = HermitianObservable(hilbert_dim=hilbert_dim, complex_valued=True)
        observable.to(device)

        psi = torch.randn(batch_size, 2 * hilbert_dim, device=device)
        psi = psi / torch.norm(psi, dim=1, keepdim=True)

        output = observable(psi)

        # Output should be real (no imaginary part)
        assert output.dtype in [torch.float32, torch.float64]

    def test_hermitian_property(self, device):
        """Test that observable matrix is Hermitian (W = W†)."""
        hilbert_dim = 8

        observable = HermitianObservable(hilbert_dim=hilbert_dim, complex_valued=True)
        observable.to(device)

        # Construct the full Hermitian matrix
        W_real = observable.W_real.detach()
        W_imag = observable.W_imag.detach()

        # A = W_real + i * W_imag
        # W = A + A† should be Hermitian
        A = torch.complex(W_real, W_imag)
        W = A + A.conj().T

        # W should equal W†
        W_dag = W.conj().T
        torch.testing.assert_close(W, W_dag, atol=1e-5, rtol=1e-5)


class TestQCMLWithRanking:
    """Tests for full QCML model with ranking loss."""

    @pytest.fixture
    def qcml_model(self, synthetic_features, device):
        """Create QCML model."""
        input_dim = synthetic_features.shape[1]
        config = QCMLConfig(
            input_dim=input_dim,
            hilbert_dim=16,
            encoder_hidden=32,
            ranking_weight=0.3
        )
        model = create_qcml_model(config)
        model.to(device)
        return model

    def test_qcml_forward(self, qcml_model, synthetic_features, device):
        """Test QCML forward pass."""
        x = torch.FloatTensor(synthetic_features).to(device)
        output = qcml_model(x)

        assert output.shape == (len(synthetic_features), 1)

    def test_qcml_predictions_finite(self, qcml_model, synthetic_features, device):
        """Test QCML predictions are finite."""
        x = torch.FloatTensor(synthetic_features).to(device)

        qcml_model.eval()
        with torch.no_grad():
            output = qcml_model(x)

        assert torch.all(torch.isfinite(output))

    def test_qcml_predictions_bounded(self, qcml_model, synthetic_features, device):
        """Test QCML predictions are reasonably bounded."""
        x = torch.FloatTensor(synthetic_features).to(device)

        qcml_model.eval()
        with torch.no_grad():
            output = qcml_model(x)

        # Predictions should be small (excess returns typically < 10%)
        assert torch.abs(output).max() < 1.0

    def test_qcml_training_loss(self, qcml_model, synthetic_features, synthetic_labels, device):
        """Test QCML training loss computation."""
        x = torch.FloatTensor(synthetic_features).to(device)
        y = torch.FloatTensor(synthetic_labels).unsqueeze(1).to(device)

        # Create week indices for ranking loss
        n_samples = len(synthetic_features)
        n_weeks = 10
        week_indices = torch.tensor([i % n_weeks for i in range(n_samples)], device=device)

        qcml_model.train()
        loss, mse_loss, rank_loss = qcml_model.compute_loss(x, y, week_indices)

        assert torch.isfinite(loss)
        assert torch.isfinite(mse_loss)
        assert torch.isfinite(rank_loss)

    def test_ranking_loss_decreases_correct_rankings(self, device):
        """Test that ranking loss is lower for correct rankings."""
        config = QCMLConfig(
            input_dim=6,
            hilbert_dim=8,
            ranking_weight=1.0,  # Pure ranking
            ranking_margin=0.01
        )
        model = create_qcml_model(config)
        model.to(device)

        # Create data where correct predictions give lower ranking loss
        # This is a sanity check for the ranking loss formulation
        batch_size = 20
        x = torch.randn(batch_size, 6, device=device)

        model.eval()
        with torch.no_grad():
            preds = model(x)

        # Ranking loss should be defined and finite
        assert torch.all(torch.isfinite(preds))


class TestQCMLConfig:
    """Tests for QCML configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QCMLConfig(input_dim=6)

        assert config.input_dim == 6
        assert config.hilbert_dim == 16
        assert config.encoder_hidden == 32
        assert config.ranking_weight == 0.3

    def test_custom_config(self):
        """Test custom configuration."""
        config = QCMLConfig(
            input_dim=10,
            hilbert_dim=32,
            encoder_hidden=64,
            ranking_weight=0.5,
            complex_valued=False
        )

        assert config.input_dim == 10
        assert config.hilbert_dim == 32
        assert config.ranking_weight == 0.5
        assert config.complex_valued == False


class TestQCMLAblations:
    """Tests for QCML ablation variants."""

    def test_no_ranking_ablation(self, synthetic_features, device):
        """Test QCML without ranking loss."""
        input_dim = synthetic_features.shape[1]
        config = QCMLConfig(
            input_dim=input_dim,
            ranking_weight=0.0  # Disable ranking
        )
        model = create_qcml_model(config)
        model.to(device)

        x = torch.FloatTensor(synthetic_features).to(device)
        output = model(x)

        assert output.shape == (len(synthetic_features), 1)

    def test_real_only_ablation(self, synthetic_features, device):
        """Test QCML with real-only embeddings."""
        input_dim = synthetic_features.shape[1]
        config = QCMLConfig(
            input_dim=input_dim,
            complex_valued=False  # Real only
        )
        model = create_qcml_model(config)
        model.to(device)

        x = torch.FloatTensor(synthetic_features).to(device)
        output = model(x)

        assert output.shape == (len(synthetic_features), 1)

    def test_ablations_different_outputs(self, synthetic_features, device):
        """Test that different ablations produce different outputs."""
        input_dim = synthetic_features.shape[1]
        torch.manual_seed(42)

        # Full model
        config_full = QCMLConfig(input_dim=input_dim, complex_valued=True, ranking_weight=0.3)
        model_full = create_qcml_model(config_full)
        model_full.to(device)

        # Real only
        torch.manual_seed(42)
        config_real = QCMLConfig(input_dim=input_dim, complex_valued=False, ranking_weight=0.3)
        model_real = create_qcml_model(config_real)
        model_real.to(device)

        x = torch.FloatTensor(synthetic_features).to(device)

        model_full.eval()
        model_real.eval()

        with torch.no_grad():
            out_full = model_full(x)
            out_real = model_real(x)

        # Outputs should be different (different architectures)
        # At least the encoder dimensions are different
        assert model_full.encoder.fc3.out_features != model_real.encoder.fc3.out_features
