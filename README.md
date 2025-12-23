# QCML: Quantum-Cognitive Market Learning

Weekly ETF rotation strategy using quantum-inspired state representations for excess return prediction.

## Overview

This project explores a novel approach to factor rotation by representing market states as quantum-like amplitude vectors in a complex Hilbert space. The core idea is that ETFs exist in a "superposition" of factor exposures, and their relative performance can be predicted by measuring observable operators on these state vectors.

The model predicts weekly excess returns (vs SPY) for a universe of sector and factor ETFs, then constructs a portfolio by going long the top-ranked ETFs and short the bottom-ranked.

## ETF Universe

**Benchmark:** SPY

**Sectors:** XLK, XLF, XLE, XLY, XLP, XLV, XLU, XLI, XLB

**Factors:** MTUM (momentum), QUAL (quality), VLUE (value), USMV (min vol)

**Style:** IWD (value), IWF (growth), IWM (small cap), IWB (large cap)

## Features

For each ETF at each weekly rebalance date:
- Log returns: 1d, 5d, 20d
- Realized volatility: 20-day annualized
- Relative vs SPY: return spread, vol ratio

Labels are next-week log returns minus SPY's next-week return.

## Models

### Baselines
1. **PCA + Ridge**: Compress features to principal components, predict with ridge regression
2. **Autoencoder + Linear**: Learn compressed representation via reconstruction, use bottleneck for prediction
3. **MLP**: Direct 2-layer feedforward network

### QCML (Quantum-Cognitive Model)
The core model implements a quantum-inspired architecture:

1. **Hilbert Space Encoder**: Maps feature vector x ∈ R^F to complex amplitudes z ∈ C^d
   - Two-layer MLP: F → hidden → 2d (real and imaginary parts)
   - Normalize to unit vector: |ψ⟩ = z / ||z||

2. **Hermitian Observable**: Learnable matrix W = A + A† (ensures Hermiticity)
   - Prediction: ŷ = Re(⟨ψ|W|ψ⟩)

3. **Loss Function**: Combined MSE + ranking loss
   - MSE term: Standard regression loss
   - Ranking term: Pairwise hinge loss within each week
   - Mimics quantum interference between competing "concept states"

**Ablations supported:**
- No ranking loss (MSE only)
- Real-only embeddings (no imaginary part)

## Project Structure

```
qcml_rotation/
├── data/           # Data loading, features, dataset classes
├── models/         # Baseline and QCML implementations
├── training/       # Training loops, losses, early stopping
├── backtest/       # Backtesting engine and metrics
└── utils/          # Config, seeding, I/O helpers

scripts/
├── run_data_prep.py    # Download data and build features
├── train_baselines.py  # Train baseline models
├── train_qcml.py       # Train QCML model
└── run_backtest.py     # Backtest and evaluate
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download data and prepare features
python scripts/run_data_prep.py

# Train baseline models
python scripts/train_baselines.py

# Train QCML model
python scripts/train_qcml.py

# With ablations (no ranking, real-only)
python scripts/train_qcml.py --run-ablations

# Run backtest (coming soon)
python scripts/run_backtest.py
```

## Configuration

Edit `configs/default.yaml` to adjust:
- ETF universe and benchmark
- Feature windows
- Train/val/test split ratios
- Model hyperparameters
- Backtest settings (top K, transaction costs)

## Backtest Metrics

- Annualized Sharpe and Sortino ratios
- Maximum drawdown and Calmar ratio
- Portfolio turnover
- Hit rate (% of picks that outperform SPY)

## References

- Busemeyer & Bruza (2012). Quantum Models of Cognition and Decision
- Khrennikov (2010). Ubiquitous Quantum Structure
- Haven & Khrennikov (2013). Quantum Social Science

## Notes

This is a research project exploring whether quantum probability structures offer advantages for financial prediction. The "quantum" aspect is purely mathematical (complex amplitudes, interference) - no actual quantum computing is involved.

Transaction costs are modeled at 5bps per side. Real-world implementation would require additional considerations around liquidity, slippage, and execution.
