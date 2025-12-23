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

### QCML (coming soon)
- Map features to complex amplitude vectors via learned encoder
- Normalize to unit vectors in C^d (Hilbert space)
- Predict via Hermitian observable: y = Re(<ψ|W|ψ>)
- Train with MSE + pairwise ranking loss (interference term)

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

# (after implementing QCML)
python scripts/train_qcml.py
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
