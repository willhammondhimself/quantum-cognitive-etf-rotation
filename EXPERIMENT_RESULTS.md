# QCML ETF Rotation - Experiment Results

## Executive Summary

After extensive experimentation with various models and strategies, here are the key findings:

### Key Result
**Simple long-only momentum rotation matches SPY performance** - no statistically significant alpha was found using ML-based approaches, but systematic momentum strategies provide comparable risk-adjusted returns.

## Detailed Results

### Walk-Forward Validation Results (Out-of-Sample)

| Strategy | Sharpe | Return | Max DD | Notes |
|----------|--------|--------|--------|-------|
| **Long-Only Momentum (k=5)** | 0.83 | +278% | -30% | Best approach |
| XGBoost (Rank target) | 0.35 | +31% | -21% | Rank helps |
| Pure L/S Momentum | 0.03 | -4% | -20% | Short leg hurts |
| XGBoost (Return target) | -0.11 | -13% | -37% | Doesn't work |
| SPY Buy-and-Hold | 0.83 | +299% | -34% | Benchmark |

### Statistical Significance

**Long-Only Momentum Strategy:**
- T-statistic: 2.74
- P-value: 0.003 (significant at 1%)
- Bootstrap 95% CI for Sharpe: [0.20, 1.46] (excludes 0)
- **BUT**: Does not significantly outperform SPY (excess p-value = 0.62)

### Key Insights

1. **ML for return prediction doesn't work**: All ML models (XGBoost, LightGBM, MLP, QCML) showed negative Sharpe ratios in walk-forward validation when predicting raw returns.

2. **Rank prediction helps ML**: Using ranking as the target instead of returns improved XGBoost Sharpe from -0.11 to +0.35.

3. **Long-only beats long-short**: The short leg consistently hurts performance. Winners keep winning more reliably than losers keep losing.

4. **Momentum is hard to beat**: Simple relative strength ranking (21/63/126-day lookbacks) performs as well as complex ML models.

5. **Trend filter reduces drawdowns**: Adding a 200-day MA filter reduces max drawdown from -30% to -14%, but at the cost of reduced returns.

## Recommended Strategy

For practical use, consider:

1. **Long-Only Relative Strength Momentum**:
   - Rank ETFs by composite momentum (21/63/126 day)
   - Hold top 5 ETFs
   - Weekly rebalancing
   - Expected Sharpe: ~0.8 (similar to SPY)

2. **Risk-Managed Version** (if lower drawdowns are desired):
   - Add 200-day MA trend filter on SPY
   - Go to cash when SPY < 200-day MA
   - Expected Sharpe: ~0.75, but Max DD: ~-15%

## QCML Architecture Improvements (Phase 1)

### Problem Diagnosis
The original QCML showed zero predictive power due to:
1. **Unit normalization** destroying magnitude information
2. **Hilbert dimension d=16** being too small (bottleneck)
3. **Complex arithmetic** adding overhead without benefit
4. **Single measurement** collapsing all information to 1 scalar

### Implemented Solutions

**SimplifiedQCML** - Stripped-down version:
- Removed unit normalization (preserves magnitude)
- Real numbers only (no complex overhead)
- Increased dimensions (hidden=64, embed=32)
- Added batch normalization and dropout

**RankingQCML** - Learning-to-rank approach:
- RankNet/ListNet loss for correct ordering
- Combined with small MSE component
- Stronger regularization (dropout=0.2, weight_decay=1e-3)

### Walk-Forward Benchmark Results

| Model | Avg Fold Corr | Overall Corr | Sign Acc | Sharpe |
|-------|---------------|--------------|----------|--------|
| Original QCML | 0.001 | 0.003 | 49.6% | -0.35 |
| Simplified QCML | -0.000 | -0.008 | 49.3% | -0.54 |
| **Ranking QCML** | -0.000 | **0.008** | **50.0%** | -0.40 |
| Momentum baseline | -0.013 | -0.012 | 49.4% | -0.67 |

### Key Findings

1. **Removing unit normalization helps on synthetic data** (98% vs 95% correlation)
2. **Ranking loss improves correlation** (0.008 vs -0.008 for SimplifiedQCML)
3. **Still insufficient for profitable trading** - all models have negative Sharpes
4. **The problem is fundamental** - not just architecture

### Conclusion
The QCML improvements show marginal gains in prediction metrics but don't translate to profitable trading. The signal in sector ETF returns is simply too weak/noisy for neural networks to extract reliably. Simple momentum remains the best practical approach.

## Technical Improvements Made

1. Added XGBoost and LightGBM models
2. Implemented dual momentum strategy
3. Added trend-following overlay
4. Added yield curve and credit spread regime indicators
5. Implemented rank-based prediction target
6. Added monthly rebalancing support
7. Created comprehensive experiment framework
8. **Added SimplifiedQCML** (without unit normalization)
9. **Added RankingQCML** (with RankNet/ListNet loss)
10. **Added benchmark script** for comparing QCML variants

## Honest Conclusions

1. **Sector rotation alpha is hard to find**: The market for sector ETFs is efficient. Simple momentum captures most of the available signal.

2. **ML adds complexity without alpha**: In walk-forward testing, ML models failed to beat simple momentum rules.

3. **The strategy matches SPY**: Long-only sector momentum provides market-like returns with market-like volatility.

4. **Value proposition is diversification**: The strategy provides exposure to sectors with positive momentum rather than market-cap weighting.

## Files Created/Modified

- `qcml_rotation/models/gradient_boosting.py` - XGBoost/LightGBM models
- `qcml_rotation/models/qcml.py` - Added SimplifiedQCML, RankingQCML, RankNetLoss, ListNetLoss
- `qcml_rotation/strategies/momentum.py` - Momentum and hybrid strategies
- `qcml_rotation/data/loader.py` - Added Treasury and Credit data loading
- `scripts/run_improved_experiments.py` - Comprehensive experiment script
- `scripts/benchmark_simplified_qcml.py` - QCML variant comparison script
