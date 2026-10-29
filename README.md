# Temporal-QBER Attack Detection for BB84 QKD

Implementation of the paper: "Beyond the QBER Threshold: A Temporal QBER Based Machine Learning Framework for Multi Attack Detection in BB84 QKD" (arXiv:2608.04047v1)

## Project Status

This repository implements the complete ML pipeline for quantum attack detection. Currently using the **QAD (Quantum Attack Dataset)** as a proxy dataset since the paper's BB84 temporal-QBER simulator is not yet implemented.

### Current Implementation (using QAD dataset)
- ✅ 8-class classification (clean + 7 quantum circuit attacks)
- ✅ Three classifiers: Random Forest, XGBoost, SVM-RBF
- ✅ Stratified train/test split, 5-fold CV
- ✅ Comprehensive evaluation (accuracy, macro-F1, ROC-AUC, confusion matrices)
- ✅ SHAP explainability for tree models
- ✅ Feature group ablation study
- ✅ Model persistence and prediction API

### Paper's Target (future work)
- BB84 simulator with 7 eavesdropping attacks
- Temporal QBER extraction (W=50 bit windows)
- 63 physics-informed features
- 24,000 balanced sessions (3,000 per class)
- Target: XGBoost 88.01% accuracy, 0.8803 macro-F1

## Quick Start

```bash
# Install dependencies
uv sync

# Run full pipeline (5000 samples, ~2 min)
uv run python scripts/run_pipeline.py --n-samples 5000

# Quick test (1000 samples, ~30 sec)
uv run python scripts/test_pipeline.py

# Demo predictions on trained models
uv run python -m temporal_qber.predict
```

## Dataset

**QAD (Quantum Attack Dataset)** from NetsLab, UCD:
- Source: [Kaggle](https://www.kaggle.com/datasets/dpdu1995/a-comprehensive-quantum-attack-dataset-qad)
- 1M samples, 44 features, 8 classes
- Classes: clean, swap_injection, gate_insertion, depolarizing_noise, amplitude_damping, measurement_tampering, pulse_level_attack, thermal_relaxation
- Balanced (100k per attack + 300k clean) and unbalanced variants
- Features: circuit structure, output distribution statistics, entropy, purity, attack-specific parameters

> **Note**: QAD uses quantum circuit attacks, not BB84 QKD attacks. This is a substitute for development until the BB84 simulator is built.

## Results (5000 samples)

| Model | Accuracy | Macro-F1 | ROC-AUC | Binary FNR |
|-------|----------|----------|---------|------------|
| XGBoost | 95.8% | 0.947 | 0.997 | 0.0% |
| Random Forest | 95.2% | 0.939 | 0.996 | 0.0% |
| SVM-RBF | 72.2% | 0.705 | 0.964 | 4.5% |

### Feature Ablation (XGBoost)
| Feature Group | Features | Acc Drop | F1 Drop |
|---------------|----------|----------|---------|
| attack_specific_params | 11 | 15.7% | 19.3% |
| circuit_structure | 12 | 6.2% | 7.8% |
| output_distribution | 11 | 0.2% | 0.3% |
| purity_diversity | 8 | 0.0% | 0.0% |
| entropy_features | 2 | 0.1% | 0.1% |
| timing | 1 | 0.4% | 0.5% |

**Key insight**: Attack-specific parameters (swap_intensity, dp_error_rate, mt_flip_prob, etc.) are most critical for detection.

## Project Structure

```
qber-ml/
├── pyproject.toml          # Dependencies (uv)
├── README.md
├── data/
│   └── qad/                # QAD dataset (auto-downloaded)
├── models/                 # Trained models (joblib)
├── results/                # Plots, metrics, SHAP results
├── scripts/
│   ├── run_pipeline.py     # Main pipeline
│   └── test_pipeline.py    # Quick test
└── src/temporal_qber/
    ├── __init__.py
    ├── config.py           # Constants, class names, TrainConfig
    ├── dataset.py          # Data loading, preprocessing
    ├── features.py         # Temporal-QBER feature extraction (paper's 63 features)
    ├── train.py            # Model training, CV
    ├── evaluate.py         # Metrics, baselines, plotting
    ├── explain.py          # SHAP, ablation
    └── predict.py          # Inference API
```

## Key Files

### Configuration (`config.py`)
- `QAD_CLASSES`: 8 class names for QAD dataset
- `BB84_CLASSES`: 8 class names for paper's attacks (reference)
- `TEMPORAL_WINDOW_SIZE = 50` (paper's W)
- `TrainConfig`: hyperparameters for all models

### Features (`features.py`)
Implements paper's 63 temporal-QBER features:
- Statistical (7): mean, var, skew, kurtosis, IQR, CV
- Burst/Instability (15): spike density, burst duration, jump energy, error gaps
- Spectral/Temporal (15): autocorr, DFT energy, spectral entropy, permutation entropy
- Basis-Dependent (10): rectilinear/diagonal QBER, basis asymmetry
- Channel-Interaction (16): key rate, transmission efficiency, multi-scale QBER

### Models (`train.py`)
- Random Forest: 200 trees, max_depth=20, balanced class weights
- XGBoost: 200 estimators, depth=10, lr=0.1
- SVM-RBF: Pipeline with StandardScaler, C=1.0, gamma='scale'

## Requirements

- Python 3.12+
- uv (for dependency management)
- See `pyproject.toml` for full list

## Future Work

1. **BB84 Simulator** (`bb84.py`, `channels.py`, `attacks.py`)
   - Implement 7 attack models: Intercept-Resend, PNS, Beam Splitting, Time-Shift, Trojan Horse, Quantum Cloning, Fake State
   - Configurable channel noise, photon loss, detector efficiency

2. **Temporal QBER Pipeline** (`qber.py`)
   - Basis reconciliation → sifted key → QBER sequence (W=50)

3. **Dataset Generation** (`dataset.py`)
   - Generate 24,000 balanced sessions with 63 features

4. **Reproduce Paper Results**
   - Target: 88.01% accuracy, 0.8803 macro-F1 (XGBoost, 10 seeds)

## License

MIT (for code). QAD dataset: MIT license per original authors.

## References

- Paper: arXiv:2608.04047v1
- QAD Dataset: Pasindu Udugahapattuwa et al., NetsLab, UCD (2026)