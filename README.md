# QBER-ML: Quantum Attack Detection with Machine Learning

This repository contains two research tracks:

1. **QAD Track (Implemented)**: Quantum-circuit attack classification using the Quantum Attack Dataset (QAD) — 8 classes, 45 features, 3 classifiers, full ML pipeline with SHAP explainability and feature ablation.

2. **BB84 Track (Partially Implemented)**: BB84 QKD simulator with temporal-QBER feature extraction for multi-attack detection — simulator foundation complete, 6/7 paper attacks remaining, dataset generation and ML evaluation not yet implemented.

**These tracks are distinct.** QAD results do not reproduce the BB84 temporal-QBER paper.

---

## Current Implementation Status

### QAD Track — Quantum-Circuit Classification ✅ IMPLEMENTED

| Component | Status |
|-----------|--------|
| Dataset loading (balanced/unbalanced QAD) | ✅ |
| Feature engineering (44 raw → 45 model features) | ✅ |
| Stratified train/test split (80/20, seed=42) | ✅ |
| Preprocessing (StandardScaler) | ⚠️ Fitted before CV (leakage risk for SVM) |
| Random Forest (200 trees, balanced weights) | ✅ |
| XGBoost (200 estimators, sample weights) | ✅ |
| SVM-RBF (Pipeline with StandardScaler) | ✅ |
| 5-fold stratified CV | ✅ |
| Comprehensive metrics (Acc, Macro-F1, ROC-AUC, per-class) | ✅ |
| Binary attack-detection metrics (FNR, FPR) | ✅ |
| SHAP explanations (TreeExplainer for RF/XGB) | ✅ |
| Feature-group ablation (leave-one-group-out) | ✅ |
| Model persistence (joblib) | ✅ |
| Inference API | ✅ |
| Centralized logging + file persistence | ✅ |
| Progress bars (tqdm) + experiment tracking (JSON) | ✅ |

**Verified Results (10,000 samples):**
| Model | Test Accuracy | Macro-F1 | ROC-AUC (OvR) | Binary FNR |
|-------|--------------|----------|---------------|------------|
| XGBoost | 95.75% | 0.9463 | 0.9976 | 0.0% |
| Random Forest | 95.45% | 0.9421 | 0.9972 | 0.0% |
| SVM-RBF | 78.30% | 0.7539 | 0.9727 | 4.8% |

**Critical Finding:** Top SHAP features for both RF and XGBoost are **attack-specific simulator parameters** (`swap_intensity`, `dp_error_rate`, `mt_flip_prob`, `has_noise_model`, etc.) that directly encode the attack configuration. The high accuracy likely reflects learning simulator settings rather than generalizable attack signatures. See notebook for details.

### BB84 Track — Temporal-QBER Detection 🔄 PARTIALLY IMPLEMENTED

| Component | Status |
|-----------|--------|
| BB84 state preparation (Alice) | ✅ |
| Basis selection & measurement (Bob) | ✅ |
| Quantum channel models (Ideal, Depolarizing, Phase Flip, Amplitude Damping, Bit Flip) | ✅ |
| Photon loss & detector imperfections | ✅ |
| Basis reconciliation & sifted-bit generation | ✅ |
| Overall session QBER calculation | ✅ |
| Temporal windowing (W=50, configurable stride) | ✅ |
| Windowed QBER extraction | ✅ |
| 63 physics-informed temporal-QBER features | ✅ |
| Intercept-resend attack (full + partial) | ✅ |
| PNS, Beam Splitting, Time-Shift, Trojan Horse, Quantum Cloning, Fake State | ❌ PLANNED |
| Multi-class BB84 simulation dataset generation | ❌ PLANNED |
| ML training/evaluation on BB84 features | ❌ PLANNED |
| Threshold/CUSUM baselines on temporal QBER | ❌ PLANNED (placeholders only) |

---

## Quick Start

### Prerequisites
- Python 3.12+
- `uv` (recommended) or pip
- MiKTeX (for PDF report generation on Windows)

### Installation
```bash
# Clone and sync dependencies
git clone <repo-url>
cd QBER-ML
uv sync
```

### QAD Track Commands
```bash
# Quick test (1000 samples, ~30 sec)
uv run python scripts/test_pipeline.py

# Full pipeline (5000 samples default, ~2 min)
uv run python scripts/run_pipeline.py --n-samples 5000

# Custom options
uv run python scripts/run_pipeline.py --n-samples 10000 --no-shap --no-ablation --models random_forest,xgboost

# Demo predictions on saved models
uv run python -m temporal_qber.predict
```

### BB84 Track Commands
```bash
# Full intercept-resend attack simulation
uv run python scripts/simulate_bb84.py --n-transmissions 100000 --attack-type intercept_resend

# Partial intercept-resend (50% interception)
uv run python scripts/simulate_bb84.py --n-transmissions 100000 --attack-type partial_intercept_resend --attack-param 0.5

# With channel noise and loss
uv run python scripts/simulate_bb84.py --n-transmissions 100000 --channel-type depolarizing --channel-param 0.1 --loss-prob 0.1

# With detector imperfections
uv run python scripts/simulate_bb84.py --n-transmissions 100000 --detection-efficiency 0.9 --dark-count-prob 0.01

# Using YAML config
uv run python scripts/simulate_bb84.py --config configs/bb84.yaml

# Extract temporal QBER features from a session
uv run python scripts/extract_temporal_features.py --n-transmissions 100000 --window-size 50 --stride 25
```

### Research Artifacts
```bash
# Full workflow: pipeline + notebook + report
uv run python scripts/build_research_artifacts.py --n-samples 10000

# Generate PDF report from saved results
uv run python scripts/generate_report.py

# Re-execute notebook then generate report
uv run python scripts/generate_report.py --execute-notebook
```

### Testing
```bash
# All tests (95 tests, ~47s)
uv run pytest

# Specific modules
uv run pytest tests/test_invariants.py -v
uv run pytest tests/test_bb84.py -v
uv run pytest tests/test_channel.py -v
uv run pytest tests/test_logging_progress_experiment.py -v
```

---

## Dataset

**QAD (Quantum Attack Dataset)** from NetsLab, UCD:
- Source: [Kaggle](https://www.kaggle.com/datasets/dpdu1995/a-comprehensive-quantum-attack-dataset-qad)
- 1M samples, 44 raw features, 8 classes
- Classes: `clean`, `swap_injection`, `gate_insertion`, `depolarizing_noise`, `amplitude_damping`, `measurement_tampering`, `pulse_level_attack`, `thermal_relaxation`
- Balanced (300k clean + 100k per attack) and unbalanced variants
- Features: circuit structure, output distribution statistics, entropy, purity, **attack-specific parameters**

> **Note:** QAD uses quantum circuit attacks, not BB84 QKD attacks. This is a substitute for development until the BB84 simulator is built.

---

## Project Structure

```
QBER-ML/
├── configs/                  # BB84 simulator configuration (bb84.yaml)
├── data/
│   ├── qad/                  # QAD dataset (NOT committed - download from Kaggle)
│   └── bb84/                 # BB84 simulation outputs
├── models/                   # Trained models (NOT committed - ~520 MB)
│   ├── *.joblib
│   └── preprocessing.joblib
├── notebooks/
│   └── qad_experiments.ipynb # Full EDA, training, evaluation, SHAP, ablation
├── report/
│   ├── main.tex              # LaTeX report source
│   ├── references.bib        # Bibliography
│   ├── sections/             # Report sections
│   ├── figures/              # Generated plots (committed)
│   └── generated/            # Build artifacts (main.pdf, tables)
├── results/
│   ├── evaluation_results.json    # Complete metrics, confusion matrices, ablation
│   ├── *.png                      # Confusion matrices, SHAP, ablation, ROC, per-class F1
│   ├── logs/                      # Pipeline log files
│   └── experiments/               # Experiment tracking JSON
├── scripts/
│   ├── run_pipeline.py           # Main QAD ML pipeline
│   ├── test_pipeline.py          # Quick test
│   ├── simulate_bb84.py          # BB84 simulation CLI
│   ├── extract_temporal_features.py  # Temporal QBER feature extraction
│   ├── build_research_artifacts.py # Orchestration
│   └── generate_report.py        # PDF report generator
├── src/temporal_qber/
│   ├── config.py                 # Constants, class names, configs
│   ├── dataset.py                # QAD loading, preprocessing
│   ├── features.py               # QAD feature groups; 63 temporal-QBER features (reference)
│   ├── train.py                  # Model training, CV, persistence
│   ├── evaluate.py               # Metrics, plotting, baselines
│   ├── explain.py                # SHAP, feature-group ablation
│   ├── predict.py                # Inference API
│   ├── logging_config.py         # Centralized logging
│   ├── progress.py               # Progress bars, callbacks
│   ├── experiment.py             # Experiment tracking
│   ├── temporal/                 # Temporal QBER feature extraction
│   └── simulator/                # BB84 simulator
└── tests/                        # 95 tests (invariants, BB84, channels, logging)
```

---

## Key Documentation

| File | Purpose |
|------|---------|
| `AGENTS.md` | Instructions for AI agents / developers |
| `ARCHITECTURE.md` | Actual system architecture (verified) |
| `ROADMAP.md` | Milestones with evidence-based statuses |
| `TODO.md` | Actionable next steps (priority ordered) |

---

## Known Issues & Limitations

1. **QAD Preprocessing Leakage**: `StandardScaler` fitted on full training set before CV. Tree models unaffected; SVM double-scales. Fix: use `Pipeline` for all models inside CV.

2. **Attack-Specific Feature Leakage**: QAD features include simulator parameters that directly reveal attack type/intensity. Models achieve >95% accuracy by reading these. No experiment yet excludes them.

3. **BB84 Track Incomplete**: Only intercept-resend attack implemented. 6 paper attacks remaining. No BB84 dataset generation or ML evaluation yet.

4. **Report PDF Build**: `generate_report.py` hardcodes MiKTeX path (`C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe`). Adjust for your system.

5. **README Title**: "Temporal-QBER Attack Detection for BB84 QKD" describes the *target paper*, not current implementation. Current work is QAD classification.

---

## License

- Code: MIT
- QAD Dataset: MIT license per original authors (Kaggle)

---

## References

- Paper: *Beyond the QBER Threshold: A Temporal QBER Based Machine Learning Framework for Multi Attack Detection in BB84 QKD* (arXiv:2608.04047v1)
- QAD Dataset: Pasindu Udugahapattuwa et al., NetsLab, UCD (2026) — DOI: 10.34740/kaggle/ds/9778273