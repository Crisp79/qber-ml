# QBER-ML: Quantum Attack Detection with Machine Learning

This repository contains two research tracks:

1. **QAD Track (Implemented)**: Quantum-circuit attack classification using the Quantum Attack Dataset (QAD) — 8 classes, 45 features, 3 classifiers, full ML pipeline with SHAP explainability and feature ablation.

2. **BB84 Track (Implemented)**: BB84 QKD simulator with temporal-QBER feature extraction for multi-attack detection — full pipeline including simulator, 7 paper attacks, multi-session dataset generation, model training/evaluation with SHAP and ablation, per-session baselines, and parameter randomization to prevent shortcut learning.

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

### BB84 Track — Temporal-QBER Detection ✅ IMPLEMENTED

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
| PNS, Beam Splitting, Time-Shift, Trojan Horse, Quantum Cloning, Fake State | ✅ |
| Multi-class BB84 simulation dataset generation | ✅ |
| ML training/evaluation on BB84 features | ✅ |
| Threshold/CUSUM baselines on temporal QBER | ✅ |
| Session-level multiprocessing with checkpoint/resume | ✅ |
| Parameter randomization per session (anti-shortcut-learning) | ✅ |
| Basis-specific sifted bits tracking | ✅ |
| Per-session loss sequence tracking | ✅ |

**Verified Results (BB84, 24,000 sessions, 8 classes, 2026-11-23):** 80/20 stratified split (seed 42), 5-fold CV on the 19,200 training sessions, held-out test set of 4,800 sessions. Decoy-state source, per-session randomized channel/detector/attack parameters. Outputs: `results/bb84_stage2_63/` and `results/bb84_stage2_monitors/`.

| Model | Features | Test Accuracy | Test Macro-F1 | CV Accuracy | ROC-AUC (OvR) | Binary FNR | Binary FPR |
|-------|----------|--------------|---------------|-------------|---------------|------------|------------|
| Random Forest | paper's 63 | 85.46% | 0.8550 | 85.05% ± 0.56 | 0.9864 | 0.0219 | 0.0983 |
| XGBoost | paper's 63 | 88.10% | 0.8810 | 88.38% ± 0.29 | 0.9903 | 0.0138 | 0.0717 |
| SVM-RBF | paper's 63 | 85.96% | 0.8589 | 85.94% ± 0.25 | 0.9882 | 0.0193 | 0.0483 |
| Random Forest | 63 + 5 monitors | 94.83% | 0.9484 | 94.44% ± 0.24 | 0.9975 | 0.0098 | 0.0633 |
| XGBoost | 63 + 5 monitors | **97.04%** | **0.9705** | 96.61% ± 0.06 | 0.9986 | 0.0033 | 0.0217 |
| SVM-RBF | 63 + 5 monitors | 95.13% | 0.9515 | 94.40% ± 0.18 | 0.9975 | 0.0088 | 0.0050 |

Binary FNR = attack sessions predicted `normal` / all attack sessions; FPR = normal sessions predicted as an attack / all normal sessions. For reference, the paper reports 75.7% (RF), 88.0% (XGBoost) and 87.3% (SVM) on its own simulation with its 63 features; our 63-feature XGBoost result matches, but the simulations are not the same.

Per-class F1, XGBoost with monitors: normal 0.978, intercept-resend 0.943, PNS 0.996, beam splitting 0.983, time shift 0.952, trojan horse 0.995, quantum cloning 0.926, fake state 0.992. With the paper's 63 features the weakest classes are PNS (0.748) and beam splitting (0.811), which differ only in photon-loss statistics; the decoy-state monitors separate them.

**Key findings:**
- The 5 monitoring features (`double_click_rate`, `empty_gate_click_rate`, `decoy_gain_ratio`, `vacuum_yield`, `single_photon_fraction`) are standard QKD monitoring statistics, not part of the paper's feature set; results are reported with and without them.
- Trojan horse is identified almost entirely by one monitor (`vacuum_yield`, single-feature AUC ≈ 0.99 — probe light clicking in vacuum pulses).
- Attack-parameter ranges are modelling choices (the paper does not give its attack models) with minimum strengths; weaker attacks hide in normal noise and are to be evaluated separately.
- Earlier results (~62% accuracy, "6 of 8 attacks indistinguishable") were artefacts of simulator bugs fixed on 2026-11-21 to 2026-11-23 — see `NEWCHANGES.md`.

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
# Generate full BB84 dataset (24,000 sessions, 3000/class, 100k pulses each; ~50 min on 16 cores)
# OMP_NUM_THREADS=1 stops numerical libraries from oversubscribing the worker processes
env OMP_NUM_THREADS=1 uv run python scripts/generate_bb84_dataset.py \
  --n-sessions-per-class 3000 \
  --n-transmissions 100000 \
  --workers 16 \
  --output-dir data/bb84

# Pilot dataset (400 sessions/class, ~7 min)
env OMP_NUM_THREADS=1 uv run python scripts/generate_bb84_dataset.py \
  --n-sessions-per-class 400 --workers 16 --output-dir data/bb84_pilot

# Train + evaluate RF, XGBoost, SVM with the paper's 63 features (5-fold CV + held-out test)
uv run python scripts/run_bb84_pipeline.py \
  --data-path data/bb84/bb84_temporal_dataset.csv \
  --models random_forest,xgboost,svm_rbf --no-shap --no-ablation \
  --results-dir results/bb84_stage2_63 --models-dir models/bb84_stage2_63

# Same, with the 5 detector/decoy-state monitoring features added
uv run python scripts/run_bb84_pipeline.py \
  --data-path data/bb84/bb84_temporal_dataset.csv \
  --models random_forest,xgboost,svm_rbf --no-shap --no-ablation --monitor-features \
  --results-dir results/bb84_stage2_monitors --models-dir models/bb84_stage2_monitors

# Drop --no-shap / --no-ablation for SHAP plots and feature-group ablation (slower)

# Single BB84 simulation session
uv run python scripts/simulate_bb84.py --n-transmissions 100000 --attack-type intercept_resend

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
# All tests (108 tests, ~56s)
uv run pytest

# Specific modules
uv run pytest tests/test_invariants.py -v
uv run pytest tests/test_bb84.py -v
uv run pytest tests/test_channel.py -v
uv run pytest tests/test_logging_progress_experiment.py -v
uv run pytest tests/test_bb84_leakage.py -v  # BB84 leakage & randomization checks
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
│   ├── generate_report.py        # PDF report generator
│   ├── generate_bb84_dataset.py  # BB84 dataset generation (multiprocessing, checkpoint/resume)
│   └── run_bb84_pipeline.py      # BB84 ML pipeline (training, evaluation, SHAP, ablation)
├── src/temporal_qber/
│   ├── config.py                 # Constants, class names, configs
│   ├── dataset.py                # QAD loading, preprocessing
│   ├── features.py               # QAD feature groups; 63 temporal-QBER features (reference)
│   ├── train.py                  # Model training, CV, persistence
│   ├── evaluate.py               # Metrics, plotting, baselines (incl. per-session BB84 baselines)
│   ├── explain.py                # SHAP, feature-group ablation
│   ├── predict.py                # Inference API
│   ├── logging_config.py         # Centralized logging
│   ├── progress.py               # Progress bars, callbacks
│   ├── experiment.py             # Experiment tracking
│   ├── temporal/                 # Temporal QBER feature extraction
│   └── simulator/                # BB84 simulator
└── tests/                        # 129 tests (invariants, BB84, channels, detector side channels, logging, leakage)
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

3. **BB84 simulator bugs — fixed 2026-11-21 to 2026-11-23** (details in `NEWCHANGES.md`): five attack classes ran no attack; lost photons were counted as bit-0 clicks; channel loss was ignored on ideal channels; beam splitting / PNS skipped most pulses; several attack models had no observable effect. The former "~62.5% accuracy ceiling" was an artefact of these bugs.

4. **BB84 open issues:**
   - `key_gen_rate` is 0 whenever QBER > 11%, so it is constant within the time-shift and quantum-cloning classes (`test_no_constant_features_within_class` fails on purpose until this is decided); `permutation_entropy` is constant overall.
   - CUSUM / EWMA / Page-Hinkley baselines are not run by the pipeline (the dataset does not store per-session QBER sequences); they are reported as `null`.
   - Attack-parameter ranges are modelling choices; a weak-attack evaluation set (below the minimum strengths) is still to be built.
   - Results are from one split and seed; repeated-seed evaluation and SHAP / ablation on the new dataset are still to do.

5. **Report PDF Build**: `generate_report.py` hardcodes MiKTeX path (`C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe`). Adjust for your system.

---

## License

- Code: MIT
- QAD Dataset: MIT license per original authors (Kaggle)

---

## References

- Paper: *Beyond the QBER Threshold: A Temporal QBER Based Machine Learning Framework for Multi Attack Detection in BB84 QKD* (arXiv:2608.04047v1)
- QAD Dataset: Pasindu Udugahapattuwa et al., NetsLab, UCD (2026) — DOI: 10.34740/kaggle/ds/9778273