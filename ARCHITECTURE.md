# QBER-ML Architecture

> This document describes the actual current system based on inspection of the checked-out source code, tests, and reproducible outputs.

## 1. Research Tracks

### Track A — QAD Quantum-Circuit Classification (IMPLEMENTED)
The current experimental direction uses the Quantum Attack Dataset (QAD) to classify quantum-circuit samples. This is an eight-class task (1 clean/benign + 7 quantum circuit attack classes) with 45 model features (42 base + 3 one-hot encoded `circuit_type`).

**This is an exploratory classification study. It does not implement BB84 simulation or temporal-QBER detection.**

Verified dataset properties:
- Source: Kaggle (dpdu1995/a-comprehensive-quantum-attack-dataset-qad)
- 1,000,000 samples (balanced: 300k clean + 100k per attack; unbalanced variant also available)
- 8 classes: `clean`, `swap_injection`, `gate_insertion`, `depolarizing_noise`, `amplitude_damping`, `measurement_tampering`, `pulse_level_attack`, `thermal_relaxation`
- 44 raw features → 45 model features after one-hot encoding `circuit_type`
- Features include circuit structure, output distribution statistics, entropy, purity/diversity metrics, **attack-specific simulator parameters** (e.g., `swap_intensity`, `dp_error_rate`, `mt_flip_prob`), and execution metadata

### Track B — BB84 Temporal-QBER Detection (IMPLEMENTED)
The BB84 simulation and temporal-QBER detection pipeline inspired by *Beyond the QBER Threshold: A Temporal QBER Based Machine Learning Framework for Multi Attack Detection in BB84 QKD* (arXiv:2608.04047v1).

Current implementation status:
- ✅ BB84 state preparation (Alice)
- ✅ Basis selection and measurement (Bob)
- ✅ Quantum channel models: Ideal, Depolarizing, Phase Flip, Amplitude Damping, Bit Flip
- ✅ Photon loss and detector imperfections (detection efficiency, dark counts)
- ✅ Basis reconciliation and sifted-bit generation
- ✅ QBER calculation (overall session)
- ✅ Temporal windowing (SessionConfig with W=50, configurable stride)
- ✅ Temporal QBER extraction per window
- ✅ 63 physics-informed temporal-QBER features (statistical, burst, spectral, basis-dependent, channel-interaction)
- ✅ Intercept-resend attack (full and partial)
- ✅ Photon Number Splitting (PNS) — `pns`, `pns_storage` variants
- ✅ Beam Splitting — `beam_splitting`, `beam_splitting_fixed` variants
- ✅ Time-Shift — `time_shift`, `time_shift_intercept` variants
- ✅ Trojan Horse — `trojan_horse`, `trojan_horse_active` variants
- ✅ Quantum Cloning — `quantum_cloning`, `quantum_cloning_measure` variants
- ✅ Fake State — `fake_state` with multiple subtypes
- ✅ Multi-class BB84 simulation dataset generation — `scripts/generate_bb84_dataset.py`
- ✅ ML training/evaluation on BB84 simulation data — `scripts/run_bb84_pipeline.py`
- ✅ Session-level multiprocessing with checkpoint/resume
- ✅ Per-session attack & channel parameter randomization (anti-shortcut-learning)
- ✅ Basis-specific sifted bits tracking
- ✅ Per-session loss sequence tracking
- ✅ Threshold/CUSUM/EWMA/Page-Hinkley baselines on temporal QBER
- ✅ SHAP explanations and feature-group ablation on BB84 data

**⚠️ Known BB84 Simulator Bugs** (identified in dataset analysis):
- Bob treats zero-norm states as detections — beam_splitting QBER shows ~0.83% but code adds no errors
- fake_state has QBER of exactly 50% — detector-mismatch attack shouldn't make bits fully random
- quantum_cloning gives 14.8% QBER, not 16.7% (1/6) as expected
- `key_gen_rate` is 0 for normal class — should be highest

**Accuracy Ceiling**: `normal`, `photon_number_splitting`, and `trojan_horse` all produce QBER ≈ 0 (indistinguishable); `intercept_resend` and `time_shift` both produce QBER ≈ 0.25 (indistinguishable). Model accuracy cannot exceed ~62.5% without simulator fixes.

**Do not combine QAD results with BB84 results. They are different tasks, datasets, and threat models.**

## 2. Repository Layout (Verified)

```text
QBER-ML/
├── configs/
│   └── bb84.yaml              # BB84 simulator configuration
├── data/
│   ├── qad/                   # QAD dataset (not committed - 300MB+ CSVs)
│   │   ├── quantum_attack_balanced_1M_dataset.csv
│   │   └── quantum_attack_unbalanced_1M_dataset.csv
│   └── bb84/                  # Empty directory for BB84 simulation outputs
├── models/                    # Trained model artifacts (not committed - 500MB+)
│   ├── random_forest.joblib   # ~500MB
│   ├── xgboost.joblib         # ~8MB
│   ├── svm_rbf.joblib         # ~2.6MB
│   ├── preprocessing.joblib   # Scaler + feature names
│   └── class_names.joblib
├── notebooks/
│   └── qad_experiments.ipynb  # EDA, training, evaluation, SHAP, ablation
├── report/
│   ├── main.tex               # LaTeX report source
│   ├── references.bib         # Bibliography
│   ├── sections/              # Report sections
│   ├── figures/               # Generated plots (committed)
│   └── generated/             # Build artifacts (main.pdf, tables)
├── results/
│   ├── evaluation_results.json    # Complete metrics, confusion matrices, ablation
│   ├── *.png                      # Confusion matrices, SHAP, ablation, ROC, per-class F1
│   ├── logs/                      # Pipeline log files
│   └── experiments/               # Experiment tracking JSON files
├── scripts/
│   ├── run_pipeline.py           # Main QAD ML pipeline
│   ├── test_pipeline.py          # Quick test (1000 samples)
│   ├── simulate_bb84.py          # BB84 simulation CLI
│   ├── extract_temporal_features.py  # Temporal QBER feature extraction CLI
│   ├── build_research_artifacts.py # Orchestration script
│   ├── generate_report.py        # PDF report generator
│   ├── generate_bb84_dataset.py  # BB84 dataset generation (multiprocessing, checkpoint/resume, param randomization)
│   └── run_bb84_pipeline.py      # BB84 ML pipeline (training, evaluation, SHAP, ablation, baselines)
├── src/
│   └── temporal_qber/           # Main Python package
│       ├── __init__.py
│       ├── config.py             # Constants, class names, TrainConfig, PipelineConfig
│       ├── dataset.py            # QAD loading, preprocessing, splitting, scaling
│       ├── features.py           # QAD feature groups; paper's 63 temporal-QBER features (reference)
│       ├── train.py              # Model creation, training, CV, persistence
│       ├── evaluate.py           # Metrics, confusion matrices, baselines (incl. per-session BB84 baselines)
│       ├── explain.py            # SHAP, feature-group ablation
│       ├── predict.py            # Inference API
│       ├── logging_config.py     # Centralized logging with file persistence
│       ├── progress.py           # Progress bars (tqdm), XGBoost callback, CV tracker
│       ├── experiment.py         # JSON-based experiment tracking
│       ├── temporal/
│       │   ├── __init__.py
│       │   ├── windowing.py      # Temporal window partitioning
│       │   ├── qber.py           # Window QBER calculation, statistics, trends
│       │   └── features.py       # 63 temporal-QBER feature extraction
│       └── simulator/
│           ├── __init__.py
│           ├── bb84.py           # Main BB84 protocol orchestration
│           ├── alice.py          # State preparation
│           ├── bob.py            # Measurement with detection efficiency/dark counts
│           ├── channel.py        # Quantum channel models
│           ├── session.py        # Session management with temporal windows
│           └── attacks/
│               ├── __init__.py
│               ├── base.py       # Attack base classes
│               ├── intercept_resend.py   # Full & partial intercept-resend
│               ├── pns.py            # Photon Number Splitting
│               ├── beam_splitting.py     # Beam splitting attack
│               ├── time_shift.py         # Time-shift attack
│               ├── trojan_horse.py       # Trojan Horse attack
│               ├── quantum_cloning.py    # Quantum Cloning attack
│               └── fake_state.py         # Fake State attack
├── tests/
│   ├── test_invariants.py        # QAD pipeline invariants (leakage, splitting, groups)
│   ├── test_bb84.py              # BB84 simulator correctness
│   ├── test_channel.py           # Quantum channel models
│   ├── test_bb84_leakage.py      # BB84 leakage detection & parameter randomization
│   └── test_logging_progress_experiment.py  # Logging, progress, tracking
├── main.py                       # Not used (entry point via scripts)
├── pyproject.toml
├── uv.lock
├── README.md
├── AGENTS.md
├── ARCHITECTURE.md
├── ROADMAP.md
└── TODO.md
```

## 3. Actual QAD Data Flow (Verified)

```mermaid
flowchart TD
    A[QAD CSV files] --> B[load_qad_dataset]
    B --> C[prepare_features_targets]
    C --> D[One-hot encode circuit_type]
    C --> E[Extract X (45 features), y (attack_label)]
    C --> F[Verify attack_type, attack_label excluded]
    E --> G[split_data: 80/20 stratified, seed=42]
    G --> H[X_train, X_test, y_train, y_test]
    H --> I[fit_scaler on X_train only]
    I --> J[StandardScaler fitted]
    J --> K[transform_data: scale train & test]
    K --> L[X_train_scaled, X_test_scaled]
    L --> M[train_all_models]
    M --> N[Create RF, XGB, SVM-RBF with balanced weights]
    N --> O[Train on full X_train_scaled]
    N --> P[5-fold CV on X_train_scaled]
    O --> Q[Evaluate on X_test_scaled]
    P --> R[Per-fold metrics + aggregate]
    Q --> S[Comprehensive metrics]
    S --> T[SHAP explanations (RF, XGB)]
    S --> U[Feature-group ablation (retrain per group)]
    T --> V[Save models, preprocessing, results]
    U --> V
    R --> V
```

**⚠️ Known Issue:** The `StandardScaler` is fitted on the **entire** `X_train` before cross-validation. This means validation folds during CV have their scaling parameters influenced by data that should be held out. For tree models (RF, XGB) this has no practical effect. For SVM, the `Pipeline` includes a second `StandardScaler`, but it operates on already-scaled data. A leakage-safe design would fit the scaler inside each CV fold (e.g., via `Pipeline` for all models).

## 4. Actual Model Training & Evaluation Flow (Verified)

### Models & Hyperparameters
| Model | Key Hyperparameters | Class Balancing |
|-------|---------------------|-----------------|
| Random Forest | `n_estimators=200`, `max_depth=20`, `random_state=42` | `class_weight='balanced'` |
| XGBoost | `n_estimators=200`, `max_depth=10`, `learning_rate=0.1`, `eval_metric='mlogloss'` | `sample_weight=compute_balanced_sample_weights(y_train)` |
| SVM-RBF | `C=1.0`, `gamma='scale'`, `probability=True` in `Pipeline(StandardScaler, SVC)` | `class_weight='balanced'` |

### Cross-Validation
- 5-fold `StratifiedKFold(shuffle=True, random_state=42)`
- Models cloned per fold via `sklearn.base.clone`
- XGBoost receives fold-specific sample weights
- Per-fold accuracy, macro-F1, train/eval time recorded

### Evaluation Metrics (Test Set)
- **Multiclass:** Accuracy, Macro-F1, Macro-Precision, Macro-Recall, ROC-AUC (OvR)
- **Per-class:** Precision, Recall, F1, Support
- **Binary (attack vs. clean):** FNR, FPR (clean=class 0, attacks=classes 1-7)

### Saved Artifacts (Verified)
- `models/{random_forest,xgboost,svm_rbf}.joblib`
- `models/preprocessing.joblib` (scaler + feature_names)
- `models/class_names.joblib`
- `results/evaluation_results.json` (complete metrics, confusion matrices, ablation)
- `results/*_confusion_matrix.png`, `*_shap_summary.png`, `*_shap_beeswarm.png`, `*_ablation.png`
- `results/per_class_f1_comparison.png`, `results/roc_curves.png`

## 5. Notebook & Report Generation Workflow

### Notebook (`notebooks/qad_experiments.ipynb`)
- Loads 10,000 sample subset for EDA
- Reproduces data loading, splitting, scaling
- Loads saved models from `models/`
- Generates all figures: class distribution, feature correlation, model comparison, confusion matrices, per-class F1, ROC curves, SHAP importance
- Documents QAD vs. BB84 distinction explicitly

### Report (`report/main.tex`)
- LaTeX source with 10 sections + appendix
- Dynamic tables generated from `evaluation_results.json` via `scripts/generate_report.py`
- Figures copied from `results/` to `report/figures/`
- Compiled with `pdflatex` (MiKTeX on Windows)
- Appendix documents package versions, experiment config, reproduction commands

### Orchestration
```bash
# Full workflow
uv run python scripts/build_research_artifacts.py --n-samples 10000

# Individual steps
uv run python scripts/run_pipeline.py --n-samples 10000
uv run python -m nbconvert --execute --to notebook --output notebooks/qad_experiments.ipynb notebooks/qad_experiments.ipynb
uv run python scripts/generate_report.py
```

## 6. Dataset & Model Artifact Locations

| Artifact | Location | Committed |
|----------|----------|-----------|
| QAD balanced CSV | `data/qad/quantum_attack_balanced_1M_dataset.csv` | No (~315 MB) |
| QAD unbalanced CSV | `data/qad/quantum_attack_unbalanced_1M_dataset.csv` | No (~308 MB) |
| Trained models | `models/*.joblib` | No (~520 MB total) |
| Preprocessing | `models/preprocessing.joblib` | No |
| Evaluation results | `results/evaluation_results.json` | Yes |
| Generated plots | `results/*.png`, `report/figures/*.png` | Yes (plots) |
| Experiment tracking | `results/experiments/*.json` | Yes |
| Pipeline logs | `results/logs/experiment_*.log` | Yes |

## 7. Configuration & Dependency Management

- **Python:** 3.12+ (enforced by `pyproject.toml`)
- **Package manager:** `uv` (lockfile `uv.lock` committed)
- **Key dependencies:** numpy, pandas, scikit-learn, xgboost, shap, matplotlib, seaborn, joblib, scipy, pytest
- **Configuration:** `configs/bb84.yaml` for BB84 simulator; `TrainConfig` and `PipelineConfig` dataclasses in `config.py` for QAD pipeline
- **Random seeds:** Fixed at 42 for data split, CV, and model initialization (documented in appendix)

## 8. Separation Between QAD and BB84 Research Tracks

| Aspect | QAD Track | BB84 Track |
|--------|-----------|------------|
| **Directory** | `src/temporal_qber/dataset.py`, `train.py`, `evaluate.py`, `explain.py` | `src/temporal_qber/simulator/`, `temporal/` |
| **Data** | `data/qad/` (external CSV) | Simulated on-the-fly |
| **Features** | 45 circuit/statistical features | 63 temporal-QBER features (paper) |
| **Classes** | 8 quantum-circuit attacks | 8 BB84 attacks (1 normal + 7 attacks) |
| **Scripts** | `run_pipeline.py`, `test_pipeline.py` | `generate_bb84_dataset.py`, `run_bb84_pipeline.py`, `simulate_bb84.py`, `extract_temporal_features.py` |
| **Notebook** | `qad_experiments.ipynb` | None yet |
| **Results** | `results/evaluation_results.json` | `results/bb84/evaluation_results.json` |
| **Status** | **Complete pipeline** | **Complete pipeline** |

## 9. Architectural Risks (Confirmed)

1. **QAD Attack-specific metadata leakage:** QAD features include `swap_intensity`, `dp_error_rate`, `mt_flip_prob`, `ad_damping_param`, etc. that directly encode attack type and intensity. Models achieve >95% accuracy largely by reading these simulator parameters. This is documented in the notebook and report.

2. **QAD Preprocessing fitted before CV folds:** `StandardScaler` fitted on full `X_train` before CV. Validation fold scaling is influenced by held-out data. Tree models unaffected; SVM double-scales.

3. **BB84 Simulator Bugs** (identified in dataset analysis):
   - Bob treats zero-norm states as detections — beam_splitting QBER ~0.83% but code adds no errors
   - fake_state QBER exactly 50% — detector-mismatch attack shouldn't randomize bits
   - quantum_cloning QBER 14.8%, not expected 16.7% (1/6)
   - `key_gen_rate` is 0 for normal class — should be highest

4. **BB84 Accuracy Ceiling** (~62.5%): `normal`, `photon_number_splitting`, `trojan_horse` all produce QBER ≈ 0; `intercept_resend`, `time_shift` both produce QBER ≈ 0.25. Model accuracy cannot exceed ~62.5% without simulator fixes.

5. **Notebook outputs stale:** Notebook shows results from 10k-sample run; full 1M-sample results may differ.

6. **Report tables manually maintained:** `generate_report.py` regenerates tables from `evaluation_results.json`, but the main LaTeX text references specific numbers that must be kept in sync.

7. **Dataset variants:** Balanced vs. unbalanced QAD variants exist; only balanced used in current experiments.

8. **SVM Pipeline double-scaling:** Data pre-scaled by external `StandardScaler`, then SVM `Pipeline` applies another `StandardScaler`.

## 10. Maintenance

When code changes, update:
- Repository tree (Section 2)
- Data flow descriptions (Section 3)
- Module responsibilities (Section 4)
- Research track ownership (Section 8)
- Unresolved decisions and risks (Section 9)
- Run `uv run pytest` and verify all tests pass
- Update `ROADMAP.md` statuses based on evidence
- Update `TODO.md` with actionable next steps