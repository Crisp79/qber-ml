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
- ✅ **Centralized logging with file persistence**
- ✅ **Progress bars with ETA for all pipeline stages**
- ✅ **Detailed cross-validation reporting per fold**
- ✅ **Experiment tracking with JSON metadata**

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

## New Features: Logging, Progress & Experiment Tracking

### Logging System
Centralized logging using Python's standard `logging` module:
- Console output with timestamps and log levels
- Persistent log files under `results/logs/experiment_<ID>.log`
- Unique log filenames using timestamps/experiment IDs
- Log levels: DEBUG, INFO, WARNING, ERROR
- Full experiment configuration logged at start
- Exceptions logged with traceback

### Progress Bars
Using `tqdm` for meaningful progress reporting:
- Dataset loading (chunked for large files)
- Feature preparation and scaling
- Model training (per-model)
- Cross-validation (per-fold with ETA)
- XGBoost boosting iterations (per-iteration)
- SHAP explanation computation
- Feature ablation (per-group)
- Evaluation and plotting

### Cross-Validation Reporting
Enhanced CV reporting for each model:
- Fold-by-fold accuracy, macro-F1, training time, evaluation time
- Mean ± std across folds
- Total and average fold duration
- Per-fold metrics preserved in experiment metadata

### Experiment Tracking
Lightweight JSON-based experiment tracking:
```
results/
├── logs/              # Log files
├── experiments/       # Experiment metadata (JSON)
│   ├── experiment_<ID>.json
│   └── latest.json    # Symlink to latest
├── metrics/           # Evaluation results
├── models/            # Saved models
└── plots/             # Generated figures
```

Each experiment records:
- Experiment ID, timestamp, git commit hash
- Dataset variant, samples, features, class distribution
- Train/test split, CV configuration, random seeds
- Model hyperparameters
- Per-fold and aggregate metrics
- Training/evaluation durations
- Paths to models and plots
- Completion status and errors

## CLI Options

```bash
uv run python scripts/run_pipeline.py --help
```

### New Options
| Option | Description |
|--------|-------------|
| `--log-level LEVEL` | Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO) |
| `--disable-progress` | Disable tqdm progress bars |
| `--experiment-name NAME` | Custom experiment ID |
| `--quiet` | Disable console logging (file only) |
| `--verbose` | Enable DEBUG logging |
| `--demo` | Run prediction demo on saved models |

### Preserved Options
| Option | Description |
|--------|-------------|
| `--n-samples N` | Number of samples (for quick testing) |
| `--unbalanced` | Use unbalanced dataset variant |
| `--no-shap` | Skip SHAP explanations |
| `--no-ablation` | Skip feature ablation |
| `--models-dir DIR` | Models output directory |
| `--results-dir DIR` | Results output directory |

## Example Terminal Output

```
2026-11-03 19:30:12 INFO ==============================
2026-11-03 19:30:12 INFO TEMPORAL-QBER ATTACK DETECTION PIPELINE
2026-11-03 19:30:12 INFO ==============================
2026-11-03 19:30:13 INFO Loading balanced QAD dataset from quantum_attack_balanced_1M_dataset.csv
2026-11-03 19:30:25 INFO Dataset loaded: 1000000 samples in 12.3s
2026-11-03 19:30:25 INFO Preparing features and targets...
2026-11-03 19:30:27 INFO Features prepared: 45 features, 8 classes in 1.8s
2026-11-03 19:30:27 INFO Dataset shape: (1000000, 47)
2026-11-03 19:30:27 INFO Feature matrix: (1000000, 45)
2026-11-03 19:30:27 INFO Number of features: 45
2026-11-03 19:30:27 INFO Number of classes: 8
2026-11-03 19:30:27 INFO Class distribution:
2026-11-03 19:30:27 INFO   clean: 300,000 (30.0%)
2026-11-03 19:30:27 INFO   swap_injection: 100,000 (10.0%)
...
2026-11-03 19:30:27 INFO Splitting data (test_size=0.2, stratified=True)...
2026-11-03 19:30:27 INFO Fitting feature scaler...
2026-11-03 19:30:28 INFO Scaler fitted in 0.5s
2026-11-03 19:30:28 INFO Transforming features...
2026-11-03 19:30:28 INFO Features scaled in 0.3s

==================================================
Model: xgboost
==================================================
2026-11-03 19:30:28 INFO Training xgboost...
2026-11-03 19:30:28 INFO [xgboost] Fold 1/5 started
2026-11-03 19:32:15 INFO [xgboost] Fold 1/5 completed in 107.2s | Acc: 0.9581 | Macro-F1: 0.9472
2026-11-03 19:34:02 INFO [xgboost] Fold 2/5 completed in 106.8s | Acc: 0.9593 | Macro-F1: 0.9485
...
2026-11-03 19:38:45 INFO [xgboost] CV complete: 5 folds, total=535.2s, avg=107.0s/fold
2026-11-03 19:38:45 INFO Training xgboost...
2026-11-03 19:42:10 INFO Completed in 204.5s
2026-11-03 19:42:10 INFO Evaluating model on test set...
2026-11-03 19:42:11 INFO Evaluation complete in 0.8s
2026-11-03 19:42:11 INFO   Accuracy: 0.9581 | Macro-F1: 0.9472 | ROC-AUC: 0.9971
```

With progress bars enabled:
```
Loading dataset: 100%|██████████| 1000000/1000000 [00:12<00:00, 82341.23rows/s]
Training xgboost: 45%|████▌     | 90/200 [01:23<01:42,  1.58iter/s]
CV xgboost: 60%|██████    | 3/5 [05:21<03:34, 107.04s/fold]
```

## Progress Support by Model

| Model | Training Progress | CV Fold Progress | Notes |
|-------|-------------------|------------------|-------|
| **XGBoost** | ✅ Per-iteration (boosting rounds) | ✅ Per-fold | Uses XGBoost callbacks |
| **Random Forest** | ⚠️ Stage-level only | ✅ Per-fold | sklearn doesn't expose tree-level progress |
| **SVM-RBF** | ⚠️ Stage-level only | ✅ Per-fold | sklearn SVC has no fit progress callback |

**SVM Limitation**: sklearn's SVC does not provide a reliable fit progress callback. The pipeline reports stage-level progress (start/end) with elapsed time. Approximate ETA is based on previous comparable runs when available.

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
│   ├── logs/               # Pipeline log files
│   ├── experiments/        # Experiment metadata (JSON)
│   └── *.png               # Generated plots
├── scripts/
│   ├── run_pipeline.py     # Main pipeline
│   ├── test_pipeline.py    # Quick test
│   ├── build_research_artifacts.py  # Full workflow
│   └── generate_report.py  # PDF report generator
└── src/temporal_qber/
    ├── __init__.py
    ├── config.py           # Constants, class names, TrainConfig, PipelineConfig
    ├── dataset.py          # Data loading, preprocessing
    ├── features.py         # Temporal-QBER feature extraction (paper's 63 features)
    ├── train.py            # Model training, CV
    ├── evaluate.py         # Metrics, baselines, plotting
    ├── explain.py          # SHAP, ablation
    ├── predict.py          # Inference API
    ├── logging_config.py   # Centralized logging
    ├── progress.py         # Progress bars & tracking
    └── experiment.py       # Experiment tracking
```

## Key Files

### Configuration (`config.py`)
- `QAD_CLASSES`: 8 class names for QAD dataset
- `BB84_CLASSES`: 8 class names for paper's attacks (reference)
- `TEMPORAL_WINDOW_SIZE = 50` (paper's W)
- `TrainConfig`: hyperparameters for all models + progress/logging settings
- `PipelineConfig`: complete pipeline configuration for experiment tracking

### New Modules
- `logging_config.py`: Centralized logging setup, file/console handlers
- `progress.py`: ProgressManager, XGBoost callback, TrainingProgressTracker
- `experiment.py`: ExperimentTracker, ExperimentConfig, ModelResult, incremental JSON saving

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
- tqdm (for progress bars) - already in dependencies
- See `pyproject.toml` for full list

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test modules
uv run pytest tests/test_invariants.py -v
uv run pytest tests/test_logging_progress_experiment.py -v

# Run pipeline test (quick)
uv run python scripts/test_pipeline.py
```

Tests verify:
- Logging configuration and file output
- Progress bar functionality (enabled/disabled)
- Experiment tracking serialization
- Model configuration invariants
- Dataset loading and preprocessing invariants
- No label leakage in features
- Pipeline runs without errors on small samples

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