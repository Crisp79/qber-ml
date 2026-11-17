# QBER-ML Architecture

> This is a starting architecture guide based on the project's stated goals and previously described layout. Inspect the current checkout before treating every path or behavior below as verified.

## 1. Research tracks

### Track A — QAD quantum-circuit classification
The current experimental direction uses the Quantum Attack Dataset (QAD) to classify quantum-circuit samples. Previously described project context says this is an eight-class task (normal/clean plus seven attack classes), but verify the exact dataset variant, labels, row counts, and feature schema from the current data and loader.

This is an exploratory classification study. It does not automatically implement BB84 simulation or temporal-QBER detection.

### Track B — BB84 temporal-QBER detection
The planned direction is a BB84 simulation and temporal feature pipeline inspired by *Beyond the QBER Threshold: A Temporal QBER Based Machine Learning Framework for Multi Attack Detection in BB84 QKD*.

The intended components include sifted-bit windows, temporal QBER features, attack simulation, and multi-class detection. Treat these as planned unless current code proves they are implemented and tested.

## 2. Repository layout

Confirm the exact paths and responsibilities in the current checkout.

```text
QBER-ML/
├── data/                  # downloaded datasets; normally not committed
├── models/                # trained model artifacts; normally not committed
├── notebooks/             # research notebooks
├── report/                # LaTeX source, sections, bibliography, generated PDF
├── results/               # metrics, predictions, plots and experiment outputs
├── scripts/               # experiment/report scripts
├── src/
│   └── temporal_qber/     # Python package; inspect actual modules
├── tests/                 # tests and invariants
├── main.py                # entry point; verify its role
├── pyproject.toml
├── uv.lock
├── README.md
└── AGENTS.md
```

## 3. Intended QAD data flow

```text
QAD files
   -> dataset loading and schema validation
   -> target/metadata separation
   -> preprocessing and feature construction
   -> stratified train/test split
   -> CV on training data only
   -> model selection and fitting
   -> held-out predictions
   -> multiclass and clean-vs-attack metrics
   -> SHAP / feature-group ablation
   -> machine-readable results and figures
   -> notebook and LaTeX report
```

This is the intended safe flow, not proof that current code follows it. Audit the actual call path and split/preprocessing order.

## 4. Intended BB84 flow

```text
BB84 simulator
   -> channel/noise and attack models
   -> basis reconciliation and sifted bits
   -> windowed temporal QBER
   -> physics-informed temporal features
   -> leakage-safe training/evaluation
   -> attack-detection results
```

Do not combine this flow with QAD results without an explicit, scientifically justified implementation.

## 5. Modules to inspect

Previously described module names include:
- `config.py`: configuration, paths, seeds, experiment parameters.
- `dataset.py`: loading, validation, labels, splitting.
- `features.py`: feature construction. A 63-feature temporal-QBER representation was previously discussed; verify whether it exists and is used.
- `train.py`: models, preprocessing, CV, fitting, persistence.
- `evaluate.py`: metrics, confusion matrices, ROC/AUC, binary evaluation.
- `explain.py`: SHAP and feature-group ablation.
- `predict.py`: inference.

Update this section if the current source differs.

## 6. Artifact boundaries

Normally version-control source, tests, scripts, notebooks, LaTeX source, bibliography, README, project metadata, and lockfile.

Normally exclude downloaded datasets, virtual environments, caches, secrets, and large model binaries. Decide explicitly whether compact results, selected plots, and the final PDF belong in Git.

## 7. Architectural risks
1. Attack-specific metadata may reveal labels.
2. Preprocessing may be fitted before CV folds are separated.
3. Notebook outputs may be stale.
4. Manually maintained report tables may drift from results.
5. QAD and BB84 terminology/results may be mixed.
6. Dataset variants or sampling may change without experiment metadata.

## 8. Maintenance
When code changes, update the repository tree, data-flow descriptions, module responsibilities, research-track ownership, and unresolved decisions.
