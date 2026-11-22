# QBER-ML — Agent Instructions

These instructions apply to the whole repository unless a more specific `AGENTS.md` overrides them.

## Project Purpose
QBER-ML explores machine-learning approaches to quantum attack detection. The current experimental track uses the Quantum Attack Dataset (QAD) for quantum-circuit attack classification. A separate planned track is a BB84 simulator and temporal-QBER detection framework.

**Do not conflate these tracks.** QAD classification results are not a reproduction of a BB84 temporal-QBER framework.

## Working Rules
1. Inspect the current repository before changing anything. Treat checked-out source, tests, configuration, and reproducible outputs as authoritative.
2. Do not assume previous documentation, notebook outputs, chat summaries, or old result files describe the current implementation.
3. Never invent metrics, dataset properties, feature definitions, attack implementations, references, test outcomes, or completed work.
4. Trace scientific claims to code, dataset metadata, experiment configuration, or reproducible output.
5. Keep QAD and BB84 work clearly separated in code, results, notebooks, and reports.
6. Prefer focused changes; avoid unrelated refactoring.
7. Use `uv` for Python dependency/environment operations and prefer `uv run ...`.
8. Keep `uv.lock` version-controlled. Do not commit datasets, secrets, local environments, caches, or large model binaries unless explicitly required.
9. Never overwrite original datasets or valuable experiment outputs. Use clearly named output locations.
10. Update documentation when implementation, commands, configuration, or project status changes.
11. Run relevant tests after code changes. Report commands and actual outcomes; never claim tests passed unless run.
12. If blocked by unavailable data, credentials, hardware, or external resources, state the limitation and add a TODO.

## Scientific and ML Standards
- Exclude target/label columns and target-derived fields from model inputs.
- Check direct and indirect leakage, including simulator metadata, attack flags, derived features, and missingness patterns.
- Split data before fitting learned preprocessing.
- Fit scalers, imputers, encoders, and feature selectors inside each CV training fold (e.g. with scikit-learn `Pipeline`).
- Keep the test set isolated from tuning and model selection.
- Use stratification where appropriate and record seeds and split configuration.
- Define the positive class and denominators for FNR/FPR.
- Distinguish cross-validation estimates from held-out test metrics.
- A zero observed error count does not imply zero future error probability.
- SHAP and ablation indicate model reliance; they do not alone establish causality.
- Record class counts, feature schema, preprocessing, model parameters, package versions, and experiment configuration.
- Do not compare QAD metrics with BB84 paper metrics as if they came from the same task and dataset.

## Documentation Conventions
- `ARCHITECTURE.md` describes the current system; mark unverified details.
- `ROADMAP.md` describes planned work and status.
- `TODO.md` is a short actionable queue linked to the roadmap.
- Generate research-report tables and figures from verified results wherever practical.
- Never invent bibliographic metadata.

## Before Finishing
Report files changed, key decisions, tests actually run and outcomes, generated artifact paths, and remaining risks/TODOs.

---

## Project-Specific Instructions

### Running the Project with `uv`
```bash
# Install/sync dependencies (uses uv.lock)
uv sync

# Run quick test pipeline (1000 samples, ~30 sec)
uv run python scripts/test_pipeline.py

# Run full pipeline (default 5000 samples, ~2 min)
uv run python scripts/run_pipeline.py --n-samples 5000

# Run with custom options
uv run python scripts/run_pipeline.py --n-samples 10000 --no-shap --no-ablation --models random_forest,xgboost

# Run BB84 simulation
uv run python scripts/simulate_bb84.py --n-transmissions 100000 --attack-type intercept_resend
uv run python scripts/simulate_bb84.py --config configs/bb84.yaml

# Extract temporal QBER features from BB84 session
uv run python scripts/extract_temporal_features.py --n-transmissions 100000 --window-size 50

# Demo predictions on trained models
uv run python -m temporal_qber.predict
```

### Running Tests
```bash
# All tests (129 tests, ~25s; test_no_constant_features_within_class fails until the key_gen_rate question is decided)
uv run pytest

# Specific test modules
uv run pytest tests/test_invariants.py -v
uv run pytest tests/test_bb84.py -v
uv run pytest tests/test_channel.py -v
uv run pytest tests/test_logging_progress_experiment.py -v
```

### Running Experiments
```bash
# Full research artifact build (pipeline + notebook + report)
uv run python scripts/build_research_artifacts.py --n-samples 10000

# Individual steps
uv run python scripts/run_pipeline.py --n-samples 10000
uv run python -m nbconvert --execute --to notebook --output notebooks/qad_experiments.ipynb notebooks/qad_experiments.ipynb
uv run python scripts/generate_report.py
```

### Generating the Report
```bash
# From saved results (requires MiKTeX on Windows)
uv run python scripts/generate_report.py

# Re-execute notebook first, then generate
uv run python scripts/generate_report.py --execute-notebook

# Custom output path
uv run python scripts/generate_report.py --output ./my_report.pdf
```
**Note:** `generate_report.py` hardcodes MiKTeX path as `C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe`. Adjust if your installation differs.

### Handling Datasets and Model Artifacts
- **QAD dataset** (~315 MB balanced, ~308 MB unbalanced): Place in `data/qad/`. Not committed. Download from Kaggle.
- **Trained models** (~520 MB total): Generated in `models/`. Not committed.
- **Results & plots**: Generated in `results/` and `report/figures/`. Plots and JSON results are committed.
- **Experiment tracking**: JSON files in `results/experiments/`. Committed.
- **Logs**: In `results/logs/`. Committed.

### Maintaining Scientific Reproducibility
- All random seeds fixed at 42 (data split, CV, model initialization, simulator)
- Experiment configuration saved in `results/experiments/experiment_<ID>.json` and `latest.json`
- `evaluation_results.json` contains complete metrics, confusion matrices, per-class results, ablation
- Package versions recorded in report appendix and `uv.lock`
- To reproduce exact results: `uv sync --locked` then run pipeline with same `--n-samples` and seed

### Keeping QAD and BB84 Experiments Separate
| Aspect | QAD Track | BB84 Track |
|--------|-----------|------------|
| Code | `dataset.py`, `train.py`, `evaluate.py`, `explain.py` | `simulator/`, `temporal/` |
| Data | `data/qad/` (external CSV) | Simulated on-the-fly |
| Features | 45 circuit/statistical | 63 temporal-QBER (paper) + 5 optional monitoring (`--monitor-features`) |
| Classes | 8 quantum-circuit attacks | 8 BB84 attacks (1 normal + 7 eavesdropping) |
| Scripts | `run_pipeline.py`, `test_pipeline.py` | `simulate_bb84.py`, `extract_temporal_features.py` |
| Notebook | `qad_experiments.ipynb` | None yet |
| Results | `results/evaluation_results.json` | `results/bb84_stage2_63/`, `results/bb84_stage2_monitors/` |

**Never:**
- Mix QAD and BB84 metrics in the same table
- Claim QAD results reproduce the BB84 paper
- Use QAD feature names for BB84 features or vice versa
- Combine QAD and BB84 datasets for training

### Key Files to Know
- `src/temporal_qber/config.py` — All constants, class names, feature columns, dataclass configs
- `src/temporal_qber/dataset.py` — QAD loading, feature prep, split, scaling
- `src/temporal_qber/train.py` — Model creation, training, CV, saving
- `src/temporal_qber/evaluate.py` — Metrics, plotting, baselines
- `src/temporal_qber/explain.py` — SHAP, feature-group ablation
- `src/temporal_qber/simulator/bb84.py` — Main BB84 protocol
- `src/temporal_qber/temporal/features.py` — 63 temporal-QBER features
- `scripts/run_pipeline.py` — Main entry point for QAD experiments
- `scripts/simulate_bb84.py` — BB84 simulation CLI
- `notebooks/qad_experiments.ipynb` — Full EDA and results
- `results/evaluation_results.json` — Machine-readable results