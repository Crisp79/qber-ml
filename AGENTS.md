# QBER-ML — Agent Instructions

These instructions apply to the whole repository unless a more specific `AGENTS.md` overrides them.

## Project purpose
QBER-ML explores machine-learning approaches to quantum attack detection. The current experimental track uses the Quantum Attack Dataset (QAD) for quantum-circuit attack classification. A separate planned track is a BB84 simulator and temporal-QBER detection framework.

**Do not conflate these tracks.** QAD classification results are not a reproduction of a BB84 temporal-QBER framework.

## Working rules
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

## Scientific and ML standards
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

## Documentation conventions
- `ARCHITECTURE.md` describes the current system; mark unverified details.
- `ROADMAP.md` describes planned work and status.
- `TODO.md` is a short actionable queue linked to the roadmap.
- Generate research-report tables and figures from verified results wherever practical.
- Never invent bibliographic metadata.

## Before finishing
Report files changed, key decisions, tests actually run and outcomes, generated artifact paths, and remaining risks/TODOs.
