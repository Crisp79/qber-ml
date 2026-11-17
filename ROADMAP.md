# QBER-ML Roadmap

Status meanings: **DONE** = verified in the current repository; **IN PROGRESS** = actively implemented; **PLANNED** = not implemented; **NEEDS VERIFICATION** = previously reported but must be checked. Do not mark work DONE without evidence.

## Milestone 0 — Repository health
- [ ] NEEDS VERIFICATION: `uv sync` works from a clean environment.
- [ ] NEEDS VERIFICATION: `uv.lock` is current and committed.
- [ ] NEEDS VERIFICATION: `.gitignore` and `.gitattributes` exclude local data, models, caches, secrets, and environments appropriately.
- [ ] NEEDS VERIFICATION: README setup/run commands work in Windows PowerShell.
- [ ] NEEDS VERIFICATION: tests run through `uv run pytest`.
- [ ] NEEDS VERIFICATION: record Python and key package versions.

## Milestone 1 — QAD dataset/schema audit
- [ ] Verify authoritative QAD source, version, licence, citation, and download instructions.
- [ ] Verify dataset variant(s), files, row counts, class labels/counts, and missing values.
- [ ] Generate a machine-readable feature inventory.
- [ ] Separate target, target-derived fields, metadata, categorical fields, and model inputs.
- [ ] Reconcile raw, encoded, and final feature counts.
- [ ] Add tests preventing labels from entering model inputs.

## Milestone 2 — Leakage-safe QAD baseline
- [ ] Verify stratified and reproducible train/test split.
- [ ] Fit learned preprocessing independently inside each CV fold.
- [ ] Verify SVM uses a fold-safe scaling pipeline.
- [ ] Verify intended inputs for tree models.
- [ ] Audit XGBoost multiclass sample weighting.
- [ ] Recompute RF, XGBoost, and SVM baseline results.
- [ ] Save predictions, class supports, configuration, metrics, and confusion matrices.
- [ ] Report CV and held-out test metrics separately.

## Milestone 3 — Leakage and robustness
- [ ] Identify attack-specific simulator parameters and direct attack indicators.
- [ ] Test without attack-specific parameters.
- [ ] Separately test removal of direct attack indicators, if present.
- [ ] Verify ablation groups against actual model columns.
- [ ] Recompute SHAP with documented sample/background selection.
- [ ] Add repeated-seed evaluation or another justified robustness check.
- [ ] Consider evaluation on independently generated circuits/configurations.
- [ ] Describe shortcut-learning evidence cautiously; do not infer causality from SHAP alone.

## Milestone 4 — Notebook and report
- [ ] Run notebook from a clean kernel using documented commands.
- [ ] Remove stale outputs or label them clearly.
- [ ] Generate plots/tables from current saved results.
- [ ] Verify report claims against code and machine-readable outputs.
- [ ] Keep QAD results distinct from published BB84 results.
- [ ] Keep the dedicated references page at the end if that remains the chosen style.
- [ ] Verify bibliography metadata using authoritative sources.
- [ ] Build PDF on Windows with documented MiKTeX commands.
- [ ] Inspect rendered PDF for placeholders, broken references, missing figures, and layout defects.

## Milestone 5 — BB84 simulator foundation
- [ ] Define protocol scope and simulation assumptions.
- [ ] Implement state preparation, basis selection, channel/noise, measurement, sifting, and key/error statistics.
- [ ] Add deterministic seeds and tests for ideal-channel behavior.
- [ ] Validate QBER under controlled noise and simple attacks.
- [ ] Document simulator assumptions and limitations.

## Milestone 6 — Temporal QBER and attack framework
- [ ] Define sifted-bit windowing and temporal QBER calculation.
- [ ] Implement/test temporal features using verified paper definitions.
- [ ] Implement attack models individually with documented parameters.
- [ ] Verify labels are not encoded in features or simulator metadata.
- [ ] Generate documented simulation datasets.
- [ ] Train/evaluate with leakage-safe splits and reproducible configurations.
- [ ] Compare threshold/CUSUM baselines only on the same simulated data and protocol.
- [ ] Distinguish reproduction from partial adaptation.

## Milestone 7 — Research-quality validation
- [ ] Run multiple seeds and report variation/uncertainty.
- [ ] Evaluate unseen attack parameters and channel conditions.
- [ ] Perform feature and attack-model ablations.
- [ ] Document compute environment and runtime.
- [ ] Archive experiment configurations and compact results.
- [ ] Update the report with verified findings and limitations.

## Current priority
First audit the current checkout and update statuses based on evidence. Do not automatically implement every planned milestone.
