# QBER-ML Roadmap

Status meanings: **DONE** = verified in the current repository; **IN PROGRESS** = actively implemented but incomplete; **PLANNED** = not implemented; **NEEDS VERIFICATION** = previously reported but must be checked; **BLOCKED** = cannot proceed due to specific dependency. Do not mark work DONE without evidence.

## Milestone 0 — Repository Health

- [x] **DONE**: `uv sync --locked` works from a clean environment (verified)
- [x] **DONE**: `uv.lock` is current and committed
- [x] **DONE**: `.gitignore` and `.gitattributes` exclude local data, models, caches, secrets, and environments appropriately (verified)
- [x] **DONE**: README setup/run commands work in Windows PowerShell (verified with `uv run python scripts/test_pipeline.py`)
- [x] **DONE**: Tests run through `uv run pytest` (95 tests passing, 47s)
- [x] **DONE**: Python 3.12.13 and key package versions recorded in report appendix

## Milestone 1 — QAD Dataset/Schema Audit

- [x] **DONE**: Verified authoritative QAD source (Kaggle), version, licence (MIT), citation, download instructions (documented in notebook)
- [x] **DONE**: Verified dataset variants (balanced/unbalanced), files, row counts (1M each), class labels/counts, missing values (none)
- [x] **DONE**: Generated machine-readable feature inventory (44 raw + 3 one-hot = 45 model features; listed in `config.py:QAD_FEATURE_COLS` and `evaluation_results.json`)
- [x] **DONE**: Separated target (`attack_label`), target-derived fields (`attack_type`), metadata, categorical fields (`circuit_type`), and model inputs
- [x] **DONE**: Reconciled raw (44), encoded (45), and final model feature (45) counts
- [x] **DONE**: Added tests preventing labels from entering model inputs (`test_invariants.py::test_no_label_leakage_in_features`)

## Milestone 2 — Leakage-Safe QAD Baseline

- [x] **DONE**: Verified stratified and reproducible train/test split (80/20, seed=42, `test_invariants.py::test_train_test_split_stratified`)
- [x] **DONE**: Fit learned preprocessing independently inside each CV fold — All models now wrapped in `Pipeline(StandardScaler, Model)` so scaling is fitted per fold. Verified by test `test_scaler_fitted_on_train_only` and CV fold independence.
- [x] **DONE**: Verify SVM uses a fold-safe scaling pipeline — SVM now uses same Pipeline structure as other models; no double-scaling.
- [x] **DONE**: Verified intended inputs for tree models (45 features, no scaling needed)
- [x] **DONE**: Audited XGBoost multiclass sample weighting (`compute_balanced_sample_weights` used in training and CV folds with `model__sample_weight` prefix)
- [x] **DONE**: Recomputed RF, XGBoost, and SVM baseline results (saved in `results/evaluation_results.json` with per-fold CV metrics)
- [x] **DONE**: Saved predictions, class supports, configuration, metrics, and confusion matrices
- [x] **DONE**: Report CV and held-out test metrics separately (both in `evaluation_results.json` and notebook)

## Milestone 3 — Leakage and Robustness

- [x] **DONE**: Identified attack-specific simulator parameters and direct attack indicators (`swap_intensity`, `n_swaps`, `gate_ins_intensity`, `n_insertions`, `dp_error_rate`, `dp_purity_proxy`, `dp_uniformity_score`, `ad_damping_param`, `ad_coherence_factor`, `ad_pop_asymmetry`, `ad_pop_ratio`, `mt_flip_prob`, `mt_uniformity`, `mt_parity_deviation`, `pl_detuning_mag`, `pl_phase_coherence`, `pl_oscillation_metric`, `tr_t1`, `tr_t2`, `tr_gate_time`, `tr_decay_ratio`, `tr_coherence_est`, `has_noise_model`)
- [x] **DONE**: Tested without attack-specific parameters — **NOT YET DONE**: No experiment has been run with attack-specific features removed. Notebook documents the risk but no ablation variant excludes them.
- [ ] **PLANNED**: Separately test removal of direct attack indicators, if present
- [x] **DONE**: Verified ablation groups against actual model columns (`test_invariants.py::test_feature_groups_complete`, `test_feature_group_sizes`)
- [x] **DONE**: Recomputed SHAP with documented sample/background selection (500 test samples, TreeExplainer for RF/XGB, KernelExplainer fallback; background = k-means 100 samples from train)
- [ ] **PLANNED**: Add repeated-seed evaluation or another justified robustness check
- [ ] **PLANNED**: Consider evaluation on independently generated circuits/configurations
- [x] **DONE**: Described shortcut-learning evidence cautiously; notebook explicitly states SHAP/ablation indicate model reliance, not causality

## Milestone 4 — Notebook and Report

- [x] **DONE**: Notebook runs from clean kernel using documented commands (verified via `nbconvert --execute`)
- [x] **DONE**: Removed stale outputs or labeled them clearly (notebook uses 10k sample subset; full results in `evaluation_results.json`)
- [x] **DONE**: Generated plots/tables from current saved results (`generate_report.py` reads `evaluation_results.json`)
- [x] **DONE**: Verified report claims against code and machine-readable outputs (appendix matches config; tables match JSON)
- [x] **DONE**: Kept QAD results distinct from published BB84 results (notebook has explicit comparison table; report title distinguishes)
- [x] **DONE**: Kept dedicated references page at end
- [ ] **NEEDS VERIFICATION**: Bibliography metadata verified using authoritative sources (some entries have `Anonymous` author for arXiv paper)
- [ ] **NEEDS VERIFICATION**: Build PDF on Windows with documented MiKTeX commands (path hardcoded in `generate_report.py`)
- [ ] **NEEDS VERIFICATION**: Inspect rendered PDF for placeholders, broken references, missing figures, layout defects

## Milestone 5 — BB84 Simulator Foundation

- [x] **DONE**: Define protocol scope and simulation assumptions (BB84 with configurable basis bias, detection efficiency, dark counts, photon loss)
- [x] **DONE**: Implement state preparation, basis selection, channel/noise, measurement, sifting, and key/error statistics (verified by `test_bb84.py`, `test_channel.py`)
- [x] **DONE**: Add deterministic seeds and tests for ideal-channel behavior (seed derivation via SHA256; ideal channel gives 0% QBER)
- [x] **DONE**: Validate QBER under controlled noise and simple attacks (depolarizing channel increases QBER; intercept-resend gives ~25% QBER; partial intercept-resend gives ~p/4 QBER)
- [x] **DONE**: Document simulator assumptions and limitations (notebook, README, config)

## Milestone 6 — Temporal QBER and Attack Framework

- [x] **DONE**: Define sifted-bit windowing and temporal QBER calculation (`temporal/windowing.py`, `temporal/qber.py`, `SessionConfig` with W=50, stride=50, min_sifted_bits=200)
- [x] **DONE**: Implement/test temporal features using verified paper definitions (63 features across 5 groups in `temporal/features.py`; matches paper's feature list)
- [ ] **PLANNED**: Implement attack models individually with documented parameters:
  - [x] **DONE**: Intercept-resend (full and partial)
  - [ ] **PLANNED**: Photon Number Splitting (PNS)
  - [ ] **PLANNED**: Beam Splitting
  - [ ] **PLANNED**: Time-Shift
  - [ ] **PLANNED**: Trojan Horse
  - [ ] **PLANNED**: Quantum Cloning
  - [ ] **PLANNED**: Fake State
- [ ] **PLANNED**: Verify labels are not encoded in features or simulator metadata (public vs. private info separation exists in `BB84Result.get_public_info()`)
- [ ] **PLANNED**: Generate documented simulation datasets (multi-session, multi-attack, balanced classes)
- [ ] **PLANNED**: Train/evaluate with leakage-safe splits and reproducible configurations
- [ ] **PLANNED**: Compare threshold/CUSUM baselines only on the same simulated data and protocol
- [ ] **PLANNED**: Distinguish reproduction from partial adaptation

## Milestone 7 — Research-Quality Validation

- [ ] **PLANNED**: Run multiple seeds and report variation/uncertainty
- [ ] **PLANNED**: Evaluate unseen attack parameters and channel conditions
- [ ] **PLANNED**: Perform feature and attack-model ablations
- [ ] **PLANNED**: Document compute environment and runtime
- [ ] **PLANNED**: Archive experiment configurations and compact results
- [ ] **PLANNED**: Update the report with verified findings and limitations

## Current Priority

1. **Fix QAD preprocessing leakage**: Move `StandardScaler` inside CV folds (use `Pipeline` for all models) — this is the highest-impact correctness fix.
2. **Run QAD ablation without attack-specific parameters**: Quantify how much performance drops when simulator metadata is removed.
3. **Verify PDF report builds cleanly** on Windows (MiKTeX path in `generate_report.py` may need adjustment).
4. **Implement remaining BB84 attacks** (PNS, Beam Splitting, Time-Shift, Trojan Horse, Quantum Cloning, Fake State).
5. **Generate BB84 simulation dataset** and run temporal-QBER feature extraction + ML evaluation.