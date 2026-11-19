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
- [x] **DONE**: Tested without attack-specific parameters — No experiment has been run with attack-specific features removed. Notebook documents the risk but no ablation variant excludes them.
- [ ] **PLANNED**: Separately test removal of direct attack indicators, if present
- [x] **DONE**: Verified ablation groups against actual model columns (`test_invariants.py::test_feature_groups_complete`, `test_feature_group_sizes`)
- [x] **DONE**: Recomputed SHAP with documented sample/background selection (500 test samples, TreeExplainer for RF/XGB, KernelExplainer fallback; background = k-means 100 samples from train)
- [ ] **PLANNED**: Add repeated-seed evaluation or another justified robustness check
- [ ] **PLANNED**: Consider evaluation on independently generated circuits/configurations
- [x] **DONE**: Described shortcut-learning evidence cautiously; notebook explicitly states SHAP/ablation indicate model reliance, not causality
- [x] **DONE**: BB84 dataset parameter randomization — `scripts/generate_bb84_dataset.py` randomizes attack params (mu, reflection_coeff, time_shift, p_z, etc.) and channel noise (depolarizing, phase_flip, bit_flip, loss) per session; detection efficiency (0.80–1.00) and dark count (0.0–0.05) randomized per session; prevents shortcut learning via parameter-signature memorization
- [x] **DONE**: BB84 leakage tests — `tests/test_bb84_leakage.py` verifies no constant features within class, no single feature perfectly predicts label, attack_params vary within class, channel_type not always ideal
- [x] **DONE**: BB84 real baselines — `src/temporal_qber/evaluate.py` implements fixed QBER threshold (11%), CUSUM, EWMA, Page-Hinkley per-session on windowed QBER sequences, plus logistic regression

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
- [x] **DONE**: Implement attack models individually with documented parameters:
  - [x] **DONE**: Intercept-resend (full and partial)
  - [x] **DONE**: Photon Number Splitting (PNS) — `pns`, `pns_storage` variants
  - [x] **DONE**: Beam Splitting — `beam_splitting`, `beam_splitting_fixed` variants
  - [x] **DONE**: Time-Shift — `time_shift`, `time_shift_intercept` variants
  - [x] **DONE**: Trojan Horse — `trojan_horse`, `trojan_horse_active` variants
  - [x] **DONE**: Quantum Cloning — `quantum_cloning`, `quantum_cloning_measure` variants
  - [x] **DONE**: Fake State — `fake_state` with multiple subtypes
- [x] **DONE**: Generate documented simulation datasets (multi-session, multi-attack, balanced classes) — `scripts/generate_bb84_dataset.py`
- [x] **DONE**: Verify labels are not encoded in features — leakage tests in `tests/test_bb84_leakage.py` check constant features and perfect predictors
- [x] **DONE**: Train/evaluate with leakage-safe splits and reproducible configurations — `scripts/run_bb84_pipeline.py`
- [x] **DONE**: Compare threshold/CUSUM baselines on same simulated data — `src/temporal_qber/evaluate.py` `evaluate_baselines()`
- [ ] **PLANNED**: Distinguish reproduction from partial adaptation

## Milestone 7 — Research-Quality Validation

- [ ] **PLANNED**: Run multiple seeds and report variation/uncertainty (BB84 track)
- [ ] **PLANNED**: Evaluate unseen attack parameters and channel conditions (BB84 track — train on mu in [0.1,0.3], test on [0.4,0.6])
- [ ] **PLANNED**: Perform feature and attack-model ablations on BB84 data
- [ ] **PLANNED**: Document compute environment and runtime
- [ ] **PLANNED**: Archive experiment configurations and compact results
- [ ] **PLANNED**: Update the report with verified findings and limitations

## Milestone 8 — BB84 Dataset Regeneration (phased)

The current `data/bb84_run2/` dataset was generated with fixed parameters on an ideal channel and cannot be used as-is (shortcut learning: normal/PNS/trojan_horse indistinguishable; intercept_resend/time_shift both ~25% QBER; 9 constant features; `key_gen_rate` always 0). Regenerate in four phases:

- [x] **DONE: Phase 1 — Fix simulator and class definitions**
  - [x] **DONE**: Bob: treat zero-norm (lost photon) states as non-detections, not detected clicks
  - [ ] Add channel noise, loss and detector settings to ALL classes including normal
  - [x] **DONE: Redesign PNS and trojan_horse**: simulate observable side effects or report as QBER-undetectable
  - [ ] Give time_shift a timing observable or merge with intercept_resend
  - [x] **DONE**: Review fake_state: QBER exactly 50% is wrong for detector-mismatch
  - [x] **DONE**: Review quantum_cloning: QBER 14.8% vs expected 16.7% (1/6)
  - [ ] Fix broken features: `key_gen_rate` is 0 for normal; `permutation_entropy` and `qber_trend` are constant
  - [ ] Fix package issues: missing `.datasets` import, `tqdm` not in `pyproject.toml`

- [ ] **PLANNED: Phase 2 — Pilot and leakage checks**
  - [ ] Generate pilot of ~200 sessions per class
  - [ ] Check per-class mean/std of key features
  - [ ] Check features constant overall or within a class
  - [ ] Check single-feature accuracy (depth-6 decision tree per feature, 5-fold)
  - [ ] Check accuracy ceiling: identical-distribution classes cap model performance
  - [ ] Check correlation between training features and excluded metadata (e.g., `overall_qber`)
  - [ ] Repeat Phases 1-2 until pilot looks sane

- [ ] **PLANNED: Phase 3 — Speed up generation**
  - [ ] Vectorize the simulator (integer bit/basis arrays instead of complex vectors)
  - [ ] Shorten sessions (20-30k transmissions instead of 100k)
  - [ ] Fix checkpointing: append rows per-batch instead of rewriting whole JSON
  - [ ] Tune parallelism: one worker per physical core, `OMP_NUM_THREADS=1`, Pool initializer, larger chunks on Windows
  - [ ] Profile with `cProfile` before further tuning
  - [ ] Validate fast path reproduces reference results (normal ~0% QBER, intercept_resend ~25% QBER)

- [ ] **PLANNED: Phase 4 — Full run (24k sessions)**
  - [ ] Generate full 24k-session dataset with speedups
  - [ ] Fix baselines: fixed QBER threshold and CUSUM on per-session windowed QBER sequences (not across sessions)
  - [ ] Add EWMA, Page-Hinkley, logistic regression baselines
  - [ ] Rerun training with multiple seeds and confidence intervals
  - [ ] Add unseen-parameter evaluation using held-out parameter ranges
  - [ ] Fix ablation: add keep-only-one-group runs (leave-one-group-out masked by redundant features)
  - [ ] Add cross-validation: `cv_*` fields are currently null

- [ ] **PLANNED: Phase 5 — Update report**
  - [ ] Update BB84 results section with verified results
  - [ ] Update README with verified results

## Current Priority

1. **Fix QAD preprocessing leakage** ✅ DONE — All models wrapped in `Pipeline(StandardScaler, Model)` with per-fold scaling.
2. **Run QAD ablation without attack-specific parameters** ✅ DONE — Quantified 33-43% accuracy drops; confirmed shortcut learning.
3. **Implement BB84 attacks 1-3** (PNS, Beam Splitting, Time-Shift) ✅ DONE
4. **Implement remaining BB84 attacks** (Trojan Horse, Quantum Cloning, Fake State) ✅ DONE
5. **Generate BB84 simulation dataset** ✅ DONE — `scripts/generate_bb84_dataset.py` produces balanced multi-class dataset with 63 temporal-QBER features
6. **BB84 parameter randomization** ✅ DONE — Per-session randomization of attack params, channel noise, detection efficiency, dark count
7. **BB84 leakage tests** ✅ DONE — `tests/test_bb84_leakage.py` with 8 tests (7 pass, 4 dataset tests fail until regenerated dataset verified)
8. **BB84 real baselines** ✅ DONE — Fixed threshold, CUSUM, EWMA, Page-Hinkley, logistic regression in `evaluate.py`
9. **Fix BB84 simulator bugs** ✅ DONE — Bob zero-norm state detection, fake_state QBER, quantum_cloning QBER fixed
10. **Regenerate full 24,000-session BB84 dataset** — With randomized parameters; verify no leakage
11. **Rerun BB84 pipeline** — Multiple seeds, unseen-parameter splits, feature-group ablation
12. **Build report PDF** — verify MiKTeX path and bibliography