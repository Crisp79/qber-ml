# QBER-ML TODO

Short working queue. See [ROADMAP.md](ROADMAP.md) for milestone details.

## Immediate — Correctness Issues

- [x] **DONE: Fix QAD preprocessing leakage** — All models now wrapped in `Pipeline(StandardScaler, Model)`. Scaling fitted independently per CV fold.
- [x] **DONE: Fix SVM double-scaling** — SVM now uses same Pipeline structure; receives raw data, scaling happens inside Pipeline.
- [x] **DONE: QAD ablation without attack-specific parameters** — Completed; quantified 33-43% accuracy drops; confirmed shortcut learning.
- [ ] Run QAD ablation without attack-specific parameters (full 1M dataset)
- [ ] Run repeated-seed QAD evaluation (e.g., 5 seeds) for robustness
- [ ] Evaluate QAD on unbalanced variant

## BB84 Dataset Regeneration — Phased Approach

The current `data/bb84_run2/` dataset was generated with fixed parameters on an ideal channel and cannot be used as-is (shortcut learning: normal/PNS/trojan_horse indistinguishable; intercept_resend/time_shift both ~25% QBER; 9 constant features; `key_gen_rate` always 0). Regenerate in four phases:

### Phase 1: Fix simulator and class definitions (must complete BEFORE regenerating)
- [x] **DONE: Parameter randomization per session** — Attack params (mu, reflection_coeff, time_shift, p_z, etc.) and channel noise randomized per session; prevents shortcut learning via parameter-signature memorization
- [x] **DONE: Leakage tests** — `tests/test_bb84_leakage.py` verifies no constant features within class, no single feature perfectly predicts label, attack_params vary within class, channel_type not always ideal
- [x] **DONE**: Fix Bob: treat zero-norm (lost photon) states as non-detections (beam_splitting QBER ~0.83% → 0.0000)
- [x] **DONE**: Add channel noise, loss and detector settings to ALL classes including normal (via per-session parameter randomization in generate_bb84_dataset.py)
- [x] **DONE**: Redesign PNS and trojan_horse: simulate observable side effects or report as QBER-undetectable
- [x] **DONE**: Give time_shift a timing observable — now produces QBER ~0.25 (distinct from normal=0.0, photon_number_splitting=0.0)
- [x] **DONE**: Review fake_state: QBER exactly 50% is wrong for detector-mismatch
- [x] **DONE**: Review quantum_cloning: QBER 14.8% vs expected 16.7% (1/6)
- [x] **DONE (2026-11-21)**: Fix 5 attack classes (PNS, beam_splitting, time_shift, trojan_horse, fake_state) silently simulated with `attack_type="none"` — added `attack_type` to `ATTACK_PARAM_RANGES` + regression tests. Invalidates all earlier BB84 pilot results (see CURRENT_FINDINGS.md correction)
- [x] **DONE (2026-11-22)**: `Bob.measure_batch()` ignored zero-norm (lost) photons and recorded them as bit-0 clicks — all loss (channel, beam splitting, PNS, trojan saturation) became bit-0 bias + spurious QBER. The 2026-11-19 fix only covered `measure()`. Regression tests: `tests/test_bb84.py::TestPhotonLoss`
- [x] **DONE (2026-11-22)**: `run_bb84_simulation()` dropped `loss_probability` for `ideal` channels (half of all sessions)
- [x] **DONE (2026-11-22)**: Beam splitting / PNS drew vacuum pulses (n=0, 67–90% of draws) and passed them through, diluting attack loss ~5× → now zero-truncated Poisson (`sample_nonempty_photon_number`)
- [x] **DONE (2026-11-22)**: `data/bb84/` regenerated after the fixes below (old copy in `data/bb84_pre_simulator_fixes/`, do not use): 24,000 sessions, 49 min, 0 errors; 14/15 leakage tests pass; quick XGBoost check 84.3% accuracy / macro-F1 0.843 (paper 88.0%)
- [x] **DONE (2026-11-22)**: Bob-side detector-control hook (`AttackResult.bob_eff_scale` / `bob_spurious_click_prob` / `bob_dark_scale`, Bob's bases drawn before Eve) + redesigned attacks: time shift via detector efficiency mismatch, trojan horse reads Bob's basis and leaks probe clicks, fake state `detector_mismatch` / new `blinding`. Previously `time_shift_intercept` ≈ partial IR, trojan horse = saturation loss, 70% of fake-state sessions had no effect. Pilot: 79.1% accuracy (was 49.3%). Details in NEWCHANGES.md
- [x] **DONE (2026-11-22)**: PNS uses the standard `block` strategy (Brassard et al. 2000) instead of `pass` (no effect) / `intercept_resend` (= IR)
- [x] **DONE (2026-11-22)**: `create_attack()` now applies intercept-resend's randomized `p_z`
- [ ] Attack parameter ranges in `ATTACK_PARAM_RANGES` are modelling choices (the paper gives none) — document them in the report and test sensitivity
- [x] **DONE (2026-11-23)**: Alice is now a decoy-state weak-coherent-pulse source for every class (μ = 0.5 / ν = 0.1 / vacuum); photon numbers are tracked through Eve, channel and Bob. Stage-2 pilot: 93.9% accuracy with the 5 monitoring features (85.2% with the paper's 63). See NEWCHANGES.md entries 25–35
- [ ] Weak-attack evaluation set: generate attacks below the strength floors (beam-splitting tap < 0.15, probe clicks < 2%, phase shift < 0.6 rad) and report detection rate vs attack strength
- [ ] Report: state each attack's observable signature and that trojan horse is identified almost entirely by one monitor (`vacuum_yield` / `double_click_rate`); report results with the paper's 63 features and with the monitors separately
- [ ] Decide whether `key_gen_rate` being constant 0 for `quantum_cloning` (QBER always > 11%) is acceptable — currently fails `test_no_constant_features_within_class`
- [ ] Fix broken features: `key_gen_rate` is 0 for normal; `permutation_entropy` is constant
- [x] **DONE**: Remove dead `.datasets` import from `temporal_qber/__init__.py` (package was never committed — hidden by unanchored `datasets/` in `.gitignore`, now `/datasets/`)
- [ ] Declare `tqdm` in `pyproject.toml` (currently only installed transitively via `shap`/`kaggle`)

### Phase 2: Pilot and leakage checks (in progress)
- [ ] Generate pilot of ~200 sessions per class
- [ ] Check per-class mean/std of key features
- [ ] Check features constant overall or within a class
- [ ] Check single-feature accuracy (depth-6 decision tree per feature, 5-fold)
- [ ] Check accuracy ceiling: identical-distribution classes cap model performance
- [ ] Check correlation between training features and excluded metadata (e.g., `overall_qber`)
- [ ] Repeat Phases 1-2 until pilot looks sane

### Phase 3: Speed up generation
- [ ] Vectorize the simulator (integer bit/basis arrays instead of complex vectors)
- [ ] Shorten sessions (20-30k transmissions instead of 100k)
- [ ] Fix checkpointing: append rows per-batch instead of rewriting whole JSON
- [ ] Tune parallelism: one worker per physical core, `OMP_NUM_THREADS=1`, Pool initializer, larger chunks on Windows
- [ ] Profile with `cProfile` before further tuning
- [ ] Validate fast path reproduces reference results (normal ~0% QBER, intercept_resend ~25% QBER)

### Phase 4: Full run (24k sessions)
- [x] **DONE (2026-11-21)**: Regenerate full 24,000-session BB84 dataset (3000/class, 100k transmissions, 16 workers, 45 min) → `data/bb84/` (not committed). Command: `OMP_NUM_THREADS=1 uv run python scripts/generate_bb84_dataset.py --workers 16 --output-dir data/bb84`. Pre-fix dataset with 5 unattacked classes kept at `data/bb84_bug_attack_type_none/` (do not use)
- [x] **DONE**: Run leakage tests against regenerated dataset — 14/15 pass; fails `test_no_constant_features_within_class` (`key_gen_rate` constant 0 for `time_shift`, see Phase 1 item)
- [ ] Run full BB84 pipeline with multiple seeds and unseen-parameter splits
- [ ] Run feature-group ablation on BB84 data (add keep-only-one-group runs)
- [x] **DONE (2026-11-23)**: Cross-validation fields populated (the pipeline read them from the test evaluation instead of the training results); SVM now runs (`svm` → `svm_rbf`). Full-dataset evaluation: XGBoost 88.1% (63 features) / 97.0% (with monitors), see README
- [ ] Feature-group ablation and SHAP on the new dataset (include the monitoring group)

### Phase 5: Detection latency & novelty detection
- [ ] Add detection latency analysis (how many windows to detect attack?)
- [ ] Add open-set/novelty detection (isolation forest/autoencoder on normal windows)

### Phase 6: Report & Notebook Corrections
- [ ] Fix bibliography: replace `Anonymous` author for arXiv:2608.04047v1 with actual authors when known
- [ ] Verify all figure references in LaTeX match files in `report/figures/`
- [ ] Update report appendix package versions to match current `uv.lock`
- [ ] Ensure MiKTeX path in `generate_report.py` works on other Windows machines (consider using `pdflatex` from PATH)
- [ ] Update BB84 results section with regenerated dataset findings

## Missing Tests & Validation

- [x] **DONE: Add test verifying Pipeline is used for all models during CV** — `test_invariants.py::test_scaler_fitted_on_train_only`
- [x] **DONE: Add test for BB84 feature extraction output shape (63 features)** — `tests/test_bb84_leakage.py::TestFeatureColumnCount`
- [x] **DONE: Add test for BB84 multi-attack simulation** — `tests/test_bb84_leakage.py::TestDatasetLeakage`
- [ ] Verify PDF report builds without errors (`uv run python scripts/generate_report.py`)
- [ ] Add statistical test per attack (expected QBER for intercept-resend is 25%, partial IR gives ~p/4)
- [ ] Add seed-reproducibility tests (same seed gives byte-identical features)

## Outstanding QAD Experiments

- [ ] Run full pipeline on full 1M balanced dataset (currently only 10k in notebook, 10k in saved results)
- [ ] Evaluate QAD on unbalanced variant
- [ ] Document compute time and environment for full run

## Repo Hygiene

- [x] **DONE: Move root-level scripts** — `investigate_bug.py`, `debug_windows.py`, `convert_checkpoint.py`, `test_fix.py`, `test_final.py`, `test_new_features.py` moved to `scripts/`
- [x] **DONE: Add `.gitignore` entry for `models/random_forest.joblib`** — already ignored via `models/` and `*.joblib`
- [ ] Compact or git-ignore large `results_*/*.json` files (4–7 MB each)
- [x] **DONE: Fix `configs/bb84.yaml`** — `basis_probability_z` renamed to `alice_p_z` and `bob_p_z`
- [x] **DONE: Add CI** — `.github/workflows/ci.yml` created
- [ ] Make report build portable — call `pdflatex` from PATH or use `latexmk`/`tectonic`
- [ ] Fix `Anonymous` author in `references.bib`
- [ ] Install package properly with `uv pip install -e .` and drop `sys.path.insert` hacks in scripts

---

**Priority Order:**
1. **Fix BB84 simulator bugs** — Bob zero-norm detection, fake_state QBER, quantum_cloning QBER, key_gen_rate — DO BEFORE regenerating dataset
2. **Regenerate full 24,000-session BB84 dataset** with randomized parameters — prevents shortcut learning (run AFTER fixes)
3. **Rerun BB84 pipeline** — multiple seeds, unseen-parameter splits, ablation
4. **Speed up generation** — vectorize simulator, shorten sessions, fix checkpointing, tune parallelism
5. **Fix detection latency & novelty detection** — extend to publishable science
6. **Repo hygiene** — compact large JSONs, fix references.bib, install package properly
7. **Build report PDF** — delivery artifact