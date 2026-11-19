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
- [ ] Fix broken features: `key_gen_rate` is 0 for normal; `permutation_entropy` is constant
- [ ] Fix package issues: missing `.datasets` import, `tqdm` not in `pyproject.toml`

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
- [ ] **DONE: Regenerate full 24,000-session BB84 dataset** — With fixed simulator and randomized parameters (`data/bb84_run2/`)
- [ ] Run leakage tests against regenerated dataset (should pass all 13 tests)
- [ ] Run full BB84 pipeline with multiple seeds and unseen-parameter splits
- [ ] Run feature-group ablation on BB84 data (add keep-only-one-group runs)
- [ ] Add cross-validation (`cv_*` fields are currently null)

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