# NEWCHANGES

Log of every change made to this repository in the Claude Code sessions starting 2026-11-20.
Newest entries are at the bottom of each section; the log is updated after every change.
Entries 1–22 are committed on branch `test` (base commit `1d4844f`); see entry 22.

---

## 2026-11-20 — Package import fix

### 1. Removed the dead `.datasets` import
- **File:** `src/temporal_qber/__init__.py`
- **Change:** deleted `from .datasets import (generate_bb84_dataset, load_bb84_dataset, dataset_to_features_labels, create_bb84_datasets)` and the four matching `__all__` entries.
- **Why:** `src/temporal_qber/datasets/` was never committed, so `import temporal_qber` (and every test) failed on a fresh clone. Nothing else in the repo used these names (`scripts/run_bb84_pipeline.py` has its own `load_bb84_dataset`).
- **Verified:** `import temporal_qber` works; test suite went from failing at import to 91 passed / 11 failed (missing QAD CSV) / 6 skipped. After the QAD CSVs were placed in `data/qad/`: 102 passed, 6 skipped.

### 2. Anchored the `datasets/` ignore rule
- **File:** `.gitignore`
- **Change:** `datasets/` → `/datasets/` (with a comment).
- **Why:** the unanchored rule matched `src/temporal_qber/datasets/` and is why that package was never committed. A root-level `datasets/` folder is still ignored.

### 3. Docs
- **Files:** `TODO.md`, `ROADMAP.md` — marked the `.datasets` import fix done; split the `tqdm` dependency item out as still open.

---

## 2026-11-21 — BB84 dataset generation and the `attack_type="none"` bug

### 4. Five attack classes were never attacked
- **File:** `scripts/generate_bb84_dataset.py` (`ATTACK_PARAM_RANGES`)
- **Change:** added the missing `"attack_type"` key for `photon_number_splitting` → `pns_storage`, `beam_splitting` → `beam_splitting`, `time_shift` → `time_shift_intercept`, `trojan_horse` → `trojan_horse`, `fake_state` → `fake_state` (names match the fixed `ATTACK_CONFIGS` table and the parameters each entry randomizes), plus a comment warning that a missing key silently falls back to `"none"`.
- **Why:** `randomize_attack_params()` defaulted to `attack_type="none"`, so these 5 of 8 classes were unattacked normal sessions under an attack label. Same bug as the earlier quantum_cloning fix, which was only applied to that one class. This invalidates the "6/8 attacks indistinguishable" conclusion in `CURRENT_FINDINGS.md`.
- **Verified:** 25-sessions/class sample showed every class running its own attack.

### 5. Regression tests for the attack-type bug
- **File:** `tests/test_bb84_leakage.py`
- **Added:** `TestParameterRandomization::test_every_attack_class_runs_a_real_attack` (every non-normal class builds a real attack via `create_attack`) and `TestDatasetLeakage::test_attack_classes_were_actually_attacked` (no attack-labelled rows with `attack_type == "none"`). The dataset test fails on the buggy dataset, so it catches the bug.

### 6. Docs
- **`CURRENT_FINDINGS.md`:** dated correction banner at the top explaining the bug and that the BB84 results below it are invalid for 5 of 8 classes.
- **`TODO.md` / `ROADMAP.md`:** recorded the fix, the unused intercept-resend `p_z` parameter, the `key_gen_rate`-constant-within-class question, the full dataset run and its command, and the leakage-test outcome.

### 7. Generated data (not committed; `data/` is git-ignored)
- `data/bb84_bug_attack_type_none/` — first full 24,000-session run (5 classes unattacked). Kept for reference; do not use.
- `data/bb84/` — second full run after change 4: 24,000 sessions × 75 columns, 3,000/class, 100k transmissions, 16 workers, 45 min, 0 errors. Command: `OMP_NUM_THREADS=1 uv run python scripts/generate_bb84_dataset.py --workers 16 --output-dir data/bb84`. **Now stale** — generated before the 2026-11-22 simulator fixes below.

---

## 2026-11-22 — Making attacks distinguishable

Baseline on `data/bb84/` (XGBoost, 80/20 stratified split, seed 42, pipeline's metadata exclusions): **49.3% accuracy, macro-F1 0.49**. Only intercept-resend and quantum cloning are learnable; all top features are QBER levels and no loss feature carries attack information.

### 8. Lost photons were recorded as bit-0 clicks
- **File:** `src/temporal_qber/simulator/bob.py` (`Bob.measure_batch`)
- **Change:** a zero-norm (vacuum) state now only clicks via a dark count: `detected_photon = has_photon & (rng.random(n) < detection_efficiency)`. Random-number draws are unchanged.
- **Why:** the batch path ignored the zero-norm check, so a lost photon was "detected" with outcome 0 (its P(1) is 0). All loss — channel loss, beam splitting, PNS, trojan-horse saturation — turned into a bit-0 bias plus spurious QBER, and the detection rate was 1.000 for every class. The 2026-11-19 fix only covered `measure()`, which data generation never calls.

### 9. Channel loss was dropped for `ideal` channels
- **File:** `src/temporal_qber/simulator/bb84.py` (`run_bb84_simulation`)
- **Change:** build the channel with `create_channel` whenever `channel_params` are given, including `channel_type="ideal"`.
- **Why:** half of all sessions use the ideal channel and their randomized `loss_probability` was silently ignored.

### 10. Beam splitting and PNS acted on only 10–33% of pulses
- **Files:** `src/temporal_qber/simulator/attacks/base.py` (new `sample_nonempty_photon_number`), `attacks/beam_splitting.py`, `attacks/pns.py`
- **Change:** photon numbers are drawn from a zero-truncated Poisson(μ) (one uniform draw, inverse CDF).
- **Why:** Alice sends one qubit per transmission, but the attacks drew n = 0 from Poisson(μ = 0.1–0.4) 67–90% of the time and passed the photon through untouched, diluting the attack ~5× (a 17.5% tap gave ~4% loss).
- **Verified (clean channel):** beam splitting now gives ~16% loss and zero QBER (was 4% loss, 2% QBER, bit-0 bias).

### 11. Regression tests for changes 8–10
- **File:** `tests/test_bb84.py` — new `TestPhotonLoss`: vacuum states are never detected; vacuum only clicks via dark counts; 30% channel loss gives ~70% detection on ideal and depolarizing channels with no spurious QBER; beam splitting loss ≈ tap fraction.
- **Test run:** 114 passed, 1 failed — `test_no_constant_features_within_class`, run against the stale `data/bb84/` (`key_gen_rate` constant 0 for time_shift).

### 12. Docs
- **`TODO.md`:** recorded changes 8–10, flagged `data/bb84/` as stale, and listed the remaining modelling problems (detector side-channel attacks cannot be expressed; PNS `pass` strategy invisible; 4 of 6 fake-state subtypes have no effect).

### Result after changes 8–10
200-sessions/class pilot (scratchpad, not kept): 45.3% accuracy (noisy — 40 test sessions/class). Beam splitting F1 0.25 → 0.63, separable by loss (transmission efficiency 0.708 vs 0.834 for normal). PNS, trojan horse, time shift and fake state remain weak.

### 13. Detector side-channel hook (decision 2026-11-22: redesign PNS, trojan horse, time shift and fake state; keep the 63 features)
- **`src/temporal_qber/simulator/attacks/base.py`:** `AttackResult` gained three keyword-only fields with no-effect defaults — `bob_eff_scale` (multipliers on Bob's bit-0/bit-1 detector efficiency), `bob_spurious_click_prob` (Eve's light clicks a random detector), `bob_dark_scale` (0 = blinded detectors). `BaseAttack` gained `bob_basis`; `attack_batch(..., bob_bases=None)` sets it per transmission.
- **`src/temporal_qber/simulator/bob.py`:** `measure_batch(..., eff_scale=None, spurious_click_prob=None, dark_scale=None)` applies outcome-dependent efficiency (photon click kept with the multiplier of the detector that fired), spurious clicks (alone → random bit; with a click in the other detector → double click, random bit) and dark-count scaling.
- **`src/temporal_qber/simulator/bb84.py`:** Bob's bases are drawn before Eve acts (same RNG call `measure_batch` used to make), passed to `attack_batch` and `measure_batch`; the three hook arrays are passed to Bob only when an attack sets them.
- **Why:** time shift, faked states, detector blinding and trojan-horse probe light act on Bob's detectors, which an attack that can only change the qubit cannot express.
- **Verified:** with no hook set the simulation is unchanged — 8 sessions (normal, intercept-resend, cloning, beam splitting) regenerated bit-identical to the previous pilot; tests 114 passed, 1 known failure (stale `data/bb84/`).

### 14. PNS: standard "block single-photon pulses" strategy
- **File:** `src/temporal_qber/simulator/attacks/pns.py` — new `single_photon_strategy="block"` with `single_photon_block_prob` (both `PNSAttack` and `PNSAttackWithStorage`; the parameter is added after `seed`, so existing positional calls still work). A blocked pulse reaches Bob as vacuum. Old `pass` / `intercept_resend` strategies kept.
- **Why:** Brassard, Lütkenhaus, Mor & Sanders, PRL 85, 1330 (2000) — paper reference [8]. `pass` (80% of sessions) had no observable effect; `intercept_resend` was plain intercept-resend.

### 15. Attack factory
- **File:** `src/temporal_qber/simulator/attacks/intercept_resend.py` (`create_attack`) — passes `single_photon_block_prob` (PNS), `detector_leak_coeff` (trojan horse, both variants), `mismatch_residual` (fake state), and now applies intercept-resend's `p_z` via `set_basis_probability` for full and partial IR (it was randomized but dropped; a biased Eve gives basis-asymmetric QBER).

### 16. Trojan horse rewritten
- **File:** `src/temporal_qber/simulator/attacks/trojan_horse.py` (whole file).
- **Model (Bob-side large-pulse attack; Vakhitov, Makarov & Hjelme, J. Mod. Opt. 48, 2023 (2001); Gisin et al., PRA 73, 022320 (2006)):** Eve reads Bob's basis from back-reflected probe light (`probe_photons × back_reflection_coeff` returned photons, read correctly with `basis_inference_prob`). Read succeeded → she measures Alice's photon in Bob's basis and resends it (no error on sifted bits). Read failed → photon untouched. Observable side effects: probe light leaking into Bob's detectors gives spurious clicks with probability `1 − exp(−probe_photons × detector_leak_coeff)` (new parameter); rare detector saturation still blanks the signal. `TrojanHorseActiveAttack` now inherits the same model with brighter defaults.
- **Before:** Eve inferred the basis but never used it; the only effect was 0–10% saturation loss — identical to beam splitting.

### 17. Time shift rewritten with detector efficiency mismatch
- **File:** `src/temporal_qber/simulator/attacks/time_shift.py` (everything from `attack()` down).
- **Pure time shift (`time_shift`; Qi, Fung, Lo & Ma, QIC 7, 73 (2007); Zhao et al., PRA 78, 042333 (2008)):** per pulse Eve favours a random detector by shifting the arrival time past its efficiency peak, away from the other detector's; the photon is untouched; Bob's efficiency for each outcome comes from the detectors' Gaussian curves via `bob_eff_scale`. Signature: outcome-dependent loss, no errors.
- **With intercept (`time_shift_intercept`, used by the dataset; Makarov, Anisimov & Skaar, PRA 74, 022313 (2006)):** with probability `intercept_prob` Eve intercept-resends and times her pulse to favour the detector of the bit she resends; QBER per intercepted bit falls from 1/4 towards 1/6 as the mismatch grows, and Bob loses clicks. Signature: errors and loss together.
- **Before:** the intercept variant was plain partial intercept-resend; the pure variant drew its own "Bob basis" and never affected Bob.

### 18. Fake state: two detector-control subtypes
- **File:** `src/temporal_qber/simulator/attacks/fake_state.py` — new `_faked_state_attack()` used by `detector_mismatch` (rewritten; Makarov & Hjelme, J. Mod. Opt. 52, 691 (2005)) and new `blinding` (Lydersen et al., Nat. Photonics 4, 686 (2010)). Eve intercept-resends; Bob clicks only when he measures in Eve's basis — outside it his detectors keep a residual efficiency `mismatch_residual` (new parameter) for `detector_mismatch`, or none and no dark counts for `blinding`. Signature: about half of Bob's clicks disappear, errors only from residual clicks. `phase_remapped` unchanged; `wavelength`, `intensity`, `time_bin`, `combined` kept but documented as legacy.
- **Before:** 4 of 6 subtypes (70% of fake-state sessions) passed Alice's photon unchanged.

### 19. Dataset generator parameter ranges
- **File:** `scripts/generate_bb84_dataset.py` (`ATTACK_PARAM_RANGES`, `randomize_attack_params`)
- PNS: strategy `block` only, `block_prob_range` (0.2, 0.7).
- Time shift: `time_shift_range` (0.05, 0.2), `det_width_range` (0.25, 0.5), new `det_separation_range` (0.15, 0.35) passed as `det1_peak_time` (efficiency ratio disfavoured/favoured ≈ 0.12–0.93); `intercept_prob_range` unchanged (0.5, 1.0). Old values (0.3–0.8 shift, 0.5–1.5 width, fixed 0.2 separation) gave a ratio of ~0.8–0.9, i.e. almost no mismatch.
- Trojan horse: new `detector_leak_coeff_range` (5e-6, 4e-5) → spurious-click probability ≈ 0.003–0.18; `detector_saturation_prob_range` lowered from (0, 0.1) to (0, 0.02) so saturation loss does not mimic beam splitting.
- Fake state: subtypes `phase_remapped` / `detector_mismatch` / `blinding` (0.34/0.33/0.33), new `mismatch_residual_range` (0.0, 0.3).
- All ranges are modelling choices, not values taken from the paper (which does not specify its attack models).

### Signatures after changes 13–19 (clean channel: ideal, efficiency 1, no dark counts; 4 sessions/class × 20k transmissions)
| class | detection rate | QBER | distinctive |
|---|---|---|---|
| normal | 1.000 | 0.000 | — |
| beam splitting | 0.839 | 0.000 | loss ≈ tap fraction |
| PNS | 0.580 | 0.000 | heavy loss |
| time shift | 0.845 | 0.187 | loss and errors together |
| trojan horse | 0.992 | 0.015 | small error rate from probe clicks |
| fake state | 0.874 ± 0.25 | 0.155 (Z 0.000 / X 0.313) | ~50% loss or Z/X asymmetry, by subtype |
| intercept-resend | 1.000 | 0.241 (Z 0.226 / X 0.255) | basis bias now active |
| quantum cloning | 1.000 | 0.157 | — |

Tests after 13–19: 114 passed, 1 known failure (stale `data/bb84/`).

### 20. Tests for the detector hook and new attack models
- **File:** `tests/test_bb84.py` — new `TestDetectorSideChannels` (9 tests): `eff_scale` blinds one detector; spurious clicks give random bits and `dark_scale=0` removes dark counts; PNS block = loss 0.5 × P(n=1|n≥1) with no errors; pure time shift = loss without errors; time shift + intercept gives QBER 0.15–0.23 (below 0.25) plus loss; trojan horse with a successful basis read is error-free; probe light adds QBER ≈ p_click/4; blinding halves clicks with no errors and no dark counts; detector-mismatch residual clicks give QBER ≈ 0.083. Expected ranges derived analytically from each model.
- **Test run:** 123 passed, 1 known failure (stale `data/bb84/`).

### Result after changes 13–20 — pilot (400 sessions/class, 100k transmissions, scratchpad, not kept)
Same XGBoost check (80/20 stratified, seed 42, 63 features): **79.1% accuracy, macro-F1 0.792** (was 49.3% / 0.49 on `data/bb84/`).

| class | F1 before (data/bb84) | F1 after (pilot) |
|---|---|---|
| quantum cloning | 0.705 | 0.902 |
| intercept-resend | 0.763 | 0.893 |
| time shift | 0.566 | 0.870 |
| fake state | 0.427 | 0.829 |
| PNS | 0.258 | 0.750 |
| trojan horse | 0.383 | 0.740 |
| normal | 0.367 | 0.686 |
| beam splitting | 0.467 | 0.667 |

Remaining confusions: PNS ↔ beam splitting (both pure loss; also the paper's main confusion) and trojan horse ↔ normal (the attack is designed to be nearly invisible). Loss features (`qber_loss_interaction`, `transmission_efficiency`, `loss_burst_corr`) are now among the top features.

Leakage checks on the pilot: best single-feature one-vs-rest AUC per class 0.81–0.95 (highest: intercept-resend via `rect_burst_energy`); best single feature alone gives 48% 8-class accuracy. Constant within a class: only `key_gen_rate` (= 0) for time shift and quantum cloning, whose QBER is always > 11%. Constant overall: `permutation_entropy` (known broken), `afterpulse_prob` (always 0 by design).

### 21. Docs
- **`TODO.md`:** marked the side-channel hook, attack redesign, PNS strategy and `p_z` items done; added two open items (document/sensitivity-test the attack parameter ranges; Alice is a single-photon source, so μ exists only inside the PNS and beam-splitting attacks).
- **`CURRENT_FINDINGS.md`:** second correction note (2026-11-22) listing the simulator bugs and the pilot result.

### 22. Committed
- All changes in entries 1–21 (17 files plus this log) committed on branch `test` on top of `7d6a5b9` (which added this file). `data/` is git-ignored, so no datasets are included.
- Push from Claude's shell failed (no GitHub credentials there); the user pushed it to `origin/test`.
- Commit message amended to drop the `Co-Authored-By: Claude` trailer at the user's request. This rewrites the already-pushed commit, so GitHub is updated with a force push.

### 23. Full BB84 dataset regenerated with all fixes (not committed; `data/` is git-ignored)
- `data/bb84/` — 24,000 sessions × 75 columns, 3,000/class, 100k transmissions, 16 workers, 49 min, 0 errors, no nulls or duplicates; every class runs its own attack. Same command as entry 7. Previous dataset moved to `data/bb84_pre_simulator_fixes/` (do not use).
- Per-class means: QBER — normal 0.026, beam splitting 0.029, PNS 0.036, trojan horse 0.046, fake state 0.125, time shift 0.190, quantum cloning 0.195, intercept-resend 0.239. Transmission efficiency — PNS 0.514, fake state 0.586, time shift 0.692, beam splitting 0.707, all others ≈ 0.84.
- Leakage tests (`tests/test_bb84_leakage.py`): 14 passed, 1 failed — `key_gen_rate` constant (0) within `time_shift`, whose QBER is always > 11%. `key_gen_rate` is computed from mean QBER only, so this is a QBER-threshold effect, not label leakage; still open in TODO.md.
- XGBoost check (80/20 stratified, seed 42, 63 features): **84.3% accuracy, macro-F1 0.843** (49.3% / 0.49 on the 2026-11-21 dataset; paper: 88.0% / 0.880). Per-class F1: intercept-resend 0.931, time shift 0.924, quantum cloning 0.916, fake state 0.886, trojan horse 0.843, PNS 0.785, normal 0.746, beam splitting 0.712. Main confusions: PNS ↔ beam splitting, normal ↔ trojan horse, normal ↔ beam splitting. Top features: `key_gen_rate`, `multi_scale_qber_8`, `rect_burst_energy` (the paper's SHAP ranking also puts `key_gen_rate` first).
- This is a single quick model check, not the full pipeline (`scripts/run_bb84_pipeline.py`, whose CV fields are still empty because of the `cv_accuracy` / `cv_accuracy_mean` key mismatch).

### 24. Docs
- **`TODO.md`, `ROADMAP.md`:** recorded the regenerated dataset and its results.

---

## 2026-11-23 — Stage 1: making the remaining attacks identifiable

Diagnosis on `data/bb84/` (5-fold out-of-fold XGBoost, parameters rebuilt from each session's seed): errors come from (a) attacks too weak to leave a trace — beam-splitting tap ≤ 0.1 (recall 0.43), trojan probe-click probability ≤ 0.02 (0.60), phase-remapping shift < 0.6 rad (0.03–0.12) — and (b) attacks whose only observable is the same: PNS / beam splitting / blinding / detector mismatch all look like "photons missing, no errors", and channel noise looks like trojan-horse probe clicks (normal sessions on noisy channels are classified correctly only 54–70% of the time). Missed as normal: beam splitting 13%, trojan 7%, fake state 4%, PNS and time shift 0%; normal false-alarm rate 24%. Plan agreed with the user: stage 1 = minimum attack strengths + double-click and empty-gate monitors; stage 2 = decoy states; report results with the paper's 63 features and with the monitors.

### 25. Bob: two-detector click model with double clicks and an empty monitoring gate
- **File:** `src/temporal_qber/simulator/bob.py` (`measure_batch`, `BobMeasurement`)
- **Change:** each slot's photon, dark count and spurious light now fire a specific detector (photon: the detector its outcome points to; dark count and probe light: a random one). If both detectors fire it is a **double click**: recorded in the new `BobMeasurement.is_double_click` and resolved with a random bit (squashing). Each slot also has an **empty monitoring gate** that only dark counts can fire (`empty_gate_click`), i.e. an in-situ dark-count measurement that drops to zero when detectors are blinded.
- **Physics change for all classes:** previously a dark count overrode the photon and gave a random bit (50% error); now a dark count in the same detector as the photon changes nothing and one in the other detector gives a double click (random bit), so dark counts coinciding with a photon cause ~25% errors instead of 50%. Random draws change, so new datasets are not bit-identical to earlier ones.
- **File:** `src/temporal_qber/simulator/bb84.py` — `BB84Result` gains optional `bob_double_click` and `bob_empty_gate_click` arrays, filled by `run()`.
- **Verified:** `tests/test_bb84.py` 59 passed. 50k-transmission check (90% efficiency, 10% loss, 4% depolarizing): double clicks per click — normal 0.0044 (dark 0.01) / 0.0194 (dark 0.04), trojan (p_click ≈ 0.05) 0.0285 / 0.0426; empty-gate rate ≈ dark-count probability for every case except blinding (0.0000).

### 26. Detector-monitoring features (separate from the paper's 63)
- **File:** `src/temporal_qber/temporal/features.py` — new `MONITORING_FEATURES = ["double_click_rate", "empty_gate_click_rate"]` and `extract_monitoring_features(bob_detected, bob_double_click, bob_empty_gate_click)`: fraction of Bob's clicks that are double clicks, and the click rate in the empty monitoring gate. Exported from `temporal_qber.temporal`. The 63-feature list and its assertion are unchanged.

### 27. Dataset generator: monitors + minimum attack strengths
- **File:** `scripts/generate_bb84_dataset.py`
- Every session now also gets the two monitoring columns (datasets go from 75 to 77 columns).
- Beam splitting `reflection_coeff_range` (0.05, 0.3) → **(0.15, 0.3)**.
- Trojan horse: `detector_leak_coeff_range` replaced by **`probe_click_prob_range` (0.02, 0.18)**; the leak coefficient is derived as −ln(1−p)/probe_photons (was 0.003–0.18 via independent draws).
- Fake state `phase_shift_range` (0, π) → **(0.6, π)**.
- Why: below these floors the attacks hide inside normal channel loss / dark counts (diagnosis above). They should go into a separate weak-attack evaluation set, not be dropped from the study (still to do).

### 28. Pipeline flag for the monitors
- **File:** `scripts/run_bb84_pipeline.py` — `prepare_features_targets(..., use_monitoring=False)`, `run_pipeline(..., monitor_features=False)` and a `--monitor-features` CLI flag. Default stays the paper's 63 features.

### 29. Tests
- **`tests/test_bb84.py`:** new `TestDetectorMonitoring` — probe light on a photon gives double clicks half the time with random bits; no double clicks from channel noise alone; empty gate measures the dark-count rate and reads zero when blinded; `extract_monitoring_features` values.
- **`tests/test_bb84_leakage.py`:** `test_minimum_attack_strengths` (the three floors); `test_feature_count_is_63` now excludes the monitoring columns.
- **Test run:** 128 passed, 1 known failure (`key_gen_rate` constant within time shift, on the 2026-11-22 `data/bb84/`).

### Result after changes 25–29 — stage-1 pilot (400 sessions/class, scratchpad, not kept; 7 min, 0 errors)
XGBoost, 80/20 stratified split, seed 42 (80 test sessions/class, so ±~3 points of noise):

| | 63 features | 63 + monitors |
|---|---|---|
| accuracy / macro-F1 | 0.850 / 0.849 | **0.887 / 0.887** |
| normal false-alarm rate | 0.138 | 0.025 |
| missed as normal | beam splitting 0.025, trojan 0.100, others 0 | beam splitting 0.025, trojan 0.037, others 0 |

Per-class F1 (63 → 63+monitors): trojan horse 0.894 → 0.981, normal 0.868 → 0.957, intercept-resend 0.911 → 0.924, quantum cloning 0.907 → 0.914, fake state 0.840 → 0.894, time shift 0.852 → 0.852, beam splitting 0.823 → 0.821, PNS 0.698 → 0.755. For reference: 0.791 accuracy on the previous pilot, 0.843 on the full 2026-11-22 dataset, 0.880 in the paper.

Remaining confusion: PNS ↔ beam splitting (21 of 80 PNS predicted beam splitting) — needs stage 2 (decoy states); fake state (blinding / mismatch) → PNS 7 of 80.

Leakage check (65 features): best single-feature one-vs-rest AUC 0.82–0.95 for every class except trojan horse, where `double_click_rate` alone gives **0.991** — the modelled physical signature (probe light firing the second detector), to be stated in the report. Each monitor alone gives weak 8-class accuracy (double clicks 0.30, empty gate 0.17). Constant within class: only `key_gen_rate` for time shift and quantum cloning.

---

## 2026-11-23 — Stage 2: decoy-state source

### 30. Attack framework: photon numbers
- **`src/temporal_qber/simulator/attacks/base.py`:** `AttackResult.photons_to_bob` (keyword-only, None = unchanged) and `BaseAttack.photon_number`, set per pulse by `attack_batch(..., photon_numbers=None)`.
- **`attacks/pns.py`, `attacks/beam_splitting.py`:** `_sample_photon_number()` returns Alice's actual photon number when it is given (decoy mode), otherwise samples as before. PNS reports `photons_to_bob` = n − 1 for multi-photon pulses (Eve keeps one) and 0/1 for blocked/passed single photons; beam splitting reports the photons not tapped.

### 31. Channel: per-photon loss
- **`src/temporal_qber/simulator/channel.py`:** new `QuantumChannel.transmit_batch_photons(states, photon_numbers)` — noise acts on the pulse's polarisation state, loss on each photon independently (Binomial(n, 1 − loss)); returns states and surviving photon numbers.

### 32. Bob: multi-photon threshold detection
- **`src/temporal_qber/simulator/bob.py`:** `measure_batch(..., photon_numbers=None)`. With photon numbers each photon is routed to a detector by the measurement and detected independently (efficiency × mismatch multiplier); a detector fires if any of its photons is detected. Without them the single-photon path is unchanged (same random draws).

### 33. Protocol: optional decoy-state source
- **`src/temporal_qber/simulator/bb84.py`:** `BB84Protocol(..., decoy_config=None)` / `run_bb84_simulation(..., decoy_config=None)`, e.g. `{"intensities": [0.5, 0.1, 0.0], "probabilities": [0.7, 0.2, 0.1]}`. Each pulse gets an intensity index (own `_decoy` RNG) and Poisson photon number; photon numbers flow through Eve (`photons_to_bob`; empty pulses forward nothing), the channel (per-photon loss) and Bob. Only signal pulses (index 0) form the sifted key and `loss_seq`; `sifting_efficiency` is per signal pulse. `BB84Result` gains `pulse_intensity`, `photon_numbers`, `decoy_intensities`. Default `None` = original single-photon model.
- **Verified:** tests 128 passed, 1 known failure (unchanged). 100k-pulse check (efficiency 0.9, dark 0.01, 10% loss): vacuum yield 0.0096 normally, 0.0000 under blinding, 0.0847 under trojan-horse probe light; decoy/signal gain ratio 0.250 normal, 0.253 beam splitting (R = 0.25), 0.256 PNS (b = 0.3), 0.237 PNS (b = 0.6) — the raw ratio barely separates PNS from beam splitting at this channel loss.

### 34. Decoy-state monitoring features + generator uses the decoy source
- **`src/temporal_qber/temporal/features.py`:** `MONITORING_FEATURES` now also has `decoy_gain_ratio` (Q_ν/Q_μ), `vacuum_yield` (Y0) and `single_photon_fraction` (lower bound on the fraction of signal clicks from single photons, Y1ᴸ·μe^−μ/Q_μ, vacuum + weak decoy bound of Ma, Qi, Zhao & Lo, PRA 72, 012326 (2005)); NaN without a decoy source. `extract_monitoring_features(..., pulse_intensity=None, intensities=None)`.
- **`scripts/generate_bb84_dataset.py`:** new `DECOY_CONFIG` (μ = 0.5 / ν = 0.1 / vacuum, probabilities 0.7 / 0.2 / 0.1) used for every session; the monitors get the pulse intensities; beam splitting and PNS no longer draw `mean_photon_number` (they act on Alice's real pulses).
- **`tests/test_bb84_leakage.py`:** `test_pns_mu_varies` → `test_pns_block_prob_varies`.

### 35. Generator fixes after the first stage-2 pilot failed
- **First stage-2 pilot:** every session failed with `bob_detected must match sifted_alice length` and the run never finished (14,020 "completions" of 3,200 after 20 min); stopped by hand.
- **Cause 1 — `scripts/generate_bb84_dataset.py` (`generate_session_task`):** Bob's detections passed to the feature extractor were filtered by matching basis only, while the key is now signal pulses only. Now filtered with the same mask as the key (matching basis, detected, signal pulse).
- **Cause 2 — same file (`generate_dataset`, parallel loop):** the next task was chosen as `len(completed_keys) + len(pending)`, which lags after every failed session, so failures resubmitted old tasks forever. Now a `next_submit` counter walks `all_tasks` once; the loop sleeps 50 ms when nothing is ready instead of busy-spinning a core.
- **Also fixed:** `session_results = {}` after loading a checkpoint discarded the sessions already completed, so a resumed run silently dropped them from the final CSV (flagged 2026-11-21). The checkpointed results are now kept.
- **Verified:** smoke test 5 sessions/class × 100k: 40 sessions, 0 failures, 1.7 s/session; tests 128 passed, 1 known failure.

### Result after changes 30–35 — stage-2 pilot (400 sessions/class, 100k pulses, scratchpad, not kept; 6 min, 0 failures)
XGBoost, 80/20 stratified split, seed 42 (80 test sessions/class, ±~3 points):

| | 63 features | 63 + 5 monitors | + calibrated channel transmittance (experiment) |
|---|---|---|---|
| accuracy / macro-F1 | 0.852 / 0.851 | **0.939 / 0.939** | 0.938 / 0.937 |
| normal false-alarm rate | 0.062 | 0.075 | 0.050 |

Per-class F1 with monitors (stage-1 pilot → stage 2): PNS 0.755 → **0.988**, beam splitting 0.821 → 0.931, fake state 0.894 → 0.962, trojan horse 0.981 → 0.994, normal 0.957 → 0.919, time shift 0.852 → 0.919, intercept-resend 0.924 → 0.897, quantum cloning 0.914 → 0.901. No PNS ↔ beam-splitting confusion left. Remaining confusions: intercept-resend ↔ quantum cloning (8 of 80) and normal ↔ beam splitting (5–6 of 80). Knowing the calibrated channel transmittance adds nothing, so it is not used.

Leakage check (68 features): no NaN monitors; best single-feature one-vs-rest AUC 0.81–0.96 for every class except trojan horse (`vacuum_yield` 0.990 — probe light clicking in vacuum pulses, the modelled signature). PNS is separated by a combination: `double_click_rate` (AUC 0.910; Eve keeps one photon of every multi-photon pulse, which removes most double clicks — 0.021 vs 0.045 beam splitting, 0.057 normal), `single_photon_fraction` (0.585 vs 0.669 / 0.702) and `transmission_efficiency`. Each monitor alone gives 17–43% 8-class accuracy. Constant within class: only `key_gen_rate` (time shift, quantum cloning).

### 36. Docs
- **`TODO.md`:** Alice single-photon-source item done; open items for the weak-attack evaluation set and for stating the per-attack signatures in the report.

### Next
- Full 24,000-session regeneration with stages 1 + 2.
- Weak-attack evaluation set (attack strengths below the new floors) for a detection-vs-strength curve.
- Weak-attack evaluation set (attack strengths below the new floors) for a detection-vs-strength curve.
