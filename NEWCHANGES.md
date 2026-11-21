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

### In progress
- Full 24,000-session regeneration into `data/bb84/` (old one moved to `data/bb84_pre_simulator_fixes/`).
