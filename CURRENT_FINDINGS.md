# QBER-ML BB84 Attacks - Current Findings

**Date**: 2026-11-19  
**Branch**: feature/bb84-attacks  
**Commit**: e566b521  

> **Current results (2026-11-23) — supersede everything below.** Full dataset `data/bb84/` (24,000
> sessions, decoy-state source, redesigned attacks, minimum attack strengths), 80/20 stratified split,
> 5-fold CV. XGBoost: **88.1% test accuracy (CV 88.4%) with the paper's 63 features**, **97.0% (CV
> 96.6%) with the 5 monitoring features added**; RF 85.5% / 94.8%, SVM-RBF 86.0% / 95.1%. With
> monitors every class reaches F1 ≥ 0.93 (PNS 0.996, trojan horse 0.995, fake state 0.992, beam
> splitting 0.983, normal 0.978, time shift 0.952, intercept-resend 0.943, quantum cloning 0.926);
> 0.3% of attacks are missed and 2.2% of normal sessions are flagged. Full table in `README.md`;
> outputs in `results/bb84_stage2_63/` and `results/bb84_stage2_monitors/`; every change that led
> here is in `NEWCHANGES.md`. The two corrections below explain why the older analysis is invalid.

> **Correction (2026-11-21): the BB84 results below are invalid for 5 of 8 classes.**
> `ATTACK_PARAM_RANGES` in `scripts/generate_bb84_dataset.py` had no `"attack_type"` for
> `photon_number_splitting`, `beam_splitting`, `time_shift`, `trojan_horse` and `fake_state`, so
> `randomize_attack_params()` fell back to `"none"`: those sessions were unattacked normal sessions
> under an attack label (the `attack_type` column of every affected dataset reads `none`). This is the
> same bug fixed for `quantum_cloning` in change 5 below, which was only fixed for that one class.
> The "6/8 attacks indistinguishable" result and the root-cause analysis are therefore artefacts of
> the bug, not evidence about passive attacks. After the fix, a 25-sessions-per-class sample gives mean
> QBER: normal 0.051, PNS 0.043, beam_splitting 0.055, trojan_horse 0.075, fake_state 0.075,
> time_shift 0.216, quantum_cloning 0.201, intercept_resend 0.243. Results must be re-derived from the
> regenerated `data/bb84/`. Regression tests: `test_every_attack_class_runs_a_real_attack`,
> `test_attack_classes_were_actually_attacked` in `tests/test_bb84_leakage.py`.
>
> **Second correction (2026-11-22):** three further simulator bugs also shaped every result below:
> `Bob.measure_batch()` counted lost photons as bit-0 clicks (so no attack or channel ever caused
> loss), `loss_probability` was ignored on ideal channels, and beam splitting / PNS skipped 67–90%
> of pulses. Several attack models also had no observable effect (PNS `pass`, 70% of fake-state
> sessions) or duplicated intercept-resend (time shift). After the fixes and the attack redesign
> (see `NEWCHANGES.md`, entries 8–20) a 400-sessions/class pilot reaches 79.1% accuracy / macro-F1
> 0.79 with XGBoost on the same 63 features, with every class at F1 ≥ 0.67.

---

## System Details

| Component | Version/Details |
|-----------|-----------------|
| Python | 3.12.14 |
| uv | Latest (manages dependencies) |
| OS | Linux |
| Key Packages | numpy, pandas, scikit-learn, xgboost, scipy, shap, pytest |
| GPU | Not used (CPU-only training) |

---

## Changes Made in This Session

### 1. Fixed Bob.measure_batch() Not Storing Measurements
**File**: `src/temporal_qber/simulator/bob.py:180`  
**Issue**: `measure_batch()` returned measurements but didn't append to `self._measurements`, causing `get_sifted_key()` to fail with "measurements 0".  
**Fix**: Added `self._measurements.extend(measurements)` before return.

### 2. Fixed Leakage Test Over-Sensitivity
**File**: `tests/test_bb84_leakage.py`  
**Issue**: Two tests flagged false positives:
- `test_no_constant_features_within_class`: Flagged features constant across ALL classes (e.g., `permutation_entropy=0`) as leakage
- `test_no_single_feature_perfectly_predicts_label`: Used exact FP comparison causing false positives  
**Fix**: 
- Only flag features constant in ONE class but varying in others (actual shortcut features)
- Use binned/rounded values for continuous features

### 3. Fixed Feature Extraction Hardcoded Detector Parameters
**File**: `src/temporal_qber/temporal/features.py` (3 locations)  
**Issue**: `detector_efficiency`, `dark_count_rate`, `afterpulse_prob` hardcoded to 1.0, 0.0, 0.0 instead of using session parameters.  
**Fix**: Pass parameters through to `_extract_channel_features()` and use them.

### 4. Added CLI Options for BB84 Pipelines
**Files**: `scripts/generate_bb84_dataset.py`, `scripts/run_bb84_pipeline.py`  
**Added**: Consistent `--input-dir`/`--in-dir` and `--output-dir`/`--out-dir` aliases.

### 5. Fixed Quantum Cloning Attack Randomization Bug
**File**: `scripts/generate_bb84_dataset.py:261`  
**Issue**: `ATTACK_PARAM_RANGES["quantum_cloning"]` missing `"attack_type": "quantum_cloning"`, causing all sessions to use `attack_type="none"`.  
**Fix**: Added missing attack_type field.

### 6. Fixed Checkpoint Resume Bug
**File**: `scripts/generate_bb84_dataset.py:702`  
**Issue**: When generating to new output directory with `--input-dir` for checkpoint resume, it loaded checkpoint from input dir but used wrong session keys.  
**Fix**: Only load checkpoint if `input_dir == output_dir` (continuing same generation).

---

## Dataset Generation Results

### Pilot Datasets Generated

| Dataset | Sessions/Class | Transmissions | Total | Status |
|---------|----------------|---------------|-------|--------|
| bb84_pilot | 200 | 10,000 | 1,600 | Old (leaky) |
| bb84_pilot2 | 300 | 50,000 | 2,400 | Old (quantum cloning broken) |
| bb84_pilot2_fixed | 300 | 50,000 | 2,400 | ✅ Fixed QC |
| bb84_100k_pilot | 100 | 100,000 | 800 | ✅ Fixed QC |
| bb84_pilot3 | 5 | 10,000 | 40 | Debug |
| bb84_pilot4 | 30 | 10,000 | 240 | Debug |

### Data Quality Checks (All Fixed Datasets Pass)
- ✅ Channel types randomized (4 types: ideal, depolarizing, bit_flip, phase_flip)
- ✅ Attack parameters randomized per session (300 unique for most classes)
- ✅ Detector efficiency varies (0.80-1.00)
- ✅ Dark count probability varies (0.0-0.05)
- ✅ No leakage: QBER distributions overlap, no perfect separation
- ✅ Quantum cloning: 4 cloning types appear (universal, phase_covariant, state_dependent, asymmetric)

---

## ML Pipeline Results

### 50k Transmissions (bb84_pilot2_fixed)

| Model | Accuracy | Macro-F1 | ROC-AUC | Classes >50% F1 |
|-------|----------|----------|---------|-----------------|
| Random Forest | 35.0% | 35.0% | 0.728 | 2/8 |
| XGBoost | 33.1% | 33.1% | 0.725 | 2/8 |
| Logistic Regression | 51.3% | 50.9% | - | - |

**Per-Class F1 (XGBoost)**:
- intercept_resend: 86% ✅
- quantum_cloning: 88% ✅
- fake_state: 34% ⚠️
- beam_splitting: 26% ❌
- photon_number_splitting: 16% ❌
- trojan_horse: 11% ❌
- time_shift: 13% ❌
- normal: 18% ❌

### 100k Transmissions (bb84_100k_pilot)

| Model | Accuracy | Macro-F1 | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | 35.0% | 34.5% | 0.734 |
| XGBoost | 38.1% | 37.9% | 0.741 |
| Logistic Regression | 62.5% | 62.6% | - |

**Per-Class F1 (XGBoost)**:
- intercept_resend: 95% ✅
- quantum_cloning: 95% ✅
- fake_state: 34% ⚠️
- beam_splitting: 26% ❌
- time_shift: 19% ❌
- photon_number_splitting: 6% ❌
- trojan_horse: 10% ❌
- normal: 18% ❌

### QBER Distribution Analysis (100k)

```
normal:                   0.00 - 0.14 (mean 0.046, std 0.036)
beam_splitting:           0.00 - 0.12 (mean 0.047, std 0.036)  ← Overlaps normal
fake_state:               0.00 - 0.12 (mean 0.047, std 0.038)  ← Overlaps normal
photon_number_splitting:  0.00 - 0.12 (mean 0.044, std 0.034)  ← Overlaps normal
time_shift:               0.00 - 0.12 (mean 0.044, std 0.034)  ← Overlaps normal
trojan_horse:             0.00 - 0.10 (mean 0.038, std 0.032)  ← Overlaps normal
intercept_resend:         0.10 - 0.31 (mean 0.250, std 0.043)  ✅ SEPARABLE
quantum_cloning:          0.15 - 0.43 (mean 0.210, std 0.077)  ✅ SEPARABLE
```

---

## Root Cause Analysis

### Why 6/8 Attacks Are Indistinguishable

1. **Passive attacks don't raise QBER significantly**: Beam splitting, PNS, time-shift, trojan horse, fake_state (detector mismatch) introduce QBER indistinguishable from normal channel noise + detector imperfections at 100k transmissions.

2. **Active attacks ARE detectable**: 
   - **Intercept-resend**: Eve measures and resends → ~25% QBER theoretical, ~10-31% observed (separable)
   - **Quantum cloning**: Imperfect clones sent to Bob → ~17% QBER theoretical, ~15-43% observed (separable)

3. **Aggregated features insufficient**: The 63 temporal-QBER features (mean, var, spectral, entropy, etc.) are statistics of overlapping distributions. No amount of feature engineering on aggregated stats can separate classes with identical QBER distributions.

4. **Logistic Regression outperforms trees**: Handles correlated noisy features better; trees overfit to noise in high-dimensional aggregated feature space.

### Feature Importance (SHAP)
- Top features for XGBoost: `diag_burst_energy`, `rect_burst_energy`, `lag2_autocorr`, `detector_efficiency`, `dark_count_rate`
- Top features for RF: `rect_burst_energy`, `diag_burst_energy`, `spectral_centroid`, `spectral_bandwidth`
- Detector parameters (efficiency, dark count) appear in top 10 → **potential leakage risk** if not properly randomized (they ARE randomized)

---

## Test Results

All **108 tests pass** after fixes:
```
tests/test_bb84.py: 44 passed
tests/test_bb84_leakage.py: 13 passed
tests/test_channel.py: 24 passed
tests/test_invariants.py: 11 passed
tests/test_logging_progress_experiment.py: 11 passed
```

---

## Recommendations

### Do NOT Run Full Pipeline (3000 sessions × 100k)
- Will take ~4 hours
- Will produce same results: only 2/8 classes detectable
- Waste of compute

### Option 1: Pivot to Temporal Sequence Modeling
- Use per-window QBER sequences (not aggregated features)
- Feed to LSTM / 1D-CNN / Temporal Transformer
- Paper methodology: "Temporal QBER" implies sequence modeling

### Option 2: Binary Active Attack Detection
- Task: intercept_resend vs quantum_cloning vs normal
- Achieves 95%+ F1
- Practical for detecting active eavesdropping

### Option 3: Adjust Attack Parameters
- Check if paper uses different parameters (higher reflection_coeff, different mu, etc.)
- Current parameters may be too conservative

### Option 4: Add Side-Channel Features
- Timing jitter, detector efficiency mismatch, afterpulsing
- Paper mentions "detector side-channels" for fake_state attack

---

## Files Modified

```
src/temporal_qber/simulator/bob.py                          # Bob.measure_batch fix
src/temporal_qber/temporal/features.py                       # Detector params fix (3 locations)
scripts/generate_bb84_dataset.py                             # QC attack_type fix, checkpoint fix, CLI options
scripts/run_bb84_pipeline.py                                 # CLI options
tests/test_bb84_leakage.py                                   # Leakage test fixes (2 tests)
```

---

## Commands for Reproduction

```bash
# Quick test pipeline (QAD track)
uv run python scripts/test_pipeline.py

# Generate fixed pilot dataset (50k)
uv run python scripts/generate_bb84_dataset.py --n-sessions-per-class 300 --n-transmissions 50000 --output-dir data/bb84_pilot2_fixed --workers 8

# Generate 100k pilot
uv run python scripts/generate_bb84_dataset.py --n-sessions-per-class 100 --n-transmissions 100000 --output-dir data/bb84_100k_pilot --workers 4

# Run BB84 pipeline
uv run python scripts/run_bb84_pipeline.py --input-dir data/bb84_100k_pilot/bb84_temporal_dataset.csv --output-dir results/bb84_100k_pilot --models random_forest,xgboost

# Run all tests
uv run pytest tests/ -v
```

---

## Conclusion

The BB84 temporal-QBER approach with **aggregated statistical features** cannot distinguish passive eavesdropping attacks from normal operation at practical transmission counts (50k-100k). Only active attacks (intercept-resend, quantum cloning) are detectable via QBER.

**Path forward**: Implement temporal sequence modeling (LSTM/1D-CNN on per-window QBER sequences) as implied by the paper's "Temporal QBER" methodology, or focus on binary active-attack detection.