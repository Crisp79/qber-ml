# QBER-ML TODO

Short working queue. See [ROADMAP.md](ROADMAP.md) for milestone details.

## Immediate — Correctness Issues

- [x] **DONE: Fix QAD preprocessing leakage** — All models now wrapped in `Pipeline(StandardScaler, Model)`. Scaling fitted independently per CV fold. No more external scaler fitted on full train set.
- [x] **DONE: Fix SVM double-scaling** — SVM now uses same Pipeline structure; receives raw data, scaling happens inside Pipeline.
- [ ] **Run QAD ablation without attack-specific parameters**: Create feature set excluding `swap_intensity`, `dp_error_rate`, `mt_flip_prob`, `ad_damping_param`, `has_noise_model`, etc. Retrain and evaluate to quantify shortcut learning.

## Missing Tests & Validation

- [ ] Add test verifying `Pipeline` is used for all models during CV (scaler fitted per fold)
- [ ] Add test for BB84 temporal feature extraction output shape (63 features)
- [ ] Add test for BB84 multi-attack simulation (when attacks implemented)
- [ ] Verify PDF report builds without errors (`uv run python scripts/generate_report.py`)

## Outstanding QAD Experiments

- [ ] Run full pipeline on full 1M balanced dataset (currently only 10k in notebook, 10k in saved results)
- [ ] Run repeated-seed evaluation (e.g., 5 seeds) for robustness
- [ ] Evaluate on unbalanced QAD variant
- [ ] Document compute time and environment for full run

## Report & Notebook Corrections

- [ ] Fix bibliography: replace `Anonymous` author for arXiv:2608.04047v1 with actual authors when known
- [ ] Verify all figure references in LaTeX match files in `report/figures/`
- [ ] Update report appendix package versions to match current `uv.lock`
- [ ] Ensure MiKTeX path in `generate_report.py` works on other Windows machines (consider using `pdflatex` from PATH)

## Next BB84 Implementation Tasks

1. **Implement PNS attack** (`simulator/attacks/pns.py`) — photon number splitting with multi-photon pulse handling
2. **Implement Beam Splitting attack** (`simulator/attacks/beam_splitting.py`) — passive tapping with beam splitter
3. **Implement Time-Shift attack** (`simulator/attacks/time_shift.py`) — detector efficiency mismatch exploitation
4. **Implement Trojan Horse attack** (`simulator/attacks/trojan_horse.py`) — back-reflection probing
5. **Implement Quantum Cloning attack** (`simulator/attacks/quantum_cloning.py`) — approximate cloning machine
6. **Implement Fake State attack** (`simulator/attacks/fake_state.py`) — tailored quantum states
7. **Create BB84 dataset generation script** (`scripts/generate_bb84_dataset.py`) — multi-session, multi-attack, balanced classes, 63 temporal features per session
8. **Add BB84 model training/evaluation** — reuse QAD pipeline (`train.py`, `evaluate.py`) on BB84 features
9. **Add threshold/CUSUM baselines** on BB84 temporal QBER (currently only placeholders in `evaluate.py`)

---

**Priority Order:**
1. Fix preprocessing leakage (highest impact on scientific correctness) ✅ DONE
2. Run ablation without attack-specific params (quantifies shortcut learning) ✅ DONE
3. **Implement BB84 attacks 1-3** (PNS, Beam Splitting, Time-Shift) ✅ DONE
4. **Implement remaining BB84 attacks** (Trojan Horse, Quantum Cloning, Fake State) ✅ DONE
5. **Generate BB84 dataset** and evaluate temporal-QBER pipeline
6. Build report PDF (delivery artifact) — **moved to last**