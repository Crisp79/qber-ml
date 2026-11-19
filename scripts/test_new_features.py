#!/usr/bin/env python3
"""Test the new features implementation."""

import numpy as np
import sys
sys.path.insert(0, 'src')

from temporal_qber.simulator import run_bb84_simulation
from temporal_qber.temporal import extract_temporal_qber_features, WindowConfig

# Test with normal
result = run_bb84_simulation(n_transmissions=10000, seed=42, attack_type='none')
print('Normal test:')
print(f'  Sifted key length: {result.sifted_key_length}')
print(f'  Has sifted_alice_rect: {hasattr(result, "sifted_alice_rect")}')
print(f'  Has loss_seq: {hasattr(result, "loss_seq")}')
if hasattr(result, 'loss_seq'):
    print(f'  loss_seq length: {len(result.loss_seq)}')
    print(f'  loss_seq sum: {np.sum(result.loss_seq)}')
if hasattr(result, 'sifted_alice_rect'):
    print(f'  sifted_alice_rect: {len(result.sifted_alice_rect)}')
    print(f'  sifted_alice_diag: {len(result.sifted_alice_diag)}')

# Test feature extraction
window_config = WindowConfig(window_size=50, stride=50, min_sifted_bits=30)
features = extract_temporal_qber_features(
    sifted_alice=result.sifted_alice_bits,
    sifted_bob=result.sifted_bob_bits,
    sifted_alice_rect=result.sifted_alice_rect,
    sifted_bob_rect=result.sifted_bob_rect,
    sifted_alice_diag=result.sifted_alice_diag,
    sifted_bob_diag=result.sifted_bob_diag,
    loss_seq=result.loss_seq,
    window_config=WindowConfig(window_size=50, stride=50, min_sifted_bits=30),
    min_sifted_bits=30,
)

nan_count = sum(1 for v in features.values() if isinstance(v, float) and np.isnan(v))
total = len(features)
print(f'\nNormal: total={total}, NaN={nan_count}, Finite={total - nan_count}')

# Test with attack
result2 = run_bb84_simulation(n_transmissions=10000, seed=42, attack_type='intercept_resend')
features2 = extract_temporal_qber_features(
    sifted_alice=result2.sifted_alice_bits,
    sifted_bob=result2.sifted_bob_bits,
    sifted_alice_rect=result2.sifted_alice_rect,
    sifted_bob_rect=result2.sifted_bob_rect,
    sifted_alice_diag=result2.sifted_alice_diag,
    sifted_bob_diag=result2.sifted_bob_diag,
    loss_seq=result2.loss_seq,
    window_config=WindowConfig(window_size=50, stride=50, min_sifted_bits=30),
    min_sifted_bits=30,
)

nan_count2 = sum(1 for v in features2.values() if isinstance(v, float) and np.isnan(v))
total2 = len(features2)
print(f'Attack: total={total2}, NaN={nan_count2}, Finite={total2 - nan_count2}')
print(f'qber_rectilinear: {features2.get("qber_rectilinear", "MISSING")}')
print(f'qber_diagonal: {features2.get("qber_diagonal", "MISSING")}')
print(f'qber_basis_diff: {features2.get("qber_basis_diff", "MISSING")}')
print(f'qber_cv: {features2.get("qber_cv", "MISSING")}')
print(f'key_gen_rate: {features2.get("key_gen_rate", "MISSING")}')
print(f'qber_loss_corr: {features2.get("qber_loss_corr", "MISSING")}')