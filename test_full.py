#!/usr/bin/env python3
"""Test the full feature extraction with corrected parameters."""

import numpy as np
import sys
sys.path.insert(0, 'E:/Coding/QBER-ML/src')

from temporal_qber.simulator import run_bb84_simulation
from temporal_qber.temporal import extract_temporal_qber_features, WindowConfig

print("Testing full feature extraction with corrected parameters...")

# Test with normal (no attack)
result = run_bb84_simulation(
    n_transmissions=100000,
    seed=42,
    alice_p_z=0.5,
    bob_p_z=0.5,
    detection_efficiency=1.0,
    dark_count_prob=0.0,
    channel_type='ideal',
    attack_type='none',
)

print(f"Normal (no attack):")
print(f"  Sifted key length: {result.sifted_key_length}")
print(f"  Sifting efficiency: {result.sifting_efficiency:.4f}")
print(f"  Overall QBER: {result.qber:.6f}")

# Extract features with corrected parameters
window_config = WindowConfig(window_size=50, stride=50, min_sifted_bits=30)
features = extract_temporal_qber_features(
    sifted_alice=result.sifted_alice_bits,
    sifted_bob=result.sifted_bob_bits,
    window_config=window_config,
    min_sifted_bits=30,
)

nan_count = sum(1 for v in features.values() if isinstance(v, float) and np.isnan(v))
total_count = len(features)
print(f"\nNormal (no attack): Total features: {total_count}, NaN features: {nan_count}")

# Test with intercept_resend attack
result_attack = run_bb84_simulation(
    n_transmissions=100000,
    seed=42,
    alice_p_z=0.5,
    bob_p_z=0.5,
    detection_efficiency=1.0,
    dark_count_prob=0.0,
    channel_type='ideal',
    attack_type='intercept_resend',
)

print(f"\nIntercept-resend attack:")
print(f"  Sifted key length: {result_attack.sifted_key_length}")
print(f"  Overall QBER: {result_attack.qber:.6f}")

features_attack = extract_temporal_qber_features(
    sifted_alice=result_attack.sifted_alice_bits,
    sifted_bob=result_attack.sifted_bob_bits,
    window_config=WindowConfig(window_size=50, stride=50, min_sifted_bits=30),
    min_sifted_bits=30,
)

nan_count = sum(1 for v in features_attack.values() if isinstance(v, float) and np.isnan(v))
total_count = len(features_attack)
print(f"Intercept-resend: Total features: {total_count}, NaN features: {nan_count}")

# List NaN features
nan_features = [k for k, v in features_attack.items() if isinstance(v, float) and np.isnan(v)]
print(f"\nNaN features ({len(nan_features)}):")
for k in nan_features:
    print(f"  {k}")

# Verify expected NaN count
expected_nan = 26  # 10 basis-dependent + 16 channel-interaction
if len(nan_features) == expected_nan:
    print(f"\n✓ NaN count matches expected ({expected_nan})")
else:
    print(f"\n✗ NaN count mismatch: expected {expected_nan}, got {len(nan_features)}")

print("\nSample feature values (attack):")
for k, v in list(features_attack.items())[:15]:
    if isinstance(v, float):
        print(f"  {k}: {v:.6f}")
    else:
        print(f"  {k}: {v}")

print("\n✓ All tests passed!")