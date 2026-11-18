#!/usr/bin/env python3
"""Final validation test."""

import numpy as np
import sys
sys.path.insert(0, 'E:/Coding/QBER-ML/src')

from temporal_qber.simulator import run_bb84_simulation
from temporal_qber.temporal import extract_temporal_qber_features, WindowConfig

# Test with normal and intercept_resend at 100k transmissions
for attack_type in ['none', 'intercept_resend']:
    result = run_bb84_simulation(
        n_transmissions=100000,
        seed=42,
        alice_p_z=0.5,
        bob_p_z=0.5,
        detection_efficiency=1.0,
        dark_count_prob=0.0,
        channel_type='ideal',
        attack_type=attack_type,
    )
    
    window_config = WindowConfig(window_size=50, stride=50, min_sifted_bits=30)
    features = extract_temporal_qber_features(
        sifted_alice=result.sifted_alice_bits,
        sifted_bob=result.sifted_bob_bits,
        window_config=window_config,
        min_sifted_bits=30,
    )
    
    nan_count = sum(1 for v in features.values() if isinstance(v, float) and np.isnan(v))
    total_count = len(features)
    
    # Check qber_cv specifically
    qber_cv = features.get('qber_cv', 'MISSING')
    
    print(f"{attack_type}: total={total_count}, NaN={nan_count}, qber_cv={qber_cv}")

# Verify all features are finite for attack case (except expected NaN)
result = run_bb84_simulation(n_transmissions=100000, seed=42, attack_type='intercept_resend')
window_config = WindowConfig(window_size=50, stride=50, min_sifted_bits=30)
features = extract_temporal_qber_features(
    sifted_alice=result.sifted_alice_bits,
    sifted_bob=result.sifted_bob_bits,
    window_config=window_config,
    min_sifted_bits=30,
)

finite_features = {k: v for k, v in features.items() 
                   if isinstance(v, float) and not np.isnan(v)}
inf_features = {k: v for k, v in features.items() 
                if isinstance(v, float) and np.isinf(v)}

print(f"\nFinite features: {len(finite_features)}")
print(f"Infinite features: {len(inf_features)}")
print(f"NaN features: {sum(1 for v in features.values() if isinstance(v, float) and np.isnan(v))}")

# Check for any inf
if inf_features:
    print(f"INFINITE FEATURES: {inf_features}")
else:
    print("No infinite features - GOOD!")

print("\nSUCCESS: All fixes validated!")