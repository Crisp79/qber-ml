import numpy as np
from temporal_qber.temporal import extract_temporal_qber_features, WindowConfig
from temporal_qber.temporal.windowing import create_windows

# Simulate what happens with typical sifted key length ~5000
np.random.seed(42)
sifted_alice = np.random.randint(0, 2, 5000)
sifted_bob = np.random.randint(0, 2, 5000)

window_config = WindowConfig(window_size=50, stride=50, min_sifted_bits=200)

windows = create_windows(sifted_alice, sifted_bob, bob_detected=None, config=window_config)

print(f'Total windows: {len(windows)}')
valid_windows = [w for w in windows if w.get('is_valid', False)]
print(f'Valid windows: {len(valid_windows)}')

for w in windows[:5]:
    print(f'  Window {w["window_index"]}: sifted_bits={w["sifted_bits"]}, is_valid={w["is_valid"]}, qber={w["qber"]}')

# Now test the full feature extraction
print("\n--- Feature Extraction ---")
features = extract_temporal_qber_features(
    sifted_alice=sifted_alice,
    sifted_bob=sifted_bob,
    window_config=window_config,
    min_sifted_bits=200,
)

nan_count = sum(1 for v in features.values() if isinstance(v, float) and np.isnan(v))
total_count = len(features)
print(f'Total features: {total_count}')
print(f'NaN features: {nan_count}')
print(f'Sample feature values:')
for k, v in list(features.items())[:10]:
    print(f'  {k}: {v}')