#!/usr/bin/env python3
"""Test the fix for qber_cv division by zero."""

import numpy as np
import sys
sys.path.insert(0, 'E:/Coding/QBER-ML/src')

from temporal_qber.temporal.qber import calculate_qber_statistics

print("Testing qber_cv fix...")

# Test case 1: All zeros (mean=0, std=0) - should return cv=0
qber_zeros = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
result = calculate_qber_statistics(qber_zeros)
print(f"All zeros: mean={result['mean']}, std={result['std']}, cv={result['cv']}")
assert result['cv'] == 0.0, f"Expected cv=0.0, got {result['cv']}"

# Test case 2: All same non-zero value (mean>0, std=0) - should return cv=0
qber_same = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
result = calculate_qber_statistics(qber_same)
print(f"All same (0.1): mean={result['mean']}, std={result['std']}, cv={result['cv']}")
assert result['cv'] == 0.0, f"Expected cv=0.0, got {result['cv']}"

# Test case 2: Mean=0 but std>0 (should be NaN)
# This would require negative values, which QBER doesn't have, so skip

# Test case 3: Normal case with variation
qber_normal = np.array([0.0, 0.02, 0.04, 0.06, 0.08])
result = calculate_qber_statistics(qber_normal)
print(f"Normal variation: mean={result['mean']:.4f}, std={result['std']:.4f}, cv={result['cv']:.4f}")
assert not np.isnan(result['cv']), f"Expected finite cv, got {result['cv']}"

# Test case 4: Mean=0, std=0 (all zeros) - cv should be 0
qber_zeros = np.array([0.0, 0.0, 0.0])
result = calculate_qber_statistics(qber_zeros)
print(f"All zeros: mean={result['mean']}, std={result['std']}, cv={result['cv']}")
assert result['cv'] == 0.0, f"Expected cv=0.0, got {result['cv']}"

print("\nAll tests passed!")