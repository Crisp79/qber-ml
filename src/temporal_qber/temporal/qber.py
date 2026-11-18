"""
QBER calculation for temporal windows.

Calculates Quantum Bit Error Rate and related statistics
for temporal windows of BB84 sifted keys.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class WindowQBERResult:
    """QBER calculation result for a single window."""
    window_index: int
    start_bit: int
    end_bit: int
    length: int
    sifted_bits: int
    errors: int
    qber: Optional[float]  # None if insufficient sifted bits
    errors_count: int
    detection_count: int
    detection_rate: float
    is_partial: bool
    is_valid: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_index": self.window_index,
            "start_bit": self.start_bit,
            "end_bit": self.end_bit,
            "length": self.length,
            "sifted_bits": self.sifted_bits,
            "errors": self.errors_count,
            "qber": self.qber,
            "errors_count": self.errors_count,
            "detection_count": self.detection_count,
            "detection_rate": self.detection_rate,
            "is_partial": self.is_partial,
            "is_valid": self.is_valid,
        }


def calculate_window_qber(
    sifted_alice: np.ndarray,
    sifted_bob: np.ndarray,
    window_start: int,
    window_end: int,
    min_sifted_bits: int = 200,
) -> Dict[str, Any]:
    """
    Calculate QBER for a single window.
    
    Args:
        sifted_alice: Alice's sifted key bits
        sifted_bob: Bob's sifted key bits
        window_start: Start index of window (inclusive)
        window_end: End index of window (exclusive)
        min_sifted_bits: Minimum sifted bits for valid QBER
        
    Returns:
        Dictionary with QBER and statistics
    """
    if window_start >= len(sifted_alice) or window_end > len(sifted_alice):
        raise ValueError("Window indices out of bounds")
    if window_start >= window_end:
        raise ValueError("window_start must be less than window_end")
    if len(sifted_alice) != len(sifted_bob):
        raise ValueError("sifted_alice and sifted_bob must have same length")
    
    window_alice = sifted_alice[window_start:window_end]
    window_bob = sifted_bob[window_start:window_end]
    
    sifted_bits = len(window_alice)
    
    if sifted_bits == 0:
        return {
            "sifted_bits": 0,
            "errors": 0,
            "qber": None,
            "errors_count": 0,
            "is_valid": False,
        }
    
    errors = np.sum(window_alice != window_bob)
    
    if sifted_bits >= min_sifted_bits:
        qber = float(errors) / sifted_bits if sifted_bits > 0 else 0.0
        is_valid = True
    else:
        qber = None
        is_valid = False
    
    return {
        "sifted_bits": sifted_bits,
        "errors": int(errors),
        "qber": qber,
        "errors_count": int(errors),
        "is_valid": is_valid,
    }


def calculate_window_qber_batch(
    sifted_alice: np.ndarray,
    sifted_bob: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    min_sifted_bits: int = 200,
) -> List[Dict[str, Any]]:
    """
    Calculate QBER for multiple windows.
    
    Args:
        sifted_alice: Alice's sifted key bits
        sifted_bob: Bob's sifted key bits
        window_starts: Array of window start indices
        window_ends: Array of window end indices
        min_sifted_bits: Minimum sifted bits for valid QBER
        
    Returns:
        List of dictionaries with QBER results
    """
    results = []
    for i, (start, end) in enumerate(zip(window_starts, window_ends)):
        result = calculate_window_qber(sifted_alice, sifted_bob, start, end, min_sifted_bits)
        result["window_index"] = i
        result["start_bit"] = int(start)
        result["end_bit"] = int(end)
        result["length"] = int(end - start)
        results.append(result)
    return results


def calculate_session_qber(
    sifted_alice: np.ndarray,
    sifted_bob: np.ndarray,
) -> float:
    """
    Calculate overall session QBER.
    
    Args:
        sifted_alice: Alice's sifted key bits
        sifted_bob: Bob's sifted key bits
        
    Returns:
        Overall QBER
    """
    if len(sifted_alice) != len(sifted_bob):
        raise ValueError("sifted_alice and sifted_bob must have same length")
    if len(sifted_alice) == 0:
        return 0.0
    
    errors = np.sum(sifted_alice != sifted_bob)
    return float(errors) / len(sifted_alice) if len(sifted_alice) > 0 else 0.0


def calculate_qber_statistics(
    qber_values: np.ndarray,
    min_windows: int = 2,
) -> Dict[str, float]:
    """
    Calculate statistics from an array of window QBER values.
    
    Args:
        qber_values: Array of valid QBER values (no None/NaN)
        min_windows: Minimum windows for valid statistics
        
    Returns:
        Dictionary of statistics
    """
    if len(qber_values) < min_windows:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "var": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "iqr": np.nan,
            "cv": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
        }
    
    # Remove NaN values
    valid_qber = qber_values[~np.isnan(qber_values)]
    if len(valid_qber) < min_windows:
        valid_qber = qber_values
    
    mean_val = np.mean(valid_qber)
    std_val = np.std(valid_qber)
    
    # Coefficient of variation: std/mean. When mean=0 and std=0, CV=0 (no variation).
    # When mean=0 but std>0, CV is infinite/undefined -> NaN.
    if mean_val > 0:
        cv = float(std_val / mean_val)
    elif mean_val == 0 and std_val == 0:
        cv = 0.0  # No variation
    else:
        cv = np.nan  # mean=0 but std>0 -> infinite CV
    
    return {
        "mean": float(mean_val),
        "median": float(np.median(valid_qber)),
        "std": float(std_val),
        "var": float(np.var(valid_qber)),
        "min": float(np.min(valid_qber)),
        "max": float(np.max(valid_qber)),
        "range": float(np.max(valid_qber) - np.min(valid_qber)),
        "iqr": float(np.percentile(valid_qber, 75) - np.percentile(valid_qber, 25)),
        "cv": cv,
        "skewness": float(_skewness(valid_qber)),
        "kurtosis": float(_kurtosis(valid_qber)),
    }


def _skewness(data: np.ndarray) -> float:
    """Calculate skewness."""
    n = len(data)
    if n < 3:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return float(np.sum(((data - mean) / std) ** 3) / n)


def _kurtosis(data: np.ndarray) -> float:
    """Calculate kurtosis (excess kurtosis)."""
    n = len(data)
    if n < 4:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return float(np.sum(((data - mean) / std) ** 4) / n - 3)


def detect_significant_changes(
    qber_values: np.ndarray,
    threshold: float = None,
) -> Dict[str, Any]:
    """
    Detect significant changes in QBER sequence.
    
    Args:
        qber_values: Array of QBER values
        threshold: Change threshold (default: 2 * std of differences)
        
    Returns:
        Dictionary with change detection results
    """
    if len(qber_values) < 2:
        return {
            "num_changes": 0,
            "change_indices": [],
            "change_magnitudes": [],
            "max_change": 0.0,
            "mean_abs_change": 0.0,
            "std_change": 0.0,
        }
    
    # Calculate first differences
    diffs = np.diff(qber_values)
    
    if threshold is None:
        threshold = 2 * np.std(diffs) if np.std(diffs) > 0 else 0.0
    
    # Significant changes
    significant = np.abs(diffs) > threshold
    change_indices = np.where(significant)[0].tolist()
    change_magnitudes = diffs[significant].tolist()
    
    return {
        "num_changes": int(np.sum(significant)),
        "change_indices": change_indices,
        "change_magnitudes": change_magnitudes,
        "max_change": float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0.0,
        "mean_abs_change": float(np.mean(np.abs(diffs))) if len(diffs) > 0 else 0.0,
        "std_change": float(np.std(diffs)) if len(diffs) > 0 else 0.0,
    }


def calculate_trend_features(
    qber_values: np.ndarray,
    window_indices: np.ndarray = None,
) -> Dict[str, float]:
    """
    Calculate trend features from QBER sequence.
    
    Args:
        qber_values: Array of QBER values
        window_indices: Optional window indices (default: 0, 1, 2, ...)
        
    Returns:
        Dictionary of trend features
    """
    if len(qber_values) < 2:
        return {
            "linear_slope": 0.0,
            "trend_direction": 0.0,
            "r_squared": 0.0,
            "early_late_diff": 0.0,
        }
    
    if window_indices is None:
        x = np.arange(len(qber_values), dtype=float)
    else:
        x = window_indices.astype(float)
    
    # Linear regression
    n = len(qber_values)
    x_mean = np.mean(x)
    y_mean = np.mean(qber_values)
    
    # Covariance and variance
    xy_cov = np.sum((x - x_mean) * (qber_values - y_mean))
    x_var = np.sum((x - x_mean) ** 2)
    
    if x_var == 0:
        slope = 0.0
        r_squared = 0.0
    else:
        slope = xy_cov / x_var
        intercept = y_mean - slope * x_mean
        
        # R-squared
        y_pred = intercept + slope * x
        ss_tot = np.sum((qber_values - y_mean) ** 2)
        ss_res = np.sum((qber_values - y_pred) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Trend direction: 1 = increasing, -1 = decreasing, 0 = flat
    if slope > 1e-6:
        trend_direction = 1.0
    elif slope < -1e-6:
        trend_direction = -1.0
    else:
        trend_direction = 0.0
    
    # Early vs late QBER difference
    mid = len(qber_values) // 2
    early_qber = np.mean(qber_values[:mid]) if mid > 0 else qber_values[0]
    late_qber = np.mean(qber_values[mid:]) if mid < len(qber_values) else qber_values[-1]
    early_late_diff = float(late_qber - early_qber)
    
    return {
        "linear_slope": float(slope),
        "trend_direction": float(trend_direction),
        "r_squared": float(r_squared),
        "early_late_diff": float(early_late_diff),
    }