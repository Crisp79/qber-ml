"""
Temporal windowing for BB84 sessions.

Partitions session data into temporal windows with configurable
window size, stride, and overlap handling.
"""

import numpy as np
from typing import List, Optional, Tuple, Literal
from dataclasses import dataclass
from enum import Enum


class WindowMode(Enum):
    """Window overlap mode."""
    NON_OVERLAPPING = "non_overlapping"
    OVERLAPPING = "overlapping"


@dataclass
class WindowConfig:
    """Configuration for temporal windowing."""
    window_size: int = 50
    stride: Optional[int] = None  # If None, defaults to window_size (non-overlapping)
    mode: WindowMode = WindowMode.NON_OVERLAPPING
    min_sifted_bits: int = 200  # Minimum valid sifted bits per window
    include_partial_trailing: bool = False  # Include incomplete trailing window
    
    def __post_init__(self):
        if self.stride is None:
            self.stride = self.window_size
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.stride > self.window_size and self.mode == WindowMode.OVERLAPPING:
            # For overlapping, stride <= window_size is typical
            pass
        if self.min_sifted_bits < 0:
            raise ValueError("min_sifted_bits must be non-negative")


@dataclass
class WindowResult:
    """Result of a single temporal window."""
    window_index: int
    start_bit: int
    end_bit: int
    length: int
    qber: Optional[float]  # None if insufficient sifted bits
    errors: int
    sifted_bits: int
    errors_count: int
    detection_count: int
    detection_rate: float
    
    def is_valid(self) -> bool:
        """Check if window has sufficient sifted bits for valid QBER."""
        return self.qber is not None


def create_windows(
    sifted_alice: np.ndarray,
    sifted_bob: np.ndarray,
    bob_detected: Optional[np.ndarray] = None,
    config: Optional[WindowConfig] = None,
) -> List[dict]:
    """
    Partition sifted key data into temporal windows.
    
    Args:
        sifted_alice: Alice's sifted key bits
        sifted_bob: Bob's sifted key bits (must match alice length)
        bob_detected: Optional boolean array indicating detected photons
        config: Window configuration
        
    Returns:
        List of window dictionaries with QBER and statistics
    """
    if config is None:
        config = WindowConfig()
    
    if len(sifted_alice) != len(sifted_bob):
        raise ValueError("sifted_alice and sifted_bob must have same length")
    
    if bob_detected is not None and len(bob_detected) != len(sifted_alice):
        raise ValueError("bob_detected must match sifted_alice length")
    
    n_sifted = len(sifted_alice)
    if n_sifted == 0:
        return []
    
    # Calculate number of windows
    if config.stride >= config.window_size:
        # Non-overlapping or exactly tiling windows
        n_windows = max(1, (n_sifted + config.stride - 1) // config.stride)
    else:
        # Overlapping windows
        n_windows = max(1, (n_sifted - config.window_size) // config.stride + 1)
    
    windows = []
    
    for w in range(n_windows):
        start = w * config.stride
        end = min(start + config.window_size, len(sifted_alice))
        
        # Check if we have a complete window
        is_partial = end - start < config.window_size
        
        # Handle partial trailing window
        if is_partial and not config.include_partial_trailing:
            break
        
        window_alice = sifted_alice[start:end]
        window_bob = sifted_bob[start:end]
        
        # Get detection info if available
        if bob_detected is not None:
            window_detected = bob_detected[start:end]
            detection_count = int(np.sum(window_detected))
            detection_rate = detection_count / len(window_alice) if len(window_alice) > 0 else 0.0
        else:
            detection_count = 0
            detection_rate = 0.0
        
        sifted_bits = len(window_alice)
        
        # Calculate QBER if sufficient sifted bits
        if sifted_bits >= config.min_sifted_bits:
            errors = np.sum(window_alice != window_bob)
            qber = float(np.sum(window_alice != window_bob)) / sifted_bits if sifted_bits > 0 else 0.0
            errors_count = int(np.sum(window_alice != window_bob))
            qber_valid = True
        else:
            errors = 0
            qber = None  # Mark as invalid/insufficient data
            errors_count = 0
            qber_valid = False
        
        window = {
            "window_index": int(w),
            "start_bit": int(start),
            "end_bit": int(end),
            "length": int(end - start),
            "qber": qber if qber_valid else None,
            "errors": errors_count,
            "sifted_bits": sifted_bits,
            "errors_count": errors_count,
            "detection_count": detection_count,
            "detection_rate": detection_rate,
            "is_partial": is_partial,
            "is_valid": qber_valid,
        }
        
        windows.append(window)
    
    return windows


def create_windows_from_session(
    sifted_alice: np.ndarray,
    sifted_bob: np.ndarray,
    bob_detected: Optional[np.ndarray] = None,
    window_size: int = 50,
    stride: Optional[int] = None,
    min_sifted_bits: int = 200,
    include_partial_trailing: bool = False,
) -> List[dict]:
    """
    Convenience function to create windows with basic configuration.
    
    Args:
        sifted_alice: Alice's sifted key bits
        sifted_bob: Bob's sifted key bits
        bob_detected: Optional detection flags
        window_size: Size of each window in bits
        stride: Stride between windows (None = window_size)
        min_sifted_bits: Minimum sifted bits for valid QBER
        include_partial_trailing: Include incomplete trailing window
        
    Returns:
        List of window dictionaries
    """
    config = WindowConfig(
        window_size=window_size,
        stride=stride,
        min_sifted_bits=min_sifted_bits,
        include_partial_trailing=include_partial_trailing,
    )
    return create_windows(sifted_alice, sifted_bob, bob_detected, config)