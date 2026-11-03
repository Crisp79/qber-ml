"""
Bob's quantum measurement for BB84 protocol.

Implements quantum measurement in Z and X bases using Born rule probabilities.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from .alice import Basis, Bit, BB84_STATES, measure_state_in_basis, sample_measurement


@dataclass
class BobMeasurement:
    """Single measurement result from Bob."""
    basis: int          # 0 (Z) or 1 (X)
    outcome: int        # 0 or 1 (measured bit)
    prob_0: float       # Probability of measuring 0
    prob_1: float       # Probability of measuring 1
    detected: bool      # Whether a detection occurred
    is_dark_count: bool # Whether this detection was a dark count


class Bob:
    """Bob's BB84 quantum measurement."""
    
    def __init__(self, seed: Optional[int] = None, p_z: float = 0.5, 
                 detection_efficiency: float = 1.0, dark_count_prob: float = 0.0):
        """
        Initialize Bob.
        
        Args:
            seed: Random seed for reproducibility
            p_z: Probability of choosing Z basis (default 0.5)
            detection_efficiency: Probability of detecting a photon (default 1.0)
            dark_count_prob: Probability of dark count per time window (default 0.0)
        """
        if not 0 <= p_z <= 1:
            raise ValueError(f"p_z must be in [0, 1], got {p_z}")
        if not 0 <= detection_efficiency <= 1:
            raise ValueError(f"detection_efficiency must be in [0, 1], got {detection_efficiency}")
        if not 0 <= dark_count_prob <= 1:
            raise ValueError(f"dark_count_prob must be in [0, 1], got {dark_count_prob}")
        
        self.rng = np.random.default_rng(seed)
        self.p_z = p_z
        self.detection_efficiency = detection_efficiency
        self.dark_count_prob = dark_count_prob
        self._measurements: List[BobMeasurement] = []
    
    def choose_basis(self) -> int:
        """Choose a random measurement basis (0=Z, 1=X) with probability p_z for Z."""
        return 0 if self.rng.random() < self.p_z else 1
    
    def measure(self, state: np.ndarray, basis: Optional[int] = None) -> BobMeasurement:
        """
        Measure a quantum state.
        
        Args:
            state: 2D complex state vector from Alice
            basis: Measurement basis (0=Z, 1=X). If None, chooses randomly.
            
        Returns:
            BobMeasurement with outcome and probabilities
        """
        if basis is None:
            basis = self.choose_basis()
        
        if basis not in (0, 1):
            raise ValueError(f"Basis must be 0 (Z) or 1 (X), got {basis}")
        
        # Calculate measurement probabilities
        prob_0, prob_1 = measure_state_in_basis(state, basis)
        
        # Check for dark count first (independent of photon arrival)
        is_dark_count = self.rng.random() < self.dark_count_prob
        
        # Check photon detection
        detected_photon = self.rng.random() < self.detection_efficiency
        
        if is_dark_count:
            # Dark count: random outcome regardless of photon
            detected = True
            is_dark_count = True
            outcome = self.rng.integers(0, 2)
        elif detected_photon:
            # Normal photon detection
            detected = True
            is_dark_count = False
            outcome = sample_measurement(state, basis, self.rng)
        else:
            # No detection event
            detected = False
            is_dark_count = False
            outcome = -1  # No detection
        
        measurement = BobMeasurement(
            basis=basis,
            outcome=outcome,
            prob_0=prob_0,
            prob_1=prob_1,
            detected=detected,
            is_dark_count=is_dark_count
        )
        
        self._measurements.append(measurement)
        return measurement
    
    def measure_batch(self, states: np.ndarray, bases: Optional[np.ndarray] = None) -> List[BobMeasurement]:
        """
        Measure a batch of quantum states.
        
        Args:
            states: Array of state vectors (n, 2) complex
            bases: Optional array of measurement bases (0 or 1). If None, chooses randomly.
            
        Returns:
            List of BobMeasurement
        """
        n = states.shape[0]
        
        if bases is None:
            bases = self.rng.integers(0, 2, size=n, dtype=np.int8)
        elif len(bases) != n:
            raise ValueError(f"Bases length {len(bases)} != states length {n}")
        
        measurements = []
        for i in range(n):
            measurement = self.measure(states[i], int(bases[i]))
            measurements.append(measurement)
        
        return measurements
    
    def get_measurements(self) -> List[BobMeasurement]:
        """Get all measurements."""
        return self._measurements.copy()
    
    def get_sifted_key(self, alice_bases: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract sifted key by comparing with Alice's bases.
        
        Args:
            alice_bases: Alice's basis choices for each transmission
            
        Returns:
            (matching_mask, bob_sifted_key, alice_sifted_key_indices)
            matching_mask: Boolean array where bases match
            bob_sifted_key: Bob's measured bits where bases match
            alice_sifted_key_indices: Indices in original arrays where bases match
        """
        if len(alice_bases) != len(self._measurements):
            raise ValueError(f"alice_bases length {len(alice_bases)} != measurements {len(self._measurements)}")
        
        bob_bases = np.array([m.basis for m in self._measurements], dtype=np.int8)
        matching_mask = (bob_bases == alice_bases)
        
        # Get Bob's outcomes where detected and bases match
        bob_outcomes = np.array([m.outcome for m in self._measurements], dtype=np.int8)
        detected = np.array([m.detected for m in self._measurements], dtype=bool)
        
        # Only consider detected events with matching bases
        valid_mask = matching_mask & detected
        indices = np.where(valid_mask)[0]
        
        return matching_mask, bob_outcomes[valid_mask], indices
    
    def reset(self, seed: Optional[int] = None):
        """Reset Bob's state."""
        self._measurements.clear()
        if seed is not None:
            self.rng = np.random.default_rng(seed)
    
    def get_config(self) -> dict:
        """Get configuration info."""
        return {
            "p_z": self.p_z,
            "detection_efficiency": self.detection_efficiency,
            "dark_count_prob": self.dark_count_prob,
            "n_measurements": len(self._measurements),
        }


def calculate_qber(alice_bits: np.ndarray, bob_bits: np.ndarray) -> float:
    """
    Calculate Quantum Bit Error Rate (QBER).
    
    Args:
        alice_bits: Alice's sifted key bits
        bob_bits: Bob's sifted key bits
        
    Returns:
        QBER (fraction of mismatched bits)
    """
    if len(alice_bits) != len(bob_bits):
        raise ValueError("Key lengths must match")
    if len(alice_bits) == 0:
        return 0.0
    
    errors = np.sum(alice_bits != bob_bits)
    return float(errors) / len(alice_bits)


def calculate_agreement(alice_bits: np.ndarray, bob_bits: np.ndarray) -> float:
    """
    Calculate agreement between Alice and Bob's sifted keys.
    
    Args:
        alice_bits: Alice's sifted key bits
        bob_bits: Bob's sifted key bits
        
    Returns:
        Agreement fraction (1 - QBER)
    """
    return 1.0 - calculate_qber(alice_bits, bob_bits)