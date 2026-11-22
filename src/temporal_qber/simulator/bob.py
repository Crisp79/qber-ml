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
    is_double_click: bool = False   # Both detectors fired (measure_batch only)
    empty_gate_click: bool = False  # Click in the no-signal monitoring gate (measure_batch only)


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
        
        # Check photon detection - only if state has non-zero norm
        # (zero-norm state means photon was lost, e.g. beam splitting sent it to Eve)
        state_norm = np.linalg.norm(state)
        if state_norm > 0:
            detected_photon = self.rng.random() < self.detection_efficiency
        else:
            detected_photon = False  # No photon to detect
        
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
    
    def measure_batch(
        self,
        states: np.ndarray,
        bases: Optional[np.ndarray] = None,
        eff_scale: Optional[np.ndarray] = None,
        spurious_click_prob: Optional[np.ndarray] = None,
        dark_scale: Optional[np.ndarray] = None,
        photon_numbers: Optional[np.ndarray] = None,
    ) -> List[BobMeasurement]:
        """
        Measure a batch of quantum states (vectorized).

        Args:
            states: Array of state vectors (n, 2) complex
            bases: Optional array of measurement bases (0 or 1). If None, chooses randomly.
            eff_scale: Optional (n, 2) multipliers (0..1) on detection efficiency when the
                bit-0 / bit-1 detector would fire (detector efficiency mismatch, e.g. time-shift).
            spurious_click_prob: Optional (n,) probability that external light (e.g. trojan-horse
                probe) makes a random detector click.
            dark_scale: Optional (n,) multiplier on the dark-count probability (0 = blinded).
            photon_numbers: Optional (n,) photons arriving per pulse (decoy-state mode). If
                None every non-vacuum state is a single photon.

        Each slot's photon, dark count and spurious light can fire either of the two
        detectors; if both fire it is a double click (recorded, random bit). Each slot also
        has an empty monitoring gate that only dark counts can fire.

        Returns:
            List of BobMeasurement
        """
        n = states.shape[0]
        
        if bases is None:
            bases = self.rng.integers(0, 2, size=n, dtype=np.int8)
        elif len(bases) != n:
            raise ValueError(f"Bases length {len(bases)} != states length {n}")
        
        # Calculate measurement probabilities for all sessions at once
        # Z basis: prob_0 = |state[0]|^2, prob_1 = |state[1]|^2
        # X basis: prob_0 = |⟨+|ψ⟩|^2, prob_1 = |⟨-|ψ⟩|^2
        prob_0_z = np.abs(states[:, 0]) ** 2
        prob_1_z = np.abs(states[:, 1]) ** 2
        
        plus = (states[:, 0] + states[:, 1]) / np.sqrt(2)
        minus = (states[:, 0] - states[:, 1]) / np.sqrt(2)
        prob_0_x = np.abs(plus) ** 2
        prob_1_x = np.abs(minus) ** 2
        
        # Select probabilities based on basis choice
        prob_0 = np.where(bases == 0, prob_0_z, prob_0_x)
        prob_1 = np.where(bases == 0, prob_1_z, prob_1_x)
        
        # Which of Bob's two detectors (bit 0 / bit 1 of the chosen basis) fire in this slot.
        # Dark count: a click in a random detector, independent of the photon.
        dark_p = self.dark_count_prob if dark_scale is None else self.dark_count_prob * np.asarray(dark_scale)
        is_dark = self.rng.random(n) < dark_p
        dark_det = self.rng.integers(0, 2, size=n).astype(np.int8)

        # Photon click. A zero-norm (vacuum) state means the photon was lost in the
        # channel or removed by Eve: only a dark count can then produce a click. Without
        # this check a lost photon was "detected" as bit 0 (prob_1 = 0), turning loss into
        # a bit-0 bias and spurious QBER. Mirrors the check in measure().
        has_photon = np.linalg.norm(states, axis=1) > 0
        if photon_numbers is None:
            # Single-photon pulses
            photon_click = has_photon & (self.rng.random(n) < self.detection_efficiency)
            photon_det = (self.rng.random(n) < prob_1).astype(np.int8)  # detector the photon reaches

            # Detector efficiency mismatch: the photon click survives with the multiplier of
            # the detector it reaches (outcome-dependent loss).
            if eff_scale is not None:
                eff_scale = np.asarray(eff_scale, dtype=np.float64)
                keep_prob = np.where(photon_det == 0, eff_scale[:, 0], eff_scale[:, 1])
                photon_click &= self.rng.random(n) < keep_prob
            photon_fired_0 = photon_click & (photon_det == 0)
            photon_fired_1 = photon_click & (photon_det == 1)
        else:
            # Multi-photon pulses (decoy-state mode): each photon is routed to a detector by
            # the measurement and detected independently (efficiency x mismatch multiplier);
            # a threshold detector fires if at least one of its photons is detected.
            n_ph = np.where(has_photon, np.asarray(photon_numbers, dtype=np.int64), 0)
            n_to_1 = self.rng.binomial(n_ph, np.clip(prob_1, 0.0, 1.0))
            n_to_0 = n_ph - n_to_1
            scale = np.ones((n, 2)) if eff_scale is None else np.asarray(eff_scale, dtype=np.float64)
            photon_fired_0 = self.rng.binomial(n_to_0, self.detection_efficiency * scale[:, 0]) > 0
            photon_fired_1 = self.rng.binomial(n_to_1, self.detection_efficiency * scale[:, 1]) > 0
            photon_click = photon_fired_0 | photon_fired_1

        # Spurious clicks from external light (e.g. trojan-horse probe): a random detector.
        if spurious_click_prob is not None:
            spurious = self.rng.random(n) < np.asarray(spurious_click_prob)
            spurious_det = self.rng.integers(0, 2, size=n).astype(np.int8)
        else:
            spurious = np.zeros(n, dtype=bool)
            spurious_det = np.zeros(n, dtype=np.int8)

        fired_0 = photon_fired_0 | (is_dark & (dark_det == 0)) | (spurious & (spurious_det == 0))
        fired_1 = photon_fired_1 | (is_dark & (dark_det == 1)) | (spurious & (spurious_det == 1))
        detected = fired_0 | fired_1
        is_double_click = fired_0 & fired_1
        outcomes = fired_1.astype(np.int8)
        # Double click: both detectors fired; Bob records it and assigns a random bit
        # (the standard "squashing" treatment, e.g. Beaudry, Moroder & Lutkenhaus 2008).
        outcomes[is_double_click] = self.rng.integers(0, 2, size=int(np.sum(is_double_click))).astype(np.int8)
        is_dark_count = detected & ~photon_click  # click without the signal photon

        # Empty-gate monitor: Bob also opens one detection gate per slot in which no signal
        # is expected and counts clicks there. Only dark counts can fire it, so it measures
        # the detectors' in-situ dark-count rate (zero when they are blinded).
        empty_gate_click = self.rng.random(n) < dark_p

        # Build result list
        measurements = []
        for i in range(n):
            measurements.append(BobMeasurement(
                basis=int(bases[i]),
                outcome=int(outcomes[i]),
                prob_0=float(prob_0[i]),
                prob_1=float(prob_1[i]),
                detected=bool(detected[i]),
                is_dark_count=bool(is_dark_count[i]),
                is_double_click=bool(is_double_click[i]),
                empty_gate_click=bool(empty_gate_click[i]),
            ))
        
        self._measurements.extend(measurements)
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