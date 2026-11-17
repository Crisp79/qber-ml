"""
Trojan Horse Attack Implementation.

The Trojan Horse attack (also called back-reflection attack) exploits the fact that
Bob's optical components (modulators, detectors) can reflect or backscatter a small
fraction of Eve's bright probe light back to Eve. By analyzing this back-reflected light,
Eve can gain information about Bob's basis choice or internal settings.

Key concept:
1. Eve injects bright probe pulses into the quantum channel (towards Bob)
2. Bob's optical components (phase modulator, Faraday mirror, detectors) back-reflect
3. Eve measures the back-reflected light to infer Bob's basis choice
4. This is a side-channel attack that doesn't directly disturb the quantum signal

Types of Trojan Horse attacks:
- Passive: Eve measures natural back-reflection from Bob's components
- Active: Eve injects bright pulses to enhance back-reflection
- Large pulse: High-power pulses to saturate Bob's detectors
- Small pulse: Low-power pulses below Bob's detection threshold

Implementation notes:
- Bob's basis modulator (or polarization controller) reflects a fraction of incident light
- The reflection encodes Bob's basis choice in the returned light's properties
- Eve can measure the returning light to infer Bob's basis
- The attack introduces some channel loss but minimal QBER
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .base import BaseAttack, AttackResult
from ..alice import Basis, Bit, BB84_STATES, measure_state_in_basis, sample_measurement


@dataclass
class TrojanHorseResult(AttackResult):
    """Result of a Trojan Horse attack on a single transmission."""
    eve_probe_photons: int  # Number of probe photons sent by Eve
    back_reflection_coeff: float  # Fraction of probe light reflected back
    bob_basis_revealed: bool  # Whether Eve correctly inferred Bob's basis
    eve_basis_guess: Optional[int] = None  # Eve's guess of Bob's basis
    # All other fields inherited from AttackResult:
    # intercepted, eve_basis, eve_outcome, original_state, transmitted_state, eve_info


class TrojanHorseAttack(BaseAttack):
    """
    Trojan Horse attack (back-reflection attack).
    
    Eve injects bright probe pulses into the quantum channel towards Bob.
    Bob's optical components (phase modulator, polarization controller) 
    back-reflect a fraction of this light, which encodes Bob's basis choice.
    Eve measures the back-reflected light to infer Bob's basis.
    
    The attack has two components:
    1. Basis inference: Eve guesses Bob's basis from back-reflection
    2. Signal disturbance: Some probe light may enter Bob's detectors
    """
    
    def __init__(
        self,
        probe_photons: int = 1000,  # Number of photons in Eve's probe pulse
        back_reflection_coeff: float = 1e-4,  # Fraction reflected back (typically 10^-4 to 10^-6)
        basis_inference_prob: float = 0.9,  # Probability Eve correctly infers basis from reflection
        detector_saturation_prob: float = 0.0,  # Probability probe light saturates Bob's detectors
        seed: Optional[int] = None
    ):
        """
        Initialize Trojan Horse attack.
        
        Args:
            probe_photons: Number of photons in Eve's probe pulse
            back_reflection_coeff: Fraction of probe light reflected by Bob's components
            basis_inference_prob: Probability Eve correctly infers Bob's basis from reflection
            detector_saturation_prob: Probability probe light causes detector saturation
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        if probe_photons <= 0:
            raise ValueError(f"probe_photons must be > 0, got {probe_photons}")
        if not 0 <= back_reflection_coeff <= 1:
            raise ValueError(f"back_reflection_coeff must be in [0, 1], got {back_reflection_coeff}")
        if not 0 <= basis_inference_prob <= 1:
            raise ValueError(f"basis_inference_prob must be in [0, 1], got {basis_inference_prob}")
        if not 0 <= detector_saturation_prob <= 1:
            raise ValueError(f"detector_saturation_prob must be in [0, 1], got {detector_saturation_prob}")
        
        self.probe_photons = probe_photons
        self.R = back_reflection_coeff
        self.basis_inference_prob = basis_inference_prob
        self.detector_saturation_prob = detector_saturation_prob
        
        # Number of photons returning to Eve (Poisson)
        self.mean_returned = probe_photons * back_reflection_coeff
    
    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """
        Perform Trojan Horse attack on a single transmission.
        
        Eve injects probe pulse towards Bob. Bob's basis modulator
        reflects a fraction back to Eve, revealing his basis choice.
        
        For simulation:
        1. Eve sends probe pulse
        2. Bob's basis choice affects back-reflection (different for Z vs X)
        3. Eve measures returned light and guesses Bob's basis
        4. With some probability, probe light saturates Bob's detectors
        """
        # Bob's basis choice (we need to know this for the attack)
        # In real attack, Eve doesn't know this - she infers it
        # For simulation, we assume Bob's basis is what we track
        
        # Simulate back-reflection: different for Z and X basis
        # This is a simplification - in reality the modulator reflection
        # properties differ between basis states
        reflected_photons = self.rng.poisson(self.mean_returned)
        
        # Eve's basis inference
        if reflected_photons > 0 and self.rng.random() < self.basis_inference_prob:
            # Eve correctly infers Bob's basis
            # In practice, Eve would need to know Bob's basis choices statistically
            # Here we model that the reflection pattern reveals the basis
            eve_basis_guess = None  # Would be inferred from reflection pattern
            bob_basis_revealed = True
        else:
            # Eve guesses randomly
            eve_basis_guess = self.rng.integers(0, 2)
            bob_basis_revealed = False
        
        # Check if probe light saturates Bob's detectors
        detector_saturated = self.rng.random() < self.detector_saturation_prob
        
        # The original quantum state is largely unaffected
        # (Trojan Horse is primarily a side-channel attack)
        transmitted_state = state.copy()
        
        # If detectors are saturated, Bob gets no detection
        if detector_saturated:
            transmitted_state = np.zeros_like(state)  # Bob sees vacuum
        
        return TrojanHorseResult(
            intercepted=True,
            eve_probe_photons=self.probe_photons,
            back_reflection_coeff=self.R,
            bob_basis_revealed=bob_basis_revealed,
            eve_basis_guess=eve_basis_guess,
            eve_basis=None,
            eve_outcome=None,
            original_state=state.copy(),
            transmitted_state=transmitted_state,
            eve_info={
                "probe_photons": self.probe_photons,
                "reflected_photons": reflected_photons,
                "mean_returned": self.mean_returned,
                "basis_inference_prob": self.basis_inference_prob,
                "bob_basis_revealed": bob_basis_revealed,
                "eve_basis_guess": eve_basis_guess,
                "detector_saturated": detector_saturated,
                "attack_type": "trojan_horse",
                "note": "Eve injects probe pulses, measures back-reflection to infer Bob's basis"
            }
        )
    
    def get_theoretical_stats(self) -> Dict[str, float]:
        """Get theoretical attack statistics."""
        return {
            "probe_photons": self.probe_photons,
            "back_reflection_coeff": self.R,
            "mean_returned_photons": self.mean_returned,
            "basis_inference_prob": self.basis_inference_prob,
            "detector_saturation_prob": self.detector_saturation_prob,
            "p_eve_detects_basis": self.basis_inference_prob * (1 - np.exp(-self.mean_returned)),
        }


class TrojanHorseActiveAttack(TrojanHorseAttack):
    """
    Active Trojan Horse attack with large pulse injection.
    
    Eve injects high-power pulses to actively probe Bob's optical components.
    This can saturate Bob's detectors or force them into linear mode.
    """
    
    def __init__(
        self,
        probe_photons: int = 10000,  # High-power probe
        back_reflection_coeff: float = 1e-4,
        basis_inference_prob: float = 0.95,  # Better inference with more photons
        detector_saturation_prob: float = 0.1,  # High power can saturate
        seed: Optional[int] = None
    ):
        super().__init__(
            probe_photons=probe_photons,
            back_reflection_coeff=back_reflection_coeff,
            basis_inference_prob=basis_inference_prob,
            detector_saturation_prob=detector_saturation_prob,
            seed=seed
        )
    
    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """
        Perform active Trojan Horse attack.
        
        High-power probe increases back-reflection but also increases
        probability of detector saturation.
        """
        # More photons = better statistics for basis inference
        reflected_photons = self.rng.poisson(self.mean_returned)
        
        # Improved basis inference with more photons
        if reflected_photons > 0 and self.rng.random() < self.basis_inference_prob:
            eve_basis_guess = None  # Would be inferred
            bob_basis_revealed = True
        else:
            eve_basis_guess = self.rng.integers(0, 2)
            bob_basis_revealed = False
        
        # Higher probability of detector saturation
        detector_saturated = self.rng.random() < self.detector_saturation_prob
        
        transmitted_state = state.copy()
        if detector_saturated:
            transmitted_state = np.zeros_like(state)
        
        return TrojanHorseResult(
            intercepted=True,
            eve_probe_photons=self.probe_photons,
            back_reflection_coeff=self.R,
            bob_basis_revealed=bob_basis_revealed,
            eve_basis_guess=eve_basis_guess,
            eve_basis=None,
            eve_outcome=None,
            original_state=state.copy(),
            transmitted_state=transmitted_state,
            eve_info={
                "probe_photons": self.probe_photons,
                "reflected_photons": reflected_photons,
                "mean_returned": self.mean_returned,
                "basis_inference_prob": self.basis_inference_prob,
                "bob_basis_revealed": bob_basis_revealed,
                "eve_basis_guess": eve_basis_guess,
                "detector_saturated": detector_saturated,
                "attack_type": "trojan_horse_active",
                "note": "Active Trojan Horse: high-power probe pulses for better inference"
            }
        )