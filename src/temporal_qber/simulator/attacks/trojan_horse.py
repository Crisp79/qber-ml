"""
Trojan Horse Attack Implementation.

The Trojan Horse attack (also called back-reflection or large-pulse attack) exploits
the fact that Bob's optical components reflect a small fraction of Eve's bright probe
light back to her. The reflected light encodes the setting of Bob's basis modulator.

Model (Bob-side attack; Vakhitov, Makarov & Hjelme, J. Mod. Opt. 48, 2023 (2001);
Gisin, Fasel, Kraus, Zbinden & Ribordy, Phys. Rev. A 73, 022320 (2006)):
1. Just before each signal arrives, Eve sends a bright probe pulse into Bob's system.
   probe_photons * back_reflection_coeff photons (Poisson) return to her; if any
   return, she reads Bob's basis correctly with probability basis_inference_prob.
2. Read-out succeeded: Eve measures Alice's photon in Bob's basis and resends her
   result. On sifted bits (Alice's basis = Bob's basis) this introduces no error.
   Read-out failed: Eve leaves the photon alone (a stealthy Eve does not guess).
3. Side effects visible at Bob - the attack's only observable signature:
   - probe light leaking into Bob's detectors inside the detection gate makes spurious
     clicks with probability 1 - exp(-probe_photons * detector_leak_coeff); a spurious
     click alone gives a random bit, together with a signal click in the other detector
     a double click that Bob resolves at random;
   - bright probes can occasionally saturate a detector (detector_saturation_prob), so
     the signal is missed.
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
    Bob-side Trojan Horse attack: read Bob's basis from back-reflected probe light,
    then measure Alice's photon in that basis. See the module docstring for the model.
    """

    label = "trojan_horse"

    def __init__(
        self,
        probe_photons: int = 1000,  # Number of photons in Eve's probe pulse
        back_reflection_coeff: float = 1e-4,  # Fraction reflected back (typically 10^-4 to 10^-6)
        basis_inference_prob: float = 0.9,  # Probability Eve reads Bob's basis correctly
        detector_saturation_prob: float = 0.0,  # Probability probe light saturates Bob's detectors
        detector_leak_coeff: float = 0.0,  # Fraction of probe photons reaching Bob's detectors in-gate
        seed: Optional[int] = None
    ):
        """
        Initialize Trojan Horse attack.

        Args:
            probe_photons: Number of photons in Eve's probe pulse
            back_reflection_coeff: Fraction of probe light reflected back by Bob's components
            basis_inference_prob: Probability Eve reads Bob's basis when light returns
            detector_saturation_prob: Probability probe light saturates Bob's detector (signal missed)
            detector_leak_coeff: Fraction of probe photons that reach Bob's detectors inside
                the detection gate (sets the spurious-click probability)
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
        if not 0 <= detector_leak_coeff <= 1:
            raise ValueError(f"detector_leak_coeff must be in [0, 1], got {detector_leak_coeff}")

        self.probe_photons = probe_photons
        self.R = back_reflection_coeff
        self.basis_inference_prob = basis_inference_prob
        self.detector_saturation_prob = detector_saturation_prob
        self.detector_leak_coeff = detector_leak_coeff

        # Number of photons returning to Eve (Poisson mean)
        self.mean_returned = probe_photons * back_reflection_coeff
        # Probability that leaked probe light makes one of Bob's detectors click
        self.probe_click_prob = float(1.0 - np.exp(-probe_photons * detector_leak_coeff))

    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """Perform the Trojan Horse attack on a single transmission."""
        reflected_photons = self.rng.poisson(self.mean_returned)
        basis_read = (
            reflected_photons > 0
            and self.bob_basis is not None
            and self.rng.random() < self.basis_inference_prob
        )

        if basis_read:
            # Eve knows Bob's basis: measure Alice's photon in it and resend the result.
            eve_basis = self.bob_basis
            prob_0, _ = measure_state_in_basis(state, eve_basis)
            eve_outcome = 0 if self.rng.random() < prob_0 else 1
            transmitted_state = BB84_STATES[(eve_basis, eve_outcome)].copy()
        else:
            # Read-out failed: leave the photon alone.
            eve_basis = None
            eve_outcome = None
            transmitted_state = state.copy()

        detector_saturated = self.rng.random() < self.detector_saturation_prob
        if detector_saturated:
            transmitted_state = np.zeros_like(state)  # signal missed

        return TrojanHorseResult(
            intercepted=bool(basis_read),
            eve_probe_photons=self.probe_photons,
            back_reflection_coeff=self.R,
            bob_basis_revealed=bool(basis_read),
            eve_basis_guess=eve_basis,
            eve_basis=eve_basis,
            eve_outcome=eve_outcome,
            original_state=state.copy(),
            transmitted_state=transmitted_state,
            eve_info={
                "probe_photons": self.probe_photons,
                "reflected_photons": int(reflected_photons),
                "mean_returned": self.mean_returned,
                "basis_inference_prob": self.basis_inference_prob,
                "bob_basis_revealed": bool(basis_read),
                "detector_saturated": bool(detector_saturated),
                "probe_click_prob": self.probe_click_prob,
                "attack_type": self.label,
            },
            bob_spurious_click_prob=self.probe_click_prob,
        )

    def get_theoretical_stats(self) -> Dict[str, float]:
        """Get theoretical attack statistics."""
        return {
            "probe_photons": self.probe_photons,
            "back_reflection_coeff": self.R,
            "mean_returned_photons": self.mean_returned,
            "basis_inference_prob": self.basis_inference_prob,
            "detector_saturation_prob": self.detector_saturation_prob,
            "probe_click_prob": self.probe_click_prob,
            "p_eve_detects_basis": self.basis_inference_prob * (1 - np.exp(-self.mean_returned)),
        }


class TrojanHorseActiveAttack(TrojanHorseAttack):
    """
    Active Trojan Horse attack with large pulse injection: brighter probes give a
    better basis read-out but leak and saturate more.
    """

    label = "trojan_horse_active"

    def __init__(
        self,
        probe_photons: int = 10000,  # High-power probe
        back_reflection_coeff: float = 1e-4,
        basis_inference_prob: float = 0.95,  # Better inference with more photons
        detector_saturation_prob: float = 0.1,  # High power can saturate
        detector_leak_coeff: float = 0.0,
        seed: Optional[int] = None
    ):
        super().__init__(
            probe_photons=probe_photons,
            back_reflection_coeff=back_reflection_coeff,
            basis_inference_prob=basis_inference_prob,
            detector_saturation_prob=detector_saturation_prob,
            detector_leak_coeff=detector_leak_coeff,
            seed=seed
        )
