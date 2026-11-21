"""
Fake State Attack Implementation.

The fake state attack (also called tailored state attack) is a powerful attack
where Eve exploits mismatches between the theoretical security proof assumptions
and the actual implementation of the QKD system.

In this attack, Eve prepares and sends quantum states that are specifically
crafted to exploit vulnerabilities in Bob's measurement apparatus, while still
appearing legitimate to the security analysis.

Types of fake state attacks:
1. Phase-remapped attack: Eve sends states with different phases than expected
2. Wavelength attack: Eve sends photons at wavelengths Bob's detectors don't expect
3. Time-bin attack: Eve manipulates timing information
4. Intensity attack: Eve sends multi-photon pulses while claiming single-photon
5. Trojan horse combination: Fake states combined with back-reflection

Key idea:
- Security proofs assume Bob's measurement device behaves in a certain way
- Real devices have implementation flaws (detector efficiency mismatch, 
  spectral sensitivity, timing response, etc.)
- Eve crafts states that exploit these flaws while passing security checks
- The attack can be tailored to a specific QKD implementation

Implementation notes:
- We model this as Eve sending states that are not the standard BB84 states
- But which Bob's measurement apparatus interprets in a way favorable to Eve
- The attack can be combined with other attacks
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .base import BaseAttack, AttackResult
from ..alice import Basis, Bit, BB84_STATES, measure_state_in_basis, sample_measurement


@dataclass
class FakeStateResult(AttackResult):
    """Result of a fake state attack on a single transmission."""
    attack_subtype: str  # Type of fake state attack
    fake_state_params: Dict[str, Any]  # Parameters of the fake state
    bob_basis_guess: Optional[int] = None
    detector_response: Optional[Dict[str, float]] = None
    # All other fields inherited from AttackResult:
    # intercepted, eve_basis, eve_outcome, original_state, transmitted_state, eve_info


class FakeStateAttack(BaseAttack):
    """
    Fake State Attack (Tailored State Attack).
    
    Eve exploits implementation flaws in Bob's measurement apparatus by
    sending carefully crafted quantum states that are not the standard
    BB84 states but exploit Bob's detector imperfections.
    
    Attack subtypes:
    - phase_remapped: Eve shifts the phase of states (basis-asymmetric errors)
    - detector_mismatch: faked states exploiting detector efficiency mismatch - Bob clicks
      (almost) only when he measures in Eve's basis (Makarov & Hjelme 2005)
    - blinding: detectors blinded by bright light and triggered by bright pulses - Bob
      clicks only in Eve's basis and has no dark counts (Lydersen et al. 2010)
    - wavelength, intensity, time_bin: legacy subtypes with no observable effect at Bob
    - combined: phase shift (as phase_remapped)
    """
    
    def __init__(
        self,
        attack_subtype: str = "phase_remapped",
        phase_shift: float = 0.0,  # Phase shift in radians
        det0_efficiency: float = 1.0,  # Detector 0 efficiency
        det1_efficiency: float = 1.0,  # Detector 1 efficiency
        multi_photon_prob: float = 0.0,  # Probability of sending multi-photon pulse
        wavelength_shift: float = 0.0,  # Wavelength shift (nm)
        seed: Optional[int] = None,
        mismatch_residual: float = 0.0,  # detector_mismatch: Bob's residual efficiency outside Eve's basis
    ):
        """
        Initialize fake state attack.
        
        Args:
            attack_subtype: Type of fake state attack
            phase_shift: Phase shift applied to states (radians)
            det0_efficiency: Detector 0 efficiency (0 to 1)
            det1_efficiency: Detector 1 efficiency (0 to 1)
            multi_photon_prob: Probability of sending multi-photon pulse
            wavelength_shift: Wavelength shift in nm
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        
        valid_subtypes = ["phase_remapped", "detector_mismatch", "blinding", "wavelength",
                         "intensity", "combined", "time_bin"]
        if not 0 <= mismatch_residual <= 1:
            raise ValueError(f"mismatch_residual must be in [0, 1], got {mismatch_residual}")
        self.mismatch_residual = mismatch_residual
        if attack_subtype not in valid_subtypes:
            raise ValueError(f"attack_subtype must be one of {valid_subtypes}, "
                           f"got {attack_subtype}")
        
        self.attack_subtype = attack_subtype
        self.phase_shift = phase_shift
        self.det0_efficiency = np.clip(det0_efficiency, 0.0, 1.0)
        self.det1_efficiency = np.clip(det1_efficiency, 0.0, 1.0)
        self.multi_photon_prob = np.clip(multi_photon_prob, 0.0, 1.0)
        self.wavelength_shift = wavelength_shift
        
        # Eve's basis choice probability (unbiased)
        self.p_z = 0.5
    
    def _apply_phase_shift(self, state: np.ndarray, basis: int) -> np.ndarray:
        """Apply phase shift to a state in given basis."""
        if basis == Basis.Z:
            # Phase shift in Z basis: Z rotation
            return np.array([
                state[0] * np.exp(1j * self.phase_shift / 2),
                state[1] * np.exp(-1j * self.phase_shift / 2)
            ], dtype=np.complex128)
        else:
            # Phase shift in X basis: rotation around X axis
            # This is more complex; we apply it in Z basis and convert
            return self._apply_phase_shift_in_basis(state, self.phase_shift)
    
    def _apply_phase_shift_in_basis(self, state: np.ndarray, phase: float) -> np.ndarray:
        """Apply phase shift (Z rotation) to any state."""
        return np.array([
            state[0] * np.exp(1j * phase / 2),
            state[1] * np.exp(-1j * phase / 2)
        ], dtype=np.complex128)
    
    def _apply_detector_mismatch(self, prob_0: float, prob_1: float, basis: int) -> tuple:
        """Apply detector efficiency mismatch to measurement probabilities."""
        if basis == Basis.Z:
            # Detector 0 measures |0>, Detector 1 measures |1>
            p0 = prob_0 * self.det0_efficiency
            p1 = prob_1 * self.det1_efficiency
        else:
            # In X basis, the mapping is different
            # This is a simplification - real mapping depends on detector design
            p0 = prob_0 * self.det0_efficiency
            p1 = prob_1 * self.det1_efficiency
        
        total = p0 + p1
        if total > 0:
            return p0 / total, p1 / total
        return 0.5, 0.5
    
    def _measure_state_in_basis(self, state: np.ndarray, basis: int) -> tuple[float, float]:
        """Calculate measurement probabilities for a state in a given basis."""
        if basis == Basis.Z:
            prob_0 = np.abs(state[0])**2
            prob_1 = np.abs(state[1])**2
        else:  # Basis.X
            plus = BB84_STATES[(Basis.X, Bit.ZERO)]
            minus = BB84_STATES[(Basis.X, Bit.ONE)]
            prob_0 = np.abs(np.vdot(plus, state))**2
            prob_1 = np.abs(np.vdot(minus, state))**2
        return float(prob_0), float(prob_1)
    
    def _prepare_state(self, basis: int, bit: int) -> np.ndarray:
        """Prepare a standard BB84 state."""
        return BB84_STATES[(basis, bit)].copy()
    
    def _create_fake_state(self, basis: int, bit: int) -> np.ndarray:
        """Create a fake state based on attack subtype."""
        if self.attack_subtype == "phase_remapped":
            # Apply phase shift to the standard state
            return self._apply_phase_shift(self._prepare_state(basis, bit), basis)
        elif self.attack_subtype == "detector_mismatch":
            # Standard state, but Bob's detectors have mismatch
            return self._prepare_state(basis, bit)
        elif self.attack_subtype == "intensity":
            # Multi-photon state (for simulation, we just track probability)
            return self._prepare_state(basis, bit)
        elif self.attack_subtype == "wavelength":
            # Wavelength shift - for simulation, this affects detector response
            return self._prepare_state(basis, bit)
        elif self.attack_subtype == "time_bin":
            # Time-bin manipulation - affects detector timing response
            return self._prepare_state(basis, bit)
        else:  # combined
            # Apply all effects
            state = self._prepare_state(basis, bit)
            state = self._apply_phase_shift(state, basis)
            return state
    
    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """
        Perform fake state attack on a single transmission.
        
        Eve intercepts Alice's state and replaces it with a carefully
        crafted fake state designed to exploit Bob's detector imperfections.
        """
        if self.attack_subtype in ("detector_mismatch", "blinding"):
            return self._faked_state_attack(state)

        # Determine which basis Eve will use for her fake state
        # Eve doesn't know Alice's basis, so she guesses or uses a fixed strategy
        eve_basis = self.rng.integers(0, 2)  # Random basis guess
        
        # Eve prepares a fake state.
        # For detector_mismatch subtype, Eve sends Alice's actual state so Bob's
        # detector mismatch causes bias, not random outcomes (which would give 50% QBER).
        # For phase_remapped and combined subtypes, Eve sends a phase-shifted copy
        # of Alice's state.
        if self.attack_subtype in ("detector_mismatch", "intensity", "wavelength"):
            # Send Alice's actual state (channel loss / detector effects handle the rest)
            fake_state = state.copy()
        elif self.attack_subtype == "phase_remapped":
            fake_state = self._apply_phase_shift(state.copy(), alice_basis)
        elif self.attack_subtype == "time_bin":
            fake_state = state.copy()
        else:  # combined
            # Apply all effects to Alice's state
            fake_state = self._apply_phase_shift(state.copy(), alice_basis)
        
        # Check if this is a multi-photon pulse
        is_multi_photon = self.rng.random() < self.multi_photon_prob
        
        # Apply detector mismatch effects
        # Bob's measurement will be affected by detector efficiencies
        
        # Bob's actual basis choice
        bob_basis = self.rng.integers(0, 2)
        
        # Calculate quantum probabilities
        prob_0, prob_1 = self._measure_state_in_basis(fake_state, bob_basis)
        
        # Apply detector mismatch if applicable
        if self.attack_subtype in ["detector_mismatch", "combined"]:
            prob_0, prob_1 = self._apply_detector_mismatch(prob_0, prob_1, bob_basis)
        
        # Sample Bob's outcome
        bob_outcome = 0 if self.rng.random() < prob_0 else 1
        
        # Eve's basis guess (she tries to infer Bob's basis from her knowledge)
        if self.attack_subtype in ["detector_mismatch", "combined"]:
            # Eve can infer Bob's basis from the mismatch pattern
            if self.rng.random() < 0.8:
                bob_basis_guess = bob_basis
            else:
                bob_basis_guess = self.rng.integers(0, 2)
        else:
            bob_basis_guess = None
        
        # Record detector response
        detector_response = {
            "det0_efficiency": self.det0_efficiency,
            "det1_efficiency": self.det1_efficiency,
        }
        
        # Record fake state parameters
        fake_state_params = {
            "attack_subtype": self.attack_subtype,
            "phase_shift": self.phase_shift,
            "det0_efficiency": self.det0_efficiency,
            "det1_efficiency": self.det1_efficiency,
            "multi_photon_prob": self.multi_photon_prob,
            "wavelength_shift": self.wavelength_shift,
            "is_multi_photon": is_multi_photon,
            "eve_basis_used": eve_basis,
        }
        
        return FakeStateResult(
            intercepted=True,
            attack_subtype=self.attack_subtype,
            fake_state_params=fake_state_params,
            bob_basis_guess=bob_basis_guess,
            detector_response=detector_response,
            eve_basis=None,
            eve_outcome=None,
            original_state=state.copy(),
            transmitted_state=fake_state,
            eve_info={
                "attack_subtype": self.attack_subtype,
                "phase_shift": self.phase_shift,
                "det0_efficiency": self.det0_efficiency,
                "det1_efficiency": self.det1_efficiency,
                "multi_photon_prob": self.multi_photon_prob,
                "wavelength_shift": self.wavelength_shift,
                "bob_basis_guess": bob_basis_guess,
                "attack_type": f"fake_state_{self.attack_subtype}",
                "note": f"Fake state attack ({self.attack_subtype}): Eve sends crafted states to exploit Bob's detector imperfections"
            }
        )
    
    def _faked_state_attack(self, state: np.ndarray) -> AttackResult:
        """
        Faked-state attacks that make Bob click only when he measures in Eve's basis.

        Eve measures Alice's photon in a random basis and resends a faked state that Bob's
        detectors register only if his basis matches hers:
        - "detector_mismatch": faked states timed into the detector efficiency-mismatch
          region (Makarov & Hjelme, J. Mod. Opt. 52, 691 (2005)). In the other basis
          Bob's detectors keep a residual efficiency mismatch_residual; those clicks are
          random, i.e. an error on half of the sifted bits they produce.
        - "blinding": detectors blinded by bright light and triggered by bright pulses
          (Lydersen et al., Nat. Photonics 4, 686 (2010)). No click in the other basis
          and no dark counts (blinded detectors are in linear mode).
        Whenever Bob clicks on a sifted bit, Eve's basis equals Alice's, so errors come
        only from residual clicks. The observable signature is the loss of about half of
        Bob's clicks (and, for blinding, of his dark counts).
        """
        eve_basis = int(self.rng.integers(0, 2))
        prob_0, _ = self._measure_state_in_basis(state, eve_basis)
        eve_outcome = 0 if self.rng.random() < prob_0 else 1
        blinding = self.attack_subtype == "blinding"
        bases_match = self.bob_basis is None or self.bob_basis == eve_basis
        scale = 1.0 if bases_match else (0.0 if blinding else self.mismatch_residual)
        return FakeStateResult(
            intercepted=True,
            attack_subtype=self.attack_subtype,
            fake_state_params={
                "attack_subtype": self.attack_subtype,
                "mismatch_residual": 0.0 if blinding else self.mismatch_residual,
                "eve_basis_used": eve_basis,
            },
            bob_basis_guess=self.bob_basis,
            eve_basis=eve_basis,
            eve_outcome=eve_outcome,
            original_state=state.copy(),
            transmitted_state=self._prepare_state(eve_basis, eve_outcome),
            eve_info={
                "attack_subtype": self.attack_subtype,
                "bob_clicks_allowed": bool(bases_match),
                "attack_type": f"fake_state_{self.attack_subtype}",
            },
            bob_eff_scale=(scale, scale),
            bob_dark_scale=0.0 if blinding else 1.0,
        )

    def get_theoretical_stats(self) -> Dict[str, float]:
        """Get theoretical attack statistics."""
        stats = {
            "attack_subtype": self.attack_subtype,
            "phase_shift": self.phase_shift,
            "det0_efficiency": self.det0_efficiency,
            "det1_efficiency": self.det1_efficiency,
            "detector_mismatch": abs(self.det0_efficiency - self.det1_efficiency),
            "multi_photon_prob": self.multi_photon_prob,
            "wavelength_shift": self.wavelength_shift,
        }
        
        if self.attack_subtype == "detector_mismatch":
            # Detector mismatch creates bias
            mismatch = abs(self.det0_efficiency - self.det1_efficiency)
            stats["expected_bias"] = mismatch / (self.det0_efficiency + self.det1_efficiency) if (self.det0_efficiency + self.det1_efficiency) > 0 else 0
        
        return stats