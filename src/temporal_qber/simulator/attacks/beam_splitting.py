"""
Beam Splitting Attack Implementation.

Eve uses a beam splitter to tap a fraction of the quantum signal.
This is a passive attack where Eve obtains a copy of the quantum state
without directly interacting with the main signal path.

Key characteristics:
- Eve couples a beam splitter into the quantum channel
- A fraction η of the signal goes to Eve, (1-η) goes to Bob
- Eve measures her portion in a randomly chosen basis
- Introduces loss (reduced intensity for Bob) but no additional errors
- With ideal single-photon sources, this reduces to a loss channel
- With weak coherent pulses (multi-photon), Eve can get full info on split photons

Implementation notes:
- For single photons: beam splitter creates a superposition; Eve gets photon with prob η
- For multi-photon pulses: Eve gets at least one photon with high probability
- Bob's detection probability is reduced by factor (1-η)
"""

import numpy as np
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .base import BaseAttack, AttackResult, sample_nonempty_photon_number
from ..alice import Basis, Bit, BB84_STATES, measure_state_in_basis, sample_measurement


@dataclass
class BeamSplitResult(AttackResult):
    """Result of a beam splitting attack on a single transmission."""
    eve_got_photon: bool
    transmission_coeff: float  # Fraction of signal that went to Bob
    reflection_coeff: float    # Fraction of signal that went to Eve
    # All other fields inherited from AttackResult:
    # intercepted, eve_basis, eve_outcome, original_state, transmitted_state, eve_info


class BeamSplittingAttack(BaseAttack):
    """
    Beam splitting attack.
    
    Eve taps a fraction of the quantum signal using a beam splitter.
    - Reflection coefficient R = fraction going to Eve
    - Transmission coefficient T = fraction going to Bob (T = 1 - R)
    
    For single-photon pulses:
    - Photon goes to Eve with probability R
    - Photon goes to Bob with probability T
    
    For multi-photon pulses (weak coherent pulses):
    - Each photon independently goes to Eve with prob R, Bob with prob T
    - Eve gets at least one photon with probability 1 - T^n
    """
    
    def __init__(
        self,
        reflection_coeff: float = 0.1,  # Fraction of signal tapped by Eve (R)
        mean_photon_number: float = 0.1,  # For multi-photon statistics
        seed: Optional[int] = None
    ):
        """
        Initialize beam splitting attack.
        
        Args:
            reflection_coeff: Fraction of signal going to Eve (0 to 1)
            mean_photon_number: Mean photon number of source (for WCP)
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        if not 0 <= reflection_coeff <= 1:
            raise ValueError(f"reflection_coeff must be in [0, 1], got {reflection_coeff}")
        if mean_photon_number <= 0:
            raise ValueError(f"mean_photon_number must be > 0, got {mean_photon_number}")
        
        self.R = reflection_coeff  # Reflection coefficient (Eve's share)
        self.T = 1.0 - reflection_coeff  # Transmission coefficient (Bob's share)
        self.mu = mean_photon_number
        
        # Pre-compute photon number probabilities
        self.photon_probs = {}
        for n in range(0, 10):
            self.photon_probs[n] = np.exp(-self.mu) * (self.mu ** n) / math.factorial(n)
        
        # Eve's basis choice probability (unbiased)
        self.p_z = 0.5
    
    def _sample_photon_number(self) -> int:
        """Sample the photon number of a non-empty pulse (see sample_nonempty_photon_number)."""
        return sample_nonempty_photon_number(self.rng, self.mu)
    
    def _choose_basis(self) -> int:
        """Eve randomly chooses a measurement basis."""
        return 0 if self.rng.random() < self.p_z else 1
    
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
        """Prepare a new quantum state based on Eve's measurement."""
        return BB84_STATES[(basis, bit)].copy()
    
    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """
        Perform beam splitting attack on a single transmission.
        
        1. Sample photon number from Poisson distribution
        2. For each photon, it goes to Eve with prob R, to Bob with prob T
        3. If Eve gets at least one photon, she measures it in random basis
        4. Bob receives remaining photons (if any)
        """
        n_photons = self._sample_photon_number()
        
        if n_photons == 0:
            # Vacuum pulse
            return BeamSplitResult(
                intercepted=False,
                eve_got_photon=False,
                transmission_coeff=self.T,
                reflection_coeff=self.R,
                eve_basis=None,
                eve_outcome=None,
                original_state=state.copy(),
                transmitted_state=state.copy(),
                eve_info={"photon_number": 0, "attack_type": "vacuum"}
            )
        
        # For each photon, determine if it goes to Eve or Bob
        n_to_eve = 0
        for _ in range(n_photons):
            if self.rng.random() < self.R:
                n_to_eve += 1
        n_to_bob = n_photons - n_to_eve
        
        eve_got_photon = n_to_eve > 0
        
        if not eve_got_photon:
            # All photons went to Bob
            return BeamSplitResult(
                intercepted=False,
                eve_got_photon=False,
                transmission_coeff=self.T,
                reflection_coeff=self.R,
                eve_basis=None,
                eve_outcome=None,
                original_state=state.copy(),
                transmitted_state=state.copy(),  # All photons to Bob
                eve_info={
                    "photon_number": n_photons,
                    "n_to_eve": 0,
                    "n_to_bob": n_to_bob,
                    "attack_type": "beam_split_bob_only"
                }
            )
        
        # Eve got at least one photon
        # She measures it in a random basis
        eve_basis = self._choose_basis()
        prob_0, prob_1 = self._measure_state_in_basis(state, eve_basis)
        eve_outcome = 0 if self.rng.random() < prob_0 else 1
        
        # Bob receives n_to_bob photons
        # For simulation, if Bob gets 0 photons, it's a loss
        # If Bob gets >= 1 photon, state is unchanged (we just track detection probability)
        if n_to_bob == 0:
            # Bob gets nothing - this is a loss event
            transmitted_state = np.zeros_like(state)  # Vacuum state for Bob
        else:
            # Bob gets at least one photon - state is effectively unchanged
            transmitted_state = state.copy()
        
        return BeamSplitResult(
            intercepted=True,
            eve_got_photon=True,
            transmission_coeff=self.T,
            reflection_coeff=self.R,
            eve_basis=eve_basis,
            eve_outcome=eve_outcome,
            original_state=state.copy(),
            transmitted_state=transmitted_state,
            eve_info={
                "photon_number": n_photons,
                "n_to_eve": n_to_eve,
                "n_to_bob": n_to_bob,
                "attack_type": "beam_split",
                "eve_basis": eve_basis,
                "eve_outcome": eve_outcome,
                "bob_gets_photon": n_to_bob > 0
            }
        )
    
    def get_theoretical_stats(self) -> Dict[str, float]:
        """Get theoretical attack statistics."""
        return {
            "reflection_coeff": self.R,
            "transmission_coeff": self.T,
            "mean_photon_number": self.mu,
            "p_eve_gets_photon": 1 - np.exp(-self.mu * self.R),  # Prob Eve gets ≥1 photon
            "p_bob_gets_photon": 1 - np.exp(-self.mu * self.T),  # Prob Bob gets ≥1 photon
            "p_both_get_photon": (1 - np.exp(-self.mu * self.R)) * (1 - np.exp(-self.mu * self.T)),
            "channel_loss_introduced": self.R,  # Fractional loss introduced
        }


class BeamSplittingAttackFixedPhoton(BeamSplittingAttack):
    """
    Beam splitting attack assuming fixed single-photon source.
    
    For ideal single-photon sources, the beam splitter acts as a loss channel:
    - Photon goes to Eve with probability R
    - Photon goes to Bob with probability T
    - No multi-photon advantages for Eve
    """
    
    def __init__(
        self,
        reflection_coeff: float = 0.1,
        seed: Optional[int] = None
    ):
        super().__init__(reflection_coeff, mean_photon_number=1.0, seed=seed)
        # Override to use exactly 1 photon
        self.fixed_photon = True
    
    def _sample_photon_number(self) -> int:
        """Always return 1 photon (ideal single-photon source)."""
        return 1
    
    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """
        Perform beam splitting on a single photon.
        
        Photon goes to Eve with prob R, to Bob with prob T.
        """
        # Single photon case
        if self.rng.random() < self.R:
            # Photon goes to Eve
            eve_basis = self._choose_basis()
            prob_0, prob_1 = self._measure_state_in_basis(state, eve_basis)
            eve_outcome = 0 if self.rng.random() < prob_0 else 1
            
            return BeamSplitResult(
                intercepted=True,
                eve_got_photon=True,
                transmission_coeff=self.T,
                reflection_coeff=self.R,
                eve_basis=eve_basis,
                eve_outcome=eve_outcome,
                original_state=state.copy(),
                transmitted_state=np.zeros_like(state),  # Bob gets vacuum
                eve_info={
                    "photon_number": 1,
                    "n_to_eve": 1,
                    "n_to_bob": 0,
                    "attack_type": "beam_split_fixed_eve",
                    "eve_basis": eve_basis,
                    "eve_outcome": eve_outcome
                }
            )
        else:
            # Photon goes to Bob
            return BeamSplitResult(
                intercepted=False,
                eve_got_photon=False,
                transmission_coeff=self.T,
                reflection_coeff=self.R,
                eve_basis=None,
                eve_outcome=None,
                original_state=state.copy(),
                transmitted_state=state.copy(),
                eve_info={
                    "photon_number": 1,
                    "n_to_eve": 0,
                    "n_to_bob": 1,
                    "attack_type": "beam_split_fixed_bob"
                }
            )