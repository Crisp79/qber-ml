"""
Photon Number Splitting (PNS) Attack Implementation.

In a practical QKD system using weak coherent pulses (WCP) instead of true single-photon sources,
some pulses contain multiple photons. Eve can exploit this by performing photon number splitting:

1. Eve measures the photon number of each pulse using a quantum non-demolition measurement
2. For multi-photon pulses (n >= 2), Eve splits off one photon and stores it
3. Eve lets the remaining photons pass through to Bob
4. Later, after basis reconciliation, Eve measures her stored photons in the correct basis
5. This gives Eve full information on multi-photon pulses without introducing errors

This attack is only possible with imperfect single-photon sources (WCP with mu > 0).
With ideal single-photon sources, PNS is not possible.

Implementation notes:
- We model the photon number distribution of weak coherent pulses using Poisson distribution
- Multi-photon probability: P(n >= 2) = 1 - P(0) - P(1) = 1 - e^(-mu) - mu*e^(-mu)
- For single-photon pulses, Eve must either let them pass or perform intercept-resend
- We assume Eve can perfectly distinguish photon numbers (ideal PNS)
"""

import numpy as np
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .base import BaseAttack, AttackResult, sample_nonempty_photon_number
from ..alice import Basis, Bit, BB84_STATES, measure_state_in_basis, sample_measurement


@dataclass
class PNSResult(AttackResult):
    """Result of a PNS attack on a single transmission."""
    photon_number: int
    # All other fields inherited from AttackResult:
    # intercepted, eve_basis, eve_outcome, original_state, transmitted_state, eve_info


class PNSAttack(BaseAttack):
    """
    Photon Number Splitting attack.
    
    Eve exploits multi-photon pulses from weak coherent pulse sources.
    For pulses with n >= 2 photons, Eve splits off one photon and stores it.
    For single-photon pulses, Eve either lets them pass or performs intercept-resend.
    """
    
    def __init__(
        self, 
        mean_photon_number: float = 0.1,  # mu parameter of WCP
        single_photon_strategy: str = "pass",  # "pass", "intercept_resend" or "block"
        seed: Optional[int] = None,
        single_photon_block_prob: float = 1.0,  # used by "block"
    ):
        """
        Initialize PNS attack.

        Args:
            mean_photon_number: Mean photon number (mu) of weak coherent pulses
            single_photon_strategy: What to do with single-photon pulses:
                - "pass": Let single-photon pulses pass unchanged (Eve gets no info;
                  the attack then has no observable effect at Bob)
                - "intercept_resend": Perform intercept-resend on single-photon pulses
                - "block": Block them with probability single_photon_block_prob - the
                  standard PNS attack (Brassard, Lutkenhaus, Mor & Sanders, PRL 85, 1330,
                  2000). Visible at Bob only as extra loss, never as errors.
            seed: Random seed for reproducibility
            single_photon_block_prob: Fraction of single-photon pulses Eve blocks ("block")
        """
        super().__init__(seed)
        if mean_photon_number <= 0:
            raise ValueError(f"mean_photon_number must be > 0, got {mean_photon_number}")
        if single_photon_strategy not in ("pass", "intercept_resend", "block"):
            raise ValueError(f"single_photon_strategy must be 'pass', 'intercept_resend' or 'block', got {single_photon_strategy}")
        if not 0 <= single_photon_block_prob <= 1:
            raise ValueError(f"single_photon_block_prob must be in [0, 1], got {single_photon_block_prob}")

        self.mu = mean_photon_number
        self.single_photon_strategy = single_photon_strategy
        self.block_prob = single_photon_block_prob
        
        # Pre-compute photon number probabilities (Poisson distribution)
        # P(n) = e^(-mu) * mu^n / n!
        self.photon_probs = {}
        for n in range(0, 10):  # Up to 9 photons should be enough for small mu
            self.photon_probs[n] = np.exp(-self.mu) * (self.mu ** n) / math.factorial(n)
        
        # Probability of multi-photon pulse (n >= 2)
        self.p_multi = 1 - self.photon_probs.get(0, 0) - self.photon_probs.get(1, 0)
        self.p_single = self.photon_probs.get(1, 0)
        self.p_vacuum = self.photon_probs.get(0, 0)
        
        # Eve's basis choice probability (unbiased)
        self.p_z = 0.5
    
    def _sample_photon_number(self) -> int:
        """Photon number of the pulse: Alice's actual number when the simulation tracks
        photon numbers (decoy-state mode), otherwise a non-empty pulse sampled here with
        mean_photon_number (see sample_nonempty_photon_number)."""
        if self.photon_number is not None:
            return self.photon_number
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

    def _block_or_pass_single(self, state: np.ndarray) -> "PNSResult":
        """"block" strategy: Eve cannot split a single photon, so she blocks it with
        probability block_prob (raising her information share of Bob's key) and lets the
        rest through. A blocked pulse reaches Bob as vacuum: loss, no error."""
        blocked = self.rng.random() < self.block_prob
        return PNSResult(
            photon_number=1,
            original_state=state.copy(),
            transmitted_state=np.zeros_like(state) if blocked else state.copy(),
            intercepted=blocked,
            eve_basis=None,
            eve_outcome=None,
            eve_info={"photon_number": 1,
                      "attack_type": "pns_single_blocked" if blocked else "pns_single_pass"},
            photons_to_bob=0 if blocked else 1,
        )

    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """
        Perform PNS attack on a single transmission.
        
        1. Sample photon number from Poisson distribution
        2. If n >= 2 (multi-photon): Eve splits one photon, measures it later in correct basis
           - For simulation: we assume Eve measures immediately in random basis
           - Eve lets remaining photons pass to Bob (state unchanged)
        3. If n == 1 (single-photon): 
           - "pass": Let it pass unchanged
           - "intercept_resend": Perform intercept-resend
        4. If n == 0 (vacuum): Nothing to do
        """
        # Sample photon number
        n_photons = self._sample_photon_number()
        
        if n_photons >= 2:
            # Multi-photon pulse: Eve splits one photon
            # She can measure it in a random basis now, or store for later
            # For simulation, we model it as immediate measurement in random basis
            eve_basis = self._choose_basis()
            prob_0, prob_1 = self._measure_state_in_basis(state, eve_basis)
            eve_outcome = 0 if self.rng.random() < prob_0 else 1
            
            # State passes through unchanged (remaining photons go to Bob)
            transmitted_state = state.copy()
            
            return PNSResult(
                photon_number=n_photons,
                original_state=state.copy(),
                transmitted_state=transmitted_state,
                intercepted=True,
                eve_basis=eve_basis,
                eve_outcome=eve_outcome,
                eve_info={
                    "photon_number": n_photons,
                    "attack_type": "pns_multi",
                    "eve_basis": eve_basis,
                    "eve_outcome": eve_outcome,
                    "note": "Eve split one photon from multi-photon pulse"
                },
                photons_to_bob=n_photons - 1,  # Eve keeps one photon
            )
        
        elif n_photons == 1:
            # Single-photon pulse
            if self.single_photon_strategy == "block":
                return self._block_or_pass_single(state)
            if self.single_photon_strategy == "pass":
                # Eve cannot attack without introducing errors
                return PNSResult(
                    photon_number=1,
                    original_state=state.copy(),
                    transmitted_state=state.copy(),
                    intercepted=False,
                    eve_basis=None,
                    eve_outcome=None,
                    eve_info={"photon_number": 1, "attack_type": "pns_single_pass"}
                )
            else:  # intercept_resend
                # Perform intercept-resend on single-photon pulses
                eve_basis = self._choose_basis()
                prob_0, prob_1 = self._measure_state_in_basis(state, eve_basis)
                eve_outcome = 0 if self.rng.random() < prob_0 else 1
                transmitted_state = self._prepare_state(eve_basis, eve_outcome)
                
                return PNSResult(
                    photon_number=1,
                    original_state=state.copy(),
                    transmitted_state=transmitted_state,
                    intercepted=True,
                    eve_basis=eve_basis,
                    eve_outcome=eve_outcome,
                    eve_info={
                        "photon_number": 1,
                        "attack_type": "pns_single_intercept",
                        "eve_basis": eve_basis,
                        "eve_outcome": eve_outcome
                    }
                )
        
        else:
            # Vacuum pulse (n = 0)
            return PNSResult(
                photon_number=0,
                original_state=state.copy(),
                transmitted_state=state.copy(),
                intercepted=False,
                eve_basis=None,
                eve_outcome=None,
                eve_info={"photon_number": 0, "attack_type": "vacuum"}
            )
    
    def get_theoretical_stats(self) -> Dict[str, float]:
        """Get theoretical attack statistics."""
        return {
            "mean_photon_number": self.mu,
            "p_vacuum": self.p_vacuum,
            "p_single": self.p_single,
            "p_multi": self.p_multi,
            "fraction_attackable": self.p_multi,  # Fraction of pulses Eve can attack perfectly
            "expected_qber_from_intercept": 0.25 * (1 - self.p_multi) if self.single_photon_strategy == "intercept_resend" else 0.0,
        }


class PNSAttackWithStorage(PNSAttack):
    """
    PNS attack with photon storage (more realistic).
    
    Eve stores the split photon and measures it AFTER basis reconciliation.
    This gives Eve perfect information on multi-photon pulses.
    """
    
    def __init__(
        self, 
        mean_photon_number: float = 0.1,
        single_photon_strategy: str = "pass",
        seed: Optional[int] = None,
        single_photon_block_prob: float = 1.0,
    ):
        super().__init__(mean_photon_number, single_photon_strategy, seed,
                         single_photon_block_prob=single_photon_block_prob)
    
    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """
        Perform PNS attack with storage.
        
        For multi-photon pulses: Eve splits one photon and stores it.
        She does NOT measure it immediately. Instead, she waits for basis
        reconciliation and then measures in the correct basis.
        
        For simulation purposes, we model this by assuming Eve will later
        measure in the correct basis, giving her perfect information.
        """
        n_photons = self._sample_photon_number()
        
        if n_photons >= 2:
            # Multi-photon: Eve splits one photon and stores it
            # She will measure in the correct basis after reconciliation
            # For now, we record the correct basis/bit for later measurement
            
            return PNSResult(
                photon_number=n_photons,
                original_state=state.copy(),
                transmitted_state=state.copy(),
                intercepted=True,
                eve_basis=alice_basis,
                eve_outcome=alice_bit,
                eve_info={
                    "photon_number": n_photons,
                    "attack_type": "pns_storage_multi",
                    "stored_photon": True,
                    "correct_basis": alice_basis,
                    "correct_bit": alice_bit,
                    "note": "Eve stores photon, measures in correct basis after reconciliation"
                },
                photons_to_bob=n_photons - 1,  # Eve keeps one photon
            )
        
        elif n_photons == 1:
            if self.single_photon_strategy == "block":
                return self._block_or_pass_single(state)
            if self.single_photon_strategy == "pass":
                return PNSResult(
                    photon_number=1,
                    original_state=state.copy(),
                    transmitted_state=state.copy(),
                    intercepted=False,
                    eve_basis=None,
                    eve_outcome=None,
                    eve_info={"photon_number": 1, "attack_type": "pns_storage_single_pass"}
                )
            else:
                eve_basis = self._choose_basis()
                prob_0, prob_1 = self._measure_state_in_basis(state, eve_basis)
                eve_outcome = 0 if self.rng.random() < prob_0 else 1
                transmitted_state = self._prepare_state(eve_basis, eve_outcome)
                
                return PNSResult(
                    photon_number=1,
                    original_state=state.copy(),
                    transmitted_state=transmitted_state,
                    intercepted=True,
                    eve_basis=eve_basis,
                    eve_outcome=eve_outcome,
                    eve_info={
                        "photon_number": 1,
                        "attack_type": "pns_storage_single_intercept",
                        "eve_basis": eve_basis,
                        "eve_outcome": eve_outcome
                    }
                )
        else:
            return PNSResult(
                photon_number=0,
                original_state=state.copy(),
                transmitted_state=state.copy(),
                intercepted=False,
                eve_basis=None,
                eve_outcome=None,
                eve_info={"photon_number": 0, "attack_type": "vacuum"}
            )