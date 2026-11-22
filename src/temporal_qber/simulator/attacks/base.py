"""
Base class for attack implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

from ..alice import Basis, Bit, BB84_STATES, measure_state_in_basis, sample_measurement


def sample_nonempty_photon_number(rng: np.random.Generator, mu: float) -> int:
    """Photon number of a non-empty weak coherent pulse (zero-truncated Poisson).

    Alice's source emits one qubit per transmission, so every transmission an attack
    sees is a non-empty pulse. Drawing n = 0 from a plain Poisson(mu) and passing the
    photon through untouched diluted multi-photon attacks by the vacuum fraction e^-mu
    (67-90% for mu = 0.1-0.4). Uses one uniform draw and inverts the CDF on [e^-mu, 1).
    """
    p = cdf = np.exp(-mu)
    u = rng.uniform(cdf, 1.0)
    n = 0
    while cdf < u:
        n += 1
        p *= mu / n
        cdf += p
    return max(n, 1)


@dataclass
class AttackResult:
    """Result of an attack on a single transmission."""
    intercepted: bool
    eve_basis: Optional[int]  # Eve's chosen basis (if intercepted)
    eve_outcome: Optional[int]  # Eve's measurement outcome (if intercepted)
    original_state: np.ndarray
    transmitted_state: np.ndarray  # State sent to Bob (modified if attacked)
    eve_info: Dict[str, Any]  # Additional attack-specific information

    # --- Detector side-channel effects on Bob (keyword-only; defaults = no effect) ---
    # Attacks such as time-shift, faked states, detector blinding and trojan-horse probe
    # light act on Bob's detectors rather than on the qubit, so they are passed to
    # Bob.measure_batch() instead of being encoded in transmitted_state.
    # Multipliers (0..1) on Bob's detection efficiency when the bit-0 / bit-1 detector
    # would fire (detector efficiency mismatch exploited by timing).
    bob_eff_scale: Tuple[float, float] = field(default=(1.0, 1.0), kw_only=True)
    # Probability that Eve's light makes one of Bob's detectors click (random detector).
    bob_spurious_click_prob: float = field(default=0.0, kw_only=True)
    # Multiplier on Bob's dark-count probability (0 = detectors blinded by bright light).
    bob_dark_scale: float = field(default=1.0, kw_only=True)
    # Photons sent on to Bob when the simulation tracks photon numbers (decoy-state
    # mode). None = the pulse's photon number is unchanged. Used by attacks that remove
    # photons (PNS, beam splitting).
    photons_to_bob: Optional[int] = field(default=None, kw_only=True)


class BaseAttack(ABC):
    """Abstract base class for BB84 attacks."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        # Bob's basis for the transmission being attacked, when the simulation provides it.
        # Only attacks that physically read Bob's apparatus (trojan horse on Bob's
        # modulator, detector blinding/faked states) may use it.
        self.bob_basis: Optional[int] = None
        # Photon number of Alice's pulse for the transmission being attacked, when the
        # simulation tracks photon numbers (decoy-state mode); None = single-photon model.
        self.photon_number: Optional[int] = None
    
    @abstractmethod
    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """
        Perform attack on a single transmission.
        
        Args:
            state: Quantum state from Alice
            alice_basis: Alice's basis (0=Z, 1=X)
            alice_bit: Alice's bit value
            
        Returns:
            AttackResult with attack details
        """
        pass
    
    def attack_batch(
        self, 
        states: np.ndarray,
        alice_bases: np.ndarray,
        alice_bits: np.ndarray,
        bob_bases: Optional[np.ndarray] = None,
        photon_numbers: Optional[np.ndarray] = None,
    ) -> list[AttackResult]:
        """
        Attack a batch of transmissions.

        Args:
            states: Array of state vectors (n, 2) complex
            alice_bases: Alice's basis choices
            alice_bits: Alice's bit values
            bob_bases: Bob's basis choices, exposed per transmission as self.bob_basis
            photon_numbers: Photon number of each of Alice's pulses, exposed as
                self.photon_number (decoy-state mode)

        Returns:
            List of AttackResult
        """
        n = states.shape[0]
        results = []
        for i in range(n):
            self.bob_basis = None if bob_bases is None else int(bob_bases[i])
            self.photon_number = None if photon_numbers is None else int(photon_numbers[i])
            result = self.attack(states[i], int(alice_bases[i]), int(alice_bits[i]))
            results.append(result)
        return results


class NoAttack(BaseAttack):
    """No attack - passes state through unchanged."""
    
    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        return AttackResult(
            intercepted=False,
            eve_basis=None,
            eve_outcome=None,
            original_state=state.copy(),
            transmitted_state=state.copy(),
            eve_info={}
        )