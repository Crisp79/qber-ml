"""
Base class for attack implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from ..alice import Basis, Bit, BB84_STATES, measure_state_in_basis, sample_measurement


@dataclass
class AttackResult:
    """Result of an attack on a single transmission."""
    intercepted: bool
    eve_basis: Optional[int]  # Eve's chosen basis (if intercepted)
    eve_outcome: Optional[int]  # Eve's measurement outcome (if intercepted)
    original_state: np.ndarray
    transmitted_state: np.ndarray  # State sent to Bob (modified if attacked)
    eve_info: Dict[str, Any]  # Additional attack-specific information


class BaseAttack(ABC):
    """Abstract base class for BB84 attacks."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
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
        alice_bits: np.ndarray
    ) -> list[AttackResult]:
        """
        Attack a batch of transmissions.
        
        Args:
            states: Array of state vectors (n, 2) complex
            alice_bases: Alice's basis choices
            alice_bits: Alice's bit values
            
        Returns:
            List of AttackResult
        """
        n = states.shape[0]
        results = []
        for i in range(n):
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