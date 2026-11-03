"""
Alice's quantum state preparation for BB84 protocol.

Implements the four BB84 states:
- Z basis, bit 0: |0> = [1, 0]
- Z basis, bit 1: |1> = [0, 1]
- X basis, bit 0: |+> = 1/sqrt(2) * [1, 1]
- X basis, bit 1: |-> = 1/sqrt(2) * [1, -1]
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import IntEnum


class Basis(IntEnum):
    """Measurement basis for BB84."""
    Z = 0  # Computational basis
    X = 1  # Hadamard basis


class Bit(IntEnum):
    """Bit value."""
    ZERO = 0
    ONE = 1


# BB84 quantum states as 2D complex vectors
# |0> = [1, 0]
# |1> = [0, 1]
# |+> = 1/sqrt(2) * [1, 1]
# |-> = 1/sqrt(2) * [1, -1]

SQRT2 = np.sqrt(2)

BB84_STATES = {
    # (basis, bit) -> state vector
    (Basis.Z, Bit.ZERO): np.array([1.0, 0.0], dtype=np.complex128),      # |0>
    (Basis.Z, Bit.ONE):  np.array([0.0, 1.0], dtype=np.complex128),      # |1>
    (Basis.X, Bit.ZERO): np.array([1.0/SQRT2, 1.0/SQRT2], dtype=np.complex128),  # |+>
    (Basis.X, Bit.ONE):  np.array([1.0/SQRT2, -1.0/SQRT2], dtype=np.complex128), # |->
}

STATE_LABELS = {
    (Basis.Z, Bit.ZERO): "|0>",
    (Basis.Z, Bit.ONE):  "|1>",
    (Basis.X, Bit.ZERO): "|+>",
    (Basis.X, Bit.ONE):  "|->",
}


@dataclass
class AliceTransmission:
    """Single transmission from Alice."""
    bit: int          # 0 or 1
    basis: int        # 0 (Z) or 1 (X)
    state: np.ndarray # 2D complex vector


class Alice:
    """Alice's BB84 quantum state preparation."""
    
    def __init__(self, seed: Optional[int] = None, p_z: float = 0.5):
        """
        Initialize Alice.
        
        Args:
            seed: Random seed for reproducibility
            p_z: Probability of choosing Z basis (default 0.5)
        """
        if not 0 <= p_z <= 1:
            raise ValueError(f"p_z must be in [0, 1], got {p_z}")
        
        self.rng = np.random.default_rng(seed)
        self.p_z = p_z
        self._transmissions: List[AliceTransmission] = []
    
    def generate_random_bit(self) -> int:
        """Generate a random bit (0 or 1)."""
        return int(self.rng.integers(0, 2))
    
    def generate_random_basis(self) -> int:
        """Generate a random basis (0=Z, 1=X) with probability p_z for Z."""
        return 0 if self.rng.random() < self.p_z else 1
    
    def prepare_state(self, bit: int, basis: int) -> np.ndarray:
        """
        Prepare a BB84 quantum state.
        
        Args:
            bit: 0 or 1
            basis: 0 (Z) or 1 (X)
            
        Returns:
            2D complex state vector
        """
        if bit not in (0, 1):
            raise ValueError(f"Bit must be 0 or 1, got {bit}")
        if basis not in (0, 1):
            raise ValueError(f"Basis must be 0 (Z) or 1 (X), got {basis}")
        
        return BB84_STATES[(basis, bit)].copy()
    
    def prepare_transmission(self) -> AliceTransmission:
        """
        Generate a single random BB84 transmission.
        
        Returns:
            AliceTransmission with bit, basis, and state
        """
        bit = self.generate_random_bit()
        basis = self.generate_random_basis()
        state = self.prepare_state(bit, basis)
        
        transmission = AliceTransmission(bit=bit, basis=basis, state=state)
        self._transmissions.append(transmission)
        return transmission
    
    def prepare_batch(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate n random BB84 transmissions.
        
        Args:
            n: Number of transmissions
            
        Returns:
            bits: Array of n bits (0 or 1)
            bases: Array of n bases (0 or 1)
            states: Array of n state vectors (n, 2) complex
        """
        bits = np.empty(n, dtype=np.int8)
        bases = np.empty(n, dtype=np.int8)
        states = np.empty((n, 2), dtype=np.complex128)
        
        for i in range(n):
            transmission = self.prepare_transmission()
            bits[i] = transmission.bit
            bases[i] = transmission.basis
            states[i] = transmission.state
        
        return bits, bases, states
    
    def get_transmissions(self) -> List[AliceTransmission]:
        """Get all prepared transmissions."""
        return self._transmissions.copy()
    
    def reset(self, seed: Optional[int] = None):
        """Reset Alice's state."""
        self._transmissions.clear()
        if seed is not None:
            self.rng = np.random.default_rng(seed)
    
    def get_state_info(self) -> dict:
        """Get configuration info."""
        return {
            "seed": getattr(self.rng, 'seed_seq', None),
            "p_z": self.p_z,
            "n_transmissions": len(self._transmissions),
        }


def measure_state_in_basis(state: np.ndarray, basis: int) -> Tuple[float, float]:
    """
    Calculate measurement probabilities for a state in a given basis.
    
    Args:
        state: 2D complex state vector
        basis: 0 (Z) or 1 (X)
        
    Returns:
        (prob_bit_0, prob_bit_1): Probabilities of measuring 0 or 1
    """
    if basis == Basis.Z:
        # Z basis measurement: |0> = [1,0], |1> = [0,1]
        prob_0 = np.abs(state[0])**2
        prob_1 = np.abs(state[1])**2
    else:  # Basis.X
        # X basis measurement: |+> = 1/sqrt(2)[1,1], |-> = 1/sqrt(2)[1,-1]
        # Project onto |+> and |->
        plus = BB84_STATES[(Basis.X, Bit.ZERO)]
        minus = BB84_STATES[(Basis.X, Bit.ONE)]
        prob_0 = np.abs(np.vdot(plus, state))**2  # |<+|state>|^2
        prob_1 = np.abs(np.vdot(minus, state))**2 # |<-|state>|^2
    
    return float(prob_0), float(prob_1)


def sample_measurement(state: np.ndarray, basis: int, rng: np.random.Generator) -> int:
    """
    Sample a measurement outcome for a state in a given basis.
    
    Args:
        state: 2D complex state vector
        basis: 0 (Z) or 1 (X)
        rng: Random number generator
        
    Returns:
        Measured bit (0 or 1)
    """
    prob_0, prob_1 = measure_state_in_basis(state, basis)
    
    # Sample according to Born rule
    return 0 if rng.random() < prob_0 else 1