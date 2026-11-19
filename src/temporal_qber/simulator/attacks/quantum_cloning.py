"""
Quantum Cloning Attack Implementation.

The quantum cloning attack exploits the fact that Eve can create approximate copies
of unknown quantum states using quantum cloning machines. While perfect cloning is
impossible (no-cloning theorem), approximate cloning is possible and can give Eve
partial information about the quantum state.

Key concepts:
- Universal Quantum Cloning Machine (UQCM): Creates N approximate copies from M inputs
- 1->2 cloning: Creates 2 copies from 1 original (optimal fidelity = 5/6)
- Phase-covariant cloning: Better fidelity for states on equator of Bloch sphere
- State-dependent cloning: Optimal for known set of states

In BB84 context:
- Eve intercepts the quantum state
- Uses a cloning machine to create a copy
- Sends one copy to Bob, keeps the other
- Measures her copy after basis reconciliation
- Optimal 1->2 universal cloning gives fidelity 5/6 ≈ 0.833
- This introduces QBER = 1 - fidelity = 1/6 ≈ 16.7%

Implementation notes:
- Universal cloning: Symmetric 1->2 cloning with fidelity 5/6
- Phase-covariant cloning: For BB84 states (on equator), fidelity = 1/2 + 1/(2√2) ≈ 0.854
- State-dependent: For known BB84 states, optimal fidelity can be higher
- We model the cloning process by applying depolarizing noise to Bob's state
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .base import BaseAttack, AttackResult
from ..alice import Basis, Bit, BB84_STATES, measure_state_in_basis, sample_measurement


@dataclass
class CloningResult(AttackResult):
    """Result of a quantum cloning attack on a single transmission."""
    cloning_fidelity: float  # Fidelity of Eve's clone
    bob_fidelity: float  # Fidelity of Bob's state
    cloning_type: str  # Type of cloning machine used
    # All other fields inherited from AttackResult:
    # intercepted, eve_basis, eve_outcome, original_state, transmitted_state, eve_info


class QuantumCloningAttack(BaseAttack):
    """
    Quantum Cloning Attack.
    
    Eve uses a quantum cloning machine to create an approximate copy of
    the quantum state. She sends one copy to Bob and keeps the other.
    After basis reconciliation, Eve measures her copy in the correct basis.
    
    Types of cloning machines:
    - Universal (1->2): Optimal fidelity 5/6 for all states
    - Phase-covariant (1->2): Fidelity 1/2 + 1/(2√2) for equator states
    - State-dependent: Optimal for known set of states (BB84: 4 states)
    """
    
    CLONING_TYPES = {
        "universal": {
            "fidelity": 5/6,  # ~0.833
            "bob_fidelity": 5/6,
            "description": "Universal 1->2 quantum cloning machine"
        },
        "phase_covariant": {
            "fidelity": 0.5 + 1/(2*np.sqrt(2)),  # ~0.854
            "bob_fidelity": 0.5 + 1/(2*np.sqrt(2)),
            "description": "Phase-covariant cloning (optimal for equator states)"
        },
        "state_dependent": {
            "fidelity": 0.5 + 1/(2*np.sqrt(2)),  # Same as phase-covariant for BB84
            "bob_fidelity": 0.5 + 1/(2*np.sqrt(2)),
            "description": "State-dependent cloning for BB84 states"
        },
        "asymmetric": {
            "fidelity": 0.9,  # Eve gets better copy
            "bob_fidelity": 0.6,  # Bob gets worse copy
            "description": "Asymmetric cloning (Eve gets better copy)"
        }
    }
    
    def __init__(
        self,
        cloning_type: str = "universal",
        custom_fidelity: Optional[float] = None,
        custom_bob_fidelity: Optional[float] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize quantum cloning attack.
        
        Args:
            cloning_type: Type of cloning machine ('universal', 'phase_covariant', 
                         'state_dependent', 'asymmetric')
            custom_fidelity: Custom fidelity for Eve's clone (0 to 1)
            custom_bob_fidelity: Custom fidelity for Bob's state (0 to 1)
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        
        if cloning_type not in self.CLONING_TYPES and custom_fidelity is None:
            raise ValueError(f"cloning_type must be one of {list(self.CLONING_TYPES.keys())} "
                           f"or provide custom_fidelity")
        
        if cloning_type in self.CLONING_TYPES:
            self.fidelity = self.CLONING_TYPES[cloning_type]["fidelity"]
            self.bob_fidelity = self.CLONING_TYPES[cloning_type]["bob_fidelity"]
            self.cloning_type = cloning_type
        else:
            if custom_fidelity is None or custom_bob_fidelity is None:
                raise ValueError("For custom cloning, both custom_fidelity and custom_bob_fidelity required")
            self.fidelity = custom_fidelity
            self.bob_fidelity = custom_bob_fidelity
            self.cloning_type = "custom"
        
        if not 0 <= self.fidelity <= 1:
            raise ValueError(f"fidelity must be in [0, 1], got {self.fidelity}")
        if not 0 <= self.bob_fidelity <= 1:
            raise ValueError(f"bob_fidelity must be in [0, 1], got {self.bob_fidelity}")
        
        # Eve's basis choice probability (unbiased)
        self.p_z = 0.5
    
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
    
    def _apply_cloning_noise(self, state: np.ndarray, fidelity: float) -> np.ndarray:
        """
        Apply cloning noise to a state.
        
        For a qubit, cloning with fidelity F is equivalent to applying
        a depolarizing channel with parameter p = 3/2 * (1 - F).
        This gives QBER = 1 - F, which matches the theoretical cloning bound.
        """
        if fidelity >= 1.0:
            return state.copy()
        
        # Depolarizing parameter: p = 3/2 * (1 - F) gives QBER = p * 2/3 = 1 - F
        p = 1.5 * (1.0 - fidelity)
        p = min(p, 1.0)
        
        # Apply depolarizing noise: with prob p, apply X, Y, or Z
        if self.rng.random() < p:
            error = self.rng.choice(['X', 'Y', 'Z'])
            if error == 'X':
                return np.array([state[1], state[0]], dtype=np.complex128)
            elif error == 'Y':
                return np.array([-1j*state[1], 1j*state[0]], dtype=np.complex128)
            else:  # 'Z'
                return np.array([state[0], -state[1]], dtype=np.complex128)
        
        return state.copy()
    
    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """
        Perform quantum cloning attack on a single transmission.
        
        Eve intercepts the state, clones it, sends one copy to Bob,
        keeps the other for later measurement after basis reconciliation.
        """
        # Eve creates her clone (with fidelity self.fidelity)
        eve_clone = self._apply_cloning_noise(state, self.fidelity)
        
        # Bob receives his copy (with fidelity self.bob_fidelity)
        bob_state = self._apply_cloning_noise(state, self.bob_fidelity)
        
        # Eve doesn't measure immediately - she waits for basis reconciliation
        # For simulation, we record the state she would measure later
        eve_basis = None
        eve_outcome = None
        
        return CloningResult(
            intercepted=True,
            cloning_fidelity=self.fidelity,
            bob_fidelity=self.bob_fidelity,
            cloning_type=self.cloning_type,
            eve_basis=eve_basis,
            eve_outcome=eve_outcome,
            original_state=state.copy(),
            transmitted_state=bob_state,
            eve_info={
                "cloning_type": self.cloning_type,
                "cloning_fidelity": self.fidelity,
                "bob_fidelity": self.bob_fidelity,
                "expected_qber": 1.0 - self.bob_fidelity,
                "eve_clone_state": eve_clone.tolist(),
                "attack_type": "quantum_cloning",
                "note": f"Eve uses {self.cloning_type} cloning machine; keeps one copy, sends one to Bob"
            }
        )
    
    def get_theoretical_stats(self) -> Dict[str, float]:
        """Get theoretical attack statistics."""
        return {
            "cloning_type": self.cloning_type,
            "cloning_fidelity": self.fidelity,
            "bob_fidelity": self.bob_fidelity,
            "expected_qber": 1.0 - self.bob_fidelity,
            "eve_information": 1.0 - self.fidelity,  # Eve's error rate
        }


class QuantumCloningAttackMeasure(QuantumCloningAttack):
    """
    Quantum cloning attack where Eve measures her clone immediately.
    
    This is a less powerful variant where Eve measures her clone
    before basis reconciliation (using a random basis).
    """
    
    def __init__(
        self,
        cloning_type: str = "universal",
        custom_fidelity: Optional[float] = None,
        custom_bob_fidelity: Optional[float] = None,
        seed: Optional[int] = None
    ):
        super().__init__(cloning_type, custom_fidelity, custom_bob_fidelity, seed)
    
    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """
        Perform quantum cloning attack with immediate measurement.
        
        Eve clones the state and immediately measures her copy
        in a randomly chosen basis.
        """
        # Eve creates her clone
        eve_clone = self._apply_cloning_noise(state, self.fidelity)
        
        # Bob receives his copy
        bob_state = self._apply_cloning_noise(state, self.bob_fidelity)
        
        # Eve immediately measures her clone in random basis
        eve_basis = self._choose_basis()
        prob_0, prob_1 = self._measure_state_in_basis(eve_clone, eve_basis)
        eve_outcome = 0 if self.rng.random() < prob_0 else 1
        
        return CloningResult(
            intercepted=True,
            cloning_fidelity=self.fidelity,
            bob_fidelity=self.bob_fidelity,
            cloning_type=self.cloning_type,
            eve_basis=eve_basis,
            eve_outcome=eve_outcome,
            original_state=state.copy(),
            transmitted_state=bob_state,
            eve_info={
                "cloning_type": self.cloning_type,
                "cloning_fidelity": self.fidelity,
                "bob_fidelity": self.bob_fidelity,
                "expected_qber": 1.0 - self.bob_fidelity,
                "eve_basis": eve_basis,
                "eve_outcome": eve_outcome,
                "attack_type": "quantum_cloning_measure",
                "note": "Eve clones state and immediately measures in random basis"
            }
        )