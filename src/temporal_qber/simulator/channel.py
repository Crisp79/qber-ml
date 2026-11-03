"""
Quantum channel models for BB84 protocol.

Provides an extensible interface for future noise and attack models.
"""

import numpy as np
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .alice import Basis, Bit, BB84_STATES


class ChannelType(Enum):
    """Channel type enumeration."""
    IDEAL = "ideal"
    DEPOLARIZING = "depolarizing"
    PHASE_FLIP = "phase_flip"
    AMPLITUDE_DAMPING = "amplitude_damping"
    BIT_FLIP = "bit_flip"
    # Future: ATTACK = "attack"


@dataclass
class ChannelConfig:
    """Configuration for a quantum channel."""
    channel_type: ChannelType = ChannelType.IDEAL
    # Common parameters
    loss_probability: float = 0.0
    # Depolarizing channel
    depolarizing_prob: float = 0.0
    # Phase flip channel
    phase_flip_prob: float = 0.0
    # Amplitude damping
    damping_gamma: float = 0.0
    # Bit flip channel
    bit_flip_prob: float = 0.0
    
    def __post_init__(self):
        if not 0 <= self.loss_probability <= 1:
            raise ValueError("loss_probability must be in [0, 1]")
        if not 0 <= self.depolarizing_prob <= 1:
            raise ValueError("depolarizing_prob must be in [0, 1]")
        if not 0 <= self.phase_flip_prob <= 1:
            raise ValueError("phase_flip_prob must be in [0, 1]")
        if not 0 <= self.damping_gamma <= 1:
            raise ValueError("damping_gamma must be in [0, 1]")
        if not 0 <= self.bit_flip_prob <= 1:
            raise ValueError("bit_flip_prob must be in [0, 1]")


class QuantumChannel(ABC):
    """Abstract base class for quantum channels."""
    
    def __init__(self, config: Optional[ChannelConfig] = None, seed: Optional[int] = None):
        self.config = config or ChannelConfig()
        self.rng = np.random.default_rng(seed)
        self._transmissions: List[dict] = []
    
    @abstractmethod
    def transmit(self, state: np.ndarray) -> np.ndarray:
        """
        Transmit a quantum state through the channel.
        
        Args:
            state: Input 2D complex state vector
            
        Returns:
            Output 2D complex state vector (may be modified by noise)
        """
        pass
    
    def transmit_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Transmit a batch of states.
        
        Args:
            states: Array of state vectors (n, 2) complex
            
        Returns:
            Array of output state vectors (n, 2) complex
        """
        n = states.shape[0]
        output = np.empty_like(states)
        for i in range(n):
            output[i] = self.transmit(states[i])
        return output
    
    def apply_loss(self, state: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Apply photon loss to a state.
        
        Args:
            state: Input state vector
            
        Returns:
            (output_state, detected): Output state and whether photon was detected
        """
        if self.config.loss_probability > 0:
            detected = self.rng.random() >= self.config.loss_probability
            if not detected:
                # Photon lost - return vacuum state (zero vector)
                return np.zeros_like(state), False
        return state, True
    
    def get_config(self) -> dict:
        """Get channel configuration."""
        return {
            "channel_type": self.config.channel_type.value,
            "loss_probability": self.config.loss_probability,
            "depolarizing_prob": self.config.depolarizing_prob,
            "phase_flip_prob": self.config.phase_flip_prob,
            "damping_gamma": self.config.damping_gamma,
            "bit_flip_prob": self.config.bit_flip_prob,
        }
    
    def reset(self, seed: Optional[int] = None):
        """Reset channel state."""
        self._transmissions.clear()
        if seed is not None:
            self.rng = np.random.default_rng(seed)


class IdealChannel(QuantumChannel):
    """Ideal quantum channel with no noise or loss."""
    
    def __init__(self, seed: Optional[int] = None):
        config = ChannelConfig(channel_type=ChannelType.IDEAL)
        super().__init__(config, seed)
    
    def transmit(self, state: np.ndarray) -> np.ndarray:
        """Ideal transmission - returns state unchanged (after loss check)."""
        # Check for loss
        output_state, detected = self.apply_loss(state)
        if not detected:
            # Return vacuum (zero) state
            return output_state
        return state.copy()


class DepolarizingChannel(QuantumChannel):
    """Depolarizing channel that applies random Pauli errors."""
    
    def __init__(self, depolarizing_prob: float = 0.01, seed: Optional[int] = None):
        config = ChannelConfig(
            channel_type=ChannelType.DEPOLARIZING,
            depolarizing_prob=depolarizing_prob
        )
        super().__init__(config, seed)
    
    def transmit(self, state: np.ndarray) -> np.ndarray:
        """Apply depolarizing noise with probability p."""
        # Check loss first
        output_state, detected = self.apply_loss(state)
        if not detected:
            return output_state
        
        # Apply depolarizing channel: with probability p, replace with maximally mixed state
        # For a qubit, this means applying X, Y, or Z each with probability p/3
        p = self.config.depolarizing_prob
        if self.rng.random() < p:
            error = self.rng.choice(['X', 'Y', 'Z'])
            if error == 'X':
                # Pauli X: [0, 1; 1, 0]
                output_state = np.array([state[1], state[0]], dtype=np.complex128)
            elif error == 'Y':
                # Pauli Y: [0, -i; i, 0]
                output_state = np.array([-1j*state[1], 1j*state[0]], dtype=np.complex128)
            else:  # 'Z'
                # Pauli Z: [1, 0; 0, -1]
                output_state = np.array([state[0], -state[1]], dtype=np.complex128)
        else:
            output_state = state.copy()
        
        return output_state


class PhaseFlipChannel(QuantumChannel):
    """Phase flip channel (Z errors only)."""
    
    def __init__(self, phase_flip_prob: float = 0.01, seed: Optional[int] = None):
        config = ChannelConfig(
            channel_type=ChannelType.PHASE_FLIP,
            phase_flip_prob=phase_flip_prob
        )
        super().__init__(config, seed)
    
    def transmit(self, state: np.ndarray) -> np.ndarray:
        """Apply phase flip (Z error) with probability p."""
        output_state, detected = self.apply_loss(state)
        if not detected:
            return output_state
        
        p = self.config.phase_flip_prob
        if self.rng.random() < p:
            # Apply Z: [1, 0; 0, -1]
            output_state = np.array([state[0], -state[1]], dtype=np.complex128)
        else:
            output_state = state.copy()
        
        return output_state


class BitFlipChannel(QuantumChannel):
    """Bit flip channel (X errors only)."""
    
    def __init__(self, bit_flip_prob: float = 0.01, seed: Optional[int] = None):
        config = ChannelConfig(
            channel_type=ChannelType.BIT_FLIP,
            bit_flip_prob=bit_flip_prob
        )
        super().__init__(config, seed)
    
    def transmit(self, state: np.ndarray) -> np.ndarray:
        """Apply bit flip (X error) with probability p."""
        output_state, detected = self.apply_loss(state)
        if not detected:
            return output_state
        
        p = self.config.bit_flip_prob
        if self.rng.random() < p:
            # Apply X: [0, 1; 1, 0]
            output_state = np.array([state[1], state[0]], dtype=np.complex128)
        else:
            output_state = state.copy()
        
        return output_state


class AmplitudeDampingChannel(QuantumChannel):
    """Amplitude damping channel (energy loss, |1> -> |0>)."""
    
    def __init__(self, damping_gamma: float = 0.01, seed: Optional[int] = None):
        config = ChannelConfig(
            channel_type=ChannelType.AMPLITUDE_DAMPING,
            damping_gamma=damping_gamma
        )
        super().__init__(config, seed)
    
    def transmit(self, state: np.ndarray) -> np.ndarray:
        """Apply amplitude damping with probability gamma."""
        output_state, detected = self.apply_loss(state)
        if not detected:
            return output_state
        
        gamma = self.config.damping_gamma
        # Kraus operators for amplitude damping:
        # E0 = [[1, 0], [0, sqrt(1-gamma)]]
        # E1 = [[0, sqrt(gamma)], [0, 0]]
        
        if self.rng.random() < gamma:
            # E1: |1> -> |0> transition
            # State becomes |0> with probability |<1|psi>|^2 * gamma
            prob_1 = np.abs(state[1])**2
            if self.rng.random() < prob_1:
                output_state = np.array([1.0, 0.0], dtype=np.complex128)
            else:
                output_state = state.copy()
        else:
            # E0: no jump, but amplitude of |1> reduced
            output_state = np.array([
                state[0],
                state[1] * np.sqrt(1 - gamma)
            ], dtype=np.complex128)
            # Renormalize
            norm = np.linalg.norm(output_state)
            if norm > 0:
                output_state = output_state / norm
        
        return output_state


def create_channel(channel_type: str, seed: Optional[int] = None, **kwargs) -> QuantumChannel:
    """
    Factory function to create quantum channels.
    
    Args:
        channel_type: Type of channel ('ideal', 'depolarizing', 'phase_flip', 'amplitude_damping', 'bit_flip')
        seed: Random seed for reproducibility
        **kwargs: Channel-specific parameters
        
    Returns:
        QuantumChannel instance
    """
    channel_type = channel_type.lower()
    
    # Extract common parameters
    loss_probability = kwargs.pop('loss_probability', 0.0)
    
    if channel_type == 'ideal':
        channel = IdealChannel(seed=seed)
    elif channel_type == 'depolarizing':
        depolarizing_prob = kwargs.pop('depolarizing_prob', 0.0)
        channel = DepolarizingChannel(depolarizing_prob=depolarizing_prob, seed=seed)
    elif channel_type == 'phase_flip':
        phase_flip_prob = kwargs.pop('phase_flip_prob', 0.0)
        channel = PhaseFlipChannel(phase_flip_prob=phase_flip_prob, seed=seed)
    elif channel_type == 'amplitude_damping':
        damping_gamma = kwargs.pop('damping_gamma', 0.0)
        channel = AmplitudeDampingChannel(damping_gamma=damping_gamma, seed=seed)
    elif channel_type == 'bit_flip':
        bit_flip_prob = kwargs.pop('bit_flip_prob', 0.0)
        channel = BitFlipChannel(bit_flip_prob=bit_flip_prob, seed=seed)
    else:
        raise ValueError(f"Unknown channel type: {channel_type}. "
                         f"Available: ideal, depolarizing, phase_flip, amplitude_damping, bit_flip")
    
    # Apply loss probability to any channel
    if loss_probability > 0:
        channel.config.loss_probability = loss_probability
    
    return channel