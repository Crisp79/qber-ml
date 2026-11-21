"""
Main BB84 protocol implementation.

Orchestrates Alice, Bob, the quantum channel, and Eve's attacks to run the full BB84 protocol.
"""

import numpy as np
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

from .alice import Alice, Basis, Bit, BB84_STATES, STATE_LABELS
from .bob import Bob, BobMeasurement, calculate_qber, calculate_agreement
from .channel import QuantumChannel, IdealChannel, create_channel, ChannelConfig, ChannelType
from .attacks import BaseAttack, NoAttack, create_attack


@dataclass
class BB84Result:
    """Complete result of a BB84 simulation."""
    # Simulation parameters
    n_transmissions: int
    seed: Optional[int]
    channel_config: dict
    alice_p_z: float
    bob_p_z: float
    detection_efficiency: float
    dark_count_prob: float
    attack_type: str
    attack_params: dict
    
    # Raw data
    alice_bits: np.ndarray
    alice_bases: np.ndarray
    bob_bases: np.ndarray
    bob_outcomes: np.ndarray
    bob_detected: np.ndarray
    bob_probs_0: np.ndarray
    bob_probs_1: np.ndarray
    bob_is_dark_count: np.ndarray
    
    # Eve's attack data
    eve_intercepted: np.ndarray
    eve_bases: np.ndarray
    eve_outcomes: np.ndarray
    
    # Sifted key
    matching_bases_mask: np.ndarray
    n_matching_bases: int
    sifted_alice_bits: np.ndarray
    sifted_bob_bits: np.ndarray
    sifted_key_length: int
    sifting_efficiency: float
    
    # Basis-specific sifted bits (for basis-dependent features)
    sifted_alice_rect: np.ndarray
    sifted_bob_rect: np.ndarray
    sifted_alice_diag: np.ndarray
    sifted_bob_diag: np.ndarray
    
    # Per-window loss sequence (for channel-interaction features)
    loss_seq: np.ndarray
    
    # QBER
    n_sifted_errors: int
    qber: float
    agreement: float
    
    # Timing
    elapsed_time: float
    
    def get_public_info(self) -> dict:
        """
        Get information available to an eavesdropper / attack detector.
        
        Does NOT include Alice's private bits or Bob's basis choices.
        """
        return {
            "n_transmissions": self.n_transmissions,
            "n_matching_bases": int(self.n_matching_bases),
            "sifted_key_length": int(self.sifted_key_length),
            "sifting_efficiency": float(self.sifting_efficiency),
            "qber": float(self.qber),
            "agreement": float(self.agreement),
            "bob_detection_rate": float(np.mean(self.bob_detected)),
            "dark_count_rate": float(np.mean(self.bob_is_dark_count)) if hasattr(self, 'bob_is_dark_count') else 0.0,
            "basis_agreement_rate": float(np.mean(self.matching_bases_mask)),
            "attack_type": self.attack_type,
            "attack_params": self.attack_params,
        }
    
    def get_private_info(self) -> dict:
        """
        Get private simulation information (for verification only).
        
        Includes Alice's bits, Bob's bases - NOT available to ML detector.
        """
        return {
            "alice_bits": self.alice_bits.tolist(),
            "alice_bases": self.alice_bases.tolist(),
            "bob_bases": self.bob_bases.tolist(),
            "bob_outcomes": self.bob_outcomes.tolist(),
            "bob_detected": self.bob_detected.tolist(),
            "bob_is_dark_count": self.bob_is_dark_count.tolist() if hasattr(self, 'bob_is_dark_count') else [],
            "matching_bases_mask": self.matching_bases_mask.tolist(),
            "sifted_alice_bits": self.sifted_alice_bits.tolist(),
            "sifted_bob_bits": self.sifted_bob_bits.tolist(),
            "eve_intercepted": self.eve_intercepted.tolist() if hasattr(self, 'eve_intercepted') else [],
            "eve_bases": self.eve_bases.tolist() if hasattr(self, 'eve_bases') else [],
            "eve_outcomes": self.eve_outcomes.tolist() if hasattr(self, 'eve_outcomes') else [],
        }


class BB84Protocol:
    """Full BB84 protocol simulation."""
    
    def __init__(
        self,
        n_transmissions: int = 100000,
        seed: Optional[int] = None,
        alice_p_z: float = 0.5,
        bob_p_z: float = 0.5,
        detection_efficiency: float = 1.0,
        dark_count_prob: float = 0.0,
        channel: Optional[QuantumChannel] = None,
        attack: Optional[str] = None,
        attack_params: Optional[dict] = None,
    ):
        """
        Initialize BB84 protocol.
        
        Args:
            n_transmissions: Number of qubits Alice sends
            seed: Random seed for reproducibility
            alice_p_z: Alice's probability of choosing Z basis
            bob_p_z: Bob's probability of choosing Z basis
            detection_efficiency: Bob's photon detection efficiency
            dark_count_prob: Bob's dark count probability
            channel: Quantum channel (default: IdealChannel)
            attack: Attack type ('none', 'intercept_resend', 'partial_intercept_resend')
            attack_params: Additional attack parameters
        """
        if n_transmissions <= 0:
            raise ValueError("n_transmissions must be positive")
        if not 0 <= alice_p_z <= 1:
            raise ValueError(f"alice_p_z must be in [0, 1], got {alice_p_z}")
        if not 0 <= bob_p_z <= 1:
            raise ValueError(f"bob_p_z must be in [0, 1], got {bob_p_z}")
        if not 0 <= detection_efficiency <= 1:
            raise ValueError(f"detection_efficiency must be in [0, 1], got {detection_efficiency}")
        if not 0 <= dark_count_prob <= 1:
            raise ValueError(f"dark_count_prob must be in [0, 1], got {dark_count_prob}")
        
        self.n_transmissions = n_transmissions
        self.seed = seed
        self.alice_p_z = alice_p_z
        self.bob_p_z = bob_p_z
        self.detection_efficiency = detection_efficiency
        self.dark_count_prob = dark_count_prob
        self.channel = channel or IdealChannel(seed=seed)
        self.attack_type = attack or "none"
        self.attack_params = attack_params or {}
        
        # Initialize Alice and Bob with independent seeds
        if seed is not None:
            # Use a simple but effective seed derivation: hash the seed with different labels
            # This ensures independent but reproducible seeds
            import hashlib
            alice_seed = int(hashlib.sha256(f"{seed}_alice".encode()).hexdigest(), 16) % (2**64)
            bob_seed = int(hashlib.sha256(f"{seed}_bob".encode()).hexdigest(), 16) % (2**64)
            channel_seed = int(hashlib.sha256(f"{seed}_channel".encode()).hexdigest(), 16) % (2**64)
            attack_seed = int(hashlib.sha256(f"{seed}_attack".encode()).hexdigest(), 16) % (2**64)
        else:
            alice_seed = bob_seed = channel_seed = attack_seed = None
        
        self.alice = Alice(seed=alice_seed, p_z=alice_p_z)
        self.bob = Bob(seed=bob_seed, p_z=bob_p_z, detection_efficiency=detection_efficiency, dark_count_prob=dark_count_prob)
        
        # Initialize attack
        self.attack = create_attack(self.attack_type, seed=attack_seed, **self.attack_params)
        
        # Recreate channel with proper seed if provided
        if channel is None and channel_seed is not None:
            self.channel = IdealChannel(seed=channel_seed)
        
        self._result: Optional[BB84Result] = None
    
    def run(self, progress_callback=None) -> BB84Result:
        """
        Run the full BB84 protocol.
        
        Args:
            progress_callback: Optional callback(step_name, progress) for progress reporting
            
        Returns:
            BB84Result with complete simulation data
        """
        start_time = time.time()
        
        if progress_callback:
            progress_callback("alice_prepare", 0)
        
        # Step 1: Alice generates bits and bases, prepares states
        alice_bits, alice_bases, alice_states = self.alice.prepare_batch(self.n_transmissions)
        
        if progress_callback:
            progress_callback("alice_prepare", 1)
            progress_callback("eve_attack", 0)
        
        # Bob's basis choices, drawn up front with the same RNG call measure_batch() would
        # make (so results are unchanged), so that attacks reading Bob's apparatus can use them.
        bob_bases_drawn = self.bob.rng.integers(0, 2, size=self.n_transmissions, dtype=np.int8)

        # Step 2: Eve's attack on transmitted states
        attack_results = self.attack.attack_batch(alice_states, alice_bases, alice_bits,
                                                  bob_bases=bob_bases_drawn)
        attacked_states = np.array([r.transmitted_state for r in attack_results], dtype=np.complex128)
        # Detector side-channel effects (passed to Bob only when some attack result sets them)
        eff_scale = np.array([r.bob_eff_scale for r in attack_results], dtype=np.float64)
        spurious_click_prob = np.array([r.bob_spurious_click_prob for r in attack_results], dtype=np.float64)
        dark_scale = np.array([r.bob_dark_scale for r in attack_results], dtype=np.float64)
        
        if progress_callback:
            progress_callback("eve_attack", 1)
            progress_callback("channel_transmit", 0)
        
        # Step 2: States pass through quantum channel
        channel_output_states = self.channel.transmit_batch(attacked_states)
        
        if progress_callback:
            progress_callback("channel_transmit", 1)
            progress_callback("bob_measure", 0)
        
        # Step 3: Bob chooses bases and measures
        bob_measurements = self.bob.measure_batch(
            channel_output_states,
            bases=bob_bases_drawn,
            eff_scale=eff_scale if np.any(eff_scale != 1.0) else None,
            spurious_click_prob=spurious_click_prob if np.any(spurious_click_prob > 0) else None,
            dark_scale=dark_scale if np.any(dark_scale != 1.0) else None,
        )
        
        if progress_callback:
            progress_callback("bob_measure", 1)
            progress_callback("basis_reconciliation", 0)
        
        # Extract Bob's data
        bob_bases = np.array([m.basis for m in bob_measurements], dtype=np.int8)
        bob_outcomes = np.array([m.outcome for m in bob_measurements], dtype=np.int8)
        bob_detected = np.array([m.detected for m in bob_measurements], dtype=bool)
        bob_probs_0 = np.array([m.prob_0 for m in bob_measurements], dtype=np.float64)
        bob_probs_1 = np.array([m.prob_1 for m in bob_measurements], dtype=np.float64)
        bob_is_dark_count = np.array([m.is_dark_count for m in bob_measurements], dtype=bool)
        
        # Extract Eve's attack data
        eve_intercepted = np.array([r.intercepted for r in attack_results], dtype=bool)
        eve_bases = np.array([r.eve_basis if r.eve_basis is not None else -1 for r in attack_results], dtype=np.int8)
        eve_outcomes = np.array([r.eve_outcome if r.eve_outcome is not None else -1 for r in attack_results], dtype=np.int8)
        
        if progress_callback:
            progress_callback("basis_reconciliation", 1)
            progress_callback("sifting", 0)
        
        # Step 4: Basis reconciliation (sifting)
        matching_bases_mask = (bob_bases == alice_bases)
        
        # Only detected events with matching bases
        valid_mask = matching_bases_mask & bob_detected
        sifted_indices = np.where(valid_mask)[0]
        
        sifted_alice_bits = alice_bits[valid_mask]
        sifted_bob_bits = bob_outcomes[valid_mask]
        
        # Track basis-specific sifted bits for basis-dependent features
        sifted_rect_mask = valid_mask & (alice_bases == Basis.Z)
        sifted_diag_mask = valid_mask & (alice_bases == Basis.X)
        
        sifted_alice_rect = alice_bits[sifted_rect_mask]
        sifted_bob_rect = bob_outcomes[sifted_rect_mask]
        sifted_alice_diag = alice_bits[sifted_diag_mask]
        sifted_bob_diag = bob_outcomes[sifted_diag_mask]
        
        # Track per-transmission loss for channel-interaction features
        # loss = 1 if photon lost (not detected), 0 if detected
        loss_seq = (~bob_detected).astype(np.float64)
        
        n_matching_bases = int(np.sum(matching_bases_mask))
        sifted_key_length = len(sifted_alice_bits)
        sifting_efficiency = sifted_key_length / self.n_transmissions if self.n_transmissions > 0 else 0.0
        
        if progress_callback:
            progress_callback("sifting", 1)
            progress_callback("qber_calculation", 0)
        
        # Step 5: QBER calculation
        if sifted_key_length > 0:
            n_sifted_errors = int(np.sum(sifted_alice_bits != sifted_bob_bits))
            qber = float(n_sifted_errors) / sifted_key_length
            agreement = 1.0 - qber
        else:
            n_sifted_errors = 0
            qber = 0.0
            agreement = 0.0
        
        elapsed_time = time.time() - start_time
        
        if progress_callback:
            progress_callback("qber_calculation", 1)
        
        # Create result
        self._result = BB84Result(
            n_transmissions=self.n_transmissions,
            seed=self.seed,
            channel_config=self.channel.get_config(),
            alice_p_z=self.alice_p_z,
            bob_p_z=self.bob_p_z,
            detection_efficiency=self.detection_efficiency,
            dark_count_prob=self.dark_count_prob,
            attack_type=self.attack_type,
            attack_params=self.attack_params,
            
            alice_bits=alice_bits,
            alice_bases=alice_bases,
            bob_bases=bob_bases,
            bob_outcomes=bob_outcomes,
            bob_detected=bob_detected,
            bob_probs_0=bob_probs_0,
            bob_probs_1=bob_probs_1,
            bob_is_dark_count=bob_is_dark_count,
            
            eve_intercepted=eve_intercepted,
            eve_bases=eve_bases,
            eve_outcomes=eve_outcomes,
            
            matching_bases_mask=matching_bases_mask,
            n_matching_bases=n_matching_bases,
            sifted_alice_bits=sifted_alice_bits,
            sifted_bob_bits=sifted_bob_bits,
            sifted_key_length=sifted_key_length,
            sifting_efficiency=sifting_efficiency,
            
            # Basis-specific sifted bits
            sifted_alice_rect=sifted_alice_rect,
            sifted_bob_rect=sifted_bob_rect,
            sifted_alice_diag=sifted_alice_diag,
            sifted_bob_diag=sifted_bob_diag,
            
            # Per-transmission loss sequence
            loss_seq=loss_seq,
            
            n_sifted_errors=n_sifted_errors,
            qber=qber,
            agreement=agreement,
            
            elapsed_time=elapsed_time,
        )
        
        return self._result
    
    def get_result(self) -> Optional[BB84Result]:
        """Get the last simulation result."""
        return self._result


def run_bb84_simulation(
    n_transmissions: int = 100000,
    seed: Optional[int] = None,
    alice_p_z: float = 0.5,
    bob_p_z: float = 0.5,
    detection_efficiency: float = 1.0,
    dark_count_prob: float = 0.0,
    channel_type: str = "ideal",
    channel_params: Optional[dict] = None,
    attack_type: str = "none",
    attack_params: Optional[dict] = None,
    progress_callback=None,
) -> BB84Result:
    """
    Convenience function to run a BB84 simulation.
    
    Args:
        n_transmissions: Number of qubits
        seed: Random seed
        alice_p_z: Alice's Z basis probability
        bob_p_z: Bob's Z basis probability
        detection_efficiency: Bob's detection efficiency
        dark_count_prob: Bob's dark count probability
        channel_type: Channel type ('ideal', 'depolarizing', etc.)
        channel_params: Additional channel parameters
        attack_type: Attack type ('none', 'intercept_resend', 'partial_intercept_resend')
        attack_params: Additional attack parameters
        progress_callback: Optional progress callback
        
    Returns:
        BB84Result
    """
    # Build the channel whenever it has parameters, including "ideal" (noiseless but
    # possibly lossy): previously loss_probability was silently dropped for ideal channels.
    channel = None
    if channel_type != "ideal" or channel_params:
        channel = create_channel(channel_type, seed=seed, **dict(channel_params or {}))
    
    protocol = BB84Protocol(
        n_transmissions=n_transmissions,
        seed=seed,
        alice_p_z=alice_p_z,
        bob_p_z=bob_p_z,
        detection_efficiency=detection_efficiency,
        dark_count_prob=dark_count_prob,
        channel=channel,
        attack=attack_type,
        attack_params=attack_params,
    )
    
    return protocol.run(progress_callback=progress_callback)