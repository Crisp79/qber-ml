"""
Intercept-resend attack implementation.

Eve intercepts Alice's quantum state, measures it in a randomly chosen basis,
and sends a new state to Bob based on her measurement result.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .base import BaseAttack, NoAttack, AttackResult
from .pns import PNSAttack, PNSAttackWithStorage
from .beam_splitting import BeamSplittingAttack, BeamSplittingAttackFixedPhoton
from .time_shift import TimeShiftAttack, TimeShiftAttackWithIntercept
from .trojan_horse import TrojanHorseAttack, TrojanHorseActiveAttack
from .quantum_cloning import QuantumCloningAttack, QuantumCloningAttackMeasure
from .fake_state import FakeStateAttack
from ..alice import Basis, Bit, BB84_STATES, measure_state_in_basis, sample_measurement


@dataclass
class InterceptResendResult(AttackResult):
    """Result of an intercept-resend attack on a single transmission."""
    eve_basis: Optional[int]  # Eve's chosen basis (0=Z, 1=X)
    eve_outcome: Optional[int]  # Eve's measurement outcome
    original_state: np.ndarray
    transmitted_state: np.ndarray  # State sent to Bob (Eve's prepared state)
    eve_info: Dict[str, Any]


class InterceptResendAttack(BaseAttack):
    """
    Full intercept-resend attack.
    
    Eve intercepts each qubit, measures it in a randomly chosen basis,
    and sends a new qubit to Bob based on her measurement result.
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        # Eve chooses Z basis with probability 0.5 (unbiased)
        self.p_z = 0.5
    
    def set_basis_probability(self, p_z: float):
        """Set Eve's probability of choosing Z basis."""
        if not 0 <= p_z <= 1:
            raise ValueError(f"p_z must be in [0, 1], got {p_z}")
        self.p_z = p_z
    
    def _choose_basis(self) -> int:
        """Eve randomly chooses a measurement basis."""
        return 0 if self.rng.random() < self.p_z else 1
    
    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """
        Perform intercept-resend attack on a single transmission.
        
        Eve:
        1. Chooses a random basis
        2. Measures Alice's state in that basis
        3. Prepares a new state based on her measurement
        4. Sends the new state to Bob
        """
        # Eve chooses a basis
        eve_basis = self._choose_basis()
        
        # Eve measures the state in her chosen basis
        prob_0, prob_1 = self._measure_state_in_basis(state, eve_basis)
        eve_outcome = 0 if self.rng.random() < prob_0 else 1
        
        # Eve prepares a new state based on her measurement
        transmitted_state = self._prepare_state(eve_basis, eve_outcome)
        
        return InterceptResendResult(
            intercepted=True,
            eve_basis=eve_basis,
            eve_outcome=eve_outcome,
            original_state=state.copy(),
            transmitted_state=transmitted_state,
            eve_info={
                "eve_basis": eve_basis,
                "eve_outcome": eve_outcome,
            }
        )
    
    def _measure_state_in_basis(self, state: np.ndarray, basis: int) -> tuple[float, float]:
        """Calculate measurement probabilities for a state in a given basis."""
        if basis == Basis.Z:
            prob_0 = np.abs(state[0])**2
            prob_1 = np.abs(state[1])**2
        else:  # Basis.X
            plus = BB84_STATES[(Basis.X, Bit.ZERO)]
            minus = BB84_STATES[(Basis.X, Bit.ONE)]
            prob_0 = np.abs(np.vdot(plus, state))**2  # |<+|state>|^2
            prob_1 = np.abs(np.vdot(minus, state))**2  # |<-|state>|^2
        
        return float(prob_0), float(prob_1)
    
    def _prepare_state(self, basis: int, bit: int) -> np.ndarray:
        """Prepare a new quantum state based on Eve's measurement."""
        return BB84_STATES[(basis, bit)].copy()


class PartialInterceptResendAttack(InterceptResendAttack):
    """
    Partial intercept-resend attack.
    
    Eve intercepts only a fraction of the transmissions.
    """
    
    def __init__(
        self, 
        interception_prob: float = 0.5, 
        seed: Optional[int] = None
    ):
        """
        Initialize partial intercept-resend attack.
        
        Args:
            interception_prob: Probability that Eve intercepts each transmission (0-1)
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        if not 0 <= interception_prob <= 1:
            raise ValueError(f"interception_prob must be in [0, 1], got {interception_prob}")
        self.interception_prob = interception_prob
    
    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """
        Perform intercept-resend attack with configurable probability.
        
        With probability interception_prob, Eve intercepts and resends.
        Otherwise, the original state passes through unchanged.
        """
        # Decide whether to intercept this transmission
        if self.rng.random() < self.interception_prob:
            # Perform full intercept-resend
            return super().attack(state, alice_basis, alice_bit)
        else:
            # No interception - pass state through unchanged
            return InterceptResendResult(
                intercepted=False,
                eve_basis=None,
                eve_outcome=None,
                original_state=state.copy(),
                transmitted_state=state.copy(),
                eve_info={"intercepted": False}
            )


def create_attack(
    attack_type: str,
    seed: Optional[int] = None,
    **kwargs
) -> BaseAttack:
    """
    Factory function to create attack instances.
    
    Args:
        attack_type: Type of attack ('none', 'intercept_resend', 'partial_intercept_resend',
                     'pns', 'pns_storage', 'beam_splitting', 'beam_splitting_fixed',
                     'time_shift', 'time_shift_intercept')
        seed: Random seed
        **kwargs: Attack-specific parameters
        
    Returns:
        BaseAttack instance
    """
    attack_type = attack_type.lower()
    
    if attack_type == 'none':
        return NoAttack(seed=seed)
    elif attack_type == 'intercept_resend':
        return InterceptResendAttack(seed=seed)
    elif attack_type == 'partial_intercept_resend':
        interception_prob = kwargs.get('interception_prob', 0.5)
        return PartialInterceptResendAttack(
            interception_prob=interception_prob, 
            seed=seed
        )
    elif attack_type == 'pns':
        mean_photon_number = kwargs.get('mean_photon_number', 0.1)
        single_photon_strategy = kwargs.get('single_photon_strategy', 'pass')
        return PNSAttack(
            mean_photon_number=mean_photon_number,
            single_photon_strategy=single_photon_strategy,
            seed=seed
        )
    elif attack_type == 'pns_storage':
        mean_photon_number = kwargs.get('mean_photon_number', 0.1)
        single_photon_strategy = kwargs.get('single_photon_strategy', 'pass')
        return PNSAttackWithStorage(
            mean_photon_number=mean_photon_number,
            single_photon_strategy=single_photon_strategy,
            seed=seed
        )
    elif attack_type == 'beam_splitting':
        reflection_coeff = kwargs.get('reflection_coeff', 0.1)
        mean_photon_number = kwargs.get('mean_photon_number', 0.1)
        return BeamSplittingAttack(
            reflection_coeff=reflection_coeff,
            mean_photon_number=mean_photon_number,
            seed=seed
        )
    elif attack_type == 'beam_splitting_fixed':
        reflection_coeff = kwargs.get('reflection_coeff', 0.1)
        return BeamSplittingAttackFixedPhoton(
            reflection_coeff=reflection_coeff,
            seed=seed
        )
    elif attack_type == 'time_shift':
        time_shift = kwargs.get('time_shift', 0.5)
        det0_peak_time = kwargs.get('det0_peak_time', 0.0)
        det1_peak_time = kwargs.get('det1_peak_time', 0.2)
        det_width = kwargs.get('det_width', 1.0)
        max_efficiency = kwargs.get('max_efficiency', 1.0)
        return TimeShiftAttack(
            time_shift=time_shift,
            det0_peak_time=det0_peak_time,
            det1_peak_time=det1_peak_time,
            det_width=det_width,
            max_efficiency=max_efficiency,
            seed=seed
        )
    elif attack_type == 'time_shift_intercept':
        time_shift = kwargs.get('time_shift', 0.5)
        det0_peak_time = kwargs.get('det0_peak_time', 0.0)
        det1_peak_time = kwargs.get('det1_peak_time', 0.2)
        det_width = kwargs.get('det_width', 1.0)
        max_efficiency = kwargs.get('max_efficiency', 1.0)
        intercept_prob = kwargs.get('intercept_prob', 1.0)
        return TimeShiftAttackWithIntercept(
            time_shift=time_shift,
            det0_peak_time=det0_peak_time,
            det1_peak_time=det1_peak_time,
            det_width=det_width,
            max_efficiency=max_efficiency,
            intercept_prob=intercept_prob,
            seed=seed
        )
    elif attack_type == 'trojan_horse':
        probe_photons = kwargs.get('probe_photons', 1000)
        back_reflection_coeff = kwargs.get('back_reflection_coeff', 1e-4)
        basis_inference_prob = kwargs.get('basis_inference_prob', 0.9)
        detector_saturation_prob = kwargs.get('detector_saturation_prob', 0.0)
        return TrojanHorseAttack(
            probe_photons=probe_photons,
            back_reflection_coeff=back_reflection_coeff,
            basis_inference_prob=basis_inference_prob,
            detector_saturation_prob=detector_saturation_prob,
            seed=seed
        )
    elif attack_type == 'trojan_horse_active':
        probe_photons = kwargs.get('probe_photons', 10000)
        back_reflection_coeff = kwargs.get('back_reflection_coeff', 1e-4)
        basis_inference_prob = kwargs.get('basis_inference_prob', 0.95)
        detector_saturation_prob = kwargs.get('detector_saturation_prob', 0.1)
        return TrojanHorseActiveAttack(
            probe_photons=probe_photons,
            back_reflection_coeff=back_reflection_coeff,
            basis_inference_prob=basis_inference_prob,
            detector_saturation_prob=detector_saturation_prob,
            seed=seed
        )
    elif attack_type == 'quantum_cloning':
        cloning_type = kwargs.get('cloning_type', 'universal')
        custom_fidelity = kwargs.get('custom_fidelity', None)
        custom_bob_fidelity = kwargs.get('custom_bob_fidelity', None)
        return QuantumCloningAttack(
            cloning_type=cloning_type,
            custom_fidelity=custom_fidelity,
            custom_bob_fidelity=custom_bob_fidelity,
            seed=seed
        )
    elif attack_type == 'quantum_cloning_measure':
        cloning_type = kwargs.get('cloning_type', 'universal')
        custom_fidelity = kwargs.get('custom_fidelity', None)
        custom_bob_fidelity = kwargs.get('custom_bob_fidelity', None)
        return QuantumCloningAttackMeasure(
            cloning_type=cloning_type,
            custom_fidelity=custom_fidelity,
            custom_bob_fidelity=custom_bob_fidelity,
            seed=seed
        )
    elif attack_type == 'fake_state':
        attack_subtype = kwargs.get('attack_subtype', 'phase_remapped')
        phase_shift = kwargs.get('phase_shift', 0.0)
        det0_efficiency = kwargs.get('det0_efficiency', 1.0)
        det1_efficiency = kwargs.get('det1_efficiency', 1.0)
        multi_photon_prob = kwargs.get('multi_photon_prob', 0.0)
        wavelength_shift = kwargs.get('wavelength_shift', 0.0)
        return FakeStateAttack(
            attack_subtype=attack_subtype,
            phase_shift=phase_shift,
            det0_efficiency=det0_efficiency,
            det1_efficiency=det1_efficiency,
            multi_photon_prob=multi_photon_prob,
            wavelength_shift=wavelength_shift,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown attack type: {attack_type}. "
                         f"Available: none, intercept_resend, partial_intercept_resend, "
                         f"pns, pns_storage, beam_splitting, beam_splitting_fixed, "
                         f"time_shift, time_shift_intercept, trojan_horse, trojan_horse_active, "
                         f"quantum_cloning, quantum_cloning_measure, fake_state")