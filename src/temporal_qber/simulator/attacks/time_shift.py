"""
Time-Shift Attack Implementation.

The time-shift attack exploits detector efficiency mismatch in Bob's detectors.
In a typical BB84 implementation, Bob uses two single-photon detectors (one for
each basis outcome). These detectors often have different efficiency vs. arrival
time profiles. Eve can exploit this by:

1. Shifting the arrival time of pulses
2. Making one detector more efficient than the other for the shifted time
3. This biases Bob's measurement outcomes, giving Eve information

Key concept:
- Detectors D0 and D1 have efficiency curves η_0(t) and η_1(t)
- Eve shifts pulse arrival time by Δt
- At shifted time, η_0(Δt) ≠ η_1(Δt), creating a bias
- Eve can learn Bob's basis choice or bit value from the bias

This attack is particularly relevant for practical QKD systems using
gated single-photon detectors (e.g., InGaAs APDs) which have significant
efficiency mismatch at different gate timings.
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .base import BaseAttack, AttackResult
from ..alice import Basis, Bit, BB84_STATES, measure_state_in_basis, sample_measurement


@dataclass
class TimeShiftResult(AttackResult):
    """Result of a time-shift attack on a single transmission."""
    time_shift: float  # Time shift applied by Eve (ns)
    det0_efficiency: float  # Detector 0 efficiency at shifted time
    det1_efficiency: float  # Detector 1 efficiency at shifted time
    bob_basis: Optional[int] = None  # Bob's chosen basis (if attacked)
    # All other fields inherited from AttackResult:
    # intercepted, eve_basis, eve_outcome, original_state, transmitted_state, eve_info


class TimeShiftAttack(BaseAttack):
    """
    Time-shift attack exploiting detector efficiency mismatch.
    
    Eve shifts the arrival time of pulses to exploit differences in
    detector efficiency curves. This can bias Bob's measurement outcomes.
    
    The attack works by:
    1. Eve shifts pulse arrival time by Δt
    2. Bob's detectors have time-dependent efficiency η(t)
    3. At time t+Δt, η_0 ≠ η_1, creating basis-dependent bias
    4. Eve can infer information from the bias
    
    For simulation, we model detector efficiency as Gaussian-like curves
    centered at different times for the two detectors.
    """
    
    def __init__(
        self,
        time_shift: float = 0.5,  # Time shift in ns (positive = later)
        det0_peak_time: float = 0.0,  # Peak efficiency time for detector 0 (ns)
        det1_peak_time: float = 0.2,  # Peak efficiency time for detector 1 (ns)
        det_width: float = 1.0,  # Width of efficiency curve (ns)
        max_efficiency: float = 1.0,  # Peak detector efficiency
        seed: Optional[int] = None
    ):
        """
        Initialize time-shift attack.
        
        Args:
            time_shift: Time shift applied by Eve (ns)
            det0_peak_time: Peak efficiency time for detector 0 (bit=0)
            det1_peak_time: Peak efficiency time for detector 1 (bit=1)
            det_width: Width of efficiency curves (ns)
            max_efficiency: Peak detector efficiency (0 to 1)
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        
        self.time_shift = time_shift
        self.det0_peak = det0_peak_time
        self.det1_peak = det1_peak_time
        self.det_width = det_width
        self.max_eff = max_efficiency
        
        if not 0 <= max_efficiency <= 1:
            raise ValueError(f"max_efficiency must be in [0, 1], got {max_efficiency}")
        
        # Eve's basis choice probability (unbiased)
        self.p_z = 0.5
    
    def _detector_efficiency(self, detector: int, time: float) -> float:
        """
        Compute detector efficiency at given time.
        
        Models efficiency as Gaussian-like curve:
        η(t) = η_max * exp(-(t - t_peak)^2 / (2 * σ^2))
        
        Args:
            detector: 0 or 1
            time: Arrival time (ns)
            
        Returns:
            Detection efficiency (0 to 1)
        """
        if detector == 0:
            peak = self.det0_peak
        else:
            peak = self.det1_peak
        
        # Gaussian efficiency curve
        eff = self.max_eff * np.exp(-0.5 * ((time - peak) / self.det_width) ** 2)
        return np.clip(eff, 0.0, 1.0)
    
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
        Perform time-shift attack on a single transmission.
        
        Eve shifts the pulse arrival time. The shifted arrival time
        affects Bob's detector efficiencies differently for the two outcomes.
        
        For simulation:
        1. Eve shifts pulse arrival time by Δt
        2. Bob's detectors have time-dependent efficiency η(t)
        3. At time t+Δt, η_0(Δt) ≠ η_1(Δt), creating bias
        4. Bob's detection becomes biased toward the more efficient detector
        5. This introduces QBER because Bob may click the "wrong" detector
        
        The attack works because:
        - Alice sends |0⟩_Z → correct detector is D0
        - Eve shifts arrival time so D0 is more efficient than D1
        - Bob's D0 clicks even when Alice sent |1⟩_Z → wrong outcome → QBER
        """
        # Nominal arrival time (without shift)
        nominal_time = 0.0
        
        # Shifted arrival time
        shifted_time = nominal_time + self.time_shift
        
        # Compute detector efficiencies at shifted time
        det0_eff = self._detector_efficiency(0, shifted_time)
        det1_eff = self._detector_efficiency(1, shifted_time)
        
        # Bob's basis choice (Bob chooses basis randomly)
        bob_basis = self._choose_basis()
        
        # Quantum measurement probabilities
        prob_0_quantum, prob_1_quantum = self._measure_state_in_basis(state, bob_basis)
        
        # Time-shift attack: detector efficiency bias causes Bob to click
        # the "wrong" detector more often, introducing QBER
        #
        # If Alice sent |0⟩_Z, correct detector is D0. But D1 may click
        # if it's more efficient at the shifted time.
        if bob_basis == Basis.Z:
            # Correct detector: D0 for |0⟩, D1 for |1⟩
            correct_det = alice_bit  # 0→D0, 1→D1
            wrong_det = 1 - alice_bit
            
            det_correct = det0_eff if correct_det == 0 else det1_eff
            det_wrong = det1_eff if correct_det == 0 else det0_eff
            
            total = det_correct + det_wrong
            if total > 0:
                p_correct = det_correct / total
            else:
                p_correct = 0.5
            
            # Bob clicks correct detector with prob p_correct, wrong with p_wrong
            bob_outcome = alice_bit if self.rng.random() < p_correct else 1 - alice_bit
            
        else:  # X basis
            # In X basis, |+⟩ and |-⟩ are superpositions of |0⟩ and |1⟩
            # Detector efficiency bias affects measurement outcome probabilities
            # directly since both detectors can fire for either basis state
            p0 = prob_0_quantum * det0_eff
            p1 = prob_1_quantum * det1_eff
            total = p0 + p1
            if total > 0:
                p0 /= total
                p1 /= total
            else:
                p0 = p1 = 0.5
            
            bob_outcome = 0 if self.rng.random() < p0 else 1
        
        # Eve just shifts time, doesn't measure the state directly
        # The state passes through with time shift applied
        transmitted_state = state.copy()
        
        return TimeShiftResult(
            intercepted=True,
            time_shift=self.time_shift,
            det0_efficiency=det0_eff,
            det1_efficiency=det1_eff,
            bob_basis=bob_basis,
            eve_basis=None,  # Pure time-shift: Eve doesn't measure
            eve_outcome=None,
            original_state=state.copy(),
            transmitted_state=transmitted_state,
            eve_info={
                "time_shift": self.time_shift,
                "shifted_time": shifted_time,
                "det0_efficiency": det0_eff,
                "det1_efficiency": det1_eff,
                "bias_factor": det0_eff / det1_eff if det1_eff > 0 else float('inf'),
                "bob_basis": bob_basis,
                "bob_outcome": bob_outcome,
                "attack_type": "time_shift",
                "note": "Eve shifts pulse arrival time to exploit detector efficiency mismatch"
            }
        )
    
    def get_detector_bias(self) -> float:
        """Get the detector efficiency bias factor at shifted time."""
        shifted_time = self.time_shift
        det0 = self._detector_efficiency(0, shifted_time)
        det1 = self._detector_efficiency(1, shifted_time)
        if det1 > 0:
            return det0 / det1
        return float('inf')


class TimeShiftAttackWithIntercept(TimeShiftAttack):
    """
    Time-shift attack combined with intercept-resend.
    
    Eve shifts the pulse time AND performs intercept-resend measurement.
    This is a more powerful attack combining both techniques.
    """
    
    def __init__(
        self,
        time_shift: float = 0.5,
        det0_peak_time: float = 0.0,
        det1_peak_time: float = 0.2,
        det_width: float = 1.0,
        max_efficiency: float = 1.0,
        intercept_prob: float = 1.0,  # Probability Eve also performs intercept-resend
        seed: Optional[int] = None
    ):
        super().__init__(
            time_shift, det0_peak_time, det1_peak_time,
            det_width, max_efficiency, seed
        )
        if not 0 <= intercept_prob <= 1:
            raise ValueError(f"intercept_prob must be in [0, 1], got {intercept_prob}")
        self.intercept_prob = intercept_prob
    
    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """
        Perform combined time-shift + intercept-resend attack.
        
        1. Eve shifts pulse arrival time
        2. With probability intercept_prob, Eve also performs intercept-resend
        3. Otherwise, Eve just lets the shifted pulse pass
        """
        # First apply time shift
        shifted_time = self.time_shift
        det0_eff = self._detector_efficiency(0, shifted_time)
        det1_eff = self._detector_efficiency(1, shifted_time)
        bias = det0_eff / det1_eff if det1_eff > 0 else float('inf')
        
        # Decide if Eve also intercepts
        if self.rng.random() < self.intercept_prob:
            # Combined attack: time-shift + intercept-resend
            eve_basis = self._choose_basis()
            prob_0, prob_1 = self._measure_state_in_basis(state, eve_basis)
            eve_outcome = 0 if self.rng.random() < prob_0 else 1
            transmitted_state = self._prepare_state(eve_basis, eve_outcome)
            
            return TimeShiftResult(
                intercepted=True,
                time_shift=self.time_shift,
                det0_efficiency=det0_eff,
                det1_efficiency=det1_eff,
                bob_basis=None,  # Eve controls the state
                eve_basis=eve_basis,
                eve_outcome=eve_outcome,
                original_state=state.copy(),
                transmitted_state=transmitted_state,
                eve_info={
                    "time_shift": self.time_shift,
                    "shifted_time": shifted_time,
                    "det0_efficiency": det0_eff,
                    "det1_efficiency": det1_eff,
                    "bias_factor": bias,
                    "eve_basis": eve_basis,
                    "eve_outcome": eve_outcome,
                    "attack_type": "time_shift_intercept",
                    "note": "Combined time-shift + intercept-resend"
                }
            )
        else:
            # Pure time-shift
            return TimeShiftResult(
                intercepted=True,
                time_shift=self.time_shift,
                det0_efficiency=det0_eff,
                det1_efficiency=det1_eff,
                bob_basis=self._choose_basis(),
                eve_basis=None,
                eve_outcome=None,
                original_state=state.copy(),
                transmitted_state=state.copy(),
                eve_info={
                    "time_shift": self.time_shift,
                    "shifted_time": shifted_time,
                    "det0_efficiency": det0_eff,
                    "det1_efficiency": det1_eff,
                    "bias_factor": bias,
                    "attack_type": "time_shift_pure",
                    "note": "Pure time-shift attack (no intercept)"
                }
            )