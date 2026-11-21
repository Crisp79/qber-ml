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
    
    def _shifted_efficiencies(self, favoured: int) -> tuple:
        """
        Bob's detector efficiencies (eta0, eta1) when Eve shifts the pulse to favour one detector.

        Eve moves the arrival time past the favoured detector's peak, away from the other
        detector's peak, by time_shift: the favoured detector keeps most of its efficiency
        while the other one is far down its curve.
        """
        direction = -1.0 if self.det1_peak >= self.det0_peak else 1.0
        if favoured == 0:
            t = self.det0_peak + direction * self.time_shift
        else:
            t = self.det1_peak - direction * self.time_shift
        return float(self._detector_efficiency(0, t)), float(self._detector_efficiency(1, t))

    def _pure_shift_result(self, state: np.ndarray) -> AttackResult:
        favoured = int(self.rng.integers(0, 2))
        eta0, eta1 = self._shifted_efficiencies(favoured)
        return TimeShiftResult(
            intercepted=True,
            time_shift=self.time_shift,
            det0_efficiency=eta0,
            det1_efficiency=eta1,
            bob_basis=None,
            eve_basis=None,  # Pure time-shift: Eve doesn't measure
            eve_outcome=None,
            original_state=state.copy(),
            transmitted_state=state.copy(),
            eve_info={
                "time_shift": self.time_shift,
                "favoured_detector": favoured,
                "det0_efficiency": eta0,
                "det1_efficiency": eta1,
                "attack_type": "time_shift_pure",
            },
            bob_eff_scale=(eta0, eta1),
        )

    def attack(self, state: np.ndarray, alice_basis: int, alice_bit: int) -> AttackResult:
        """
        Pure time-shift attack (Qi, Fung, Lo & Ma, Quantum Inf. Comput. 7, 73 (2007);
        Zhao, Fung, Qi, Chen & Lo, Phys. Rev. A 78, 042333 (2008)).

        For each pulse Eve randomly picks one of Bob's detectors to favour and shifts the
        arrival time away from the other detector's efficiency peak. The photon itself is
        untouched, so no error is introduced; but a click from the disfavoured detector
        becomes unlikely. Observable at Bob: outcome-dependent loss (lower detection rate),
        while a click tells Eve which bit Bob most likely obtained.
        """
        return self._pure_shift_result(state)

    def get_detector_bias(self) -> float:
        """Efficiency ratio favoured/other detector at the shifted time (mismatch strength)."""
        eta_fav, eta_other = self._shifted_efficiencies(0)
        if eta_other > 0:
            return eta_fav / eta_other
        return float('inf')


class TimeShiftAttackWithIntercept(TimeShiftAttack):
    """
    Time-shift attack combined with intercept-resend, exploiting detector efficiency
    mismatch (faked-state style; Makarov, Anisimov & Skaar, Phys. Rev. A 74, 022313 (2006)).
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
        With probability intercept_prob Eve measures Alice's photon in a random basis,
        resends her result and times it to favour the detector of the bit she resends.
        When Bob measures in Eve's basis he clicks on the favoured detector; when he
        measures in the other basis (where intercept-resend errors arise) half of his
        would-be clicks land on the disfavoured detector and are suppressed. Per
        intercepted sifted bit the QBER falls from 1/4 towards 1/6 as the efficiency
        ratio eta_other/eta_favoured goes from 1 to 0, and Bob loses extra clicks.
        Otherwise Eve applies the pure time shift.
        """
        if self.rng.random() >= self.intercept_prob:
            return self._pure_shift_result(state)

        eve_basis = self._choose_basis()
        prob_0, _ = self._measure_state_in_basis(state, eve_basis)
        eve_outcome = 0 if self.rng.random() < prob_0 else 1
        eta0, eta1 = self._shifted_efficiencies(eve_outcome)
        return TimeShiftResult(
            intercepted=True,
            time_shift=self.time_shift,
            det0_efficiency=eta0,
            det1_efficiency=eta1,
            bob_basis=None,  # Eve controls the state
            eve_basis=eve_basis,
            eve_outcome=eve_outcome,
            original_state=state.copy(),
            transmitted_state=self._prepare_state(eve_basis, eve_outcome),
            eve_info={
                "time_shift": self.time_shift,
                "favoured_detector": eve_outcome,
                "det0_efficiency": eta0,
                "det1_efficiency": eta1,
                "eve_basis": eve_basis,
                "eve_outcome": eve_outcome,
                "attack_type": "time_shift_intercept",
            },
            bob_eff_scale=(eta0, eta1),
        )
