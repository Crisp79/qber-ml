"""
BB84 Quantum Key Distribution Simulator
"""

from .alice import Alice, Basis, Bit, BB84_STATES, STATE_LABELS, measure_state_in_basis, sample_measurement
from .bob import Bob, BobMeasurement, calculate_qber, calculate_agreement
from .channel import QuantumChannel, IdealChannel, DepolarizingChannel, PhaseFlipChannel, AmplitudeDampingChannel, BitFlipChannel, create_channel
from .bb84 import BB84Protocol, BB84Result, run_bb84_simulation
from .session import SessionConfig, SessionResult, run_session, run_multi_session

__all__ = [
    "Alice",
    "Bob",
    "Basis",
    "Bit",
    "BB84_STATES",
    "STATE_LABELS",
    "measure_state_in_basis",
    "sample_measurement",
    "BobMeasurement",
    "calculate_qber",
    "calculate_agreement",
    "QuantumChannel",
    "IdealChannel",
    "DepolarizingChannel",
    "PhaseFlipChannel",
    "AmplitudeDampingChannel",
    "BitFlipChannel",
    "create_channel",
    "BB84Protocol",
    "BB84Result",
    "run_bb84_simulation",
    "SessionConfig",
    "SessionResult",
    "run_session",
    "run_multi_session",
]