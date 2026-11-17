"""
Temporal QBER feature extraction for BB84 sessions.

Provides temporal windowing, QBER calculation, and feature extraction
for BB84 quantum key distribution sessions.
"""

from .windowing import (
    WindowConfig,
    WindowMode,
    WindowResult,
    create_windows,
    create_windows_from_session,
)

from .qber import (
    calculate_window_qber,
    calculate_window_qber_batch,
    calculate_session_qber,
    calculate_qber_statistics,
    detect_significant_changes,
    calculate_trend_features,
)

from .features import (
    extract_temporal_qber_features,
    TEMPORAL_QBER_FEATURES,
    TEMPORAL_QBER_FEATURE_GROUPS,
)

from ..simulator.session import SessionConfig, run_session, SessionResult

__all__ = [
    "WindowConfig",
    "WindowMode", 
    "WindowResult",
    "create_windows",
    "create_windows_from_session",
    "calculate_window_qber",
    "calculate_window_qber_batch",
    "calculate_session_qber",
    "calculate_qber_statistics",
    "detect_significant_changes",
    "calculate_trend_features",
    "extract_temporal_qber_features",
    "TEMPORAL_QBER_FEATURES",
    "TEMPORAL_QBER_FEATURE_GROUPS",
    "SessionConfig",
    "SessionResult",
    "run_session",
]