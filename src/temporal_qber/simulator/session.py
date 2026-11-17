"""
Session management for BB84 simulations.

Handles temporal windowing and session-based QBER calculation.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from .bb84 import BB84Result, BB84Protocol, run_bb84_simulation


@dataclass
class SessionConfig:
    """Configuration for a BB84 session."""
    n_transmissions: int = 100000
    window_size: int = 50  # Bits per temporal window (paper's W)
    stride: int = 50  # Stride for temporal windows (default = window_size for non-overlapping)
    min_sifted_bits: int = 200  # Minimum bits for valid session
    seed: Optional[int] = None
    alice_p_z: float = 0.5
    bob_p_z: float = 0.5
    detection_efficiency: float = 1.0
    channel_type: str = "ideal"
    channel_params: Optional[dict] = None
    include_partial_trailing: bool = False  # Include incomplete trailing window


@dataclass
class SessionResult:
    """Result of a BB84 session with temporal windows."""
    session_config: SessionConfig
    bb84_result: BB84Result
    windows: List[dict] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate temporal windows after initialization."""
        self._calculate_windows()
    
    def _calculate_windows(self):
        """Divide sifted key into temporal windows."""
        sifted_alice = self.bb84_result.sifted_alice_bits
        sifted_bob = self.bb84_result.sifted_bob_bits
        
        n_sifted = len(sifted_alice)
        if n_sifted == 0:
            self.windows = []
            return
        
        n_windows = max(1, n_sifted // self.session_config.window_size)
        
        for w in range(n_windows):
            start = w * self.session_config.window_size
            end = min((w + 1) * self.session_config.window_size, n_sifted)
            
            window_alice = sifted_alice[start:end]
            window_bob = sifted_bob[start:end]
            
            if len(window_alice) >= self.session_config.min_sifted_bits:
                errors = np.sum(window_alice != window_bob)
                qber = errors / len(window_alice) if len(window_alice) > 0 else 0.0
            else:
                qber = 0.0
            
            self.windows.append({
                "window_index": w,
                "start_bit": start,
                "end_bit": end,
                "length": end - start,
                "qber": qber,
                "errors": int(errors) if len(window_alice) >= self.session_config.min_sifted_bits else 0,
            })


def run_session(config: SessionConfig, progress_callback=None) -> SessionResult:
    """
    Run a BB84 session with temporal windowing.
    
    Args:
        config: Session configuration
        progress_callback: Optional progress callback
        
    Returns:
        SessionResult with BB84 result and temporal windows
    """
    bb84_result = run_bb84_simulation(
        n_transmissions=config.n_transmissions,
        seed=config.seed,
        alice_p_z=config.alice_p_z,
        bob_p_z=config.bob_p_z,
        detection_efficiency=config.detection_efficiency,
        channel_type=config.channel_type,
        channel_params=config.channel_params,
        progress_callback=progress_callback,
    )
    
    return SessionResult(
        session_config=config,
        bb84_result=bb84_result,
    )


def run_multi_session(
    configs: List[SessionConfig],
    progress_callback=None,
) -> List[SessionResult]:
    """
    Run multiple BB84 sessions.
    
    Args:
        configs: List of session configurations
        progress_callback: Optional progress callback(step, session_idx, total)
        
    Returns:
        List of SessionResult
    """
    results = []
    for i, config in enumerate(configs):
        if progress_callback:
            progress_callback("session_start", i, len(configs))
        result = run_session(config, progress_callback=progress_callback)
        if progress_callback:
            progress_callback("session_complete", i, len(configs))
        results.append(result)
    return results


def create_sessions_from_yaml(yaml_path: str) -> List[SessionConfig]:
    """
    Load session configurations from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        List of SessionConfig
    """
    import yaml
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'sessions' in data:
        # Multiple sessions
        configs = []
        for sess_data in data['sessions']:
            configs.append(SessionConfig(**sess_data))
        return configs
    else:
        # Single session
        return [SessionConfig(**data)]