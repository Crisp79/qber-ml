#!/usr/bin/env python3
"""
Generate BB84 Simulation Dataset with Temporal-QBER Features

Generates a multi-session, multi-attack BB84 dataset with 63 temporal-QBER features
per session for machine learning-based attack detection.

This script reproduces the dataset generation methodology from the paper:
"Beyond the QBER Threshold: A Temporal QBER Based Machine Learning Framework 
for Multi Attack Detection in BB84 QKD" (arXiv:2608.04047v1)

Target: 24,000 balanced sessions (3,000 per class for 8 classes)
"""

import argparse
import sys
import json
import time
import os
import signal
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from temporal_qber.simulator import (
    BB84Protocol, run_bb84_simulation, SessionConfig, run_session
)
from temporal_qber.temporal import extract_temporal_qber_features, SessionConfig as TemporalSessionConfig, WindowConfig
from temporal_qber.logging_config import setup_logging, get_logger
from temporal_qber.progress import create_progress_manager

# ============================================================
# Live Status Display
# ============================================================

class LiveStatusDisplay:
    """Live-updating status display with global counter and recent activity log."""
    
    def __init__(self, total_sessions: int, logger, update_interval: int = 10, recent_size: int = 8):
        self.total = total_sessions
        self.logger = logger
        self.update_interval = update_interval
        self.recent = deque(maxlen=recent_size)
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.last_update = 0
        self._first_print = True
        self._lines_printed = 0
        
    def add_completion(self, session_key: str, class_name: str, elapsed: float, success: bool = True):
        """Record a completed session."""
        self.completed += 1
        if not success:
            self.failed += 1
        timestamp = time.strftime("%H:%M:%S")
        status = "OK" if success else "ERR"
        self.recent.append(f"[{timestamp}] {status} {class_name:25s} ({session_key})  {elapsed:.1f}s")
        
    def update(self, force: bool = False):
        """Update the display (respects update_interval)."""
        if not force and (self.completed - self.last_update) < self.update_interval:
            return
        self.last_update = self.completed
        self._render()
        
    def _render(self):
        """Render the live status display."""
        elapsed = time.time() - self.start_time
        pct = (self.completed / self.total * 100) if self.total > 0 else 0
        avg_time = elapsed / self.completed if self.completed > 0 else 0
        remaining = self.total - self.completed
        eta = avg_time * remaining if remaining > 0 else 0
        
        # Build status line (ASCII for Windows compatibility)
        bar_width = 40
        filled = int(bar_width * self.completed / self.total) if self.total > 0 else 0
        bar = "#" * filled + "-" * (bar_width - filled)
        
        status_line = (
            f"\r{'='*78}\n"
            f"BB84 DATASET GENERATION  |  {self.completed:>5}/{self.total:<5}  ({pct:5.1f}%)  "
            f"[{bar}]  OK:{self.completed-self.failed}  ERR:{self.failed}  "
            f"Avg: {avg_time:.1f}s  ETA: {self._fmt_time(eta)}  Elapsed: {self._fmt_time(elapsed)}\n"
            f"{'-'*78}\n"
        )
        
        # Recent activity
        if self.recent:
            status_line += "RECENT:\n"
            for line in self.recent:
                status_line += f"  {line}\n"
        else:
            status_line += "RECENT: (waiting for first completion...)\n"
            
        status_line += f"{'='*78}"
        
        # Clear previous lines and print new status
        if self._first_print:
            sys.stdout.write(status_line + "\n")
            self._first_print = False
            self._lines_printed = status_line.count('\n') + 1
        else:
            # Move cursor up and clear lines
            sys.stdout.write(f"\033[{self._lines_printed}A")
            sys.stdout.write("\033[J")  # Clear from cursor to end
            sys.stdout.write(status_line + "\n")
            self._lines_printed = status_line.count('\n') + 1
        sys.stdout.flush()
        
    def _fmt_time(self, seconds: float) -> str:
        """Format seconds as H:MM:SS or M:SS."""
        if seconds < 0:
            return "??:??"
        s = int(seconds)
        h, m = divmod(s, 3600)
        m, s = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"
        
    def finalize(self):
        """Final update and newline."""
        self.update(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()


# Paper's attack classes (8 classes: 1 normal + 7 attacks)
BB84_CLASSES = [
    "normal",             # 0 - no eavesdropper
    "intercept_resend",   # 1
    "photon_number_splitting",  # 2
    "beam_splitting",     # 3
    "time_shift",         # 4
    "trojan_horse",       # 5
    "quantum_cloning",    # 6
    "fake_state",         # 7
]

# Checkpoint file for resume capability
CHECKPOINT_FILE = "bb84_dataset_checkpoint.json"

# Attack configurations (paper's parameters where specified)
ATTACK_CONFIGS = {
    "normal": {
        "attack_type": "none",
        "attack_params": {},
        "channel_type": "ideal",
        "channel_params": {},
    },
    "intercept_resend": {
        "attack_type": "intercept_resend",
        "attack_params": {},
        "channel_type": "ideal",
        "channel_params": {},
    },
    "photon_number_splitting": {
        "attack_type": "pns_storage",
        "attack_params": {"mean_photon_number": 0.2},  # Typical WCP mu
        "channel_type": "ideal",
        "channel_params": {},
    },
    "beam_splitting": {
        "attack_type": "beam_splitting",
        "attack_params": {"reflection_coeff": 0.1, "mean_photon_number": 0.2},
        "channel_type": "ideal",
        "channel_params": {},
    },
    "time_shift": {
        "attack_type": "time_shift_intercept",
        "attack_params": {"time_shift": 0.5},
        "channel_type": "ideal",
        "channel_params": {},
    },
    "trojan_horse": {
        "attack_type": "trojan_horse",
        "attack_params": {"probe_photons": 1000},
        "channel_type": "ideal",
        "channel_params": {},
    },
    "quantum_cloning": {
        "attack_type": "quantum_cloning",
        "attack_params": {"cloning_type": "universal"},
        "channel_type": "ideal",
        "channel_params": {},
    },
    "fake_state": {
        "attack_type": "fake_state",
        "attack_params": {
            "attack_subtype": "detector_mismatch",
            "det0_efficiency": 1.0,
            "det1_efficiency": 0.8
        },
        "channel_type": "ideal",
        "channel_params": {},
    },
}

# ============================================================
# Parameter ranges for per-session randomization
# ============================================================
# Prevents shortcut learning: models can't memorize fixed parameter signatures.
# Each session gets randomized attack + channel parameters within these ranges.
# Hold-out ranges can be used for testing (train on [0.1,0.3], test on [0.4,0.6]).

# Base channel noise ranges applied to ALL classes (including normal)
BASE_CHANNEL_PARAMS = {
    "channel_types": ["ideal", "depolarizing", "phase_flip", "bit_flip"],
    "channel_type_probs": [0.5, 0.2, 0.2, 0.1],  # Mostly ideal, some noise
    "loss_prob_range": (0.0, 0.15),      # Photon loss 0-15%
    "depolarizing_prob_range": (0.0, 0.08),  # Depolarizing noise 0-8%
    "phase_flip_prob_range": (0.0, 0.08),   # Phase flip 0-8%
    "bit_flip_prob_range": (0.0, 0.08),     # Bit flip 0-8%
}

# Detection efficiency range (Bob's detector efficiency 80-100%)
DETECTION_EFFICIENCY_RANGE = (0.80, 1.00)

# Dark count probability range (0-5%)
DARK_COUNT_PROB_RANGE = (0.0, 0.05)

# Decoy-state source used for every class (vacuum + weak decoy; Ma, Qi, Zhao & Lo 2005).
# Alice sends weak coherent pulses: signal mu = 0.5 (70% of pulses, the only ones used for
# the key and the 63 temporal features), weak decoy nu = 0.1 (20%), vacuum (10%). Fixed
# protocol constants, identical for all classes.
DECOY_CONFIG = {"intensities": [0.5, 0.1, 0.0], "probabilities": [0.7, 0.2, 0.1]}

# Per-attack parameter ranges (sampled per session)
ATTACK_PARAM_RANGES = {
    "normal": {
        # Normal: no attack, only channel noise
        "attack_type": "none",
    },
    "intercept_resend": {
        "attack_types": ["intercept_resend", "partial_intercept_resend"],
        "attack_type_probs": [0.7, 0.3],
        "interception_prob_range": (0.3, 1.0),  # for partial IR
        "p_z_range": (0.4, 0.6),  # Eve's Z basis bias (0.5 = unbiased)
    },
    # Every attack class MUST set "attack_type" (a create_attack() name); a missing key
    # silently falls back to "none" and produces unattacked sessions under an attack label.
    "photon_number_splitting": {
        "attack_type": "pns_storage",
        # Photon numbers come from Alice's decoy-state source (DECOY_CONFIG).
        # Standard PNS (Brassard et al. 2000): block single-photon pulses. "pass" has no
        # observable effect at Bob and "intercept_resend" is plain intercept-resend.
        "strategies": ["block"],
        "strategy_probs": [1.0],
        "block_prob_range": (0.2, 0.7),  # fraction of single-photon pulses Eve blocks
    },
    "beam_splitting": {
        "attack_type": "beam_splitting",
        # Eve's tap fraction. Floor 0.15: weaker taps hide inside normal channel loss
        # (0-15%) and were detected <50% of the time; they belong in a separate
        # weak-attack evaluation set, not the training data.
        "reflection_coeff_range": (0.15, 0.3),
        # Photon numbers come from Alice's decoy-state source (DECOY_CONFIG).
    },
    "time_shift": {
        "attack_type": "time_shift_intercept",  # intercept_prob below is only used by this variant
        # Gaussian efficiency curves of Bob's two detectors (ns). With these ranges the
        # efficiency ratio disfavoured/favoured detector at the shifted time spans ~0.1-0.9.
        "time_shift_range": (0.05, 0.2),  # shift past the favoured detector's peak
        "det_width_range": (0.25, 0.5),   # detector efficiency-curve width
        "det_separation_range": (0.15, 0.35),  # offset between the two detectors' peaks
        "intercept_prob_range": (0.5, 1.0),  # for combined time_shift+IR
    },
    "trojan_horse": {
        "attack_type": "trojan_horse",
        "probe_photons_range": (500, 5000),
        "back_reflection_coeff_range": (1e-6, 1e-3),
        "basis_inference_prob_range": (0.7, 0.95),
        "detector_saturation_prob_range": (0.0, 0.02),
        # Spurious-click probability from probe light leaking into Bob's detectors; the
        # leak coefficient is derived from it and probe_photons. Floor 0.02: below that the
        # probe clicks hide inside dark counts and were detected ~60% of the time.
        "probe_click_prob_range": (0.02, 0.18),
    },
    "quantum_cloning": {
        "attack_type": "quantum_cloning",
        "cloning_types": ["universal", "phase_covariant", "state_dependent", "asymmetric"],
        "cloning_type_probs": [0.4, 0.3, 0.2, 0.1],
    },
    "fake_state": {
        "attack_type": "fake_state",
        # "wavelength", "intensity", "combined" and "time_bin" are no longer sampled: the
        # first three have no observable effect at Bob and "combined" duplicates
        # phase_remapped.
        "subtypes": ["phase_remapped", "detector_mismatch", "blinding"],
        "subtype_probs": [0.34, 0.33, 0.33],
        "mismatch_residual_range": (0.0, 0.3),  # detector_mismatch: residual efficiency outside Eve's basis
        # Floor 0.6 rad: smaller remapping phases cause almost no errors (detected 3-12%).
        "phase_shift_range": (0.6, np.pi),
        "det0_efficiency_range": (0.7, 1.0),
        "det1_efficiency_range": (0.7, 1.0),
        "multi_photon_prob_range": (0.0, 0.3),
        "wavelength_shift_range": (0.0, 10.0),
    },
}


def randomize_attack_params(class_name: str, rng: np.random.Generator) -> Dict[str, Any]:
    """
    Randomize attack and channel parameters for a single session.
    
    Prevents shortcut learning by ensuring each session has unique
    parameter values within defined ranges.
    
    Args:
        class_name: The attack class name
        rng: Random number generator for reproducibility
        
    Returns:
        Dict with 'attack_type', 'attack_params', 'channel_type',
        'channel_params', 'detection_efficiency', 'dark_count_prob'
    """
    ranges = ATTACK_PARAM_RANGES.get(class_name, {})
    
    # --- Attack type selection ---
    attack_type = ranges.get("attack_type", "none")
    
    if class_name == "intercept_resend":
        attack_type = rng.choice(
            ranges["attack_types"], p=ranges["attack_type_probs"]
        )
        attack_params = {}
        if attack_type == "partial_intercept_resend":
            attack_params["interception_prob"] = float(
                rng.uniform(*ranges["interception_prob_range"])
            )
        attack_params["p_z"] = float(rng.uniform(*ranges["p_z_range"]))
    
    elif class_name == "photon_number_splitting":
        attack_params = {
            "single_photon_strategy": rng.choice(
                ranges["strategies"], p=ranges["strategy_probs"]
            ),
            "single_photon_block_prob": float(rng.uniform(*ranges["block_prob_range"])),
        }
    
    elif class_name == "beam_splitting":
        attack_params = {
            "reflection_coeff": float(rng.uniform(*ranges["reflection_coeff_range"])),
        }
    
    elif class_name == "time_shift":
        attack_params = {
            "time_shift": float(rng.uniform(*ranges["time_shift_range"])),
            "det_width": float(rng.uniform(*ranges["det_width_range"])),
            "det1_peak_time": float(rng.uniform(*ranges["det_separation_range"])),
            "intercept_prob": float(rng.uniform(*ranges["intercept_prob_range"])),
        }
    
    elif class_name == "trojan_horse":
        attack_params = {
            "probe_photons": int(rng.integers(*ranges["probe_photons_range"])),
            "back_reflection_coeff": float(
                rng.uniform(*ranges["back_reflection_coeff_range"])
            ),
            "basis_inference_prob": float(
                rng.uniform(*ranges["basis_inference_prob_range"])
            ),
            "detector_saturation_prob": float(
                rng.uniform(*ranges["detector_saturation_prob_range"])
            ),
        }
        probe_click_prob = float(rng.uniform(*ranges["probe_click_prob_range"]))
        attack_params["detector_leak_coeff"] = float(
            -np.log1p(-probe_click_prob) / attack_params["probe_photons"]
        )
    
    elif class_name == "quantum_cloning":
        cloning_type = rng.choice(
            ranges["cloning_types"], p=ranges["cloning_type_probs"]
        )
        attack_params = {"cloning_type": cloning_type}
    
    elif class_name == "fake_state":
        subtype = rng.choice(ranges["subtypes"], p=ranges["subtype_probs"])
        attack_params = {
            "attack_subtype": subtype,
            "phase_shift": float(rng.uniform(*ranges["phase_shift_range"])),
            "det0_efficiency": float(rng.uniform(*ranges["det0_efficiency_range"])),
            "det1_efficiency": float(rng.uniform(*ranges["det1_efficiency_range"])),
            "multi_photon_prob": float(rng.uniform(*ranges["multi_photon_prob_range"])),
            "wavelength_shift": float(rng.uniform(*ranges["wavelength_shift_range"])),
            "mismatch_residual": float(rng.uniform(*ranges["mismatch_residual_range"])),
        }
    
    else:
        # normal: no attack
        attack_params = {}
    
    # --- Channel noise (applied to ALL classes including normal) ---
    channel_type = rng.choice(
        BASE_CHANNEL_PARAMS["channel_types"],
        p=BASE_CHANNEL_PARAMS["channel_type_probs"],
    )
    
    channel_params = {}
    if channel_type == "depolarizing":
        channel_params["depolarizing_prob"] = float(
            rng.uniform(*BASE_CHANNEL_PARAMS["depolarizing_prob_range"])
        )
    elif channel_type == "phase_flip":
        channel_params["phase_flip_prob"] = float(
            rng.uniform(*BASE_CHANNEL_PARAMS["phase_flip_prob_range"])
        )
    elif channel_type == "bit_flip":
        channel_params["bit_flip_prob"] = float(
            rng.uniform(*BASE_CHANNEL_PARAMS["bit_flip_prob_range"])
        )
    channel_params["loss_probability"] = float(
        rng.uniform(*BASE_CHANNEL_PARAMS["loss_prob_range"])
    )
    
    # --- Detector parameters ---
    detection_efficiency = float(rng.uniform(*DETECTION_EFFICIENCY_RANGE))
    dark_count_prob = float(rng.uniform(*DARK_COUNT_PROB_RANGE))
    
    return {
        "attack_type": attack_type,
        "attack_params": attack_params,
        "channel_type": channel_type,
        "channel_params": channel_params,
        "detection_efficiency": detection_efficiency,
        "dark_count_prob": dark_count_prob,
    }


def load_checkpoint(output_path: Path) -> Dict[str, Any]:
    """Load checkpoint data if exists."""
    checkpoint_path = output_path / CHECKPOINT_FILE
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {
        "completed_sessions": [],
        "session_results": {},
        "next_session_id": 0,
        "total_sessions": 0,
        "start_time": time.time(),
    }


def save_checkpoint(output_path: Path, checkpoint: Dict[str, Any]):
    """Save checkpoint data."""
    checkpoint_path = output_path / CHECKPOINT_FILE
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)


def session_key(class_idx: int, session_idx: int, n_sessions_per_class: int) -> str:
    """Generate unique session key for checkpoint tracking."""
    return f"{class_idx}_{session_idx}"


def generate_session_task(args: Tuple) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Worker function to generate a single BB84 session with features.
    
    Args:
        args: Tuple of (class_idx, class_name, session_idx, n_transmissions, 
                      window_size, stride, min_sifted_bits, seed, attack_config)
        attack_config can be a dict (fixed config) or None (use randomization)
    
    Returns:
        Tuple of (session_key, features_dict, error_message)
    """
    class_idx, class_name, session_idx, n_transmissions, window_size, stride, \
    min_sifted_bits, seed, attack_config = args
    
    try:
        # Import here to avoid pickling issues
        from temporal_qber.simulator import run_bb84_simulation
        from temporal_qber.temporal import extract_temporal_qber_features, extract_monitoring_features, WindowConfig
        from temporal_qber.simulator.session import SessionConfig as TemporalSessionConfig
        
        # Create per-session RNG for parameter randomization
        # Derive a unique sub-seed from the session seed
        session_rng = np.random.default_rng(seed * 1000003 + class_idx * 97 + session_idx)
        
        # Randomize parameters per session (overrides fixed config)
        randomized = randomize_attack_params(class_name, session_rng)
        
        # Use randomized params, falling back to fixed config if provided
        attack_type = attack_config.get('attack_type', randomized['attack_type']) if attack_config else randomized['attack_type']
        attack_params = attack_config.get('attack_params', randomized['attack_params']) if attack_config else randomized['attack_params']
        channel_type = attack_config.get('channel_type', randomized['channel_type']) if attack_config else randomized['channel_type']
        channel_params = attack_config.get('channel_params', randomized['channel_params']) if attack_config else randomized['channel_params']
        detection_efficiency = attack_config.get('detection_efficiency', randomized['detection_efficiency']) if attack_config else randomized['detection_efficiency']
        dark_count_prob = attack_config.get('dark_count_prob', randomized['dark_count_prob']) if attack_config else randomized['dark_count_prob']
        
        # Create session config
        session_config = TemporalSessionConfig(
            n_transmissions=n_transmissions,
            window_size=window_size,
            stride=stride,
            min_sifted_bits=min_sifted_bits,
            seed=seed,
            alice_p_z=0.5,
            bob_p_z=0.5,
            detection_efficiency=detection_efficiency,
            channel_type=channel_type,
            channel_params=channel_params,
        )
        
        # Run BB84 simulation
        bb84_config = {
            'n_transmissions': n_transmissions,
            'seed': seed,
            'alice_p_z': 0.5,
            'bob_p_z': 0.5,
            'detection_efficiency': detection_efficiency,
            'dark_count_prob': dark_count_prob,
            'channel_type': channel_type,
            'channel_params': channel_params,
            'attack_type': attack_type,
            'attack_params': attack_params,
            'decoy_config': DECOY_CONFIG,
        }

        result = run_bb84_simulation(**bb84_config)
        
        # Extract temporal features. Bob's detections are filtered exactly like the sifted
        # key: matching basis, detected and - with the decoy-state source - signal pulses only.
        key_mask = result.matching_bases_mask & result.bob_detected
        if result.pulse_intensity is not None:
            key_mask = key_mask & (result.pulse_intensity == 0)
        bob_detected_filtered = result.bob_detected[key_mask]

        window_config = WindowConfig(
            window_size=window_size,
            stride=stride,
            min_sifted_bits=min_sifted_bits,
            include_partial_trailing=False,
        )
        
        # Get loss sequence from result
        loss_seq = getattr(result, 'loss_seq', None)
        
        features = extract_temporal_qber_features(
            sifted_alice=result.sifted_alice_bits,
            sifted_bob=result.sifted_bob_bits,
            bob_detected=bob_detected_filtered,
            # Basis-specific sifted bits for basis-dependent features
            sifted_alice_rect=result.sifted_alice_rect,
            sifted_bob_rect=result.sifted_bob_rect,
            sifted_alice_diag=result.sifted_alice_diag,
            sifted_bob_diag=result.sifted_bob_diag,
            # Loss sequence for channel-interaction features
            loss_seq=loss_seq,
            window_config=WindowConfig(
                window_size=window_size,
                stride=stride,
                min_sifted_bits=min_sifted_bits,
            ),
            min_sifted_bits=min_sifted_bits,
            detection_efficiency=detection_efficiency,
            dark_count_prob=dark_count_prob,
            afterpulse_prob=0.0,
        )
        
        # Detector monitoring statistics (separate from the paper's 63 features)
        features.update(extract_monitoring_features(
            result.bob_detected, result.bob_double_click, result.bob_empty_gate_click,
            pulse_intensity=result.pulse_intensity, intensities=result.decoy_intensities))

        # Add metadata (class_name/class_idx from function args)
        features.update({
            'class_label': class_name,
            'class_index': class_idx,
            'n_transmissions': result.n_transmissions,
            'seed': result.seed,
            'channel_type': result.channel_config.get('channel_type', 'unknown'),
            'attack_type': result.attack_type,
            'attack_params': str(result.attack_params),
            'sifted_key_length': result.sifted_key_length,
            'sifting_efficiency': result.sifting_efficiency,
            'overall_qber': result.qber,
            'elapsed_time': result.elapsed_time,
        })
        
        session_key_str = f"{args[0]}_{args[2]}"
        return session_key_str, features, None
        
    except Exception as e:
        session_key_str = f"{args[0]}_{args[2]}"
        return session_key_str, {}, f"Class {args[1]}, session {args[2]}, seed {args[7]}: {str(e)}"


def generate_dataset(
    n_sessions_per_class: int = 3000,
    n_transmissions: int = 100000,
    window_size: int = 50,
    stride: int = 50,
    min_sifted_bits: int = 30,
    output_dir: str = "data/bb84",
    log_level: str = "INFO",
    disable_progress: bool = False,
    workers: int = 1,
) -> str:
    """Generate the full BB84 dataset with multiprocessing support."""


def create_session_config(
    n_transmissions: int,
    window_size: int = 50,
    stride: int = 50,
    min_sifted_bits: int = 200,
    seed: Optional[int] = None,
    **protocol_params
) -> TemporalSessionConfig:
    """Create a temporal session configuration."""
    # Filter out params not in SessionConfig
    valid_params = {
        'n_transmissions', 'window_size', 'stride', 'min_sifted_bits',
        'seed', 'alice_p_z', 'bob_p_z', 'detection_efficiency',
        'channel_type', 'channel_params', 'include_partial_trailing'
    }
    filtered = {k: v for k, v in protocol_params.items() if k in valid_params}
    return TemporalSessionConfig(
        n_transmissions=n_transmissions,
        window_size=window_size,
        stride=stride,
        min_sifted_bits=min_sifted_bits,
        seed=seed,
        **filtered
    )


def run_bb84_session_with_features(
    config: TemporalSessionConfig,
    class_name: str,
    class_idx: int,
    logger,
    session_rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Run a BB84 session with attack and extract temporal features.
    
    Uses per-session parameter randomization to prevent shortcut learning.
    """
    
    # Create per-session RNG if not provided
    if session_rng is None:
        session_rng = np.random.default_rng(config.seed * 1000003 + class_idx * 97)
    
    # Randomize parameters per session
    randomized = randomize_attack_params(class_name, session_rng)
    
    # Run BB84 simulation
    bb84_config = {
        'n_transmissions': config.n_transmissions,
        'seed': config.seed,
        'alice_p_z': config.alice_p_z,
        'bob_p_z': config.bob_p_z,
        'detection_efficiency': randomized['detection_efficiency'],
        'dark_count_prob': randomized['dark_count_prob'],
        'channel_type': randomized['channel_type'],
        'channel_params': randomized['channel_params'],
        'attack_type': randomized['attack_type'],
        'attack_params': randomized['attack_params'],
    }
    
    result = run_bb84_simulation(**bb84_config)
    
    # Extract temporal features
    # Filter bob_detected to match sifted key length
    bob_detected_filtered = result.bob_detected[result.matching_bases_mask & result.bob_detected]
    
    # Create WindowConfig from temporal config
    window_config = WindowConfig(
        window_size=config.window_size,
        stride=config.stride,
        min_sifted_bits=config.min_sifted_bits,
        include_partial_trailing=config.include_partial_trailing,
    )
    
    features = extract_temporal_qber_features(
        sifted_alice=result.sifted_alice_bits,
        sifted_bob=result.sifted_bob_bits,
        bob_detected=bob_detected_filtered,
        window_config=window_config,
        min_sifted_bits=config.min_sifted_bits,
        detection_efficiency=randomized['detection_efficiency'],
        dark_count_prob=randomized['dark_count_prob'],
        afterpulse_prob=0.0,
    )
    
    # Add metadata
    features.update({
        'class_label': class_name,
        'class_index': class_idx,
        'n_transmissions': result.n_transmissions,
        'seed': result.seed,
        'channel_type': result.channel_config.get('channel_type', 'unknown'),
        'attack_type': result.attack_type,
        'attack_params': str(result.attack_params),
        'sifted_key_length': result.sifted_key_length,
        'sifting_efficiency': result.sifting_efficiency,
        'overall_qber': result.qber,
        'elapsed_time': result.elapsed_time,
    })
    
    return features


def generate_dataset(
    n_sessions_per_class: int = 3000,
    n_transmissions: int = 100000,
    window_size: int = 50,
    stride: int = 50,
    min_sifted_bits: int = 30,
    output_dir: str = "data/bb84",
    input_dir: str = None,
    log_level: str = "INFO",
    disable_progress: bool = False,
    workers: int = 1,
) -> str:
    """Generate the full BB84 dataset with multiprocessing support."""
    
    if input_dir is None:
        input_dir = output_dir
    
    # Setup logging
    logger = setup_logging(
        log_level=getattr(__import__('logging'), log_level.upper()),
        log_dir=f"{output_dir}/logs",
        disable_console=False
    )
    
    # Setup progress manager
    progress = create_progress_manager(enabled=not disable_progress, logger=logger)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Load checkpoint if exists (only resume if input_dir == output_dir, i.e., continuing same generation)
    checkpoint = load_checkpoint(output_path) if input_dir == output_dir else load_checkpoint(input_path)
    # Only use checkpoint if we're continuing the same generation
    if input_dir != output_dir:
        checkpoint = {"completed_sessions": [], "session_results": {}, "next_session_id": 0}
    completed_keys = set(checkpoint.get("completed_sessions", []))
    session_results = checkpoint.get("session_results", {})
    session_id = checkpoint.get("next_session_id", 0)
    
    logger.info("=" * 60)
    logger.info("BB84 DATASET GENERATION")
    logger.info("=" * 60)
    logger.info(f"Classes: {len(BB84_CLASSES)}")
    logger.info(f"Sessions per class: {n_sessions_per_class}")
    logger.info(f"Total sessions: {n_sessions_per_class * len(BB84_CLASSES)}")
    logger.info(f"Transmissions per session: {n_transmissions:,}")
    logger.info(f"Window size: {window_size}, Stride: {stride}")
    logger.info(f"Min sifted bits per window: {min_sifted_bits}")
    logger.info(f"Workers: {workers}")
    if completed_keys:
        logger.info(f"Resuming from checkpoint: {len(completed_keys)} sessions already completed")
    
    # Prepare all session tasks
    # Attack config is passed as None to trigger per-session parameter randomization
    # in generate_session_task. This prevents shortcut learning by ensuring each
    # session has unique parameter values.
    all_tasks = []
    for class_idx, class_name in enumerate(BB84_CLASSES):
        for i in range(n_sessions_per_class):
            session_key_str = session_key(class_idx, i, n_sessions_per_class)
            if session_key_str in completed_keys:
                continue
            
            seed = (class_idx * n_sessions_per_class + i) * 1000 + 42
            task_args = (
                class_idx, class_name, i, n_transmissions, window_size, stride,
                min_sifted_bits, seed, None,  # None triggers parameter randomization
            )
            all_tasks.append((session_key_str, task_args))
    
    logger.info(f"Total sessions to generate: {len(all_tasks)}")
    
    # Initialize live status display
    total_to_generate = len(all_tasks)
    status = LiveStatusDisplay(total_to_generate, logger, update_interval=10, recent_size=8)
    
    # Generate sessions
    all_sessions = []
    # session_results keeps the sessions loaded from a checkpoint (resetting it here used to
    # drop them from the final CSV when a run was resumed)
    checkpoint_interval = 100  # Save checkpoint every N sessions
    
    start_time = time.time()
    
    if workers == 1:
        # Sequential processing
        for i, (session_key_str, task_args) in enumerate(all_tasks):
            session_key_str, features, error = generate_session_task(task_args)
            
            class_name = task_args[1]
            session_elapsed = time.time() - start_time
            
            if error:
                logger.error(f"Session {session_key_str} failed: {error}")
                status.add_completion(session_key_str, class_name, 0, success=False)
            else:
                session_results[session_key_str] = features
                completed_keys.add(session_key_str)
                status.add_completion(session_key_str, class_name, session_elapsed, success=True)
            
            status.update()
            
            # Checkpoint periodically
            if (i + 1) % checkpoint_interval == 0:
                elapsed = time.time() - start_time
                checkpoint.update({
                    "completed_sessions": list(completed_keys),
                    "session_results": session_results,
                    "next_session_id": session_id + len(completed_keys),
                    "total_sessions": len(all_tasks) + len(completed_keys),
                    "elapsed_time": elapsed,
                    "workers": workers,
                })
                save_checkpoint(output_path, checkpoint)
    else:
        # Parallel processing with ProcessPoolExecutor
        # Use bounded submission to avoid queuing all tasks at once
        max_queue_size = workers * 4
        
        with multiprocessing.Pool(processes=workers) as pool:
            # Submit tasks in batches
            pending = {}
            completed = 0
            
            # Submit the initial batch; next_submit indexes the next task in all_tasks.
            # (The old index, len(completed_keys) + len(pending), lagged behind after every
            # failed session and after a resume, resubmitting old tasks - an endless loop
            # once sessions fail.)
            next_submit = min(max_queue_size, len(all_tasks))
            for session_key_str, task_args in all_tasks[:next_submit]:
                future = pool.apply_async(generate_session_task, (task_args,))
                pending[future] = (session_key_str, task_args[1])  # store class_name too

            # Process completed tasks and submit new ones
            while pending:
                ready = [f for f in pending if f.ready()]
                if not ready:
                    time.sleep(0.05)  # wait without busy-spinning a core
                    continue
                future = ready[0]
                session_key_str, class_name = pending.pop(future)
                try:
                    _, features, error = future.get()
                    if error:
                        logger.error(f"Session {session_key_str} failed: {error}")
                        status.add_completion(session_key_str, class_name, 0, success=False)
                    else:
                        session_results[session_key_str] = features
                        completed_keys.add(session_key_str)
                        status.add_completion(session_key_str, class_name, time.time() - start_time, success=True)
                except Exception as e:
                    logger.error(f"Session {session_key_str} failed with exception: {e}")
                    status.add_completion(session_key_str, class_name, 0, success=False)

                completed += 1
                status.update()

                # Submit next task if available
                if next_submit < len(all_tasks):
                    next_key, next_args = all_tasks[next_submit]
                    next_submit += 1
                    new_future = pool.apply_async(generate_session_task, (next_args,))
                    pending[new_future] = (next_key, next_args[1])

                # Checkpoint periodically
                if completed % checkpoint_interval == 0:
                    elapsed = time.time() - start_time
                    checkpoint.update({
                        "completed_sessions": list(completed_keys),
                        "session_results": session_results,
                        "next_session_id": session_id + len(completed_keys),
                        "total_sessions": len(all_tasks) + len(completed_keys),
                        "elapsed_time": elapsed,
                        "workers": workers,
                    })
                    save_checkpoint(output_path, checkpoint)
    
    # Finalize status display
    status.finalize()
    
    # Collect results in deterministic order
    all_sessions = []
    for class_idx, class_name in enumerate(BB84_CLASSES):
        for i in range(n_sessions_per_class):
            session_key_str = session_key(class_idx, i, n_sessions_per_class)
            if session_key_str in session_results:
                features = session_results[session_key_str].copy()
                features['session_id'] = session_id
                all_sessions.append(features)
                session_id += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(all_sessions)
    
    # Save as CSV
    csv_path = output_path / "bb84_temporal_dataset.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nDataset saved to {csv_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Classes: {df['class_label'].value_counts().to_dict()}")
    
    # Save metadata with multiprocessing info
    elapsed_total = time.time() - start_time
    metadata = {
        'n_sessions_total': len(df),
        'n_sessions_per_class': n_sessions_per_class,
        'n_classes': len(BB84_CLASSES),
        'class_names': BB84_CLASSES,
        'n_transmissions': n_transmissions,
        'window_size': window_size,
        'stride': stride,
        'min_sifted_bits': min_sifted_bits,
        'n_features': len(df.columns) - 1,  # Exclude class_label
        'feature_names': [c for c in df.columns if c != 'class_label'],
        'generation_time': time.time(),
        'elapsed_seconds': elapsed_total,
        'workers': workers,
        'attack_configs': ATTACK_CONFIGS,
    }
    
    meta_path = output_path / "bb84_dataset_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to {meta_path}")
    
    # Remove checkpoint on successful completion
    checkpoint_path = output_path / CHECKPOINT_FILE
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    logger.info(f"\nGeneration complete in {elapsed_total:.1f}s with {workers} workers")
    return str(csv_path)


def main():
    parser = argparse.ArgumentParser(description="Generate BB84 Temporal-QBER Dataset")
    parser.add_argument('--n-sessions-per-class', type=int, default=3000,
                        help='Number of sessions per class (default: 3000)')
    parser.add_argument('--n-transmissions', type=int, default=100000,
                        help='Number of qubit transmissions per session (default: 100000)')
    parser.add_argument('--window-size', type=int, default=50,
                        help='Temporal window size in sifted bits (default: 50, paper W=50)')
    parser.add_argument('--stride', type=int, default=50,
                        help='Window stride in sifted bits (default: 50)')
    parser.add_argument('--min-sifted-bits', type=int, default=30,
                        help='Minimum sifted bits per window for valid QBER (default: 30)')
    parser.add_argument('--output-dir', '--out-dir', type=str, default='data/bb84',
                        dest='output_dir',
                        help='Output directory (default: data/bb84)')
    parser.add_argument('--input-dir', '--in-dir', type=str, default=None,
                        dest='input_dir',
                        help='Input directory for checkpoint resume (default: same as output-dir)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--disable-progress', action='store_true',
                        help='Disable progress bars')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker processes for parallel generation (default: 1)')
    
    args = parser.parse_args()
    
    input_dir = args.input_dir or args.output_dir
    
    generate_dataset(
        n_sessions_per_class=args.n_sessions_per_class,
        n_transmissions=args.n_transmissions,
        window_size=args.window_size,
        stride=args.stride,
        min_sifted_bits=args.min_sifted_bits,
        output_dir=args.output_dir,
        input_dir=input_dir,
        log_level=args.log_level,
        disable_progress=args.disable_progress,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()