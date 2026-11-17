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
from pathlib import Path
from typing import List, Dict, Any, Optional
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
    attack_config: Dict[str, Any],
    logger
) -> Dict[str, Any]:
    """Run a BB84 session with attack and extract temporal features."""
    
    # Run BB84 simulation
    bb84_config = {
        'n_transmissions': config.n_transmissions,
        'seed': config.seed,
        'alice_p_z': config.alice_p_z,
        'bob_p_z': config.bob_p_z,
        'detection_efficiency': config.detection_efficiency,
        'dark_count_prob': 0.0,  # Default value since not in SessionConfig
        'channel_type': attack_config.get('channel_type', 'ideal'),
        'channel_params': attack_config.get('channel_params', {}),
        'attack_type': attack_config.get('attack_type', 'none'),
        'attack_params': attack_config.get('attack_params', {}),
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
    )
    
    # Add metadata
    features.update({
        'class_label': attack_config.get('class_label', 'unknown'),
        'class_index': attack_config.get('class_index', -1),
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
    min_sifted_bits: int = 200,
    output_dir: str = "data/bb84",
    log_level: str = "INFO",
    disable_progress: bool = False,
) -> str:
    """Generate the full BB84 dataset."""
    
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
    
    logger.info("=" * 60)
    logger.info("BB84 DATASET GENERATION")
    logger.info("=" * 60)
    logger.info(f"Classes: {len(BB84_CLASSES)}")
    logger.info(f"Sessions per class: {n_sessions_per_class}")
    logger.info(f"Total sessions: {n_sessions_per_class * len(BB84_CLASSES)}")
    logger.info(f"Transmissions per session: {n_transmissions:,}")
    logger.info(f"Window size: {window_size}, Stride: {stride}")
    logger.info(f"Min sifted bits per window: {min_sifted_bits}")
    
    all_sessions = []
    session_id = 0
    
    # For each attack class
    for class_idx, class_name in enumerate(BB84_CLASSES):
        attack_config = ATTACK_CONFIGS[class_name].copy()
        attack_config['class_label'] = class_name
        attack_config['class_index'] = class_idx
        
        logger.info(f"\nGenerating class {class_idx}: {class_name} ({n_sessions_per_class} sessions)")
        
        if progress.enabled:
            with progress.progress(
                total=n_sessions_per_class,
                desc=f"Class {class_name}",
                unit="session"
            ) as pbar:
                for i in range(n_sessions_per_class):
                    # Create session config with unique seed
                    seed = (class_idx * n_sessions_per_class + i) * 1000 + 42
                    session_config = create_session_config(
                        n_transmissions=n_transmissions,
                        window_size=window_size,
                        stride=stride,
                        min_sifted_bits=min_sifted_bits,
                        seed=seed,
                        alice_p_z=0.5,
                        bob_p_z=0.5,
                        detection_efficiency=1.0,
                        dark_count_prob=0.0,
                    )
                    
                    try:
                        features = run_bb84_session_with_features(
                            session_config, attack_config, logger
                        )
                        features['session_id'] = session_id
                        all_sessions.append(features)
                        session_id += 1
                    except Exception as e:
                        logger.error(f"Session {session_id} failed: {e}")
                        continue
                    
                    pbar.update(1)
        else:
            for i in range(n_sessions_per_class):
                seed = (class_idx * n_sessions_per_class + i) * 1000 + 42
                session_config = create_session_config(
                    n_transmissions=n_transmissions,
                    window_size=window_size,
                    stride=stride,
                    min_sifted_bits=min_sifted_bits,
                    seed=seed,
                    alice_p_z=0.5,
                    bob_p_z=0.5,
                    detection_efficiency=1.0,
                    dark_count_prob=0.0,
                )
                
                try:
                    features = run_bb84_session_with_features(
                        session_config, attack_config, logger
                    )
                    features['session_id'] = session_id
                    all_sessions.append(features)
                    session_id += 1
                except Exception as e:
                    logger.error(f"Session {session_id} failed: {e}")
                    continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_sessions)
    
    # Save as CSV
    csv_path = output_path / "bb84_temporal_dataset.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nDataset saved to {csv_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Classes: {df['class_label'].value_counts().to_dict()}")
    
    # Save metadata
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
        'attack_configs': ATTACK_CONFIGS,
    }
    
    meta_path = output_path / "bb84_dataset_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to {meta_path}")
    
    return str(csv_path)


def main():
    parser = argparse.ArgumentParser(description="Generate BB84 Temporal-QBER Dataset")
    parser.add_argument('--n-sessions-per-class', type=int, default=3000,
                        help='Number of sessions per class (default: 3000)')
    parser.add_argument('--n-transmissions', type=int, default=100000,
                        help='Number of qubit transmissions per session (default: 100000)')
    parser.add_argument('--window-size', type=int, default=50,
                        help='Temporal window size in bits (default: 50)')
    parser.add_argument('--stride', type=int, default=50,
                        help='Window stride in bits (default: 50)')
    parser.add_argument('--min-sifted-bits', type=int, default=200,
                        help='Minimum sifted bits per window (default: 200)')
    parser.add_argument('--output-dir', type=str, default='data/bb84',
                        help='Output directory (default: data/bb84)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--disable-progress', action='store_true',
                        help='Disable progress bars')
    
    args = parser.parse_args()
    
    generate_dataset(
        n_sessions_per_class=args.n_sessions_per_class,
        n_transmissions=args.n_transmissions,
        window_size=args.window_size,
        stride=args.stride,
        min_sifted_bits=args.min_sifted_bits,
        output_dir=args.output_dir,
        log_level=args.log_level,
        disable_progress=args.disable_progress,
    )


if __name__ == "__main__":
    main()