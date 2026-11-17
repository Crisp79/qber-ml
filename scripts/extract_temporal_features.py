#!/usr/bin/env python3
"""
Temporal QBER Feature Extraction CLI

Extracts temporal QBER features from BB84 simulation sessions.

Usage:
    uv run python scripts/extract_temporal_features.py --config configs/bb84.yaml
    uv run python scripts/extract_temporal_features.py --n-transmissions 100000 --window-size 50 --stride 25
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.temporal_qber.simulator import (
    BB84Protocol,
    run_bb84_simulation,
    IdealChannel,
    create_channel,
)
from src.temporal_qber.temporal import (
    SessionConfig,
    run_session,
    extract_temporal_qber_features,
    calculate_session_qber,
)
from src.temporal_qber.logging_config import setup_logging, get_logger
from src.temporal_qber.progress import create_progress_manager


def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_results(result: dict, logger):
    """Print simulation results."""
    logger.info("=" * 60)
    logger.info("TEMPORAL QBER FEATURE EXTRACTION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Transmissions:         {result.get('n_transmissions', 'N/A'):,}")
    logger.info(f"Windows:               {result.get('total_windows', 0)}")
    logger.info(f"Valid windows:         {result.get('valid_windows', 0)}")
    logger.info(f"Sifted key length:     {result.get('sifted_key_length', 0):,}")
    logger.info(f"Sifting efficiency:    {result.get('sifting_efficiency', 0):.4f}")
    logger.info(f"QBER:                  {result.get('qber', 0):.6f}")
    logger.info(f"Agreement:             {result.get('agreement', 0):.6f}")
    logger.info(f"Elapsed time:          {result.get('elapsed_time', 0):.2f}s")
    logger.info("=" * 60)
    
    # Print key feature statistics
    logger.info("Key Features:")
    logger.info(f"  QBER Mean:           {result.get('qber_mean', 0):.6f}")
    logger.info(f"  QBER Std:            {result.get('qber_std', 0):.6f}")
    logger.info(f"  QBER Skew:           {result.get('qber_skew', 0):.6f}")
    logger.info(f"  QBER Kurtosis:       {result.get('qber_kurtosis', 0):.6f}")
    logger.info(f"  Spike Density:       {result.get('spike_density', 0):.6f}")
    logger.info(f"  Burst Count:         {result.get('burst_count', 0):.2f}")
    logger.info(f"  Jump Energy:         {result.get('jump_energy', 0):.6f}")
    logger.info(f"  Temporal Drift:      {result.get('temporal_drift', 0):.6f}")
    logger.info(f"  Linear Slope:        {result.get('linear_slope', 0):.6f}")
    logger.info(f"  Trend Direction:     {result.get('trend_direction', 0):.2f}")
    logger.info(f"  R-squared:           {result.get('r_squared', 0):.6f}")
    logger.info(f"  Early-Late Diff:     {result.get('early_late_diff', 0):.6f}")
    logger.info(f"  Num Changes:         {result.get('num_changes', 0)}")
    logger.info(f"  Max Change:          {result.get('max_change', 0):.6f}")
    logger.info(f"  Mean Abs Change:     {result.get('mean_abs_change', 0):.6f}")


def save_features_json(result: dict, output_path: str):
    """Save features to JSON file."""
    # Convert numpy types to native Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert(result), f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Temporal QBER Feature Extraction for BB84")
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--n-transmissions', type=int, default=100000, 
                        help='Number of qubit transmissions')
    parser.add_argument('--window-size', type=int, default=50, 
                        help='Temporal window size in bits')
    parser.add_argument('--stride', type=int, default=None, 
                        help='Window stride (default: window_size for non-overlapping)')
    parser.add_argument('--min-sifted-bits', type=int, default=200, 
                        help='Minimum sifted bits per window for valid QBER')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Random seed for reproducibility')
    parser.add_argument('--alice-p-z', type=float, default=0.5,
                        help='Alice Z basis probability')
    parser.add_argument('--bob-p-z', type=float, default=0.5,
                        help='Bob Z basis probability')
    parser.add_argument('--detection-efficiency', type=float, default=1.0,
                        help='Bob detection efficiency [0, 1]')
    parser.add_argument('--channel-type', type=str, default='ideal',
                        choices=['ideal', 'depolarizing', 'phase_flip', 'amplitude_damping', 'bit_flip'],
                        help='Quantum channel type')
    parser.add_argument('--channel-param', type=float, default=0.0,
                        help='Channel parameter (depolarizing prob, phase flip prob, damping gamma, bit flip prob)')
    parser.add_argument('--loss-prob', type=float, default=0.0,
                        help='Photon loss probability [0, 1]')
    parser.add_argument('--bit-flip-prob', type=float, default=0.0,
                        help='Bit flip probability [0, 1]')
    parser.add_argument('--dark-count-prob', type=float, default=0.0,
                        help='Dark count probability [0, 1]')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--disable-progress', action='store_true',
                        help='Disable progress bars')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for features')
    parser.add_argument('--quiet', action='store_true',
                        help='Disable console logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(__import__('logging'), args.log_level.upper())
    logger = setup_logging(
        log_level=log_level,
        log_dir="results/bb84/logs",
        disable_console=args.quiet
    )
    
    # Setup progress manager
    progress_manager = create_progress_manager(enabled=not args.disable_progress, logger=logger)
    
    logger.info("=" * 60)
    logger.info("TEMPORAL QBER FEATURE EXTRACTION")
    logger.info("=" * 60)
    
    try:
        start_time = time.time()
        
        # Determine channel
        channel = None
        if args.channel_type != 'ideal':
            from src.temporal_qber.simulator.channel import create_channel
            channel_params = {}
            if args.channel_type == 'depolarizing':
                channel_params['depolarizing_prob'] = args.channel_param
            elif args.channel_type == 'phase_flip':
                channel_params['phase_flip_prob'] = args.channel_param
            elif args.channel_type == 'amplitude_damping':
                channel_params['damping_gamma'] = args.channel_param
            elif args.channel_type == 'bit_flip':
                channel_params['bit_flip_prob'] = args.channel_param
            channel_params['loss_probability'] = args.loss_prob
            channel = create_channel(args.channel_type, seed=args.seed, **channel_params)
        
        # Run simulation
        protocol = BB84Protocol(
            n_transmissions=args.n_transmissions,
            seed=args.seed,
            alice_p_z=args.alice_p_z,
            bob_p_z=args.bob_p_z,
            detection_efficiency=args.detection_efficiency,
            dark_count_prob=args.dark_count_prob,
            channel=channel,
        )
        
        def progress_callback(step, progress):
            if progress_manager.enabled:
                logger.info(f"Step: {step} ({progress*100:.0f}%)")
        
        result = protocol.run(progress_callback=progress_callback)
        
        # Extract temporal features
        from src.temporal_qber.temporal import extract_temporal_qber_features, SessionConfig, WindowConfig
        
        session_config = SessionConfig(
            n_transmissions=args.n_transmissions,
            window_size=args.window_size,
            stride=args.stride,
            min_sifted_bits=50,  # Lower for testing with smaller windows
            seed=args.seed,
            alice_p_z=args.alice_p_z,
            bob_p_z=args.bob_p_z,
            detection_efficiency=args.detection_efficiency,
            channel_type=args.channel_type,
            channel_params={},
        )
        
        # Run session to get BB84 result
        session_result = run_session(session_config)
        bb84_result = session_result.bb84_result
        
        # Extract temporal features
        # Note: bob_detected has length = n_transmissions, while sifted keys have different length
        # We need to filter bob_detected to match sifted key length
        bob_detected_filtered = bb84_result.bob_detected[bb84_result.matching_bases_mask & bb84_result.bob_detected]
        
        # Create WindowConfig from SessionConfig
        window_config = WindowConfig(
            window_size=session_config.window_size,
            stride=session_config.stride,
            min_sifted_bits=session_config.min_sifted_bits,
            include_partial_trailing=session_config.include_partial_trailing,
        )
        
        features = extract_temporal_qber_features(
            sifted_alice=bb84_result.sifted_alice_bits,
            sifted_bob=bb84_result.sifted_bob_bits,
            bob_detected=bob_detected_filtered,
            window_config=window_config,
            min_sifted_bits=session_config.min_sifted_bits,
        )
        
        # Combine results
        result = {
            "n_transmissions": bb84_result.n_transmissions,
            "seed": bb84_result.seed,
            "channel_config": bb84_result.channel_config,
            "alice_p_z": bb84_result.alice_p_z,
            "bob_p_z": bb84_result.bob_p_z,
            "detection_efficiency": bb84_result.detection_efficiency,
            "dark_count_prob": bb84_result.dark_count_prob,
            "attack_type": bb84_result.attack_type,
            "attack_params": bb84_result.attack_params,
            "sifted_key_length": bb84_result.sifted_key_length,
            "sifting_efficiency": bb84_result.sifting_efficiency,
            "qber": bb84_result.qber,
            "agreement": bb84_result.agreement,
            "elapsed_time": bb84_result.elapsed_time,
            "total_windows": bb84_result.total_windows if hasattr(bb84_result, 'total_windows') else 0,
            "valid_windows": bb84_result.valid_windows if hasattr(bb84_result, 'valid_windows') else 0,
            **features,
        }
        
        # Print results
        print_results(result, logger)
        
        # Save output if requested
        if args.output:
            save_features_json(result, args.output)
            logger.info(f"Features saved to {args.output}")
        
        elapsed = time.time() - start_time
        logger.info(f"Total elapsed time: {elapsed:.2f}s")
        
    except Exception as e:
        logger.exception(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()