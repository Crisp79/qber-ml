#!/usr/bin/env python3
"""
BB84 Simulation CLI

Runs the BB84 quantum key distribution protocol simulation.

Usage:
    uv run python scripts/simulate_bb84.py --config configs/bb84.yaml
    uv run python scripts/simulate_bb84.py --n-transmissions 100000 --seed 42
"""

import argparse
import sys
import time
import json
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
from src.temporal_qber.simulator.session import SessionConfig, run_session
from src.temporal_qber.logging_config import setup_logging, get_logger, log_experiment_config
from src.temporal_qber.progress import create_progress_manager


def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_simulation_from_config(config: dict, logger, progress_manager) -> dict:
    """Run simulation from loaded YAML config."""
    sim = config.get('simulation', {})
    channel_cfg = config.get('channel', {})
    session_cfg = config.get('session', {})
    
    n_transmissions = sim.get('n_transmissions', 100000)
    seed = sim.get('seed')
    alice_p_z = sim.get('protocol', {}).get('basis_probability_z', 0.5)
    bob_p_z = sim.get('protocol', {}).get('basis_probability_z', 0.5)
    detection_efficiency = sim.get('protocol', {}).get('detection_efficiency', 1.0)
    
    channel_type = channel_cfg.get('type', 'ideal')
    channel_params = {}
    if channel_type == 'depolarizing':
        channel_params['depolarizing_prob'] = channel_cfg.get('depolarizing_prob', 0.0)
    elif channel_type == 'phase_flip':
        channel_params['phase_flip_prob'] = channel_cfg.get('phase_flip_prob', 0.0)
    elif channel_type == 'amplitude_damping':
        channel_params['damping_gamma'] = channel_cfg.get('damping_gamma', 0.0)
    
    channel_params['loss_probability'] = channel_cfg.get('loss_probability', 0.0)
    
    if channel_type != 'ideal':
        from src.temporal_qber.simulator.channel import create_channel
        channel = create_channel(channel_type, seed=seed, **channel_params)
    else:
        channel = IdealChannel(seed=seed)
    
    protocol = BB84Protocol(
        n_transmissions=n_transmissions,
        seed=seed,
        alice_p_z=alice_p_z,
        bob_p_z=bob_p_z,
        detection_efficiency=detection_efficiency,
        channel=channel,
    )
    
    def progress_callback(step, progress):
        if progress_manager.enabled:
            logger.info(f"Step: {step} ({progress*100:.0f}%)")
    
    result = protocol.run(progress_callback=progress_callback)
    
    return result


def print_results(result, logger):
    """Print simulation results."""
    logger.info("=" * 60)
    logger.info("BB84 SIMULATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Transmissions:         {result.n_transmissions:,}")
    logger.info(f"Matching bases:        {result.n_matching_bases:,}")
    logger.info(f"Sifted key length:     {result.sifted_key_length:,}")
    logger.info(f"Sifting efficiency:    {result.sifting_efficiency:.4f}")
    logger.info(f"Sifted key errors:     {result.n_sifted_errors:,}")
    logger.info(f"QBER:                  {result.qber:.6f}")
    logger.info(f"Agreement:             {result.agreement:.6f}")
    logger.info(f"Elapsed time:          {result.elapsed_time:.2f}s")
    logger.info(f"Channel:               {result.channel_config['channel_type']}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BB84 Quantum Key Distribution Simulator")
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--n-transmissions', type=int, default=100000, 
                        help='Number of qubit transmissions')
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
                        help='Channel parameter (depolarizing prob, phase flip prob, damping gamma, or bit flip prob)')
    parser.add_argument('--loss-prob', type=float, default=0.0,
                        help='Photon loss probability [0, 1]')
    parser.add_argument('--bit-flip-prob', type=float, default=0.0,
                        help='Bit flip probability [0, 1]')
    parser.add_argument('--dark-count-prob', type=float, default=0.0,
                        help='Dark count probability [0, 1]')
    parser.add_argument('--attack-type', type=str, default='none',
                        choices=['none', 'intercept_resend', 'partial_intercept_resend',
                                 'pns', 'pns_storage', 'beam_splitting', 'beam_splitting_fixed',
                                 'time_shift', 'time_shift_intercept',
                                 'trojan_horse', 'trojan_horse_active',
                                 'quantum_cloning', 'quantum_cloning_measure',
                                 'fake_state'],
                        help='Attack type')
    parser.add_argument('--attack-param', type=float, default=0.0,
                        help='Attack parameter (meaning depends on attack type)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--disable-progress', action='store_true',
                        help='Disable progress bars')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
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
    logger.info("BB84 QUANTUM KEY DISTRIBUTION SIMULATOR")
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
        
        # Determine attack
        attack_params = {}
        if args.attack_type == 'partial_intercept_resend':
            attack_params['interception_prob'] = args.attack_param
        elif args.attack_type in ('pns', 'pns_storage'):
            attack_params['mean_photon_number'] = args.attack_param if args.attack_param > 0 else 0.1
        elif args.attack_type == 'beam_splitting':
            attack_params['reflection_coeff'] = args.attack_param if args.attack_param > 0 else 0.1
        elif args.attack_type == 'beam_splitting_fixed':
            attack_params['reflection_coeff'] = args.attack_param if args.attack_param > 0 else 0.1
        elif args.attack_type == 'time_shift':
            attack_params['time_shift'] = args.attack_param if args.attack_param > 0 else 0.5
        elif args.attack_type == 'time_shift_intercept':
            attack_params['time_shift'] = args.attack_param if args.attack_param > 0 else 0.5
            attack_params['intercept_prob'] = 1.0
        elif args.attack_type in ('trojan_horse', 'trojan_horse_active'):
            attack_params['probe_photons'] = int(args.attack_param) if args.attack_param > 0 else 1000
        elif args.attack_type in ('quantum_cloning', 'quantum_cloning_measure'):
            attack_params['cloning_type'] = 'universal'
        elif args.attack_type == 'fake_state':
            attack_params['attack_subtype'] = 'detector_mismatch'
            attack_params['det0_efficiency'] = 1.0
            attack_params['det1_efficiency'] = 0.8
        
        # Run simulation
        if args.config:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            result = run_simulation_from_config(config, logger, progress_manager)
        else:
            protocol = BB84Protocol(
                n_transmissions=args.n_transmissions,
                seed=args.seed,
                alice_p_z=args.alice_p_z,
                bob_p_z=args.bob_p_z,
                detection_efficiency=args.detection_efficiency,
                dark_count_prob=args.dark_count_prob,
                channel=channel,
                attack=args.attack_type,
                attack_params=attack_params,
            )
            
            def progress_callback(step, progress):
                if progress_manager.enabled:
                    logger.info(f"Step: {step} ({progress*100:.0f}%)")
            
            result = protocol.run(progress_callback=progress_callback)
        
        # Print results
        print_results(result, logger)
        
        # Save output if requested
        if args.output:
            output_data = {
                "public_info": result.get_public_info(),
                "private_info": result.get_private_info(),
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        elapsed = time.time() - start_time
        logger.info(f"Total elapsed time: {elapsed:.2f}s")
        
    except Exception as e:
        logger.exception(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()