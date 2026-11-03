"""
Configuration for Temporal-QBER Attack Detection
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

# QAD Dataset classes (quantum circuit attacks, not BB84 QKD attacks)
QAD_CLASSES = [
    "clean",              # 0 - benign
    "swap_injection",     # 1
    "gate_insertion",     # 2
    "depolarizing_noise", # 3
    "amplitude_damping",  # 4
    "measurement_tampering", # 5
    "pulse_level_attack", # 6
    "thermal_relaxation", # 7
]

# Paper's BB84 attack classes (for reference - not in QAD)
BB84_CLASSES = [
    "normal",             # 0 - no eavesdropper
    "intercept_resend",   # 1
    "photon_number_splitting", # 2
    "beam_splitting",     # 3
    "time_shift",         # 4
    "trojan_horse",       # 5
    "quantum_cloning",    # 6
    "fake_state",         # 7
]

# Feature columns in QAD dataset (excluding attack_type, attack_label)
QAD_FEATURE_COLS = [
    'n_qubits', 'circuit_depth', 'n_gates_total', 'n_single_qubit_gates',
    'n_two_qubit_gates', 'gate_density', 'circuit_type', 'n_outcomes',
    'entropy', 'normalized_entropy', 'prob_std', 'prob_max', 'prob_min',
    'prob_range', 'prob_cv', 'purity_proxy', 'most_common_count',
    'least_common_count', 'outcome_ratio', 'exec_time_sec', 'has_noise_model',
    'swap_intensity', 'n_swaps', 'gate_ins_intensity', 'n_insertions',
    'dp_error_rate', 'dp_purity_proxy', 'dp_uniformity_score',
    'ad_damping_param', 'ad_coherence_factor', 'ad_pop_asymmetry', 'ad_pop_ratio',
    'mt_flip_prob', 'mt_uniformity', 'mt_parity_deviation',
    'pl_detuning_mag', 'pl_phase_coherence', 'pl_oscillation_metric',
    'tr_t1', 'tr_t2', 'tr_gate_time', 'tr_decay_ratio', 'tr_coherence_est',
]

# Temporal QBER parameters (from paper)
TEMPORAL_WINDOW_SIZE = 50  # bits per window
MIN_SIFTED_BITS = 200  # minimum bits for valid session

# Training configuration
@dataclass
class TrainConfig:
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    n_jobs: int = -1
    
    # Model hyperparameters
    rf_n_estimators: int = 200
    rf_max_depth: int = 20
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 10
    xgb_learning_rate: float = 0.1
    svm_C: float = 1.0
    svm_gamma: str = 'scale'
    
    # Progress and logging
    show_progress: bool = True
    log_level: str = "INFO"
    experiment_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for experiment tracking."""
        return {
            'test_size': self.test_size,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'rf_n_estimators': self.rf_n_estimators,
            'rf_max_depth': self.rf_max_depth,
            'xgb_n_estimators': self.xgb_n_estimators,
            'xgb_max_depth': self.xgb_max_depth,
            'xgb_learning_rate': self.xgb_learning_rate,
            'svm_C': self.svm_C,
            'svm_gamma': self.svm_gamma,
            'show_progress': self.show_progress,
            'log_level': self.log_level,
        }


@dataclass
class PipelineConfig:
    """Complete pipeline configuration for experiment tracking."""
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: Dict[str, Any] = field(default_factory=dict)
    models: List[str] = field(default_factory=lambda: ['random_forest', 'xgboost', 'svm_rbf'])
    do_shap: bool = True
    do_ablation: bool = True
    models_dir: str = "models"
    results_dir: str = "results"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'train': self.train.to_dict(),
            'dataset': self.dataset,
            'models': self.models,
            'do_shap': self.do_shap,
            'do_ablation': self.do_ablation,
            'models_dir': self.models_dir,
            'results_dir': self.results_dir,
        }

# Paths
DATA_DIR = "data/qad"
MODELS_DIR = "models"
RESULTS_DIR = "results"