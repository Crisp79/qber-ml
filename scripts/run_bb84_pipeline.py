#!/usr/bin/env python3
"""
BB84 Temporal-QBER ML Pipeline

Runs the ML training/evaluation pipeline on the generated BB84 dataset.
Adapts the QAD pipeline logic for BB84's 8 classes and 63 features.
"""

import argparse
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from temporal_qber.train import train_all_models, save_models, TrainConfig
from temporal_qber.evaluate import (
    evaluate_model, plot_confusion_matrix, plot_per_class_f1,
    evaluate_baselines, save_results, print_summary
)
from temporal_qber.explain import explain_model_shap, feature_group_ablation, save_shap_results
from temporal_qber.logging_config import setup_logging, get_logger, log_experiment_config
from temporal_qber.progress import create_progress_manager
from temporal_qber.experiment import ExperimentTracker, ExperimentConfig
from temporal_qber.temporal import MONITORING_FEATURES

# BB84 class definitions (matching generation)
BB84_CLASSES = [
    "normal",             # 0
    "intercept_resend",   # 1
    "photon_number_splitting",  # 2
    "beam_splitting",     # 3
    "time_shift",         # 4
    "trojan_horse",       # 5
    "quantum_cloning",    # 6
    "fake_state",         # 7
]

# Feature columns to exclude from training (metadata)
METADATA_COLS = [
    'class_label', 'class_index', 'n_transmissions', 'seed', 
    'channel_type', 'attack_type', 'attack_params', 
    'sifted_key_length', 'sifting_efficiency', 'overall_qber', 
    'elapsed_time', 'session_id'
]

# BB84 feature groups for ablation (matches feature extraction groups)
BB84_FEATURE_GROUPS = {
    "A_statistical": [
        "qber_mean", "qber_var", "qber_std", "qber_skew", "qber_kurtosis", "qber_iqr", "qber_cv"
    ],
    "B_burst": [
        "spike_density", "burst_count", "burst_mean_duration", "burst_max_duration",
        "burst_total_duration", "jump_energy"
    ],
    "C_temporal_drift": [
        "temporal_drift", "error_gap_mean", "error_gap_std", "error_gap_cv", "error_gap_skew"
    ],
    "D_error_clustering": [
        "consecutive_error_runs", "max_consecutive_errors", "error_clustering_coeff", "instability_index"
    ],
    "E_autocorr_spectral": [
        "lag1_autocorr", "lag2_autocorr", "lag3_autocorr",
        "dft_energy_low", "dft_energy_mid", "dft_energy_high",
        "spectral_entropy", "spectral_centroid", "spectral_bandwidth"
    ],
    "F_entropy_complexity": [
        "shannon_entropy", "permutation_entropy", "sample_entropy",
        "hurst_exponent", "lyapunov_exp", "correlation_dim"
    ],
    "G_basis_dependent": [
        "qber_rectilinear", "qber_diagonal", "qber_basis_diff", "qber_basis_ratio",
        "rect_burst_energy", "diag_burst_energy", "basis_asymmetry",
        "basis_switch_rate", "basis_error_corr", "basis_drift"
    ],
    "H_channel_interaction": [
        "key_gen_rate", "transmission_efficiency", "qber_loss_corr", "qber_loss_interaction",
        "multi_scale_qber_1", "multi_scale_qber_2", "multi_scale_qber_4", 
        "multi_scale_qber_8", "multi_scale_qber_16", "multi_scale_qber_32",
        "loss_burst_corr", "detector_efficiency", "dark_count_rate", 
        "afterpulse_prob", "channel_stability", "qber_trend"
    ],
}


def load_bb84_dataset(csv_path: str, n_samples: int = None, progress=None):
    """Load BB84 dataset from CSV."""
    if progress:
        # Use context manager for stage
        with progress.stage("Loading BB84 dataset"):
            df = pd.read_csv(csv_path)
            
            if n_samples is not None and n_samples < len(df):
                # Stratified sample
                df = df.groupby('class_label').apply(
                    lambda x: x.sample(min(len(x), n_samples // 8), random_state=42)
                ).reset_index(drop=True)
            
            return df
    
    df = pd.read_csv(csv_path)
    
    if n_samples is not None and n_samples < len(df):
        # Stratified sample
        df = df.groupby('class_label').apply(
            lambda x: x.sample(min(len(x), n_samples // 8), random_state=42)
        ).reset_index(drop=True)
    
    return df


def prepare_features_targets(df: pd.DataFrame, progress=None, use_monitoring: bool = False):
    """Extract features and targets from BB84 dataframe.

    By default only the paper's 63 temporal-QBER features are used; with
    use_monitoring=True the detector-monitoring statistics (MONITORING_FEATURES) are added.
    """
    excluded = set(METADATA_COLS) | (set() if use_monitoring else set(MONITORING_FEATURES))
    feature_cols = [c for c in df.columns if c not in excluded]
    X = df[feature_cols].values
    y = df['class_index'].values
    feature_names = feature_cols
    
    return X, y, feature_names


def get_class_distribution(y):
    """Get class distribution as dict."""
    unique, counts = np.unique(y, return_counts=True)
    return {BB84_CLASSES[int(u)]: int(c) for u, c in zip(unique, counts)}


def run_pipeline(
    data_path: str = "data/bb84/bb84_temporal_dataset.csv",
    n_samples: int = None,
    do_shap: bool = True,
    do_ablation: bool = True,
    models_dir: str = "models/bb84",
    results_dir: str = "results/bb84",
    log_level: str = "INFO",
    show_progress: bool = True,
    experiment_name: str = "",
    disable_console_log: bool = False,
    models: list = None,
    monitor_features: bool = False,
):
    """Run the complete ML pipeline on BB84 data."""
    
    # Names must match temporal_qber.config.AVAILABLE_MODELS; "svm" is accepted as an alias
    # for "svm_rbf" (it used to be passed through unchanged and crashed create_models).
    AVAILABLE_MODELS = ['random_forest', 'xgboost', 'svm_rbf']
    if models is None:
        models = AVAILABLE_MODELS.copy()
    else:
        models = ['svm_rbf' if m == 'svm' else m for m in models]
        for m in models:
            if m not in AVAILABLE_MODELS:
                raise ValueError(f"Unknown model: {m}. Available: {AVAILABLE_MODELS} (or 'svm')")
    
    # Setup logging
    experiment_id = experiment_name or f"bb84_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(
        log_level=getattr(__import__('logging'), log_level.upper(), __import__('logging').INFO),
        log_dir=f"{results_dir}/logs",
        experiment_id=experiment_id,
        disable_console=disable_console_log
    )
    
    progress = create_progress_manager(enabled=show_progress, logger=logger)
    tracker = ExperimentTracker(
        experiment_id=experiment_id,
        output_dir=f"{results_dir}/experiments",
        logger=logger
    )
    
    pipeline_start = time.time()
    
    logger.info("="*60)
    logger.info("BB84 TEMPORAL-QBER ATTACK DETECTION PIPELINE")
    logger.info("="*60)
    
    try:
        # 1. Load data
        logger.info("\n[1/7] Loading BB84 dataset...")
        df = load_bb84_dataset(data_path, n_samples, progress)
        
        # 2. Prepare features and targets
        logger.info("\n[2/7] Preparing features...")
        X, y, feature_names = prepare_features_targets(df, progress, use_monitoring=monitor_features)
        
        class_dist = get_class_distribution(y)
        logger.info(f"Features: {len(feature_names)} | Classes: {len(np.unique(y))}")
        logger.info(f"Class distribution: {class_dist}")
        
        # 3. Split data
        logger.info("\n[3/7] Splitting data...")
        config = TrainConfig()
        from temporal_qber.dataset import split_data
        X_train, X_test, y_train, y_test = split_data(X, y, config)
        
        # 4. Save feature names
        logger.info("\n[4/7] Saving feature names...")
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        from temporal_qber.dataset import save_preprocessing
        save_preprocessing(None, feature_names, f"{models_dir}/preprocessing.joblib")
        
        # 5. Train models
        logger.info("\n[5/7] Training models...")
        train_results = train_all_models(
            X_train, y_train, X_test, y_test,
            config, progress=progress, model_names=models
        )
        # Persist trained models so tracker-recorded model paths exist
        save_models(train_results, feature_names, models_dir, class_names=BB84_CLASSES)
        
        # 6. Evaluate models
        logger.info("\n[6/7] Evaluating models...")
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        # Create experiment config manually (avoid QAD_CLASSES mapping issue)
        exp_config = ExperimentConfig(
            experiment_id="",  # Will be set by tracker
            timestamp="",      # Will be set by tracker
            dataset_variant="balanced",
            n_samples=n_samples,
            n_features=len(feature_names),
            class_distribution=class_dist,
            train_test_split={
                'test_size': config.test_size,
                'stratified': True
            },
            cv_config={
                'folds': config.cv_folds,
                'shuffle': True
            },
            random_seeds={
                'data_split': 42,
                'models': 42,
                'cv': 42
            },
            model_hyperparameters=config.to_dict(),
        )
        exp_config.models = models
        tracker.start_experiment(exp_config)
        
        for model_name, result in train_results.items():
            model = result['model']
            tracker.record_model_start(model_name)
            
            eval_result = evaluate_model(model, X_test, y_test, BB84_CLASSES, progress=progress)
            result['evaluation'] = eval_result
            
            # Confusion matrix
            cm_path = f"{results_dir}/{model_name}_confusion_matrix.png"
            plot_confusion_matrix(
                eval_result['confusion_matrix'],
                BB84_CLASSES,
                title=f"Confusion Matrix - {model_name}",
                save_path=cm_path
            )
            plots = [cm_path]
            
            # Per-class F1
            f1_path = f"{results_dir}/{model_name}_per_class_f1.png"
            plot_per_class_f1(
                {model_name: {'evaluation': eval_result}},
                BB84_CLASSES,
                save_path=f1_path
            )
            plots.append(f1_path)
            
            # SHAP explanations (tree models)
            base_model = model.named_steps['model'] if hasattr(model, 'named_steps') else model
            shap_result = None
            if do_shap and model_name in ['random_forest', 'xgboost']:
                logger.info(f"\nGenerating SHAP explanations for {model_name}...")
                shap_result = explain_model_shap(
                    base_model, X_train, X_test,
                    feature_names=feature_names,
                    class_names=BB84_CLASSES,
                    model_name=model_name,
                    output_dir=results_dir,
                    max_samples=500,
                    progress=progress
                )
                if shap_result:
                    shap_path = f"{results_dir}/{model_name}_shap_summary.png"
                    # explain_model_shap already saves plots to output_dir
                    plots.append(shap_path)
                    
                    save_shap_results(shap_result, results_dir, model_name)
            
            # Feature group ablation
            ablation_result = None
            if do_ablation and model_name in ['random_forest', 'xgboost']:
                logger.info(f"\nRunning feature group ablation for {model_name}...")
                # Convert feature group names to indices
                feature_name_to_idx = {name: i for i, name in enumerate(feature_names)}
                ablation_groups = {}
                for group_name, group_features in BB84_FEATURE_GROUPS.items():
                    ablation_groups[group_name] = [
                        feature_name_to_idx[f] for f in group_features if f in feature_name_to_idx
                    ]
                ablation_result = feature_group_ablation(
                    base_model, X_train, y_train, X_test, y_test,
                    feature_names, ablation_groups,
                    model_name, results_dir, progress=progress
                )
            
            result['shap'] = shap_result
            result['ablation'] = ablation_result
            result['plots'] = plots
            
            tracker.record_model_result(
                model_name=model_name,
                train_time=result.get('train_time', 0),
                eval_time=eval_result.get('eval_time', 0),
                test_accuracy=eval_result['accuracy'],
                # Cross-validation runs during training (train_all_models), so its results
                # live in `result`, not in the test-set evaluation (which never has them).
                cv_results={
                    'cv_scores': result.get('cv_scores', []),
                    'cv_f1_scores': result.get('cv_f1_scores', []),
                },
                fold_metrics=result.get('fold_metrics', []),
                aggregate_metrics={
                    'accuracy': result.get('cv_accuracy_mean', 0),
                    'macro_f1': result.get('cv_f1_macro_mean', 0),
                },
                model_path=f"{models_dir}/{model_name}.joblib",
                plots=plots,
                shap_results=shap_result,
                ablation_results=ablation_result,
            )
            all_results[model_name] = result
        
        # 7. Baselines and summary
        logger.info("\n[7/7] Running baselines and generating summary...")
        # Find QBER feature index for baselines
        qber_idx = feature_names.index('qber_mean') if 'qber_mean' in feature_names else None
        baseline_results = evaluate_baselines(X_test, y_test, qber_feature_idx=qber_idx)
        
        # Save all results (strip model objects, keep only serializable metrics from evaluation)
        serializable_results = {}
        for model_name, result in all_results.items():
            eval_result = result.get('evaluation', {})
            serializable_results[model_name] = {
                'test_accuracy': eval_result.get('accuracy'),
                'test_macro_f1': eval_result.get('macro_f1'),
                'test_macro_precision': eval_result.get('macro_precision'),
                'test_macro_recall': eval_result.get('macro_recall'),
                'test_roc_auc': eval_result.get('roc_auc_ovr'),
                'cv_accuracy': result.get('cv_accuracy_mean'),
                'cv_macro_f1': result.get('cv_f1_macro_mean'),
                'cv_std': result.get('cv_accuracy_std'),
                'cv_macro_f1_std': result.get('cv_f1_macro_std'),
                'cv_scores': result.get('cv_scores'),
                'confusion_matrix': eval_result.get('confusion_matrix'),
                'per_class': eval_result.get('per_class'),
                'fnr_binary': eval_result.get('fnr_binary'),
                'fpr_binary': eval_result.get('fpr_binary'),
                'train_time': result.get('train_time'),
                'eval_time': eval_result.get('eval_time'),
            }
        
        combined_results = {
            'models': serializable_results,
            'baselines': baseline_results,
            'classes': BB84_CLASSES,
        }
        save_results(combined_results, results_dir)
        
        elapsed = time.time() - pipeline_start
        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
        logger.info(f"{'='*60}")
        
        # Print summary using evaluation results format
        summary_results = {}
        for model_name, result in all_results.items():
            summary_results[model_name] = {'evaluation': result.get('evaluation', {})}
        print_summary(summary_results)
        
        return all_results, baseline_results
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Run BB84 Temporal-QBER ML Pipeline")
    parser.add_argument("--data-path", "--input-dir", "--in-dir", type=str, default="data/bb84/bb84_temporal_dataset.csv",
                        dest="data_path",
                        help="Path to BB84 dataset CSV (default: data/bb84/bb84_temporal_dataset.csv)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples to use (stratified, default: all)")
    parser.add_argument("--no-shap", action="store_true", help="Disable SHAP explanations")
    parser.add_argument("--no-ablation", action="store_true", help="Disable feature group ablation")
    parser.add_argument("--models", type=str, default="random_forest,xgboost,svm_rbf",
                        help="Comma-separated list of models to train (random_forest, xgboost, svm_rbf; 'svm' = svm_rbf)")
    parser.add_argument("--models-dir", type=str, default="models/bb84", help="Models output directory")
    parser.add_argument("--results-dir", "--output-dir", "--out-dir", type=str, default="results/bb84",
                        dest="results_dir",
                        help="Results output directory (default: results/bb84)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    parser.add_argument("--experiment-name", type=str, default="", help="Experiment name")
    parser.add_argument("--disable-console-log", action="store_true", help="Disable console logging")
    parser.add_argument("--monitor-features", action="store_true",
                        help="Add detector-monitoring features (double_click_rate, empty_gate_click_rate) "
                             "to the paper's 63 temporal-QBER features")

    args = parser.parse_args()
    
    models = [m.strip() for m in args.models.split(",")]
    
    run_pipeline(
        data_path=args.data_path,
        n_samples=args.n_samples,
        do_shap=not args.no_shap,
        do_ablation=not args.no_ablation,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        log_level=args.log_level,
        show_progress=not args.no_progress,
        experiment_name=args.experiment_name,
        disable_console_log=args.disable_console_log,
        models=models,
        monitor_features=args.monitor_features,
    )


if __name__ == "__main__":
    main()