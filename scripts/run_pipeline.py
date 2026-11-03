"""
Main pipeline runner for Temporal-QBER Attack Detection
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from temporal_qber.dataset import (
    load_qad_dataset, prepare_features_targets, split_data, 
    fit_scaler, transform_data, save_preprocessing, get_class_distribution
)
from temporal_qber.train import train_all_models, save_models, TrainConfig
from temporal_qber.evaluate import (
    evaluate_model, plot_confusion_matrix, plot_per_class_f1, 
    evaluate_baselines, save_results, print_summary
)
from temporal_qber.explain import (
    explain_model_shap, feature_group_ablation, define_qad_feature_groups, 
    save_shap_results
)
from temporal_qber.config import QAD_CLASSES, PipelineConfig
from temporal_qber.logging_config import setup_logging, get_logger, log_experiment_config
from temporal_qber.progress import ProgressManager, create_progress_manager
from temporal_qber.experiment import ExperimentTracker, ExperimentConfig, create_experiment_config_from_args


def run_pipeline(
    n_samples: int = None,
    balanced: bool = True,
    do_shap: bool = True,
    do_ablation: bool = True,
    models_dir: str = "models",
    results_dir: str = "results",
    log_level: str = "INFO",
    show_progress: bool = True,
    experiment_name: str = "",
    disable_console_log: bool = False,
):
    """Run the complete ML pipeline with logging, progress tracking, and experiment tracking."""
    
    # Setup logging
    experiment_id = experiment_name or time.strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(
        log_level=getattr(__import__('logging'), log_level.upper(), __import__('logging').INFO),
        log_dir=f"{results_dir}/logs",
        experiment_id=experiment_id,
        disable_console=disable_console_log
    )
    
    # Setup progress manager
    progress = create_progress_manager(enabled=show_progress, logger=logger)
    
    # Setup experiment tracker
    tracker = ExperimentTracker(
        experiment_id=experiment_id,
        output_dir=f"{results_dir}/experiments",
        logger=logger
    )
    
    pipeline_start = time.time()
    
    logger.info("="*60)
    logger.info("TEMPORAL-QBER ATTACK DETECTION PIPELINE")
    logger.info("="*60)
    
    try:
        # 1. Load data
        logger.info("\n[1/7] Loading dataset...")
        df = load_qad_dataset(balanced=balanced, n_samples=n_samples, progress=progress)
        
        # 2. Prepare features and targets
        logger.info("\n[2/7] Preparing features...")
        X, y, feature_names = prepare_features_targets(df, progress=progress)
        
        class_dist = get_class_distribution(y)
        logger.info(f"Features: {len(feature_names)} | Classes: {len(np.unique(y))}")
        logger.info(f"Class distribution: {class_dist}")
        
        # 3. Split data
        logger.info("\n[3/7] Splitting data...")
        config = TrainConfig()
        X_train, X_test, y_train, y_test = split_data(X, y, config)
        
        # 4. Scale features
        logger.info("\n[4/7] Scaling features...")
        scaler = fit_scaler(X_train, progress=progress)
        X_train_scaled, X_test_scaled = transform_data(scaler, X_train, X_test, progress=progress)
        save_preprocessing(scaler, feature_names, f"{models_dir}/preprocessing.joblib")
        
        # 5. Train models
        logger.info("\n[5/7] Training models...")
        train_results = train_all_models(
            X_train_scaled, y_train, X_test_scaled, y_test, 
            config, progress=progress
        )
        
        # 6. Evaluate models
        logger.info("\n[6/7] Evaluating models...")
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        # Create experiment config for tracking
        exp_config = create_experiment_config_from_args(
            type('Args', (), {
                'balanced': balanced,
                'n_samples': n_samples,
                'test_size': config.test_size,
                'cv_folds': config.cv_folds,
            })(),
            n_features=len(feature_names),
            class_dist=class_dist
        )
        exp_config.model_hyperparameters = config.to_dict()
        tracker.start_experiment(exp_config)
        
        for model_name, result in train_results.items():
            model = result['model']
            
            tracker.record_model_start(model_name)
            
            # Comprehensive evaluation
            eval_result = evaluate_model(model, X_test_scaled, y_test, QAD_CLASSES, progress=progress)
            result['evaluation'] = eval_result
            
            # Confusion matrix plot
            cm_path = f"{results_dir}/{model_name}_confusion_matrix.png"
            plot_confusion_matrix(
                eval_result['confusion_matrix'], 
                QAD_CLASSES,
                title=f"Confusion Matrix - {model_name}",
                save_path=cm_path
            )
            
            plots = [cm_path]
            
            # SHAP explanations (for tree models)
            shap_result = None
            if do_shap and model_name in ['random_forest', 'xgboost']:
                logger.info(f"\nGenerating SHAP explanations for {model_name}...")
                shap_result = explain_model_shap(
                    model, X_train_scaled, X_test_scaled,
                    feature_names, QAD_CLASSES,
                    model_name, results_dir,
                    progress=progress
                )
                save_shap_results(shap_result, results_dir, model_name)
                result['shap'] = shap_result
                plots.append(f"{results_dir}/{model_name}_shap_summary.png")
                plots.append(f"{results_dir}/{model_name}_shap_beeswarm.png")
            
            # Feature group ablation
            ablation_result = None
            if do_ablation:
                logger.info(f"\nRunning feature ablation for {model_name}...")
                feature_groups = define_qad_feature_groups(feature_names)
                ablation_result = feature_group_ablation(
                    model, X_train_scaled, y_train, X_test_scaled, y_test,
                    feature_names, feature_groups, model_name, results_dir,
                    progress=progress
                )
                result['ablation'] = ablation_result
                plots.append(f"{results_dir}/{model_name}_ablation.png")
            
            # Prepare model hyperparameters for tracking
            model_hparams = {}
            if hasattr(model, 'get_params'):
                model_hparams = model.get_params()
            
            # Extract fold metrics from CV results
            fold_metrics = result.get('fold_metrics', [])
            cv_results = {
                'cv_accuracy_mean': result.get('cv_accuracy_mean', 0),
                'cv_accuracy_std': result.get('cv_accuracy_std', 0),
                'cv_f1_macro_mean': result.get('cv_f1_macro_mean', 0),
                'cv_f1_macro_std': result.get('cv_f1_macro_std', 0),
                'cv_scores': result.get('cv_scores', []),
                'cv_f1_scores': result.get('cv_f1_scores', []),
            }
            
            aggregate_metrics = {
                'test_accuracy': result.get('test_accuracy', 0),
                'macro_f1': eval_result.get('macro_f1', 0),
                'macro_precision': eval_result.get('macro_precision', 0),
                'macro_recall': eval_result.get('macro_recall', 0),
                'roc_auc_ovr': eval_result.get('roc_auc_ovr', 0),
                'fnr_binary': eval_result.get('fnr_binary', 0),
                'fpr_binary': eval_result.get('fpr_binary', 0),
            }
            
            model_path = f"{models_dir}/{model_name}.joblib"
            
            tracker.record_model_result(
                model_name=model_name,
                train_time=result.get('train_time', 0),
                eval_time=result.get('eval_time', 0),
                test_accuracy=result.get('test_accuracy', 0),
                cv_results=cv_results,
                fold_metrics=fold_metrics,
                aggregate_metrics=aggregate_metrics,
                model_path=model_path,
                plots=plots,
                shap_results=shap_result,
                ablation_results=ablation_result,
                status="completed"
            )
            
            all_results[model_name] = result
        
        # Per-class F1 comparison plot
        f1_comp_path = f"{results_dir}/per_class_f1_comparison.png"
        plot_per_class_f1(all_results, QAD_CLASSES, save_path=f1_comp_path)
        
        # Baseline evaluation (placeholder for QAD)
        logger.info("\nEvaluating baselines...")
        baselines = evaluate_baselines(X_test_scaled, y_test)
        logger.info(f"Baselines: {baselines}")
        tracker.record_baselines(baselines)
        
        # Save all results
        save_results({
            'models': {name: {k: v for k, v in res.items() if k != 'model'} for name, res in all_results.items()},
            'baselines': baselines,
            'dataset_info': {
                'n_samples': len(df),
                'n_features': len(feature_names),
                'feature_names': feature_names,
                'class_names': QAD_CLASSES,
                'class_distribution': {QAD_CLASSES[i]: int(np.sum(y == i)) for i in range(len(QAD_CLASSES))}
            }
        }, results_dir)
        
        # Print summary
        print_summary(all_results)
        
        # 7. Save models
        logger.info("\n[7/7] Saving models...")
        save_models(train_results, feature_names, scaler, models_dir)
        
        total_duration = time.time() - pipeline_start
        logger.info(f"\nPipeline complete in {total_duration:.1f}s!")
        logger.info(f"Results in {results_dir}/, models in {models_dir}/")
        logger.info(f"Experiment ID: {experiment_id}")
        
        tracker.finalize(status="completed")
        
        return all_results
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        tracker.finalize(status="failed", error=str(e))
        raise


def run_demo(models_dir: str = "models"):
    """Run prediction demo."""
    from temporal_qber.predict import demo_prediction
    demo_prediction(models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal-QBER Attack Detection Pipeline")
    parser.add_argument('--n-samples', type=int, default=None, help='Number of samples (for quick testing)')
    parser.add_argument('--unbalanced', action='store_true', help='Use unbalanced dataset')
    parser.add_argument('--no-shap', action='store_true', help='Skip SHAP explanations')
    parser.add_argument('--no-ablation', action='store_true', help='Skip feature ablation')
    parser.add_argument('--models-dir', default='models', help='Models directory')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--demo', action='store_true', help='Run prediction demo')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--disable-progress', action='store_true', help='Disable progress bars')
    parser.add_argument('--experiment-name', default='', help='Custom experiment name/ID')
    parser.add_argument('--quiet', action='store_true', help='Disable console logging (file only)')
    parser.add_argument('--verbose', action='store_true', help='Enable DEBUG logging')
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo(args.models_dir)
    else:
        log_level = 'DEBUG' if args.verbose else args.log_level
        run_pipeline(
            n_samples=args.n_samples,
            balanced=not args.unbalanced,
            do_shap=not args.no_shap,
            do_ablation=not args.no_ablation,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
            log_level=log_level,
            show_progress=not args.disable_progress,
            experiment_name=args.experiment_name,
            disable_console_log=args.quiet
        )