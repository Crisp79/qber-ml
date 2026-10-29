"""
Main pipeline runner for Temporal-QBER Attack Detection
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from temporal_qber.dataset import (
    load_qad_dataset, prepare_features_targets, split_data, 
    fit_scaler, transform_data, save_preprocessing, print_dataset_info
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
from temporal_qber.config import QAD_CLASSES


def run_pipeline(
    n_samples: int = None,
    balanced: bool = True,
    do_shap: bool = True,
    do_ablation: bool = True,
    models_dir: str = "models",
    results_dir: str = "results"
):
    """Run the complete ML pipeline."""
    
    print("="*60)
    print("TEMPORAL-QBER ATTACK DETECTION PIPELINE")
    print("="*60)
    
    # 1. Load data
    print("\n[1/6] Loading dataset...")
    df = load_qad_dataset(balanced=balanced, n_samples=n_samples)
    
    # 2. Prepare features and targets
    print("\n[2/6] Preparing features...")
    X, y, feature_names = prepare_features_targets(df)
    print_dataset_info(df, X, y, feature_names)
    
    # 3. Split data
    print("\n[3/6] Splitting data...")
    config = TrainConfig()
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    
    # 4. Scale features
    print("\n[4/6] Scaling features...")
    scaler = fit_scaler(X_train)
    X_train_scaled, X_test_scaled = transform_data(scaler, X_train, X_test)
    save_preprocessing(scaler, feature_names, f"{models_dir}/preprocessing.joblib")
    
    # 5. Train models
    print("\n[5/6] Training models...")
    train_results = train_all_models(X_train_scaled, y_train, X_test_scaled, y_test, config)
    
    # 6. Evaluate models
    print("\n[6/6] Evaluating models...")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_name, result in train_results.items():
        model = result['model']
        
        # Comprehensive evaluation
        eval_result = evaluate_model(model, X_test_scaled, y_test, QAD_CLASSES)
        result['evaluation'] = eval_result
        
        # Confusion matrix plot
        plot_confusion_matrix(
            eval_result['confusion_matrix'], 
            QAD_CLASSES,
            title=f"Confusion Matrix - {model_name}",
            save_path=f"{results_dir}/{model_name}_confusion_matrix.png"
        )
        
        # SHAP explanations (for tree models)
        if do_shap and model_name in ['random_forest', 'xgboost']:
            print(f"\nGenerating SHAP explanations for {model_name}...")
            shap_result = explain_model_shap(
                model, X_train_scaled, X_test_scaled,
                feature_names, QAD_CLASSES,
                model_name, results_dir
            )
            save_shap_results(shap_result, results_dir, model_name)
            result['shap'] = shap_result
        
        # Feature group ablation
        if do_ablation:
            print(f"\nRunning feature ablation for {model_name}...")
            feature_groups = define_qad_feature_groups(feature_names)
            ablation_result = feature_group_ablation(
                model, X_train_scaled, y_train, X_test_scaled, y_test,
                feature_names, feature_groups, model_name, results_dir
            )
            result['ablation'] = ablation_result
        
        all_results[model_name] = result
    
    # Per-class F1 comparison plot
    plot_per_class_f1(all_results, QAD_CLASSES, 
                      save_path=f"{results_dir}/per_class_f1_comparison.png")
    
    # Baseline evaluation (placeholder for QAD)
    print("\nEvaluating baselines...")
    # Note: QAD doesn't have temporal QBER, so baselines are not directly applicable
    baselines = evaluate_baselines(X_test_scaled, y_test)
    print(f"Baselines: {baselines}")
    
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
    
    # Save models
    save_models(train_results, feature_names, scaler, models_dir)
    
    print(f"\nPipeline complete! Results in {results_dir}/, models in {models_dir}/")
    
    return all_results


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
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo(args.models_dir)
    else:
        run_pipeline(
            n_samples=args.n_samples,
            balanced=not args.unbalanced,
            do_shap=not args.no_shap,
            do_ablation=not args.no_ablation,
            models_dir=args.models_dir,
            results_dir=args.results_dir
        )