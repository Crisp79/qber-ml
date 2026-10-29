"""
SHAP explainability and feature ablation for trained models
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from .config import QAD_CLASSES


def explain_model_shap(model: Any, X_train: np.ndarray, X_test: np.ndarray,
                       feature_names: List[str], class_names: List[str],
                       model_name: str, output_dir: str, 
                       max_samples: int = 500) -> Dict[str, Any]:
    """
    Generate SHAP explanations for tree-based models (XGBoost, Random Forest).
    
    Args:
        model: Trained model
        X_train: Training data (for background)
        X_test: Test data (for explanations)
        feature_names: Feature names
        class_names: Class names
        model_name: Name for output files
        output_dir: Directory to save plots
        max_samples: Max samples to explain (for speed)
    
    Returns:
        Dictionary with SHAP values and summary info
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Sample data for speed
    n_explain = min(max_samples, len(X_test))
    X_explain = X_test[:n_explain]
    
    # Create explainer
    print(f"Creating SHAP explainer for {model_name}...")
    
    # For tree models, use TreeExplainer
    if hasattr(model, 'get_booster') or hasattr(model, 'estimators_'):
        explainer = shap.TreeExplainer(model)
    else:
        # For other models, use KernelExplainer (slower)
        # Use a small background sample
        bg_size = min(100, len(X_train))
        background = shap.kmeans(X_train, bg_size)
        explainer = shap.KernelExplainer(model.predict_proba, background)
    
    # Compute SHAP values
    print(f"Computing SHAP values for {n_explain} samples...")
    shap_values = explainer.shap_values(X_explain)
    
    # Handle different SHAP value formats
    # TreeExplainer for multi-class returns (n_samples, n_features, n_classes)
    # KernelExplainer returns list of arrays
    if isinstance(shap_values, list):
        # Shape: [n_classes, n_samples, n_features]
        shap_array = np.array(shap_values)
        # Mean absolute SHAP per class
        mean_abs_shap = np.mean(np.abs(shap_array), axis=1)  # [n_classes, n_features]
        # Global mean absolute SHAP
        global_mean_abs = np.mean(mean_abs_shap, axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Shape: [n_samples, n_features, n_classes]
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0).T  # [n_classes, n_features]
        global_mean_abs = np.mean(mean_abs_shap, axis=0)
    else:
        # Binary or single output
        global_mean_abs = np.mean(np.abs(shap_values), axis=0)
    
    # Ensure it's a 1D numpy array
    global_mean_abs = np.asarray(global_mean_abs).flatten()
    
    # Validate length matches feature_names
    if len(global_mean_abs) != len(feature_names):
        print(f"Warning: SHAP values length ({len(global_mean_abs)}) != feature_names ({len(feature_names)})")
        # Truncate or pad
        min_len = min(len(global_mean_abs), len(feature_names))
        global_mean_abs = global_mean_abs[:min_len]
        feature_names = feature_names[:min_len]
    
    # Plot global summary
    plt.figure(figsize=(10, 8))
    if isinstance(shap_values, list):
        # Multi-class: plot for each class
        n_classes = len(shap_values)
        fig, axes = plt.subplots(1, min(n_classes, 4), figsize=(5*min(n_classes, 4), 8))
        if n_classes == 1:
            axes = [axes]
        for c in range(min(n_classes, 4)):
            shap.summary_plot(shap_values[c], X_explain, feature_names=feature_names,
                            show=False, plot_type='bar', ax=axes[c])
            axes[c].set_title(f'{class_names[c]}')
    else:
        shap.summary_plot(shap_values, X_explain, feature_names=feature_names,
                        show=False, plot_type='bar')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{model_name}_shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Detailed beeswarm plot (for first class or global)
    plt.figure(figsize=(10, 8))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], X_explain, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{model_name}_shap_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP plots saved to {output_dir}")
    
    # Feature importance ranking
    importance_idx = np.argsort(global_mean_abs)[::-1]
    top_features = [(feature_names[i], float(global_mean_abs[i])) for i in importance_idx[:20]]
    
    return {
        'shap_values': shap_values,
        'global_importance': global_mean_abs.tolist(),
        'top_features': top_features,
        'feature_names': feature_names,
        'class_names': class_names
    }


def feature_group_ablation(model: Any, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           feature_names: List[str], feature_groups: Dict[str, List[int]],
                           model_name: str, output_dir: str) -> Dict[str, Any]:
    """
    Leave-one-feature-group-out ablation study.
    
    Args:
        model: Trained model (will be retrained for each ablation)
        X_train, y_train: Training data
        X_test, y_test: Test data
        feature_names: Feature names
        feature_groups: Dict mapping group name to list of feature indices
        model_name: Name for output
        output_dir: Output directory
    
    Returns:
        Ablation results
    """
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, f1_score
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Baseline performance with all features
    baseline_model = clone(model)
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)
    baseline_acc = accuracy_score(y_test, y_pred_base)
    baseline_f1 = f1_score(y_test, y_pred_base, average='macro')
    
    print(f"Baseline {model_name}: Acc={baseline_acc:.4f}, F1={baseline_f1:.4f}")
    
    ablation_results = {}
    
    for group_name, feat_indices in feature_groups.items():
        print(f"Ablating group: {group_name} ({len(feat_indices)} features)...")
        
        # Create mask to exclude this group
        mask = np.ones(len(feature_names), dtype=bool)
        mask[feat_indices] = False
        
        X_train_ablated = X_train[:, mask]
        X_test_ablated = X_test[:, mask]
        
        # Retrain
        ablated_model = clone(model)
        ablated_model.fit(X_train_ablated, y_train)
        y_pred = ablated_model.predict(X_test_ablated)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        drop_acc = baseline_acc - acc
        drop_f1 = baseline_f1 - f1
        
        ablation_results[group_name] = {
            'accuracy': acc,
            'f1_macro': f1,
            'drop_accuracy': drop_acc,
            'drop_f1': drop_f1,
            'n_features_removed': len(feat_indices)
        }
        
        print(f"  Acc={acc:.4f} (-{drop_acc:.4f}), F1={f1:.4f} (-{drop_f1:.4f})")
    
    # Plot ablation results
    plt.figure(figsize=(10, 6))
    groups = list(ablation_results.keys())
    drops_acc = [ablation_results[g]['drop_accuracy'] for g in groups]
    drops_f1 = [ablation_results[g]['drop_f1'] for g in groups]
    
    x = np.arange(len(groups))
    width = 0.35
    plt.bar(x - width/2, drops_acc, width, label='Accuracy Drop', color='coral', alpha=0.8)
    plt.bar(x + width/2, drops_f1, width, label='F1 Drop', color='steelblue', alpha=0.8)
    plt.xticks(x, groups, rotation=45, ha='right')
    plt.ylabel('Performance Drop')
    plt.title(f'Feature Group Ablation - {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{model_name}_ablation.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'baseline_accuracy': baseline_acc,
        'baseline_f1': baseline_f1,
        'ablation_results': ablation_results
    }


def define_qad_feature_groups(feature_names: List[str]) -> Dict[str, List[int]]:
    """
    Define feature groups for QAD dataset based on feature semantics.
    """
    groups = {
        'circuit_structure': [],
        'output_distribution': [],
        'entropy_features': [],
        'statistical_moments': [],
        'purity_diversity': [],
        'attack_specific_params': [],
        'timing': [],
    }
    
    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        if any(kw in name_lower for kw in ['n_qubit', 'depth', 'gate', 'circuit_type']):
            groups['circuit_structure'].append(i)
        elif any(kw in name_lower for kw in ['outcome', 'prob_', 'most_common', 'least_common', 'ratio']):
            groups['output_distribution'].append(i)
        elif any(kw in name_lower for kw in ['entropy', 'normalized_entropy']):
            groups['entropy_features'].append(i)
        elif any(kw in name_lower for kw in ['std', 'max', 'min', 'range', 'cv', 'skew', 'kurt']):
            groups['statistical_moments'].append(i)
        elif any(kw in name_lower for kw in ['purity', 'uniformity', 'coherence', 'asymmetry']):
            groups['purity_diversity'].append(i)
        elif any(kw in name_lower for kw in ['swap', 'insert', 'dp_', 'ad_', 'mt_', 'pl_', 'tr_']):
            groups['attack_specific_params'].append(i)
        elif any(kw in name_lower for kw in ['time', 'exec']):
            groups['timing'].append(i)
        else:
            # Default to circuit structure
            groups['circuit_structure'].append(i)
    
    # Remove empty groups
    return {k: v for k, v in groups.items() if v}


def save_shap_results(shap_results: Dict[str, Any], output_dir: str, model_name: str):
    """Save SHAP results to disk."""
    import json
    
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
        return obj
    
    with open(Path(output_dir) / f"{model_name}_shap_results.json", 'w') as f:
        json.dump(convert(shap_results), f, indent=2)