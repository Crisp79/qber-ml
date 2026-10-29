"""
Model evaluation and baseline comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

from .config import QAD_CLASSES, TrainConfig


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                   class_names: List[str] = None) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Returns:
        Dictionary with all metrics
    """
    if class_names is None:
        class_names = QAD_CLASSES
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    
    # ROC-AUC (OvR)
    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        except:
            roc_auc = None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Per-class results
    class_results = {}
    for i, name in enumerate(class_names):
        class_results[name] = {
            'precision': float(per_class_precision[i]) if i < len(per_class_precision) else 0.0,
            'recall': float(per_class_recall[i]) if i < len(per_class_recall) else 0.0,
            'f1': float(per_class_f1[i]) if i < len(per_class_f1) else 0.0,
            'support': int(np.sum(y_test == i))
        }
    
    # False Negative Rate for binary task (attack vs normal)
    # Normal = class 0, Attacks = classes 1-7
    y_test_binary = (y_test > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    fnr = np.sum((y_test_binary == 1) & (y_pred_binary == 0)) / np.sum(y_test_binary == 1) if np.sum(y_test_binary == 1) > 0 else 0.0
    fpr = np.sum((y_test_binary == 0) & (y_pred_binary == 1)) / np.sum(y_test_binary == 0) if np.sum(y_test_binary == 0) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'roc_auc_ovr': roc_auc,
        'confusion_matrix': cm.tolist(),
        'per_class': class_results,
        'fnr_binary': fnr,
        'fpr_binary': fpr,
        'classification_report': classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                          title: str = "Confusion Matrix", save_path: str = None):
    """Plot and optionally save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    plt.close()


def plot_per_class_f1(results: Dict[str, Dict[str, Any]], class_names: List[str],
                      save_path: str = None):
    """Plot per-class F1 scores for all models."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, result) in enumerate(results.items()):
        ax = axes[idx]
        per_class = result.get('evaluation', {}).get('per_class', {})
        f1_scores = [per_class.get(name, {}).get('f1', 0) for name in class_names]
        
        bars = ax.bar(range(len(class_names)), f1_scores, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('F1 Score')
        ax.set_title(f'{model_name}\nMacro-F1: {result.get("evaluation", {}).get("macro_f1", 0):.3f}')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved per-class F1 plot to {save_path}")
    plt.close()


def evaluate_baselines(X_test: np.ndarray, y_test: np.ndarray, 
                       qber_feature_idx: int = None) -> Dict[str, Any]:
    """
    Evaluate baseline detectors:
    1. Fixed QBER threshold (11%)
    2. CUSUM detector
    
    Note: Requires QBER feature index in feature matrix.
    For QAD dataset, we approximate using available features.
    """
    # For QAD, we don't have direct QBER. Use a proxy feature.
    # Using 'entropy' or 'normalized_entropy' as proxy for anomaly score
    if qber_feature_idx is None:
        # Find entropy column
        # This is a placeholder - in real use, you'd have temporal QBER
        return {
            'fixed_threshold': {'accuracy': 0.0, 'fnr': 1.0, 'note': 'Requires temporal QBER feature'},
            'cusum': {'accuracy': 0.0, 'fnr': 1.0, 'note': 'Requires temporal QBER sequence'}
        }
    
    # Get QBER-like feature
    qber_proxy = X_test[:, qber_feature_idx]
    
    # Fixed threshold (11% = 0.11)
    threshold = 0.11
    y_pred_thresh = (qber_proxy > threshold).astype(int)
    y_true_binary = (y_test > 0).astype(int)
    
    thresh_acc = accuracy_score(y_true_binary, y_pred_thresh)
    thresh_fnr = np.sum((y_true_binary == 1) & (y_pred_thresh == 0)) / np.sum(y_true_binary == 1) if np.sum(y_true_binary == 1) > 0 else 0.0
    
    # CUSUM detector
    cusum_pred = cusum_detector(qber_proxy, threshold=0.11, drift=0.5)
    cusum_acc = accuracy_score(y_true_binary, cusum_pred)
    cusum_fnr = np.sum((y_true_binary == 1) & (cusum_pred == 0)) / np.sum(y_true_binary == 1) if np.sum(y_true_binary == 1) > 0 else 0.0
    
    return {
        'fixed_threshold': {
            'accuracy': thresh_acc,
            'fnr': thresh_fnr,
            'threshold': threshold
        },
        'cusum': {
            'accuracy': cusum_acc,
            'fnr': cusum_fnr,
            'threshold': 0.11,
            'drift': 0.5
        }
    }


def cusum_detector(sequence: np.ndarray, threshold: float = 0.11, 
                   drift: float = 0.5) -> np.ndarray:
    """
    CUSUM sequential change detector.
    
    Args:
        sequence: Input sequence (QBER values)
        threshold: Detection threshold
        drift: Reference drift parameter
    
    Returns:
        Binary predictions (0=normal, 1=attack)
    """
    n = len(sequence)
    predictions = np.zeros(n, dtype=int)
    
    # CUSUM statistics
    g_plus = 0.0
    g_minus = 0.0
    
    for i in range(n):
        # Upper CUSUM (detect increase)
        g_plus = max(0, g_plus + sequence[i] - threshold - drift)
        # Lower CUSUM (detect decrease)
        g_minus = min(0, g_minus + sequence[i] - threshold + drift)
        
        # Alarm if either exceeds threshold
        if g_plus > threshold or abs(g_minus) > threshold:
            predictions[i] = 1
            # Reset after alarm
            g_plus = 0.0
            g_minus = 0.0
    
    return predictions


def save_results(results: Dict[str, Any], output_dir: str):
    """Save evaluation results to JSON."""
    import json
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python
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
    
    with open(Path(output_dir) / "evaluation_results.json", 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"Saved evaluation results to {output_dir}/evaluation_results.json")


def print_summary(results: Dict[str, Dict[str, Any]]):
    """Print evaluation summary table."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Macro-Prec':>10} {'Macro-Rec':>10} {'ROC-AUC':>10} {'Binary-FNR':>10}")
    print("-"*80)
    
    for model_name, result in results.items():
        eval_result = result.get('evaluation', {})
        print(f"{model_name:<20} "
              f"{eval_result.get('accuracy', 0):>10.4f} "
              f"{eval_result.get('macro_f1', 0):>10.4f} "
              f"{eval_result.get('macro_precision', 0):>10.4f} "
              f"{eval_result.get('macro_recall', 0):>10.4f} "
              f"{eval_result.get('roc_auc_ovr', 0):>10.4f} "
              f"{eval_result.get('fnr_binary', 0):>10.4f}")
    print("="*80)