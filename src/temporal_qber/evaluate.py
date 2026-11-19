"""
Model evaluation and baseline comparisons
"""

# Set matplotlib backend before importing pyplot (for headless environments)
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import time

from .config import QAD_CLASSES, TrainConfig
from .logging_config import get_logger
from .progress import ProgressManager


def evaluate_model(
    model: Any, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    class_names: List[str] = None,
    progress: Optional[ProgressManager] = None
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Returns:
        Dictionary with all metrics
    """
    logger = get_logger()
    
    if class_names is None:
        class_names = QAD_CLASSES
    
    logger.info("Evaluating model on test set...")
    start = time.time()
    
    if progress and progress.enabled:
        with progress.progress(total=3, desc="Evaluation", unit="step") as pbar:
            y_pred = model.predict(X_test)
            pbar.update(1)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            pbar.update(1)
            
            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            pbar.update(1)
    else:
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
    
    elapsed = time.time() - start
    logger.info(f"Evaluation complete in {elapsed:.1f}s")
    roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
    logger.info(f"  Accuracy: {accuracy:.4f} | Macro-F1: {macro_f1:.4f} | ROC-AUC: {roc_auc_str}")
    logger.info(f"  Binary FNR: {fnr:.4f} | FPR: {fpr:.4f}")
    
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
        'eval_time': elapsed,
        'classification_report': classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                          title: str = "Confusion Matrix", save_path: str = None):
    """Plot and optionally save confusion matrix."""
    logger = get_logger()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    plt.close()


def plot_per_class_f1(results: Dict[str, Dict[str, Any]], class_names: List[str],
                      save_path: str = None):
    """Plot per-class F1 scores for all models."""
    logger = get_logger()
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
        logger.info(f"Saved per-class F1 plot to {save_path}")
    plt.close()


def evaluate_baselines(X_test: np.ndarray, y_test: np.ndarray, 
                       qber_feature_idx: int = None,
                       qber_sequences: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
    """
    Evaluate baseline detectors on per-session QBER sequences.
    
    Baselines:
    1. Fixed QBER threshold (11%) - session-level QBER mean
    2. CUSUM detector - per-session windowed QBER sequence
    3. EWMA detector - per-session windowed QBER sequence
    4. Page-Hinkley test - per-session windowed QBER sequence
    
    Args:
        X_test: Feature matrix (n_sessions, n_features)
        y_test: Session labels (n_sessions,)
        qber_feature_idx: Index of session-level QBER feature in X_test (for threshold)
        qber_sequences: List of per-session QBER sequences (one array per session)
    
    Returns:
        Dictionary with per-baseline results
    """
    logger = get_logger()
    logger.info("Evaluating baseline detectors (per-session)...")
    
    y_true_binary = (y_test > 0).astype(int)
    n_sessions = len(y_test)
    
    # ============================================================
    # Baseline 1: Fixed QBER threshold (11%)
    # Applied to session-level QBER mean from feature matrix
    # ============================================================
    threshold = 0.11
    
    if qber_feature_idx is not None and qber_feature_idx < X_test.shape[1]:
        session_qber_mean = X_test[:, qber_feature_idx]
    else:
        # Fallback: use mean of windowed QBER sequences if available
        if qber_sequences is not None:
            session_qber_mean = np.array([np.mean(seq) for seq in qber_sequences])
        else:
            logger.warning("No QBER feature or sequences provided - threshold baseline not applicable")
            session_qber_mean = None
    
    if session_qber_mean is not None:
        y_pred_thresh = (session_qber_mean > threshold).astype(int)
        thresh_acc = accuracy_score(y_true_binary, y_pred_thresh)
        thresh_fnr = _binary_fnr(y_true_binary, y_pred_thresh)
        thresh_fpr = _binary_fpr(y_true_binary, y_pred_thresh)
    else:
        thresh_acc = 0.0
        thresh_fnr = 1.0
        thresh_fpr = 0.0
        y_pred_thresh = np.zeros(n_sessions, dtype=int)
    
    logger.info(f"Fixed threshold (11%): Acc={thresh_acc:.4f}, FNR={thresh_fnr:.4f}, FPR={thresh_fpr:.4f}")
    
    # ============================================================
    # Baseline 2-4: Sequential detectors on per-session QBER sequences
    # ============================================================
    if qber_sequences is None:
        logger.warning("No per-session QBER sequences provided - sequential baselines not applicable")
        seq_results = {
            'cusum': {'accuracy': 0.0, 'fnr': 1.0, 'fpr': 0.0, 'note': 'Requires QBER sequences'},
            'ewma': {'accuracy': 0.0, 'fnr': 1.0, 'fpr': 0.0, 'note': 'Requires QBER sequences'},
            'page_hinkley': {'accuracy': 0.0, 'fnr': 1.0, 'fpr': 0.0, 'note': 'Requires QBER sequences'},
        }
    else:
        # CUSUM detector
        cusum_preds = np.array([
            cusum_detector(seq, threshold=0.11, drift=0.05).max()  # Session-level: attack if any window triggers
            for seq in qber_sequences
        ])
        cusum_acc = accuracy_score(y_true_binary, cusum_preds)
        cusum_fnr = _binary_fnr(y_true_binary, cusum_preds)
        cusum_fpr = _binary_fpr(y_true_binary, cusum_preds)
        logger.info(f"CUSUM: Acc={cusum_acc:.4f}, FNR={cusum_fnr:.4f}, FPR={cusum_fpr:.4f}")
        
        # EWMA detector
        ewma_preds = np.array([
            ewma_detector(seq, threshold=0.11, alpha=0.3).max()
            for seq in qber_sequences
        ])
        ewma_acc = accuracy_score(y_true_binary, ewma_preds)
        ewma_fnr = _binary_fnr(y_true_binary, ewma_preds)
        ewma_fpr = _binary_fpr(y_true_binary, ewma_preds)
        logger.info(f"EWMA: Acc={ewma_acc:.4f}, FNR={ewma_fnr:.4f}, FPR={ewma_fpr:.4f}")
        
        # Page-Hinkley detector
        ph_preds = np.array([
            page_hinkley_detector(seq, threshold=5.0, drift=0.005).max()
            for seq in qber_sequences
        ])
        ph_acc = accuracy_score(y_true_binary, ph_preds)
        ph_fnr = _binary_fnr(y_true_binary, ph_preds)
        ph_fpr = _binary_fpr(y_true_binary, ph_preds)
        logger.info(f"Page-Hinkley: Acc={ph_acc:.4f}, FNR={ph_fnr:.4f}, FPR={ph_fpr:.4f}")
        
        seq_results = {
            'cusum': {
                'accuracy': cusum_acc,
                'fnr': cusum_fnr,
                'fpr': cusum_fpr,
                'threshold': 0.11,
                'drift': 0.05,
            },
            'ewma': {
                'accuracy': ewma_acc,
                'fnr': ewma_fnr,
                'fpr': ewma_fpr,
                'threshold': 0.11,
                'alpha': 0.3,
            },
            'page_hinkley': {
                'accuracy': ph_acc,
                'fnr': ph_fnr,
                'fpr': ph_fpr,
                'threshold': 5.0,
                'drift': 0.005,
            },
        }
    
    # ============================================================
    # Baseline 5: Logistic regression on aggregated features
    # ============================================================
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Use only the QBER-related features (exclude metadata)
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])
        lr_pipeline.fit(X_test, y_test)
        lr_pred = lr_pipeline.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_pred)
        lr_macro_f1 = f1_score(y_test, lr_pred, average='macro', zero_division=0)
        logger.info(f"Logistic regression: Acc={lr_acc:.4f}, Macro-F1={lr_macro_f1:.4f}")
        
        lr_result = {
            'accuracy': lr_acc,
            'macro_f1': lr_macro_f1,
        }
    except ImportError:
        logger.warning("scikit-learn not available - logistic regression baseline skipped")
        lr_result = {'accuracy': 0.0, 'macro_f1': 0.0, 'note': 'scikit-learn not available'}
    
    return {
        'fixed_threshold': {
            'accuracy': thresh_acc,
            'fnr': thresh_fnr,
            'fpr': thresh_fpr,
            'threshold': threshold,
        },
        **seq_results,
        'logistic_regression': lr_result,
    }


def _binary_fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary False Negative Rate (attack=positive)."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return fn / (tp + fn) if (tp + fn) > 0 else 0.0


def _binary_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary False Positive Rate (attack=positive)."""
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def cusum_detector(sequence: np.ndarray, threshold: float = 0.11, 
                   drift: float = 0.05) -> np.ndarray:
    """
    CUSUM sequential change detector.
    
    Applied per-session to windowed QBER sequence. Returns per-window
    predictions; session-level prediction is max across windows.
    
    Args:
        sequence: Input sequence (QBER values per window)
        threshold: Detection threshold
        drift: Reference drift parameter
    
    Returns:
        Binary predictions per window (0=normal, 1=attack)
    """
    n = len(sequence)
    predictions = np.zeros(n, dtype=int)
    
    g_plus = 0.0
    g_minus = 0.0
    
    for i in range(n):
        g_plus = max(0, g_plus + sequence[i] - threshold - drift)
        g_minus = min(0, g_minus + sequence[i] - threshold + drift)
        
        if g_plus > threshold or abs(g_minus) > threshold:
            predictions[i] = 1
            g_plus = 0.0
            g_minus = 0.0
    
    return predictions


def ewma_detector(sequence: np.ndarray, threshold: float = 0.11, 
                  alpha: float = 0.3) -> np.ndarray:
    """
    Exponentially Weighted Moving Average detector.
    
    Flags a window if the EWMA of QBER exceeds threshold.
    
    Args:
        sequence: Input sequence (QBER values per window)
        threshold: Detection threshold
        alpha: Smoothing factor (0-1)
    
    Returns:
        Binary predictions per window
    """
    n = len(sequence)
    predictions = np.zeros(n, dtype=int)
    
    ewma = sequence[0] if n > 0 else 0.0
    for i in range(n):
        ewma = alpha * sequence[i] + (1 - alpha) * ewma
        if ewma > threshold:
            predictions[i] = 1
    
    return predictions


def page_hinkley_detector(sequence: np.ndarray, threshold: float = 5.0, 
                          drift: float = 0.005) -> np.ndarray:
    """
    Page-Hinkley sequential change detector.
    
    Detects changes in the mean of a sequence by monitoring the cumulative
    deviation from the mean.
    
    Args:
        sequence: Input sequence (QBER values per window)
        threshold: Detection threshold
        drift: Drift parameter (allows small mean changes)
    
    Returns:
        Binary predictions per window
    """
    n = len(sequence)
    predictions = np.zeros(n, dtype=int)
    
    if n == 0:
        return predictions
    
    mean_x = sequence[0]
    ph_stat = 0.0
    
    for i in range(n):
        mean_x = mean_x + (sequence[i] - mean_x) / (i + 1)
        ph_stat = ph_stat + (sequence[i] - mean_x - drift)
        
        if ph_stat > threshold:
            predictions[i] = 1
            ph_stat = 0.0  # Reset after alarm
    
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
    
    logger = get_logger()
    logger.info(f"Saved evaluation results to {output_dir}/evaluation_results.json")


def print_summary(results: Dict[str, Dict[str, Any]]):
    """Print evaluation summary table."""
    logger = get_logger()
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Model':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Macro-Prec':>10} {'Macro-Rec':>10} {'ROC-AUC':>10} {'Binary-FNR':>10}")
    logger.info("-"*80)
    
    for model_name, result in results.items():
        eval_result = result.get('evaluation', {})
        logger.info(f"{model_name:<20} "
              f"{eval_result.get('accuracy', 0):>10.4f} "
              f"{eval_result.get('macro_f1', 0):>10.4f} "
              f"{eval_result.get('macro_precision', 0):>10.4f} "
              f"{eval_result.get('macro_recall', 0):>10.4f} "
              f"{eval_result.get('roc_auc_ovr', 0):>10.4f} "
              f"{eval_result.get('fnr_binary', 0):>10.4f}")
    logger.info("="*80)