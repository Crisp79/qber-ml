"""
Model training for Temporal-QBER attack detection
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import time

from .config import TrainConfig, QAD_CLASSES, AVAILABLE_MODELS
from .logging_config import get_logger
from .progress import (
    ProgressManager, 
    TrainingProgressTracker, 
    create_xgb_callback,
    track_training_time
)


def compute_balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced sample weights for multiclass classification."""
    return compute_sample_weight(class_weight='balanced', y=y)


def create_models(config: TrainConfig, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create the three classifiers with configured hyperparameters.
    
    Args:
        config: Training configuration
        model_names: Optional list of model names to create. If None, creates all available models.
    
    Returns:
        Dictionary of model instances
    """
    if model_names is None:
        model_names = AVAILABLE_MODELS
    
    # Validate model names
    for name in model_names:
        if name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {name}. Available: {AVAILABLE_MODELS}")
    
    all_models = {
        'random_forest': RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            class_weight='balanced'
        ),
        'xgboost': XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            eval_metric='mlogloss',
            verbosity=0
        ),
        'svm_rbf': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf',
                C=config.svm_C,
                gamma=config.svm_gamma,
                random_state=config.random_state,
                probability=True,
                class_weight='balanced'
            ))
        ])
    }
    
    return {name: all_models[name] for name in model_names}


def train_model(
    model: Any, 
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    model_name: str, 
    sample_weights: Optional[np.ndarray] = None,
    progress: Optional[ProgressManager] = None
) -> Tuple[Any, float]:
    """Train a single model and return training time."""
    logger = get_logger()
    
    start = time.time()
    logger.info(f"Training {model_name}...")
    
    # Train model - no fake progress bars, just log start/end
    if model_name == 'xgboost':
        if sample_weights is not None:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
        
    elapsed = time.time() - start
    logger.info(f"  Completed in {elapsed:.1f}s")
    return model, elapsed


def cross_validate_model(
    model: Any, 
    X: np.ndarray, 
    y: np.ndarray, 
    config: TrainConfig, 
    model_name: str,
    progress: Optional[ProgressManager] = None
) -> Dict[str, Any]:
    """Perform stratified cross-validation with detailed fold reporting."""
    logger = get_logger()
    logger.info(f"Cross-validating {model_name} ({config.cv_folds}-fold)...")
    
    cv = StratifiedKFold(
        n_splits=config.cv_folds, 
        shuffle=True, 
        random_state=config.random_state
    )
    
    # Progress tracker for CV
    tracker = TrainingProgressTracker(
        model_name, 
        config.cv_folds, 
        logger=logger, 
        progress_manager=progress
    )
    
    scores = []
    f1_scores = []
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        fold_num = fold_idx + 1
        tracker.start_fold(fold_num)
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Clone and fit model for this fold
        from sklearn.base import clone
        fold_model = clone(model)
        
        fold_start = time.time()
        fold_error = None
        
        try:
            if model_name == 'xgboost':
                sample_weights_fold = compute_balanced_sample_weights(y_train_fold)
                fold_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
            else:
                fold_model.fit(X_train_fold, y_train_fold)
        except Exception as e:
            fold_error = e
            logger.error(f"  Fold {fold_num} failed: {e}")
            fold_train_time = time.time() - fold_start
            fold_metric = {
                'fold': fold_num,
                'accuracy': 0.0,
                'macro_f1': 0.0,
                'train_time': fold_train_time,
                'eval_time': 0.0,
                'error': str(e),
            }
            fold_metrics.append(fold_metric)
            tracker.end_fold({'accuracy': 0.0, 'macro_f1': 0.0})
            continue
        
        fold_train_time = time.time() - fold_start
        
        # Evaluate
        eval_start = time.time()
        try:
            y_pred = fold_model.predict(X_val_fold)
            acc = accuracy_score(y_val_fold, y_pred)
            f1 = f1_score(y_val_fold, y_pred, average='macro')
        except Exception as e:
            logger.error(f"  Fold {fold_num} evaluation failed: {e}")
            fold_error = e
            acc = 0.0
            f1 = 0.0
        eval_time = time.time() - eval_start
        
        scores.append(acc)
        f1_scores.append(f1)
        
        fold_metric = {
            'fold': fold_num,
            'accuracy': float(acc),
            'macro_f1': float(f1),
            'train_time': fold_train_time,
            'eval_time': eval_time,
        }
        if fold_error:
            fold_metric['error'] = str(fold_error)
        fold_metrics.append(fold_metric)
        
        tracker.end_fold({'accuracy': acc, 'macro_f1': f1})
    
    cv_summary = tracker.finish()
    
    # Compute aggregate metrics only from successful folds
    valid_scores = [s for s in scores if s > 0]
    valid_f1 = [f for f in f1_scores if f > 0]
    
    return {
        'cv_accuracy_mean': float(np.mean(valid_scores)) if valid_scores else 0.0,
        'cv_accuracy_std': float(np.std(valid_scores)) if valid_scores else 0.0,
        'cv_f1_macro_mean': float(np.mean(valid_f1)) if valid_f1 else 0.0,
        'cv_f1_macro_std': float(np.std(valid_f1)) if valid_f1 else 0.0,
        'cv_scores': [float(s) for s in scores],
        'cv_f1_scores': [float(s) for s in f1_scores],
        'fold_metrics': fold_metrics,
        'cv_summary': cv_summary,
    }


def train_all_models(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    config: TrainConfig,
    progress: Optional[ProgressManager] = None,
    model_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Train specified models and evaluate on test set.
    
    Args:
        X_train, y_train, X_test, y_test: Data splits
        config: Training configuration
        progress: Optional progress manager
        model_names: List of model names to train. If None, trains all available models.
    
    Returns:
        Dictionary with model results
    """
    logger = get_logger()
    models = create_models(config, model_names)
    results = {}
    
    # Compute balanced sample weights for XGBoost
    xgb_sample_weights = compute_balanced_sample_weights(y_train)
    
    for name, model in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Model: {name}")
        logger.info(f"{'='*50}")
        
        # Train with sample weights for XGBoost
        if name == 'xgboost':
            model, train_time = train_model(
                model, X_train, y_train, name, 
                xgb_sample_weights, progress
            )
        else:
            model, train_time = train_model(model, X_train, y_train, name, progress=progress)
        
        # Test accuracy
        eval_start = time.time()
        test_acc = model.score(X_test, y_test)
        eval_time = time.time() - eval_start
        
        # Cross-validation
        cv_results = cross_validate_model(model, X_train, y_train, config, name, progress)
        
        results[name] = {
            'model': model,
            'train_time': train_time,
            'eval_time': eval_time,
            'test_accuracy': test_acc,
            **cv_results
        }
        
        logger.info(f"  Test Accuracy: {test_acc:.4f} (eval time: {eval_time:.1f}s)")
        logger.info(f"  CV Accuracy: {cv_results['cv_accuracy_mean']:.4f} (+/- {cv_results['cv_accuracy_std']:.4f})")
        logger.info(f"  CV Macro-F1: {cv_results['cv_f1_macro_mean']:.4f} (+/- {cv_results['cv_f1_macro_std']:.4f})")
        logger.info(f"  Train time: {train_time:.1f}s")
    
    return results


def save_models(results: Dict[str, Dict[str, Any]], feature_names: list, 
                scaler: Any, output_dir: str):
    """Save trained models and preprocessing artifacts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for name, result in results.items():
        model_path = Path(output_dir) / f"{name}.joblib"
        joblib.dump(result['model'], model_path)
    
    # Save feature names and scaler
    joblib.dump({'feature_names': feature_names, 'scaler': scaler}, 
                Path(output_dir) / "preprocessing.joblib")
    
    # Save class names
    joblib.dump(QAD_CLASSES, Path(output_dir) / "class_names.joblib")


def load_model(model_path: str):
    """Load a trained model."""
    return joblib.load(model_path)


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Make predictions."""
    return model.predict(X)


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Get prediction probabilities."""
    return model.predict_proba(X)