"""
Model training for Temporal-QBER attack detection
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import time

from .config import TrainConfig, QAD_CLASSES


def compute_balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced sample weights for multiclass classification."""
    return compute_sample_weight(class_weight='balanced', y=y)


def create_models(config: TrainConfig) -> Dict[str, Any]:
    """Create the three classifiers with configured hyperparameters."""
    
    models = {
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
    
    return models


def train_model(model: Any, X_train: np.ndarray, y_train: np.ndarray, 
                model_name: str, sample_weights: Optional[np.ndarray] = None) -> Tuple[Any, float]:
    """Train a single model and return training time."""
    print(f"Training {model_name}...")
    start = time.time()
    
    # XGBoost supports sample_weight parameter
    if model_name == 'xgboost' and sample_weights is not None:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
        
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    return model, elapsed


def cross_validate_model(model: Any, X: np.ndarray, y: np.ndarray, 
                         config: TrainConfig, model_name: str) -> Dict[str, float]:
    """Perform stratified cross-validation."""
    print(f"Cross-validating {model_name}...")
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    
    scores = []
    f1_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Clone and fit model for this fold
        from sklearn.base import clone
        fold_model = clone(model)
        
        if model_name == 'xgboost':
            # Compute sample weights for this fold
            sample_weights_fold = compute_balanced_sample_weights(y_train_fold)
            fold_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
        else:
            fold_model.fit(X_train_fold, y_train_fold)
        
        # Evaluate
        y_pred = fold_model.predict(X_val_fold)
        from sklearn.metrics import accuracy_score, f1_score
        scores.append(accuracy_score(y_val_fold, y_pred))
        f1_scores.append(f1_score(y_val_fold, y_pred, average='macro'))
    
    return {
        'cv_accuracy_mean': np.mean(scores),
        'cv_accuracy_std': np.std(scores),
        'cv_f1_macro_mean': np.mean(f1_scores),
        'cv_f1_macro_std': np.std(f1_scores),
        'cv_scores': [float(s) for s in scores],
        'cv_f1_scores': [float(s) for s in f1_scores],
    }


def train_all_models(X_train: np.ndarray, y_train: np.ndarray, 
                     X_test: np.ndarray, y_test: np.ndarray,
                     config: TrainConfig) -> Dict[str, Dict[str, Any]]:
    """
    Train all three models and evaluate on test set.
    
    Returns:
        Dictionary with model results
    """
    models = create_models(config)
    results = {}
    
    # Compute balanced sample weights for XGBoost
    xgb_sample_weights = compute_balanced_sample_weights(y_train)
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"{'='*50}")
        
        # Train with sample weights for XGBoost
        if name == 'xgboost':
            model, train_time = train_model(model, X_train, y_train, name, xgb_sample_weights)
        else:
            model, train_time = train_model(model, X_train, y_train, name)
        
        # Test accuracy
        test_acc = model.score(X_test, y_test)
        
        # Cross-validation
        cv_results = cross_validate_model(model, X_train, y_train, config, name)
        
        results[name] = {
            'model': model,
            'train_time': train_time,
            'test_accuracy': test_acc,
            **cv_results
        }
        
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  CV Accuracy: {cv_results['cv_accuracy_mean']:.4f} (+/- {cv_results['cv_accuracy_std']:.4f})")
        print(f"  CV Macro-F1: {cv_results['cv_f1_macro_mean']:.4f} (+/- {cv_results['cv_f1_macro_std']:.4f})")
    
    return results


def save_models(results: Dict[str, Dict[str, Any]], feature_names: list, 
                scaler: Any, output_dir: str):
    """Save trained models and preprocessing artifacts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for name, result in results.items():
        model_path = Path(output_dir) / f"{name}.joblib"
        joblib.dump(result['model'], model_path)
        print(f"Saved {name} to {model_path}")
    
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