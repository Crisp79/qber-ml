"""
Model training for Temporal-QBER attack detection
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import time

from .config import TrainConfig, QAD_CLASSES


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
                model_name: str) -> Tuple[Any, float]:
    """Train a single model and return training time."""
    print(f"Training {model_name}...")
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    return model, elapsed


def cross_validate_model(model: Any, X: np.ndarray, y: np.ndarray, 
                         config: TrainConfig, model_name: str) -> Dict[str, float]:
    """Perform stratified cross-validation."""
    print(f"Cross-validating {model_name}...")
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=config.n_jobs)
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=config.n_jobs)
    
    return {
        'cv_accuracy_mean': np.mean(scores),
        'cv_accuracy_std': np.std(scores),
        'cv_f1_macro_mean': np.mean(f1_scores),
        'cv_f1_macro_std': np.std(f1_scores),
        'cv_scores': scores.tolist(),
        'cv_f1_scores': f1_scores.tolist(),
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
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"{'='*50}")
        
        # Train
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