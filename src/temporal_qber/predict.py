"""
Prediction on new samples
"""

import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import QAD_CLASSES


def load_artifacts(models_dir: str) -> Dict[str, Any]:
    """Load all saved models and preprocessing artifacts."""
    artifacts = {}
    
    # Load preprocessing
    prep_path = Path(models_dir) / "preprocessing.joblib"
    if prep_path.exists():
        prep = joblib.load(prep_path)
        artifacts['scaler'] = prep['scaler']
        artifacts['feature_names'] = prep['feature_names']
    
    # Load class names
    class_path = Path(models_dir) / "class_names.joblib"
    if class_path.exists():
        artifacts['class_names'] = joblib.load(class_path)
    else:
        artifacts['class_names'] = QAD_CLASSES
    
    # Load models
    model_files = list(Path(models_dir).glob("*.joblib"))
    for f in model_files:
        if f.name not in ['preprocessing.joblib', 'class_names.joblib']:
            model_name = f.stem
            artifacts[model_name] = joblib.load(f)
    
    return artifacts


def preprocess_sample(sample: np.ndarray, scaler: Any, 
                      feature_names: List[str] = None) -> np.ndarray:
    """Preprocess a single sample or batch."""
    # Ensure 2D
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    
    # Apply scaling
    return scaler.transform(sample)


def predict_single(model: Any, sample: np.ndarray, 
                   class_names: List[str]) -> Dict[str, Any]:
    """
    Predict class for a single sample.
    
    Returns:
        Dictionary with prediction, confidence, and all class probabilities
    """
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    
    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0] if hasattr(model, 'predict_proba') else None
    
    result = {
        'predicted_class': int(pred),
        'predicted_label': class_names[pred] if pred < len(class_names) else str(pred),
        'confidence': float(np.max(proba)) if proba is not None else 1.0,
    }
    
    if proba is not None:
        result['all_probabilities'] = {
            class_names[i] if i < len(class_names) else f'class_{i}': float(p)
            for i, p in enumerate(proba)
        }
    
    return result


def predict_batch(model: Any, X: np.ndarray, 
                  class_names: List[str]) -> List[Dict[str, Any]]:
    """Predict classes for a batch of samples."""
    preds = model.predict(X)
    probas = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
    
    results = []
    for i, pred in enumerate(preds):
        result = {
            'predicted_class': int(pred),
            'predicted_label': class_names[pred] if pred < len(class_names) else str(pred),
        }
        if probas is not None:
            result['confidence'] = float(np.max(probas[i]))
            result['all_probabilities'] = {
                class_names[j] if j < len(class_names) else f'class_{j}': float(p)
                for j, p in enumerate(probas[i])
            }
        results.append(result)
    
    return results


def demo_prediction(models_dir: str = "models", n_samples: int = 5):
    """Run demo predictions on test samples."""
    from .dataset import load_qad_dataset, prepare_features_targets, split_data
    from .config import TrainConfig
    
    print("Loading test data...")
    df = load_qad_dataset(balanced=True, n_samples=1000)
    X, y, feature_names = prepare_features_targets(df)
    
    config = TrainConfig()
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    
    # Load artifacts
    artifacts = load_artifacts(models_dir)
    scaler = artifacts['scaler']
    class_names = artifacts['class_names']
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Demo with each model
    for model_name in ['random_forest', 'xgboost', 'svm_rbf']:
        if model_name not in artifacts:
            print(f"Model {model_name} not found, skipping...")
            continue
        
        model = artifacts[model_name]
        print(f"\n--- {model_name} Demo ---")
        
        for i in range(min(n_samples, len(X_test))):
            sample = X_test_scaled[i:i+1]
            true_label = class_names[y_test[i]]
            pred = predict_single(model, sample, class_names)
            print(f"  Sample {i}: True={true_label}, Pred={pred['predicted_label']}, Conf={pred['confidence']:.3f}")


if __name__ == "__main__":
    demo_prediction()