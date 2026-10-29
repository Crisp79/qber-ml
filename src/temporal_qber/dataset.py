"""
Dataset loading and preprocessing for QAD dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from .config import QAD_FEATURE_COLS, QAD_CLASSES, TrainConfig


def load_qad_dataset(balanced: bool = True, n_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load QAD dataset from CSV.
    
    Args:
        balanced: If True, load balanced dataset; else unbalanced
        n_samples: If set, randomly sample this many rows (for quick testing)
    
    Returns:
        DataFrame with features and labels
    """
    fname = "quantum_attack_balanced_1M_dataset.csv" if balanced else "quantum_attack_unbalanced_1M_dataset.csv"
    fpath = Path("data/qad") / fname
    
    if not fpath.exists():
        raise FileNotFoundError(f"Dataset not found at {fpath}")
    
    df = pd.read_csv(fpath)
    
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    
    return df


def prepare_features_targets(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract feature matrix X and target vector y from DataFrame.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names (defaults to QAD_FEATURE_COLS)
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        feature_names: List of feature names in order
    """
    if feature_cols is None:
        feature_cols = QAD_FEATURE_COLS
    
    # Handle circuit_type categorical
    X = df[feature_cols].copy()
    
    # Encode circuit_type if present
    if 'circuit_type' in X.columns:
        X = pd.get_dummies(X, columns=['circuit_type'], prefix='circuit')
    
    feature_names = list(X.columns)
    X = X.values.astype(np.float32)
    y = df['attack_label'].values.astype(np.int64)
    
    return X, y, feature_names


def split_data(X: np.ndarray, y: np.ndarray, config: TrainConfig) -> Tuple:
    """
    Split data into train/test with stratification.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on training data only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_data(scaler: StandardScaler, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform train and test data using fitted scaler."""
    return scaler.transform(X_train), scaler.transform(X_test)


def save_preprocessing(scaler: StandardScaler, feature_names: List[str], path: str):
    """Save scaler and feature names."""
    joblib.dump({'scaler': scaler, 'feature_names': feature_names}, path)


def load_preprocessing(path: str) -> Tuple[StandardScaler, List[str]]:
    """Load scaler and feature names."""
    data = joblib.load(path)
    return data['scaler'], data['feature_names']


def get_class_distribution(y: np.ndarray) -> dict:
    """Get class distribution as dictionary."""
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))


def print_dataset_info(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    """Print dataset summary."""
    print(f"Dataset shape: {df.shape}")
    print(f"Feature matrix: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of classes: {len(np.unique(y))}")
    print("\nClass distribution:")
    dist = get_class_distribution(y)
    for cls, count in sorted(dist.items()):
        print(f"  {QAD_CLASSES[cls] if cls < len(QAD_CLASSES) else cls}: {count:,} ({count/len(y)*100:.1f}%)")
    print(f"\nFeatures: {feature_names[:10]}... ({len(feature_names)} total)")