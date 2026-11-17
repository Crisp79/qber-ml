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
import time

from .config import QAD_FEATURE_COLS, QAD_CLASSES, TrainConfig
from .logging_config import get_logger
from .progress import ProgressManager, create_progress_manager


def load_qad_dataset(
    balanced: bool = True, 
    n_samples: Optional[int] = None,
    progress: Optional[ProgressManager] = None
) -> pd.DataFrame:
    """
    Load QAD dataset from CSV.
    
    Args:
        balanced: If True, load balanced dataset; else unbalanced
        n_samples: If set, randomly sample this many rows (for quick testing)
        progress: Optional progress manager for progress reporting
    
    Returns:
        DataFrame with features and labels
    """
    logger = get_logger()
    fname = "quantum_attack_balanced_1M_dataset.csv" if balanced else "quantum_attack_unbalanced_1M_dataset.csv"
    base_dir = Path(__file__).parent.parent.parent
    fpath = base_dir / "data" / "qad" / fname
    
    if not fpath.exists():
        raise FileNotFoundError(f"Dataset not found at {fpath}")
    
    logger.info(f"Loading {'balanced' if balanced else 'unbalanced'} QAD dataset from {fname}")
    start = time.time()
    
    # For small samples, use reservoir sampling or skiprows to avoid loading full dataset
    if n_samples and n_samples < 100000:
        # Use pandas read_csv with skiprows for efficient sampling
        # First, get total rows (fast - just counts lines)
        with open(fpath, 'r') as f:
            total_rows = sum(1 for _ in f) - 1  # subtract header
        
        if n_samples >= total_rows:
            # Need all rows, just load normally
            if progress and progress.enabled:
                with progress.progress(desc="Loading dataset", unit="rows") as pbar:
                    df = pd.read_csv(fpath)
                    pbar.update(len(df))
            else:
                df = pd.read_csv(fpath)
        else:
            # Use random skiprows for efficient sampling without loading full dataset
            # Generate sorted skip indices
            import random
            random.seed(42)
            skip_indices = sorted(random.sample(range(1, total_rows + 1), total_rows - n_samples))
            
            if progress and progress.enabled:
                with progress.progress(desc="Loading dataset", unit="rows", total=n_samples) as pbar:
                    df = pd.read_csv(fpath, skiprows=skip_indices)
                    pbar.update(len(df))
            else:
                df = pd.read_csv(fpath, skiprows=skip_indices)
    else:
        # For large samples or full dataset, use chunked loading
        if progress and progress.enabled:
            with progress.progress(desc="Loading dataset", unit="rows") as pbar:
                chunk_size = 100000
                chunks = []
                total_rows = 0
                for chunk in pd.read_csv(fpath, chunksize=chunk_size):
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    pbar.update(len(chunk))
                    if n_samples and total_rows >= n_samples:
                        break
                df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(fpath)
        
        if n_samples and n_samples < len(df):
            df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    
    elapsed = time.time() - start
    logger.info(f"Dataset loaded: {len(df):,} samples in {elapsed:.1f}s")
    
    return df


def prepare_features_targets(
    df: pd.DataFrame, 
    feature_cols: Optional[List[str]] = None,
    progress: Optional[ProgressManager] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract feature matrix X and target vector y from DataFrame.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names (defaults to QAD_FEATURE_COLS)
        progress: Optional progress manager for progress reporting
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        feature_names: List of feature names in order
    """
    logger = get_logger()
    
    if feature_cols is None:
        feature_cols = QAD_FEATURE_COLS
    
    logger.info("Preparing features and targets...")
    start = time.time()
    
    # Handle circuit_type categorical
    X = df[feature_cols].copy()
    
    if progress and progress.enabled:
        with progress.progress(total=2, desc="Feature prep", unit="step") as pbar:
            # Encode circuit_type if present
            if 'circuit_type' in X.columns:
                X = pd.get_dummies(X, columns=['circuit_type'], prefix='circuit')
            pbar.update(1)
            
            feature_names = list(X.columns)
            X = X.values.astype(np.float32)
            y = df['attack_label'].values.astype(np.int64)
            pbar.update(1)
    else:
        if 'circuit_type' in X.columns:
            X = pd.get_dummies(X, columns=['circuit_type'], prefix='circuit')
        
        feature_names = list(X.columns)
        X = X.values.astype(np.float32)
        y = df['attack_label'].values.astype(np.int64)
    
    elapsed = time.time() - start
    logger.info(f"Features prepared: {X.shape[1]} features, {len(np.unique(y))} classes in {elapsed:.1f}s")
    
    return X, y, feature_names


def split_data(X: np.ndarray, y: np.ndarray, config: TrainConfig) -> Tuple:
    """
    Split data into train/test with stratification.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger = get_logger()
    logger.info(f"Splitting data (test_size={config.test_size}, stratified=True)...")
    
    return train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )


def fit_scaler(X_train: np.ndarray, progress: Optional[ProgressManager] = None) -> StandardScaler:
    """Fit StandardScaler on training data only."""
    logger = get_logger()
    logger.info("Fitting feature scaler...")
    start = time.time()
    
    scaler = StandardScaler()
    
    if progress and progress.enabled:
        with progress.progress(total=1, desc="Fitting scaler", unit="step") as pbar:
            scaler.fit(X_train)
            pbar.update(1)
    else:
        scaler.fit(X_train)
    
    elapsed = time.time() - start
    logger.info(f"Scaler fitted in {elapsed:.1f}s")
    return scaler


def transform_data(
    scaler: StandardScaler, 
    X_train: np.ndarray, 
    X_test: np.ndarray,
    progress: Optional[ProgressManager] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform train and test data using fitted scaler."""
    logger = get_logger()
    logger.info("Transforming features...")
    start = time.time()
    
    if progress and progress.enabled:
        with progress.progress(total=2, desc="Scaling data", unit="split") as pbar:
            X_train_scaled = scaler.transform(X_train)
            pbar.update(1)
            X_test_scaled = scaler.transform(X_test)
            pbar.update(1)
    else:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    elapsed = time.time() - start
    logger.info(f"Features scaled in {elapsed:.1f}s")
    return X_train_scaled, X_test_scaled


def save_preprocessing(scaler: Optional[StandardScaler], feature_names: List[str], path: str):
    """Save scaler (optional) and feature names.
    
    Scaler is now part of each model's Pipeline, so this only saves feature names.
    """
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
    logger = get_logger()
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Number of features: {len(feature_names)}")
    logger.info(f"Number of classes: {len(np.unique(y))}")
    logger.info("Class distribution:")
    dist = get_class_distribution(y)
    for cls, count in sorted(dist.items()):
        cls_name = QAD_CLASSES[cls] if cls < len(QAD_CLASSES) else str(cls)
        logger.info(f"  {cls_name}: {count:,} ({count/len(y)*100:.1f}%)")
    logger.info(f"Features: {feature_names[:10]}... ({len(feature_names)} total)")