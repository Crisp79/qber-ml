"""
Quick test script to verify pipeline works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from temporal_qber.dataset import load_qad_dataset, prepare_features_targets, split_data, fit_scaler, transform_data
from temporal_qber.train import train_all_models, TrainConfig
from temporal_qber.evaluate import evaluate_model, print_summary
from temporal_qber.logging_config import setup_logging
from temporal_qber.progress import create_progress_manager
import logging

# Setup logging for test (console output enabled, INFO level)
logger = setup_logging(
    log_level=logging.INFO,
    log_dir="results/logs",
    experiment_id="test_pipeline",
    disable_console=False
)

# Setup progress manager (disabled for tests)
progress = create_progress_manager(enabled=False, logger=logger)

# Use small sample for quick test
logger.info("Loading small sample (1000 samples)...")
df = load_qad_dataset(balanced=True, n_samples=1000, progress=progress)

logger.info("Preparing features...")
X, y, feature_names = prepare_features_targets(df, progress=progress)
logger.info(f"Features: {X.shape}, Labels: {y.shape}")

config = TrainConfig()
X_train, X_test, y_train, y_test = split_data(X, y, config)

scaler = fit_scaler(X_train, progress=progress)
X_train_scaled, X_test_scaled = transform_data(scaler, X_train, X_test, progress=progress)

logger.info("Training models (quick test)...")
results = train_all_models(X_train_scaled, y_train, X_test_scaled, y_test, config, progress=progress)

logger.info("\nEvaluating...")
all_results = {}
for name, result in results.items():
    eval_result = evaluate_model(result['model'], X_test_scaled, y_test, progress=progress)
    result['evaluation'] = eval_result
    all_results[name] = result

print_summary(all_results)
logger.info("\nTest passed!")