"""
Quick test script to verify pipeline works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from temporal_qber.dataset import load_qad_dataset, prepare_features_targets, split_data, fit_scaler, transform_data
from temporal_qber.train import train_all_models, TrainConfig
from temporal_qber.evaluate import evaluate_model, print_summary

# Use small sample for quick test
print("Loading small sample (1000 samples)...")
df = load_qad_dataset(balanced=True, n_samples=1000)

print("Preparing features...")
X, y, feature_names = prepare_features_targets(df)
print(f"Features: {X.shape}, Labels: {y.shape}")

config = TrainConfig()
X_train, X_test, y_train, y_test = split_data(X, y, config)

scaler = fit_scaler(X_train)
X_train_scaled, X_test_scaled = transform_data(scaler, X_train, X_test)

print("Training models (quick test)...")
results = train_all_models(X_train_scaled, y_train, X_test_scaled, y_test, config)

print("\nEvaluating...")
all_results = {}
for name, result in results.items():
    eval_result = evaluate_model(result['model'], X_test_scaled, y_test)
    result['evaluation'] = eval_result
    all_results[name] = result

print_summary(all_results)
print("\nTest passed!")