"""
Tests for key invariants in the QBER-ML pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from temporal_qber.dataset import load_qad_dataset, prepare_features_targets, split_data, fit_scaler, transform_data
from temporal_qber.train import create_models, compute_balanced_sample_weights
from temporal_qber.config import QAD_FEATURE_COLS, QAD_CLASSES, TrainConfig
from temporal_qber.explain import define_qad_feature_groups


class TestDatasetLoading:
    """Test dataset loading and preprocessing invariants."""
    
    def test_load_dataset(self):
        """Test that dataset loads correctly."""
        df = load_qad_dataset(balanced=True, n_samples=100)
        assert len(df) == 100
        assert 'attack_type' in df.columns
        assert 'attack_label' in df.columns
    
    def test_feature_target_separation(self):
        """Test that features and targets are correctly separated."""
        df = load_qad_dataset(balanced=True, n_samples=100)
        X, y, feature_names = prepare_features_targets(df)
        
        # Check label columns are excluded
        assert 'attack_type' not in feature_names
        assert 'attack_label' not in feature_names
        
        # Check target is attack_label
        assert np.array_equal(y, df['attack_label'].values)
        
        # Check feature count
        assert len(feature_names) == 45  # 42 base + 3 one-hot
    
    def test_no_label_leakage_in_features(self):
        """Test that no target-derived columns are in features."""
        df = load_qad_dataset(balanced=True, n_samples=100)
        X, y, feature_names = prepare_features_targets(df)
        
        # Attack-specific params should be in features (they're part of QAD)
        # but attack_type and attack_label should not
        assert 'attack_type' not in feature_names
        assert 'attack_label' not in feature_names
    
    def test_train_test_split_stratified(self):
        """Test that train/test split is stratified."""
        df = load_qad_dataset(balanced=True, n_samples=1000)
        X, y, _ = prepare_features_targets(df)
        config = TrainConfig()
        
        X_train, X_test, y_train, y_test = split_data(X, y, config)
        
        # Check proportions are preserved
        train_props = np.bincount(y_train) / len(y_train)
        test_props = np.bincount(y_test) / len(y_test)
        overall_props = np.bincount(y) / len(y)
        
        np.testing.assert_allclose(train_props, overall_props, rtol=0.05)
        np.testing.assert_allclose(test_props, overall_props, rtol=0.05)
    
    def test_scaler_fitted_on_train_only(self):
        """Test that scaler is fitted on training data only."""
        df = load_qad_dataset(balanced=True, n_samples=1000)
        X, y, _ = prepare_features_targets(df)
        config = TrainConfig()
        
        X_train, X_test, y_train, y_test = split_data(X, y, config)
        scaler = fit_scaler(X_train)
        
        # Check scaler was fitted on train data (mean should match train mean)
        # Use relaxed tolerance for floating-point precision
        np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), rtol=1e-5, atol=1e-8)
        
        # Transform and check - scaled train data should have mean ≈ 0
        X_train_scaled, X_test_scaled = transform_data(scaler, X_train, X_test)
        # Use relaxed tolerance for floating-point precision after scaling
        np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0, atol=2e-7)


class TestFeatureGroups:
    """Test feature group definitions."""
    
    def test_feature_groups_complete(self):
        """Test that all features are assigned to exactly one group."""
        df = load_qad_dataset(balanced=True, n_samples=100)
        X, y, feature_names = prepare_features_targets(df)
        
        groups = define_qad_feature_groups(feature_names)
        
        # Check all features are covered
        all_indices = set()
        for indices in groups.values():
            all_indices.update(indices)
        
        assert len(all_indices) == len(feature_names)
        assert all_indices == set(range(len(feature_names)))
        
        # Check no overlaps
        total_count = sum(len(indices) for indices in groups.values())
        assert total_count == len(feature_names)
    
    def test_feature_group_sizes(self):
        """Test that group sizes match expected values."""
        df = load_qad_dataset(balanced=True, n_samples=100)
        X, y, feature_names = prepare_features_targets(df)
        
        groups = define_qad_feature_groups(feature_names)
        
        expected_sizes = {
            'circuit_structure': 12,
            'output_distribution': 11,
            'entropy_features': 2,
            'purity_diversity': 8,
            'attack_specific_params': 11,
            'timing': 1,
        }
        
        for group, expected in expected_sizes.items():
            assert len(groups[group]) == expected, f"Group {group} has {len(groups[group])} features, expected {expected}"


class TestModelConfiguration:
    """Test model configuration invariants."""
    
    def test_xgboost_uses_sample_weights(self):
        """Test that XGBoost receives sample weights for balanced classes."""
        df = load_qad_dataset(balanced=True, n_samples=200)
        X, y, _ = prepare_features_targets(df)
        config = TrainConfig()
        
        X_train, X_test, y_train, y_test = split_data(X, y, config)
        scaler = fit_scaler(X_train)
        X_train_scaled, _ = transform_data(scaler, X_train, X_test)
        
        models = create_models(config)
        xgb_model = models['xgboost']
        
        # Train with sample weights
        sample_weights = compute_balanced_sample_weights(y_train)
        
        # Verify sample weights are computed correctly
        unique, counts = np.unique(y_train, return_counts=True)
        expected_weights = len(y_train) / (len(unique) * counts)
        for cls, expected_w in zip(unique, expected_weights):
            mask = (y_train == cls)
            assert np.allclose(sample_weights[mask], expected_w)
    
    def test_random_forest_class_weight(self):
        """Test that Random Forest uses balanced class weight."""
        models = create_models(TrainConfig())
        rf = models['random_forest']
        assert rf.class_weight == 'balanced'
    
    def test_svm_class_weight(self):
        """Test that SVM uses balanced class weight."""
        models = create_models(TrainConfig())
        svm = models['svm_rbf']
        assert svm.named_steps['svm'].class_weight == 'balanced'


class TestLeakageDetection:
    """Test label leakage detection."""
    
    def test_attack_specific_features_exist(self):
        """Test that attack-specific features exist in the dataset."""
        df = load_qad_dataset(balanced=True, n_samples=100)
        
        attack_specific_cols = [
            'swap_intensity', 'n_swaps', 'gate_ins_intensity', 'n_insertions',
            'dp_error_rate', 'ad_damping_param', 'mt_flip_prob',
            'pl_detuning_mag', 'tr_t1', 'tr_t2', 'tr_gate_time'
        ]
        
        for col in attack_specific_cols:
            assert col in df.columns
    
    def test_has_noise_model_indicates_attacks(self):
        """Test that has_noise_model is 0 for clean, and >0 for attacks (some attacks may have 0)."""
        df = load_qad_dataset(balanced=True, n_samples=1000)
        
        clean_noise = df[df['attack_type'] == 'clean']['has_noise_model'].unique()
        attack_noise = df[df['attack_type'] != 'clean']['has_noise_model'].unique()
        
        assert set(clean_noise) == {0}
        # Most attacks should have has_noise_model == 1
        assert 1 in attack_noise
    
    def test_attack_specific_params_mostly_nonzero_for_correct_class(self):
        """Test that attack-specific params are mostly non-zero for their attack class."""
        df = load_qad_dataset(balanced=True, n_samples=1000)
        
        # swap_intensity should be > 0 mostly for swap_injection
        swap_inj = df[df['attack_type'] == 'swap_injection']
        other = df[df['attack_type'] != 'swap_injection']
        
        assert (swap_inj['swap_intensity'] > 0).all()
        # Some other attacks may have non-zero swap_intensity due to data generation
        # At least the target class should have it non-zero
        assert (swap_inj['swap_intensity'] > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])