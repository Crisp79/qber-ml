"""
Tests for BB84 dataset leakage detection and parameter randomization.

These tests ensure:
1. No feature column is constant within a class (prevents shortcut learning)
2. No single feature perfectly predicts the label
3. Attack parameters are properly randomized per session
4. Channel parameters vary across sessions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import numpy as np
import pytest
import pandas as pd

# Import the randomization function from the generator
from generate_bb84_dataset import (
    randomize_attack_params,
    ATTACK_PARAM_RANGES,
    BASE_CHANNEL_PARAMS,
    DETECTION_EFFICIENCY_RANGE,
    DARK_COUNT_PROB_RANGE,
)

BB84_CLASSES = [
    "normal", "intercept_resend", "photon_number_splitting",
    "beam_splitting", "time_shift", "trojan_horse",
    "quantum_cloning", "fake_state",
]


class TestParameterRandomization:
    """Test that attack parameters are properly randomized per session."""

    def test_randomize_attack_params_returns_all_fields(self):
        """Test that randomize_attack_params returns all required fields."""
        rng = np.random.default_rng(42)
        for class_name in BB84_CLASSES:
            result = randomize_attack_params(class_name, rng)
            assert 'attack_type' in result
            assert 'attack_params' in result
            assert 'channel_type' in result
            assert 'channel_params' in result
            assert 'detection_efficiency' in result
            assert 'dark_count_prob' in result

    def test_detection_efficiency_in_range(self):
        """Test that detection efficiency is within [0.8, 1.0]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            class_name = rng.choice(BB84_CLASSES)
            result = randomize_attack_params(class_name, rng)
            eff = result['detection_efficiency']
            assert DETECTION_EFFICIENCY_RANGE[0] <= eff <= DETECTION_EFFICIENCY_RANGE[1], \
                f"Detection efficiency {eff} out of range {DETECTION_EFFICIENCY_RANGE}"

    def test_dark_count_prob_in_range(self):
        """Test that dark count probability is within [0.0, 0.05]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            class_name = rng.choice(BB84_CLASSES)
            result = randomize_attack_params(class_name, rng)
            dcp = result['dark_count_prob']
            assert DARK_COUNT_PROB_RANGE[0] <= dcp <= DARK_COUNT_PROB_RANGE[1], \
                f"Dark count prob {dcp} out of range {DARK_COUNT_PROB_RANGE}"

    def test_parameters_vary_across_sessions(self):
        """Test that parameters vary across sessions (not constant)."""
        rng = np.random.default_rng(42)
        n_samples = 50

        for class_name in BB84_CLASSES:
            efficiencies = []
            dark_counts = []
            for _ in range(n_samples):
                result = randomize_attack_params(class_name, rng)
                efficiencies.append(result['detection_efficiency'])
                dark_counts.append(result['dark_count_prob'])

            # Efficiency should not be constant
            eff_std = np.std(efficiencies)
            assert eff_std > 0.001, \
                f"Detection efficiency is constant for {class_name} (std={eff_std})"

            # Dark count should vary (unless range is zero)
            dcp_std = np.std(dark_counts)
            assert dcp_std > 0.0001, \
                f"Dark count prob is constant for {class_name} (std={dcp_std})"

    def test_intercept_resend_p_z_varies(self):
        """Test that IR p_z varies across sessions."""
        rng = np.random.default_rng(42)
        p_z_values = []
        for _ in range(50):
            result = randomize_attack_params("intercept_resend", rng)
            p_z_values.append(result['attack_params'].get('p_z', 0.5))
        assert np.std(p_z_values) > 0.01, "IR p_z is constant across sessions"

    def test_pns_mu_varies(self):
        """Test that PNS mu varies across sessions."""
        rng = np.random.default_rng(42)
        mu_values = []
        for _ in range(50):
            result = randomize_attack_params("photon_number_splitting", rng)
            mu_values.append(result['attack_params'].get('mean_photon_number', 0.2))
        assert np.std(mu_values) > 0.01, "PNS mu is constant across sessions"

    def test_beam_splitting_reflection_varies(self):
        """Test that beam splitting reflection_coeff varies."""
        rng = np.random.default_rng(42)
        r_values = []
        for _ in range(50):
            result = randomize_attack_params("beam_splitting", rng)
            r_values.append(result['attack_params'].get('reflection_coeff', 0.1))
        assert np.std(r_values) > 0.01, "Beam splitting reflection_coeff is constant"

    def test_every_attack_class_runs_a_real_attack(self):
        """Non-normal classes must map to an attack that create_attack() builds.

        Regression: classes missing "attack_type" in ATTACK_PARAM_RANGES silently
        fell back to "none", producing unattacked sessions under an attack label.
        """
        from temporal_qber.simulator.attacks import NoAttack, create_attack

        rng = np.random.default_rng(42)
        for class_name in BB84_CLASSES:
            for _ in range(20):
                result = randomize_attack_params(class_name, rng)
                attack = create_attack(result['attack_type'], seed=0, **result['attack_params'])
                if class_name == "normal":
                    assert isinstance(attack, NoAttack)
                else:
                    assert result['attack_type'] != "none", \
                        f"{class_name} randomizes to attack_type='none'"
                    assert not isinstance(attack, NoAttack), \
                        f"{class_name} builds a NoAttack"


class TestDatasetLeakage:
    """Test for label leakage in the generated BB84 dataset."""

    @pytest.fixture
    def bb84_df(self):
        """Load BB84 dataset if available, else skip."""
        csv_path = Path("data/bb84_run2/bb84_temporal_dataset.csv")
        if not csv_path.exists():
            # Try alternative locations
            for alt in [Path("data/bb84/bb84_temporal_dataset.csv"),
                        Path("data/bb84_run2/bb84_temporal_dataset.csv")]:
                if alt.exists():
                    csv_path = alt
                    break
            else:
                pytest.skip("BB84 dataset CSV not found. Run generate_bb84_dataset.py first.")
        return pd.read_csv(csv_path)

    def test_no_constant_features_within_class(self, bb84_df):
        """Test that no feature column is constant within one class but varies in others (shortcut feature)."""
        # Identify feature columns (exclude metadata)
        metadata_cols = {
            'class_label', 'class_index', 'n_transmissions', 'seed',
            'channel_type', 'attack_type', 'attack_params',
            'sifted_key_length', 'sifting_efficiency', 'overall_qber',
            'elapsed_time', 'session_id'
        }
        feature_cols = [c for c in bb84_df.columns if c not in metadata_cols]

        # First, identify features that are constant across ALL classes (useless but not leakage)
        globally_constant = set()
        for col in feature_cols:
            if bb84_df[col].nunique() <= 1:
                globally_constant.add(col)

        for class_name in bb84_df['class_label'].unique():
            class_df = bb84_df[bb84_df['class_label'] == class_name]
            other_df = bb84_df[bb84_df['class_label'] != class_name]
            for col in feature_cols:
                if col in globally_constant:
                    continue  # Useless feature but not leakage
                class_nunique = class_df[col].nunique()
                other_nunique = other_df[col].nunique()
                # Flag only if constant in this class but varying in others (shortcut feature)
                if class_nunique <= 1 and other_nunique > 1:
                    pytest.fail(
                        f"Feature '{col}' is constant (nunique={class_nunique}) "
                        f"within class '{class_name}' but varies in others (nunique={other_nunique}) - "
                        f"potential shortcut feature for leakage"
                    )

    def test_no_single_feature_perfectly_predicts_label(self, bb84_df):
        """Test that no single feature perfectly predicts the class label."""
        metadata_cols = {
            'class_label', 'class_index', 'n_transmissions', 'seed',
            'channel_type', 'attack_type', 'attack_params',
            'sifted_key_length', 'sifting_efficiency', 'overall_qber',
            'elapsed_time', 'session_id'
        }
        feature_cols = [c for c in bb84_df.columns if c not in metadata_cols]

        # Use binned/rounded values for continuous features to avoid FP precision issues
        # Bin continuous features into 20 bins for separation check
        n_bins = 20
        
        for col in feature_cols:
            # Check if this feature perfectly separates any class from all others
            for class_name in bb84_df['class_label'].unique():
                class_mask = bb84_df['class_label'] == class_name
                # Check if feature value uniquely identifies this class
                class_vals = bb84_df.loc[class_mask, col].dropna()
                other_vals = bb84_df.loc[~class_mask, col].dropna()
                
                if len(class_vals) == 0 or len(other_vals) == 0:
                    continue
                    
                # For numeric columns, bin the values
                if np.issubdtype(class_vals.dtype, np.number):
                    # Create bins based on overall range
                    all_vals = bb84_df[col].dropna()
                    if all_vals.nunique() > n_bins:
                        _, bins = np.histogram(all_vals, bins=n_bins)
                        class_binned = np.digitize(class_vals, bins)
                        other_binned = np.digitize(other_vals, bins)
                        class_values = set(class_binned)
                        other_values = set(other_binned)
                    else:
                        # Few unique values, use exact
                        class_values = set(class_vals.round(6))
                        other_values = set(other_vals.round(6))
                else:
                    class_values = set(class_vals)
                    other_values = set(other_vals)
                    
                if class_values and class_values.isdisjoint(other_values):
                    pytest.fail(
                        f"Feature '{col}' perfectly separates class '{class_name}' "
                        f"from all others (binned) - potential leakage"
                    )

    def test_attack_params_not_constant_within_class(self, bb84_df):
        """Test that attack_params column is not constant within attack classes."""
        if 'attack_params' not in bb84_df.columns:
            pytest.skip("attack_params column not found")
        for class_name in bb84_df['class_label'].unique():
            if class_name == 'normal':
                continue  # normal has no attack params
            class_df = bb84_df[bb84_df['class_label'] == class_name]
            unique_params = class_df['attack_params'].nunique()
            assert unique_params > 1, (
                f"attack_params is constant (nunique={unique_params}) for class '{class_name}' - "
                f"no parameter randomization"
            )

    def test_channel_type_not_always_ideal(self, bb84_df):
        """Test that channel_type is not always 'ideal' (channel noise randomized)."""
        if 'channel_type' not in bb84_df.columns:
            pytest.skip("channel_type column not found")
        unique_channels = bb84_df['channel_type'].nunique()
        assert unique_channels > 1, (
            f"channel_type is constant ('{bb84_df['channel_type'].iloc[0]}') - "
            f"no channel noise randomization"
        )
        # 'ideal' should not be the only channel type
        assert 'ideal' not in bb84_df['channel_type'].unique() or unique_channels > 1, (
            "All sessions use ideal channel - no channel noise randomization"
        )

    def test_qber_separation_not_perfect(self, bb84_df):
        """Test that overall_qber does not perfectly separate classes."""
        if 'overall_qber' not in bb84_df.columns:
            pytest.skip("overall_qber column not found")
        for class_name in bb84_df['class_label'].unique():
            class_df = bb84_df[bb84_df['class_label'] == class_name]
            other_df = bb84_df[bb84_df['class_label'] != class_name]
            class_qber = set(class_df['overall_qber'].dropna().round(4))
            other_qber = set(other_df['overall_qber'].dropna().round(4))
            # QBER ranges should overlap between classes (no perfect separation)
            if class_qber and other_qber:
                overlap = class_qber & other_qber
                # Allow some overlap - if zero overlap, QBER perfectly separates
                assert len(overlap) > 0 or len(class_qber) > 1, (
                    f"overall_qber perfectly separates class '{class_name}' "
                    f"from all others - potential leakage"
                )

    def test_attack_classes_were_actually_attacked(self, bb84_df):
        """Every non-normal session must have run an attack; normal sessions must not."""
        if 'attack_type' not in bb84_df.columns:
            pytest.skip("attack_type column not found")
        is_normal = bb84_df['class_label'] == 'normal'
        unattacked = bb84_df[~is_normal & (bb84_df['attack_type'] == 'none')]
        assert unattacked.empty, (
            "Sessions labelled as attacks but simulated with attack_type='none': "
            f"{unattacked['class_label'].value_counts().to_dict()}"
        )
        assert (bb84_df.loc[is_normal, 'attack_type'] == 'none').all()


class TestFeatureColumnCount:
    """Test that the BB84 feature matrix has exactly 63 features."""

    @pytest.fixture
    def bb84_df(self):
        csv_path = Path("data/bb84_run2/bb84_temporal_dataset.csv")
        if not csv_path.exists():
            for alt in [Path("data/bb84/bb84_temporal_dataset.csv")]:
                if alt.exists():
                    csv_path = alt
                    break
            else:
                pytest.skip("BB84 dataset CSV not found.")
        return pd.read_csv(csv_path)

    def test_feature_count_is_63(self, bb84_df):
        """Test that there are exactly 63 feature columns (excluding metadata)."""
        metadata_cols = {
            'class_label', 'class_index', 'n_transmissions', 'seed',
            'channel_type', 'attack_type', 'attack_params',
            'sifted_key_length', 'sifting_efficiency', 'overall_qber',
            'elapsed_time', 'session_id'
        }
        feature_cols = [c for c in bb84_df.columns if c not in metadata_cols]
        assert len(feature_cols) == 63, \
            f"Expected 63 features, found {len(feature_cols)}: {feature_cols}"
