"""
Temporal QBER feature extraction for BB84 sessions.

Extracts numerical feature vectors from temporal QBER windows
for machine learning-based attack detection.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks

from .qber import (
    calculate_window_qber_batch,
    calculate_session_qber,
    calculate_qber_statistics,
    detect_significant_changes,
    calculate_trend_features,
)
from .windowing import create_windows, WindowConfig


# Feature group definitions matching the paper's 63 temporal-QBER features
TEMPORAL_QBER_FEATURE_GROUPS = {
    "statistical": [
        "qber_mean", "qber_var", "qber_std", "qber_skew", 
        "qber_kurtosis", "qber_iqr", "qber_cv"
    ],
    "burst_instability": [
        "spike_density", "burst_count", "burst_mean_duration",
        "burst_max_duration", "burst_total_duration",
        "jump_energy", "temporal_drift",
        "error_gap_mean", "error_gap_std", "error_gap_cv",
        "error_gap_skew", "consecutive_error_runs",
        "max_consecutive_errors", "error_clustering_coeff",
        "instability_index"
    ],
    "spectral_temporal": [
        "lag1_autocorr", "lag2_autocorr", "lag3_autocorr",
        "dft_energy_low", "dft_energy_mid", "dft_energy_high",
        "spectral_entropy", "spectral_centroid", "spectral_bandwidth",
        "shannon_entropy", "permutation_entropy", "sample_entropy",
        "hurst_exponent", "lyapunov_exp", "correlation_dim"
    ],
    "basis_dependent": [
        "qber_rectilinear", "qber_diagonal", "qber_basis_diff",
        "qber_basis_ratio", "rect_burst_energy", "diag_burst_energy",
        "basis_asymmetry", "basis_switch_rate", "basis_error_corr",
        "basis_drift"
    ],
    "channel_interaction": [
        "key_gen_rate", "transmission_efficiency", "qber_loss_corr",
        "qber_loss_interaction", "multi_scale_qber_1", "multi_scale_qber_2",
        "multi_scale_qber_4", "multi_scale_qber_8", "multi_scale_qber_16",
        "multi_scale_qber_32", "loss_burst_corr", "detector_efficiency",
        "dark_count_rate", "afterpulse_prob", "channel_stability", "qber_trend"
    ],
}

# Flatten feature names in fixed order
TEMPORAL_QBER_FEATURES = (
    TEMPORAL_QBER_FEATURE_GROUPS["statistical"] +
    TEMPORAL_QBER_FEATURE_GROUPS["burst_instability"] +
    TEMPORAL_QBER_FEATURE_GROUPS["spectral_temporal"] +
    TEMPORAL_QBER_FEATURE_GROUPS["basis_dependent"] +
    TEMPORAL_QBER_FEATURE_GROUPS["channel_interaction"]
)

# Verify feature count
assert len(TEMPORAL_QBER_FEATURES) == 63, f"Expected 63 features, got {len(TEMPORAL_QBER_FEATURES)}"


def extract_temporal_qber_features(
    sifted_alice: np.ndarray,
    sifted_bob: np.ndarray,
    bob_detected: np.ndarray = None,
    sifted_alice_rect: np.ndarray = None,
    sifted_bob_rect: np.ndarray = None,
    sifted_alice_diag: np.ndarray = None,
    sifted_bob_diag: np.ndarray = None,
    loss_seq: Optional[np.ndarray] = None,
    detection_efficiency: float = 1.0,
    dark_count_prob: float = 0.0,
    afterpulse_prob: float = 0.0,
    window_config: Optional["WindowConfig"] = None,
    min_sifted_bits: int = 200,
) -> Dict[str, Any]:
    """
    Extract all 63 temporal-QBER features from a BB84 session.
    
    Args:
        sifted_alice: Alice's sifted key bits
        sifted_bob: Bob's sifted key bits
        bob_detected: Bob's photon detection flags
        sifted_alice_rect: Alice's rectilinear basis sifted bits
        sifted_bob_rect: Bob's rectilinear basis sifted bits
        sifted_alice_diag: Alice's diagonal basis sifted bits
        sifted_bob_diag: Bob's diagonal basis sifted bits
        loss_seq: Photon loss sequence per window
        window_config: Temporal window configuration
        min_sifted_bits: Minimum sifted bits per window for valid QBER
        
    Returns:
        Dictionary with all 63 features
    """
    from .windowing import create_windows, WindowConfig
    from .qber import (
        calculate_window_qber_batch,
        calculate_session_qber,
        calculate_qber_statistics,
        detect_significant_changes,
        calculate_trend_features,
    )
    
    if window_config is None:
        window_config = WindowConfig()
    
    # Create temporal windows
    windows = create_windows(
        sifted_alice, sifted_bob, 
        bob_detected=bob_detected,
        config=WindowConfig(
            window_size=window_config.window_size,
            stride=window_config.stride,
            min_sifted_bits=min_sifted_bits,
            include_partial_trailing=window_config.include_partial_trailing,
        )
    )
    
    if len(windows) == 0:
        return {f: np.nan for f in TEMPORAL_QBER_FEATURES}
    
    # Extract QBER values for valid windows
    valid_windows = [w for w in windows if w.get("is_valid", False)]
    qber_values = np.array([w["qber"] for w in valid_windows])
    
    if len(qber_values) == 0:
        return {f: np.nan for f in TEMPORAL_QBER_FEATURES}
    
    # ============================================================
    # A. QBER Statistics (7 features)
    # ============================================================
    from .qber import calculate_qber_statistics, detect_significant_changes, calculate_trend_features
    
    qber_stats = calculate_qber_statistics(np.array([w["qber"] for w in valid_windows]))
    features = {
        "qber_mean": qber_stats["mean"],
        "qber_var": qber_stats["var"],
        "qber_std": qber_stats["std"],
        "qber_skew": qber_stats["skewness"],
        "qber_kurtosis": qber_stats["kurtosis"],
        "qber_iqr": qber_stats["iqr"],
        "qber_cv": qber_stats["cv"],
    }
    
    # ============================================================
    # B. Burst and Instability Features (~15 features)
    # ============================================================
    qber_seq = qber_values
    burst_features = _extract_burst_features(qber_seq)
    features = {**features, **burst_features}
    
    # ============================================================
    # C. Spectral and Temporal Features (~15 features)
    # ============================================================
    spectral_features = _extract_spectral_features(qber_seq)
    features = {**features, **spectral_features}
    
    # ============================================================
    # D. Trend Features (4 features)
    # ============================================================
    trend_features = calculate_trend_features(qber_seq)
    features = {**features, **trend_features}
    
    # ============================================================
    # E. Change Detection Features (6 features)
    # ============================================================
    change_features = detect_significant_changes(qber_seq)
    features.update({
        "num_changes": change_features["num_changes"],
        "max_change": change_features["max_change"],
        "mean_abs_change": change_features["mean_abs_change"],
        "std_change": change_features["std_change"],
        "change_rate": change_features["num_changes"] / len(qber_seq) if len(qber_seq) > 0 else 0.0,
    })
    
    # ============================================================
    # F. Basis-Dependent Features (if available)
    # ============================================================
    if sifted_alice_rect is not None and sifted_bob_rect is not None and \
       sifted_alice_diag is not None and sifted_bob_diag is not None:
        basis_features = _extract_basis_features(
            sifted_alice_rect, sifted_bob_rect,
            sifted_alice_diag, sifted_bob_diag,
        )
        features.update(basis_features)
    else:
        features.update({
            "qber_rectilinear": np.nan,
            "qber_diagonal": np.nan,
            "qber_basis_diff": np.nan,
            "qber_basis_ratio": np.nan,
            "rect_burst_energy": np.nan,
            "diag_burst_energy": np.nan,
            "basis_asymmetry": np.nan,
            "basis_switch_rate": np.nan,
            "basis_error_corr": np.nan,
            "basis_drift": np.nan,
        })
    
    # ============================================================
    # G. Channel-Interaction Features (~16 features)
    # ============================================================
    if loss_seq is not None:
        channel_features = _extract_channel_features(qber_seq, loss_seq, detection_efficiency, dark_count_prob, afterpulse_prob)
        features.update(channel_features)
    else:
        features.update({
            "key_gen_rate": np.nan,
            "transmission_efficiency": np.nan,
            "qber_loss_corr": np.nan,
            "qber_loss_interaction": np.nan,
            "multi_scale_qber_1": np.nan,
            "multi_scale_qber_2": np.nan,
            "multi_scale_qber_4": np.nan,
            "multi_scale_qber_8": np.nan,
            "multi_scale_qber_16": np.nan,
            "multi_scale_qber_32": np.nan,
            "loss_burst_corr": np.nan,
            "detector_efficiency": float(detection_efficiency),
            "dark_count_rate": float(dark_count_prob),
            "afterpulse_prob": float(afterpulse_prob),
            "channel_stability": np.nan,
            "qber_trend": np.nan,
        })
    
    # Add detection and sifting statistics
    if len(windows) > 0:
        features["mean_detection_rate"] = float(np.mean([w.get("detection_rate", 0) for w in windows]))
        features["detection_rate_std"] = float(np.std([w.get("detection_rate", 0) for w in windows]))
        features["mean_sifting_efficiency"] = float(np.mean([w.get("sifted_bits", 0) / max(1, w.get("length", 1)) for w in windows]))
        features["valid_window_fraction"] = len(valid_windows) / len(windows) if windows else 0.0
        features["mean_sifted_bits_per_window"] = float(np.mean([w.get("sifted_bits", 0) for w in valid_windows])) if valid_windows else 0.0
        features["total_windows"] = len(windows)
        features["valid_windows"] = len(valid_windows)
    
    # Ensure all features are present and in correct order
    result = {}
    for f in TEMPORAL_QBER_FEATURES:
        result[f] = features.get(f, np.nan)
    
    return result


def _extract_burst_features(qber_seq: np.ndarray) -> Dict[str, float]:
    """Extract burst and instability features from QBER sequence."""
    if len(qber_seq) == 0:
        return {f: 0.0 for f in TEMPORAL_QBER_FEATURE_GROUPS["burst_instability"]}
    
    # Threshold for spike detection
    threshold = np.mean(qber_seq) + 2 * np.std(qber_seq) if np.std(qber_seq) > 0 else np.mean(qber_seq)
    spikes = qber_seq > threshold
    spike_density = np.mean(spikes)
    
    # Burst detection (consecutive spikes)
    burst_starts = np.where(np.diff(spikes.astype(int)) == 1)[0] + 1
    burst_ends = np.where(np.diff(spikes.astype(int)) == -1)[0] + 1
    
    if spikes[0]:
        burst_starts = np.insert(burst_starts, 0, 0)
    if spikes[-1]:
        burst_ends = np.append(burst_ends, len(qber_seq))
    
    burst_durations = burst_ends - burst_starts
    burst_count = len(burst_durations)
    
    # Jump energy (sum of squared differences)
    jumps = np.diff(qber_seq)
    jump_energy = np.sum(jumps ** 2)
    
    # Temporal drift (linear trend)
    x = np.arange(len(qber_seq))
    if len(qber_seq) > 1:
        slope, _ = np.polyfit(x, qber_seq, 1)
        temporal_drift = slope
    else:
        temporal_drift = 0.0
    
    # Error gap statistics (gaps between errors)
    error_indices = np.where(qber_seq > 0)[0]
    if len(error_indices) > 1:
        gaps = np.diff(error_indices)
        error_gap_mean = np.mean(gaps)
        error_gap_std = np.std(gaps)
        error_gap_cv = error_gap_std / error_gap_mean if error_gap_mean > 0 else 0.0
        error_gap_skew = stats.skew(gaps) if error_gap_std > 0 else 0.0
    else:
        error_gap_mean = error_gap_std = error_gap_cv = error_gap_skew = 0.0
    
    # Consecutive error runs
    error_runs = []
    current_run = 0
    for val in qber_seq:
        if val > 0:
            current_run += 1
        else:
            if current_run > 0:
                error_runs.append(current_run)
                current_run = 0
    if current_run > 0:
        error_runs.append(current_run)
    
    consecutive_error_runs = len(error_runs)
    max_consecutive_errors = max(error_runs) if error_runs else 0
    
    return {
        "spike_density": float(spike_density),
        "burst_count": float(burst_count),
        "burst_mean_duration": float(np.mean(burst_durations)) if burst_count > 0 else 0.0,
        "burst_max_duration": float(np.max(burst_durations)) if burst_count > 0 else 0.0,
        "burst_total_duration": float(np.sum(burst_durations)) if burst_count > 0 else 0.0,
        "jump_energy": float(jump_energy),
        "temporal_drift": float(temporal_drift),
        "error_gap_mean": float(error_gap_mean),
        "error_gap_std": float(error_gap_std),
        "error_gap_cv": float(error_gap_cv),
        "error_gap_skew": float(error_gap_skew),
        "consecutive_error_runs": float(consecutive_error_runs),
        "max_consecutive_errors": float(max_consecutive_errors),
        "error_clustering_coeff": float(burst_count / len(qber_seq)) if len(qber_seq) > 0 else 0.0,
        "instability_index": float(jump_energy / len(qber_seq)) if len(qber_seq) > 0 else 0.0,
    }


def _extract_spectral_features(qber_seq: np.ndarray) -> Dict[str, float]:
    """Extract spectral and temporal features from QBER sequence."""
    if len(qber_seq) < 4:
        return {f: 0.0 for f in TEMPORAL_QBER_FEATURE_GROUPS["spectral_temporal"]}
    
    # Autocorrelation
    n = len(qber_seq)
    mean_val = np.mean(qber_seq)
    var_val = np.var(qber_seq)
    
    if var_val == 0:
        lag1_ac = lag2_ac = lag3_ac = 0.0
    else:
        # Use numpy correlate for autocorrelation
        lag1_ac = np.correlate(qber_seq - mean_val, qber_seq - mean_val, mode='full')[n] / (n * var_val)
        lag2_ac = np.correlate(qber_seq[:-2] - mean_val, qber_seq[2:] - mean_val, mode='valid')[0] / ((n-2) * var_val) if n > 2 else 0.0
        lag3_ac = np.correlate(qber_seq[:-3] - mean_val, qber_seq[3:] - mean_val, mode='valid')[0] / ((n-3) * var_val) if n > 3 else 0.0
    
    # DFT
    dft = fft(qber_seq)
    psd = np.abs(dft) ** 2
    total_energy = np.sum(psd)
    
    if total_energy > 0:
        n_freq = len(psd)
        low_end = n_freq // 4
        mid_end = n_freq // 2
        
        dft_energy_low = np.sum(psd[:low_end]) / total_energy
        dft_energy_mid = np.sum(psd[low_end:mid_end]) / total_energy
        dft_energy_high = np.sum(psd[mid_end:]) / total_energy
        
        # Spectral entropy
        psd_norm = psd / total_energy
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        
        # Spectral centroid
        freqs = np.arange(n_freq)
        spectral_centroid = np.sum(freqs * psd_norm)
        spectral_bandwidth = np.sqrt(np.sum((freqs - spectral_centroid) ** 2 * psd_norm))
    else:
        dft_energy_low = dft_energy_mid = dft_energy_high = 0.0
        spectral_entropy = spectral_centroid = spectral_bandwidth = 0.0
    
    # Shannon entropy of QBER values
    hist, _ = np.histogram(qber_seq, bins=min(10, len(qber_seq)), density=True)
    hist = hist[hist > 0]
    shannon_entropy = -np.sum(hist * np.log2(hist))
    
    # Permutation entropy (simplified)
    if n >= 3:
        permutations = []
        for i in range(n - 2):
            triplet = qber_seq[i:i+3]
            perm = tuple(np.argsort(triplet))
            permutations.append(perm)
        _, perm_counts = np.unique(permutations, return_counts=True)
        perm_probs = perm_counts / len(permutations)
        permutation_entropy = -np.sum(perm_probs * np.log2(perm_probs))
    else:
        permutation_entropy = 0.0
    
    # Sample entropy (approximate using approximate entropy)
    # Approximate entropy (ApEn) as approximation of sample entropy
    def _approx_entropy(data, m=2, r_factor=0.2):
        """Compute approximate entropy (ApEn) as approximation of sample entropy."""
        n = len(data)
        if n < m + 1:
            return 0.0
        
        r = r_factor * np.std(data)
        if r == 0:
            return 0.0
        
        def _phi(m):
            patterns = np.array([data[i:i+m] for i in range(n - m + 1)])
            C = np.zeros(n - m + 1)
            for i in range(n - m + 1):
                template = patterns[i]
                dists = np.max(np.abs(patterns - template), axis=1)
                C[i] = np.sum(dists <= r) / (n - m + 1)
            return np.mean(np.log(C + 1e-12))
        
        return _phi(m) - _phi(m + 1)
    
    sample_entropy = _approx_entropy(qber_seq)
    
    # Hurst exponent (simplified R/S analysis)
    def _hurst_exponent(data):
        """Simplified Hurst exponent using R/S analysis."""
        n = len(data)
        if n < 20:
            return 0.5
        
        # R/S analysis
        mean_val = np.mean(data)
        data_demeaned = data - mean_val
        
        # Cumulative deviation
        cum_dev = np.cumsum(data_demeaned)
        
        # Range
        R = np.max(cum_dev) - np.min(cum_dev)
        S = np.std(data)
        
        if S == 0:
            return 0.5
        
        # R/S ratio
        rs = R / S
        
        # For large n, H ≈ log(R/S) / log(n) (simplified)
        if rs > 0 and n > 1:
            return float(np.log(rs) / np.log(n))
        return 0.5
    
    hurst_exponent = _hurst_exponent(qber_seq)
    
    # Lyapunov exponent (simplified)
    def _lyapunov_exponent(data):
        n = len(data)
        if n < 10:
            return 0.0
        
        # Simplified: estimate using nearest neighbor distances
        m = 2  # embedding dimension
        tau = 1  # delay
        
        # Create embedded vectors
        vectors = np.array([data[i:i+m*tau:tau] for i in range(len(data) - (m-1)*tau)])
        # Actually simpler: just use successive differences
        diffs = np.abs(np.diff(data))
        if len(diffs) > 1 and np.mean(diffs) > 0:
            # Estimate using divergence of nearby trajectories
            # Simplified: use log of divergence rate
            return float(np.log(np.mean(np.abs(np.diff(data)) + 1e-12)))
        return 0.0
    
    lyapunov_exp = _lyapunov_exponent(qber_seq)
    
    # Correlation dimension (simplified Grassberger-Procaccia)
    def _correlation_dimension(data):
        n = len(data)
        if n < 10:
            return 0.0
        
        # Simplified: use embedding and correlation integral
        m = 3  # embedding dimension
        tau = 1
        N = len(data)
        
        if N < m + 2:
            return 0.0
        
        # Create embedded vectors
        M = N - (m - 1) * tau
        vectors = np.array([data[i:i + m*tau:tau] for i in range(M)])
        
        # Compute pairwise distances
        distances = []
        for i in range(min(M, 100)):  # Sample for efficiency
            for j in range(i+1, min(M, i+50)):
                dist = np.sqrt(np.sum((vectors[i] - vectors[j])**2))
                distances.append(dist)
        
        if len(distances) < 10:
            return 0.0
        
        distances = np.array(distances)
        # Estimate correlation dimension from slope of log(C(r)) vs log(r)
        r_vals = np.logspace(np.log10(np.min(distances) + 1e-12), 
                             np.log10(np.max(distances)), 10)
        C_r = []
        for r in r_vals:
            C_r.append(np.sum(np.array(distances) < r) / len(distances))
        
        # Linear fit to log(C(r)) vs log(r)
        log_r = np.log(r_vals)
        log_C = np.log(np.array(C_r) + 1e-12)
        if len(log_r) > 2 and np.std(log_r) > 0:
            coeff = np.polyfit(log_r, log_C, 1)
            return float(max(0, coeff[0]))
        return 0.0
    
    correlation_dim = _correlation_dimension(qber_seq)
    
    return {
        "lag1_autocorr": float(lag1_ac),
        "lag2_autocorr": float(lag2_ac),
        "lag3_autocorr": float(lag3_ac),
        "dft_energy_low": float(dft_energy_low),
        "dft_energy_mid": float(dft_energy_mid),
        "dft_energy_high": float(dft_energy_high),
        "spectral_entropy": float(spectral_entropy),
        "spectral_centroid": float(spectral_centroid),
        "spectral_bandwidth": float(spectral_bandwidth),
        "shannon_entropy": float(shannon_entropy),
        "permutation_entropy": float(permutation_entropy),
        "sample_entropy": float(sample_entropy),
        "hurst_exponent": float(hurst_exponent),
        "lyapunov_exp": float(lyapunov_exp),
        "correlation_dim": float(correlation_dim),
    }


def _extract_basis_features(
    sifted_alice_rect: np.ndarray,
    sifted_bob_rect: np.ndarray,
    sifted_alice_diag: np.ndarray,
    sifted_bob_diag: np.ndarray,
) -> Dict[str, float]:
    """Extract 10 basis-dependent features from rectilinear and diagonal basis sifted bits."""
    features = {}
    
    # QBER for rectilinear basis
    if len(sifted_alice_rect) > 0 and len(sifted_bob_rect) > 0:
        errors_rect = np.sum(sifted_alice_rect != sifted_bob_rect)
        qber_rect = float(errors_rect) / len(sifted_alice_rect) if len(sifted_alice_rect) > 0 else 0.0
    else:
        qber_rect = 0.0
    
    # QBER for diagonal basis
    if len(sifted_alice_diag) > 0 and len(sifted_bob_diag) > 0:
        errors_diag = np.sum(sifted_alice_diag != sifted_bob_diag)
        qber_diag = float(errors_diag) / len(sifted_alice_diag) if len(sifted_alice_diag) > 0 else 0.0
    else:
        qber_diag = 0.0
    
    features["qber_rectilinear"] = qber_rect
    features["qber_diagonal"] = qber_diag
    features["qber_basis_diff"] = qber_rect - qber_diag
    features["qber_basis_ratio"] = qber_rect / qber_diag if qber_diag > 0 else 0.0
    
    # Burst energy for rectilinear basis
    if len(sifted_alice_rect) > 0:
        errors_rect = (sifted_alice_rect != sifted_bob_rect).astype(float)
        rect_burst_energy = float(np.sum(errors_rect ** 2))
    else:
        rect_burst_energy = 0.0
    
    # Burst energy for diagonal basis
    if len(sifted_alice_diag) > 0:
        errors_diag = (sifted_alice_diag != sifted_bob_diag).astype(float)
        diag_burst_energy = float(np.sum(errors_diag ** 2))
    else:
        diag_burst_energy = 0.0
    
    features["rect_burst_energy"] = rect_burst_energy
    features["diag_burst_energy"] = diag_burst_energy
    
    # Basis asymmetry: (QBER_rect - QBER_diag) / (QBER_rect + QBER_diag)
    basis_sum = qber_rect + qber_diag
    features["basis_asymmetry"] = (qber_rect - qber_diag) / basis_sum if basis_sum > 0 else 0.0
    
    # Basis switch rate: how often the basis changes in the sifted key
    # We need the basis sequence for the sifted key - this is a simplified version
    # For now, we estimate based on the proportion of rectilinear vs diagonal
    total_sifted = len(sifted_alice_rect) + len(sifted_alice_diag)
    if total_sifted > 0:
        features["basis_switch_rate"] = 2 * len(sifted_alice_rect) * len(sifted_alice_diag) / (total_sifted * (total_sifted - 1)) if total_sifted > 1 else 0.0
    else:
        features["basis_switch_rate"] = 0.0
    
    # Basis error correlation: correlation between rectilinear and diagonal errors
    # We need aligned error sequences - simplified version
    if len(sifted_alice_rect) > 1 and len(sifted_alice_diag) > 1:
        min_len = min(len(sifted_alice_rect), len(sifted_alice_diag))
        errors_rect = (sifted_alice_rect[:min_len] != sifted_bob_rect[:min_len]).astype(float)
        errors_diag = (sifted_alice_diag[:min_len] != sifted_bob_diag[:min_len]).astype(float)
        if np.std(errors_rect) > 0 and np.std(errors_diag) > 0:
            corr = np.corrcoef(errors_rect, errors_diag)[0, 1]
            features["basis_error_corr"] = float(corr) if not np.isnan(corr) else 0.0
        else:
            features["basis_error_corr"] = 0.0
    else:
        features["basis_error_corr"] = 0.0
    
    # Basis drift: linear trend difference between rectilinear and diagonal QBER over time
    # Simplified: compare first half vs second half QBER for each basis
    if len(sifted_alice_rect) > 2 and len(sifted_alice_diag) > 2:
        mid_rect = len(sifted_alice_rect) // 2
        mid_diag = len(sifted_alice_diag) // 2
        
        qber_rect_early = np.mean(sifted_alice_rect[:mid_rect] != sifted_bob_rect[:mid_rect]) if mid_rect > 0 else 0
        qber_rect_late = np.mean(sifted_alice_rect[mid_rect:] != sifted_bob_rect[mid_rect:]) if mid_rect < len(sifted_alice_rect) else 0
        qber_diag_early = np.mean(sifted_alice_diag[:mid_diag] != sifted_bob_diag[:mid_diag]) if mid_diag > 0 else 0
        qber_diag_late = np.mean(sifted_alice_diag[mid_diag:] != sifted_bob_diag[mid_diag:]) if mid_diag < len(sifted_alice_diag) else 0
        
        drift_rect = qber_rect_late - qber_rect_early
        drift_diag = qber_diag_late - qber_diag_early
        features["basis_drift"] = float(drift_rect - drift_diag)
    else:
        features["basis_drift"] = 0.0
    
    return features


def _extract_channel_features(qber_seq: np.ndarray, loss_seq: np.ndarray, detection_efficiency: float = 1.0, dark_count_prob: float = 0.0, afterpulse_prob: float = 0.0) -> Dict[str, float]:
    """Extract 16 channel-interaction features from QBER sequence and loss sequence."""
    features = {}
    
    if len(qber_seq) == 0 or len(loss_seq) == 0:
        return {
            "key_gen_rate": 0.0,
            "transmission_efficiency": 0.0,
            "qber_loss_corr": 0.0,
            "qber_loss_interaction": 0.0,
            "multi_scale_qber_1": 0.0,
            "multi_scale_qber_2": 0.0,
            "multi_scale_qber_4": 0.0,
            "multi_scale_qber_8": 0.0,
            "multi_scale_qber_16": 0.0,
            "multi_scale_qber_32": 0.0,
            "loss_burst_corr": 0.0,
            "detector_efficiency": float(detection_efficiency),
            "dark_count_rate": float(dark_count_prob),
            "afterpulse_prob": float(afterpulse_prob),
            "channel_stability": 0.0,
            "qber_trend": 0.0,
        }
    
    # Ensure same length
    min_len = min(len(qber_seq), len(loss_seq))
    qber_seq = qber_seq[:min_len]
    loss_seq = loss_seq[:min_len]
    
    # 1. Key generation rate: sifted key rate * (1 - h(QBER) - f * h(QBER))
    # Simplified: transmission efficiency * (1 - H2(QBER))
    mean_qber = np.mean(qber_seq)
    if 0 <= mean_qber < 0.5:
        h_qber = -mean_qber * np.log2(mean_qber) - (1 - mean_qber) * np.log2(1 - mean_qber)
        key_gen_rate = max(0.0, 1 - 2 * h_qber)  # Simplified key rate
    else:
        key_gen_rate = 0.0
    
    # Transmission efficiency: fraction of transmitted qubits that contribute to sifted key
    transmission_efficiency = 1.0 - np.mean(np.array(loss_seq))  # Simplified
    
    # QBER-Loss correlation
    if len(qber_seq) > 1 and np.std(qber_seq) > 0 and np.std(loss_seq) > 0:
        qber_loss_corr = np.corrcoef(qber_seq, loss_seq)[0, 1]
        qber_loss_corr = float(qber_loss_corr) if not np.isnan(qber_loss_corr) else 0.0
    else:
        qber_loss_corr = 0.0
    
    # QBER-Loss interaction: mean(QBER * loss)
    qber_loss_interaction = float(np.mean(qber_seq * loss_seq))
    
    # Multi-scale QBER (average QBER at different scales)
    scales = [1, 2, 4, 8, 16, 32]
    multi_scale = {}
    for scale in scales:
        if len(qber_seq) >= scale:
            n_windows = len(qber_seq) // scale
            if n_windows > 0:
                qber_scaled = qber_seq[:n_windows * scale].reshape(n_windows, scale).mean(axis=1)
                multi_scale[f"multi_scale_qber_{scale}"] = float(np.mean(qber_scaled))
            else:
                multi_scale[f"multi_scale_qber_{scale}"] = 0.0
        else:
            multi_scale[f"multi_scale_qber_{scale}"] = 0.0
    
    features.update(multi_scale)
    
    # Loss burst correlation
    if len(loss_seq) > 1 and np.std(loss_seq) > 0:
        loss_diff = np.diff(loss_seq)
        loss_bursts = (loss_seq[:-1] > np.mean(loss_seq)) & (loss_seq[1:] > np.mean(loss_seq))
        if np.sum(loss_bursts) > 0:
            loss_burst_corr = float(np.mean(loss_bursts.astype(float)))
        else:
            loss_burst_corr = 0.0
    else:
        loss_burst_corr = 0.0
    
    features.update({
        "key_gen_rate": float(key_gen_rate),
        "transmission_efficiency": float(transmission_efficiency),
        "qber_loss_corr": float(qber_loss_corr),
        "qber_loss_interaction": float(qber_loss_interaction),
        "loss_burst_corr": float(loss_burst_corr),
        "detector_efficiency": float(detection_efficiency),
        "dark_count_rate": float(dark_count_prob),
        "afterpulse_prob": float(afterpulse_prob),
        "channel_stability": 1.0 - float(np.std(qber_seq)) if len(qber_seq) > 1 else 1.0,
    })
    
    return features