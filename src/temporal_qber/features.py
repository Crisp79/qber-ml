"""
Feature engineering for Temporal-QBER pipeline.

Note: QAD dataset already provides 44 quantum-circuit features.
This module implements the paper's 63 temporal-QBER features for BB84 sessions.
For QAD, we use the existing features directly.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks


# Feature registry for paper's 63 temporal-QBER features
TEMPORAL_QBER_FEATURES = [
    # Statistical features (7)
    "qber_mean",
    "qber_var",
    "qber_std",
    "qber_skew",
    "qber_kurtosis",
    "qber_iqr",
    "qber_cv",
    
    # Burst and instability features (~15)
    "spike_density",
    "burst_count",
    "burst_mean_duration",
    "burst_max_duration",
    "burst_total_duration",
    "jump_energy",
    "temporal_drift",
    "error_gap_mean",
    "error_gap_std",
    "error_gap_cv",
    "error_gap_skew",
    "consecutive_error_runs",
    "max_consecutive_errors",
    "error_clustering_coeff",
    "instability_index",
    
    # Spectral and temporal features (~15)
    "lag1_autocorr",
    "lag2_autocorr",
    "lag3_autocorr",
    "dft_energy_low",
    "dft_energy_mid",
    "dft_energy_high",
    "spectral_entropy",
    "spectral_centroid",
    "spectral_bandwidth",
    "shannon_entropy",
    "permutation_entropy",
    "sample_entropy",
    "hurst_exponent",
    "lyapunov_exp",
    "correlation_dim",
    
    # Basis-dependent features (~10)
    "qber_rectilinear",
    "qber_diagonal",
    "qber_basis_diff",
    "qber_basis_ratio",
    "rect_burst_energy",
    "diag_burst_energy",
    "basis_asymmetry",
    "basis_switch_rate",
    "basis_error_corr",
    "basis_drift",
    
    # Channel-interaction features (~16)
    "key_gen_rate",
    "transmission_efficiency",
    "qber_loss_corr",
    "qber_loss_interaction",
    "multi_scale_qber_1",
    "multi_scale_qber_2",
    "multi_scale_qber_4",
    "multi_scale_qber_8",
    "multi_scale_qber_16",
    "multi_scale_qber_32",
    "loss_burst_corr",
    "detector_efficiency",
    "dark_count_rate",
    "afterpulse_prob",
    "channel_stability",
    "qber_trend",
]


def extract_statistical_features(qber_seq: np.ndarray) -> Dict[str, float]:
    """Extract 7 statistical features from QBER sequence."""
    if len(qber_seq) == 0:
        return {f: 0.0 for f in TEMPORAL_QBER_FEATURES[:7]}
    
    mean_val = np.mean(qber_seq)
    std_val = np.std(qber_seq)
    
    return {
        "qber_mean": mean_val,
        "qber_var": np.var(qber_seq),
        "qber_std": std_val,
        "qber_skew": stats.skew(qber_seq) if std_val > 0 else 0.0,
        "qber_kurtosis": stats.kurtosis(qber_seq) if std_val > 0 else 0.0,
        "qber_iqr": stats.iqr(qber_seq) if len(qber_seq) > 1 else 0.0,
        "qber_cv": std_val / mean_val if mean_val > 0 else 0.0,
    }


def extract_burst_features(qber_seq: np.ndarray, threshold: float = None) -> Dict[str, float]:
    """Extract burst and instability features."""
    if len(qber_seq) == 0:
        return {f: 0.0 for f in TEMPORAL_QBER_FEATURES[7:22]}
    
    if threshold is None:
        threshold = np.mean(qber_seq) + 2 * np.std(qber_seq)
    
    # Spike detection
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
        "spike_density": spike_density,
        "burst_count": float(burst_count),
        "burst_mean_duration": np.mean(burst_durations) if burst_count > 0 else 0.0,
        "burst_max_duration": np.max(burst_durations) if burst_count > 0 else 0.0,
        "burst_total_duration": np.sum(burst_durations) if burst_count > 0 else 0.0,
        "jump_energy": jump_energy,
        "temporal_drift": temporal_drift,
        "error_gap_mean": error_gap_mean,
        "error_gap_std": error_gap_std,
        "error_gap_cv": error_gap_cv,
        "error_gap_skew": error_gap_skew,
        "consecutive_error_runs": float(consecutive_error_runs),
        "max_consecutive_errors": float(max_consecutive_errors),
        "error_clustering_coeff": burst_count / len(qber_seq) if len(qber_seq) > 0 else 0.0,
        "instability_index": jump_energy / len(qber_seq) if len(qber_seq) > 0 else 0.0,
    }


def extract_spectral_features(qber_seq: np.ndarray) -> Dict[str, float]:
    """Extract spectral and temporal features."""
    if len(qber_seq) < 4:
        return {f: 0.0 for f in TEMPORAL_QBER_FEATURES[22:37]}
    
    # Autocorrelation
    n = len(qber_seq)
    mean_val = np.mean(qber_seq)
    var_val = np.var(qber_seq)
    
    if var_val == 0:
        lag1_ac = lag2_ac = lag3_ac = 0.0
    else:
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
    
    # Sample entropy (approximate)
    sample_entropy = 0.0  # placeholder
    
    # Hurst exponent (simplified R/S analysis)
    hurst_exponent = 0.5  # placeholder
    
    # Lyapunov exponent (placeholder)
    lyapunov_exp = 0.0
    
    # Correlation dimension (placeholder)
    correlation_dim = 0.0
    
    return {
        "lag1_autocorr": lag1_ac,
        "lag2_autocorr": lag2_ac,
        "lag3_autocorr": lag3_ac,
        "dft_energy_low": dft_energy_low,
        "dft_energy_mid": dft_energy_mid,
        "dft_energy_high": dft_energy_high,
        "spectral_entropy": spectral_entropy,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "shannon_entropy": shannon_entropy,
        "permutation_entropy": permutation_entropy,
        "sample_entropy": sample_entropy,
        "hurst_exponent": hurst_exponent,
        "lyapunov_exp": lyapunov_exp,
        "correlation_dim": correlation_dim,
    }


def extract_basis_features(qber_rect: np.ndarray, qber_diag: np.ndarray) -> Dict[str, float]:
    """Extract basis-dependent features."""
    if len(qber_rect) == 0 or len(qber_diag) == 0:
        return {f: 0.0 for f in TEMPORAL_QBER_FEATURES[37:47]}
    
    qber_rect_mean = np.mean(qber_rect)
    qber_diag_mean = np.mean(qber_diag)
    
    # Basis burst energy
    rect_bursts = qber_rect > (np.mean(qber_rect) + 2 * np.std(qber_rect))
    diag_bursts = qber_diag > (np.mean(qber_diag) + 2 * np.std(qber_diag))
    
    return {
        "qber_rectilinear": qber_rect_mean,
        "qber_diagonal": qber_diag_mean,
        "qber_basis_diff": qber_rect_mean - qber_diag_mean,
        "qber_basis_ratio": qber_rect_mean / qber_diag_mean if qber_diag_mean > 0 else 0.0,
        "rect_burst_energy": np.sum(rect_bursts * qber_rect),
        "diag_burst_energy": np.sum(diag_bursts * qber_diag),
        "basis_asymmetry": (qber_rect_mean - qber_diag_mean) / (qber_rect_mean + qber_diag_mean) if (qber_rect_mean + qber_diag_mean) > 0 else 0.0,
        "basis_switch_rate": 0.0,  # placeholder
        "basis_error_corr": np.corrcoef(qber_rect, qber_diag)[0,1] if len(qber_rect) == len(qber_diag) and len(qber_rect) > 1 else 0.0,
        "basis_drift": 0.0,  # placeholder
    }


def extract_channel_features(qber_seq: np.ndarray, loss_seq: np.ndarray = None, 
                             key_rate: float = 0.0, det_eff: float = 0.0) -> Dict[str, float]:
    """Extract channel-interaction features."""
    if len(qber_seq) == 0:
        return {f: 0.0 for f in TEMPORAL_QBER_FEATURES[47:]}
    
    if loss_seq is None:
        loss_seq = np.zeros_like(qber_seq)
    
    # Multi-scale QBER (average over different window sizes)
    multi_scales = []
    for scale in [1, 2, 4, 8, 16, 32]:
        if len(qber_seq) >= scale:
            n_windows = len(qber_seq) // scale
            if n_windows > 0:
                windowed = qber_seq[:n_windows*scale].reshape(n_windows, scale).mean(axis=1)
                multi_scales.append(np.mean(windowed))
            else:
                multi_scales.append(0.0)
        else:
            multi_scales.append(0.0)
    
    # QBER-loss correlation
    if len(loss_seq) == len(qber_seq) and len(qber_seq) > 1:
        qber_loss_corr = np.corrcoef(qber_seq, loss_seq)[0,1]
        if np.isnan(qber_loss_corr):
            qber_loss_corr = 0.0
    else:
        qber_loss_corr = 0.0
    
    return {
        "key_gen_rate": key_rate,
        "transmission_efficiency": 1.0 - np.mean(loss_seq) if len(loss_seq) > 0 else 1.0,
        "qber_loss_corr": qber_loss_corr,
        "qber_loss_interaction": np.mean(qber_seq * loss_seq) if len(loss_seq) == len(qber_seq) else 0.0,
        "multi_scale_qber_1": multi_scales[0],
        "multi_scale_qber_2": multi_scales[1],
        "multi_scale_qber_4": multi_scales[2],
        "multi_scale_qber_8": multi_scales[3],
        "multi_scale_qber_16": multi_scales[4],
        "multi_scale_qber_32": multi_scales[5],
        "loss_burst_corr": 0.0,  # placeholder
        "detector_efficiency": det_eff,
        "dark_count_rate": 0.0,  # placeholder
        "afterpulse_prob": 0.0,  # placeholder
        "channel_stability": 1.0 / (1.0 + np.std(qber_seq)) if np.std(qber_seq) > 0 else 1.0,
        "qber_trend": 0.0,  # placeholder
    }


def extract_temporal_qber_features(qber_seq: np.ndarray, 
                                    qber_rect: np.ndarray = None,
                                    qber_diag: np.ndarray = None,
                                    loss_seq: np.ndarray = None,
                                    key_rate: float = 0.0,
                                    det_eff: float = 0.0) -> np.ndarray:
    """
    Extract all 63 temporal-QBER features.
    
    Args:
        qber_seq: Overall temporal QBER sequence (windows x 1)
        qber_rect: Rectilinear basis QBER sequence
        qber_diag: Diagonal basis QBER sequence
        loss_seq: Photon loss sequence per window
        key_rate: Key generation rate
        det_eff: Detector efficiency
    
    Returns:
        Array of 63 features in fixed order
    """
    features = {}
    
    # Statistical
    features.update(extract_statistical_features(qber_seq))
    
    # Burst
    features.update(extract_burst_features(qber_seq))
    
    # Spectral
    features.update(extract_spectral_features(qber_seq))
    
    # Basis-dependent (use overall if basis-specific not provided)
    if qber_rect is None:
        qber_rect = qber_seq
    if qber_diag is None:
        qber_diag = qber_seq
    features.update(extract_basis_features(qber_rect, qber_diag))
    
    # Channel-interaction
    features.update(extract_channel_features(qber_seq, loss_seq, key_rate, det_eff))
    
    # Return in fixed order
    return np.array([features[f] for f in TEMPORAL_QBER_FEATURES], dtype=np.float32)


# For QAD dataset - use existing features directly
def get_qad_feature_names() -> List[str]:
    """Get QAD feature names (after one-hot encoding circuit_type)."""
    # Base features (42) + circuit_type one-hot (3) = 45
    return None  # Will be determined at runtime after one-hot encoding