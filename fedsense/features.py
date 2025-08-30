"""
Feature extraction and data preprocessing for time-series anomaly detection.
Handles windowing, standardization, and optional FFT features.
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def make_windows(
    df: pd.DataFrame, 
    window_len: int = 250, 
    stride: int = 50,
    target_col: str = "label"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from time-series data.
    
    Args:
        df: DataFrame with columns [timestamp, hr, accel_x, accel_y, accel_z, label]
        window_len: Window length (number of samples)
        stride: Stride between windows
        target_col: Name of the target column
        
    Returns:
        X: Windows of shape (n_windows, window_len, n_features)
        y: Labels of shape (n_windows,)
    """
    # Feature columns (excluding timestamp and label)
    feature_cols = [col for col in df.columns if col not in ['timestamp', target_col]]
    
    # Extract features and labels
    features = df[feature_cols].values
    labels = df[target_col].values
    
    n_samples, n_features = features.shape
    
    # Calculate number of windows
    n_windows = max(0, (n_samples - window_len) // stride + 1)
    
    if n_windows == 0:
        logger.warning(f"No windows generated. Data length: {n_samples}, window_len: {window_len}")
        return np.empty((0, window_len, n_features)), np.empty((0,))
    
    # Create windows
    X = np.zeros((n_windows, window_len, n_features))
    y = np.zeros(n_windows)
    
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_len
        X[i] = features[start_idx:end_idx]
        # Use majority vote for window label
        y[i] = np.mean(labels[start_idx:end_idx]) > 0.5
    
    logger.info(f"Generated {n_windows} windows of shape {X.shape[1:]} from {n_samples} samples")
    return X, y.astype(np.int32)


def standardize_features(
    X_train: np.ndarray, 
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, ...]:
    """
    Standardize features using training statistics only.
    
    Args:
        X_train: Training windows of shape (n_windows, window_len, n_features)
        X_val: Optional validation windows
        X_test: Optional test windows
        
    Returns:
        Tuple of standardized arrays and fitted scaler
    """
    n_windows, window_len, n_features = X_train.shape
    
    # Reshape for StandardScaler (samples x features)
    X_train_reshaped = X_train.reshape(-1, n_features)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(n_windows, window_len, n_features)
    
    results = [X_train_scaled]
    
    # Transform validation data
    if X_val is not None:
        n_val_windows = X_val.shape[0]
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_val_scaled = scaler.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled.reshape(n_val_windows, window_len, n_features)
        results.append(X_val_scaled)
    
    # Transform test data
    if X_test is not None:
        n_test_windows = X_test.shape[0]
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_scaled = scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled.reshape(n_test_windows, window_len, n_features)
        results.append(X_test_scaled)
    
    results.append(scaler)
    return tuple(results)


def extract_fft_features(X: np.ndarray, fs: float = 50.0) -> np.ndarray:
    """
    Extract frequency domain features using FFT.
    
    Args:
        X: Time-series windows of shape (n_windows, window_len, n_features)
        fs: Sampling frequency in Hz
        
    Returns:
        FFT features of shape (n_windows, n_fft_features)
    """
    n_windows, window_len, n_features = X.shape
    
    # Define frequency bands (in Hz)
    bands = {
        'very_low': (0.0, 0.04),
        'low': (0.04, 0.15), 
        'high': (0.15, 0.4),
        'very_high': (0.4, fs/2)
    }
    
    n_fft_features = len(bands) * n_features
    fft_features = np.zeros((n_windows, n_fft_features))
    
    freqs = np.fft.rfftfreq(window_len, 1/fs)
    
    for i in range(n_windows):
        feature_idx = 0
        for j in range(n_features):
            # Compute FFT for this feature channel
            fft_vals = np.abs(np.fft.rfft(X[i, :, j]))
            
            # Extract power in each frequency band
            for band_name, (low_freq, high_freq) in bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.sum(fft_vals[band_mask] ** 2)
                fft_features[i, feature_idx] = band_power
                feature_idx += 1
    
    logger.info(f"Extracted FFT features of shape {fft_features.shape}")
    return fft_features


def train_val_test_split(
    X: np.ndarray, 
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/validation/test sets with stratification.
    
    Args:
        X: Feature windows
        y: Labels
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    np.random.seed(random_seed)
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    # Calculate split indices
    train_end = int(train_ratio * n_samples)
    val_end = int((train_ratio + val_ratio) * n_samples)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Train anomaly rate: {np.mean(y_train):.3f}")
    logger.info(f"Val anomaly rate: {np.mean(y_val):.3f}")
    logger.info(f"Test anomaly rate: {np.mean(y_test):.3f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_dataset_stats(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Compute dataset statistics for logging.
    
    Args:
        X: Feature windows
        y: Labels
        
    Returns:
        Dictionary of dataset statistics
    """
    stats = {
        'n_samples': len(X),
        'n_features': X.shape[-1],
        'window_length': X.shape[1],
        'anomaly_rate': float(np.mean(y)),
        'feature_means': X.mean(axis=(0, 1)).tolist(),
        'feature_stds': X.std(axis=(0, 1)).tolist(),
    }
    return stats
