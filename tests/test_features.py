"""
Test suite for FedSense feature extraction and preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from fedsense.features import (
    make_windows, standardize_features, train_val_test_split,
    extract_fft_features, get_dataset_stats
)


class TestMakeWindows:
    """Test windowing functionality."""
    
    def test_basic_windowing(self):
        """Test basic windowing with simple data."""
        # Create simple test data
        n_samples = 1000
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='20ms'),
            'hr': np.linspace(70, 80, n_samples),
            'accel_x': np.sin(np.linspace(0, 4*np.pi, n_samples)),
            'accel_y': np.cos(np.linspace(0, 4*np.pi, n_samples)),
            'accel_z': np.ones(n_samples) * 9.8,
            'label': np.zeros(n_samples, dtype=int)
        })
        
        window_len = 100
        stride = 50
        
        X, y = make_windows(df, window_len, stride)
        
        # Check shapes
        expected_n_windows = (n_samples - window_len) // stride + 1
        assert X.shape == (expected_n_windows, window_len, 4)
        assert y.shape == (expected_n_windows,)
        assert y.dtype == np.int32
    
    def test_windowing_with_anomalies(self):
        """Test windowing with anomalous samples."""
        n_samples = 500
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='20ms'),
            'hr': np.full(n_samples, 75.0),
            'accel_x': np.zeros(n_samples),
            'accel_y': np.zeros(n_samples),
            'accel_z': np.full(n_samples, 9.8),
            'label': np.zeros(n_samples, dtype=int)
        })
        
        # Add some anomalies
        anomaly_indices = [100, 150, 300, 350, 400]
        df.loc[anomaly_indices, 'label'] = 1
        
        window_len = 50
        stride = 25
        
        X, y = make_windows(df, window_len, stride)
        
        # Check that some windows have anomaly labels
        assert y.sum() > 0
        assert y.sum() < len(y)  # Not all windows should be anomalous
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='20ms'),
            'hr': np.full(10, 75.0),
            'accel_x': np.zeros(10),
            'accel_y': np.zeros(10),
            'accel_z': np.full(10, 9.8),
            'label': np.zeros(10, dtype=int)
        })
        
        window_len = 50  # Larger than data
        stride = 25
        
        X, y = make_windows(df, window_len, stride)
        
        # Should return empty arrays
        assert X.shape == (0, window_len, 4)
        assert y.shape == (0,)


class TestStandardizeFeatures:
    """Test feature standardization."""
    
    def test_standardization_shapes(self):
        """Test that standardization preserves shapes."""
        n_windows, window_len, n_features = 100, 50, 4
        
        # Create random data
        np.random.seed(42)
        X_train = np.random.randn(n_windows, window_len, n_features)
        X_val = np.random.randn(20, window_len, n_features)
        X_test = np.random.randn(30, window_len, n_features)
        
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_features(
            X_train, X_val, X_test
        )
        
        # Check shapes are preserved
        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled.shape == X_val.shape
        assert X_test_scaled.shape == X_test.shape
    
    def test_standardization_properties(self):
        """Test that standardization has correct statistical properties."""
        np.random.seed(42)
        X_train = np.random.randn(100, 50, 4) * 10 + 5  # Mean=5, std≈10
        
        X_train_scaled, scaler = standardize_features(X_train)
        
        # Check that training data is standardized (mean≈0, std≈1)
        train_reshaped = X_train_scaled.reshape(-1, 4)
        np.testing.assert_allclose(train_reshaped.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(train_reshaped.std(axis=0), 1, atol=1e-6)
    
    def test_validation_uses_training_stats(self):
        """Test that validation data uses training statistics."""
        np.random.seed(42)
        
        # Training data with known statistics
        X_train = np.ones((50, 25, 2)) * 10  # Mean=10, std=0
        
        # Validation data with different statistics  
        X_val = np.ones((20, 25, 2)) * 20   # Mean=20, std=0
        
        X_train_scaled, X_val_scaled, scaler = standardize_features(X_train, X_val)
        
        # Training data should be standardized
        assert np.allclose(X_train_scaled, (10 - 10) / 1e-8)  # Should be ~0 (with numerical issues)
        
        # Validation data should use training stats (mean=10, std=tiny)
        # So (20-10)/tiny ≈ large positive number
        assert np.all(X_val_scaled > X_train_scaled)


class TestTrainValTestSplit:
    """Test data splitting functionality."""
    
    def test_split_ratios(self):
        """Test that split ratios are approximately correct."""
        n_samples = 1000
        X = np.random.randn(n_samples, 50, 4)
        y = np.random.randint(0, 2, n_samples)
        
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, train_ratio=0.7, val_ratio=0.15, random_seed=42
        )
        
        # Check approximate ratios (within 1% due to integer division)
        assert abs(len(X_train) / n_samples - 0.7) < 0.01
        assert abs(len(X_val) / n_samples - 0.15) < 0.01  
        assert abs(len(X_test) / n_samples - 0.15) < 0.01
        
        # Check no data leakage (total samples)
        assert len(X_train) + len(X_val) + len(X_test) == n_samples
    
    def test_split_reproducibility(self):
        """Test that splits are reproducible with same seed."""
        X = np.random.randn(100, 20, 3)
        y = np.random.randint(0, 2, 100)
        
        # Two identical splits
        split1 = train_val_test_split(X, y, random_seed=123)
        split2 = train_val_test_split(X, y, random_seed=123)
        
        # Should be identical
        for i in range(6):  # 6 returned arrays
            np.testing.assert_array_equal(split1[i], split2[i])


class TestExtractFFTFeatures:
    """Test FFT feature extraction."""
    
    def test_fft_shapes(self):
        """Test FFT feature shapes."""
        n_windows, window_len, n_features = 50, 250, 4
        X = np.random.randn(n_windows, window_len, n_features)
        
        fft_features = extract_fft_features(X, fs=50.0)
        
        # Should have 4 frequency bands × 4 features = 16 FFT features
        expected_fft_features = 4 * n_features
        assert fft_features.shape == (n_windows, expected_fft_features)
    
    def test_fft_with_sine_wave(self):
        """Test FFT with known sine wave."""
        fs = 50.0
        duration = 5.0  # 5 seconds
        t = np.linspace(0, duration, int(fs * duration))
        
        # Create sine wave at 2 Hz
        freq = 2.0
        sine_wave = np.sin(2 * np.pi * freq * t)
        
        # Create input array (single window, single feature)
        X = np.zeros((1, len(t), 1))
        X[0, :, 0] = sine_wave
        
        fft_features = extract_fft_features(X, fs=fs)
        
        # Should have power in the low frequency band (0.04-0.15 Hz contains our 2Hz signal)
        # Actually, 2Hz is in the "high" band (0.15-0.4 Hz) based on typical HRV bands
        assert fft_features.shape == (1, 4)  # 4 bands × 1 feature
        assert np.all(fft_features >= 0)  # Power should be non-negative


class TestGetDatasetStats:
    """Test dataset statistics computation."""
    
    def test_stats_computation(self):
        """Test basic statistics computation."""
        # Create simple test data
        n_samples, window_len, n_features = 100, 50, 4
        X = np.random.randn(n_samples, window_len, n_features) 
        y = np.random.randint(0, 2, n_samples)
        
        stats = get_dataset_stats(X, y)
        
        # Check all expected keys are present
        expected_keys = [
            'n_samples', 'n_features', 'window_length', 'anomaly_rate',
            'feature_means', 'feature_stds'
        ]
        for key in expected_keys:
            assert key in stats
        
        # Check values make sense
        assert stats['n_samples'] == n_samples
        assert stats['n_features'] == n_features
        assert stats['window_length'] == window_len
        assert 0 <= stats['anomaly_rate'] <= 1
        assert len(stats['feature_means']) == n_features
        assert len(stats['feature_stds']) == n_features


if __name__ == "__main__":
    pytest.main([__file__])
