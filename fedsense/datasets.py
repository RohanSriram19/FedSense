"""
Dataset utilities for JAX and PyTorch models.
Handles data loading, batching, and client data partitioning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterator, Tuple, List, Dict, Any, Optional
import logging
from .features import make_windows, standardize_features, train_val_test_split
from .config import FedSenseConfig

logger = logging.getLogger(__name__)


class FedSenseDataset:
    """Dataset class for FedSense time-series anomaly detection."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Feature windows of shape (n_samples, window_len, n_features)
            y: Labels of shape (n_samples,)
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int32)
        self.n_samples = len(X)
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[idx], self.y[idx]
    
    def batch_iterator(self, batch_size: int, shuffle: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Create batches for JAX training.
        
        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
            
        Yields:
            Batches of (X_batch, y_batch)
        """
        indices = np.arange(self.n_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, self.n_samples, batch_size):
            end_idx = min(start_idx + batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield self.X[batch_indices], self.y[batch_indices]


def generate_synthetic_data(
    n_samples: int = 10000,
    fs: float = 50.0,
    window_len: int = 250,
    anomaly_rate: float = 0.1,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic wearable sensor data for testing.
    
    Args:
        n_samples: Number of samples to generate
        fs: Sampling frequency in Hz  
        window_len: Length of each window
        anomaly_rate: Proportion of anomalous samples
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns [timestamp, hr, accel_x, accel_y, accel_z, label]
    """
    np.random.seed(random_seed)
    
    # Generate timestamps
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq=f'{1000/fs:.0f}ms')
    
    # Generate normal heart rate (70-80 bpm baseline with random walk)
    hr_baseline = 75.0
    hr_noise = np.random.normal(0, 2.0, n_samples)
    hr_walk = np.cumsum(np.random.normal(0, 0.1, n_samples))
    hr = hr_baseline + hr_walk + hr_noise
    hr = np.clip(hr, 50, 120)  # Physiological limits
    
    # Generate accelerometer data (normal activity)
    accel_x = np.random.normal(0, 0.5, n_samples) + 0.1 * np.sin(2 * np.pi * 0.5 * np.arange(n_samples) / fs)
    accel_y = np.random.normal(0, 0.5, n_samples) + 0.1 * np.cos(2 * np.pi * 0.3 * np.arange(n_samples) / fs)
    accel_z = np.random.normal(9.8, 0.2, n_samples)  # Gravity + small variations
    
    # Generate anomaly labels
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    labels = np.zeros(n_samples, dtype=int)
    labels[anomaly_indices] = 1
    
    # Inject anomalies (elevated HR + high acceleration)
    for idx in anomaly_indices:
        # Create anomalous patterns in windows around the anomaly
        window_start = max(0, idx - window_len//2)
        window_end = min(n_samples, idx + window_len//2)
        
        # Elevated heart rate
        hr[window_start:window_end] *= np.random.uniform(1.3, 1.8)
        hr[window_start:window_end] = np.clip(hr[window_start:window_end], 50, 180)
        
        # High acceleration (exercise/stress)
        scale_factor = np.random.uniform(2.0, 4.0)
        accel_x[window_start:window_end] *= scale_factor
        accel_y[window_start:window_end] *= scale_factor
        accel_z[window_start:window_end] += np.random.normal(0, 1.0, window_end - window_start)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'hr': hr,
        'accel_x': accel_x,
        'accel_y': accel_y,
        'accel_z': accel_z,
        'label': labels
    })
    
    logger.info(f"Generated synthetic data: {len(df)} samples, {anomaly_rate:.1%} anomalies")
    return df


def create_federated_splits(
    df: pd.DataFrame,
    n_clients: int = 8,
    alpha: float = 0.1,
    random_seed: int = 42
) -> Dict[int, pd.DataFrame]:
    """
    Create non-IID federated data splits using Dirichlet distribution.
    
    Args:
        df: Full dataset
        n_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping client_id to client DataFrame
    """
    np.random.seed(random_seed)
    
    # Group by label for non-IID split
    normal_data = df[df['label'] == 0].copy()
    anomaly_data = df[df['label'] == 1].copy()
    
    client_splits = {}
    
    # Distribute normal data using Dirichlet
    normal_proportions = np.random.dirichlet([alpha] * n_clients)
    normal_splits = np.random.multinomial(len(normal_data), normal_proportions)
    
    # Distribute anomaly data using Dirichlet  
    anomaly_proportions = np.random.dirichlet([alpha] * n_clients)
    anomaly_splits = np.random.multinomial(len(anomaly_data), anomaly_proportions)
    
    normal_idx = 0
    anomaly_idx = 0
    
    for client_id in range(n_clients):
        # Get client's portion of normal data
        client_normal = normal_data.iloc[normal_idx:normal_idx + normal_splits[client_id]].copy()
        normal_idx += normal_splits[client_id]
        
        # Get client's portion of anomaly data  
        client_anomaly = anomaly_data.iloc[anomaly_idx:anomaly_idx + anomaly_splits[client_id]].copy()
        anomaly_idx += anomaly_splits[client_id]
        
        # Combine and shuffle
        client_data = pd.concat([client_normal, client_anomaly], ignore_index=True)
        client_data = client_data.sample(frac=1, random_state=random_seed + client_id).reset_index(drop=True)
        
        client_splits[client_id] = client_data
        
        logger.info(f"Client {client_id}: {len(client_data)} samples, "
                   f"{np.mean(client_data['label']):.3f} anomaly rate")
    
    return client_splits


def load_client_data(
    client_id: int,
    config: FedSenseConfig,
    data_splits: Optional[Dict[int, pd.DataFrame]] = None
) -> Tuple[FedSenseDataset, FedSenseDataset]:
    """
    Load and preprocess data for a specific federated client.
    
    Args:
        client_id: ID of the client
        config: FedSense configuration
        data_splits: Pre-computed data splits (if None, will load from files)
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    if data_splits is not None:
        # Use provided data splits
        client_df = data_splits[client_id]
    else:
        # Load from file
        client_data_path = config.data_dir / f"client_{client_id}" / "data.csv"
        if not client_data_path.exists():
            raise FileNotFoundError(f"Client data not found: {client_data_path}")
        client_df = pd.read_csv(client_data_path)
    
    # Create windows
    X, y = make_windows(client_df, config.window_len, config.stride)
    
    # Train/val split for this client
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(
        X, y, train_ratio=0.8, val_ratio=0.2, random_seed=config.random_seed
    )
    
    # Standardize features (fit on train only)
    X_train_scaled, X_val_scaled, scaler = standardize_features(X_train, X_val)
    
    # Create datasets
    train_dataset = FedSenseDataset(X_train_scaled, y_train)
    val_dataset = FedSenseDataset(X_val_scaled, y_val)
    
    logger.info(f"Client {client_id} data loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def save_client_data(data_splits: Dict[int, pd.DataFrame], config: FedSenseConfig) -> None:
    """
    Save federated client data splits to disk.
    
    Args:
        data_splits: Dictionary mapping client_id to DataFrame
        config: FedSense configuration
    """
    for client_id, client_df in data_splits.items():
        client_dir = config.data_dir / f"client_{client_id}"
        client_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = client_dir / "data.csv"
        client_df.to_csv(output_path, index=False)
        logger.info(f"Saved client {client_id} data to {output_path}")
    
    # Save global test set (equal samples from all clients)
    all_data = pd.concat(data_splits.values(), ignore_index=True)
    test_data = all_data.sample(frac=0.2, random_state=config.random_seed)
    
    test_path = config.data_dir / "global_test.csv"
    test_data.to_csv(test_path, index=False)
    logger.info(f"Saved global test data to {test_path}")
