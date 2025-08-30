"""
Test suite for FedSense datasets and data generation.
"""

import numpy as np
from fedsense.datasets import (
    generate_synthetic_wearable_data, create_federated_splits,
    get_data_loaders, WearableDataset
)


class TestGenerateSyntheticWearableData:
    """Test synthetic data generation."""
    
    def test_basic_data_generation(self):
        """Test basic data generation with default parameters."""
        df = generate_synthetic_wearable_data()
        
        # Check required columns exist
        required_cols = ['timestamp', 'hr', 'accel_x', 'accel_y', 'accel_z', 'label']
        for col in required_cols:
            assert col in df.columns
        
        # Check data types
        assert df['timestamp'].dtype.name.startswith('datetime')
        assert df['label'].dtype == np.int32
        
        # Check reasonable value ranges
        assert (df['hr'] >= 40).all()  # Reasonable heart rate
        assert (df['hr'] <= 200).all()
        assert (df['accel_z'].abs() > 5).any()  # Should have gravity component
    
    def test_custom_parameters(self):
        """Test data generation with custom parameters."""
        n_samples = 5000
        anomaly_rate = 0.15
        random_seed = 123
        
        df = generate_synthetic_wearable_data(
            n_samples=n_samples,
            anomaly_rate=anomaly_rate,
            random_seed=random_seed
        )
        
        # Check sample count
        assert len(df) == n_samples
        
        # Check anomaly rate (within 5% tolerance)
        actual_rate = df['label'].mean()
        assert abs(actual_rate - anomaly_rate) < 0.05
    
    def test_reproducibility(self):
        """Test that data generation is reproducible with same seed."""
        seed = 42
        
        df1 = generate_synthetic_wearable_data(n_samples=1000, random_seed=seed)
        df2 = generate_synthetic_wearable_data(n_samples=1000, random_seed=seed)
        
        # Should be identical
        assert df1.equals(df2)
    
    def test_no_anomalies(self):
        """Test data generation with no anomalies."""
        df = generate_synthetic_wearable_data(
            n_samples=1000,
            anomaly_rate=0.0,
            random_seed=42
        )
        
        # Should have no anomalies
        assert df['label'].sum() == 0
    
    def test_all_anomalies(self):
        """Test data generation with all anomalies."""
        df = generate_synthetic_wearable_data(
            n_samples=1000,
            anomaly_rate=1.0,
            random_seed=42
        )
        
        # Should all be anomalies
        assert df['label'].sum() == 1000


class TestCreateFederatedSplits:
    """Test federated data splitting."""
    
    def test_basic_splitting(self):
        """Test basic federated splitting."""
        # Generate test data
        df = generate_synthetic_wearable_data(n_samples=10000, random_seed=42)
        
        n_clients = 5
        client_dfs = create_federated_splits(df, n_clients, random_seed=42)
        
        # Check number of clients
        assert len(client_dfs) == n_clients
        
        # Check that all data is distributed
        total_samples = sum(len(client_df) for client_df in client_dfs)
        assert total_samples == len(df)
        
        # Check no data leakage (each sample appears exactly once)
        all_indices = []
        for client_df in client_dfs:
            all_indices.extend(client_df.index.tolist())
        
        assert len(all_indices) == len(df)
        assert len(set(all_indices)) == len(df)  # No duplicates
    
    def test_non_iid_splitting(self):
        """Test non-IID splitting (heterogeneous distribution)."""
        # Generate data with clear anomaly patterns
        df = generate_synthetic_wearable_data(
            n_samples=5000, 
            anomaly_rate=0.2, 
            random_seed=42
        )
        
        n_clients = 4
        client_dfs = create_federated_splits(
            df, n_clients, 
            alpha=0.1,  # High heterogeneity
            random_seed=42
        )
        
        # Calculate anomaly rates per client
        client_rates = [client_df['label'].mean() for client_df in client_dfs]
        
        # Should have different anomaly rates (heterogeneous)
        rate_std = np.std(client_rates)
        assert rate_std > 0.01  # Some variation in anomaly rates
    
    def test_iid_splitting(self):
        """Test IID splitting (homogeneous distribution)."""
        df = generate_synthetic_wearable_data(
            n_samples=5000,
            anomaly_rate=0.2,
            random_seed=42
        )
        
        n_clients = 4
        client_dfs = create_federated_splits(
            df, n_clients,
            alpha=100,  # Very low heterogeneity (more IID)
            random_seed=42
        )
        
        # Calculate anomaly rates per client
        client_rates = [client_df['label'].mean() for client_df in client_dfs]
        
        # Should have similar anomaly rates
        rate_std = np.std(client_rates)
        assert rate_std < 0.05  # Less variation
    
    def test_single_client(self):
        """Test splitting with single client."""
        df = generate_synthetic_wearable_data(n_samples=1000, random_seed=42)
        
        client_dfs = create_federated_splits(df, n_clients=1, random_seed=42)
        
        # Single client should get all data
        assert len(client_dfs) == 1
        assert len(client_dfs[0]) == len(df)


class TestWearableDataset:
    """Test WearableDataset class."""
    
    def test_dataset_creation(self):
        """Test dataset creation from numpy arrays."""
        n_samples = 100
        window_len = 50
        n_features = 4
        
        X = np.random.randn(n_samples, window_len, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        dataset = WearableDataset(X, y)
        
        assert len(dataset) == n_samples
        
        # Test indexing
        sample_x, sample_y = dataset[0]
        assert sample_x.shape == (window_len, n_features)
        assert isinstance(sample_y, (int, np.integer))
    
    def test_dataset_iteration(self):
        """Test dataset iteration."""
        X = np.random.randn(10, 20, 3)
        y = np.random.randint(0, 2, 10)
        
        dataset = WearableDataset(X, y)
        
        # Test iteration
        samples = list(dataset)
        assert len(samples) == 10
        
        for i, (x, y_val) in enumerate(samples):
            assert x.shape == (20, 3)
            assert y_val == y[i]


class TestGetDataLoaders:
    """Test data loader creation."""
    
    def test_dataloader_creation(self):
        """Test basic data loader creation."""
        # Create sample datasets
        n_samples = 200
        X = np.random.randn(n_samples, 50, 4)
        y = np.random.randint(0, 2, n_samples)
        
        train_ds = WearableDataset(X[:150], y[:150])
        val_ds = WearableDataset(X[150:175], y[150:175])
        test_ds = WearableDataset(X[175:], y[175:])
        
        train_loader, val_loader, test_loader = get_data_loaders(
            train_ds, val_ds, test_ds,
            batch_size=32,
            num_workers=0  # Avoid multiprocessing in tests
        )
        
        # Test loader properties
        assert train_loader.batch_size == 32
        assert val_loader.batch_size == 32
        assert test_loader.batch_size == 32
        
        # Test that we can iterate
        train_batch = next(iter(train_loader))
        batch_x, batch_y = train_batch
        
        assert batch_x.shape[0] <= 32  # Batch size
        assert batch_x.shape[1:] == (50, 4)  # Window and feature dims
        assert len(batch_y) <= 32
    
    def test_shuffle_behavior(self):
        """Test that training data is shuffled."""
        X = np.arange(100 * 20 * 2).reshape(100, 20, 2)
        y = np.arange(100)
        
        train_ds = WearableDataset(X, y)
        
        train_loader, _, _ = get_data_loaders(
            train_ds, train_ds, train_ds,  # Use same dataset
            batch_size=10,
            num_workers=0
        )
        
        # Get first batch
        batch_x, batch_y = next(iter(train_loader))
        
        # Due to shuffling, first batch shouldn't be [0, 1, 2, ..., 9]
        # This is probabilistic, but very likely to pass
        expected_order = np.arange(10)
        assert not np.array_equal(batch_y.numpy(), expected_order)


if __name__ == "__main__":
    # Simple test runner without pytest dependency
    test_classes = [
        TestGenerateSyntheticWearableData,
        TestCreateFederatedSplits,
        TestWearableDataset,
        TestGetDataLoaders
    ]
    
    for test_class in test_classes:
        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith('test_')]
        
        print(f"Running {test_class.__name__}:")
        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✓ {method_name}")
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
        print()
