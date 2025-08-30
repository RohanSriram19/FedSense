"""
Test suite for FedSense JAX models and training utilities.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from fedsense.model_jax import (
    AnomalyDetectionCNN, create_train_state, train_step, eval_step,
    train_epoch, evaluate_model, compute_metrics
)


class TestAnomalyDetectionCNN:
    """Test the JAX/Flax CNN model."""
    
    def test_model_initialization(self):
        """Test model initialization and parameter shapes."""
        key = random.PRNGKey(42)
        model = AnomalyDetectionCNN(hidden_dims=[32, 16], dropout_rate=0.1)
        
        # Create dummy input
        batch_size, window_len, n_features = 8, 100, 4
        x = jnp.ones((batch_size, window_len, n_features))
        
        # Initialize parameters
        params = model.init(key, x, training=False)
        
        # Check that parameters were created
        assert 'params' in params
        assert len(params['params']) > 0
    
    def test_model_forward_pass(self):
        """Test forward pass through the model."""
        key = random.PRNGKey(42)
        model = AnomalyDetectionCNN(hidden_dims=[16, 8], dropout_rate=0.0)
        
        batch_size, window_len, n_features = 4, 50, 4
        x = random.normal(key, (batch_size, window_len, n_features))
        
        # Initialize and run
        params = model.init(key, x, training=False)
        logits = model.apply(params, x, training=False)
        
        # Check output shape (binary classification)
        assert logits.shape == (batch_size, 2)
        
        # Check that outputs are finite
        assert jnp.all(jnp.isfinite(logits))
    
    def test_model_with_dropout(self):
        """Test model behavior with dropout enabled."""
        key = random.PRNGKey(42)
        model = AnomalyDetectionCNN(hidden_dims=[32], dropout_rate=0.5)
        
        batch_size, window_len, n_features = 2, 30, 4
        x = random.normal(key, (batch_size, window_len, n_features))
        
        # Initialize parameters
        params = model.init(key, x, training=True)
        
        # Run with training=True (dropout active)
        key1, key2 = random.split(key)
        logits1 = model.apply(params, x, training=True, rngs={'dropout': key1})
        logits2 = model.apply(params, x, training=True, rngs={'dropout': key2})
        
        # With dropout, outputs should be different
        assert not jnp.allclose(logits1, logits2, atol=1e-6)
        
        # Run with training=False (no dropout)
        logits_eval1 = model.apply(params, x, training=False)
        logits_eval2 = model.apply(params, x, training=False)
        
        # Without dropout, outputs should be identical
        assert jnp.allclose(logits_eval1, logits_eval2)


class TestTrainingUtilities:
    """Test training and evaluation utilities."""
    
    def test_create_train_state(self):
        """Test training state creation."""
        key = random.PRNGKey(42)
        
        # Create model and dummy data for initialization
        model = AnomalyDetectionCNN(hidden_dims=[16])
        x_dummy = jnp.ones((1, 50, 4))
        
        state = create_train_state(model, key, x_dummy, learning_rate=0.001)
        
        # Check that state has required components
        assert hasattr(state, 'params')
        assert hasattr(state, 'opt_state')
        assert hasattr(state, 'apply_fn')
        assert state.step == 0
    
    def test_train_step(self):
        """Test single training step."""
        key = random.PRNGKey(42)
        
        # Setup
        model = AnomalyDetectionCNN(hidden_dims=[8])
        batch_size, window_len, n_features = 4, 25, 4
        
        x = random.normal(key, (batch_size, window_len, n_features))
        y = random.randint(random.split(key)[0], (batch_size,), 0, 2)
        
        # Create initial state
        state = create_train_state(model, key, x, learning_rate=0.01)
        
        # Take training step
        new_state, loss, accuracy = train_step(state, x, y, dropout_key=key)
        
        # Check that state was updated
        assert new_state.step == state.step + 1
        
        # Check that loss is reasonable
        assert jnp.isfinite(loss)
        assert loss >= 0
        
        # Check accuracy is in [0, 1]
        assert 0 <= accuracy <= 1
    
    def test_eval_step(self):
        """Test evaluation step."""
        key = random.PRNGKey(42)
        
        # Setup
        model = AnomalyDetectionCNN(hidden_dims=[8])
        batch_size, window_len, n_features = 4, 25, 4
        
        x = random.normal(key, (batch_size, window_len, n_features))
        y = random.randint(random.split(key)[0], (batch_size,), 0, 2)
        
        # Create state
        state = create_train_state(model, key, x, learning_rate=0.01)
        
        # Evaluation step
        loss, accuracy, predictions = eval_step(state, x, y)
        
        # Check outputs
        assert jnp.isfinite(loss)
        assert 0 <= accuracy <= 1
        assert predictions.shape == (batch_size,)
        assert jnp.all((predictions >= 0) & (predictions <= 1))  # Binary predictions


class TestTrainingLoop:
    """Test training loop components."""
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        key = random.PRNGKey(42)
        
        # Create synthetic data
        n_samples, window_len, n_features = 32, 20, 4
        X = random.normal(key, (n_samples, window_len, n_features))
        y = random.randint(random.split(key)[0], (n_samples,), 0, 2)
        
        # Create data loader (simple batch iteration)
        batch_size = 8
        batches = []
        for i in range(0, n_samples, batch_size):
            batch_x = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            batches.append((batch_x, batch_y))
        
        # Setup model and state
        model = AnomalyDetectionCNN(hidden_dims=[8])
        state = create_train_state(model, key, X[:1], learning_rate=0.01)
        
        # Train epoch
        new_state, avg_loss, avg_accuracy = train_epoch(state, batches, key)
        
        # Check that state was updated
        assert new_state.step > state.step
        
        # Check metrics
        assert jnp.isfinite(avg_loss)
        assert 0 <= avg_accuracy <= 1
    
    def test_evaluate_model(self):
        """Test model evaluation on dataset."""
        key = random.PRNGKey(42)
        
        # Create synthetic data
        n_samples, window_len, n_features = 24, 20, 4
        X = random.normal(key, (n_samples, window_len, n_features))
        y = random.randint(random.split(key)[0], (n_samples,), 0, 2)
        
        # Create batches
        batch_size = 8
        batches = []
        for i in range(0, n_samples, batch_size):
            batch_x = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            batches.append((batch_x, batch_y))
        
        # Setup model
        model = AnomalyDetectionCNN(hidden_dims=[8])
        state = create_train_state(model, key, X[:1], learning_rate=0.01)
        
        # Evaluate
        metrics = evaluate_model(state, batches)
        
        # Check metrics
        expected_keys = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']
        for key in expected_keys:
            assert key in metrics
            assert jnp.isfinite(metrics[key])
        
        # Check value ranges
        assert metrics['accuracy'] >= 0 and metrics['accuracy'] <= 1
        assert metrics['precision'] >= 0 and metrics['precision'] <= 1
        assert metrics['recall'] >= 0 and metrics['recall'] <= 1
        assert metrics['f1'] >= 0 and metrics['f1'] <= 1
        assert metrics['auc'] >= 0 and metrics['auc'] <= 1


class TestMetrics:
    """Test metric computation functions."""
    
    def test_compute_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = jnp.array([0, 0, 1, 1])
        y_pred = jnp.array([0, 0, 1, 1])
        y_prob = jnp.array([0.1, 0.2, 0.8, 0.9])
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        # Perfect predictions should give perfect scores
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['auc'] == 1.0
    
    def test_compute_metrics_worst_predictions(self):
        """Test metrics with worst possible predictions."""
        y_true = jnp.array([0, 0, 1, 1])
        y_pred = jnp.array([1, 1, 0, 0])  # Completely wrong
        y_prob = jnp.array([0.9, 0.8, 0.2, 0.1])  # Confidently wrong
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        # Should have poor performance
        assert metrics['accuracy'] == 0.0
        assert metrics['auc'] == 0.0  # Perfectly wrong ranking
    
    def test_compute_metrics_all_same_class(self):
        """Test metrics when all samples are same class."""
        y_true = jnp.array([0, 0, 0, 0])
        y_pred = jnp.array([0, 0, 0, 0])
        y_prob = jnp.array([0.1, 0.2, 0.1, 0.3])
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        # Accuracy should be perfect
        assert metrics['accuracy'] == 1.0
        
        # Precision and recall might be undefined (handled gracefully)
        assert jnp.isfinite(metrics['precision']) or jnp.isnan(metrics['precision'])
        assert jnp.isfinite(metrics['recall']) or jnp.isnan(metrics['recall'])


if __name__ == "__main__":
    # Simple test runner
    test_classes = [
        TestAnomalyDetectionCNN,
        TestTrainingUtilities,
        TestTrainingLoop,
        TestMetrics
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
