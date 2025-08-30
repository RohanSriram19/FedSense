"""
JAX/Flax model implementation for time-series anomaly detection.
Uses 1D CNN architecture with global average pooling.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from flax.core import freeze, unfreeze
from typing import Tuple, Dict, Any, Callable
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import logging

logger = logging.getLogger(__name__)


class AnomalyDetectionCNN(nn.Module):
    """1D CNN for time-series anomaly detection."""
    
    hidden_dims: Tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the CNN.
        
        Args:
            x: Input tensor of shape (batch_size, window_len, n_features)
            training: Whether in training mode (for dropout)
            
        Returns:
            Output probabilities of shape (batch_size, 1)
        """
        # First conv layer
        x = nn.Conv(features=self.hidden_dims[0], kernel_size=[7], padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Second conv layer
        x = nn.Conv(features=self.hidden_dims[1], kernel_size=[5], padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=1)  # Shape: (batch_size, hidden_dims[1])
        
        # Final classification layer
        x = nn.Dense(features=1)(x)
        x = nn.sigmoid(x)
        
        return x


def create_train_state(
    rng: jax.random.PRNGKey,
    input_shape: Tuple[int, ...],
    learning_rate: float = 1e-3,
    hidden_dims: Tuple[int, ...] = (64, 32),
    dropout_rate: float = 0.1
) -> train_state.TrainState:
    """
    Create a training state with initialized parameters.
    
    Args:
        rng: Random key for initialization
        input_shape: Shape of input (window_len, n_features)
        learning_rate: Learning rate for optimizer
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout rate
        
    Returns:
        Initialized TrainState
    """
    model = AnomalyDetectionCNN(hidden_dims=hidden_dims, dropout_rate=dropout_rate)
    
    # Initialize with dummy input
    dummy_input = jnp.ones((1, *input_shape))
    params = model.init(rng, dummy_input, training=False)
    
    # Create optimizer
    optimizer = optax.adamw(learning_rate=learning_rate)
    
    # Create training state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    logger.info(f"Initialized model with {count_parameters(params)} parameters")
    return state


def count_parameters(params) -> int:
    """Count total number of model parameters."""
    total = 0
    for layer_params in jax.tree_util.tree_leaves(params):
        if hasattr(layer_params, 'size'):
            total += layer_params.size
    return total


@jax.jit
def train_step(
    state: train_state.TrainState, 
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    rng: jax.random.PRNGKey
) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """
    Perform one training step.
    
    Args:
        state: Current training state
        batch: Tuple of (X, y) where X is (batch_size, window_len, n_features)
        rng: Random key for dropout
        
    Returns:
        Updated state and metrics dictionary
    """
    X, y = batch
    
    def loss_fn(params):
        # Forward pass
        logits = state.apply_fn(params, X, training=True, rngs={'dropout': rng})
        logits = logits.squeeze(-1)  # Remove last dimension
        
        # Binary cross-entropy loss
        loss = optax.sigmoid_binary_cross_entropy(logits, y.astype(jnp.float32))
        return jnp.mean(loss)
    
    # Compute gradients
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Apply gradients
    state = state.apply_gradients(grads=grads)
    
    metrics = {'loss': loss}
    return state, metrics


@jax.jit  
def eval_step(
    state: train_state.TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Perform one evaluation step.
    
    Args:
        state: Current training state
        batch: Tuple of (X, y)
        
    Returns:
        Predictions, true labels, and loss
    """
    X, y = batch
    
    # Forward pass (no dropout in eval)
    logits = state.apply_fn(state.params, X, training=False)
    logits = logits.squeeze(-1)
    
    # Compute loss
    loss = optax.sigmoid_binary_cross_entropy(logits, y.astype(jnp.float32))
    loss = jnp.mean(loss)
    
    # Convert to probabilities
    probs = nn.sigmoid(logits)
    
    return probs, y, loss


def evaluate_model(
    state: train_state.TrainState,
    dataset: Any,  # FedSenseDataset
    batch_size: int = 64
) -> Dict[str, float]:
    """
    Evaluate model on a dataset and compute metrics.
    
    Args:
        state: Training state
        dataset: FedSenseDataset to evaluate on
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataset.batch_iterator(batch_size, shuffle=False):
        probs, labels, loss = eval_step(state, batch)
        
        all_probs.extend(np.array(probs))
        all_labels.extend(np.array(labels))
        total_loss += float(loss)
        n_batches += 1
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    
    # Find optimal threshold for F1
    thresholds = np.linspace(0, 1, 101)
    best_f1 = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        pred_labels = (all_probs >= threshold).astype(int)
        if np.sum(pred_labels) > 0 and np.sum(all_labels) > 0:  # Avoid division by zero
            f1 = f1_score(all_labels, pred_labels)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    metrics = {
        'loss': total_loss / n_batches,
        'auroc': auroc,
        'auprc': auprc,
        'f1': best_f1,
        'best_threshold': best_threshold,
        'n_samples': len(all_labels)
    }
    
    return metrics


def apply_differential_privacy(
    grads: Dict[str, Any],
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    rng: jax.random.PRNGKey = None
) -> Dict[str, Any]:
    """
    Apply differential privacy to gradients (clipping + noise).
    
    Args:
        grads: Gradient dictionary
        clip_norm: L2 norm clipping threshold
        noise_multiplier: Noise multiplier for Gaussian noise
        rng: Random key for noise generation
        
    Returns:
        DP gradients
    """
    # Flatten gradients for norm computation
    flat_grads = jax.tree_util.tree_leaves(grads)
    flat_grads_concat = jnp.concatenate([g.flatten() for g in flat_grads])
    
    # Compute L2 norm
    grad_norm = jnp.linalg.norm(flat_grads_concat)
    
    # Clip gradients
    clip_factor = jnp.minimum(1.0, clip_norm / (grad_norm + 1e-8))
    clipped_grads = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)
    
    # Add Gaussian noise
    if noise_multiplier > 0 and rng is not None:
        noise_grads = jax.tree_util.tree_map(
            lambda g: g + noise_multiplier * clip_norm * jax.random.normal(rng, g.shape),
            clipped_grads
        )
        return noise_grads
    
    return clipped_grads


def params_to_numpy(params) -> Dict[str, Any]:
    """Convert JAX parameters to NumPy for serialization."""
    return jax.tree_util.tree_map(lambda x: np.array(x), params)


def numpy_to_params(numpy_params: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Convert numpy arrays back to JAX parameters."""
    return jax.tree_util.tree_map(lambda x: jnp.array(x), numpy_params)


def save_model_params(params: Dict[str, Any], filepath: str) -> None:
    """Save model parameters to disk."""
    numpy_params = params_to_numpy(params)
    np.savez_compressed(filepath, **jax.tree_util.tree_flatten(numpy_params)[0])
    logger.info(f"Saved model parameters to {filepath}")


def load_model_params(filepath: str, template_params: Dict[str, Any]) -> Dict[str, Any]:
    """Load model parameters from disk."""
    loaded_data = np.load(filepath)
    flat_params = [loaded_data[f'arr_{i}'] for i in range(len(loaded_data.files))]
    numpy_params = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template_params), flat_params)
    params = numpy_to_params(numpy_params)
    logger.info(f"Loaded model parameters from {filepath}")
    return params
    return params
