"""
Flower federated learning client implementation.
Handles local training with JAX/Flax models and optional differential privacy.
"""

import flwr as fl
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import os

from .model_jax import (
    create_train_state, train_step, evaluate_model,
    params_to_numpy, numpy_to_params
)
from .datasets import load_client_data, FedSenseDataset
from .config import get_config
from .dp_utils import (
    DPTrainingState, PrivacyAccountant, apply_dp_to_gradients,
    validate_dp_parameters
)
from .utils_logging import log_client_metrics

logger = logging.getLogger(__name__)


class FedSenseClient(fl.client.NumPyClient):
    """Federated learning client for anomaly detection."""
    
    def __init__(self, 
                 client_id: int,
                 config: Optional[Any] = None,
                 data_splits: Optional[Dict[int, Any]] = None):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            config: FedSense configuration
            data_splits: Pre-computed data splits
        """
        self.client_id = client_id
        self.config = config or get_config()
        
        # Load client data
        self.train_dataset, self.val_dataset = load_client_data(
            client_id, self.config, data_splits
        )
        
        # Initialize model state
        rng = jax.random.PRNGKey(self.config.random_seed + client_id)
        input_shape = (self.config.window_len, 4)  # HR + 3 accel channels
        
        self.state = create_train_state(
            rng=rng,
            input_shape=input_shape,
            learning_rate=self.config.learning_rate,
            hidden_dims=self.config.hidden_dims,
            dropout_rate=self.config.dropout_rate
        )
        
        # Setup differential privacy if enabled
        self.dp_enabled = self.config.use_dp
        if self.dp_enabled:
            # Validate DP parameters
            is_valid = validate_dp_parameters(
                epsilon=self.config.dp_epsilon,
                delta=self.config.dp_delta,
                n_samples=len(self.train_dataset),
                clip_norm=self.config.clip_norm,
                noise_multiplier=self.config.noise_multiplier
            )
            
            if not is_valid:
                logger.error(f"Invalid DP parameters for client {client_id}")
                self.dp_enabled = False
            else:
                # Create privacy accountant
                self.privacy_accountant = PrivacyAccountant(
                    n_samples=len(self.train_dataset),
                    noise_multiplier=self.config.noise_multiplier,
                    delta=self.config.dp_delta
                )
                
                # Wrap state with DP
                self.dp_state = DPTrainingState(
                    base_state=self.state,
                    privacy_accountant=self.privacy_accountant,
                    clip_norm=self.config.clip_norm,
                    noise_multiplier=self.config.noise_multiplier
                )
                
                logger.info(f"Client {client_id}: DP enabled with ε={self.config.dp_epsilon}, "
                           f"δ={self.config.dp_delta}")
        
        logger.info(f"Initialized client {client_id} with {len(self.train_dataset)} "
                   f"training samples, {len(self.val_dataset)} validation samples")
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Get model parameters as numpy arrays.
        
        Args:
            config: Configuration from server
            
        Returns:
            List of parameter arrays
        """
        current_state = self.dp_state.base_state if self.dp_enabled else self.state
        numpy_params = params_to_numpy(current_state.params)
        
        # Flatten parameters for transmission
        flat_params = []
        for layer_name, layer_params in numpy_params.items():
            if isinstance(layer_params, dict):
                for param_name, param_array in layer_params.items():
                    flat_params.append(param_array.flatten())
            else:
                flat_params.append(layer_params.flatten())
        
        return flat_params
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from numpy arrays.
        
        Args:
            parameters: List of parameter arrays from server
        """
        # Reconstruct parameter structure
        # This is simplified - in production, you'd need proper parameter mapping
        logger.info(f"Client {self.client_id}: Received {len(parameters)} parameter arrays")
        
        # For now, we'll recreate the state with new parameters
        # In practice, you'd need to properly map the flat arrays back to the tree structure
        # This is a placeholder implementation
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train model locally.
        
        Args:
            parameters: Global model parameters
            config: Training configuration
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        logger.info(f"Client {self.client_id}: Starting local training round")
        
        # Set global parameters (if provided)
        if parameters:
            self.set_parameters(parameters)
        
        # Local training
        metrics = self._local_training()
        
        # Get updated parameters
        updated_params = self.get_parameters(config)
        
        # Log metrics
        log_client_metrics(self.client_id, metrics, round_num=config.get('round', 0))
        
        return updated_params, len(self.train_dataset), metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model locally.
        
        Args:
            parameters: Global model parameters  
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        logger.info(f"Client {self.client_id}: Starting local evaluation")
        
        # Set parameters
        if parameters:
            self.set_parameters(parameters)
        
        # Evaluate on validation set
        current_state = self.dp_state.base_state if self.dp_enabled else self.state
        val_metrics = evaluate_model(current_state, self.val_dataset, self.config.batch_size)
        
        logger.info(f"Client {self.client_id} validation: "
                   f"Loss={val_metrics['loss']:.4f}, AUROC={val_metrics['auroc']:.4f}")
        
        return val_metrics['loss'], len(self.val_dataset), val_metrics
    
    def _local_training(self) -> Dict[str, Any]:
        """
        Perform local training for specified number of epochs.
        
        Returns:
            Training metrics
        """
        current_state = self.dp_state.base_state if self.dp_enabled else self.state
        rng = jax.random.PRNGKey(self.config.random_seed + self.client_id)
        
        metrics = {
            'train_loss': 0.0,
            'train_batches': 0,
            'grad_norm': 0.0
        }
        
        # Add DP metrics if enabled
        if self.dp_enabled:
            metrics.update({
                'dp_clip_norm': self.config.clip_norm,
                'dp_noise_multiplier': self.config.noise_multiplier,
                'dp_clipped_batches': 0
            })
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            # Training loop
            for batch in self.train_dataset.batch_iterator(self.config.batch_size, shuffle=True):
                rng, step_rng = jax.random.split(rng)
                
                if self.dp_enabled:
                    # DP training step
                    updated_dp_state, step_metrics = self._dp_train_step(batch, step_rng)
                    self.dp_state = updated_dp_state
                    current_state = self.dp_state.base_state
                    
                    # Track DP metrics
                    if step_metrics.get('clipped', False):
                        metrics['dp_clipped_batches'] += 1
                else:
                    # Regular training step
                    updated_state, step_metrics = train_step(current_state, batch, step_rng)
                    self.state = updated_state
                    current_state = self.state
                
                epoch_loss += step_metrics['loss']
                epoch_batches += 1
                metrics['grad_norm'] += step_metrics.get('original_grad_norm', 0.0)
            
            metrics['train_loss'] += epoch_loss / epoch_batches
            metrics['train_batches'] += epoch_batches
            
            logger.info(f"Client {self.client_id} Epoch {epoch+1}: "
                       f"Loss={epoch_loss/epoch_batches:.4f}")
        
        # Average metrics
        metrics['train_loss'] /= self.config.local_epochs
        metrics['grad_norm'] /= metrics['train_batches']
        
        # Add privacy metrics if DP enabled
        if self.dp_enabled:
            epsilon, delta = self.privacy_accountant.get_privacy_spent()
            metrics.update({
                'privacy_epsilon': epsilon,
                'privacy_delta': delta,
                'privacy_steps': self.privacy_accountant.steps
            })
            
            logger.info(f"Client {self.client_id} privacy spent: ε={epsilon:.3f}, δ={delta:.2e}")
        
        return metrics
    
    def _dp_train_step(self, batch: Tuple[jnp.ndarray, jnp.ndarray], rng: jax.random.PRNGKey) -> Tuple[DPTrainingState, Dict[str, Any]]:
        """
        Perform one DP training step.
        
        Args:
            batch: Training batch
            rng: Random key
            
        Returns:
            Updated DP state and metrics
        """
        X, y = batch
        current_state = self.dp_state.base_state
        
        def loss_fn(params):
            logits = current_state.apply_fn(params, X, training=True, rngs={'dropout': rng})
            logits = logits.squeeze(-1)
            loss = jnp.mean(jax.nn.sigmoid_cross_entropy_with_logits(logits, y.astype(jnp.float32)))
            return loss
        
        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(current_state.params)
        
        # Apply DP to gradients and update state
        rng, dp_rng = jax.random.split(rng)
        updated_dp_state, dp_metrics = self.dp_state.apply_dp_gradients(grads, dp_rng)
        
        # Combine metrics
        step_metrics = {'loss': loss}
        step_metrics.update(dp_metrics)
        
        return updated_dp_state, step_metrics


def start_client(client_id: int, server_address: str = "localhost:8080") -> None:
    """
    Start a federated learning client.
    
    Args:
        client_id: Client identifier
        server_address: Server address and port
    """
    # Load configuration
    config = get_config()
    
    # Create client
    client = FedSenseClient(client_id=client_id, config=config)
    
    logger.info(f"Starting client {client_id}, connecting to {server_address}")
    
    # Start client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
        grpc_max_message_length=1024*1024*1024  # 1GB message limit
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FedSense Federated Client")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--server", type=str, default="localhost:8080", help="Server address")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - Client-{args.client_id} - %(levelname)s - %(message)s"
    )
    
    start_client(args.client_id, args.server)
