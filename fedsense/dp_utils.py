"""
Differential privacy utilities for federated learning.
Implements per-sample gradient clipping, Gaussian noise addition,
and privacy accounting.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)


class PrivacyAccountant:
    """Simple privacy accountant using moments accountant."""
    
    def __init__(self, 
                 n_samples: int,
                 noise_multiplier: float,
                 sample_rate: float = 1.0,
                 delta: float = 1e-5):
        """
        Initialize privacy accountant.
        
        Args:
            n_samples: Number of samples in the dataset
            noise_multiplier: Gaussian noise multiplier (sigma)
            sample_rate: Sampling rate for each round
            delta: Target delta for (epsilon, delta)-DP
        """
        self.n_samples = n_samples
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self.steps = 0
        
    def step(self) -> None:
        """Record one DP step (e.g., one training round)."""
        self.steps += 1
        
    def get_epsilon(self) -> float:
        """
        Compute current privacy epsilon using RDP conversion.
        This is a simplified implementation. For production use,
        consider using libraries like tensorflow-privacy or opacus.
        
        Returns:
            Current epsilon value
        """
        if self.noise_multiplier == 0 or self.steps == 0:
            return float('inf')
        
        # Simplified RDP-based epsilon computation
        # Based on: https://arxiv.org/abs/1607.00133
        
        # RDP parameters
        orders = np.arange(2, 64)  # Range of alpha values
        
        # Compute RDP for each order
        rdp_values = []
        for alpha in orders:
            if self.sample_rate == 1.0:
                # Full batch case
                rdp = alpha * self.steps / (2 * self.noise_multiplier ** 2)
            else:
                # Subsampling case (more complex formula)
                rdp = (alpha * self.sample_rate ** 2 * self.steps) / (2 * self.noise_multiplier ** 2)
            rdp_values.append(rdp)
        
        rdp_values = np.array(rdp_values)
        
        # Convert RDP to (epsilon, delta)-DP
        epsilons = rdp_values - np.log(self.delta) / (orders - 1)
        
        return float(np.min(epsilons))
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get current privacy budget spent.
        
        Returns:
            Tuple of (epsilon, delta)
        """
        epsilon = self.get_epsilon()
        return epsilon, self.delta


def clip_gradients_by_norm(
    grads: Dict[str, Any],
    clip_norm: float = 1.0
) -> Tuple[Dict[str, Any], float]:
    """
    Clip gradients by their global L2 norm.
    
    Args:
        grads: Gradient dictionary
        clip_norm: Maximum allowed L2 norm
        
    Returns:
        Tuple of (clipped_gradients, original_norm)
    """
    # Flatten all gradients
    flat_grads = jax.tree_leaves(grads)
    flat_grads_concat = jnp.concatenate([g.flatten() for g in flat_grads])
    
    # Compute global L2 norm
    global_norm = jnp.linalg.norm(flat_grads_concat)
    
    # Clip if necessary
    clip_factor = jnp.minimum(1.0, clip_norm / (global_norm + 1e-8))
    clipped_grads = jax.tree_map(lambda g: g * clip_factor, grads)
    
    return clipped_grads, float(global_norm)


def add_gaussian_noise(
    grads: Dict[str, Any],
    noise_multiplier: float,
    clip_norm: float,
    rng: jax.random.PRNGKey
) -> Dict[str, Any]:
    """
    Add Gaussian noise to gradients for differential privacy.
    
    Args:
        grads: Gradient dictionary
        noise_multiplier: Noise multiplier (sigma)
        clip_norm: Clipping norm (used to scale noise)
        rng: Random key for noise generation
        
    Returns:
        Noisy gradients
    """
    def add_noise_to_tensor(grad, key):
        noise_std = noise_multiplier * clip_norm
        noise = jax.random.normal(key, grad.shape) * noise_std
        return grad + noise
    
    # Generate different random keys for each gradient tensor
    keys = jax.random.split(rng, len(jax.tree_leaves(grads)))
    key_tree = jax.tree_unflatten(jax.tree_structure(grads), keys)
    
    noisy_grads = jax.tree_map(add_noise_to_tensor, grads, key_tree)
    return noisy_grads


def apply_dp_to_gradients(
    grads: Dict[str, Any],
    clip_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    rng: Optional[jax.random.PRNGKey] = None
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Apply differential privacy (clipping + noise) to gradients.
    
    Args:
        grads: Gradient dictionary
        clip_norm: L2 norm clipping threshold
        noise_multiplier: Noise multiplier for Gaussian noise
        rng: Random key for noise generation
        
    Returns:
        Tuple of (dp_gradients, dp_metrics)
    """
    # Clip gradients
    clipped_grads, original_norm = clip_gradients_by_norm(grads, clip_norm)
    
    dp_metrics = {
        'original_grad_norm': original_norm,
        'clip_norm': clip_norm,
        'noise_multiplier': noise_multiplier,
        'clipped': original_norm > clip_norm
    }
    
    # Add noise if specified
    if noise_multiplier > 0 and rng is not None:
        noisy_grads = add_gaussian_noise(clipped_grads, noise_multiplier, clip_norm, rng)
        return noisy_grads, dp_metrics
    
    return clipped_grads, dp_metrics


class DPTrainingState:
    """Enhanced training state with differential privacy tracking."""
    
    def __init__(self,
                 base_state: Any,  # JAX TrainState
                 privacy_accountant: PrivacyAccountant,
                 clip_norm: float = 1.0,
                 noise_multiplier: float = 1.0):
        """
        Initialize DP training state.
        
        Args:
            base_state: Base JAX training state
            privacy_accountant: Privacy accountant instance
            clip_norm: Gradient clipping norm
            noise_multiplier: Noise multiplier
        """
        self.base_state = base_state
        self.privacy_accountant = privacy_accountant
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        
    def apply_dp_gradients(self, 
                          grads: Dict[str, Any], 
                          rng: jax.random.PRNGKey) -> Tuple[Any, Dict[str, float]]:
        """
        Apply DP to gradients and update state.
        
        Args:
            grads: Raw gradients
            rng: Random key for noise
            
        Returns:
            Tuple of (updated_state, dp_metrics)
        """
        # Apply differential privacy
        dp_grads, dp_metrics = apply_dp_to_gradients(
            grads, self.clip_norm, self.noise_multiplier, rng
        )
        
        # Update base state
        updated_base_state = self.base_state.apply_gradients(grads=dp_grads)
        
        # Update privacy accounting
        self.privacy_accountant.step()
        
        # Create new DP state
        updated_state = DPTrainingState(
            base_state=updated_base_state,
            privacy_accountant=self.privacy_accountant,
            clip_norm=self.clip_norm,
            noise_multiplier=self.noise_multiplier
        )
        
        # Add privacy metrics
        epsilon, delta = self.privacy_accountant.get_privacy_spent()
        dp_metrics.update({
            'privacy_epsilon': epsilon,
            'privacy_delta': delta,
            'privacy_steps': self.privacy_accountant.steps
        })
        
        return updated_state, dp_metrics


def compute_dp_noise_scale(
    target_epsilon: float,
    target_delta: float,
    n_samples: int,
    n_rounds: int,
    sample_rate: float = 1.0
) -> float:
    """
    Compute required noise multiplier for target privacy parameters.
    This is a simplified calculation. For production use, consider
    more sophisticated privacy analysis tools.
    
    Args:
        target_epsilon: Target epsilon
        target_delta: Target delta
        n_samples: Number of samples
        n_rounds: Number of training rounds
        sample_rate: Sampling rate per round
        
    Returns:
        Required noise multiplier
    """
    if target_epsilon <= 0 or target_epsilon >= 10:
        logger.warning(f"Target epsilon {target_epsilon} may be too small or large")
    
    # Simplified calculation based on composition theorem
    # For more accurate calculation, use tools like tensorflow-privacy
    
    # Per-round epsilon for composition
    per_round_epsilon = target_epsilon / n_rounds
    
    # Approximate noise multiplier (this is very simplified)
    # Real implementation should use RDP or other advanced techniques
    if per_round_epsilon > 0:
        noise_multiplier = math.sqrt(2 * math.log(1.25 / target_delta)) / per_round_epsilon
    else:
        noise_multiplier = 10.0  # Large noise for very small epsilon
    
    logger.info(f"Computed noise multiplier: {noise_multiplier:.3f} for "
               f"ε={target_epsilon}, δ={target_delta}, rounds={n_rounds}")
    
    return max(0.1, min(10.0, noise_multiplier))  # Reasonable bounds


def validate_dp_parameters(
    epsilon: float,
    delta: float,
    n_samples: int,
    clip_norm: float,
    noise_multiplier: float
) -> bool:
    """
    Validate differential privacy parameters.
    
    Args:
        epsilon: Privacy epsilon
        delta: Privacy delta
        n_samples: Number of samples
        clip_norm: Gradient clipping norm
        noise_multiplier: Noise multiplier
        
    Returns:
        True if parameters are valid
    """
    issues = []
    
    if epsilon <= 0:
        issues.append("Epsilon must be positive")
    elif epsilon > 10:
        issues.append("Epsilon is very large (weak privacy)")
    
    if delta <= 0:
        issues.append("Delta must be positive")
    elif delta >= 1.0 / n_samples:
        issues.append("Delta should be much smaller than 1/n")
    
    if clip_norm <= 0:
        issues.append("Clip norm must be positive")
    
    if noise_multiplier < 0:
        issues.append("Noise multiplier must be non-negative")
    elif noise_multiplier == 0:
        issues.append("Zero noise provides no privacy protection")
    
    if issues:
        for issue in issues:
            logger.warning(f"DP parameter issue: {issue}")
        return False
    
    return True
