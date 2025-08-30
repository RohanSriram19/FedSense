"""
Local training baseline for FedSense anomaly detection.
Trains a single model on pooled data for comparison with federated approaches.
"""

import jax
import jax.numpy as jnp
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import time

from .config import get_config
from .datasets import (
    generate_synthetic_data, create_federated_splits, FedSenseDataset,
    save_client_data
)
from .features import make_windows, standardize_features, train_val_test_split
from .model_jax import (
    create_train_state, train_step, evaluate_model, 
    save_model_params, count_parameters
)
from .utils_logging import (
    setup_mlflow, start_federated_run, log_server_metrics, 
    create_training_plots, log_evaluation_results
)
from .eval import evaluate_model_comprehensive

logger = logging.getLogger(__name__)


def prepare_pooled_data(config: Any) -> Tuple[FedSenseDataset, FedSenseDataset, FedSenseDataset]:
    """
    Generate and prepare pooled dataset for centralized training.
    
    Args:
        config: FedSense configuration
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    logger.info("Generating synthetic data for centralized training")
    
    # Generate larger dataset for centralized training
    synthetic_df = generate_synthetic_data(
        n_samples=50000,
        fs=50.0,
        window_len=config.window_len,
        anomaly_rate=0.15,
        random_seed=config.random_seed
    )
    
    # Create windows
    X, y = make_windows(synthetic_df, config.window_len, config.stride)
    logger.info(f"Generated {len(X)} windows from {len(synthetic_df)} samples")
    
    # Train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_ratio=0.7, val_ratio=0.15, random_seed=config.random_seed
    )
    
    # Standardize features (fit on train only)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_features(
        X_train, X_val, X_test
    )
    
    # Create datasets
    train_dataset = FedSenseDataset(X_train_scaled, y_train)
    val_dataset = FedSenseDataset(X_val_scaled, y_val)
    test_dataset = FedSenseDataset(X_test_scaled, y_test)
    
    logger.info(f"Dataset splits - Train: {len(train_dataset)}, "
               f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def train_epoch(
    state: Any,
    dataset: FedSenseDataset,
    batch_size: int,
    rng: jax.random.PRNGKey,
    epoch: int
) -> Tuple[Any, Dict[str, float]]:
    """
    Train for one epoch.
    
    Args:
        state: Training state
        dataset: Training dataset
        batch_size: Batch size
        rng: Random key
        epoch: Current epoch number
        
    Returns:
        Updated state and metrics
    """
    epoch_metrics = {
        'train_loss': 0.0,
        'n_batches': 0
    }
    
    batch_count = 0
    for batch in dataset.batch_iterator(batch_size, shuffle=True):
        rng, step_rng = jax.random.split(rng)
        state, step_metrics = train_step(state, batch, step_rng)
        
        epoch_metrics['train_loss'] += step_metrics['loss']
        batch_count += 1
    
    # Average metrics
    if batch_count > 0:
        epoch_metrics['train_loss'] /= batch_count
        epoch_metrics['n_batches'] = batch_count
    
    return state, epoch_metrics


def run_local_training(config: Any = None) -> Dict[str, Any]:
    """
    Run local (centralized) training baseline.
    
    Args:
        config: FedSense configuration
        
    Returns:
        Training results and metrics
    """
    if config is None:
        config = get_config()
    
    logger.info("Starting centralized training baseline")
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Start MLflow run
    run_name = "centralized_baseline"
    run_id = start_federated_run(config, run_name)
    
    # Prepare data
    train_dataset, val_dataset, test_dataset = prepare_pooled_data(config)
    
    # Initialize model
    rng = jax.random.PRNGKey(config.random_seed)
    input_shape = (config.window_len, 4)  # HR + 3 accel channels
    
    state = create_train_state(
        rng=rng,
        input_shape=input_shape,
        learning_rate=config.learning_rate,
        hidden_dims=config.hidden_dims,
        dropout_rate=config.dropout_rate
    )
    
    param_count = count_parameters(state.params)
    logger.info(f"Training model with {param_count} parameters")
    
    # Training loop
    n_epochs = config.rounds * config.local_epochs  # Match federated training compute
    best_val_auroc = 0.0
    best_epoch = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_auroc': [],
        'val_f1': []
    }
    
    total_start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        
        # Train one epoch
        rng, epoch_rng = jax.random.split(rng)
        state, train_metrics = train_epoch(
            state, train_dataset, config.batch_size, epoch_rng, epoch
        )
        
        # Validation
        val_metrics = evaluate_model(state, val_dataset, config.batch_size)
        
        # Track metrics
        training_history['train_loss'].append(train_metrics['train_loss'])
        training_history['val_loss'].append(val_metrics['loss'])
        training_history['val_auroc'].append(val_metrics['auroc'])
        training_history['val_f1'].append(val_metrics['f1'])
        
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['train_loss'],
            'val_loss': val_metrics['loss'],
            'val_auroc': val_metrics['auroc'],
            'val_f1': val_metrics['f1'],
            'val_precision': val_metrics.get('precision', 0.0),
            'val_recall': val_metrics.get('recall', 0.0),
            'epoch_time': epoch_time,
            'best_threshold': val_metrics.get('best_threshold', 0.5)
        }
        
        log_server_metrics(epoch_metrics, epoch + 1)
        
        # Check for best model
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            best_epoch = epoch + 1
            
            # Save best model
            best_model_path = f"best_centralized_model_epoch_{epoch+1}.npz"
            save_model_params(state.params, best_model_path)
        
        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}/{n_epochs}: "
                       f"Train Loss: {train_metrics['train_loss']:.4f}, "
                       f"Val AUROC: {val_metrics['auroc']:.4f}, "
                       f"Val F1: {val_metrics['f1']:.4f}, "
                       f"Time: {epoch_time:.1f}s")
    
    total_training_time = time.time() - total_start_time
    
    # Final evaluation on test set
    logger.info("Evaluating on test set")
    test_results = evaluate_model_comprehensive(
        model_state=state,
        dataset=test_dataset,
        model_name="centralized",
        batch_size=config.batch_size
    )
    
    # Log final results
    final_metrics = {
        'final_test_auroc': test_results['metrics']['auroc'],
        'final_test_f1': test_results['metrics']['f1'],
        'final_test_precision': test_results['metrics']['precision'],
        'final_test_recall': test_results['metrics']['recall'],
        'final_test_auprc': test_results['metrics']['auprc'],
        'best_val_auroc': best_val_auroc,
        'best_epoch': best_epoch,
        'total_epochs': n_epochs,
        'total_training_time': total_training_time,
        'avg_epoch_time': total_training_time / n_epochs,
        'model_parameters': param_count,
        'training_samples': len(train_dataset),
        'test_samples': len(test_dataset)
    }
    
    log_server_metrics(final_metrics, n_epochs)
    
    # Log evaluation results with plots
    log_evaluation_results(
        metrics=test_results['metrics'],
        y_true=test_results['predictions']['y_true'],
        y_scores=test_results['predictions']['y_scores']
    )
    
    # Create and log training plots
    plot_dir = Path("training_plots")
    plot_files = create_training_plots(training_history, plot_dir)
    
    # Log plots as artifacts
    import mlflow
    for plot_file in plot_files:
        mlflow.log_artifact(plot_file, "training_plots")
    
    # Cleanup temp plots
    import shutil
    shutil.rmtree(plot_dir, ignore_errors=True)
    
    logger.info(f"Centralized training completed in {total_training_time:.1f}s")
    logger.info(f"Best validation AUROC: {best_val_auroc:.4f} at epoch {best_epoch}")
    logger.info(f"Final test AUROC: {test_results['metrics']['auroc']:.4f}")
    
    return {
        'final_metrics': final_metrics,
        'training_history': training_history,
        'test_results': test_results,
        'best_model_path': f"best_centralized_model_epoch_{best_epoch}.npz"
    }


def main():
    """Main function for local training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FedSense Centralized Training")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load config with overrides
    config = get_config()
    
    if args.epochs:
        # Override total epochs (adjust rounds to match)
        config.rounds = max(1, args.epochs // config.local_epochs)
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.seed:
        config.random_seed = args.seed
    
    logger.info(f"Config: {config}")
    
    # Run training
    results = run_local_training(config)
    
    logger.info("Training completed successfully!")
    logger.info(f"Results summary: {results['final_metrics']}")


if __name__ == "__main__":
    main()
