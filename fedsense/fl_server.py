"""
Flower federated learning server implementation.
Orchestrates federated training with FedAvg and MLflow logging.
"""

import flwr as fl
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path

from .config import get_config
from .datasets import generate_synthetic_data, create_federated_splits, save_client_data
from .model_jax import create_train_state, evaluate_model, save_model_params
from .utils_logging import setup_mlflow, log_server_metrics, log_federated_results
from .eval import evaluate_federated_model

logger = logging.getLogger(__name__)


class FedAvgStrategy(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy with MLflow logging."""
    
    def __init__(self,
                 config: Any,
                 global_test_data: Optional[Any] = None,
                 **kwargs):
        """
        Initialize custom FedAvg strategy.
        
        Args:
            config: FedSense configuration
            global_test_data: Global test dataset for evaluation
            **kwargs: Additional arguments for base FedAvg
        """
        super().__init__(**kwargs)
        self.config = config
        self.global_test_data = global_test_data
        self.round_num = 0
        self.best_auroc = 0.0
        self.best_round = 0
        
        # Initialize global model for evaluation
        import jax
        rng = jax.random.PRNGKey(config.random_seed)
        input_shape = (config.window_len, 4)
        
        self.global_state = create_train_state(
            rng=rng,
            input_shape=input_shape,
            learning_rate=config.learning_rate,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate
        )
        
        logger.info(f"Initialized FedAvg strategy with {config.min_fit_clients} min fit clients")
    
    def aggregate_fit(self,
                     server_round: int,
                     results: List[Tuple[fl.client.Client, fl.common.FitRes]],
                     failures: List[Union[Tuple[fl.client.Client, fl.common.FitRes], BaseException]],
                     ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
        """
        Aggregate training results from clients.
        
        Args:
            server_round: Current round number
            results: Training results from clients
            failures: Failed client results
            
        Returns:
            Aggregated parameters and metrics
        """
        self.round_num = server_round
        
        logger.info(f"Round {server_round}: Aggregating {len(results)} client results")
        
        # Call parent aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Collect client metrics
        client_metrics = {}
        total_examples = 0
        
        for client, fit_res in results:
            client_metrics[f"client_{client.cid}"] = fit_res.metrics
            total_examples += fit_res.num_examples
        
        # Log round metrics
        round_metrics = {
            'round': server_round,
            'participating_clients': len(results),
            'failed_clients': len(failures),
            'total_examples': total_examples,
        }
        
        # Compute average client metrics
        if results:
            avg_train_loss = np.mean([fit_res.metrics.get('train_loss', 0) for _, fit_res in results])
            avg_grad_norm = np.mean([fit_res.metrics.get('grad_norm', 0) for _, fit_res in results])
            round_metrics.update({
                'avg_train_loss': avg_train_loss,
                'avg_grad_norm': avg_grad_norm
            })
            
            # DP metrics if available
            dp_clients = sum(1 for _, fit_res in results if 'privacy_epsilon' in fit_res.metrics)
            if dp_clients > 0:
                avg_epsilon = np.mean([
                    fit_res.metrics.get('privacy_epsilon', float('inf')) 
                    for _, fit_res in results if 'privacy_epsilon' in fit_res.metrics
                ])
                round_metrics['dp_clients'] = dp_clients
                round_metrics['avg_privacy_epsilon'] = avg_epsilon
        
        # Log to MLflow
        log_server_metrics(round_metrics, server_round)
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self,
                          server_round: int,
                          results: List[Tuple[fl.client.Client, fl.common.EvaluateRes]],
                          failures: List[Union[Tuple[fl.client.Client, fl.common.EvaluateRes], BaseException]],
                          ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Aggregate evaluation results from clients.
        
        Args:
            server_round: Current round number
            results: Evaluation results from clients
            failures: Failed client evaluations
            
        Returns:
            Aggregated loss and metrics
        """
        logger.info(f"Round {server_round}: Aggregating {len(results)} evaluation results")
        
        # Call parent aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Collect evaluation metrics
        if results:
            client_losses = [eval_res.loss for _, eval_res in results]
            client_aurocs = [eval_res.metrics.get('auroc', 0) for _, eval_res in results]
            client_f1s = [eval_res.metrics.get('f1', 0) for _, eval_res in results]
            
            eval_metrics = {
                'round': server_round,
                'avg_val_loss': np.mean(client_losses),
                'std_val_loss': np.std(client_losses),
                'avg_val_auroc': np.mean(client_aurocs),
                'std_val_auroc': np.std(client_aurocs),
                'avg_val_f1': np.mean(client_f1s),
                'std_val_f1': np.std(client_f1s),
                'evaluating_clients': len(results)
            }
            
            # Global model evaluation on test set
            if self.global_test_data is not None:
                # TODO: Update global state with aggregated parameters
                # global_metrics = evaluate_model(self.global_state, self.global_test_data)
                # eval_metrics.update({f'global_{k}': v for k, v in global_metrics.items()})
                pass
            
            # Track best model
            current_auroc = eval_metrics['avg_val_auroc']
            if current_auroc > self.best_auroc:
                self.best_auroc = current_auroc
                self.best_round = server_round
                
                # Save best model parameters
                if hasattr(self, 'global_state'):
                    save_path = Path(f"best_global_model_round_{server_round}.npz")
                    save_model_params(self.global_state.params, str(save_path))
            
            eval_metrics.update({
                'best_auroc': self.best_auroc,
                'best_round': self.best_round
            })
            
            # Log evaluation metrics
            log_server_metrics(eval_metrics, server_round, prefix='eval_')
            
            logger.info(f"Round {server_round} evaluation: "
                       f"AUROC={current_auroc:.4f} (best: {self.best_auroc:.4f} @ round {self.best_round})")
        
        return aggregated_loss, aggregated_metrics


def create_federated_data(config: Any) -> Dict[int, Any]:
    """
    Create federated data splits for all clients.
    
    Args:
        config: FedSense configuration
        
    Returns:
        Dictionary mapping client_id to data
    """
    logger.info("Generating synthetic federated data")
    
    # Generate base dataset
    base_data = generate_synthetic_data(
        n_samples=50000,  # Large dataset for splitting
        fs=50.0,
        window_len=config.window_len,
        anomaly_rate=0.15,
        random_seed=config.random_seed
    )
    
    # Create federated splits
    data_splits = create_federated_splits(
        df=base_data,
        n_clients=config.n_clients,
        alpha=0.1,  # Non-IID parameter
        random_seed=config.random_seed
    )
    
    # Save to disk
    save_client_data(data_splits, config)
    
    return data_splits


def start_server(config: Optional[Any] = None) -> None:
    """
    Start the federated learning server.
    
    Args:
        config: FedSense configuration
    """
    if config is None:
        config = get_config()
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Create federated data
    data_splits = create_federated_data(config)
    
    # Create strategy
    strategy = FedAvgStrategy(
        config=config,
        min_fit_clients=config.min_fit_clients,
        min_eval_clients=config.min_eval_clients,
        min_available_clients=config.min_fit_clients,
        evaluate_metrics_aggregation_fn=lambda metrics: {
            "avg_auroc": np.mean([m["auroc"] for _, m in metrics]),
            "avg_f1": np.mean([m["f1"] for _, m in metrics])
        },
        fit_metrics_aggregation_fn=lambda metrics: {
            "avg_train_loss": np.mean([m["train_loss"] for _, m in metrics]),
            "avg_grad_norm": np.mean([m["grad_norm"] for _, m in metrics])
        }
    )
    
    # Start server
    logger.info(f"Starting FL server for {config.rounds} rounds with {config.n_clients} clients")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config.rounds),
        strategy=strategy,
        grpc_max_message_length=1024*1024*1024  # 1GB message limit
    )
    
    # Final evaluation and model export
    logger.info("Federated training completed")
    
    # Log final results
    log_federated_results({
        'total_rounds': config.rounds,
        'participating_clients': config.n_clients,
        'best_auroc': strategy.best_auroc,
        'best_round': strategy.best_round,
        'dp_enabled': config.use_dp,
    })


def run_federated_simulation(config: Optional[Any] = None) -> Dict[str, Any]:
    """
    Run federated learning simulation (all clients in one process).
    Useful for development and testing.
    
    Args:
        config: FedSense configuration
        
    Returns:
        Simulation results
    """
    if config is None:
        config = get_config()
    
    logger.info("Starting federated simulation")
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Create data
    data_splits = create_federated_data(config)
    
    # Import client class
    from .fl_client import FedSenseClient
    
    # Create client function
    def client_fn(cid: str) -> FedSenseClient:
        return FedSenseClient(
            client_id=int(cid),
            config=config,
            data_splits=data_splits
        )
    
    # Create strategy
    strategy = FedAvgStrategy(
        config=config,
        min_fit_clients=config.min_fit_clients,
        min_eval_clients=config.min_eval_clients,
        min_available_clients=config.min_fit_clients
    )
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.n_clients,
        config=fl.server.ServerConfig(num_rounds=config.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )
    
    logger.info("Federated simulation completed")
    
    # Extract results
    results = {
        'rounds_completed': len(history.losses_distributed),
        'final_loss': history.losses_distributed[-1][1] if history.losses_distributed else None,
        'metrics_history': history.metrics_distributed,
        'best_auroc': strategy.best_auroc,
        'best_round': strategy.best_round
    }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FedSense Federated Server")
    parser.add_argument("--simulate", action="store_true", help="Run simulation instead of real server")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--clients", type=int, default=8, help="Number of clients")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - Server - %(levelname)s - %(message)s"
    )
    
    # Load config with overrides
    config = get_config()
    config.rounds = args.rounds
    config.n_clients = args.clients
    
    if args.simulate:
        results = run_federated_simulation(config)
        logger.info(f"Simulation results: {results}")
    else:
        start_server(config)
