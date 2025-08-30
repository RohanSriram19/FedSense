"""
MLflow utilities for experiment tracking and logging.
Provides structured logging for federated learning experiments.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_mlflow(config: Any) -> None:
    """
    Setup MLflow tracking and create experiment.
    
    Args:
        config: FedSense configuration
    """
    # Set tracking URI
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    
    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(
            name=config.experiment_name,
            tags={
                "project": "fedsense",
                "type": "federated_learning",
                "created_at": datetime.now().isoformat()
            }
        )
        logger.info(f"Created new MLflow experiment: {config.experiment_name}")
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(config.experiment_name)
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment: {config.experiment_name}")
    
    mlflow.set_experiment(experiment_id=experiment_id)


def start_federated_run(config: Any, run_name: Optional[str] = None) -> str:
    """
    Start a new MLflow run for federated learning.
    
    Args:
        config: FedSense configuration
        run_name: Optional run name
        
    Returns:
        MLflow run ID
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dp_suffix = "_dp" if config.use_dp else ""
        run_name = f"federated_{config.n_clients}clients_{config.rounds}rounds{dp_suffix}_{timestamp}"
    
    run = mlflow.start_run(run_name=run_name)
    
    # Log configuration parameters
    config_dict = {
        # Data parameters
        "window_len": config.window_len,
        "stride": config.stride,
        "use_fft": config.use_fft,
        
        # Model parameters
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "hidden_dims": str(config.hidden_dims),
        "dropout_rate": config.dropout_rate,
        
        # Federated learning parameters
        "n_clients": config.n_clients,
        "rounds": config.rounds,
        "local_epochs": config.local_epochs,
        "min_fit_clients": config.min_fit_clients,
        "min_eval_clients": config.min_eval_clients,
        
        # Differential privacy parameters
        "use_dp": config.use_dp,
        "clip_norm": config.clip_norm,
        "noise_multiplier": config.noise_multiplier,
        "dp_epsilon": config.dp_epsilon,
        "dp_delta": config.dp_delta,
        
        # Other parameters
        "random_seed": config.random_seed,
    }
    
    mlflow.log_params(config_dict)
    
    # Add tags
    mlflow.set_tags({
        "model_type": "jax_flax_cnn",
        "federated_learning": "flower",
        "differential_privacy": str(config.use_dp),
        "data_type": "synthetic_wearables"
    })
    
    logger.info(f"Started MLflow run: {run_name} ({run.info.run_id})")
    return run.info.run_id


def log_server_metrics(metrics: Dict[str, Any], round_num: int, prefix: str = "") -> None:
    """
    Log server-side metrics to MLflow.
    
    Args:
        metrics: Dictionary of metrics to log
        round_num: Current round number
        prefix: Optional prefix for metric names
    """
    with mlflow.start_run():
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_name = f"{prefix}{key}" if prefix else key
                mlflow.log_metric(metric_name, value, step=round_num)


def log_client_metrics(client_id: int, metrics: Dict[str, Any], round_num: int) -> None:
    """
    Log client-side metrics to MLflow.
    
    Args:
        client_id: Client identifier
        metrics: Dictionary of metrics to log
        round_num: Current round number
    """
    # Note: In a real federated setting, client metrics might be logged
    # to a separate MLflow instance or aggregated by the server
    
    with mlflow.start_run():
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_name = f"client_{client_id}_{key}"
                mlflow.log_metric(metric_name, value, step=round_num)


def log_model_artifacts(model_params: Dict[str, Any], 
                       model_path: str,
                       round_num: Optional[int] = None) -> None:
    """
    Log model parameters and artifacts to MLflow.
    
    Args:
        model_params: Model parameters
        model_path: Path to saved model file
        round_num: Optional round number
    """
    with mlflow.start_run():
        # Log model file
        mlflow.log_artifact(model_path, "models")
        
        # Log model metadata
        model_info = {
            "parameter_count": sum(np.prod(p.shape) for p in model_params.values() if hasattr(p, 'shape')),
            "model_type": "jax_flax_cnn",
            "saved_at": datetime.now().isoformat()
        }
        
        if round_num is not None:
            model_info["round"] = round_num
            
        mlflow.log_dict(model_info, "model_info.json")


def create_training_plots(history: Dict[str, List[float]], 
                         output_dir: Path) -> List[str]:
    """
    Create training progress plots.
    
    Args:
        history: Dictionary of training metrics over time
        output_dir: Directory to save plots
        
    Returns:
        List of plot file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_files = []
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Training loss plot
    if 'train_loss' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], 'b-', linewidth=2, label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Federated Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        loss_plot = output_dir / 'training_loss.png'
        plt.savefig(loss_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(loss_plot))
    
    # AUROC plot
    if 'auroc' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['auroc'], 'g-', linewidth=2, label='AUROC')
        plt.xlabel('Round')
        plt.ylabel('AUROC')
        plt.title('Federated Training AUROC')
        plt.ylim(0.5, 1.0)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        auroc_plot = output_dir / 'training_auroc.png'
        plt.savefig(auroc_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(auroc_plot))
    
    # Privacy epsilon plot (if DP enabled)
    if 'privacy_epsilon' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['privacy_epsilon'], 'purple', linewidth=2, label='Privacy ε')
        plt.xlabel('Round')
        plt.ylabel('Epsilon (ε)')
        plt.title('Privacy Budget Consumption')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        privacy_plot = output_dir / 'privacy_epsilon.png'
        plt.savefig(privacy_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(privacy_plot))
    
    return plot_files


def create_evaluation_plots(y_true: np.ndarray, 
                          y_scores: np.ndarray,
                          output_dir: Path) -> List[str]:
    """
    Create evaluation plots (ROC curve, PR curve, etc.).
    
    Args:
        y_true: True labels
        y_scores: Predicted scores
        output_dir: Directory to save plots
        
    Returns:
        List of plot file paths
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_files = []
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    roc_plot = output_dir / 'roc_curve.png'
    plt.savefig(roc_plot, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(str(roc_plot))
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'r-', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    pr_plot = output_dir / 'pr_curve.png'
    plt.savefig(pr_plot, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(str(pr_plot))
    
    return plot_files


def log_evaluation_results(metrics: Dict[str, float],
                          y_true: Optional[np.ndarray] = None,
                          y_scores: Optional[np.ndarray] = None) -> None:
    """
    Log evaluation results and plots to MLflow.
    
    Args:
        metrics: Dictionary of evaluation metrics
        y_true: True labels (for plotting)
        y_scores: Predicted scores (for plotting)
    """
    with mlflow.start_run():
        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"final_{key}", value)
        
        # Create and log plots
        if y_true is not None and y_scores is not None:
            plot_dir = Path("temp_plots")
            plot_files = create_evaluation_plots(y_true, y_scores, plot_dir)
            
            for plot_file in plot_files:
                mlflow.log_artifact(plot_file, "evaluation_plots")
            
            # Cleanup temp plots
            import shutil
            shutil.rmtree(plot_dir, ignore_errors=True)


def log_federated_results(summary: Dict[str, Any]) -> None:
    """
    Log final federated learning results summary.
    
    Args:
        summary: Summary of federated learning results
    """
    with mlflow.start_run():
        # Log summary metrics
        mlflow.log_metrics({
            "final_best_auroc": summary.get('best_auroc', 0.0),
            "final_best_round": summary.get('best_round', 0),
            "total_rounds": summary.get('total_rounds', 0),
            "participating_clients": summary.get('participating_clients', 0)
        })
        
        # Log summary as artifact
        mlflow.log_dict(summary, "federated_summary.json")
        
        logger.info(f"Logged federated learning summary: {summary}")


def get_best_run(experiment_name: str, metric_name: str = "final_best_auroc") -> Optional[Dict[str, Any]]:
    """
    Get the best run from an experiment based on a metric.
    
    Args:
        experiment_name: Name of the MLflow experiment
        metric_name: Metric to optimize
        
    Returns:
        Dictionary with best run information
    """
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return None
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1
        )
        
        if not runs:
            logger.warning(f"No runs found in experiment '{experiment_name}'")
            return None
        
        best_run = runs[0]
        
        return {
            "run_id": best_run.info.run_id,
            "metrics": best_run.data.metrics,
            "params": best_run.data.params,
            "tags": best_run.data.tags,
            "start_time": best_run.info.start_time,
            "end_time": best_run.info.end_time,
        }
        
    except Exception as e:
        logger.error(f"Error retrieving best run: {e}")
        return None
