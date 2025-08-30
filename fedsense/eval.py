"""
Evaluation utilities for comparing centralized, federated, and DP models.
Provides comprehensive metrics and visualization capabilities.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_recall_fscore_support, confusion_matrix,
    roc_curve, precision_recall_curve
)
from typing import Dict, Any, Tuple, List, Optional
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import time

from .model_jax import evaluate_model
from .datasets import FedSenseDataset
from .utils_logging import log_evaluation_results, create_evaluation_plots

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray, 
    y_scores: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    
    # Threshold-based metrics
    y_pred = (y_scores >= threshold).astype(int)
    
    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = recall  # Same as recall
    balanced_accuracy = (sensitivity + specificity) / 2
    
    metrics = {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'sensitivity': float(sensitivity),
        'balanced_accuracy': float(balanced_accuracy),
        'threshold': float(threshold),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'n_samples': len(y_true),
        'anomaly_rate': float(np.mean(y_true))
    }
    
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold based on a metric.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores
        metric: Metric to optimize ('f1', 'balanced_accuracy', 'youden')
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_value = 0.0
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        if metric == 'f1':
            if np.sum(y_pred) > 0 and np.sum(y_true) > 0:
                value = f1_score(y_true, y_pred)
            else:
                value = 0.0
        elif metric == 'balanced_accuracy':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            value = (sensitivity + specificity) / 2
        elif metric == 'youden':
            # Youden's J statistic = Sensitivity + Specificity - 1
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            value = sensitivity + specificity - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if value > best_value:
            best_value = value
            best_threshold = threshold
    
    return best_threshold, best_value


def evaluate_model_comprehensive(
    model_state: Any,
    dataset: FedSenseDataset,
    model_name: str = "model",
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        model_state: Trained model state (JAX)
        dataset: Test dataset
        model_name: Name for logging
        batch_size: Batch size for evaluation
        
    Returns:
        Comprehensive evaluation results
    """
    logger.info(f"Evaluating {model_name} on {len(dataset)} samples")
    
    start_time = time.time()
    
    # Get predictions
    all_probs = []
    all_labels = []
    
    for batch in dataset.batch_iterator(batch_size, shuffle=False):
        from .model_jax import eval_step
        probs, labels, _ = eval_step(model_state, batch)
        all_probs.extend(np.array(probs))
        all_labels.extend(np.array(labels))
    
    y_true = np.array(all_labels)
    y_scores = np.array(all_probs)
    
    inference_time = time.time() - start_time
    
    # Find optimal threshold
    opt_threshold, opt_f1 = find_optimal_threshold(y_true, y_scores, 'f1')
    
    # Compute metrics with optimal threshold
    metrics = compute_classification_metrics(y_true, y_scores, opt_threshold)
    
    # Add timing metrics
    metrics.update({
        'inference_time': inference_time,
        'inference_time_per_sample': inference_time / len(y_true),
        'samples_per_second': len(y_true) / inference_time,
        'model_name': model_name
    })
    
    logger.info(f"{model_name} evaluation: AUROC={metrics['auroc']:.4f}, "
               f"F1={metrics['f1']:.4f}, Threshold={opt_threshold:.3f}")
    
    return {
        'metrics': metrics,
        'predictions': {
            'y_true': y_true,
            'y_scores': y_scores,
            'optimal_threshold': opt_threshold
        }
    }


def compare_models(
    centralized_results: Dict[str, Any],
    federated_results: Dict[str, Any],
    dp_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compare different model training approaches.
    
    Args:
        centralized_results: Results from centralized training
        federated_results: Results from federated training
        dp_results: Optional results from DP federated training
        
    Returns:
        Comparison results
    """
    logger.info("Comparing model performance across training approaches")
    
    comparison = {
        'centralized': centralized_results['metrics'],
        'federated': federated_results['metrics']
    }
    
    if dp_results is not None:
        comparison['federated_dp'] = dp_results['metrics']
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison).round(4)
    
    # Compute relative performance
    centralized_auroc = centralized_results['metrics']['auroc']
    federated_auroc = federated_results['metrics']['auroc']
    
    performance_drop = {
        'federated_vs_centralized': {
            'auroc_drop': centralized_auroc - federated_auroc,
            'auroc_relative_drop': (centralized_auroc - federated_auroc) / centralized_auroc,
        }
    }
    
    if dp_results is not None:
        dp_auroc = dp_results['metrics']['auroc']
        performance_drop['dp_vs_federated'] = {
            'auroc_drop': federated_auroc - dp_auroc,
            'auroc_relative_drop': (federated_auroc - dp_auroc) / federated_auroc,
        }
        performance_drop['dp_vs_centralized'] = {
            'auroc_drop': centralized_auroc - dp_auroc,
            'auroc_relative_drop': (centralized_auroc - dp_auroc) / centralized_auroc,
        }
    
    return {
        'comparison_table': comparison_df,
        'performance_drop': performance_drop,
        'best_model': max(comparison.keys(), key=lambda k: comparison[k]['auroc']),
        'summary': _create_comparison_summary(comparison, performance_drop)
    }


def _create_comparison_summary(comparison: Dict[str, Any], 
                              performance_drop: Dict[str, Any]) -> str:
    """Create a text summary of model comparison results."""
    
    centralized_auroc = comparison['centralized']['auroc']
    federated_auroc = comparison['federated']['auroc']
    
    summary = f"""
Model Performance Comparison:

Centralized Training:
  - AUROC: {centralized_auroc:.4f}
  - F1 Score: {comparison['centralized']['f1']:.4f}
  - Precision: {comparison['centralized']['precision']:.4f}
  - Recall: {comparison['centralized']['recall']:.4f}

Federated Training:  
  - AUROC: {federated_auroc:.4f}
  - F1 Score: {comparison['federated']['f1']:.4f}
  - Precision: {comparison['federated']['precision']:.4f}
  - Recall: {comparison['federated']['recall']:.4f}
  - Performance drop: {performance_drop['federated_vs_centralized']['auroc_relative_drop']:.2%}
"""

    if 'federated_dp' in comparison:
        dp_auroc = comparison['federated_dp']['auroc']
        summary += f"""
Federated Training with Differential Privacy:
  - AUROC: {dp_auroc:.4f}
  - F1 Score: {comparison['federated_dp']['f1']:.4f}
  - Precision: {comparison['federated_dp']['precision']:.4f}
  - Recall: {comparison['federated_dp']['recall']:.4f}
  - Performance drop from federated: {performance_drop['dp_vs_federated']['auroc_relative_drop']:.2%}
  - Performance drop from centralized: {performance_drop['dp_vs_centralized']['auroc_relative_drop']:.2%}
"""

    return summary.strip()


def create_comparison_plots(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> List[str]:
    """
    Create comparison plots for different training approaches.
    
    Args:
        results: Dictionary mapping approach name to results
        output_dir: Directory to save plots
        
    Returns:
        List of created plot files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_files = []
    
    # ROC comparison
    plt.figure(figsize=(10, 8))
    for approach, result in results.items():
        y_true = result['predictions']['y_true']
        y_scores = result['predictions']['y_scores']
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auroc = result['metrics']['auroc']
        
        plt.plot(fpr, tpr, linewidth=2, label=f'{approach.title()} (AUROC = {auroc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    roc_plot = output_dir / 'roc_comparison.png'
    plt.savefig(roc_plot, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(str(roc_plot))
    
    # PR curve comparison  
    plt.figure(figsize=(10, 8))
    for approach, result in results.items():
        y_true = result['predictions']['y_true']
        y_scores = result['predictions']['y_scores']
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auprc = result['metrics']['auprc']
        
        plt.plot(recall, precision, linewidth=2, label=f'{approach.title()} (AUPRC = {auprc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    pr_plot = output_dir / 'pr_comparison.png'
    plt.savefig(pr_plot, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(str(pr_plot))
    
    # Metrics comparison bar chart
    metrics_to_compare = ['auroc', 'auprc', 'f1', 'precision', 'recall']
    approaches = list(results.keys())
    
    fig, axes = plt.subplots(1, len(metrics_to_compare), figsize=(20, 6))
    
    for i, metric in enumerate(metrics_to_compare):
        values = [results[approach]['metrics'][metric] for approach in approaches]
        
        axes[i].bar(approaches, values, alpha=0.7, color=['blue', 'orange', 'green'][:len(approaches)])
        axes[i].set_title(metric.upper())
        axes[i].set_ylim([0, 1])
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    metrics_plot = output_dir / 'metrics_comparison.png'
    plt.savefig(metrics_plot, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(str(metrics_plot))
    
    return plot_files


def evaluate_federated_model(
    global_model_path: str,
    test_dataset: FedSenseDataset,
    config: Any,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluate a federated model and create comprehensive report.
    
    Args:
        global_model_path: Path to saved global model parameters
        test_dataset: Test dataset
        config: FedSense configuration
        output_dir: Optional directory for plots
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating federated model from {global_model_path}")
    
    # Load model
    from .model_jax import create_train_state, load_model_params
    import jax
    
    rng = jax.random.PRNGKey(config.random_seed)
    input_shape = (config.window_len, 4)
    
    model_state = create_train_state(
        rng=rng,
        input_shape=input_shape,
        learning_rate=config.learning_rate,
        hidden_dims=config.hidden_dims,
        dropout_rate=config.dropout_rate
    )
    
    # Load saved parameters
    loaded_params = load_model_params(global_model_path, model_state.params)
    model_state = model_state.replace(params=loaded_params)
    
    # Comprehensive evaluation
    results = evaluate_model_comprehensive(
        model_state=model_state,
        dataset=test_dataset,
        model_name="federated"
    )
    
    # Create plots if output directory provided
    if output_dir:
        plot_files = create_evaluation_plots(
            results['predictions']['y_true'],
            results['predictions']['y_scores'],
            output_dir
        )
        results['plot_files'] = plot_files
    
    return results
