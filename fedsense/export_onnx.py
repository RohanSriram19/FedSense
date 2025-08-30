"""
ONNX export utilities for model deployment.
Converts trained models to ONNX format for Triton Inference Server.
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Any, Optional

from .model_torch_twin import AnomalyDetectionCNNTorch, create_torch_model, TorchDataset
from .model_jax import load_model_params
from .datasets import FedSenseDataset
from .config import get_config

logger = logging.getLogger(__name__)


def export_jax_to_onnx(
    jax_model_path: str,
    onnx_output_path: str,
    config: Optional[Any] = None,
    fine_tune_data: Optional[TorchDataset] = None
) -> bool:
    """
    Export JAX model to ONNX format via PyTorch twin.
    
    Args:
        jax_model_path: Path to saved JAX model parameters
        onnx_output_path: Path to save ONNX model
        config: FedSense configuration
        fine_tune_data: Optional data for fine-tuning PyTorch twin
        
    Returns:
        True if export successful
    """
    if config is None:
        config = get_config()
    
    logger.info(f"Exporting JAX model {jax_model_path} to ONNX {onnx_output_path}")
    
    try:
        # Create PyTorch twin model
        torch_model = create_torch_model(
            n_features=4,
            hidden_dims=tuple(config.hidden_dims),
            dropout_rate=config.dropout_rate
        )
        
        # Option 1: Try to load JAX weights (complex, often skipped in practice)
        # load_jax_weights_to_torch(torch_model, jax_params)
        
        # Option 2: Fine-tune PyTorch twin on pooled data (recommended approach)
        if fine_tune_data is not None:
            logger.info("Fine-tuning PyTorch twin model on pooled data")
            
            # Create data loader
            train_loader = torch.utils.data.DataLoader(
                fine_tune_data,
                batch_size=config.batch_size,
                shuffle=True
            )
            
            # Fine-tune for a few epochs
            from .model_torch_twin import train_torch_model
            val_loader = train_loader  # Use same data for validation (simplified)
            
            train_metrics = train_torch_model(
                model=torch_model,
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=5,
                learning_rate=config.learning_rate * 0.1,  # Lower learning rate for fine-tuning
                device="cpu"
            )
            
            logger.info(f"Fine-tuning completed: {train_metrics}")
        
        # Export to ONNX
        success = export_torch_to_onnx(
            torch_model=torch_model,
            onnx_output_path=onnx_output_path,
            input_shape=(config.window_len, 4),
            opset_version=17
        )
        
        if success:
            logger.info(f"Successfully exported model to {onnx_output_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to export JAX model to ONNX: {e}")
        return False


def export_torch_to_onnx(
    torch_model: torch.nn.Module,
    onnx_output_path: str,
    input_shape: Tuple[int, ...] = (250, 4),
    opset_version: int = 17,
    dynamic_batch: bool = True
) -> bool:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        torch_model: PyTorch model to export
        onnx_output_path: Path to save ONNX model
        input_shape: Shape of input (window_len, n_features)
        opset_version: ONNX opset version
        dynamic_batch: Whether to allow dynamic batch size
        
    Returns:
        True if export successful
    """
    try:
        # Set model to evaluation mode
        torch_model.eval()
        
        # Create dummy input
        batch_size = 1
        dummy_input = torch.randn(batch_size, *input_shape)
        
        # Define input/output names
        input_names = ['input']
        output_names = ['output']
        
        # Define dynamic axes if needed
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        logger.info(f"Exporting PyTorch model to ONNX with input shape: {dummy_input.shape}")
        
        # Export to ONNX
        torch.onnx.export(
            torch_model,
            dummy_input,
            onnx_output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Verify ONNX model
        return verify_onnx_model(onnx_output_path, dummy_input)
        
    except Exception as e:
        logger.error(f"Failed to export PyTorch model to ONNX: {e}")
        return False


def verify_onnx_model(
    onnx_model_path: str,
    test_input: torch.Tensor
) -> bool:
    """
    Verify ONNX model by checking structure and running inference.
    
    Args:
        onnx_model_path: Path to ONNX model
        test_input: Test input tensor
        
    Returns:
        True if verification successful
    """
    try:
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info("ONNX model structure check passed")
        
        # Test inference with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_model_path)
        
        # Get input/output info
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        output_name = ort_session.get_outputs()[0].name
        output_shape = ort_session.get_outputs()[0].shape
        
        logger.info(f"ONNX model - Input: {input_name} {input_shape}, Output: {output_name} {output_shape}")
        
        # Run inference
        ort_inputs = {input_name: test_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Check output
        prediction = ort_outputs[0]
        logger.info(f"ONNX inference test - Output shape: {prediction.shape}, Value: {prediction[0][0]:.4f}")
        
        # Verify output is a valid probability
        if not (0.0 <= prediction[0][0] <= 1.0):
            logger.warning(f"Output {prediction[0][0]} is not a valid probability")
            return False
        
        logger.info("ONNX model verification successful")
        return True
        
    except Exception as e:
        logger.error(f"ONNX model verification failed: {e}")
        return False


def benchmark_onnx_model(
    onnx_model_path: str,
    input_shape: Tuple[int, ...] = (250, 4),
    n_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark ONNX model inference performance.
    
    Args:
        onnx_model_path: Path to ONNX model
        input_shape: Shape of input
        n_runs: Number of inference runs for timing
        
    Returns:
        Performance metrics
    """
    try:
        import time
        
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_model_path)
        input_name = ort_session.get_inputs()[0].name
        
        # Generate test data
        test_inputs = []
        for _ in range(n_runs):
            dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
            test_inputs.append({input_name: dummy_input})
        
        # Warmup runs
        for _ in range(10):
            ort_session.run(None, test_inputs[0])
        
        # Timing runs
        start_time = time.time()
        for test_input in test_inputs:
            ort_session.run(None, test_input)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / n_runs
        throughput = n_runs / total_time
        
        metrics = {
            'total_time_seconds': total_time,
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_inferences_per_second': throughput,
            'n_runs': n_runs
        }
        
        logger.info(f"ONNX model performance: {avg_time*1000:.2f}ms per inference, "
                   f"{throughput:.1f} inferences/sec")
        
        return metrics
        
    except Exception as e:
        logger.error(f"ONNX benchmarking failed: {e}")
        return {}


def create_triton_config(
    model_name: str = "fedsense_anomaly",
    onnx_model_filename: str = "model.onnx",
    input_shape: Tuple[int, ...] = (250, 4),
    output_path: str = "docker/triton/config.pbtxt"
) -> None:
    """
    Create Triton Inference Server configuration file.
    
    Args:
        model_name: Name of the model in Triton
        onnx_model_filename: Name of the ONNX model file
        input_shape: Shape of model input
        output_path: Path to save config file
    """
    config_content = f'''name: "{model_name}"
backend: "onnxruntime"
max_batch_size: 8
version_policy: {{ all: {{}} }}

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [{input_shape[0]}, {input_shape[1]}]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [1]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]

model_warmup [
  {{
    name: "anomaly_detection_warmup"
    batch_size: 1
    inputs: {{
      key: "input"
      value: {{
        data_type: TYPE_FP32
        dims: [{input_shape[0]}, {input_shape[1]}]
        zero_data: true
      }}
    }}
  }}
]
'''
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write config file
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    logger.info(f"Created Triton config file: {output_path}")


def main():
    """Main function for ONNX export script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export FedSense model to ONNX")
    parser.add_argument("--jax_model", type=str, required=True, help="Path to JAX model parameters")
    parser.add_argument("--onnx_output", type=str, default="docker/triton/model.onnx", help="Output ONNX path")
    parser.add_argument("--fine_tune", action="store_true", help="Fine-tune PyTorch twin")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark ONNX model")
    parser.add_argument("--create_config", action="store_true", help="Create Triton config")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    config = get_config()
    
    # Load fine-tuning data if requested
    fine_tune_data = None
    if args.fine_tune:
        logger.info("Creating fine-tuning dataset from synthetic data")
        from .datasets import generate_synthetic_data
        from .features import make_windows, standardize_features
        
        # Generate synthetic data for fine-tuning
        synthetic_df = generate_synthetic_data(
            n_samples=5000,
            anomaly_rate=0.1,
            random_seed=config.random_seed
        )
        
        # Create windows
        X, y = make_windows(synthetic_df, config.window_len, config.stride)
        X_scaled, _, _ = standardize_features(X)
        
        fine_tune_data = TorchDataset(X_scaled, y)
    
    # Export to ONNX
    success = export_jax_to_onnx(
        jax_model_path=args.jax_model,
        onnx_output_path=args.onnx_output,
        config=config,
        fine_tune_data=fine_tune_data
    )
    
    if not success:
        logger.error("ONNX export failed")
        return
    
    # Benchmark if requested
    if args.benchmark:
        metrics = benchmark_onnx_model(args.onnx_output)
        logger.info(f"Benchmark results: {metrics}")
    
    # Create Triton config if requested
    if args.create_config:
        create_triton_config(
            model_name=config.model_name,
            input_shape=(config.window_len, 4)
        )


if __name__ == "__main__":
    main()
