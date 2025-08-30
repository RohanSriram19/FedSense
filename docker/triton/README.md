# Triton Inference Server Setup

This directory contains configuration for deploying FedSense models with Triton Inference Server.

## Quick Start

1. **Export model to ONNX**:
   ```bash
   make export
   ```

2. **Start Triton server**:
   ```bash
   make triton_up
   ```

3. **Test inference**:
   ```bash
   curl -X POST http://localhost:8001/v2/models/fedsense_anomaly/infer \
     -H "Content-Type: application/json" \
     -d @test_request.json
   ```

## Directory Structure

```
triton/
├── config.pbtxt              # Triton model configuration
├── model.onnx               # ONNX model (created by export)
├── README.md                # This file
└── test_request.json        # Example inference request
```

## Model Configuration

- **Model Name**: `fedsense_anomaly`
- **Backend**: ONNX Runtime
- **Input**: `input` (float32, shape: [-1, 250, 4])
  - Batch dimension is dynamic
  - 250 timesteps (5 seconds @ 50Hz)
  - 4 features: [HR, accel_x, accel_y, accel_z]
- **Output**: `output` (float32, shape: [-1, 1])
  - Anomaly probability [0, 1]

## Running Triton Server

### Using Docker

```bash
# Pull Triton image
docker pull nvcr.io/nvidia/tritonserver:23.08-py3

# Start server (assumes model.onnx exists)
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd):/models \
  nvcr.io/nvidia/tritonserver:23.08-py3 \
  tritonserver --model-repository=/models
```

### Local Installation

```bash
# Install Triton (Ubuntu/Debian)
wget https://github.com/triton-inference-server/server/releases/download/v2.37.0/tritonserver2.37.0-jetpack5.1.tgz
tar xzf tritonserver2.37.0-jetpack5.1.tgz

# Start server
./bin/tritonserver --model-repository=/path/to/models
```

## API Endpoints

- **HTTP**: `http://localhost:8000`
- **gRPC**: `localhost:8001` 
- **Metrics**: `http://localhost:8002/metrics`

### Health Check
```bash
curl http://localhost:8000/v2/health/ready
```

### Model Status
```bash
curl http://localhost:8000/v2/models/fedsense_anomaly
```

### Inference
```bash
curl -X POST http://localhost:8001/v2/models/fedsense_anomaly/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "input",
        "shape": [1, 250, 4],
        "datatype": "FP32", 
        "data": [[...]]
      }
    ]
  }'
```

## Performance Tuning

### CPU Optimization
```
instance_group [
  {
    count: 2  # Multiple CPU instances
    kind: KIND_CPU
  }
]
```

### GPU Support
```
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

### Dynamic Batching
```
dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100
}
```

## Monitoring

Triton provides Prometheus metrics at `/metrics`:

- `nv_inference_request_duration_us` - Inference latency
- `nv_inference_request_success` - Success rate
- `nv_inference_queue_duration_us` - Queue time
- `nv_gpu_utilization` - GPU utilization

## Troubleshooting

### Common Issues

1. **Model not loading**:
   - Check `config.pbtxt` syntax
   - Verify `model.onnx` exists and is valid
   - Check Triton logs for errors

2. **ONNX Runtime errors**:
   - Ensure ONNX model opset compatibility
   - Verify input/output shapes match config

3. **Performance issues**:
   - Increase instance count for CPU
   - Enable dynamic batching
   - Use GPU if available

### Debug Commands

```bash
# Check model status
curl http://localhost:8000/v2/models/fedsense_anomaly/ready

# View server stats
curl http://localhost:8000/v2/models/fedsense_anomaly/stats

# Monitor logs
docker logs -f triton_server
```
