<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# FedSense - Federated Time-Series Anomaly Detection

## Project Overview
This is a production-quality federated learning project for time-series anomaly detection on wearable device data. The system uses JAX/Flax for deep learning, Flower for federated training, and includes differential privacy features.

## Code Style & Practices
- Use type hints throughout (Python 3.11+ features preferred)
- Follow pydantic for configuration and data validation
- Implement proper logging with structured formats
- Use JAX pure functions and avoid side effects in model code
- Prefer composition over inheritance
- Document complex federated learning and differential privacy logic thoroughly

## Architecture Guidelines
- **Models**: JAX/Flax primary, PyTorch twin for ONNX export only
- **FL**: Flower client-server with FedAvg strategy
- **Privacy**: Optional DP-SGD with per-sample clipping
- **Serving**: FastAPI → Triton inference server → ONNX models
- **Frontend**: Next.js 14 App Router with TypeScript and Tailwind

## Key Dependencies
- JAX/Flax for ML models and training
- Flower for federated learning orchestration  
- MLflow for experiment tracking
- FastAPI for API serving
- Triton Inference Server for model deployment
- Next.js for the web dashboard

## Testing Guidelines
- Unit tests for data processing, model shapes, DP utilities
- Integration tests for FL client-server communication
- Mock external services (MLflow, Triton) in tests
- Test differential privacy guarantees and privacy accounting
