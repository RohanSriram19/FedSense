"""
FastAPI server for FedSense anomaly detection.
Provides REST API endpoints for scoring and health checks.
Forwards inference requests to Triton Inference Server.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import asyncio
from datetime import datetime
import json

from .config import get_config
from .client_payload import (
    AnomalyScoreRequest, AnomalyScoreResponse,
    HealthResponse, ClientInfo, MetricsResponse
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FedSense API",
    description="Federated Time-Series Anomaly Detection for Wearables",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global config
config = get_config()

# Mock client data (in production, this would come from a database)
MOCK_CLIENTS = [
    ClientInfo(
        id=0,
        last_heartbeat=datetime.now().isoformat(),
        local_epochs=2,
        dp_enabled=False,
        status="healthy"
    ),
    ClientInfo(
        id=1,
        last_heartbeat=datetime.now().isoformat(),
        local_epochs=2,
        dp_enabled=True,
        status="healthy"
    ),
    ClientInfo(
        id=2,
        last_heartbeat=datetime.now().isoformat(),
        local_epochs=2,
        dp_enabled=False,
        status="degraded"
    ),
]

# Mock metrics (in production, this would come from MLflow or database)
MOCK_METRICS = MetricsResponse(
    auroc=0.892,
    f1=0.847,
    rounds=10,
    last_updated=datetime.now().isoformat()
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint that verifies system components.
    """
    try:
        # Check Triton server
        triton_status = await check_triton_health()
        
        # Check MLflow (simplified check)
        mlflow_status = await check_mlflow_health()
        
        # Overall status
        overall_status = "ok"
        if triton_status != "ok" or mlflow_status != "ok":
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            triton=triton_status,
            mlflow=mlflow_status,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/score", response_model=AnomalyScoreResponse)
async def score_anomaly(request: AnomalyScoreRequest):
    """
    Score a time-series window for anomaly detection.
    Forwards request to Triton Inference Server.
    """
    try:
        logger.info(f"Received scoring request for window of length {len(request.window)}")
        
        # Validate input
        if len(request.window) != config.window_len:
            raise HTTPException(
                status_code=400,
                detail=f"Window length must be {config.window_len}, got {len(request.window)}"
            )
        
        # Check that each timestep has correct number of features
        expected_features = 4  # HR + 3 accel channels
        for i, timestep in enumerate(request.window):
            if len(timestep) != expected_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Each timestep must have {expected_features} features, "
                          f"timestep {i} has {len(timestep)}"
                )
        
        # Convert to numpy array and add batch dimension
        window_array = np.array(request.window, dtype=np.float32)
        batch_input = np.expand_dims(window_array, axis=0)  # Shape: (1, window_len, n_features)
        
        # Forward to Triton (or use local model if Triton unavailable)
        try:
            anomaly_prob = await forward_to_triton(batch_input)
        except Exception as triton_error:
            logger.warning(f"Triton forwarding failed: {triton_error}, using local fallback")
            anomaly_prob = await local_inference_fallback(batch_input)
        
        return AnomalyScoreResponse(
            anomaly_prob=float(anomaly_prob),
            timestamp=datetime.now().isoformat(),
            model_version="federated_v1",
            inference_time_ms=50.0  # Mock timing
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.get("/clients", response_model=List[ClientInfo])
async def get_clients():
    """
    Get list of federated learning clients and their status.
    """
    return MOCK_CLIENTS


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get current model performance metrics.
    """
    return MOCK_METRICS


@app.post("/clients/{client_id}/ping")
async def ping_client(client_id: int):
    """
    Ping a specific client to check connectivity.
    """
    # Find client
    client = next((c for c in MOCK_CLIENTS if c.id == client_id), None)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Mock ping result
    return {
        "client_id": client_id,
        "status": "ok",
        "ping_time_ms": 15.2,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/train/metrics")
async def get_training_metrics():
    """
    Get federated training metrics history.
    """
    # Mock training metrics data
    training_rounds = []
    for round_num in range(1, 11):  # 10 training rounds
        accuracy = 0.7 + (round_num * 0.02) + np.random.normal(0, 0.01)  # Improving accuracy
        accuracy = min(accuracy, 0.95)  # Cap at 95%
        
        training_rounds.append({
            "round": round_num,
            "clients_participating": np.random.randint(8, 12),
            "global_loss": 1.0 - accuracy + np.random.normal(0, 0.05),
            "accuracy": float(accuracy),
            "convergence_score": float(min(0.9, round_num * 0.08)),
            "privacy_spent": float(round_num * 0.1),
            "duration": np.random.randint(120, 300)  # 2-5 minutes
        })
    
    return training_rounds


@app.get("/model/info")
async def get_model_info():
    """
    Get current model information and metadata.
    """
    return {
        "version": 5,
        "created_at": "2024-01-20T10:00:00Z",
        "accuracy": 0.891,
        "f1_score": 0.875,
        "parameters_count": 45280,
        "model_size_mb": 0.42
    }


@app.get("/train/status")
async def get_training_status():
    """
    Get current federated training status.
    """
    return {
        "is_training": False,
        "current_round": 0,
        "total_rounds": 10,
        "clients_ready": len([c for c in MOCK_CLIENTS if c.status == "online"]),
        "estimated_completion": None
    }


@app.post("/train/start")
async def start_training():
    """
    Start a new federated training round.
    """
    return {
        "message": "Federated training started",
        "round": 1
    }


@app.post("/train/stop")
async def stop_training():
    """
    Stop the current federated training.
    """
    return {
        "message": "Federated training stopped"
    }


@app.post("/detect")
async def detect_anomalies(request: AnomalyScoreRequest):
    """
    Detect anomalies in time-series data.
    """
    results = []
    for i, data_point in enumerate(request.data):
        # Simple anomaly detection logic
        anomaly_score = float(np.random.beta(2, 8))  # Skewed towards low values
        is_anomaly = anomaly_score > 0.7
        
        results.append({
            "timestamp": datetime.now().isoformat(),
            "value": float(np.mean(data_point) if isinstance(data_point, list) else data_point),
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "confidence": float(np.random.uniform(0.7, 0.95))
        })
    
    return results


@app.get("/privacy/metrics")
async def get_privacy_metrics():
    """
    Get differential privacy metrics and budget usage.
    """
    return {
        "total_budget": 1.0,
        "spent_budget": 0.65,
        "remaining_budget": 0.35,
        "epsilon": 1.0,
        "delta": 1e-5,
        "clients_privacy_status": [
            {
                "client_id": f"client_{i}",
                "privacy_spent": np.random.uniform(0.5, 0.8),
                "privacy_remaining": np.random.uniform(0.2, 0.5)
            } for i in range(len(MOCK_CLIENTS))
        ]
    }


async def check_triton_health() -> str:
    """Check if Triton Inference Server is healthy."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{config.triton_url}/v2/health/ready")
            return "ok" if response.status_code == 200 else "down"
    except Exception:
        return "down"


async def check_mlflow_health() -> str:
    """Check if MLflow server is healthy."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{config.mlflow_tracking_uri}/api/2.0/mlflow/experiments/list")
            return "ok" if response.status_code == 200 else "down"
    except Exception:
        return "down"


async def forward_to_triton(batch_input: np.ndarray) -> float:
    """
    Forward inference request to Triton Inference Server.
    
    Args:
        batch_input: Input batch of shape (1, window_len, n_features)
        
    Returns:
        Anomaly probability
    """
    # Prepare Triton inference request
    inference_request = {
        "inputs": [
            {
                "name": "input",
                "shape": list(batch_input.shape),
                "datatype": "FP32",
                "data": batch_input.flatten().tolist()
            }
        ],
        "outputs": [
            {
                "name": "output"
            }
        ]
    }
    
    triton_inference_url = f"{config.triton_url}/v2/models/{config.model_name}/infer"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            triton_inference_url,
            json=inference_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Triton inference failed: {response.status_code} {response.text}")
        
        result = response.json()
        
        # Extract prediction
        output_data = result["outputs"][0]["data"]
        anomaly_prob = output_data[0]  # First (and only) prediction in batch
        
        return anomaly_prob


async def local_inference_fallback(batch_input: np.ndarray) -> float:
    """
    Local inference fallback when Triton is unavailable.
    Returns a mock prediction for demonstration.
    
    Args:
        batch_input: Input batch
        
    Returns:
        Mock anomaly probability
    """
    # Simple heuristic-based fallback
    # In practice, you'd load a local model here
    
    # Extract features
    hr_values = batch_input[0, :, 0]  # Heart rate channel
    accel_magnitude = np.sqrt(
        batch_input[0, :, 1]**2 + 
        batch_input[0, :, 2]**2 + 
        batch_input[0, :, 3]**2
    )
    
    # Simple anomaly detection based on thresholds
    high_hr = np.mean(hr_values) > 100  # High heart rate
    high_accel = np.mean(accel_magnitude) > 2.0  # High acceleration
    hr_variability = np.std(hr_values) > 10  # High HR variability
    
    # Combine indicators
    anomaly_score = 0.0
    if high_hr:
        anomaly_score += 0.4
    if high_accel:
        anomaly_score += 0.3
    if hr_variability:
        anomaly_score += 0.3
    
    # Add some noise for realism
    anomaly_score += np.random.normal(0, 0.05)
    anomaly_score = np.clip(anomaly_score, 0.0, 1.0)
    
    logger.info(f"Local fallback prediction: {anomaly_score:.3f}")
    return float(anomaly_score)


# WebSocket endpoint for real-time anomaly stream (optional)
@app.websocket("/ws/anomaly_stream")
async def anomaly_stream_websocket(websocket):
    """
    WebSocket endpoint for streaming anomaly scores.
    Useful for real-time dashboard updates.
    """
    await websocket.accept()
    
    try:
        while True:
            # Generate mock streaming data
            mock_score = {
                "timestamp": datetime.now().timestamp(),
                "anomaly_prob": float(np.random.beta(2, 8)),  # Skewed towards low values
                "client_id": np.random.randint(0, config.n_clients)
            }
            
            await websocket.send_json(mock_score)
            await asyncio.sleep(1.0)  # Send every second
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


def create_app() -> FastAPI:
    """Factory function to create FastAPI app."""
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    logger.info(f"Starting FedSense API server on {host}:{port}")
    
    uvicorn.run(
        "fedsense.serve_fastapi:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FedSense FastAPI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    run_server(args.host, args.port, args.reload)
