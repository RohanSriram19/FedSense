"""
Pydantic models for API request/response schemas.
Defines the data structures for FastAPI endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
from enum import Enum


class ClientStatus(str, Enum):
    """Client status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class AnomalyScoreRequest(BaseModel):
    """Request model for anomaly scoring."""
    
    window: List[List[float]] = Field(
        ...,
        description="Time-series window of shape (window_len, n_features). "
                   "Features should be [hr, accel_x, accel_y, accel_z]",
        example=[
            [75.5, 0.1, 0.2, 9.8],
            [76.0, 0.0, 0.1, 9.9],
            [75.8, -0.1, 0.3, 9.7]
        ]
    )
    
    @validator('window')
    def validate_window(cls, v):
        """Validate window format and dimensions."""
        if not v:
            raise ValueError("Window cannot be empty")
        
        # Check that all timesteps have the same number of features
        n_features = len(v[0])
        for i, timestep in enumerate(v):
            if len(timestep) != n_features:
                raise ValueError(f"Inconsistent feature count at timestep {i}")
        
        # Check expected number of features
        if n_features != 4:
            raise ValueError(f"Expected 4 features (HR, accel_x, accel_y, accel_z), got {n_features}")
        
        return v


class AnomalyScoreResponse(BaseModel):
    """Response model for anomaly scoring."""
    
    anomaly_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Anomaly probability between 0 and 1"
    )
    
    timestamp: str = Field(
        ...,
        description="Timestamp of the prediction"
    )
    
    model_version: str = Field(
        default="federated_v1",
        description="Version of the model used for prediction"
    )
    
    inference_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference time in milliseconds"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(
        ...,
        description="Overall system status",
        example="ok"
    )
    
    triton: str = Field(
        ...,
        description="Triton Inference Server status",
        example="ok"
    )
    
    mlflow: str = Field(
        ...,
        description="MLflow server status",
        example="ok"
    )
    
    timestamp: str = Field(
        ...,
        description="Health check timestamp"
    )


class ClientInfo(BaseModel):
    """Information about a federated learning client."""
    
    id: int = Field(
        ...,
        ge=0,
        description="Unique client identifier"
    )
    
    last_heartbeat: str = Field(
        ...,
        description="Timestamp of last heartbeat from client"
    )
    
    local_epochs: int = Field(
        ...,
        ge=0,
        description="Number of local training epochs per round"
    )
    
    dp_enabled: bool = Field(
        ...,
        description="Whether differential privacy is enabled for this client"
    )
    
    status: ClientStatus = Field(
        ...,
        description="Current client status"
    )


class MetricsResponse(BaseModel):
    """Response model for current model metrics."""
    
    auroc: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Area Under ROC Curve"
    )
    
    f1: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="F1 Score"
    )
    
    rounds: int = Field(
        ...,
        ge=0,
        description="Number of completed federated learning rounds"
    )
    
    last_updated: str = Field(
        ...,
        description="Timestamp when metrics were last updated"
    )


class SyntheticWindowRequest(BaseModel):
    """Request model for generating synthetic test windows."""
    
    window_length: int = Field(
        default=250,
        ge=10,
        le=1000,
        description="Length of the synthetic window"
    )
    
    anomaly: bool = Field(
        default=False,
        description="Whether to inject anomaly patterns"
    )
    
    noise_level: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Amount of noise to add to the signal"
    )
    
    heart_rate_base: float = Field(
        default=75.0,
        ge=40.0,
        le=200.0,
        description="Base heart rate in BPM"
    )


class SyntheticWindowResponse(BaseModel):
    """Response model for synthetic window generation."""
    
    window: List[List[float]] = Field(
        ...,
        description="Generated synthetic window"
    )
    
    true_anomaly: bool = Field(
        ...,
        description="Whether this window contains anomalies"
    )
    
    metadata: dict = Field(
        ...,
        description="Metadata about the generated window"
    )


class WebSocketMessage(BaseModel):
    """WebSocket message for real-time updates."""
    
    timestamp: float = Field(
        ...,
        description="Unix timestamp"
    )
    
    anomaly_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Anomaly probability"
    )
    
    client_id: Optional[int] = Field(
        None,
        description="Client that generated this score (if applicable)"
    )


class PingResponse(BaseModel):
    """Response model for client ping."""
    
    client_id: int = Field(
        ...,
        description="Client identifier"
    )
    
    status: str = Field(
        ...,
        description="Ping result status"
    )
    
    ping_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Round-trip ping time in milliseconds"
    )
    
    timestamp: str = Field(
        ...,
        description="Timestamp of the ping"
    )


# Example data for documentation
EXAMPLE_WINDOW = [
    [75.5, 0.1, 0.2, 9.8],
    [76.0, 0.0, 0.1, 9.9],
    [75.8, -0.1, 0.3, 9.7],
    [77.2, 0.2, -0.1, 9.8],
    [76.5, 0.0, 0.0, 9.9]
] * 50  # Repeat to make 250 timesteps

EXAMPLE_ANOMALY_WINDOW = [
    [95.5, 2.1, 1.2, 11.8],  # Elevated HR and acceleration
    [98.0, 1.8, 1.1, 11.9],
    [96.8, 2.2, 1.5, 11.2],
    [99.2, 2.5, 1.8, 12.1],
    [97.5, 1.9, 1.3, 11.5]
] * 50  # Repeat to make 250 timesteps


# Update examples in the request models
AnomalyScoreRequest.model_config = {
    "json_schema_extra": {
        "examples": [
            {
                "window": EXAMPLE_WINDOW[:5]  # Show first 5 timesteps
            }
        ]
    }
}

SyntheticWindowRequest.model_config = {
    "json_schema_extra": {
        "examples": [
            {
                "window_length": 250,
                "anomaly": False,
                "noise_level": 0.1,
                "heart_rate_base": 75.0
            },
            {
                "window_length": 250,
                "anomaly": True,
                "noise_level": 0.15,
                "heart_rate_base": 80.0
            }
        ]
    }
}
