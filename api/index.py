"""
Vercel-compatible serverless function wrapper for FedSense FastAPI backend.
This is a simplified version for serverless deployment.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import random
import numpy as np
from datetime import datetime

# Create FastAPI app
app = FastAPI(title="FedSense API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class TimeSeriesData(BaseModel):
    timestamp: str
    heart_rate: float
    steps: int
    sleep_quality: float

class AnomalyPrediction(BaseModel):
    timestamp: str
    anomaly_score: float
    is_anomaly: bool
    confidence: float

class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_updated: str

# Mock data and functions for serverless deployment
def generate_mock_data(days: int = 7) -> List[TimeSeriesData]:
    """Generate mock wearable device data."""
    data = []
    base_time = datetime.now()
    
    for i in range(days * 24):  # Hourly data
        timestamp = base_time.replace(hour=i % 24).isoformat()
        data.append(TimeSeriesData(
            timestamp=timestamp,
            heart_rate=70 + random.gauss(0, 10),
            steps=random.randint(0, 1000),
            sleep_quality=random.uniform(0.3, 1.0)
        ))
    
    return data

def detect_anomalies(data: List[TimeSeriesData]) -> List[AnomalyPrediction]:
    """Simple anomaly detection using statistical thresholds."""
    predictions = []
    
    heart_rates = [d.heart_rate for d in data]
    hr_mean = np.mean(heart_rates)
    hr_std = np.std(heart_rates)
    
    for d in data:
        # Simple threshold-based anomaly detection
        z_score = abs(d.heart_rate - hr_mean) / hr_std
        is_anomaly = z_score > 2.0
        
        predictions.append(AnomalyPrediction(
            timestamp=d.timestamp,
            anomaly_score=min(z_score / 3.0, 1.0),
            is_anomaly=is_anomaly,
            confidence=0.8 if is_anomaly else 0.95
        ))
    
    return predictions

# API Endpoints
@app.get("/")
async def root():
    return {"message": "FedSense API - Federated Anomaly Detection", "status": "healthy"}

@app.get("/api/health")
async def health():
    return {"status": "healthy", "platform": "vercel", "timestamp": datetime.now().isoformat()}

@app.get("/api/data", response_model=List[TimeSeriesData])
async def get_data(days: int = 7):
    """Get mock wearable device data."""
    if days > 30:
        raise HTTPException(status_code=400, detail="Maximum 30 days of data")
    return generate_mock_data(days)

@app.post("/api/predict", response_model=List[AnomalyPrediction])
async def predict_anomalies(data: List[TimeSeriesData]):
    """Predict anomalies in the provided time series data."""
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="No data provided")
    if len(data) > 1000:
        raise HTTPException(status_code=400, detail="Too many data points (max 1000)")
    
    predictions = detect_anomalies(data)
    return predictions

@app.get("/api/metrics", response_model=ModelMetrics)
async def get_metrics():
    """Get current model performance metrics."""
    return ModelMetrics(
        accuracy=0.94,
        precision=0.89,
        recall=0.87,
        f1_score=0.88,
        last_updated=datetime.now().isoformat()
    )

@app.get("/api/status")
async def get_status():
    """Get FL training status."""
    return {
        "fl_round": 15,
        "active_clients": 8,
        "global_accuracy": 0.94,
        "last_update": datetime.now().isoformat(),
        "privacy_budget": 0.75
    }

# Vercel handler
handler = Mangum(app, lifespan="off")
