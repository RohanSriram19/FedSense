"""
FedSense: Federated Time-Series Anomaly Detection for Wearables

A production-quality federated learning system for anomaly detection on wearable
device time-series data (heart rate, accelerometer). Built with JAX/Flax, 
Flower federated learning, and optional differential privacy.
"""

__version__ = "0.1.0"
__author__ = "FedSense Team"

from .config import FedSenseConfig

__all__ = ["FedSenseConfig"]
