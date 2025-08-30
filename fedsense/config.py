"""
Configuration management using Pydantic settings.
Supports environment variables and .env files.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class FedSenseConfig(BaseSettings):
    """Main configuration for FedSense system."""
    
    # Data settings
    data_dir: Path = Field(default=Path("data"), description="Directory containing client data")
    window_len: int = Field(default=250, description="Window length (5s @ 50Hz)")
    stride: int = Field(default=50, description="Window stride for sliding window")
    use_fft: bool = Field(default=False, description="Include FFT features")
    
    # Model hyperparameters
    learning_rate: float = Field(default=1e-3, description="Learning rate for local training")
    batch_size: int = Field(default=64, description="Batch size for training")
    hidden_dims: list[int] = Field(default=[64, 32], description="Hidden layer dimensions")
    dropout_rate: float = Field(default=0.1, description="Dropout rate")
    
    # Federated learning settings
    n_clients: int = Field(default=8, description="Number of federated clients")
    rounds: int = Field(default=10, description="Number of federated rounds")
    local_epochs: int = Field(default=2, description="Local epochs per round")
    min_fit_clients: int = Field(default=6, description="Minimum clients for training")
    min_eval_clients: int = Field(default=4, description="Minimum clients for evaluation")
    
    # Differential privacy settings
    use_dp: bool = Field(default=False, description="Enable differential privacy")
    clip_norm: float = Field(default=1.0, description="Gradient clipping L2 norm")
    noise_multiplier: float = Field(default=1.0, description="DP noise multiplier")
    dp_epsilon: float = Field(default=8.0, description="Target privacy epsilon")
    dp_delta: float = Field(default=1e-5, description="Target privacy delta")
    
    # MLflow settings
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", description="MLflow server URI")
    experiment_name: str = Field(default="fedsense", description="MLflow experiment name")
    
    # API/Serving settings
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")
    triton_url: str = Field(default="http://localhost:8001", description="Triton inference server URL")
    model_name: str = Field(default="fedsense_anomaly", description="Model name in Triton")
    
    # Random seed
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    
    @validator('data_dir', pre=True)
    def validate_data_dir(cls, v):
        """Ensure data directory exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @validator('min_fit_clients')
    def validate_min_fit_clients(cls, v, values):
        """Ensure min_fit_clients <= n_clients."""
        n_clients = values.get('n_clients', 8)
        if v > n_clients:
            raise ValueError(f"min_fit_clients ({v}) cannot exceed n_clients ({n_clients})")
        return v
    
    @validator('min_eval_clients')
    def validate_min_eval_clients(cls, v, values):
        """Ensure min_eval_clients <= n_clients."""
        n_clients = values.get('n_clients', 8)
        if v > n_clients:
            raise ValueError(f"min_eval_clients ({v}) cannot exceed n_clients ({n_clients})")
        return v

    class Config:
        env_file = ".env"
        env_prefix = "FEDSENSE_"


def get_config() -> FedSenseConfig:
    """Get the global configuration instance."""
    return FedSenseConfig()
