"""
PyTorch twin model for ONNX export.
Architecturally mirrors the JAX/Flax model for compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AnomalyDetectionCNNTorch(nn.Module):
    """PyTorch twin of the JAX/Flax anomaly detection model."""
    
    def __init__(
        self,
        n_features: int = 4,
        hidden_dims: Tuple[int, ...] = (64, 32),
        dropout_rate: float = 0.1
    ):
        """
        Initialize the PyTorch model.
        
        Args:
            n_features: Number of input features
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # First conv layer (kernel_size=7, same padding)
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=hidden_dims[0],
            kernel_size=7,
            padding=3  # SAME padding for kernel_size=7
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second conv layer (kernel_size=5, same padding)  
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dims[0],
            out_channels=hidden_dims[1],
            kernel_size=5,
            padding=2  # SAME padding for kernel_size=5
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Global average pooling (handled in forward)
        # Final dense layer
        self.classifier = nn.Linear(hidden_dims[1], 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, window_len, n_features)
            
        Returns:
            Output probabilities of shape (batch_size, 1)
        """
        # Transpose for Conv1d: (batch_size, n_features, window_len)
        x = x.transpose(1, 2)
        
        # First conv layer
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second conv layer
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Global average pooling: (batch_size, hidden_dims[1])
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        
        # Final classification
        x = self.classifier(x)
        x = torch.sigmoid(x)
        
        return x


def load_jax_weights_to_torch(
    torch_model: AnomalyDetectionCNNTorch,
    jax_params: Dict[str, Any]
) -> None:
    """
    Load JAX/Flax parameters into PyTorch model.
    Note: This is complex due to parameter structure differences.
    For production use, consider fine-tuning the PyTorch model instead.
    
    Args:
        torch_model: PyTorch model to load weights into
        jax_params: JAX model parameters
    """
    # TODO: Implement weight conversion if needed
    # This is non-trivial due to different parameter naming conventions
    # and tensor layouts between JAX and PyTorch
    logger.warning("JAX to PyTorch weight loading not implemented. "
                  "Consider fine-tuning PyTorch model on pooled data.")


def train_torch_model(
    model: AnomalyDetectionCNNTorch,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Train the PyTorch twin model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Training metrics
    """
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    best_val_loss = float('inf')
    metrics = {}
    
    for epoch in range(n_epochs):
        train_loss = 0.0
        model.train()
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).float()
                
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            metrics = {
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'best_epoch': epoch + 1
            }
    
    return metrics


class TorchDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper for FedSense data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Feature windows of shape (n_samples, window_len, n_features)
            y: Labels of shape (n_samples,)
        """
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_torch_model(
    n_features: int = 4,
    hidden_dims: Tuple[int, ...] = (64, 32),
    dropout_rate: float = 0.1
) -> AnomalyDetectionCNNTorch:
    """
    Create and initialize a PyTorch twin model.
    
    Args:
        n_features: Number of input features
        hidden_dims: Hidden layer dimensions  
        dropout_rate: Dropout rate
        
    Returns:
        Initialized PyTorch model
    """
    model = AnomalyDetectionCNNTorch(
        n_features=n_features,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created PyTorch model with {total_params} total parameters "
               f"({trainable_params} trainable)")
    
    return model


def evaluate_torch_model(
    model: AnomalyDetectionCNNTorch,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate PyTorch model and compute metrics.
    
    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        device: Device to evaluate on
        
    Returns:
        Evaluation metrics
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    
    model = model.to(device)
    model.eval()
    
    all_probs = []
    all_labels = []
    total_loss = 0.0
    
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()
            
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            total_loss += loss.item()
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    
    # Find optimal threshold for F1
    thresholds = np.linspace(0, 1, 101)
    best_f1 = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        pred_labels = (all_probs >= threshold).astype(int)
        if np.sum(pred_labels) > 0 and np.sum(all_labels) > 0:
            f1 = f1_score(all_labels, pred_labels)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    metrics = {
        'loss': total_loss / len(data_loader),
        'auroc': auroc,
        'auprc': auprc,
        'f1': best_f1,
        'best_threshold': best_threshold,
        'n_samples': len(all_labels)
    }
    
    return metrics
