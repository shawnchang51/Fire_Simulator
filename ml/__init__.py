"""
Fire Simulation Surrogate Model

A PyTorch CNN+MLP model for predicting fire simulation metrics
from floor plan configurations.
"""

from .config import ModelConfig
from .dataset import FireSimulationDataset, create_dataloaders
from .model import FireSimulationSurrogate
from .train import train_model
from .evaluate import evaluate_model, compute_metrics

__all__ = [
    'ModelConfig',
    'FireSimulationDataset',
    'create_dataloaders',
    'FireSimulationSurrogate',
    'train_model',
    'evaluate_model',
    'compute_metrics',
]
