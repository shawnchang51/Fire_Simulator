"""
Machine Learning Models for Fire Simulation

This package contains:
- surrogate: Regression model for predicting simulation metrics
- ranking: Pairwise ranking model for comparing configurations
"""

# Re-export surrogate model for backward compatibility
from .surrogate import (
    ModelConfig,
    FireSimulationDataset,
    create_dataloaders,
    FireSimulationSurrogate,
    train_model,
    evaluate_model,
    compute_metrics,
)

__all__ = [
    # Surrogate model
    'ModelConfig',
    'FireSimulationDataset',
    'create_dataloaders',
    'FireSimulationSurrogate',
    'train_model',
    'evaluate_model',
    'compute_metrics',
    # Submodules
    'surrogate',
    'ranking',
]
