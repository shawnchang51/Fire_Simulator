"""
Configuration for Fire Simulation Surrogate Model
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path


@dataclass
class ModelConfig:
    """Hyperparameters and configuration for the surrogate model."""

    # Data paths
    data_dir: str = "combined_fast"
    floor_plans_dir: str = "combined_fast/floor_plans"

    # Grid processing
    target_grid_size: Tuple[int, int] = (96, 128)  # (H, W) - covers max observed sizes
    num_grid_channels: int = 4  # wall, passable, doors, exits

    # CNN architecture
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    cnn_kernel_size: int = 3

    # Scenario MLP
    scenario_input_dim: int = 4  # agent_count, num_fires, fire_spread_rate, fire_discovery_delay
    scenario_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])

    # Combined head
    head_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2

    # Output
    num_outputs: int = 4  # survival_rate, avg_evacuation_time, steps, avg_fire_damage
    output_names: List[str] = field(default_factory=lambda: [
        'survival_rate', 'avg_evacuation_time', 'steps', 'avg_fire_damage'
    ])

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 10
    num_workers: int = 4

    # Data loading
    max_plans: Optional[int] = None  # Limit floor plans (None = all)

    # Normalization stats (computed from training data)
    scenario_means: Optional[List[float]] = None
    scenario_stds: Optional[List[float]] = None
    target_means: Optional[List[float]] = None
    target_stds: Optional[List[float]] = None

    # Checkpointing
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self):
        """Ensure directories exist."""
        Path(self.checkpoint_dir).mkdir(exist_ok=True)


# Default configuration
DEFAULT_CONFIG = ModelConfig()
