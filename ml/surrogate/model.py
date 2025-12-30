"""
CNN+MLP Surrogate Model for Fire Simulation
"""

from typing import List

import torch
import torch.nn as nn

from .config import ModelConfig


class FireSimulationCNN(nn.Module):
    """
    CNN for processing floor plan grids.

    Input: (B, 4, H, W) - 4 channels (wall, passable, doors, exits)
    Output: (B, feature_dim) - flattened spatial features
    """

    def __init__(
        self,
        in_channels: int = 4,
        channel_dims: List[int] = None,
        kernel_size: int = 3
    ):
        super().__init__()

        if channel_dims is None:
            channel_dims = [32, 64, 128, 256]

        layers = []
        prev_ch = in_channels

        for ch in channel_dims:
            layers.extend([
                nn.Conv2d(prev_ch, ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  # Reduces spatial dims by 2x each layer
            ])
            prev_ch = ch

        self.conv_layers = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Fixed output size
        self.output_dim = channel_dims[-1] * 4 * 4  # e.g., 256 * 16 = 4096

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        return x.view(x.size(0), -1)


class ScenarioMLP(nn.Module):
    """
    MLP for processing scenario scalar features.

    Input: (B, 4) - agent_count, num_fires, fire_spread_rate, fire_discovery_delay
    Output: (B, hidden_dim)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: List[int] = None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FireSimulationSurrogate(nn.Module):
    """
    Combined surrogate model: CNN for grid + MLP for scenario -> multi-output prediction.

    Predicts 4 outputs (no activation):
    - survival_rate
    - avg_evacuation_time
    - steps
    - avg_fire_damage
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()

        if config is None:
            config = ModelConfig()

        # CNN for grid processing
        self.cnn = FireSimulationCNN(
            in_channels=config.num_grid_channels,
            channel_dims=config.cnn_channels,
            kernel_size=config.cnn_kernel_size
        )

        # MLP for scenario parameters
        self.scenario_mlp = ScenarioMLP(
            input_dim=config.scenario_input_dim,
            hidden_dims=config.scenario_hidden_dims
        )

        # Combined head
        combined_input_dim = self.cnn.output_dim + self.scenario_mlp.output_dim

        head_layers = []
        prev_dim = combined_input_dim

        for dim in config.head_hidden_dims:
            head_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout)
            ])
            prev_dim = dim

        # Final output layer (no activation - raw outputs)
        head_layers.append(nn.Linear(prev_dim, config.num_outputs))

        self.head = nn.Sequential(*head_layers)

        # Store config for reference
        self.config = config

    def forward(
        self,
        grid: torch.Tensor,
        scenario: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            grid: (B, 4, H, W) floor plan grid tensor
            scenario: (B, 4) scenario parameters tensor

        Returns:
            (B, 4) predictions for each output
        """
        grid_features = self.cnn(grid)
        scenario_features = self.scenario_mlp(scenario)
        combined = torch.cat([grid_features, scenario_features], dim=1)
        return self.head(combined)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: ModelConfig = None) -> FireSimulationSurrogate:
    """Create a new model instance."""
    return FireSimulationSurrogate(config)
