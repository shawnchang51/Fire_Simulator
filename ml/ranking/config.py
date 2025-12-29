"""
Configuration for Pairwise Ranking Model
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class RankingConfig:
    """
    Configuration for the pairwise ranking model.

    Attributes:
        data_dir: Root directory containing training data
        floor_plans_dir: Directory containing floor plan NPZ files
        target_grid_size: Target size (H, W) for grid padding
        cnn_channels: Channel dimensions for CNN layers
        latent_dim: Dimension of latent vector (K)
        scenario_hidden_dim: Hidden dimension for scenario MLP
        scenario_output_dim: Output dimension for scenario MLP
        scoring_hidden_dim: Hidden dimension for scoring head
        dropout: Dropout rate for scoring head
        loss_type: Loss function type ('ranknet' or 'hinge')
        margin: Margin for hinge loss
        sigma: Scaling factor for RankNet logistic
        l1_lambda: L1 regularization strength for latent sparsity
        weight_decay: L2 regularization (AdamW weight decay)
        batch_size: Number of pairs per batch
        learning_rate: Initial learning rate
        warmup_epochs: Number of warmup epochs
        epochs: Maximum training epochs
        early_stopping_patience: Patience for early stopping
        num_workers: Number of data loading workers
        checkpoint_dir: Directory for saving checkpoints
        log_dir: Directory for TensorBoard logs
        seed: Random seed for reproducibility
    """

    # Data paths
    data_dir: str = "combined_fast"
    floor_plans_dir: str = "combined_fast/floor_plans"
    target_grid_size: Tuple[int, int] = (96, 128)

    # Grid encoding
    num_grid_channels: int = 5  # wall, passable, doors, exits, valid_mask

    # CNN Encoder
    cnn_channels: List[int] = field(default_factory=lambda: [16, 32, 64])
    latent_dim: int = 8  # K=8 for interpretability

    # Scenario MLP (4 → 32 → 16)
    scenario_input_dim: int = 4  # agent_count, num_fires, fire_spread_rate, fire_discovery_delay
    scenario_hidden_dim: int = 32
    scenario_output_dim: int = 16

    # Scoring Head (latent + scenario → score)
    scoring_hidden_dim: int = 32
    dropout: float = 0.3

    # Loss function
    loss_type: str = "ranknet"  # "ranknet" or "hinge"
    margin: float = 0.1  # For hinge loss
    sigma: float = 1.0  # For RankNet

    # Regularization
    l1_lambda: float = 0.005  # Latent sparsity
    weight_decay: float = 1e-4

    # Training
    batch_size: int = 128  # pairs per batch
    learning_rate: float = 1e-3
    warmup_epochs: int = 5
    epochs: int = 100
    early_stopping_patience: int = 15
    num_workers: int = 4

    # Checkpoint and logging
    checkpoint_dir: str = "checkpoints/ranking"
    log_dir: str = "logs/ranking"

    # Reproducibility
    seed: Optional[int] = 42

    # Data Augmentation
    augment_shift: bool = True  # Random shift augmentation for training

    # Normalization stats (computed from training data)
    scenario_means: Optional[List[float]] = None
    scenario_stds: Optional[List[float]] = None

    @property
    def feature_dim(self) -> int:
        """Total feature dimension after combining latent and scenario."""
        return self.latent_dim + self.scenario_output_dim

    def __post_init__(self):
        """Validate configuration."""
        assert len(self.cnn_channels) >= 1, "Need at least one CNN channel"
        assert self.latent_dim > 0, "Latent dimension must be positive"
        assert self.loss_type in ("ranknet", "hinge"), f"Unknown loss type: {self.loss_type}"
        assert 0 <= self.dropout < 1, "Dropout must be in [0, 1)"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
