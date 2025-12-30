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
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    latent_dim: int = 64  # Increased from 8 for better capacity
    use_residual: bool = True  # Use residual connections in CNN

    # Scenario MLP (4 → 64 → 32)
    scenario_input_dim: int = 4  # agent_count, num_fires, fire_spread_rate, fire_discovery_delay
    scenario_hidden_dim: int = 64  # Increased from 32
    scenario_output_dim: int = 32  # Increased from 16

    # Scoring Head (latent + scenario → score)
    scoring_hidden_dim: int = 64  # Increased from 32
    scoring_num_layers: int = 3  # Number of hidden layers in scoring head
    use_layer_norm: bool = True  # Use LayerNorm in scoring head
    dropout: float = 0.1  # Reduced from 0.3

    # Loss function
    loss_type: str = "ranknet"  # "ranknet" or "hinge"
    margin: float = 0.1  # For hinge loss
    sigma: float = 1.0  # For RankNet

    # Regularization
    l1_lambda: float = 0.001  # Reduced from 0.005 for larger model
    weight_decay: float = 5e-5  # Reduced from 1e-4
    label_smoothing: float = 0.0  # Label smoothing for noisy pairs

    # Training
    batch_size: int = 128  # pairs per batch
    gradient_accumulation_steps: int = 1  # Accumulate gradients for larger effective batch
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
        assert self.scoring_num_layers >= 1, "Scoring head must have at least 1 layer"
        assert 0 <= self.label_smoothing < 0.5, "Label smoothing must be in [0, 0.5)"


def get_high_capacity_config() -> RankingConfig:
    """
    Configuration for high AUC training (target: 0.85-0.90).

    Prioritizes model capacity over interpretability.
    """
    return RankingConfig(
        # Expanded architecture
        cnn_channels=[32, 64, 128, 256],
        latent_dim=64,
        use_residual=True,

        # Deeper scoring head
        scoring_hidden_dim=64,
        scoring_num_layers=3,
        use_layer_norm=True,

        # Larger scenario encoder
        scenario_hidden_dim=64,
        scenario_output_dim=32,

        # Reduced regularization
        dropout=0.1,
        l1_lambda=0.001,
        weight_decay=5e-5,

        # Training
        loss_type='ranknet',
        sigma=1.0,
        batch_size=256,
        learning_rate=5e-4,
        warmup_epochs=10,
        epochs=150,
        early_stopping_patience=20,

        # Augmentation
        augment_shift=True
    )


def get_balanced_config() -> RankingConfig:
    """
    Balanced configuration for moderate capacity increase.

    Good balance between interpretability and performance.
    """
    return RankingConfig(
        cnn_channels=[32, 64, 128],
        latent_dim=32,
        use_residual=True,

        scoring_hidden_dim=48,
        scoring_num_layers=2,
        use_layer_norm=False,

        scenario_hidden_dim=48,
        scenario_output_dim=24,

        dropout=0.15,
        l1_lambda=0.002,

        batch_size=192,
        learning_rate=7e-4,
        warmup_epochs=7,
        epochs=120,
    )


def get_legacy_config() -> RankingConfig:
    """
    Legacy configuration matching original K=8 architecture.

    For backward compatibility and comparison.
    """
    return RankingConfig(
        cnn_channels=[16, 32, 64],
        latent_dim=8,
        use_residual=False,

        scoring_hidden_dim=32,
        scoring_num_layers=2,
        use_layer_norm=False,

        scenario_hidden_dim=32,
        scenario_output_dim=16,

        dropout=0.3,
        l1_lambda=0.005,
        weight_decay=1e-4,

        batch_size=128,
        learning_rate=1e-3,
    )
