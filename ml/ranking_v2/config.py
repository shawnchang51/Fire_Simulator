"""
Configuration for Pairwise Ranking Model V2

Enhanced with:
- Cross-Attention settings
- Hard Negative Mining settings
- Auxiliary Task settings
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class RankingV2Config:
    """
    Configuration for the enhanced pairwise ranking model V2.

    New features over V1:
    - Cross-attention between A and B configurations
    - Hard negative mining for better boundary learning
    - Auxiliary task learning (survival_rate, steps, etc.)

    Attributes:
        # Data paths
        data_dir: Root directory containing training data
        floor_plans_dir: Directory containing floor plan NPZ files
        target_grid_size: Target size (H, W) for grid padding

        # Grid encoding
        num_grid_channels: Number of input channels (5 for V2)

        # CNN Encoder
        cnn_channels: Channel dimensions for CNN layers
        latent_dim: Dimension of latent vector (K)
        use_residual: Whether to use residual connections

        # Scenario MLP
        scenario_input_dim: Input dimension for scenario (4 features)
        scenario_hidden_dim: Hidden dimension for scenario MLP
        scenario_output_dim: Output dimension for scenario MLP

        # Scoring Head
        scoring_hidden_dim: Hidden dimension for scoring head
        scoring_num_layers: Number of hidden layers in scoring head
        use_layer_norm: Whether to use LayerNorm in scoring head
        dropout: Dropout rate for scoring head

        # Cross-Attention (NEW)
        use_cross_attention: Whether to enable cross-attention
        attention_heads: Number of attention heads
        attention_dim: Projection dimension for attention
        num_attention_layers: Number of stacked attention layers
        attention_dropout: Dropout rate for attention
        use_attention_ffn: Whether to use FFN after attention

        # Hard Negative Mining (NEW)
        mining_strategy: Mining strategy (online, offline, curriculum, none)
        hard_negative_ratio: Fraction of batch that should be hard negatives
        margin_threshold: Pairs with |score_diff| < threshold are "hard"
        mining_refresh_epochs: Refresh hard negatives every N epochs (offline)
        curriculum_warmup_epochs: Gradually increase hard ratio over N epochs

        # Auxiliary Task (NEW)
        auxiliary_tasks: List of auxiliary tasks to predict
        aux_loss_weight: Base weight for auxiliary losses
        aux_loss_schedule: Schedule for aux weight (constant, warmup, decay)
        aux_hidden_dim: Hidden dimension for auxiliary heads
        aux_survival_weight: Weight for survival_rate auxiliary loss
        aux_steps_weight: Weight for steps auxiliary loss
        aux_fire_damage_weight: Weight for avg_fire_damage auxiliary loss

        # Loss function
        loss_type: Loss function type (ranknet or hinge)
        margin: Margin for hinge loss
        sigma: Scaling factor for RankNet logistic
        label_smoothing: Label smoothing for noisy pairs
        ranking_loss_weight: Weight for main ranking loss

        # Regularization
        l1_lambda: L1 regularization strength for latent sparsity
        weight_decay: L2 regularization (AdamW weight decay)

        # Training
        batch_size: Number of pairs per batch
        gradient_accumulation_steps: Accumulate gradients for larger effective batch
        learning_rate: Initial learning rate
        warmup_epochs: Number of warmup epochs
        epochs: Maximum training epochs
        early_stopping_patience: Patience for early stopping
        num_workers: Number of data loading workers

        # Checkpoint and logging
        checkpoint_dir: Directory for saving checkpoints
        log_dir: Directory for TensorBoard logs

        # Reproducibility
        seed: Random seed for reproducibility

        # Data Augmentation
        augment_shift: Random shift augmentation for training
        augment_rotate90: Random 90-degree rotation augmentation (0°, 90°, 180°, 270°)

        # Normalization stats
        scenario_means: Precomputed scenario means
        scenario_stds: Precomputed scenario stds
    """

    # Data paths
    data_dir: str = "combined_fast"
    floor_plans_dir: str = "combined_fast/floor_plans"
    target_grid_size: Tuple[int, int] = (96, 128)

    # Grid encoding
    num_grid_channels: int = 5  # wall, passable, doors, exits, valid_mask

    # CNN Encoder
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    latent_dim: int = 64
    use_residual: bool = True

    # Scenario MLP
    scenario_input_dim: int = 4  # agent_count, num_fires, fire_spread_rate, fire_discovery_delay
    scenario_hidden_dim: int = 64
    scenario_output_dim: int = 32

    # Scoring Head
    scoring_hidden_dim: int = 64
    scoring_num_layers: int = 3
    use_layer_norm: bool = True
    dropout: float = 0.1

    # === NEW: Cross-Attention Settings ===
    use_cross_attention: bool = True
    attention_heads: int = 4
    attention_dim: int = 64  # Should match latent_dim for residual
    num_attention_layers: int = 2
    attention_dropout: float = 0.1
    use_attention_ffn: bool = True

    # === NEW: Hard Negative Mining Settings ===
    mining_strategy: str = "curriculum"  # "online", "offline", "curriculum", "none"
    hard_negative_ratio: float = 0.5  # Fraction of batch that should be hard
    margin_threshold: float = 0.3  # |score_diff| < threshold = hard
    mining_refresh_epochs: int = 5  # For offline mining
    curriculum_warmup_epochs: int = 10  # Gradually increase hard ratio

    # === NEW: Auxiliary Task Settings ===
    auxiliary_tasks: List[str] = field(default_factory=lambda: ["survival_rate", "steps"])
    # Available: "survival_rate", "steps", "avg_fire_damage"
    aux_loss_weight: float = 0.3  # Base weight for all auxiliary losses
    aux_loss_schedule: str = "warmup"  # "constant", "warmup", "decay"
    aux_hidden_dim: int = 64
    aux_survival_weight: float = 1.0  # Per-task weight multipliers
    aux_steps_weight: float = 0.5
    aux_fire_damage_weight: float = 0.5

    # Loss function
    loss_type: str = "ranknet"  # "ranknet" or "hinge"
    margin: float = 0.1  # For hinge loss
    sigma: float = 1.0  # For RankNet
    label_smoothing: float = 0.0  # Label smoothing
    ranking_loss_weight: float = 1.0  # Weight for main ranking loss

    # Regularization
    l1_lambda: float = 0.001
    weight_decay: float = 5e-5

    # Training
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    warmup_epochs: int = 5
    epochs: int = 100
    early_stopping_patience: int = 15
    num_workers: int = 16

    # Checkpoint and logging
    checkpoint_dir: str = "checkpoints/ranking_v2"
    log_dir: str = "logs/ranking_v2"

    # Reproducibility
    seed: Optional[int] = 42

    # Data Augmentation
    augment_shift: bool = True
    augment_rotate90: bool = False  # Random 90-degree rotation (0°, 90°, 180°, 270°)

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

        # V2-specific validations
        assert self.mining_strategy in ("online", "offline", "curriculum", "none"), \
            f"Unknown mining strategy: {self.mining_strategy}"
        assert 0 <= self.hard_negative_ratio <= 1, "Hard negative ratio must be in [0, 1]"
        assert self.aux_loss_schedule in ("constant", "warmup", "decay"), \
            f"Unknown aux loss schedule: {self.aux_loss_schedule}"
        assert self.attention_heads > 0, "Attention heads must be positive"
        assert self.attention_dim % self.attention_heads == 0, \
            "Attention dim must be divisible by number of heads"

        # Validate auxiliary tasks
        valid_tasks = {"survival_rate", "steps", "avg_fire_damage"}
        for task in self.auxiliary_tasks:
            assert task in valid_tasks, f"Unknown auxiliary task: {task}"


def get_full_config() -> RankingV2Config:
    """
    Full configuration with all V2 features enabled.
    Target: Maximum performance with cross-attention + mining + auxiliary.
    """
    return RankingV2Config(
        # Architecture
        cnn_channels=[32, 64, 128, 256],
        latent_dim=64,
        use_residual=True,

        # Cross-Attention
        use_cross_attention=True,
        attention_heads=4,
        attention_dim=64,
        num_attention_layers=2,
        use_attention_ffn=True,

        # Deeper scoring head
        scoring_hidden_dim=64,
        scoring_num_layers=3,
        use_layer_norm=True,

        # Mining
        mining_strategy="curriculum",
        hard_negative_ratio=0.5,
        margin_threshold=0.3,
        curriculum_warmup_epochs=10,

        # Auxiliary tasks
        auxiliary_tasks=["survival_rate", "steps"],
        aux_loss_weight=0.3,
        aux_loss_schedule="warmup",

        # Regularization
        dropout=0.1,
        l1_lambda=0.001,
        weight_decay=5e-5,

        # Training
        loss_type='ranknet',
        batch_size=128,
        learning_rate=5e-4,
        warmup_epochs=10,
        epochs=150,
        early_stopping_patience=20,

        # Augmentation
        augment_shift=True,
    )


def get_attention_only_config() -> RankingV2Config:
    """
    Configuration with cross-attention but no mining or auxiliary tasks.
    Good for ablation studies.
    """
    return RankingV2Config(
        # Architecture
        cnn_channels=[32, 64, 128, 256],
        latent_dim=64,
        use_residual=True,

        # Cross-Attention enabled
        use_cross_attention=True,
        attention_heads=4,
        attention_dim=64,
        num_attention_layers=2,

        # No mining
        mining_strategy="none",

        # No auxiliary tasks
        auxiliary_tasks=[],
        aux_loss_weight=0.0,

        # Training
        batch_size=128,
        learning_rate=5e-4,
        epochs=100,
    )


def get_mining_only_config() -> RankingV2Config:
    """
    Configuration with hard negative mining but no cross-attention.
    Good for ablation studies.
    """
    return RankingV2Config(
        # Architecture (same as V1 high-capacity)
        cnn_channels=[32, 64, 128, 256],
        latent_dim=64,
        use_residual=True,

        # No cross-attention
        use_cross_attention=False,

        # Mining enabled
        mining_strategy="curriculum",
        hard_negative_ratio=0.5,
        margin_threshold=0.3,
        curriculum_warmup_epochs=10,

        # No auxiliary tasks
        auxiliary_tasks=[],
        aux_loss_weight=0.0,

        # Training
        batch_size=128,
        learning_rate=5e-4,
        epochs=100,
    )


def get_auxiliary_only_config() -> RankingV2Config:
    """
    Configuration with auxiliary tasks but no cross-attention or mining.
    Good for ablation studies.
    """
    return RankingV2Config(
        # Architecture
        cnn_channels=[32, 64, 128, 256],
        latent_dim=64,
        use_residual=True,

        # No cross-attention
        use_cross_attention=False,

        # No mining
        mining_strategy="none",

        # Auxiliary tasks enabled
        auxiliary_tasks=["survival_rate", "steps"],
        aux_loss_weight=0.3,
        aux_loss_schedule="warmup",

        # Training
        batch_size=128,
        learning_rate=5e-4,
        epochs=100,
    )


def get_lightweight_config() -> RankingV2Config:
    """
    Lightweight configuration for faster training/testing.
    """
    return RankingV2Config(
        # Smaller architecture
        cnn_channels=[32, 64, 128],
        latent_dim=32,
        use_residual=True,

        # Lighter cross-attention
        use_cross_attention=True,
        attention_heads=2,
        attention_dim=32,
        num_attention_layers=1,

        # Smaller scoring head
        scoring_hidden_dim=32,
        scoring_num_layers=2,

        # Simple mining
        mining_strategy="curriculum",
        hard_negative_ratio=0.3,

        # Fewer auxiliary tasks
        auxiliary_tasks=["survival_rate"],
        aux_loss_weight=0.2,

        # Faster training
        batch_size=256,
        learning_rate=1e-3,
        epochs=50,
        early_stopping_patience=10,
    )
