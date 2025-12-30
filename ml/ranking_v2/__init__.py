"""
Pairwise Ranking Model V2

Enhanced ranking model with:
- Hard Negative Mining: Prioritize difficult-to-distinguish pairs during training
- Cross-Attention Fusion: Enable A/B configurations to interact before scoring
- Auxiliary Task Learning: Multi-task learning to predict survival_rate, steps, etc.

Usage:
    # Training with full features
    python -m ml.ranking_v2.run_training --mode train --preset full

    # Training with specific features
    python -m ml.ranking_v2.run_training --mode train \\
        --mining-strategy curriculum \\
        --use-cross-attention \\
        --auxiliary-tasks survival_rate,steps

    # Evaluation
    python -m ml.ranking_v2.run_training --mode eval \\
        --checkpoint checkpoints/ranking_v2/best_model.pt

Example programmatic usage:
    from ml.ranking_v2 import (
        RankingV2Config,
        CrossAttentionRanker,
        create_pairwise_dataloaders,
        train_ranking_model,
        get_full_config
    )

    # Create config with all V2 features
    config = get_full_config()

    # Create model
    model = CrossAttentionRanker(config)

    # Create dataloaders
    train_loader, val_loader, test_loader, stats = create_pairwise_dataloaders(config)

    # Train
    model, history = train_ranking_model(config, train_loader, val_loader, device)
"""

# Configuration
from .config import (
    RankingV2Config,
    get_full_config,
    get_attention_only_config,
    get_mining_only_config,
    get_auxiliary_only_config,
    get_lightweight_config,
)

# Dataset and Sampling
from .dataset import (
    PairwiseDatasetV2,
    SingleConfigDataset,
    create_pairwise_dataloaders,
    create_train_loader_with_sampler,
    compute_scenario_stats,
)

from .sampler import (
    HardNegativeSampler,
    HardNegativeBatchSampler,
    AdaptiveHardNegativeSampler,
    create_sampler,
)

# Model Components
from .attention import (
    CrossAttentionLayer,
    CrossAttentionBlock,
    CrossAttentionStack,
    FeedForward,
    DifferenceAttention,
)

from .model import (
    CrossAttentionRanker,
    FloorPlanEncoder,
    ScenarioEncoder,
    AuxiliaryHead,
)

# Loss Functions
from .losses import (
    RankNetLoss,
    MarginHingeLoss,
    MultiTaskRankingLoss,
    FocalRankNetLoss,
    get_loss_function,
    create_multi_task_loss,
)

# Training
from .train import (
    train_ranking_model,
    train_epoch,
    validate_epoch,
    save_checkpoint,
    load_checkpoint,
    load_resume_checkpoint,
    get_cosine_schedule_with_warmup,
)

# Evaluation
from .evaluate import (
    evaluate_pairwise,
    evaluate_per_plan_ranking,
    evaluate_auxiliary,
    evaluate_model_full,
    compute_ndcg,
    print_evaluation_report,
)

# Visualization
from .visualize import (
    GradCAM,
    plot_training_history,
    plot_auxiliary_predictions,
    visualize_gradcam_sample,
    plot_latent_pca,
    generate_all_visualizations,
)

__all__ = [
    # Config
    'RankingV2Config',
    'get_full_config',
    'get_attention_only_config',
    'get_mining_only_config',
    'get_auxiliary_only_config',
    'get_lightweight_config',
    # Dataset
    'PairwiseDatasetV2',
    'SingleConfigDataset',
    'create_pairwise_dataloaders',
    'create_train_loader_with_sampler',
    'compute_scenario_stats',
    # Sampler
    'HardNegativeSampler',
    'HardNegativeBatchSampler',
    'AdaptiveHardNegativeSampler',
    'create_sampler',
    # Attention
    'CrossAttentionLayer',
    'CrossAttentionBlock',
    'CrossAttentionStack',
    'FeedForward',
    'DifferenceAttention',
    # Model
    'CrossAttentionRanker',
    'FloorPlanEncoder',
    'ScenarioEncoder',
    'AuxiliaryHead',
    # Losses
    'RankNetLoss',
    'MarginHingeLoss',
    'MultiTaskRankingLoss',
    'FocalRankNetLoss',
    'get_loss_function',
    'create_multi_task_loss',
    # Training
    'train_ranking_model',
    'train_epoch',
    'validate_epoch',
    'save_checkpoint',
    'load_checkpoint',
    'load_resume_checkpoint',
    'get_cosine_schedule_with_warmup',
    # Evaluation
    'evaluate_pairwise',
    'evaluate_per_plan_ranking',
    'evaluate_auxiliary',
    'evaluate_model_full',
    'compute_ndcg',
    'print_evaluation_report',
    # Visualization
    'GradCAM',
    'plot_training_history',
    'plot_auxiliary_predictions',
    'visualize_gradcam_sample',
    'plot_latent_pca',
    'generate_all_visualizations',
]

__version__ = '2.0.0'
