"""
Pairwise Ranking Model V2 - Enhanced with Advanced ML Capabilities

Core Features:
- Hard Negative Mining: Prioritize difficult-to-distinguish pairs during training
- Cross-Attention Fusion: Enable A/B configurations to interact before scoring
- Auxiliary Task Learning: Multi-task learning to predict survival_rate, steps, etc.

Advanced Enhancements (V2.1):
- Uncertainty Quantification: MC Dropout, Deep Ensembles, Evidential Learning
- GNN Encoder: Graph-based floor plan representation
- Contrastive Pre-training: SimCLR, MoCo for self-supervised learning
- Active Learning: Smart sample selection to reduce simulation costs
- Configuration Generator: Evolutionary, MCTS, gradient-based optimization
- Explainable AI: Feature attribution, attention analysis, natural language reports
- Multi-Objective Ranking: Pareto optimization for multiple criteria
- Transfer Learning: Cross-building knowledge transfer
- Continual Learning: Online updates without catastrophic forgetting

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

    # Advanced: Uncertainty-aware predictions
    from ml.ranking_v2 import MCDropoutWrapper
    mc_model = MCDropoutWrapper(model, n_samples=30)
    uncertainty = mc_model.predict_with_uncertainty(grid_a, scenario_a, grid_b, scenario_b)

    # Advanced: Generate optimal configurations
    from ml.ranking_v2 import EvolutionaryOptimizer, ConfigurationScorer
    scorer = ConfigurationScorer(model, base_grid, scenario, device)
    optimizer = EvolutionaryOptimizer(scorer, valid_positions)
    result = optimizer.optimize(floor_plan_id)
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

# === NEW: Uncertainty Quantification ===
from .uncertainty import (
    MCDropoutWrapper,
    DeepEnsemble,
    EvidentialRanker,
    EvidentialLoss,
    UncertaintyEstimate,
    UncertaintyCalibrator,
    create_uncertainty_model,
    compute_uncertainty_metrics,
)

# === NEW: GNN Encoder ===
from .gnn_encoder import (
    GNNEncoder,
    GATv2Layer,
    GraphSAGELayer,
    GINLayer,
    HybridEncoder,
    GridToGraphConverter,
    FloorPlanGraph,
    create_gnn_encoder,
)

# === NEW: Contrastive Learning ===
from .contrastive import (
    SimCLRModel,
    MoCoModel,
    NTXentLoss,
    InfoNCELoss,
    FloorPlanAugmenter,
    ContrastivePretrainer,
    create_contrastive_model,
    transfer_pretrained_encoder,
)

# === NEW: Active Learning ===
from .active_learning import (
    ActiveLearningLoop,
    UncertaintySampling,
    QueryByCommittee,
    ExpectedModelChange,
    DiversitySampling,
    BatchModeSampler,
    SimulationOracle,
    create_acquisition_function,
)

# === NEW: Configuration Generation ===
from .config_generator import (
    Configuration,
    ConfigurationScorer,
    EvolutionaryOptimizer,
    MCTSOptimizer,
    GradientOptimizer,
    ConfigurationVAE,
    GenerationResult,
    create_optimizer,
)

# === NEW: Explainability ===
from .explainer import (
    ExplanationPipeline,
    FeatureAttributor,
    AttentionAnalyzer,
    CounterfactualExplainer,
    NaturalLanguageGenerator,
    ExplanationReport,
    Recommendation,
)

# === NEW: Multi-Objective Ranking ===
from .multi_objective import (
    MultiObjectiveRanker,
    MultiObjectiveHead,
    MultiObjectiveLoss,
    ParetoOptimizer,
    PreferenceLearner,
    ObjectiveConfig,
    ObjectiveType,
    create_default_objectives,
    create_multi_objective_ranker,
)

# === NEW: Transfer Learning ===
from .transfer import (
    TransferLearningPipeline,
    DomainAdaptationRanker,
    MAML,
    FeatureExtractor,
    DomainBank,
    TransferResult,
)

# === NEW: Continual Learning ===
from .continual import (
    ContinualLearner,
    ExperienceReplayBuffer,
    EWC,
    ProgressiveNetwork,
    KnowledgeDistillation,
    OnlineLearner,
    create_continual_learner,
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
    # === NEW: Uncertainty ===
    'MCDropoutWrapper',
    'DeepEnsemble',
    'EvidentialRanker',
    'EvidentialLoss',
    'UncertaintyEstimate',
    'UncertaintyCalibrator',
    'create_uncertainty_model',
    'compute_uncertainty_metrics',
    # === NEW: GNN ===
    'GNNEncoder',
    'GATv2Layer',
    'GraphSAGELayer',
    'GINLayer',
    'HybridEncoder',
    'GridToGraphConverter',
    'FloorPlanGraph',
    'create_gnn_encoder',
    # === NEW: Contrastive ===
    'SimCLRModel',
    'MoCoModel',
    'NTXentLoss',
    'InfoNCELoss',
    'FloorPlanAugmenter',
    'ContrastivePretrainer',
    'create_contrastive_model',
    'transfer_pretrained_encoder',
    # === NEW: Active Learning ===
    'ActiveLearningLoop',
    'UncertaintySampling',
    'QueryByCommittee',
    'ExpectedModelChange',
    'DiversitySampling',
    'BatchModeSampler',
    'SimulationOracle',
    'create_acquisition_function',
    # === NEW: Config Generation ===
    'Configuration',
    'ConfigurationScorer',
    'EvolutionaryOptimizer',
    'MCTSOptimizer',
    'GradientOptimizer',
    'ConfigurationVAE',
    'GenerationResult',
    'create_optimizer',
    # === NEW: Explainability ===
    'ExplanationPipeline',
    'FeatureAttributor',
    'AttentionAnalyzer',
    'CounterfactualExplainer',
    'NaturalLanguageGenerator',
    'ExplanationReport',
    'Recommendation',
    # === NEW: Multi-Objective ===
    'MultiObjectiveRanker',
    'MultiObjectiveHead',
    'MultiObjectiveLoss',
    'ParetoOptimizer',
    'PreferenceLearner',
    'ObjectiveConfig',
    'ObjectiveType',
    'create_default_objectives',
    'create_multi_objective_ranker',
    # === NEW: Transfer Learning ===
    'TransferLearningPipeline',
    'DomainAdaptationRanker',
    'MAML',
    'FeatureExtractor',
    'DomainBank',
    'TransferResult',
    # === NEW: Continual Learning ===
    'ContinualLearner',
    'ExperienceReplayBuffer',
    'EWC',
    'ProgressiveNetwork',
    'KnowledgeDistillation',
    'OnlineLearner',
    'create_continual_learner',
]

__version__ = '2.1.0'
