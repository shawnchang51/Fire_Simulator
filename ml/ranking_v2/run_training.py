"""
CLI Entry Point for Pairwise Ranking Model V2

Enhanced version with:
- Hard Negative Mining
- Cross-Attention Fusion
- Auxiliary Task Learning

Usage:
    # Train with all V2 features (full preset)
    python -m ml.ranking_v2.run_training --mode train --preset full

    # Train with only cross-attention
    python -m ml.ranking_v2.run_training --mode train --preset attention_only

    # Train with only hard negative mining
    python -m ml.ranking_v2.run_training --mode train --preset mining_only

    # Train with custom configuration
    python -m ml.ranking_v2.run_training --mode train --epochs 100 \
        --mining-strategy curriculum --use-cross-attention --auxiliary-tasks survival_rate,steps

    # Train with YAML config
    python -m ml.ranking_v2.run_training --mode train --config config_v2.yaml

    # Resume training
    python -m ml.ranking_v2.run_training --mode train --resume-from checkpoints/ranking_v2

    # Evaluate
    python -m ml.ranking_v2.run_training --mode eval --checkpoint checkpoints/ranking_v2/best_model.pt

    # Visualize (includes attention maps and auxiliary predictions)
    python -m ml.ranking_v2.run_training --mode visualize --checkpoint checkpoints/ranking_v2/best_model.pt

Example YAML config:
    # Data paths
    data_dir: combined_fast
    floor_plans_dir: combined_fast/floor_plans

    # V2 Cross-Attention settings
    use_cross_attention: true
    attention_heads: 4
    attention_dim: 64
    num_attention_layers: 2

    # V2 Hard Negative Mining settings
    mining_strategy: curriculum
    hard_negative_ratio: 0.5
    margin_threshold: 0.3
    curriculum_warmup_epochs: 10

    # V2 Auxiliary Task settings
    auxiliary_tasks: [survival_rate, steps, avg_fire_damage]
    aux_loss_weight: 0.3
    aux_loss_schedule: warmup

    # Training
    epochs: 100
    batch_size: 128
    learning_rate: 0.001
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from .config import (
    RankingV2Config,
    get_full_config,
    get_attention_only_config,
    get_mining_only_config,
    get_auxiliary_only_config,
    get_lightweight_config
)
from .dataset import (
    create_pairwise_dataloaders,
    SingleConfigDataset,
    compute_scenario_stats
)
from .model import CrossAttentionRanker
from .train import train_ranking_model, load_checkpoint, load_resume_checkpoint
from .evaluate import (
    evaluate_pairwise,
    evaluate_per_plan_ranking,
    evaluate_auxiliary,
    print_evaluation_report
)
from .visualize import (
    generate_all_visualizations,
    plot_training_history,
    visualize_gradcam_sample,
    generate_gradcam_from_floor_plans
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pairwise Ranking Model V2 with Hard Negative Mining, "
                    "Cross-Attention, and Auxiliary Tasks"
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval', 'visualize'],
        required=True,
        help="Mode: train, eval, or visualize"
    )

    # Config file argument
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="Path to YAML config file. CLI arguments override config file values."
    )

    # Configuration preset
    parser.add_argument(
        '--preset',
        type=str,
        choices=['full', 'attention_only', 'mining_only', 'auxiliary_only', 'lightweight'],
        default=None,
        help="Use predefined config preset: "
             "'full' (all V2 features), "
             "'attention_only' (cross-attention only), "
             "'mining_only' (hard negative mining only), "
             "'auxiliary_only' (auxiliary tasks only), "
             "'lightweight' (minimal config for testing)"
    )

    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help="Root directory containing training data (default: combined_fast)"
    )
    parser.add_argument(
        '--floor-plans-dir',
        type=str,
        default=None,
        help="Directory containing floor plan NPZ files (default: combined_fast/floor_plans)"
    )

    # =====================
    # V2 Cross-Attention arguments
    # =====================
    parser.add_argument(
        '--use-cross-attention',
        action='store_true',
        help="Enable cross-attention between A and B configurations"
    )
    parser.add_argument(
        '--no-cross-attention',
        action='store_true',
        help="Disable cross-attention (default if preset not specified)"
    )
    parser.add_argument(
        '--attention-heads',
        type=int,
        default=None,
        help="Number of attention heads (default: 4)"
    )
    parser.add_argument(
        '--attention-dim',
        type=int,
        default=None,
        help="Attention dimension per head (default: 64)"
    )
    parser.add_argument(
        '--num-attention-layers',
        type=int,
        default=None,
        help="Number of cross-attention layers (default: 2)"
    )

    # =====================
    # V2 Hard Negative Mining arguments
    # =====================
    parser.add_argument(
        '--mining-strategy',
        type=str,
        choices=['none', 'online', 'offline', 'curriculum'],
        default=None,
        help="Hard negative mining strategy: "
             "'none' (uniform sampling), "
             "'online' (sample hard from current batch), "
             "'offline' (precompute hardness), "
             "'curriculum' (gradual hardness increase)"
    )
    parser.add_argument(
        '--hard-negative-ratio',
        type=float,
        default=None,
        help="Ratio of hard negatives in each batch (default: 0.5)"
    )
    parser.add_argument(
        '--margin-threshold',
        type=float,
        default=None,
        help="Score difference threshold for hard negatives (default: 0.3)"
    )
    parser.add_argument(
        '--curriculum-warmup-epochs',
        type=int,
        default=None,
        help="Epochs before full hard negative ratio (default: 10)"
    )

    # =====================
    # V2 Auxiliary Task arguments
    # =====================
    parser.add_argument(
        '--auxiliary-tasks',
        type=str,
        default=None,
        help="Comma-separated list of auxiliary tasks: survival_rate,steps,avg_fire_damage"
    )
    parser.add_argument(
        '--aux-loss-weight',
        type=float,
        default=None,
        help="Weight for auxiliary losses (default: 0.3)"
    )
    parser.add_argument(
        '--aux-loss-schedule',
        type=str,
        choices=['constant', 'warmup', 'decay'],
        default=None,
        help="Auxiliary loss weight schedule (default: constant)"
    )

    # =====================
    # Model architecture arguments
    # =====================
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=None,
        help="Latent dimension (default: 64)"
    )
    parser.add_argument(
        '--use-residual',
        action='store_true',
        help="Use residual blocks in encoder"
    )
    parser.add_argument(
        '--loss-type',
        type=str,
        choices=['ranknet', 'hinge'],
        default=None,
        help="Loss function type (default: ranknet)"
    )

    # =====================
    # Training arguments
    # =====================
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help="Batch size (default: 128)"
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help="Maximum epochs (default: 100)"
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=None,
        help="Warmup epochs for LR scheduler (default: 5)"
    )
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=None,
        help="Early stopping patience (default: 15)"
    )
    parser.add_argument(
        '--l1-lambda',
        type=float,
        default=None,
        help="L1 regularization for latent sparsity (default: 0.001)"
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=None,
        help="Weight decay (default: 1e-4)"
    )
    parser.add_argument(
        '--gradient-accumulation',
        type=int,
        default=None,
        help="Gradient accumulation steps (default: 1)"
    )

    # =====================
    # Checkpoint arguments
    # =====================
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help="Path to checkpoint for eval/visualize mode"
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help="Directory for saving checkpoints (default: checkpoints/ranking_v2)"
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help="Resume training from checkpoint directory"
    )
    parser.add_argument(
        '--resume-in-place',
        action='store_true',
        help="Save resumed checkpoints to same directory"
    )

    # =====================
    # Output arguments
    # =====================
    parser.add_argument(
        '--output',
        type=str,
        default='viz_v2',
        help="Output directory for visualizations"
    )

    # =====================
    # Other arguments
    # =====================
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help="Number of data loading workers (default: 16)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Device to use (default: auto-detect)"
    )

    return parser.parse_args()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config if config else {}


def save_config_to_yaml(
    config: RankingV2Config,
    output_path: str,
    extra_params: Optional[Dict] = None
) -> None:
    """Save current configuration to a YAML file for reproducibility."""
    config_dict = {
        # Data paths
        'data_dir': config.data_dir,
        'floor_plans_dir': config.floor_plans_dir,
        'target_grid_size': list(config.target_grid_size),

        # Grid encoding
        'num_grid_channels': config.num_grid_channels,

        # CNN Encoder
        'cnn_channels': config.cnn_channels,
        'latent_dim': config.latent_dim,
        'use_residual': config.use_residual,

        # Scenario MLP
        'scenario_input_dim': config.scenario_input_dim,
        'scenario_hidden_dim': config.scenario_hidden_dim,
        'scenario_output_dim': config.scenario_output_dim,

        # Scoring Head
        'scoring_hidden_dim': config.scoring_hidden_dim,
        'scoring_num_layers': config.scoring_num_layers,
        'use_layer_norm': config.use_layer_norm,
        'dropout': config.dropout,

        # V2: Cross-Attention
        'use_cross_attention': config.use_cross_attention,
        'attention_heads': config.attention_heads,
        'attention_dim': config.attention_dim,
        'num_attention_layers': config.num_attention_layers,
        'attention_dropout': config.attention_dropout,
        'use_attention_ffn': config.use_attention_ffn,

        # V2: Hard Negative Mining
        'mining_strategy': config.mining_strategy,
        'hard_negative_ratio': config.hard_negative_ratio,
        'margin_threshold': config.margin_threshold,
        'curriculum_warmup_epochs': config.curriculum_warmup_epochs,
        'mining_refresh_epochs': config.mining_refresh_epochs,

        # V2: Auxiliary Tasks
        'auxiliary_tasks': config.auxiliary_tasks,
        'aux_loss_weight': config.aux_loss_weight,
        'aux_hidden_dim': config.aux_hidden_dim,
        'aux_loss_schedule': config.aux_loss_schedule,
        'aux_survival_weight': config.aux_survival_weight,
        'aux_steps_weight': config.aux_steps_weight,
        'aux_fire_damage_weight': config.aux_fire_damage_weight,

        # Loss function
        'loss_type': config.loss_type,
        'margin': config.margin,
        'sigma': config.sigma,
        'label_smoothing': config.label_smoothing,
        'ranking_loss_weight': config.ranking_loss_weight,

        # Regularization
        'l1_lambda': config.l1_lambda,
        'weight_decay': config.weight_decay,

        # Training
        'batch_size': config.batch_size,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'learning_rate': config.learning_rate,
        'warmup_epochs': config.warmup_epochs,
        'epochs': config.epochs,
        'early_stopping_patience': config.early_stopping_patience,
        'num_workers': config.num_workers,

        # Checkpoint and logging
        'checkpoint_dir': config.checkpoint_dir,
        'log_dir': config.log_dir,

        # Reproducibility
        'seed': config.seed,

        # Data Augmentation
        'augment_shift': config.augment_shift,
    }

    if extra_params:
        config_dict.update(extra_params)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def create_config_from_args(args) -> RankingV2Config:
    """
    Create RankingV2Config from command line arguments, preset, and optional YAML config.

    Priority (highest to lowest):
    1. CLI arguments (if explicitly provided)
    2. YAML config file values
    3. Preset configuration
    4. RankingV2Config defaults
    """
    # Start with preset if specified
    preset = getattr(args, 'preset', None)
    if preset == 'full':
        base_config = get_full_config()
        print("Using preset: full (all V2 features)")
    elif preset == 'attention_only':
        base_config = get_attention_only_config()
        print("Using preset: attention_only")
    elif preset == 'mining_only':
        base_config = get_mining_only_config()
        print("Using preset: mining_only")
    elif preset == 'auxiliary_only':
        base_config = get_auxiliary_only_config()
        print("Using preset: auxiliary_only")
    elif preset == 'lightweight':
        base_config = get_lightweight_config()
        print("Using preset: lightweight")
    else:
        base_config = None

    # Extract config fields from preset
    if base_config is not None:
        config_kwargs = {
            field: getattr(base_config, field)
            for field in base_config.__dataclass_fields__
            if getattr(base_config, field) is not None
        }
    else:
        config_kwargs = {}

    # Load YAML config if provided (overrides preset)
    if args.config:
        print(f"Loading config from: {args.config}")
        yaml_config = load_yaml_config(args.config)

        # Map YAML keys to config fields
        yaml_mapping = [
            'data_dir', 'floor_plans_dir', 'target_grid_size',
            'num_grid_channels', 'cnn_channels', 'latent_dim', 'use_residual',
            'scenario_input_dim', 'scenario_hidden_dim', 'scenario_output_dim',
            'scoring_hidden_dim', 'scoring_num_layers', 'use_layer_norm', 'dropout',
            # V2 fields
            'use_cross_attention', 'attention_heads', 'attention_dim',
            'num_attention_layers', 'attention_dropout', 'use_attention_ffn',
            'mining_strategy', 'hard_negative_ratio', 'margin_threshold',
            'curriculum_warmup_epochs', 'mining_refresh_epochs',
            'auxiliary_tasks', 'aux_loss_weight', 'aux_hidden_dim',
            'aux_loss_schedule', 'aux_survival_weight', 'aux_steps_weight',
            'aux_fire_damage_weight',
            # Loss and training
            'loss_type', 'margin', 'sigma', 'label_smoothing', 'ranking_loss_weight',
            'l1_lambda', 'weight_decay', 'batch_size', 'gradient_accumulation_steps',
            'learning_rate', 'warmup_epochs', 'epochs', 'early_stopping_patience',
            'num_workers', 'checkpoint_dir', 'log_dir', 'seed', 'augment_shift',
            'scenario_means', 'scenario_stds',
        ]

        for key in yaml_mapping:
            if key in yaml_config:
                value = yaml_config[key]
                # Convert list to tuple for target_grid_size
                if key == 'target_grid_size' and isinstance(value, list):
                    value = tuple(value)
                config_kwargs[key] = value

    # CLI arguments override YAML config (only if explicitly set)
    cli_mappings = [
        ('data_dir', 'data_dir'),
        ('floor_plans_dir', 'floor_plans_dir'),
        ('latent_dim', 'latent_dim'),
        ('use_residual', 'use_residual'),
        ('loss_type', 'loss_type'),
        ('batch_size', 'batch_size'),
        ('learning_rate', 'learning_rate'),
        ('epochs', 'epochs'),
        ('warmup_epochs', 'warmup_epochs'),
        ('early_stopping', 'early_stopping_patience'),
        ('l1_lambda', 'l1_lambda'),
        ('weight_decay', 'weight_decay'),
        ('gradient_accumulation', 'gradient_accumulation_steps'),
        ('checkpoint_dir', 'checkpoint_dir'),
        ('seed', 'seed'),
        ('num_workers', 'num_workers'),
        # V2 cross-attention
        ('attention_heads', 'attention_heads'),
        ('attention_dim', 'attention_dim'),
        ('num_attention_layers', 'num_attention_layers'),
        # V2 mining
        ('mining_strategy', 'mining_strategy'),
        ('hard_negative_ratio', 'hard_negative_ratio'),
        ('margin_threshold', 'margin_threshold'),
        ('curriculum_warmup_epochs', 'curriculum_warmup_epochs'),
        # V2 auxiliary
        ('aux_loss_weight', 'aux_loss_weight'),
        ('aux_loss_schedule', 'aux_loss_schedule'),
    ]

    for cli_arg, config_key in cli_mappings:
        cli_value = getattr(args, cli_arg.replace('-', '_'), None)
        if cli_value is not None:
            config_kwargs[config_key] = cli_value

    # Handle boolean flags for cross-attention
    if args.use_cross_attention:
        config_kwargs['use_cross_attention'] = True
    elif args.no_cross_attention:
        config_kwargs['use_cross_attention'] = False

    # Handle boolean flag for use_residual
    if args.use_residual:
        config_kwargs['use_residual'] = True

    # Handle auxiliary tasks as comma-separated string
    if args.auxiliary_tasks:
        tasks = [t.strip() for t in args.auxiliary_tasks.split(',')]
        config_kwargs['auxiliary_tasks'] = tasks

    # Set default checkpoint_dir if not specified
    if 'checkpoint_dir' not in config_kwargs or config_kwargs['checkpoint_dir'] is None:
        config_kwargs['checkpoint_dir'] = 'checkpoints/ranking_v2'

    # Set default log_dir
    if 'log_dir' not in config_kwargs or config_kwargs['log_dir'] is None:
        config_kwargs['log_dir'] = 'logs/ranking_v2'

    return RankingV2Config(**config_kwargs)


def mode_train(args):
    """Training mode."""
    print("=" * 60)

    is_resuming = args.resume_from is not None

    if is_resuming:
        print("PAIRWISE RANKING MODEL V2 - RESUME TRAINING")
        print("=" * 60)

        resume_dir = Path(args.resume_from)

        # Load config from resume directory
        resume_config_file = resume_dir / "config.yaml"
        if resume_config_file.exists():
            print(f"Loading config from: {resume_config_file}")
            original_config_arg = args.config
            args.config = str(resume_config_file)
            config = create_config_from_args(args)
            args.config = original_config_arg
        else:
            print("WARNING: No config.yaml found in resume directory")
            config = create_config_from_args(args)

        # Set seed from config
        torch.manual_seed(config.seed)

        # Set device
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load resume checkpoint
        model, optimizer, history, start_epoch, best_val_auc, scenario_stats = \
            load_resume_checkpoint(str(resume_dir), config, device)

        # Determine checkpoint directory
        if args.resume_in_place:
            config.checkpoint_dir = str(resume_dir)
            config.log_dir = str(resume_dir.parent / "logs" / resume_dir.name)
            print(f"\nSaving to same directory: {config.checkpoint_dir}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_dir_name = f"{resume_dir.name}_resumed_{timestamp}"
            config.checkpoint_dir = str(resume_dir.parent / new_dir_name)
            config.log_dir = str(resume_dir.parent / "logs" / new_dir_name)
            print(f"\nSaving to new directory: {config.checkpoint_dir}")

        print(f"\nConfiguration:")
        print(f"  Resuming from epoch: {start_epoch}")
        print(f"  Previous best AUC: {best_val_auc:.4f}")
        print(f"  Total epochs: {config.epochs}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Mining strategy: {config.mining_strategy}")
        print(f"  Cross-attention: {config.use_cross_attention}")
        print(f"  Auxiliary tasks: {config.auxiliary_tasks}")
        print(f"  Device: {device}")
        print()

        # Create dataloaders
        print("Loading data with original normalization stats...")
        config.scenario_means = scenario_stats['means']
        config.scenario_stds = scenario_stats['stds']
        train_loader, val_loader, test_loader, _ = create_pairwise_dataloaders(
            config, compute_stats=False
        )

        print(f"  Train pairs: {len(train_loader.dataset):,}")
        print(f"  Val pairs: {len(val_loader.dataset):,}")
        print(f"  Test pairs: {len(test_loader.dataset):,}")

        # Prepare resume checkpoint
        resume_checkpoint = {
            'model': model,
            'optimizer': optimizer,
            'history': history,
            'start_epoch': start_epoch,
            'best_val_auc': best_val_auc
        }

        # Continue training (train_ranking_model handles sampler creation internally)
        model, history = train_ranking_model(
            config, train_loader, val_loader, device,
            scenario_stats=scenario_stats,
            resume_checkpoint=resume_checkpoint,
            train_dataset=train_loader.dataset if config.mining_strategy != 'none' else None
        )

        # Save final config
        config_output_path = Path(config.checkpoint_dir) / "config.yaml"
        save_config_to_yaml(config, str(config_output_path),
                          extra_params={'device': str(device),
                                       'resumed_from': str(resume_dir)})
        print(f"Saved config to {config_output_path}")

        # Save scenario stats
        stats_path = Path(config.checkpoint_dir) / "scenario_stats.json"
        with open(stats_path, 'w') as f:
            json.dump({'scenario_stats': scenario_stats}, f, indent=2)

    else:
        # Fresh training
        print("PAIRWISE RANKING MODEL V2 - TRAINING")
        print("=" * 60)

        config = create_config_from_args(args)
        torch.manual_seed(config.seed)

        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"\nConfiguration:")
        print(f"  Data dir: {config.data_dir}")
        print(f"  Latent dim: {config.latent_dim}")
        print(f"  Loss type: {config.loss_type}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Device: {device}")
        print(f"\nV2 Features:")
        print(f"  Cross-attention: {config.use_cross_attention}")
        if config.use_cross_attention:
            print(f"    Heads: {config.attention_heads}")
            print(f"    Layers: {config.num_attention_layers}")
        print(f"  Mining strategy: {config.mining_strategy}")
        if config.mining_strategy != 'none':
            print(f"    Hard ratio: {config.hard_negative_ratio}")
            print(f"    Threshold: {config.margin_threshold}")
        print(f"  Auxiliary tasks: {config.auxiliary_tasks}")
        if config.auxiliary_tasks:
            print(f"    Aux weight: {config.aux_loss_weight}")
            print(f"    Schedule: {config.aux_loss_schedule}")
        print()

        # Create dataloaders
        print(f"Loading data from {config.data_dir}...")
        train_loader, val_loader, test_loader, stats = create_pairwise_dataloaders(config)

        print(f"  Train pairs: {len(train_loader.dataset):,}")
        print(f"  Val pairs: {len(val_loader.dataset):,}")
        print(f"  Test pairs: {len(test_loader.dataset):,}")

        # Train model (train_ranking_model handles sampler creation internally)
        model, history = train_ranking_model(
            config, train_loader, val_loader, device,
            scenario_stats=stats.get('scenario_stats'),
            train_dataset=train_loader.dataset if config.mining_strategy != 'none' else None
        )

        # Save config
        config_output_path = Path(config.checkpoint_dir) / "config.yaml"
        save_config_to_yaml(config, str(config_output_path),
                          extra_params={'device': str(device)})
        print(f"Saved config to {config_output_path}")

        # Evaluate on test set
        print("\nEvaluating on test set...")
        pairwise_metrics = evaluate_pairwise(model, test_loader, device)

        # Per-plan and auxiliary evaluation
        ranking_metrics = None
        auxiliary_metrics = None
        try:
            data_dir = Path(config.data_dir)
            eval_dataset = SingleConfigDataset(
                simulation_results_file=str(data_dir / "simulation_results.jsonl"),
                floor_plans_dir=config.floor_plans_dir,
                target_size=config.target_grid_size,
                max_configs=10000
            )
            ranking_metrics = evaluate_per_plan_ranking(model, eval_dataset, device)
            if config.auxiliary_tasks:
                auxiliary_metrics = evaluate_auxiliary(model, eval_dataset, device)
        except Exception as e:
            print(f"Skipping per-plan evaluation: {e}")

        print_evaluation_report(pairwise_metrics, ranking_metrics, auxiliary_metrics)

        # Save stats
        stats_path = Path(config.checkpoint_dir) / "scenario_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

    print("\nTraining complete!")


def mode_eval(args):
    """Evaluation mode."""
    print("=" * 60)
    print("PAIRWISE RANKING MODEL V2 - EVALUATION")
    print("=" * 60)

    if args.checkpoint is None:
        print("Error: --checkpoint required for eval mode")
        sys.exit(1)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    print(f"  Loaded from epoch {checkpoint['epoch'] + 1}")
    print(f"  Best val AUC: {checkpoint['val_auc']:.4f}")

    # Create config from checkpoint
    config = RankingV2Config(**checkpoint['config'])
    config.data_dir = args.data_dir
    config.floor_plans_dir = args.floor_plans_dir

    print(f"\nV2 Features in model:")
    print(f"  Cross-attention: {config.use_cross_attention}")
    print(f"  Auxiliary tasks: {config.auxiliary_tasks}")

    # Create test dataloader
    print("\nLoading test data...")
    _, _, test_loader, _ = create_pairwise_dataloaders(config, compute_stats=True)
    print(f"  Test pairs: {len(test_loader.dataset):,}")

    # Pairwise evaluation
    print("\nEvaluating pairwise metrics...")
    pairwise_metrics = evaluate_pairwise(model, test_loader, device)

    # Per-plan and auxiliary evaluation
    ranking_metrics = None
    auxiliary_metrics = None
    try:
        data_dir = Path(config.data_dir)
        eval_dataset = SingleConfigDataset(
            simulation_results_file=str(data_dir / "simulation_results.jsonl"),
            floor_plans_dir=config.floor_plans_dir,
            target_size=config.target_grid_size,
            max_configs=10000
        )
        print("\nEvaluating per-plan ranking metrics...")
        ranking_metrics = evaluate_per_plan_ranking(model, eval_dataset, device)

        if config.auxiliary_tasks:
            print("\nEvaluating auxiliary task metrics...")
            auxiliary_metrics = evaluate_auxiliary(model, eval_dataset, device)
    except Exception as e:
        print(f"Skipping per-plan evaluation: {e}")

    print_evaluation_report(pairwise_metrics, ranking_metrics, auxiliary_metrics)


def mode_visualize(args):
    """Visualization mode."""
    print("=" * 60)
    print("PAIRWISE RANKING MODEL V2 - VISUALIZATION")
    print("=" * 60)

    if args.checkpoint is None:
        print("Error: --checkpoint required for visualize mode")
        sys.exit(1)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)

    # Create config from checkpoint
    config = RankingV2Config(**checkpoint['config'])
    config.data_dir = args.data_dir
    config.floor_plans_dir = args.floor_plans_dir

    # Load history
    history = checkpoint.get('history', {})

    # Create evaluation dataset
    print("\nLoading evaluation data...")
    data_dir = Path(config.data_dir)
    try:
        eval_dataset = SingleConfigDataset(
            simulation_results_file=str(data_dir / "simulation_results.jsonl"),
            floor_plans_dir=config.floor_plans_dir,
            target_size=config.target_grid_size,
            max_configs=2000
        )
        print(f"  Loaded {len(eval_dataset):,} configs")
    except Exception as e:
        print(f"Failed to load evaluation data: {e}")
        eval_dataset = None

    # Generate visualizations
    output_dir = args.output
    print(f"\nGenerating visualizations to {output_dir}...")

    if history:
        plot_training_history(
            history,
            output_path=str(Path(output_dir) / "training_history.png")
        )

    if eval_dataset is not None:
        generate_all_visualizations(
            model=model,
            dataset=eval_dataset,
            history=history,
            output_dir=output_dir,
            n_samples=500,
            device=device
        )
    else:
        # Fallback: generate GradCAM directly from floor plan NPZ files
        print("\nGenerating GradCAM from floor plans (no simulation_results.jsonl)...")
        generate_gradcam_from_floor_plans(
            model=model,
            floor_plans_dir=config.floor_plans_dir,
            output_dir=output_dir,
            target_size=config.target_grid_size,
            n_samples=5,
            device=device
        )

    print("\nVisualization complete!")


def main():
    """Main entry point."""
    args = parse_args()

    if args.mode == 'train':
        mode_train(args)
    elif args.mode == 'eval':
        mode_eval(args)
    elif args.mode == 'visualize':
        mode_visualize(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == '__main__':
    main()
