"""
CLI Entry Point for Pairwise Ranking Model

Usage:
    # Train model with CLI arguments
    python -m ml.ranking.run_training --mode train --epochs 50

    # Train model with YAML config
    python -m ml.ranking.run_training --mode train --config config.yaml

    # Train with YAML config + CLI overrides (CLI takes priority)
    python -m ml.ranking.run_training --mode train --config config.yaml --epochs 100 --learning-rate 0.0005

    # Evaluate on test set
    python -m ml.ranking.run_training --mode eval --checkpoint checkpoints/ranking/best_model.pt

    # Generate visualizations
    python -m ml.ranking.run_training --mode visualize --checkpoint checkpoints/ranking/best_model.pt --output viz/

Example YAML config (config.yaml):
    # Data paths
    data_dir: combined_fast
    floor_plans_dir: combined_fast/floor_plans
    target_grid_size: [96, 128]
    
    # Model architecture
    cnn_channels: [16, 32, 64]
    latent_dim: 8
    scenario_hidden_dim: 32
    scenario_output_dim: 16
    scoring_hidden_dim: 32
    dropout: 0.3
    
    # Loss function
    loss_type: ranknet  # or 'hinge'
    margin: 0.1
    sigma: 1.0
    
    # Regularization
    l1_lambda: 0.005
    weight_decay: 0.0001
    
    # Training
    batch_size: 128
    learning_rate: 0.001
    warmup_epochs: 5
    epochs: 100
    early_stopping_patience: 15
    num_workers: 4
    
    # Checkpoint and logging
    checkpoint_dir: checkpoints/ranking
    log_dir: logs/ranking
    
    # Reproducibility
    seed: 42
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from .config import RankingConfig
from .dataset import (
    create_pairwise_dataloaders,
    SingleConfigDataset,
    compute_scenario_stats
)
from .model import SiameseRanker
from .train import train_ranking_model, load_checkpoint
from .evaluate import evaluate_pairwise, evaluate_per_plan_ranking, print_evaluation_report
from .visualize import generate_all_visualizations, plot_training_history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pairwise Ranking Model for Floor Plan Evacuation Quality"
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

    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='combined_fast',
        help="Root directory containing training data"
    )
    parser.add_argument(
        '--floor-plans-dir',
        type=str,
        default='combined_fast/floor_plans',
        help="Directory containing floor plan NPZ files"
    )

    # Model arguments
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=8,
        help="Latent dimension K (default: 8)"
    )
    parser.add_argument(
        '--loss-type',
        type=str,
        choices=['ranknet', 'hinge'],
        default='ranknet',
        help="Loss function type (default: ranknet)"
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=0.1,
        help="Margin for hinge loss (default: 0.1)"
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=1.0,
        help="Sigma for RankNet loss (default: 1.0)"
    )

    # Training arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help="Batch size (default: 128)"
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="Maximum epochs (default: 100)"
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=5,
        help="Warmup epochs (default: 5)"
    )
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=15,
        help="Early stopping patience (default: 15)"
    )
    parser.add_argument(
        '--l1-lambda',
        type=float,
        default=0.005,
        help="L1 regularization for latent sparsity (default: 0.005)"
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)"
    )

    # Checkpoint arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help="Path to checkpoint for eval/visualize mode"
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/ranking',
        help="Directory for saving checkpoints"
    )

    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        default='viz',
        help="Output directory for visualizations"
    )

    # Other arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)"
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


def save_config_to_yaml(config: RankingConfig, output_path: str, extra_params: Optional[Dict] = None) -> None:
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
        
        # Scenario MLP
        'scenario_input_dim': config.scenario_input_dim,
        'scenario_hidden_dim': config.scenario_hidden_dim,
        'scenario_output_dim': config.scenario_output_dim,
        
        # Scoring Head
        'scoring_hidden_dim': config.scoring_hidden_dim,
        'dropout': config.dropout,
        
        # Loss function
        'loss_type': config.loss_type,
        'margin': config.margin,
        'sigma': config.sigma,
        
        # Regularization
        'l1_lambda': config.l1_lambda,
        'weight_decay': config.weight_decay,
        
        # Training
        'batch_size': config.batch_size,
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


def create_config_from_args(args) -> RankingConfig:
    """
    Create RankingConfig from command line arguments and optional YAML config.
    
    Priority (highest to lowest):
    1. CLI arguments (if explicitly provided)
    2. YAML config file values
    3. RankingConfig defaults
    """
    # Start with defaults from RankingConfig
    config_kwargs = {}
    
    # Load YAML config if provided
    yaml_config = {}
    if args.config:
        print(f"Loading config from: {args.config}")
        yaml_config = load_yaml_config(args.config)
        
        # Map YAML keys to RankingConfig fields
        yaml_mapping = {
            'data_dir': 'data_dir',
            'floor_plans_dir': 'floor_plans_dir',
            'target_grid_size': 'target_grid_size',
            'num_grid_channels': 'num_grid_channels',
            'cnn_channels': 'cnn_channels',
            'latent_dim': 'latent_dim',
            'scenario_input_dim': 'scenario_input_dim',
            'scenario_hidden_dim': 'scenario_hidden_dim',
            'scenario_output_dim': 'scenario_output_dim',
            'scoring_hidden_dim': 'scoring_hidden_dim',
            'dropout': 'dropout',
            'loss_type': 'loss_type',
            'margin': 'margin',
            'sigma': 'sigma',
            'l1_lambda': 'l1_lambda',
            'weight_decay': 'weight_decay',
            'batch_size': 'batch_size',
            'learning_rate': 'learning_rate',
            'warmup_epochs': 'warmup_epochs',
            'epochs': 'epochs',
            'early_stopping_patience': 'early_stopping_patience',
            'num_workers': 'num_workers',
            'checkpoint_dir': 'checkpoint_dir',
            'log_dir': 'log_dir',
            'seed': 'seed',
            'augment_shift': 'augment_shift',
            'scenario_means': 'scenario_means',
            'scenario_stds': 'scenario_stds',
        }
        
        for yaml_key, config_key in yaml_mapping.items():
            if yaml_key in yaml_config:
                value = yaml_config[yaml_key]
                # Convert list to tuple for target_grid_size
                if yaml_key == 'target_grid_size' and isinstance(value, list):
                    value = tuple(value)
                config_kwargs[config_key] = value
    
    # CLI arguments override YAML config (check against parser defaults)
    parser_defaults = {
        'data_dir': 'combined_fast',
        'floor_plans_dir': 'combined_fast/floor_plans',
        'latent_dim': 8,
        'loss_type': 'ranknet',
        'margin': 0.1,
        'sigma': 1.0,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'epochs': 100,
        'warmup_epochs': 5,
        'early_stopping': 15,
        'l1_lambda': 0.005,
        'weight_decay': 1e-4,
        'checkpoint_dir': 'checkpoints/ranking',
        'seed': 42,
        'num_workers': 4,
    }
    
    # Map CLI args to config kwargs (override if not default)
    cli_to_config = [
        ('data_dir', 'data_dir'),
        ('floor_plans_dir', 'floor_plans_dir'),
        ('latent_dim', 'latent_dim'),
        ('loss_type', 'loss_type'),
        ('margin', 'margin'),
        ('sigma', 'sigma'),
        ('batch_size', 'batch_size'),
        ('learning_rate', 'learning_rate'),
        ('epochs', 'epochs'),
        ('warmup_epochs', 'warmup_epochs'),
        ('early_stopping', 'early_stopping_patience'),
        ('l1_lambda', 'l1_lambda'),
        ('weight_decay', 'weight_decay'),
        ('checkpoint_dir', 'checkpoint_dir'),
        ('seed', 'seed'),
        ('num_workers', 'num_workers'),
    ]
    
    for cli_arg, config_key in cli_to_config:
        cli_value = getattr(args, cli_arg, None)
        default_value = parser_defaults.get(cli_arg)
        
        # Override if CLI value is different from default (user explicitly set it)
        # or if no YAML config was provided
        if cli_value is not None:
            if not args.config or cli_value != default_value:
                # User explicitly set CLI value or no YAML config
                config_kwargs[config_key] = cli_value
            elif config_key not in config_kwargs:
                # YAML config provided but doesn't have this key - use CLI default
                config_kwargs[config_key] = cli_value
    
    # Ensure num_workers is always set (fallback to default if still None)
    if config_kwargs.get('num_workers') is None:
        config_kwargs['num_workers'] = 4
    
    return RankingConfig(**config_kwargs)


def mode_train(args):
    """Training mode."""
    print("=" * 60)
    print("PAIRWISE RANKING MODEL - TRAINING")
    print("=" * 60)

    # Set seed
    torch.manual_seed(args.seed)

    # Create config
    config = create_config_from_args(args)

    # Set device
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
    print(f"  Augment shift: {config.augment_shift}")
    print(f"  Device: {device}")
    print()

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader, stats = create_pairwise_dataloaders(config)
    print(f"  Train pairs: {len(train_loader.dataset):,}")
    print(f"  Val pairs: {len(val_loader.dataset):,}")
    print(f"  Test pairs: {len(test_loader.dataset):,}")

    # Train model
    model, history = train_ranking_model(
        config, train_loader, val_loader, device
    )

    # Save config for reproducibility
    config_output_path = Path(config.checkpoint_dir) / "config.yaml"
    save_config_to_yaml(config, str(config_output_path), extra_params={'device': str(device)})
    print(f"Saved config to {config_output_path}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    pairwise_metrics = evaluate_pairwise(model, test_loader, device)
    print_evaluation_report(pairwise_metrics)

    # Save stats
    stats_path = Path(config.checkpoint_dir) / "scenario_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved scenario stats to {stats_path}")

    print("\nTraining complete!")


def mode_eval(args):
    """Evaluation mode."""
    print("=" * 60)
    print("PAIRWISE RANKING MODEL - EVALUATION")
    print("=" * 60)

    if args.checkpoint is None:
        print("Error: --checkpoint required for eval mode")
        sys.exit(1)

    # Set device
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
    config = RankingConfig(**checkpoint['config'])
    config.data_dir = args.data_dir
    config.floor_plans_dir = args.floor_plans_dir

    # Create test dataloader
    print("\nLoading test data...")
    _, _, test_loader, _ = create_pairwise_dataloaders(config, compute_stats=True)
    print(f"  Test pairs: {len(test_loader.dataset):,}")

    # Pairwise evaluation
    print("\nEvaluating pairwise metrics...")
    pairwise_metrics = evaluate_pairwise(model, test_loader, device)

    # Per-plan evaluation (optional)
    ranking_metrics = None
    try:
        data_dir = Path(config.data_dir)
        eval_dataset = SingleConfigDataset(
            simulation_results_file=str(data_dir / "simulation_results.jsonl"),
            floor_plans_dir=config.floor_plans_dir,
            target_size=config.target_grid_size,
            max_configs=10000  # Limit for speed
        )
        print("\nEvaluating per-plan ranking metrics...")
        ranking_metrics = evaluate_per_plan_ranking(model, eval_dataset, device)
    except Exception as e:
        print(f"Skipping per-plan evaluation: {e}")

    # Print report
    print_evaluation_report(pairwise_metrics, ranking_metrics)


def mode_visualize(args):
    """Visualization mode."""
    print("=" * 60)
    print("PAIRWISE RANKING MODEL - VISUALIZATION")
    print("=" * 60)

    if args.checkpoint is None:
        print("Error: --checkpoint required for visualize mode")
        sys.exit(1)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)

    # Create config from checkpoint
    config = RankingConfig(**checkpoint['config'])
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
            max_configs=2000  # Limit for visualization
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
