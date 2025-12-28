"""
CLI Entry Point for Pairwise Ranking Model

Usage:
    # Train model
    python -m ml.ranking.run_training --mode train --epochs 50

    # Evaluate on test set
    python -m ml.ranking.run_training --mode eval --checkpoint checkpoints/ranking/best_model.pt

    # Generate visualizations
    python -m ml.ranking.run_training --mode visualize --checkpoint checkpoints/ranking/best_model.pt --output viz/
"""

import argparse
import json
import sys
from pathlib import Path

import torch

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


def create_config_from_args(args) -> RankingConfig:
    """Create RankingConfig from command line arguments."""
    return RankingConfig(
        data_dir=args.data_dir,
        floor_plans_dir=args.floor_plans_dir,
        latent_dim=args.latent_dim,
        loss_type=args.loss_type,
        margin=args.margin,
        sigma=args.sigma,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.early_stopping,
        l1_lambda=args.l1_lambda,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        num_workers=args.num_workers
    )


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
