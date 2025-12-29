"""
Main entry point for training the Fire Simulation Surrogate Model

Usage:
    python -m ml.run_training
    python -m ml.run_training --epochs 50 --batch_size 128
    python -m ml.run_training --max_plans 100  # Quick test with subset
    python -m ml.run_training --config config.yaml  # Load from YAML config
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from .config import ModelConfig
from .dataset import create_dataloaders
from .model import create_model
from .train import train_model, load_checkpoint
from .evaluate import evaluate_model, print_evaluation_report


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Fire Simulation Surrogate Model")

    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides other arguments)")

    # Data
    parser.add_argument("--data_dir", type=str, default="combined_fast",
                        help="Directory containing training data")
    parser.add_argument("--max_plans", type=int, default=None,
                        help="Limit number of floor plans (default: all)")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")

    # Model
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")

    # System
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, default: auto)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")

    # Evaluation
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate existing checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for evaluation")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load YAML config if provided
    yaml_config = {}
    if args.config:
        print(f"Loading config from: {args.config}")
        yaml_config = load_yaml_config(args.config)

    # Helper function to get config value (YAML takes priority, then CLI args)
    def get_config(key, default=None):
        if key in yaml_config:
            return yaml_config[key]
        return getattr(args, key, default)

    # Create config
    config = ModelConfig(
        data_dir=get_config('data_dir', 'combined_fast'),
        floor_plans_dir=get_config('floor_plans_dir', f"{get_config('data_dir', 'combined_fast')}/floor_plans"),
        epochs=get_config('epochs', 100),
        batch_size=get_config('batch_size', 64),
        learning_rate=get_config('lr', 1e-3),
        weight_decay=get_config('weight_decay', 1e-4),
        early_stopping_patience=get_config('patience', 10),
        dropout=get_config('dropout', 0.2),
        num_workers=get_config('num_workers', 4),
        checkpoint_dir=get_config('checkpoint_dir', 'checkpoints'),
        max_plans=get_config('max_plans', None)
    )

    # Device
    device_str = get_config('device', None)
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Eval only mode
    eval_only = get_config('eval_only', False)
    checkpoint_path = get_config('checkpoint', None)

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, stats = create_dataloaders(config)

    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Val samples: {len(val_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")

    # Print normalization stats
    print("\nNormalization statistics:")
    print("  Scenario (agent_count, num_fires, fire_spread_rate, fire_discovery_delay):")
    print(f"    Means: {stats['scenario_stats']['means']}")
    print(f"    Stds:  {stats['scenario_stats']['stds']}")
    print("  Targets (survival_rate, avg_evacuation_time, steps, avg_fire_damage):")
    print(f"    Means: {stats['target_stats']['means']}")
    print(f"    Stds:  {stats['target_stats']['stds']}")

    # Create model
    model = create_model(config)
    print(f"\nModel parameters: {model.count_parameters():,}")

    if eval_only:
        # Load checkpoint and evaluate
        ckpt_path = checkpoint_path or Path(config.checkpoint_dir) / "best_model.pt"
        print(f"\nLoading checkpoint: {ckpt_path}")
        load_checkpoint(model, ckpt_path, device)

        print("\nEvaluating on test set...")
        metrics = evaluate_model(
            model, test_loader, device, config,
            denormalize=True, target_stats=stats['target_stats']
        )
        print_evaluation_report(metrics, "Test Set Evaluation (Denormalized)")
    else:
        # Train model
        history = train_model(model, train_loader, val_loader, config, device)

        # Save training history
        history_path = Path(config.checkpoint_dir) / "training_history.json"
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {
            k: [x.tolist() if hasattr(x, 'tolist') else x for x in v]
            for k, v in history.items()
        }
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        print(f"Training history saved to: {history_path}")

        # Final evaluation on test set
        print("\n" + "=" * 70)
        print(" FINAL TEST SET EVALUATION")
        print("=" * 70)

        # Evaluate with normalized values (what the model sees)
        metrics_normalized = evaluate_model(model, test_loader, device, config)
        print_evaluation_report(metrics_normalized, "Test Set Evaluation (Normalized)")

        # Evaluate with denormalized values (actual scale)
        metrics_denormalized = evaluate_model(
            model, test_loader, device, config,
            denormalize=True, target_stats=stats['target_stats']
        )
        print_evaluation_report(metrics_denormalized, "Test Set Evaluation (Denormalized)")

        # Save stats for inference
        stats_path = Path(config.checkpoint_dir) / "normalization_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Normalization stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
