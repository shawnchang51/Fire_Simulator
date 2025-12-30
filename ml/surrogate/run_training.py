"""
Main entry point for training the Fire Simulation Surrogate Model

Usage:
    python -m ml.surrogate.run_training
    python -m ml.surrogate.run_training --epochs 50 --batch_size 128
    python -m ml.surrogate.run_training --max_plans 100  # Quick test with subset
    python -m ml.surrogate.run_training --config config.yaml  # Load from YAML config

Visualization:
    python -m ml.surrogate.run_training --visualize
    python -m ml.surrogate.run_training --visualize --viz_samples 0,5,10
    python -m ml.surrogate.run_training --visualize --viz_samples random:5
    python -m ml.surrogate.run_training --visualize --viz_type gradcam
    python -m ml.surrogate.run_training --visualize --viz_output ./my_viz/
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


def parse_sample_indices(viz_samples: str, dataset_size: int) -> list:
    """Parse visualization sample indices from string.

    Supports:
        - Comma-separated: "0,1,2,5,10"
        - Random: "random:5"
    """
    import random

    if viz_samples.startswith("random:"):
        n = int(viz_samples.split(":")[1])
        return random.sample(range(dataset_size), min(n, dataset_size))
    else:
        return [int(x.strip()) for x in viz_samples.split(",")]


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

    # Visualization
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations (GradCAM and Counterfactual)")
    parser.add_argument("--viz_samples", type=str, default="0,1,2",
                        help="Sample indices to visualize (comma-separated, or 'random:N')")
    parser.add_argument("--viz_type", type=str, default="all",
                        choices=["gradcam", "counterfactual", "all"],
                        help="Type of visualization to generate")
    parser.add_argument("--viz_output", type=str, default=None,
                        help="Output directory for visualizations (default: checkpoint_dir/visualizations)")

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
        # Data paths
        data_dir=get_config('data_dir', 'combined_fast'),
        floor_plans_dir=get_config('floor_plans_dir', f"{get_config('data_dir', 'combined_fast')}/floor_plans"),
        
        # Grid processing
        target_grid_size=tuple(get_config('target_grid_size', [96, 128])),
        num_grid_channels=get_config('num_grid_channels', 4),
        
        # CNN architecture
        cnn_channels=get_config('cnn_channels', [32, 64, 128, 256]),
        cnn_kernel_size=get_config('cnn_kernel_size', 3),
        
        # Scenario MLP
        scenario_input_dim=get_config('scenario_input_dim', 4),
        scenario_hidden_dims=get_config('scenario_hidden_dims', [64, 32]),
        
        # Combined head
        head_hidden_dims=get_config('head_hidden_dims', [256, 128, 64]),
        dropout=get_config('dropout', 0.2),
        
        # Output
        num_outputs=get_config('num_outputs', 4),
        output_names=get_config('output_names', ['survival_rate', 'avg_evacuation_time', 'steps', 'avg_fire_damage']),
        
        # Training
        batch_size=get_config('batch_size', 64),
        learning_rate=get_config('lr', 1e-3),
        weight_decay=get_config('weight_decay', 1e-4),
        epochs=get_config('epochs', 100),
        early_stopping_patience=get_config('patience', 10),
        num_workers=get_config('num_workers', 4),
        
        # Data loading
        max_plans=get_config('max_plans', None),
        
        # Normalization stats (optional)
        scenario_means=get_config('scenario_means', None),
        scenario_stds=get_config('scenario_stds', None),
        target_means=get_config('target_means', None),
        target_stds=get_config('target_stds', None),
        
        # Checkpointing
        checkpoint_dir=get_config('checkpoint_dir', 'checkpoints'),
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

    # Visualization mode
    visualize_mode = get_config('visualize', False)
    viz_samples = get_config('viz_samples', '0,1,2')
    viz_type = get_config('viz_type', 'all')
    viz_output = get_config('viz_output', None)

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

    if visualize_mode:
        # Visualization mode
        from .visualize import visualize_sample, generate_visualizations

        ckpt_path = checkpoint_path or Path(config.checkpoint_dir) / "best_model.pt"
        print(f"\nLoading checkpoint: {ckpt_path}")
        load_checkpoint(model, ckpt_path, device)
        model.to(device)

        # Parse sample indices
        dataset = test_loader.dataset
        sample_indices = parse_sample_indices(viz_samples, len(dataset))
        print(f"Visualizing samples: {sample_indices}")

        # Set output directory
        output_dir = viz_output or str(Path(config.checkpoint_dir) / "visualizations")
        print(f"Output directory: {output_dir}")

        # Generate visualizations
        compute_cf = viz_type in ['counterfactual', 'all']

        if len(sample_indices) == 1:
            # Single sample
            sample = dataset[sample_indices[0]]
            sample_output_dir = str(Path(output_dir) / f"sample_{sample_indices[0]:04d}")
            visualize_sample(
                model=model,
                grid=sample['grid'],
                scenario=sample['scenario'],
                output_dir=sample_output_dir,
                sample_name=f"sample_{sample_indices[0]:04d}",
                compute_counterfactual=compute_cf,
                device=device,
                show_progress=True
            )
        else:
            # Multiple samples
            generate_visualizations(
                model=model,
                dataset=dataset,
                sample_indices=sample_indices,
                output_dir=output_dir,
                compute_counterfactual=compute_cf,
                device=device
            )

    elif eval_only:
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
