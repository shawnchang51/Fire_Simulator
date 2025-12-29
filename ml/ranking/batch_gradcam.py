"""
Batch Grad-CAM Generator for Multiple Floor Plans

This script generates individual Grad-CAM visualizations for multiple plans.
The original `--mode visualize` functionality remains unchanged.

Usage:
    # Generate Grad-CAM for specific plan indices
    python -m ml.ranking.batch_gradcam --checkpoint checkpoints/ranking/best_model.pt --plans 0 5 10 25 50 100

    # Generate for a range of plans (0 to 99)
    python -m ml.ranking.batch_gradcam --checkpoint checkpoints/ranking/best_model.pt --range 0 100

    # Generate for random N plans
    python -m ml.ranking.batch_gradcam --checkpoint checkpoints/ranking/best_model.pt --random 20

    # Specify output directory
    python -m ml.ranking.batch_gradcam --checkpoint checkpoints/ranking/best_model.pt --plans 0 5 10 --output gradcam_outputs/

    # Generate combined grid image instead of individual images
    python -m ml.ranking.batch_gradcam --checkpoint checkpoints/ranking/best_model.pt --plans 0 5 10 --combined

    # Use different data directory
    python -m ml.ranking.batch_gradcam --checkpoint checkpoints/ranking/best_model.pt --plans 0 5 10 --data-dir training_data_v5_combined
"""

import argparse
import random
import sys
from pathlib import Path

import torch

from .config import RankingConfig
from .dataset import SingleConfigDataset
from .train import load_checkpoint
from .visualize import visualize_multiple_gradcam, visualize_gradcam_sample


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch Grad-CAM Generator for Floor Plan Analysis"
    )

    # Required
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )

    # Plan selection (mutually exclusive)
    plan_group = parser.add_mutually_exclusive_group(required=True)
    plan_group.add_argument(
        '--plans',
        type=int,
        nargs='+',
        help="Specific plan indices to visualize (e.g., --plans 0 5 10 25)"
    )
    plan_group.add_argument(
        '--range',
        type=int,
        nargs=2,
        metavar=('START', 'END'),
        help="Range of plan indices [start, end) (e.g., --range 0 100)"
    )
    plan_group.add_argument(
        '--random',
        type=int,
        metavar='N',
        help="Randomly select N plans (e.g., --random 20)"
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='gradcam_outputs',
        help="Output directory for images (default: gradcam_outputs/)"
    )
    parser.add_argument(
        '--combined',
        action='store_true',
        help="Save as one combined grid image instead of individual files"
    )
    parser.add_argument(
        '--cols',
        type=int,
        default=3,
        help="Number of columns for combined grid (default: 3)"
    )

    # Data options
    parser.add_argument(
        '--data-dir',
        type=str,
        default='combined_fast',
        help="Root directory containing training data"
    )
    parser.add_argument(
        '--floor-plans-dir',
        type=str,
        default=None,
        help="Directory containing floor plan NPZ files (default: <data-dir>/floor_plans)"
    )
    parser.add_argument(
        '--max-configs',
        type=int,
        default=None,
        help="Maximum number of configs to load (default: all)"
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )

    # Seed for reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for --random mode (default: 42)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("BATCH GRAD-CAM GENERATOR")
    print("=" * 60)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    # Get config from checkpoint
    config = RankingConfig(**checkpoint['config'])

    # Override data paths if provided
    data_dir = Path(args.data_dir)
    floor_plans_dir = args.floor_plans_dir or str(data_dir / "floor_plans")

    # Load dataset
    print(f"\nLoading data from {data_dir}...")
    try:
        dataset = SingleConfigDataset(
            simulation_results_file=str(data_dir / "simulation_results.jsonl"),
            floor_plans_dir=floor_plans_dir,
            target_size=config.target_grid_size,
            max_configs=args.max_configs
        )
        print(f"  Loaded {len(dataset):,} configs")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Determine plan indices
    if args.plans:
        plan_indices = args.plans
    elif args.range:
        start, end = args.range
        plan_indices = list(range(start, min(end, len(dataset))))
    elif args.random:
        random.seed(args.seed)
        n = min(args.random, len(dataset))
        plan_indices = sorted(random.sample(range(len(dataset)), n))
    else:
        print("Error: Must specify --plans, --range, or --random")
        sys.exit(1)

    # Validate indices
    valid_indices = [i for i in plan_indices if 0 <= i < len(dataset)]
    if len(valid_indices) != len(plan_indices):
        invalid = set(plan_indices) - set(valid_indices)
        print(f"Warning: Invalid indices (out of range): {sorted(invalid)}")

    if not valid_indices:
        print("Error: No valid plan indices!")
        sys.exit(1)

    print(f"\nWill generate Grad-CAM for {len(valid_indices)} plans:")
    if len(valid_indices) <= 20:
        print(f"  Indices: {valid_indices}")
    else:
        print(f"  Indices: {valid_indices[:10]} ... {valid_indices[-5:]}")

    # Generate visualizations
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.combined:
        # Combined grid image
        output_path = output_dir / "gradcam_combined.png"
        print(f"\nGenerating combined grid image to {output_path}...")
        visualize_multiple_gradcam(
            model=model,
            dataset=dataset,
            plan_indices=valid_indices,
            output_path=str(output_path),
            device=device,
            cols=args.cols,
            save_individual=False
        )
    else:
        # Individual images
        print(f"\nGenerating individual images to {output_dir}/...")
        visualize_multiple_gradcam(
            model=model,
            dataset=dataset,
            plan_indices=valid_indices,
            output_dir=str(output_dir),
            device=device,
            save_individual=True
        )

    print("\n" + "=" * 60)
    print("BATCH GRAD-CAM GENERATION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
