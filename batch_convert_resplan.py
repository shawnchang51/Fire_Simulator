"""
Batch convert multiple ResPlan floor plans to NPZ format.

Usage:
    # Convert first 100 plans
    python batch_convert_resplan.py --start 0 --end 100 --output-dir npz_plans/

    # Convert 50 random plans
    python batch_convert_resplan.py --random 50 --output-dir npz_plans/

    # Convert specific plans
    python batch_convert_resplan.py --indices 0,10,42,100,500 --output-dir npz_plans/
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from resplan_to_npz import load_resplan_dataset, ResPlanToNPZ


def batch_convert(plan_indices, output_dir: Path, cell_size: float = 0.3,
                 wall_thickness: int = 2, pkl_path: str = 'ResPlan/ResPlan.pkl'):
    """
    Batch convert multiple plans to NPZ.

    Args:
        plan_indices: List of plan indices to convert
        output_dir: Directory to save NPZ files
        cell_size: Cell size in meters
        wall_thickness: Wall thickness in cells
        pkl_path: Path to ResPlan.pkl
    """
    # Load dataset
    print(f"Loading ResPlan dataset from {pkl_path}...")
    plans = load_resplan_dataset(pkl_path)
    print(f"Loaded {len(plans)} plans\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    successful = 0
    failed = 0
    failed_plans = []

    # Convert each plan
    for idx in tqdm(plan_indices, desc="Converting plans"):
        if idx >= len(plans):
            print(f"Warning: Plan index {idx} out of range, skipping")
            failed += 1
            continue

        plan = plans[idx]
        plan_id = plan.get('id', idx)
        output_path = output_dir / f"plan_{idx:05d}.npz"

        try:
            converter = ResPlanToNPZ(
                plan,
                cell_size=cell_size,
                wall_thickness=wall_thickness
            )
            converter.save_npz(str(output_path))
            successful += 1

        except Exception as e:
            print(f"\nError converting plan {idx} (ID {plan_id}): {e}")
            failed += 1
            failed_plans.append((idx, plan_id, str(e)))

    # Summary
    print(f"\n{'='*60}")
    print(f"Batch Conversion Summary")
    print(f"{'='*60}")
    print(f"Total attempts: {len(plan_indices)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir.absolute()}")

    if failed_plans:
        print(f"\nFailed plans:")
        for idx, plan_id, error in failed_plans[:10]:  # Show first 10
            print(f"  Plan {idx} (ID {plan_id}): {error[:60]}")
        if len(failed_plans) > 10:
            print(f"  ... and {len(failed_plans) - 10} more")


def main():
    parser = argparse.ArgumentParser(description="Batch convert ResPlan floor plans to NPZ")

    # Plan selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start-end', type=str, metavar='START,END',
                      help='Convert plans from START to END (e.g., "0,100")')
    group.add_argument('--random', type=int, metavar='N',
                      help='Convert N random plans')
    group.add_argument('--indices', type=str,
                      help='Comma-separated list of indices (e.g., "0,10,42,100")')

    # Output settings
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for NPZ files')
    parser.add_argument('--cell-size', type=float, default=0.3,
                       help='Cell size in meters (default: 0.3)')
    parser.add_argument('--wall-thickness', type=int, default=2,
                       help='Wall thickness in cells (default: 2)')
    parser.add_argument('--pkl-path', type=str, default='ResPlan/ResPlan.pkl',
                       help='Path to ResPlan.pkl (default: ResPlan/ResPlan.pkl)')

    args = parser.parse_args()

    # Determine plan indices to convert
    if args.start_end:
        try:
            start, end = map(int, args.start_end.split(','))
            plan_indices = list(range(start, end))
        except ValueError:
            parser.error("--start-end must be in format 'START,END' (e.g., '0,100')")

    elif args.random:
        # Load to get total count
        plans = load_resplan_dataset(args.pkl_path)
        plan_indices = np.random.choice(len(plans), size=args.random, replace=False).tolist()
        print(f"Randomly selected {args.random} plans")

    elif args.indices:
        try:
            plan_indices = [int(x.strip()) for x in args.indices.split(',')]
        except ValueError:
            parser.error("--indices must be comma-separated integers (e.g., '0,10,42')")

    # Run batch conversion
    output_dir = Path(args.output_dir)
    batch_convert(
        plan_indices,
        output_dir,
        cell_size=args.cell_size,
        wall_thickness=args.wall_thickness,
        pkl_path=args.pkl_path
    )


if __name__ == "__main__":
    main()
