"""
Find ResPlan floor plans with valid net_area (required for NPZ conversion).

Usage:
    python find_valid_resplan.py --min-area 50 --max-area 150 --limit 20
"""

import argparse
import pickle
import sys
from pathlib import Path

# Add ResPlan utilities
sys.path.insert(0, str(Path(__file__).parent / "ResPlan"))
from resplan_utils import normalize_keys


def find_valid_plans(pkl_path='ResPlan/ResPlan.pkl', min_area=0, max_area=float('inf'), limit=None):
    """
    Find plans with valid net_area.

    Args:
        pkl_path: Path to ResPlan.pkl
        min_area: Minimum net_area in sqm (default: 0)
        max_area: Maximum net_area in sqm (default: inf)
        limit: Maximum number of results (default: all)

    Returns:
        List of (index, plan_id, net_area, unit_type) tuples
    """
    print(f"Loading ResPlan dataset from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        plans = pickle.load(f)

    for plan in plans:
        normalize_keys(plan)

    print(f"Loaded {len(plans)} plans\n")

    valid_plans = []
    for idx, plan in enumerate(plans):
        net_area = plan.get('net_area', 0)
        if net_area > 0 and min_area <= net_area <= max_area:
            plan_id = plan.get('id', idx)
            unit_type = plan.get('unitType', 'unknown')
            valid_plans.append((idx, plan_id, net_area, unit_type))

            if limit and len(valid_plans) >= limit:
                break

    return valid_plans


def main():
    parser = argparse.ArgumentParser(
        description="Find ResPlan floor plans with valid net_area"
    )
    parser.add_argument('--min-area', type=float, default=0,
                       help='Minimum net_area in sqm (default: 0)')
    parser.add_argument('--max-area', type=float, default=float('inf'),
                       help='Maximum net_area in sqm (default: unlimited)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of results (default: all)')
    parser.add_argument('--pkl-path', type=str, default='ResPlan/ResPlan.pkl',
                       help='Path to ResPlan.pkl (default: ResPlan/ResPlan.pkl)')
    parser.add_argument('--save', type=str,
                       help='Save indices to text file (one per line)')

    args = parser.parse_args()

    # Find valid plans
    valid_plans = find_valid_plans(
        pkl_path=args.pkl_path,
        min_area=args.min_area,
        max_area=args.max_area,
        limit=args.limit
    )

    # Display results
    print(f"Found {len(valid_plans)} valid plans")
    print(f"{'Index':<8} {'Plan ID':<10} {'Area (sqm)':<12} {'Unit Type':<15}")
    print("-" * 50)

    for idx, plan_id, net_area, unit_type in valid_plans[:20]:  # Show first 20
        print(f"{idx:<8} {plan_id:<10} {net_area:<12.2f} {unit_type:<15}")

    if len(valid_plans) > 20:
        print(f"... and {len(valid_plans) - 20} more")

    # Save indices if requested
    if args.save:
        save_path = Path(args.save)
        with open(save_path, 'w') as f:
            for idx, _, _, _ in valid_plans:
                f.write(f"{idx}\n")
        print(f"\nSaved {len(valid_plans)} indices to: {save_path}")

    # Print statistics
    if valid_plans:
        areas = [area for _, _, area, _ in valid_plans]
        print(f"\nStatistics:")
        print(f"  Total valid plans: {len(valid_plans)}")
        print(f"  Area range: {min(areas):.2f} - {max(areas):.2f} sqm")
        print(f"  Average area: {sum(areas)/len(areas):.2f} sqm")

        # Unit type distribution
        unit_types = {}
        for _, _, _, unit_type in valid_plans:
            unit_types[unit_type] = unit_types.get(unit_type, 0) + 1
        print(f"\n  Unit types:")
        for unit_type, count in sorted(unit_types.items(), key=lambda x: -x[1]):
            print(f"    {unit_type}: {count}")


if __name__ == "__main__":
    main()
