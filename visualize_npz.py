"""
Visualize ResPlan NPZ floor plans.

Usage:
    python visualize_npz.py test_plan_0.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_npz(npz_path: str, save_path: str = None):
    """
    Visualize NPZ floor plan.

    Args:
        npz_path: Path to NPZ file
        save_path: Optional path to save figure (if None, displays interactively)
    """
    # Load NPZ
    data = np.load(npz_path, allow_pickle=True)

    grid = data['grid']
    door_positions = data['door_positions']
    exit_positions = data['exit_positions']

    # Extract metadata
    plan_id = int(data['plan_id'])
    unit_type = str(data['unit_type'])
    net_area = float(data['net_area'])
    cell_size = float(data['cell_size'])

    print(f"Plan ID: {plan_id}")
    print(f"Unit type: {unit_type}")
    print(f"Net area: {net_area:.2f} square meters")
    print(f"Grid shape: {grid.shape}")
    print(f"Cell size: {cell_size}m")
    print(f"Doors: {len(door_positions)}")
    print(f"Exits: {len(exit_positions)}")
    print(f"Passable cells: {np.sum(grid == 0)}")
    print(f"Wall cells: {np.sum(grid == -2)}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Display grid (walls = black, passable = white)
    display_grid = np.ones_like(grid) * 0.5  # Gray default
    display_grid[grid == 0] = 1.0   # White for passable
    display_grid[grid == -2] = 0.0  # Black for walls

    ax.imshow(display_grid, cmap='gray', origin='upper', interpolation='nearest')

    # Mark doors as green circles
    if len(door_positions) > 0:
        door_rows = door_positions[:, 0]
        door_cols = door_positions[:, 1]
        ax.scatter(door_cols, door_rows, c='green', s=50, marker='o',
                  label=f'Doors ({len(door_positions)})', edgecolors='black', linewidths=1)

    # Mark exits as red stars
    if len(exit_positions) > 0:
        exit_rows = exit_positions[:, 0]
        exit_cols = exit_positions[:, 1]
        ax.scatter(exit_cols, exit_rows, c='red', s=200, marker='*',
                  label=f'Exits ({len(exit_positions)})', edgecolors='black', linewidths=1)

    ax.set_title(f'ResPlan Floor Plan #{plan_id} ({unit_type}, {net_area:.1f} sqm)\n'
                f'Grid: {grid.shape[0]}x{grid.shape[1]} @ {cell_size}m/cell',
                fontsize=14)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.legend(loc='upper right')
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize ResPlan NPZ floor plans")
    parser.add_argument('npz_path', type=str, help='Path to NPZ file')
    parser.add_argument('--save', type=str, help='Save figure to path (default: show interactively)')

    args = parser.parse_args()

    if not Path(args.npz_path).exists():
        print(f"Error: File not found: {args.npz_path}")
        return

    visualize_npz(args.npz_path, args.save)


if __name__ == "__main__":
    main()
