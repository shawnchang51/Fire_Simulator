"""
Visualize wall thickness and straightness in NPZ floor plans.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_wall_thickness(grid: np.ndarray) -> dict:
    """
    Analyze wall thickness by checking connected wall components.

    Returns:
        dict: Statistics about wall thickness
    """
    from scipy import ndimage

    # Identify walls
    wall_mask = (grid == -2).astype(np.uint8)

    # Label connected wall components
    labeled, num_features = ndimage.label(wall_mask)

    # Find thickness of each wall segment
    wall_thicknesses = []

    rows, cols = grid.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid[r, c] == -2:
                # Check 3x3 neighborhood
                neighborhood = grid[r-1:r+2, c-1:c+2]
                wall_count = np.sum(neighborhood == -2)

                # If there are multiple walls in neighborhood, it's thick
                if wall_count > 3:  # More than a simple line
                    wall_thicknesses.append(wall_count)

    return {
        'total_walls': np.sum(wall_mask),
        'thick_walls': len(wall_thicknesses),
        'avg_thickness': np.mean(wall_thicknesses) if wall_thicknesses else 0,
        'max_thickness': max(wall_thicknesses) if wall_thicknesses else 0
    }


def visualize_npz(npz_path: str):
    """
    Visualize NPZ floor plan and analyze walls.
    """
    # Load NPZ
    data = np.load(npz_path)
    grid = data['grid']

    # Analyze walls
    stats = analyze_wall_thickness(grid)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Full grid
    im1 = ax1.imshow(grid, cmap='RdYlGn', interpolation='nearest')
    ax1.set_title('Floor Plan Grid')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    plt.colorbar(im1, ax=ax1, label='Cell Value')

    # Plot 2: Walls only
    wall_only = np.where(grid == -2, 1, 0)
    im2 = ax2.imshow(wall_only, cmap='binary', interpolation='nearest')
    ax2.set_title('Walls Only')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')

    # Add statistics text
    stats_text = f"""
    Total wall cells: {stats['total_walls']}
    Thick wall cells: {stats['thick_walls']}
    Avg thickness: {stats['avg_thickness']:.2f}
    Max thickness: {stats['max_thickness']}
    """
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save visualization
    output_path = Path(npz_path).with_suffix('.png')
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to: {output_path}")

    # Print statistics
    print("\nWall Statistics:")
    print(f"  Total wall cells: {stats['total_walls']}")
    print(f"  Thick wall cells: {stats['thick_walls']}")
    print(f"  Average thickness: {stats['avg_thickness']:.2f}")
    print(f"  Max thickness: {stats['max_thickness']}")

    if stats['thick_walls'] == 0:
        print("\n✓ All walls are 1 cell in width!")
    else:
        print(f"\n✗ Found {stats['thick_walls']} cells with thick walls")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize NPZ floor plan walls")
    parser.add_argument('npz_path', type=str, help='Path to NPZ file')

    args = parser.parse_args()
    visualize_npz(args.npz_path)
