"""
Simple script to check if walls are truly 1 cell wide.
"""

import numpy as np
import sys


def check_wall_width(npz_path: str):
    """Check if walls are 1 cell wide by examining the grid."""
    data = np.load(npz_path)
    grid = data['grid']

    rows, cols = grid.shape

    # Check for horizontal thick walls (more than 1 row thick)
    horizontal_thick = []
    for r in range(1, rows - 1):
        for c in range(cols):
            if grid[r, c] == -2 and grid[r-1, c] == -2 and grid[r+1, c] == -2:
                horizontal_thick.append((r, c))

    # Check for vertical thick walls (more than 1 col thick)
    vertical_thick = []
    for r in range(rows):
        for c in range(1, cols - 1):
            if grid[r, c] == -2 and grid[r, c-1] == -2 and grid[r, c+1] == -2:
                vertical_thick.append((r, c))

    print(f"Grid shape: {grid.shape}")
    print(f"Horizontal thick walls (3+ rows): {len(horizontal_thick)}")
    print(f"Vertical thick walls (3+ cols): {len(vertical_thick)}")

    if horizontal_thick:
        print(f"\nSample horizontal thick walls: {horizontal_thick[:10]}")
    if vertical_thick:
        print(f"\nSample vertical thick walls: {vertical_thick[:10]}")

    # Show a sample of the grid
    print(f"\nTop-left corner (rows 0-10, cols 0-15):")
    print(grid[0:10, 0:15].astype(int))

    # Count consecutive walls
    print(f"\nChecking for thick wall segments...")
    max_h_thickness = 0
    max_v_thickness = 0

    for r in range(rows):
        thickness = 0
        for c in range(cols):
            if grid[r, c] == -2:
                thickness += 1
            else:
                if thickness > max_h_thickness:
                    max_h_thickness = thickness
                thickness = 0

    for c in range(cols):
        thickness = 0
        for r in range(rows):
            if grid[r, c] == -2:
                thickness += 1
            else:
                if thickness > max_v_thickness:
                    max_v_thickness = thickness
                thickness = 0

    print(f"Max consecutive horizontal wall cells: {max_h_thickness}")
    print(f"Max consecutive vertical wall cells: {max_v_thickness}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_wall_width.py <npz_file>")
        sys.exit(1)

    check_wall_width(sys.argv[1])
