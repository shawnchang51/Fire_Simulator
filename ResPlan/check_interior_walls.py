"""
Check if INTERIOR walls (walls between rooms) are 1 cell wide.
"""

import numpy as np
import sys


def check_interior_wall_width(npz_path: str):
    """Check if interior walls are 1 cell wide."""
    data = np.load(npz_path)
    grid = data['grid']

    rows, cols = grid.shape

    # Identify interior walls: wall cells (-2) adjacent to passable cells (0)
    interior_walls = np.zeros_like(grid, dtype=bool)

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == -2:
                # Check if this wall is adjacent to interior (has at least one 0 neighbor)
                has_interior_neighbor = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid[nr, nc] == 0:
                                has_interior_neighbor = True
                                break
                    if has_interior_neighbor:
                        break

                if has_interior_neighbor:
                    interior_walls[r, c] = True

    print(f"Total interior wall cells: {np.sum(interior_walls)}")

    # Check for thick interior walls
    thick_h = 0
    thick_v = 0

    for r in range(1, rows - 1):
        for c in range(cols):
            if (interior_walls[r, c] and
                interior_walls[r-1, c] and
                interior_walls[r+1, c]):
                thick_h += 1

    for r in range(rows):
        for c in range(1, cols - 1):
            if (interior_walls[r, c] and
                interior_walls[r, c-1] and
                interior_walls[r, c+1]):
                thick_v += 1

    print(f"Interior walls with 3+ rows thick: {thick_h}")
    print(f"Interior walls with 3+ cols thick: {thick_v}")

    if thick_h == 0 and thick_v == 0:
        print("\n[OK] All interior walls are 1 cell wide!")
    else:
        print(f"\n[ERROR] Found thick interior walls")

    # Show sample of interior walls
    print(f"\nSample interior wall region (rows 10-20, cols 15-30):")
    sample = grid[10:20, 15:30].copy()
    # Mark interior walls with 'W', passable with '.', exterior with '#'
    for r in range(10, 20):
        for c in range(15, 30):
            if interior_walls[r, c]:
                print('W', end='')
            elif grid[r, c] == 0:
                print('.', end='')
            else:
                print('#', end='')
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_interior_walls.py <npz_file>")
        sys.exit(1)

    check_interior_wall_width(sys.argv[1])
