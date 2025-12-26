"""
Test if walls properly block 8-connected agent movement.
"""

import numpy as np
import sys
from collections import deque


def can_reach_8connected(grid: np.ndarray, start: tuple, end: tuple) -> bool:
    """
    Check if an 8-connected agent can reach from start to end position.

    Args:
        grid: 2D array where -2 is wall, 0 is passable
        start: (row, col) starting position
        end: (row, col) ending position

    Returns:
        True if reachable, False if blocked by walls
    """
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)

    queue = deque([start])
    visited[start] = True

    # 8-connected movement (including diagonals)
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    while queue:
        r, c = queue.popleft()

        if (r, c) == end:
            return True

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc

            if (0 <= nr < rows and 0 <= nc < cols and
                not visited[nr, nc] and grid[nr, nc] == 0):
                visited[nr, nc] = True
                queue.append((nr, nc))

    return False


def test_wall_integrity(npz_path: str):
    """
    Test that walls properly separate rooms for 8-connected agents.
    """
    data = np.load(npz_path)
    grid = data['grid']

    rows, cols = grid.shape

    print(f"Grid shape: {grid.shape}")
    print(f"Testing wall integrity for 8-connected agents...\n")

    # Find all interior regions
    from scipy import ndimage

    interior_mask = (grid == 0)
    labeled, num_regions = ndimage.label(interior_mask, structure=np.ones((3, 3)))

    print(f"Found {num_regions} separated interior regions")

    if num_regions <= 1:
        print("\nAll interior cells are connected - walls may not be separating rooms properly!")
        return False

    # For each pair of regions, verify they're separated by walls
    issues_found = 0

    for region_a in range(1, num_regions + 1):
        mask_a = (labeled == region_a)
        # Find a cell in region A
        cells_a = np.argwhere(mask_a)
        if len(cells_a) == 0:
            continue

        start_cell = tuple(cells_a[0])

        for region_b in range(region_a + 1, num_regions + 1):
            mask_b = (labeled == region_b)
            cells_b = np.argwhere(mask_b)
            if len(cells_b) == 0:
                continue

            end_cell = tuple(cells_b[0])

            # These regions should NOT be reachable from each other
            if can_reach_8connected(grid, start_cell, end_cell):
                print(f"[ERROR] Region {region_a} can reach Region {region_b} despite walls!")
                issues_found += 1

    if issues_found == 0:
        print("\n[OK] All walls properly block 8-connected movement!")
        return True
    else:
        print(f"\n[ERROR] Found {issues_found} wall integrity issues!")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_8connectivity.py <npz_file>")
        sys.exit(1)

    success = test_wall_integrity(sys.argv[1])
    sys.exit(0 if success else 1)
