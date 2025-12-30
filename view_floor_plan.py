"""
Simple floor plan viewer for .npz files
"""

import sys
import numpy as np

def visualize_floor_plan(filepath):
    """Load and display a floor plan from .npz file"""

    # Load data
    data = np.load(filepath)

    grid = data['grid']
    size = tuple(data['size'])
    room_count = int(data['room_count'])
    method = str(data['generation_method'])
    exit_positions = data['exit_positions']

    # Get obstacle density if available
    obstacle_density = float(data['obstacle_density']) if 'obstacle_density' in data else 0.0

    print("="*70)
    print(f"Floor Plan: {filepath.split('/')[-1]}")
    print("="*70)
    print(f"Method: {method}")
    print(f"Size: {size[0]}x{size[1]} ({size[0] * size[1]} cells)")
    print(f"Rooms: {room_count}")
    print(f"Exits: {len(exit_positions)} at {list(map(tuple, exit_positions))}")
    print(f"Obstacle density: {obstacle_density:.1%}")

    passable = np.sum(grid == 0)
    total = grid.size
    print(f"Passable area: {passable}/{total} cells ({100*passable/total:.1f}%)")
    print()

    # Create visualization
    rows, cols = grid.shape

    # Mark exits
    exit_set = set(tuple(pos) for pos in exit_positions)

    print("Visualization:")
    print("  # = wall")
    print("  . = passable")
    print("  E = exit")
    print()

    for y in range(rows):
        row_str = ""
        for x in range(cols):
            if (x, y) in exit_set:
                row_str += "E"
            elif grid[y, x] == -2:
                row_str += "#"
            else:
                row_str += "."
        print("  " + row_str)

    print()
    print("="*70)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python view_floor_plan.py <npz_file>")
        sys.exit(1)

    visualize_floor_plan(sys.argv[1])
