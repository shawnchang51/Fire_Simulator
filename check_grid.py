import numpy as np
import sys

data = np.load(sys.argv[1])
grid = data['grid']
doors = data['door_positions']
exits = data['exit_positions']

print(f"Grid shape: {grid.shape}")
print(f"Passable cells (0): {np.sum(grid == 0)}")
print(f"Wall cells (-2): {np.sum(grid == -2)}")
print(f"Other values: {np.sum((grid != 0) & (grid != -2))}")
print(f"\nDoor positions (first 5):")
for i, (r, c) in enumerate(doors[:5]):
    print(f"  Door {i}: ({r}, {c}) -> grid value = {grid[r, c]}")
print(f"\nExit positions:")
for i, (r, c) in enumerate(exits):
    print(f"  Exit {i}: ({r}, {c}) -> grid value = {grid[r, c]}")
