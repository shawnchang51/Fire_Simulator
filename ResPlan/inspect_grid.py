"""Inspect specific grid regions"""
import numpy as np
import sys

npz_path = sys.argv[1] if len(sys.argv) > 1 else "test_plan_fixed.npz"
data = np.load(npz_path)
grid = data['grid']

print(f"Grid shape: {grid.shape}\n")

# Show the problematic region (rows 10-20, cols 15-30)
print("Rows 10-20, Cols 15-30:")
print("Col:  ", "".join(f"{c:3}" for c in range(15, 30)))
for r in range(10, 20):
    row_str = f"R{r:2}: "
    for c in range(15, 30):
        val = grid[r, c]
        if val == -2:
            row_str += "  #"
        elif val == 0:
            row_str += "  ."
        else:
            row_str += f"{val:3.0f}"
    print(row_str)

print(f"\nLegend: # = wall (-2), . = passable (0)")
