# Quick Reference: Vector-to-Grid Conversion

## Essential Formulas

### 1. Scale Factor Calculation
```python
# Given: real_area (m²), polygon
scale_factor = sqrt(real_area / polygon.area)
```

**Example:**
```
Real area: 73.13 m²
Vector area: 28,970 units²
Scale: sqrt(73.13 / 28,970) = 0.0532 units→meters
```

### 2. Coordinate Conversion
```python
# Vector (x, y) → Grid (row, col)
x_rel = x - x_min
y_rel = y - y_min
x_meters = x_rel * scale_factor
y_meters = y_rel * scale_factor
col = int(x_meters / cell_size)
row = int(y_meters / cell_size)
```

### 3. Grid Dimensions
```python
width_meters = (x_max - x_min) * scale_factor
height_meters = (y_max - y_min) * scale_factor
grid_cols = ceil(width_meters / cell_size)
grid_rows = ceil(height_meters / cell_size)
```

## Key Libraries

```python
import numpy as np              # Grid arrays
from shapely.geometry import Polygon, LineString  # Vector geometry
import cv2                      # Rasterization
import matplotlib.pyplot as plt # Visualization
```

## Common Operations

### Get Polygon Area
```python
area = polygon.area  # NOT (x_max - x_min) * (y_max - y_min)
```

### Get Bounding Box
```python
x_min, y_min, x_max, y_max = polygon.bounds
```

### Get Centroid
```python
center = polygon.centroid  # Returns Point(x, y)
x, y = center.x, center.y
```

### Rasterize Polygon
```python
coords = np.array([(col, row) for x, y in polygon.exterior.coords
                   for row, col in [world_to_grid(x, y)]], dtype=np.int32)
cv2.fillPoly(grid, [coords], color=value)
```

### Draw Line
```python
coords = np.array([(col, row) for x, y in linestring.coords
                   for row, col in [world_to_grid(x, y)]], dtype=np.int32)
cv2.polylines(grid, [coords], isClosed=False, color=value, thickness=1)
```

## NPZ File Structure

### Save
```python
np.savez_compressed(
    'output.npz',
    grid=grid,                  # 2D array
    door_positions=doors,       # (N, 2) array
    exit_positions=exits,       # (M, 2) array
    cell_size=0.3,             # Scalar
    scale_factor=0.0532        # Scalar
)
```

### Load
```python
data = np.load('output.npz', allow_pickle=True)
grid = data['grid']
doors = data['door_positions']
cell_size = float(data['cell_size'])
```

## Grid Values

| Value | Meaning | Use |
|-------|---------|-----|
| 0 | Passable | Interior space, walkable |
| -2 | Wall/Outside | Obstacles, exterior |

## Coordinate Systems

**Shapely (Vector):**
- Format: `(x, y)`
- Origin: Arbitrary, usually bottom-left
- Units: Arbitrary (needs scaling)

**NumPy (Grid):**
- Format: `[row, col]` = `[y, x]`
- Origin: Top-left
- Units: Grid cells

**OpenCV:**
- Format: `[(x, y), ...]` = `[(col, row), ...]`
- Origin: Top-left

## Validation Checklist

- [ ] `net_area > 0`
- [ ] `polygon is not None and not polygon.is_empty`
- [ ] `0 <= row < grid_rows` and `0 <= col < grid_cols`
- [ ] `grid.dtype == np.float32` (not int8/int16)
- [ ] Area ratio: `0.95 <= (grid_area / real_area) <= 1.05`

## Common Errors

### Wrong scale calculation
```python
# ❌ WRONG
scale = sqrt(real_area / bbox_area)

# ✓ CORRECT
scale = sqrt(real_area / polygon.area)
```

### Coordinate order confusion
```python
# ❌ WRONG
grid[col, row] = value

# ✓ CORRECT
grid[row, col] = value
```

### Missing geometry check
```python
# ❌ WRONG
area = polygon.area  # Crashes if None

# ✓ CORRECT
if polygon is None or polygon.is_empty:
    raise ValueError("Invalid geometry")
area = polygon.area
```

## Performance Tips

1. **Use cv2.fillPoly** instead of manual pixel iteration
2. **Pre-allocate grid** with `np.full()` instead of nested loops
3. **Batch conversions** with tqdm progress bars
4. **Validate once** at start, not per-polygon
5. **Use float32** not float64 (half the memory)

## Typical Values

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `cell_size` | 0.2 - 0.5 m | 0.3m = shoulder width |
| `scale_factor` | 0.01 - 0.1 | Depends on vector units |
| `grid_size` | 20×20 to 200×200 | For residential plans |
| `real_area` | 30 - 200 m² | Apartments/houses |

## One-Liner Conversions

```python
# Polygon area
area = polygon.area

# Bounding box
bbox = polygon.bounds

# Centroid
center = polygon.centroid

# Buffer (expand/shrink)
bigger = polygon.buffer(1.0)
smaller = polygon.buffer(-1.0)

# Check point inside
is_inside = polygon.contains(Point(x, y))

# Intersection
overlap = polygon1.intersection(polygon2)

# Union
combined = polygon1.union(polygon2)
```

## Debugging Commands

```python
# Check grid stats
print(f"Shape: {grid.shape}")
print(f"Passable: {np.sum(grid == 0)}")
print(f"Walls: {np.sum(grid == -2)}")
print(f"Min: {grid.min()}, Max: {grid.max()}")

# Visualize quickly
plt.imshow(grid == 0, cmap='gray')
plt.show()

# Find unique values
print(np.unique(grid))

# Check specific cell
row, col = 10, 20
print(f"Grid[{row},{col}] = {grid[row, col]}")
```
