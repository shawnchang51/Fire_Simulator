# Tutorial: Building a Vector-to-Grid Floor Plan Converter

This tutorial teaches you how to convert vector-based floor plans (Shapely geometries) to grid-based representations suitable for pathfinding and simulation.

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Architecture Overview](#architecture-overview)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Advanced Topics](#advanced-topics)
5. [Common Pitfalls](#common-pitfalls)

---

## Core Concepts

### 1. Vector vs. Grid Representations

**Vector (Continuous Space):**
```python
# Walls defined as geometric shapes
wall = Polygon([(0, 0), (10, 0), (10, 0.5), (0, 0.5)])
door = LineString([(5, 0), (5, 1)])
room = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)])
```
- Precise, scalable, resolution-independent
- Used by: CAD software, architectural tools, ResPlan
- Library: Shapely (Python wrapper for GEOS)

**Grid (Discrete Space):**
```python
# Same structures as 2D array
grid = np.array([
    [-2, -2, -2, -2, -2],  # -2 = wall
    [ 0,  0,  0,  0,  0],  # 0 = passable
    [ 0,  0,  0,  0,  0],
])
```
- Fixed resolution, efficient for pathfinding algorithms
- Used by: Games, robotics, A*/D* Lite pathfinding
- Library: NumPy arrays

### 2. Coordinate System Transformations

**Problem:** Vector coordinates are arbitrary units, not real-world meters.

**Solution:** Scale based on known real-world area.

```python
# Given:
bounding_box = (256 units wide × 153.54 units tall)
real_area = 73.13 m²

# Step 1: Calculate vector area
vector_area = inner_polygon.area  # Use ACTUAL polygon area, not bbox!
# Why? Polygon could be L-shaped, have holes, etc.

# Step 2: Find scale factor
# Area scales as length²
scale_factor = sqrt(real_area / vector_area)
# Example: sqrt(73.13 / 28970) = 0.0532

# Step 3: Convert dimensions
width_meters = 256 * 0.0532 = 13.62 m
height_meters = 153.54 * 0.0532 = 8.17 m

# Step 4: Verify (area should match)
13.62 × 8.17 ≈ 111 m²  # Bounding box
# But actual polygon area = 73.13 m² ✓
```

**Key Insight:** Use polygon area, not bounding box area!

### 3. Rasterization

Converting vector shapes to grid cells:

```python
def point_to_grid(x, y, x_min, y_min, scale, cell_size):
    """Convert vector coordinates to grid indices."""
    # 1. Translate to origin
    x_rel = x - x_min
    y_rel = y - y_min

    # 2. Scale to meters
    x_meters = x_rel * scale
    y_meters = y_rel * scale

    # 3. Discretize to grid
    col = int(x_meters / cell_size)
    row = int(y_meters / cell_size)

    return (row, col)
```

For complex shapes, use OpenCV:
```python
import cv2

def rasterize_polygon(polygon, grid, value):
    """Fill polygon on grid."""
    # Get exterior coordinates
    coords = np.array(polygon.exterior.coords, dtype=np.int32)

    # Convert to grid coordinates (using point_to_grid for each)
    grid_coords = convert_coords_to_grid(coords)

    # Fill polygon
    cv2.fillPoly(grid, [grid_coords], color=value)

    # Handle holes (interiors)
    for hole in polygon.interiors:
        hole_coords = convert_coords_to_grid(np.array(hole.coords))
        cv2.fillPoly(grid, [hole_coords], color=opposite_value)

    return grid
```

---

## Architecture Overview

### Design Principles

1. **Separation of Concerns:**
   - Coordinate conversion (math)
   - Rasterization (rendering)
   - Data extraction (parsing)
   - File I/O (serialization)

2. **Single Responsibility:**
   ```python
   class Converter:
       def _compute_dimensions()    # Calculate grid size
       def _world_to_grid()         # Coordinate conversion
       def create_grid()            # Rasterization
       def extract_doors()          # Feature extraction
       def save_npz()               # Serialization
   ```

3. **Fail Fast:**
   ```python
   if net_area <= 0:
       raise ValueError(f"Invalid net_area: {net_area}")
   ```

### Data Flow

```
ResPlan PKL → Converter → NPZ File
     ↓            ↓           ↓
 Shapely      NumPy      Compressed
 Polygons     Arrays      Binary
```

**Detailed Flow:**
```
1. Load PKL → Dictionary with Shapely geometries
2. Extract 'inner' → Compute bounds & area
3. Calculate scale → Vector units → Meters
4. Create grid → Initialize all -2 (walls/outside)
5. Rasterize 'inner' → Fill with 0 (passable)
6. Rasterize walls → Overwrite with -2
7. Rasterize doors → Overwrite with -2
8. Extract door positions → Save centroids
9. Save NPZ → Compressed NumPy format
```

---

## Step-by-Step Implementation

### Step 1: Set Up Dependencies

```bash
pip install numpy shapely opencv-python
```

**Why each library:**
- `numpy`: Fast array operations, grid storage
- `shapely`: Vector geometry operations (area, bounds, centroid)
- `opencv-python`: Polygon rasterization (cv2.fillPoly)

### Step 2: Load Vector Data

```python
import pickle
from shapely.geometry import Polygon, LineString
from typing import List, Dict, Any

def load_floor_plan(pkl_path: str, plan_index: int) -> Dict[str, Any]:
    """Load a single floor plan from dataset."""
    with open(pkl_path, 'rb') as f:
        plans = pickle.load(f)

    plan = plans[plan_index]

    # Normalize common typos/variations
    if "balacony" in plan and "balcony" not in plan:
        plan["balcony"] = plan.pop("balacony")

    return plan
```

**What to expect in plan dict:**
```python
{
    'inner': Polygon(...),           # Building outline
    'wall': MultiLineString(...),    # Wall geometries
    'door': MultiLineString(...),    # Door locations
    'front_door': LineString(...),   # Exit
    'bedroom': Polygon(...),         # Room shapes
    'net_area': 73.13,              # Real area in m²
    'id': 13,                        # Plan identifier
    ...
}
```

### Step 3: Compute Grid Dimensions

```python
import numpy as np

class FloorPlanConverter:
    def __init__(self, plan: Dict, cell_size: float = 0.3):
        self.plan = plan
        self.cell_size = cell_size
        self._compute_dimensions()

    def _compute_dimensions(self):
        """Calculate grid size from plan dimensions."""
        # 1. Get interior polygon
        inner = self.plan.get('inner')
        if inner is None or inner.is_empty:
            raise ValueError("No 'inner' geometry found")

        # 2. Get bounds (bounding box)
        x_min, y_min, x_max, y_max = inner.bounds
        self.x_min = x_min
        self.y_min = y_min
        width_units = x_max - x_min
        height_units = y_max - y_min

        # 3. Get real-world area
        net_area_m2 = self.plan.get('net_area', 0)
        if net_area_m2 <= 0:
            raise ValueError(f"Invalid net_area: {net_area_m2}")

        # 4. Calculate scale factor
        # CRITICAL: Use polygon.area, NOT bounding box area!
        vector_area = inner.area  # Actual polygon area
        self.scale_factor = np.sqrt(net_area_m2 / vector_area)

        # 5. Convert dimensions to meters
        width_meters = width_units * self.scale_factor
        height_meters = height_units * self.scale_factor

        # 6. Calculate grid dimensions
        self.grid_cols = int(np.ceil(width_meters / self.cell_size))
        self.grid_rows = int(np.ceil(height_meters / self.cell_size))

        print(f"Grid: {self.grid_rows} × {self.grid_cols} cells")
        print(f"Scale: {self.scale_factor:.4f} (vector units → meters)")
```

**Why polygon.area not bbox area?**
```
Rectangular room:  bbox_area ≈ polygon_area ✓
L-shaped room:     bbox_area > polygon_area ✗
Room with holes:   bbox_area > polygon_area ✗
```

### Step 4: Coordinate Conversion

```python
def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
    """Convert vector coordinates to grid indices."""
    # Step 1: Translate to origin (0, 0)
    x_rel = x - self.x_min
    y_rel = y - self.y_min

    # Step 2: Scale to meters
    x_meters = x_rel * self.scale_factor
    y_meters = y_rel * self.scale_factor

    # Step 3: Convert to grid indices
    col = int(x_meters / self.cell_size)
    row = int(y_meters / self.cell_size)

    return (row, col)
```

**Example:**
```python
# Vector point at (128, 76.77) in arbitrary units
x, y = 128, 76.77

# With x_min=0, y_min=51.23, scale=0.0532, cell_size=0.3
x_rel = 128 - 0 = 128
y_rel = 76.77 - 51.23 = 25.54

x_meters = 128 * 0.0532 = 6.81 m
y_meters = 25.54 * 0.0532 = 1.36 m

col = int(6.81 / 0.3) = 22
row = int(1.36 / 0.3) = 4

# Grid[4, 22] corresponds to vector point (128, 76.77)
```

### Step 5: Create Base Grid

```python
def create_grid(self) -> np.ndarray:
    """Create the floor plan grid."""
    # 1. Initialize all cells as walls/outside (-2)
    grid = np.full((self.grid_rows, self.grid_cols), -2, dtype=np.float32)

    # 2. Fill interior space with passable (0)
    inner = self.plan.get('inner')
    if inner is not None:
        self._rasterize_polygon(inner, value=0, grid=grid)

    # 3. Draw walls over interior
    self._draw_walls(grid)

    # 4. Draw doors as walls (for post-processing)
    self._draw_doors(grid)

    return grid
```

### Step 6: Polygon Rasterization

```python
import cv2

def _rasterize_polygon(self, polygon, value: float, grid: np.ndarray):
    """Rasterize a Shapely polygon onto the grid."""
    # Extract exterior coordinates
    coords = np.array(polygon.exterior.coords)

    # Convert to grid coordinates
    grid_coords = []
    for x, y in coords:
        row, col = self._world_to_grid(x, y)
        grid_coords.append([col, row])  # Note: cv2 uses (x, y) = (col, row)

    grid_coords = np.array(grid_coords, dtype=np.int32)

    # Fill polygon
    cv2.fillPoly(grid, [grid_coords], color=value)

    # Handle holes (interior rings)
    for interior in polygon.interiors:
        hole_coords = []
        for x, y in np.array(interior.coords):
            row, col = self._world_to_grid(x, y)
            hole_coords.append([col, row])
        hole_coords = np.array(hole_coords, dtype=np.int32)

        # Holes should be opposite value
        hole_value = -2 if value == 0 else 0
        cv2.fillPoly(grid, [hole_coords], color=hole_value)
```

**Why cv2.fillPoly?**
- Fast C++ implementation
- Handles complex polygons (concave, with holes)
- Anti-aliasing optional
- Alternative: `skimage.draw.polygon` (pure Python, slower)

### Step 7: Draw Walls

```python
def _draw_walls(self, grid: np.ndarray):
    """Draw walls on the grid."""
    wall_geom = self.plan.get('wall')
    if wall_geom is None:
        return

    # Handle different geometry types
    from shapely.geometry import LineString, MultiLineString

    # Get individual geometries
    if isinstance(wall_geom, LineString):
        wall_geoms = [wall_geom]
    elif isinstance(wall_geom, MultiLineString):
        wall_geoms = list(wall_geom.geoms)
    else:
        return

    # Draw each wall segment
    for wall in wall_geoms:
        coords = np.array(wall.coords)
        grid_coords = []

        for x, y in coords:
            row, col = self._world_to_grid(x, y)
            grid_coords.append([col, row])

        grid_coords = np.array(grid_coords, dtype=np.int32)

        # Draw line with thickness=1
        cv2.polylines(grid, [grid_coords],
                     isClosed=False,
                     color=-2,        # Wall value
                     thickness=1)     # 1 cell wide
```

**Wall thickness discussion:**
```python
thickness=1  # Thin walls, more interior space
thickness=2  # Thicker walls, looks more realistic
thickness=3  # Very thick, reduces pathfinding space

# For this project: Always use thickness=1
```

### Step 8: Extract Feature Positions

```python
def extract_doors(self) -> List[tuple[int, int]]:
    """Extract door positions (centroids) for post-processing."""
    door_positions = []

    door_geoms = self.plan.get('door')
    if door_geoms is None:
        return door_positions

    # Get individual door geometries
    from shapely.geometry import LineString, MultiLineString
    if isinstance(door_geoms, LineString):
        door_list = [door_geoms]
    elif isinstance(door_geoms, MultiLineString):
        door_list = list(door_geoms.geoms)
    else:
        return door_positions

    # Extract centroid of each door
    for door in door_list:
        # Get centroid (middle point of door)
        centroid = door.centroid
        row, col = self._world_to_grid(centroid.x, centroid.y)

        # Validate bounds
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            door_positions.append((row, col))

    return door_positions
```

**Why centroid?**
- Simple, always works
- Alternative: Sample multiple points along door
- Alternative: Find closest wall cell to door center

### Step 9: Save to NPZ Format

```python
def save_npz(self, output_path: str):
    """Save grid and metadata to NPZ file."""
    grid = self.create_grid()
    door_positions = self.extract_doors()
    exit_positions = self.extract_exits()

    # Save with compression
    np.savez_compressed(
        output_path,
        # Main data
        grid=grid,
        door_positions=np.array(door_positions, dtype=np.int32),
        exit_positions=np.array(exit_positions, dtype=np.int32),

        # Metadata (saved as separate keys)
        plan_id=self.plan.get('id', -1),
        net_area=self.plan.get('net_area', 0.0),
        cell_size=self.cell_size,
        grid_rows=self.grid_rows,
        grid_cols=self.grid_cols,
        scale_factor=self.scale_factor
    )

    print(f"Saved: {output_path}")
```

**Loading NPZ files:**
```python
data = np.load('plan_0.npz', allow_pickle=True)

grid = data['grid']                    # (rows, cols) array
doors = data['door_positions']         # (N, 2) array
exits = data['exit_positions']         # (M, 2) array
plan_id = int(data['plan_id'])        # Scalar
net_area = float(data['net_area'])    # Scalar
```

---

## Advanced Topics

### Topic 1: Handling Multi-Geometry Types

ResPlan can have different geometry types for same feature:

```python
def get_geometries(geom_data):
    """Extract individual geometries from any Shapely type."""
    if geom_data is None:
        return []

    from shapely.geometry import (
        Polygon, MultiPolygon,
        LineString, MultiLineString,
        GeometryCollection
    )

    # Single geometries
    if isinstance(geom_data, (Polygon, LineString)):
        return [geom_data] if not geom_data.is_empty else []

    # Multi-geometries
    if isinstance(geom_data, (MultiPolygon, MultiLineString)):
        return [g for g in geom_data.geoms if not g.is_empty]

    # Collections
    if isinstance(geom_data, GeometryCollection):
        return [g for g in geom_data.geoms if not g.is_empty]

    return []
```

### Topic 2: Optimizing for Large Datasets

**Batch Processing:**
```python
from tqdm import tqdm  # Progress bar

def batch_convert(plan_indices, output_dir):
    """Convert multiple plans efficiently."""
    plans = load_dataset()

    for idx in tqdm(plan_indices, desc="Converting"):
        try:
            plan = plans[idx]
            converter = FloorPlanConverter(plan)
            converter.save_npz(f"{output_dir}/plan_{idx:05d}.npz")
        except Exception as e:
            print(f"Failed plan {idx}: {e}")
            continue
```

**Memory Efficiency:**
```python
# Bad: Load all plans into memory
plans = load_all_plans()
for plan in plans:
    convert(plan)

# Good: Stream one at a time
for idx in plan_indices:
    plan = load_single_plan(idx)
    convert(plan)
    del plan  # Free memory
```

### Topic 3: Validation and Quality Control

```python
def validate_conversion(plan, grid, door_positions):
    """Check conversion quality."""
    issues = []

    # 1. Check area preservation (within 5% tolerance)
    real_area = plan.get('net_area', 0)
    grid_area = np.sum(grid == 0) * (0.3 ** 2)  # Passable cells × cell_area
    ratio = grid_area / real_area

    if not (0.95 <= ratio <= 1.05):
        issues.append(f"Area mismatch: {ratio:.2%} of expected")

    # 2. Check connectivity (all passable cells reachable)
    from scipy.ndimage import label
    labeled, num_components = label(grid == 0)

    if num_components > 1:
        issues.append(f"Disconnected: {num_components} separate regions")

    # 3. Check door positions are valid
    for row, col in door_positions:
        if grid[row, col] != -2:
            issues.append(f"Door at ({row},{col}) not on wall")

    return issues
```

### Topic 4: Visualization for Debugging

```python
import matplotlib.pyplot as plt

def visualize_conversion(grid, door_positions, exit_positions):
    """Visual debugging tool."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display grid
    display = np.ones_like(grid)
    display[grid == 0] = 1.0   # White = passable
    display[grid == -2] = 0.0  # Black = walls

    ax.imshow(display, cmap='gray', origin='upper')

    # Mark doors (green)
    if len(door_positions) > 0:
        ax.scatter(door_positions[:, 1], door_positions[:, 0],
                  c='green', s=50, marker='o', label='Doors')

    # Mark exits (red)
    if len(exit_positions) > 0:
        ax.scatter(exit_positions[:, 1], exit_positions[:, 0],
                  c='red', s=100, marker='*', label='Exits')

    ax.legend()
    ax.set_title('Floor Plan Grid')
    plt.show()
```

---

## Common Pitfalls

### Pitfall 1: Coordinate Order Confusion

**Problem:**
```python
# Shapely uses (x, y)
point = Point(10, 20)  # x=10, y=20

# NumPy arrays use [row, col] = [y, x]
grid[20, 10]  # row=20, col=10

# cv2.fillPoly uses [(x, y), ...]
coords = np.array([[10, 20], [15, 25]])  # x, y order
```

**Solution:** Be explicit:
```python
def _world_to_grid(self, x, y):
    # Input: x, y (Shapely convention)
    col = int(x_scaled / cell_size)
    row = int(y_scaled / cell_size)
    # Output: row, col (NumPy convention)
    return (row, col)
```

### Pitfall 2: Using Bounding Box Area

**Wrong:**
```python
x_min, y_min, x_max, y_max = polygon.bounds
area = (x_max - x_min) * (y_max - y_min)  # WRONG for non-rectangular!
```

**Right:**
```python
area = polygon.area  # Always correct
```

### Pitfall 3: Integer Overflow with Large Grids

**Problem:**
```python
# Large plan: 50m × 50m with 0.05m cells
grid_size = int(50 / 0.05)  # 1000 × 1000
grid = np.zeros((grid_size, grid_size), dtype=np.int8)
# Only values: -128 to 127 ❌
```

**Solution:**
```python
grid = np.zeros((grid_size, grid_size), dtype=np.float32)
# Values: -3.4e38 to 3.4e38 ✓
# Memory: 4 bytes/cell (acceptable)
```

### Pitfall 4: Forgetting Empty Geometry Checks

**Problem:**
```python
inner = plan.get('inner')
x_min, y_min, x_max, y_max = inner.bounds  # Crashes if inner is None!
```

**Solution:**
```python
inner = plan.get('inner')
if inner is None or inner.is_empty:
    raise ValueError("Invalid geometry")
x_min, y_min, x_max, y_max = inner.bounds
```

### Pitfall 5: Not Handling Missing Data

**Problem:**
```python
net_area = plan['net_area']  # KeyError if missing
scale = sqrt(net_area / vector_area)  # ZeroDivisionError if 0
```

**Solution:**
```python
net_area = plan.get('net_area', 0)
if net_area <= 0:
    raise ValueError(f"Invalid net_area: {net_area}")
scale = sqrt(net_area / vector_area)
```

---

## Complete Example: Minimal Converter

Here's a minimal working converter (simplified):

```python
import numpy as np
import cv2
from shapely.geometry import Polygon

class MinimalConverter:
    def __init__(self, plan, cell_size=0.3):
        self.plan = plan
        self.cell_size = cell_size

        # Get dimensions
        inner = plan['inner']
        self.x_min, self.y_min, x_max, y_max = inner.bounds

        # Calculate scale
        net_area = plan['net_area']
        scale = np.sqrt(net_area / inner.area)
        self.scale = scale

        # Grid size
        width = (x_max - self.x_min) * scale
        height = (y_max - self.y_min) * scale
        self.grid_rows = int(np.ceil(height / cell_size))
        self.grid_cols = int(np.ceil(width / cell_size))

    def convert(self):
        # Initialize grid
        grid = np.full((self.grid_rows, self.grid_cols), -2, dtype=np.float32)

        # Rasterize interior
        inner = self.plan['inner']
        coords = np.array(inner.exterior.coords)
        grid_coords = []

        for x, y in coords:
            x_m = (x - self.x_min) * self.scale
            y_m = (y - self.y_min) * self.scale
            col = int(x_m / self.cell_size)
            row = int(y_m / self.cell_size)
            grid_coords.append([col, row])

        cv2.fillPoly(grid, [np.array(grid_coords, dtype=np.int32)], color=0)

        return grid

# Usage
plan = load_plan('plan_0.pkl')
converter = MinimalConverter(plan, cell_size=0.3)
grid = converter.convert()
np.savez_compressed('output.npz', grid=grid)
```

---

## Exercise: Build Your Own

**Challenge:** Convert a simple floor plan to grid.

```python
# 1. Create test data
from shapely.geometry import Polygon, LineString

test_plan = {
    'inner': Polygon([(0, 0), (10, 0), (10, 8), (0, 8)]),  # 10×8 rectangle
    'wall': LineString([(5, 0), (5, 8)]),  # Vertical wall in middle
    'door': LineString([(5, 3.5), (5, 4.5)]),  # Door in wall
    'net_area': 80.0  # 10m × 8m = 80 m²
}

# 2. Expected result
# Grid with cell_size=1.0 should be 10×8
# Should have vertical wall down middle
# Door at row 4, col 5

# 3. Implement converter
# ... your code here ...

# 4. Verify
assert grid.shape == (8, 10)
assert grid[4, 5] == -2  # Door is wall (for post-processing)
assert np.sum(grid == 0) < 80  # Some cells are walls
```

**Solution:** See `resplan_to_npz.py` :)

---

## Further Reading

- **Shapely Documentation:** https://shapely.readthedocs.io/
- **NumPy User Guide:** https://numpy.org/doc/stable/user/index.html
- **OpenCV Tutorials:** https://docs.opencv.org/4.x/
- **ResPlan Paper:** https://arxiv.org/abs/2508.14006
- **Grid-Based Pathfinding:** http://theory.stanford.edu/~amitp/GameProgramming/

---

## Summary

**Key Takeaways:**

1. **Scale using polygon area**, not bounding box
2. **Coordinate systems matter**: (x,y) vs [row,col]
3. **Validate inputs**: Check for None, empty, zero values
4. **Use libraries**: cv2.fillPoly > manual rasterization
5. **Test incrementally**: Visualize at each step

You now understand:
- ✓ Why we need coordinate scaling
- ✓ How to convert vector → grid
- ✓ How to rasterize complex shapes
- ✓ How to handle edge cases
- ✓ How to organize code for maintainability

**Next steps:**
- Add support for curved walls
- Implement multi-floor buildings
- Add furniture/obstacles rasterization
- Optimize for real-time conversion
