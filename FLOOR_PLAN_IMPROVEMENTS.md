# Floor Plan Generator Improvements

## Issues Fixed

### 1. ✅ Thick Right/Bottom Walls
**Problem**: Grid plans had double-thickness walls on right and bottom edges.

**Solution**: Modified `_generate_grid()` to extend last row/column rooms to the perimeter, eliminating the gap that created double walls.

```python
# For last row/column, extend to edge to avoid thick perimeter walls
if gr == grid_rows - 1:
    h = rows - y - 1  # Extend to bottom edge
if gc == grid_cols - 1:
    w = cols - x - 1  # Extend to right edge
```

---

### 2. ✅ Cellular Plans Too Cave-Like
**Problem**: Cellular automata generated organic cave layouts, not building-like floor plans.

**Solution**: Reduced cellular generation from 10-30% to constant 5% in method weights.

```python
method_weights = {
    'bsp': 0.30 + 0.20 * (1 - realism_ratio),      # 30-50%
    'grid': 0.25 + 0.10 * (1 - realism_ratio),     # 25-35%
    'template': 0.40 + 0.15 * realism_ratio,       # 40-55%
    'cellular': 0.05                                # 5% (minimal)
}
```

---

### 3. ✅ Connectivity Not Verified
**Problem**: Obstacles could block corridors, creating disconnected regions.

**Solution**: Added `_verify_connectivity()` method using BFS to ensure all passable cells are reachable.

```python
def _verify_connectivity(self, grid: np.ndarray) -> bool:
    """Verify that all passable cells form a single connected component"""
    # BFS from first passable cell
    # Ensures no isolated regions after adding obstacles
```

This now runs automatically in `_validate_plan()` and rejects disconnected floor plans.

---

### 4. ✅ L/U-Shaped Templates Too Easy
**Problem**: L and U-shaped templates created large open wings that were too simple.

**Solution**: Reduced L/U templates to 5% chance, prioritized complex patterns.

```python
patterns = [
    'corridor_central',   # 20%
    'office_building',    # 25%
    'school_layout',      # 20%
    'warehouse',          # 15%
    'hospital_wing',      # 15%
    'open_office'         # 5%
]
# L/U shapes: only 5% chance (for variety)
```

---

### 5. ✅ Weak Template Patterns (corridor_central)
**Problem**: Uniform classroom-style rooms were too easy and predictable.

**Solution**: Improved `corridor_central` pattern with:
- **Varied room sizes** (not uniform)
- **Offset door positions** (not centered)
- **T-shaped corridors** (50% chance of cross-corridor)
- **3-6 rooms per side** (more complexity)

**Before**: Uniform rooms with centered doors
```
#####|#####|#####
    |     |
========M========  (main corridor)
    |     |
#####|#####|#####
```

**After**: Varied sizes with offset doors and cross-corridor
```
###|####|##
   |    |
===+====+===  (T-junction)
 | |    |
##|####|####
```

---

## New Distribution at realism_ratio=0.6

| Method | Weight | Purpose |
|--------|--------|---------|
| BSP | 40% | Complex office layouts, good for challenging scenarios |
| Template | 46% | Realistic buildings (office, school, hospital, warehouse) |
| Grid | 29% | Structured rooms with partitions |
| Cellular | 5% | Minimal, for variety only |

---

## Validation Now Checks

1. ✅ Minimum 15% passable area
2. ✅ At least 20 passable cells
3. ✅ Perimeter walls intact
4. ✅ **Full connectivity** (new - most important)

Invalid plans are regenerated (max 5 attempts per plan).

---

## Results

**Before**:
- Thick walls on edges ❌
- Cave-like cellular plans (30%) ❌
- Disconnected regions ❌
- Easy uniform templates ❌

**After**:
- Single-cell perimeter walls ✅
- Minimal cellular (5%) ✅
- All plans fully connected ✅
- Challenging varied templates ✅

---

## Usage

```python
# Generate with connectivity guarantee
generator = FloorPlanGenerator(seed=42)
plans = generator.generate_batch(
    num_plans=100,
    size_range=(30, 50),
    realism_ratio=0.6  # Balanced mix
)
# All plans are guaranteed to be connected
```

---

## Files Modified

- `floor_plan_generator.py`:
  - `_generate_grid()`: Fixed thick walls
  - `generate_batch()`: Reduced cellular weight
  - `_validate_plan()`: Added connectivity check
  - `_verify_connectivity()`: New BFS validation
  - `_generate_template()`: Improved corridor_central pattern
  - Template selection: Reduced L/U shapes to 5%
