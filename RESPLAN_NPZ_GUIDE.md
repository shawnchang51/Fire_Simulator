# ResPlan NPZ Converter Guide

Convert ResPlan's 17,000+ residential floor plans to NPZ format for fire evacuation simulation.

## Output Format

Each NPZ file contains:

```python
{
    'grid': np.ndarray,              # 2D array (rows x cols) with -2=walls/outside, 0=passable
    'door_positions': np.ndarray,    # Nx2 array of (row, col) for door openings
    'exit_positions': np.ndarray,    # Mx2 array of (row, col) for exits
    'plan_id': int,                  # Original ResPlan ID
    'unit_type': str,                # "Apartment", "House", etc.
    'net_area': float,               # Net area in square meters
    'cell_size': float,              # Grid cell size in meters
    'grid_rows': int,                # Grid dimensions
    'grid_cols': int,
    'world_bounds': dict,            # Original coordinate bounds
    'num_doors': int,                # Number of doors
    'num_exits': int                 # Number of exits
}
```

## Quick Start

### 1. Convert Single Plan

```bash
# Convert plan by index (0-17106)
python resplan_to_npz.py --plan-index 100 --cell-size 0.3 --output plan_100.npz

# Convert random plan
python resplan_to_npz.py --random --output random_plan.npz

# Custom cell size and wall thickness
python resplan_to_npz.py --plan-index 42 --cell-size 0.2 --wall-thickness 3 --output plan_42.npz
```

### 2. Visualize NPZ File

```bash
# Display interactively
python visualize_npz.py plan_100.npz

# Save to PNG
python visualize_npz.py plan_100.npz --save plan_100_viz.png
```

### 3. Batch Convert Multiple Plans

```bash
# Convert first 100 plans
python batch_convert_resplan.py --start-end 0,100 --output-dir npz_plans/

# Convert 50 random plans
python batch_convert_resplan.py --random 50 --output-dir npz_plans/

# Convert specific plans
python batch_convert_resplan.py --indices 0,10,42,100,500,1000 --output-dir npz_plans/
```

## Conversion Details

### Grid Values

- **0**: Passable interior space (where agents can move)
- **-2**: Walls and any space outside the building

### Door Handling

Each door has **exactly one passable cell** (value = 0) at the door's centroid:
- Internal doors connect rooms
- The rest of the door geometry remains as wall (-2)
- Doors are automatically detected from ResPlan's door geometries

### Exit Handling

Front doors (exits) are treated similarly:
- One passable cell per exit
- Exits are marked separately in `exit_positions` array
- Some plans may have 0 exits if front_door geometry is missing

### Interior Space Definition

Uses ResPlan's `inner` polygon to define the building interior:
- Excludes balconies (as desired)
- Determines the grid dimensions based on actual size
- Everything outside `inner` is marked as -2

### No Auto-Scaling

Unlike the JSON converter, this maintains **actual dimensions**:
- Grid size determined by: `ceil(plan_width / cell_size)` x `ceil(plan_height / cell_size)`
- Example: 73 sqm apartment with 0.3m cells → ~512x854 grid
- Cell size is configurable (default: 0.3m = human shoulder width)

## Command-Line Options

### resplan_to_npz.py

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--plan-index` | int | - | Specific plan to convert (0-17106) |
| `--random` | flag | - | Select random plan |
| `--cell-size` | float | 0.3 | Cell size in meters |
| `--wall-thickness` | int | 2 | Wall thickness in grid cells |
| `--output` | str | **required** | Output NPZ file path |
| `--pkl-path` | str | ResPlan/ResPlan.pkl | Path to dataset |

### batch_convert_resplan.py

| Option | Type | Description |
|--------|------|-------------|
| `--start-end` | str | Range "START,END" (e.g., "0,100") |
| `--random` | int | Convert N random plans |
| `--indices` | str | Comma-separated indices (e.g., "0,10,42") |
| `--output-dir` | str | **required** - Output directory |
| `--cell-size` | float | Cell size in meters (default: 0.3) |
| `--wall-thickness` | int | Wall thickness in cells (default: 2) |

## Loading NPZ Files

### Python

```python
import numpy as np

# Load NPZ
data = np.load('plan_100.npz', allow_pickle=True)

# Access data
grid = data['grid']                    # 2D floor plan array
door_positions = data['door_positions']  # Nx2 array
exit_positions = data['exit_positions']  # Mx2 array

# Metadata
plan_id = int(data['plan_id'])
unit_type = str(data['unit_type'])
net_area = float(data['net_area'])
cell_size = float(data['cell_size'])

print(f"Grid shape: {grid.shape}")
print(f"Passable cells: {np.sum(grid == 0)}")
print(f"Doors: {len(door_positions)}")
```

### Using the Helper Function

```python
from resplan_to_npz import load_npz

# Load with parsed metadata
plan_data = load_npz('plan_100.npz')

grid = plan_data['grid']
doors = plan_data['door_positions']
exits = plan_data['exit_positions']
metadata = plan_data['metadata']

print(f"Plan ID: {metadata['plan_id']}")
print(f"Unit type: {metadata['unit_type']}")
print(f"Area: {metadata['net_area']} sqm")
```

## Integration with Simulator

### Convert NPZ to Simulation Config

```python
import numpy as np
import json

# Load NPZ
data = np.load('plan_100.npz', allow_pickle=True)
grid = data['grid']
exit_positions = data['exit_positions']
door_positions = data['door_positions']

# Create simulation config
config = {
    "map_rows": grid.shape[0],
    "map_cols": grid.shape[1],
    "initial_fire_map": grid.tolist(),
    "cell_size": float(data['cell_size']),

    # Place agents randomly in passable areas
    "agent_num": 10,
    "start_positions": generate_agent_positions(grid, num=10),

    # Set exit as target
    "targets": [f"x{exit_positions[0,1]}y{exit_positions[0,0]}"] if len(exit_positions) > 0 else [],

    # Door configurations for hierarchical pathfinding
    "door_configs": [
        {"id": f"d{i}", "position": f"x{col}y{row}", "type": "door"}
        for i, (row, col) in enumerate(door_positions)
    ] + [
        {"id": f"exit{i}", "position": f"x{col}y{row}", "type": "exit"}
        for i, (row, col) in enumerate(exit_positions)
    ]
}

def generate_agent_positions(grid, num=10):
    """Generate random agent starting positions in passable cells."""
    passable = np.argwhere(grid == 0)
    if len(passable) < num:
        num = len(passable)
    indices = np.random.choice(len(passable), size=num, replace=False)
    positions = []
    for idx in indices:
        row, col = passable[idx]
        positions.append(f"x{col}y{row}")
    return positions

# Save config
with open('sim_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

## Cell Size Selection

Choose based on your needs:

| Cell Size | Grid Size (70 sqm apt) | Use Case |
|-----------|------------------------|----------|
| 0.5m | ~300x500 | Fast simulation, coarse detail |
| 0.3m | ~500x850 | **Recommended** - balanced |
| 0.2m | ~750x1280 | High detail, slower |
| 0.1m | ~1500x2560 | Very fine detail, very slow |

**Recommendation**: Use **0.3m** (default) for most cases. This matches human shoulder width and provides good detail without excessive computational cost.

## Examples

### Example 1: Dataset Exploration

```python
from resplan_to_npz import load_resplan_dataset
import numpy as np

# Load dataset
plans = load_resplan_dataset('ResPlan/ResPlan.pkl')

# Find plans with specific characteristics
apartments = [p for p in plans if p.get('unitType') == 'Apartment']
medium_size = [p for p in apartments if 50 <= p.get('net_area', 0) <= 100]

print(f"Total plans: {len(plans)}")
print(f"Apartments: {len(apartments)}")
print(f"Medium size (50-100 sqm): {len(medium_size)}")

# Convert a medium-sized apartment
plan_idx = plans.index(medium_size[0])
print(f"Converting plan {plan_idx}...")

# Use batch converter for the medium-sized apartments
indices = [plans.index(p) for p in medium_size[:20]]
# Then use batch_convert_resplan.py with --indices
```

### Example 2: Quality Filtering

```bash
# Convert 100 plans and filter by characteristics
python batch_convert_resplan.py --random 100 --output-dir npz_candidates/
```

Then filter programmatically:

```python
import numpy as np
from pathlib import Path

npz_dir = Path('npz_candidates')
good_plans = []

for npz_file in npz_dir.glob('*.npz'):
    data = np.load(npz_file, allow_pickle=True)

    # Quality criteria
    has_exit = len(data['exit_positions']) > 0
    reasonable_size = 200 <= data['grid_rows'] * data['grid_cols'] <= 1000000
    has_doors = len(data['door_positions']) >= 3

    if has_exit and reasonable_size and has_doors:
        good_plans.append(npz_file)

print(f"Found {len(good_plans)} good quality plans")

# Copy to final directory
import shutil
final_dir = Path('npz_final')
final_dir.mkdir(exist_ok=True)
for plan in good_plans:
    shutil.copy(plan, final_dir / plan.name)
```

### Example 3: Batch Processing for ML Training

```python
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Load all NPZ files
npz_dir = Path('npz_plans')
plans = []

for npz_file in tqdm(list(npz_dir.glob('*.npz')), desc="Loading plans"):
    data = np.load(npz_file, allow_pickle=True)
    plans.append({
        'grid': data['grid'],
        'doors': data['door_positions'],
        'exits': data['exit_positions'],
        'plan_id': int(data['plan_id'])
    })

print(f"Loaded {len(plans)} plans")

# Create training dataset
# Example: predict evacuation difficulty from floor plan
X = np.array([p['grid'].flatten() for p in plans])  # Flatten grids
y = np.array([len(p['exits']) for p in plans])      # Target: number of exits

print(f"Training data shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Use for RL training, scoring network, etc.
```

## Troubleshooting

### Issue: Grid too large (> 1M cells)

**Cause**: Large building with small cell size
**Solution**: Increase `--cell-size` to 0.5 or 1.0

```bash
python resplan_to_npz.py --plan-index 42 --cell-size 0.5 --output plan_42.npz
```

### Issue: No exits found

**Cause**: Some ResPlan entries have missing `front_door` geometry
**Solutions**:
- Try different plan indices
- Manually add exit positions after conversion
- Filter during batch processing

### Issue: Doors not connecting rooms

**Cause**: Door centroid calculation may not align perfectly with passable space
**Solutions**:
- Increase `--wall-thickness` to ensure thicker walls
- Manually adjust door positions in post-processing
- Use smaller `--cell-size` for better resolution

### Issue: Too much passable space (should be wall)

**Cause**: Wall geometry may have gaps in ResPlan data
**Solution**: Increase `--wall-thickness` parameter

```bash
python resplan_to_npz.py --plan-index 42 --wall-thickness 3 --output plan_42.npz
```

## File Sizes

Approximate NPZ file sizes:

| Grid Size | Approx. File Size (compressed) |
|-----------|-------------------------------|
| 300x500 | ~150 KB |
| 500x850 | ~400 KB |
| 750x1280 | ~900 KB |
| 1000x1500 | ~1.5 MB |

For batch conversion of 1000 plans @ 500x850 grid: ~400 MB total

## Performance

Conversion speed (approximate):

| Grid Size | Time per Plan |
|-----------|---------------|
| 300x500 | ~0.5 sec |
| 500x850 | ~1-2 sec |
| 1000x1500 | ~3-5 sec |

Batch converting 100 plans @ 500x850: ~2-3 minutes

## Best Practices

1. **Start with visualization**: Always visualize a few converted plans to verify quality
2. **Filter by quality**: Not all ResPlan entries have complete data - filter after conversion
3. **Consistent cell size**: Use the same cell size for all plans in a dataset
4. **Batch processing**: Use batch converter for multiple plans
5. **Storage**: Compress directories of NPZ files for long-term storage

## References

- **ResPlan Paper**: [arXiv:2508.14006](https://arxiv.org/abs/2508.14006)
- **ResPlan Dataset**: `ResPlan/ResPlan.pkl` (17,107 floor plans)
- **Fire Simulator**: See `CLAUDE.md` for simulation details
