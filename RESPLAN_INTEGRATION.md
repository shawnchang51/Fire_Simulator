# ResPlan Integration Guide

This guide explains how to use the **ResPlan dataset** (17,000 residential floor plans) to generate realistic floor plans for fire evacuation simulations.

## Overview

ResPlan provides:
- **Vector-based floor plans** with Shapely geometries (walls, doors, windows, rooms)
- **17,107 residential layouts** with varied room configurations
- **Room types**: living rooms, bedrooms, bathrooms, kitchens, balconies
- **Architectural elements**: doors, windows, walls with accurate dimensions

The converter (`resplan_to_sim_converter.py`) transforms these vector plans into grid-based configurations compatible with the fire evacuation simulator.

## Quick Start

### 1. Convert a ResPlan Floor Plan

```bash
# Convert a specific plan by index (0-17106)
python resplan_to_sim_converter.py \
    --plan-index 42 \
    --grid-size 100 \
    --num-agents 10 \
    --output configs/resplan_42.json

# Convert a random plan
python resplan_to_sim_converter.py \
    --random \
    --grid-size 150 \
    --num-agents 20 \
    --fire-model realistic \
    --output configs/random_plan.json
```

### 2. Run Simulation with Generated Config

```bash
# Standard simulation with visualization
python simulation.py --config configs/resplan_42.json

# Fast simulation (Phase 2, for RL training)
python run_phase2_visual.py --config configs/resplan_42.json

# Monte Carlo analysis
python monte_carlo.py --config configs/resplan_42.json --runs 100 --parallel
```

## Converter Options

### Required Arguments
- `--output PATH`: Output JSON configuration file path

### Plan Selection (one required)
- `--plan-index N`: Use specific plan (0 to 17106)
- `--random`: Randomly select a plan

### Optional Arguments
- `--grid-size N`: Grid dimensions (default: 100x100)
  - Smaller (50-80): Faster simulation, less detail
  - Medium (100-150): Balanced performance/realism
  - Large (200-300): High detail, slower simulation

- `--num-agents N`: Number of agents to place (default: 10)
  - Agents are placed at room centroids (bedrooms, living rooms, kitchens)

- `--cell-size M`: Cell size in meters (default: 0.3)
  - Standard: 0.3m (human shoulder width)
  - Fine detail: 0.2m
  - Coarse: 0.5m

- `--fire-model TYPE`: Fire propagation model (default: realistic)
  - `realistic`: 3-6 min to flashover, physics-based
  - `aggressive`: 30-60 sec to flashover, stress testing
  - `default`: Original model

- `--pkl-path PATH`: Path to ResPlan.pkl (default: ResPlan/ResPlan.pkl)

## Conversion Process

The converter performs the following transformations:

### 1. **Coordinate Scaling**
- ResPlan uses arbitrary vector coordinates
- Automatically scales to fit grid with 5% margin
- Preserves aspect ratio

### 2. **Wall Rasterization**
- Converts Shapely wall geometries to grid cells
- Walls marked as `-2` (permanent obstacles)
- Configurable wall thickness (default: 2 cells)

### 3. **Door Extraction**
- Internal doors → `{"id": "d0", "position": "x12y5", "type": "door"}`
- Front doors → `{"id": "exit0", "position": "x0y4", "type": "exit"}`
- Enables hierarchical pathfinding with door graphs

### 4. **Agent Placement**
- Extracts room centroids from:
  - Bedrooms (sleeping areas)
  - Living rooms (common areas)
  - Kitchens (potential fire sources)
- Randomly samples positions for agents
- Ensures positions not on walls

### 5. **Fire Initialization**
- Default: Places fire in kitchen (common origin)
- Falls back to center if no kitchen found
- Can be customized in generated JSON

### 6. **Exit Identification**
- Front doors become evacuation targets
- Multiple exits supported (first exit used by default)

## Generated Configuration Structure

```json
{
  "map_rows": 100,
  "map_cols": 100,
  "agent_num": 10,
  "start_positions": ["x21y13", "x20y35", ...],
  "targets": ["x0y4"],
  "initial_fire_map": [...],  // 2D array: 0=clear, -2=wall, 1.0=fire
  "door_configs": [
    {"id": "d0", "position": "x83y21", "type": "door"},
    {"id": "exit0", "position": "x0y4", "type": "exit"}
  ],
  "cell_size": 0.3,
  "fire_model_type": "realistic",
  "metadata": {
    "source": "ResPlan",
    "plan_id": 13,
    "unit_type": "Apartment",
    "net_area": 73.13,
    "scale_factor": 0.371
  }
}
```

## Advanced Usage

### Batch Conversion

Convert multiple plans for comparative analysis:

```python
import subprocess
import numpy as np

# Convert 100 random plans
for i in range(100):
    subprocess.run([
        'python', 'resplan_to_sim_converter.py',
        '--random',
        '--grid-size', '120',
        '--num-agents', '15',
        '--output', f'configs/batch/plan_{i:03d}.json'
    ])
```

### Custom Fire Placement

After generation, modify the JSON to place fires strategically:

```python
import json
import numpy as np

# Load generated config
with open('configs/resplan_42.json', 'r') as f:
    config = json.load(f)

# Add fires in multiple locations
fire_map = np.array(config['initial_fire_map'])
fire_map[30, 40] = 1.0  # Additional fire location
fire_map[60, 70] = 1.0

config['initial_fire_map'] = fire_map.tolist()

# Save modified config
with open('configs/resplan_42_multi_fire.json', 'w') as f:
    json.dump(config, f, indent=2)
```

### Programmatic Conversion

Use the converter as a Python module:

```python
from resplan_to_sim_converter import load_resplan_dataset, ResPlanConverter
import numpy as np

# Load dataset
plans = load_resplan_dataset('ResPlan/ResPlan.pkl')

# Convert specific plan
plan = plans[42]
converter = ResPlanConverter(plan, grid_size=120, cell_size=0.3)

# Generate config with custom parameters
config = converter.generate_config(
    num_agents=15,
    fire_locations=[(60, 30), (80, 40)],  # Multiple fires
    fire_model_type='aggressive'
)

# Access components
fire_map = converter.create_fire_map()
doors = converter.extract_doors()
room_centroids = converter.get_room_centroids()
exits = converter.get_exits()
```

## Integration with RL Training

For reinforcement learning (floor plan optimization), use with Phase 2 fast simulation:

```python
from resplan_to_sim_converter import load_resplan_dataset, ResPlanConverter
from fast_simulation import FastEvacuationSim
import numpy as np

# Load ResPlan dataset
plans = load_resplan_dataset()

# Convert and evaluate multiple plans
scores = []
for i in range(100):
    plan = plans[np.random.randint(len(plans))]
    converter = ResPlanConverter(plan, grid_size=100)
    config = converter.generate_config(num_agents=20)

    # Run fast simulation
    sim = FastEvacuationSim(config)
    sim.run(max_steps=500)

    # Evaluate (e.g., evacuation success rate)
    score = sim.get_evacuation_rate()
    scores.append((i, score))

# Identify best floor plan configurations
top_plans = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
print(f"Top 10 plans: {top_plans}")
```

## Visualization

### Viewing Original ResPlan Floor Plan

Use the provided utilities to visualize the vector-based plan:

```python
import pickle
from ResPlan.resplan_utils import plot_plan, plot_plan_and_graph
import matplotlib.pyplot as plt

# Load plan
with open('ResPlan/ResPlan.pkl', 'rb') as f:
    plans = pickle.load(f)

# Visualize plan with room graph overlay
plot_plan_and_graph(plans[42], title='ResPlan Floor Plan #42')
plt.show()
```

### Viewing Converted Grid-Based Map

After conversion, visualize with the simulator:

```bash
# Pygame visualization (real-time)
python run_phase2_visual.py --config configs/resplan_42.json

# MATLAB-style environmental visualization
python run_with_matlab_viz.py --config configs/resplan_42.json
```

## Troubleshooting

### "No valid room centroids found"
- Plan may have no recognizable rooms
- Try a different plan index
- Check metadata: `config['metadata']['unit_type']`

### "No exits found"
- Plan missing front_door geometry
- Manually add exit in JSON: `"targets": ["x50y50"]`

### Agents stuck or not moving
- Check walls don't completely block paths
- Increase `viewing_range` in config (default: 5)
- Use visual configurator to inspect: `python visual_configurator.py`

### Fire spreads too fast/slow
- Adjust `fire_model_type`: realistic, aggressive, or default
- Modify `fire_update_interval` (default: 4 timesteps = 2 seconds)
- For Phase 2: adjust `fire_spread_rate` and `fire_intensity_growth`

## Dataset Statistics

ResPlan provides diverse floor plans:
- **17,107 total plans**
- **Unit types**: Apartments, houses, studios
- **Size range**: 20-250 m² net area
- **Room variety**: 1-6 bedrooms, 1-4 bathrooms
- **Complexity**: Simple studios to multi-room layouts

Sample plan distribution:
```python
import pickle
plans = pickle.load(open('ResPlan/ResPlan.pkl', 'rb'))

unit_types = [p.get('unitType') for p in plans]
areas = [p.get('net_area', 0) for p in plans]

print(f"Unit types: {set(unit_types)}")
print(f"Area range: {min(areas):.1f} - {max(areas):.1f} m²")
print(f"Average area: {sum(areas)/len(areas):.1f} m²")
```

## Best Practices

1. **Grid Size Selection**
   - Small apartments (< 50m²): 80-100 grid
   - Medium apartments (50-100m²): 100-150 grid
   - Large houses (> 100m²): 150-250 grid

2. **Agent Count**
   - Realistic density: ~1 agent per 10-15 m²
   - For 70m² apartment: 5-7 agents
   - Stress testing: 2-3x normal density

3. **Performance Optimization**
   - Use `--grid-size 100` for balanced performance
   - Phase 2 fast simulation for batch processing
   - `--parallel` flag for Monte Carlo runs > 10

4. **Realistic Scenarios**
   - Fire in kitchen (`fire_model_type='realistic'`)
   - Agents in bedrooms (nighttime evacuation)
   - Consider `fire_discovery_delay` for sleeping agents

## References

- ResPlan Paper: [arXiv:2508.14006](https://arxiv.org/abs/2508.14006)
- ResPlan GitHub: [github.com/m-agour/ResPlan](https://github.com/m-agour/ResPlan)
- Fire Simulator Docs: `CLAUDE.md`, `PHASE2_README.md`
