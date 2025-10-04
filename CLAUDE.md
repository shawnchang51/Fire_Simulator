# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fire Evacuation Simulator - A Python-based evacuation simulation system that models agent pathfinding and fire spread in dynamic environments using the D* Lite algorithm.

## Running and Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation with default configuration
python simulation.py

# Run with custom configuration
python simulation.py  # Edit example_configuration.json first
```

## Architecture

### Core System Components

**Simulation Engine** ([simulation.py](simulation.py))
- `EvacuationSimulation`: Main simulation orchestrator
- `EvacuationAgent`: Individual agent with D* Lite pathfinding
- `SimulationConfig`: Configuration dataclass loaded from JSON
- Agents share a single `occupancy` grid to coordinate movements and avoid excessive density

**Fire Model** ([fire_model_float.py](fire_model_float.py))
- `AdvancedFireModel`: Cellular automata-based fire spread simulation
- `EnvironmentalParameters`: Environmental conditions (wind, oxygen, humidity, fuel, temperature)
- Tracks oxygen consumption, temperature, smoke density, fuel depletion
- Fire states: 0 (clear), 1-4 (fire intensity), -1 (inaccessible), -2 (permanent obstacle)

**Fire Monitoring** ([fire_monitor.py](fire_monitor.py))
- `FireMonitor`: Tracks simulation history and environmental statistics
- Collects oxygen levels, temperatures, smoke density, fuel consumption per step
- Exports monitoring data to JSON and CSV formats

**D* Lite Pathfinding** ([d_star_lite/](d_star_lite/))
- `d_star_lite.py`: Core D* Lite algorithm implementation with dynamic replanning
- `grid.py`: GridWorld graph with 8-connectivity and terrain cost calculations
- `graph.py`: Base graph structure with nodes, parents, children
- `utils.py`: Coordinate conversion utilities (`stateNameToCoords`)

**Visualization**
- [pygame_visualizer.py](pygame_visualizer.py): Graphical visualization (recommended, requires pygame)
- [snapshot_ainmator.py](snapshot_ainmator.py): Real-time grid animation
- [visual_configurator.py](visual_configurator.py): Visual configuration tools

### Key Interactions

1. **Simulation Loop**: `EvacuationSimulation.run()` orchestrates:
   - Each agent moves using D* Lite pathfinding
   - Fire model updates fire spread based on environmental conditions
   - Changes propagate to all agent graphs via `update_environment()`
   - Visualization updates (pygame or text-based)

2. **Agent Movement**: Each `EvacuationAgent`:
   - Maintains its own `GridWorld` graph representing known terrain
   - Scans for obstacles within `viewing_range` (default: 3-5 cells)
   - Replans path dynamically when fire/obstacles detected
   - Updates shared `occupancy` grid to coordinate with other agents
   - Visits targets sequentially until evacuation complete

3. **Fire Propagation**: Fire spreads based on:
   - Neighboring fire intensity (distance-weighted)
   - Wind direction and speed (directional bias)
   - Oxygen availability (fire struggles below 16% O2)
   - Fuel density and moisture
   - Temperature preheating effects
   - Smoke density (reduces oxygen)
   - Changes returned as `{"x{col}y{row}": new_value}` dictionary

4. **Graph Updates**: When fire spreads or obstacles appear:
   - `FireMonitor.monitor_step()` applies changes to fire state
   - `EvacuationSimulation.update_environment()` broadcasts changes
   - Each agent's `update_graph()` modifies terrain costs
   - D* Lite incrementally replans paths without full recomputation

### Coordinate System

- Grid positions use format: `"x{col}y{row}"` (e.g., `"x12y9"`)
- Conversion: `stateNameToCoords("x5y7")` → `(5, 7)` (col, row)
- Grid access: `grid[row][col]` or `grid[y][x]`

### Configuration Format

JSON configuration ([example_configuration.json](example_configuration.json)):
```json
{
  "map_rows": 20,
  "map_cols": 20,
  "max_occupancy": 2,
  "agent_num": 5,
  "viewing_range": 3,
  "start_positions": ["x12y9", "x7y11"],
  "targets": ["x17y2", "x17y17"],
  "initial_fire_map": [[...]]
}
```

- `max_occupancy`: Maximum agents per cell (prevents clustering)
- `viewing_range`: Agent's obstacle detection radius
- `initial_fire_map`: 2D array where 0=passable, -2=obstacle, 0.0-1.0=fire intensity

### D* Lite Algorithm Details

The pathfinding system uses D* Lite for efficient dynamic replanning:

- **OBS_VAL**: Configurable obstacle value (set via `set_OBS_VAL()`, default -1)
- **Replanning**: When terrain changes, D* Lite incrementally updates path instead of full recomputation
- **Partial Observability**: Agents only scan within `viewing_range` and discover obstacles dynamically
- **Cycle Detection**: Agents detect position cycles and perform aggressive rescans to break loops
- **Stuck Handling**: When no valid path exists, agent returns "stuck" status and is removed from simulation

### Important Implementation Notes

- Fire model uses separate environmental maps (oxygen, temperature, smoke, fuel) that interact
- Agents become "stuck" when surrounded by obstacles/fire or when `max_occupancy` prevents movement
- The `k_m` parameter in D* Lite tracks cumulative heuristic changes for consistency
- Terrain costs calculated in `GridWorld.getTerrainCost()`: obstacles → infinity, fire intensity → higher cost
- Graph uses 8-connectivity with diagonal movement cost of √2
- When agents reach a target, they call `reset_for_new_planning()` and `initDStarLite()` for the next target
