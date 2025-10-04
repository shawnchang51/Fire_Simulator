# Fire Evacuation Simulator

A Python-based evacuation simulation system that models agent pathfinding and fire spread in dynamic environments using D* Lite algorithm.

## Features

- **Dynamic Pathfinding**: Agents use D* Lite algorithm to navigate around obstacles and fire
- **Fire Spread Simulation**: Real-time fire propagation with environmental monitoring
- **Multiple Visualization Modes**:
  - Pygame graphical interface (recommended)
  - Text-based console output
- **Multi-Agent System**: Supports multiple agents with individual targets and viewing ranges
- **Configurable Environments**: JSON-based configuration for map layout, obstacles, and initial fire positions

## Requirements

- Python 3.7+
- numpy
- matplotlib
- pygame (optional, for graphical visualization)

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For graphical visualization, ensure pygame is installed:
```bash
pip install pygame
```

## Project Structure

```
Fire_Simulator/
├── simulation.py              # Main simulation engine
├── fire_model_float.py        # Fire spread model
├── fire_monitor.py            # Fire monitoring system
├── pygame_visualizer.py       # Pygame-based visualization
├── visual_configurator.py     # Visual configuration tools
├── snapshot_ainmator.py       # Real-time grid animation
├── example_configuration.json # Sample configuration file
├── d_star_lite/               # D* Lite pathfinding implementation
│   ├── d_star_lite.py
│   ├── grid.py
│   ├── graph.py
│   └── utils.py
└── requirements.txt
```

## Quick Start

Run the simulation with default configuration:

```bash
python simulation.py
```

This will:
- Load configuration from [example_configuration.json](example_configuration.json)
- Initialize 5 agents with predefined start positions and targets
- Launch pygame visualization (if available)
- Simulate fire spread and agent evacuation

## Configuration

Create a JSON configuration file with the following structure:

```json
{
  "map_rows": 20,
  "map_cols": 20,
  "max_occupancy": 2,
  "agent_num": 5,
  "viewing_range": 3,
  "start_positions": ["x12y9", "x7y11", "x6y8", "x12y14", "x13y3"],
  "targets": ["x17y2", "x17y17", "x2y17", "x2y2"],
  "initial_fire_map": [[...]]
}
```

### Configuration Parameters

- `map_rows`, `map_cols`: Grid dimensions
- `max_occupancy`: Maximum agents per cell
- `agent_num`: Number of evacuation agents
- `viewing_range`: Agent's obstacle detection range
- `start_positions`: Agent starting coordinates (format: "x{col}y{row}")
- `targets`: Evacuation target waypoints (agents visit in order)
- `initial_fire_map`: 2D array where:
  - `0` = passable cell
  - `-2` = obstacle/wall
  - `0.0-1.0` = fire intensity

## Usage Examples

### Custom Configuration

```python
import json
from simulation import EvacuationSimulation, SimulationConfig

# Load configuration
with open('my_config.json', 'r') as f:
    config_data = json.load(f)

config = SimulationConfig.from_json(config_data)
sim = EvacuationSimulation(config)
sim.run(max_steps=500)
```

### Programmatic Configuration

```python
from simulation import EvacuationSimulation, SimulationConfig

config = SimulationConfig(
    map_rows=20,
    map_cols=20,
    max_occupancy=2,
    agent_num=3,
    viewing_range=5,
    start_positions=['x0y0', 'x5y5', 'x10y10'],
    targets=['x19y19'],
    initial_fire_map=[[0]*20 for _ in range(20)]
)

sim = EvacuationSimulation(config)
sim.run(max_steps=1000, show_visualization=True, use_pygame=True)
```

### Run Parameters

- `max_steps`: Maximum simulation steps (default: 1000)
- `show_visualization`: Enable text-based visualization (default: True)
- `use_pygame`: Use pygame graphical interface (default: True)

## Visualization

### Pygame (Graphical)
- **Green circles**: Agents with their ID numbers
- **Red squares**: Targets
- **Gray squares**: Obstacles
- **Red/orange gradient**: Fire intensity
- Close window or press ESC to quit

### Text-based (Console)
- `A#`: Agent with ID number
- `T#`: Target waypoint
- `.`: Empty cell

## Algorithm Details

The simulation uses **D* Lite** for dynamic pathfinding, which:
- Efficiently recalculates paths when environment changes
- Handles moving obstacles and fire spread
- Supports partial observability (agents have limited viewing range)

## Troubleshooting

**Pygame not found**: Install with `pip install pygame` or run with text visualization

**Agent stuck**: Occurs when no valid path exists; check obstacle configuration

**Import errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`

## License

This project is provided as-is for educational and research purposes.