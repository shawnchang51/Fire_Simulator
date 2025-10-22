# Fire Evacuation Simulator

A Python-based evacuation simulation system that models agent pathfinding and fire spread in dynamic environments using D* Lite algorithm.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Monte Carlo Simulations (Parallel Execution)](#monte-carlo-simulations-parallel-execution)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Visualization](#visualization)
- [Algorithm Details](#algorithm-details)
- [Troubleshooting](#troubleshooting)

## Features

- **Dynamic Pathfinding**: Agents use D* Lite algorithm to navigate around obstacles and fire
- **Fire Spread Simulation**: Real-time fire propagation with environmental monitoring
- **Monte Carlo Simulations**: Parallel execution for statistical analysis (8-10x speedup)
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

## Quick Start

**Recommended**: Use the visual configurator for easy setup:

```bash
python visual_configurator.py
```

This provides an interactive interface to:
- Design custom map layouts
- Place agents and targets visually
- Add obstacles and initial fire positions
- Configure simulation parameters
- Launch simulations directly

Or run the simulation with default configuration:

```bash
python simulation.py
```

This will:
- Load configuration from [example_configuration.json](example_configuration.json)
- Initialize 5 agents with predefined start positions and targets
- Launch pygame visualization (if available)
- Simulate fire spread and agent evacuation

## Monte Carlo Simulations (Parallel Execution)

For statistical analysis and large-scale testing, use the **Monte Carlo simulation module** which supports **parallel execution** to utilize all CPU cores:

### Quick Start

```bash
# Run 100 simulations in parallel (FASTEST - uses all CPU cores)
python monte_carlo.py --runs 100 --parallel

# Run 50 simulations in serial mode (for debugging)
python monte_carlo.py --runs 50

# Benchmark serial vs parallel performance
python benchmark_parallel.py
```

### Features

- 🚀 **Parallel Execution**: Utilizes all CPU cores for 8-10x speedup
- 🎲 **Random Scenarios**: Each run uses randomized fire and agent positions
- 📊 **Statistical Analysis**: Aggregates results across all runs
- 🔄 **Reproducible**: Control randomness with seed parameter
- ⚙️ **Configurable**: Adjust number of processes and runs

### Command-Line Options

```bash
python monte_carlo.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | string | `example_configuration.json` | Path to configuration file |
| `--runs` | int | 10 | Number of simulation runs |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--parallel` | flag | False | Enable parallel execution |
| `--processes` | int | All cores | Number of parallel processes |

### Usage Examples

```bash
# Small test run (debugging)
python monte_carlo.py --runs 10

# Medium run with parallel execution (recommended)
python monte_carlo.py --runs 100 --parallel

# Large-scale study with custom configuration
python monte_carlo.py --config custom_map.json --runs 500 --parallel

# Use specific number of processes
python monte_carlo.py --runs 200 --parallel --processes 8

# Reproducible results with fixed seed
python monte_carlo.py --runs 100 --parallel --seed 12345
```

### Performance

On a typical multi-core system:

| Cores | Runs | Serial Time | Parallel Time | Speedup |
|-------|------|-------------|---------------|---------|
| 4     | 100  | ~600s       | ~80s          | 7.5x    |
| 8     | 100  | ~600s       | ~70s          | 8.5x    |
| 12    | 100  | ~600s       | ~65s          | 9.2x    |
| 16    | 100  | ~600s       | ~62s          | 9.7x    |

**Recommendation**: Always use `--parallel` for runs > 10

### Output Statistics

After completion, you'll see comprehensive statistics:

```
============================================================
MONTE CARLO SIMULATION SUMMARY
============================================================
Total runs: 100
Mode: Parallel
Processes used: 12
Time elapsed: 65.23 seconds
Average time per run: 0.65 seconds

Statistics:
  Average steps: 145.32
  Average fire damage: 23.45
  Average peak temperature: 850.21
  Average temperature: 425.67
  Total evacuated agents: 450
  Total survived agents: 480
============================================================
```

### Programmatic Usage

```python
from simulation import SimulationConfig
from monte_carlo import run_monte_carlo_parallel
import json

# Load configuration
with open('config.json') as f:
    config = SimulationConfig.from_json(json.load(f))

# Run parallel simulations
results, statistics = run_monte_carlo_parallel(
    config,
    num_runs=1000,
    random_seed=42,
    num_processes=8
)

# Analyze results
print(f"Average evacuation time: {statistics['average_steps']:.2f} steps")
print(f"Survival rate: {statistics['survived_agents']/config.agent_num/1000*100:.1f}%")
```

### Advanced Features

**Random Fire Placement**:
```python
from monte_carlo import replace_fire

# Randomly place 10 fires on valid positions
config = replace_fire(config, num_fires=10)
```

**Random Agent Placement**:
```python
from monte_carlo import replace_agents

# Randomly place 20 agents on valid positions
config = replace_agents(config, num_agents=20)
```

**Export Results**:
```python
import pandas as pd

# Convert results to DataFrame for analysis
df = pd.DataFrame(results)
df.to_csv('monte_carlo_results.csv', index=False)
```

### Documentation

For detailed information, see:
- [MONTE_CARLO_README.md](MONTE_CARLO_README.md) - Complete guide
- [PARALLEL_USAGE.txt](PARALLEL_USAGE.txt) - Quick reference

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
