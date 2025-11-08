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
- **Hierarchical Navigation**: Optional door graph system for strategic multi-room pathfinding
- **Fire Spread Simulation**: Three fire models (realistic, aggressive, default) with environmental monitoring
- **Agent Knowledge Sharing**: Cooperative pathfinding through proximity-based information exchange
- **Monte Carlo Simulations**: Parallel execution for statistical analysis (8-10x speedup)
- **Multiple Visualization Modes**:
  - Pygame graphical interface (recommended)
  - MATLAB-style environmental visualization with interpolation
  - Text-based console output
- **Multi-Agent System**: Supports multiple agents with individual targets, viewing ranges, and fear factors
- **Configurable Environments**: JSON-based configuration for map layout, obstacles, fire positions, and behavior parameters

## Requirements

- Python 3.7+
- numpy
- matplotlib
- pygame (optional, for graphical visualization)
- scipy (optional, for MATLAB-style visualization)
- pandas (optional, for data analysis)
- tqdm (optional, for progress bars in Monte Carlo simulations)

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For advanced features:
```bash
# Graphical visualization
pip install pygame

# MATLAB-style environmental visualization
pip install scipy

# Monte Carlo progress bars
pip install tqdm
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
- Initialize agents with predefined start positions and targets
- Use realistic fire model (3-6 min to flashover, 0.1-0.2 m/s spread)
- Launch pygame visualization (if available)
- Simulate fire spread and agent evacuation
- Export results to `./data/` directory

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
| `--output` | string | `./monte_carlo_results` | Output directory for results |
| `--no-full-results` | flag | False | Save only statistics (memory-efficient for 600+ agents) |

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

# Memory-efficient mode for large simulations
python monte_carlo.py --runs 100 --parallel --no-full-results

# Custom output directory
python monte_carlo.py --runs 100 --parallel --output ./my_results
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

### Output Files

Results are saved to `./monte_carlo_results/{config_name}_{timestamp}/`:
- `full_results.json`: Complete simulation data (optional, can be disabled with `--no-full-results`)
- `summary.txt`: Human-readable summary with statistics
- `statistics.json`: Aggregated metrics (averages, percentiles, distributions)
- `config_used.json`: Configuration snapshot for reproducibility

## Configuration

Create a JSON configuration file with the following structure:

```json
{
  "map_rows": 60,
  "map_cols": 60,
  "max_occupancy": 2,
  "agent_num": 5,
  "viewing_range": 5,
  "cell_size": 0.3,
  "timestep_duration": 0.5,
  "fire_update_interval": 4,
  "fire_model_type": "realistic",
  "start_positions": ["x12y9", "x7y11", "x6y8", "x12y14", "x13y3"],
  "targets": ["x17y2", "x17y17"],
  "initial_fire_map": [[...]],
  "door_configs": [
    {"id": "d1", "position": "x15y3", "type": "door"},
    {"id": "e1", "position": "x50y35", "type": "exit"}
  ],
  "agent_fearness": [1.0, 1.2, 0.8, 1.0, 1.5],
  "consider_env_factors": false,
  "wall_preference": 0.0,
  "communication_range": 15.0,
  "sharing_interval": 5
}
```

### Configuration Parameters

**Required:**
- `map_rows`, `map_cols`: Grid dimensions
- `agent_num`: Number of evacuation agents
- `start_positions`: Agent starting coordinates (format: "x{col}y{row}")
- `targets`: Evacuation target waypoints (agents visit in order)
- `initial_fire_map`: 2D array where:
  - `0` = passable cell
  - `-2` = obstacle/wall
  - `0.0-1.0` = fire intensity

**Optional (with defaults):**
- `max_occupancy`: Maximum agents per cell (default: 2)
- `viewing_range`: Agent's obstacle detection radius (default: 5, auto-scaled based on cell_size)
- `cell_size`: Physical size of each cell in meters (default: 0.3)
- `timestep_duration`: Duration of each timestep in seconds (default: 0.5)
- `fire_update_interval`: Update fire every N timesteps (default: 4)
- `fire_model_type`: Fire model - "realistic", "aggressive", or "default" (default: "realistic")
- `agent_fearness`: Per-agent fear multipliers or single value for all (default: 1.0)
- `door_configs`: Door/exit configurations for hierarchical pathfinding (default: [])
- `consider_env_factors`: Use temperature/smoke in pathfinding costs (default: false)
- `wall_preference`: Wall-following preference (0=none, higher=stronger) (default: 0.0)
- `communication_range`: Distance for agent knowledge sharing in cells (default: 15.0)
- `sharing_interval`: Share knowledge every N timesteps (default: 5)
- `sector_size`: Spatial index sector size (default: auto-calculated)

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
- `use_matlab`: Use MATLAB-style environmental visualization (default: False)

## Visualization

### Pygame (Graphical)
- **Green circles**: Agents with their ID numbers
- **Red squares**: Targets
- **Gray squares**: Obstacles
- **Red/orange gradient**: Fire intensity
- **Blue circles**: Doors (if door_configs enabled)
- Close window or press ESC to quit

### MATLAB-Style Environmental Visualization
Requires scipy. Enable with `use_matlab=True`:
- **Temperature map**: Hot colormap with cubic interpolation
- **Oxygen levels**: Blues_r colormap
- **Smoke density**: Grayscale
- **Fuel remaining**: Greens colormap
- **Fire intensity**: YlOrRd colormap
- **Agent trajectories**: Last 10 positions per agent with anti-overlap offsets
- **Interactive checkboxes**: Toggle layers in real-time

### Text-based (Console)
- `A#`: Agent with ID number
- `T#`: Target waypoint
- `.`: Empty cell
- `#`: Obstacle
- `F`: Fire

## Algorithm Details

### Pathfinding
The simulation uses **D* Lite** for dynamic pathfinding:
- Efficiently recalculates paths when environment changes
- Handles moving obstacles and fire spread
- Supports partial observability (agents have limited viewing range)
- 8-connectivity with diagonal movement cost of √2

### Hierarchical Navigation (Optional)
When `door_configs` is provided:
- **High-level planning**: Door graph using Dijkstra to plan route through doors/exits
- **Low-level navigation**: D* Lite for tactical movement between doors
- **Lazy updates**: Edge weights updated when agents enter rooms
- **Per-agent knowledge**: Each agent maintains independent door graph copy

### Fire Models
Three available models via `fire_model_type`:
- **Realistic** (default): 3-6 min to flashover, 0.1-0.2 m/s spread, suitable for accurate evacuation planning
- **Aggressive**: 30-60 sec to flashover, 0.3-0.5 m/s spread, for stress-testing algorithms
- **Default**: Original model from fire_model_float.py

Fire propagation considers:
- Neighboring fire intensity (distance-weighted)
- Wind direction and speed
- Oxygen availability (fire struggles below 16% O2)
- Fuel density and moisture
- Temperature preheating
- Smoke density

### Knowledge Sharing
When `communication_range` > 0:
- Agents share door graph knowledge with nearby agents
- Spatial indexing enables O(n) proximity queries instead of O(n²)
- Sharing occurs every `sharing_interval` timesteps
- Enables cooperative pathfinding through information exchange

## Troubleshooting

**Pygame not found**: Install with `pip install pygame` or run with text visualization

**Scipy not found for MATLAB visualization**: Install with `pip install scipy`

**Agent stuck**: Occurs when no valid path exists; check obstacle configuration and viewing range

**Import errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Slow Monte Carlo simulations**: Always use `--parallel` flag for runs > 10

**Memory issues with large simulations**: Use `--no-full-results` flag to save only statistics

**Fire spreads too fast/slow**: Adjust `fire_model_type` or `fire_update_interval` in configuration

## Performance Optimizations

🚀 **NEW: Optimized implementations available!**

We've created optimized versions of the core components with **dramatic performance improvements**:

- **Fire Model: 21.72x faster** (95.4% time reduction)
- **Memory Usage: 73.9% reduction** for fire calculations
- **D* Lite Grid: 1.47x faster** (32% time reduction)
- **Overall: ~3-4x faster** simulations

### Using Optimizations

```python
# Option 1: Import optimized modules directly
from fire_model_aggressive_optimized import AdvancedFireModel
from d_star_lite.grid_optimized import GridWorld

# Option 2: Use benchmark script to compare
python benchmark_optimizations.py
```

**Performance Comparison** (60×60 map, 200 steps):
- Original: 68 seconds, 90 MB peak memory
- Optimized: ~22 seconds, 55 MB peak memory
- **Speedup: 3.1x, Memory saved: 39%**

See **[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md)** for detailed analysis.

### Optimizations Implemented

1. **NumPy Arrays** - Replace nested lists with numpy for 2-3x faster operations
2. **Sparse Updates** - Only process cells with active fire (50-70% reduction)
3. **Vectorized Operations** - Bulk environmental updates using numpy masking
4. **Cost Caching** - Cache terrain cost calculations in D* Lite grid
5. **Dirty Tracking** - Only update changed cells and their neighbors

### Profiling Tools

```bash
# Analyze performance bottlenecks
python profile_simulator.py

# Compare original vs optimized
python benchmark_optimizations.py
```

## Project Structure

```
Fire_Simulator/
├── simulation.py                        # Main simulation engine
├── monte_carlo.py                       # Monte Carlo simulations with parallel execution
├── distribution_analysis.py             # Statistical analysis for Monte Carlo results
├── visual_configurator.py               # Interactive map design tool
├── fire_model_realistic.py              # Realistic fire model (default)
├── fire_model_aggressive.py             # Aggressive fire model for stress-testing
├── fire_model_aggressive_optimized.py   # OPTIMIZED fire model (21x faster)
├── fire_model_float.py                  # Original fire model
├── fire_monitor.py                      # Fire monitoring and data export
├── door_graph.py                        # Hierarchical pathfinding system
├── spatial_index.py                     # Spatial indexing for agent proximity
├── pygame_visualizer.py                 # Pygame graphical visualization
├── matlab_visualizer.py                 # MATLAB-style environmental visualization
├── d_star_lite/                         # D* Lite pathfinding algorithm
│   ├── d_star_lite.py                  # Core algorithm
│   ├── grid.py                         # GridWorld graph
│   ├── grid_optimized.py               # OPTIMIZED grid (1.5x faster)
│   ├── graph.py                        # Base graph structure
│   └── utils.py                        # Utilities
├── profile_simulator.py                 # Profiling tool (time & memory)
├── benchmark_optimizations.py           # Benchmark original vs optimized
├── PROFILING_REPORT.md                  # Initial profiling analysis
├── OPTIMIZATION_REPORT.md               # Detailed optimization results
├── example_configuration.json           # Example configuration file
└── requirements.txt                     # Python dependencies
```

## Additional Resources

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive developer guide with architecture details
- **Example Configuration**: See [example_configuration.json](example_configuration.json) for a complete configuration
- **Data Output**: Simulation results automatically saved to `./data/` directory
- **Monte Carlo Results**: Parallel simulation results in `./monte_carlo_results/`

## License

This project is provided as-is for educational and research purposes.
