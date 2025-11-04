# Aggressive Fire Model API Documentation

## Overview

The Aggressive Fire Model ([fire_model_aggressive.py](fire_model_aggressive.py)) is a high-intensity fire spread simulation designed for stress-testing evacuation algorithms and worst-case scenario planning. It implements a cellular automata-based fire spread system with realistic physics modeling but accelerated parameters.

**Purpose**: Validate evacuation algorithms under extreme conditions, not for accurate evacuation planning.

**Key Characteristics**:
- Time to flashover: 30-60 seconds (vs. 3-6 minutes realistic)
- Spread speed: 0.3-0.5 m/s (vs. 0.1-0.2 m/s realistic)
- Adjacent cell ignition: 40-60% probability (vs. 15-30% realistic)
- Burn duration: 40+ seconds before decay (vs. 20-30 seconds realistic)

---

## Core Concepts

### Spatial and Temporal Scale

```
Cell size: 0.3m × 0.3m (fine-grained indoor space)
Timestep: 0.5 seconds (configured in simulation)
Fire update interval: Every 2-3 timesteps (1-1.5 seconds)
```

### Fire Intensity Scale

The model uses a continuous float scale representing fire development stages:

| Value | Physical State | Temperature | Heat Flux |
|-------|---------------|-------------|-----------|
| 0.0 | Clear/Unburned | 20°C | 0 kW/m² |
| 1.0 | Ignition/Smoldering | 100-300°C | <100 kW/m² |
| 2.0 | Growth phase | 300-500°C | 100-500 kW/m² |
| 3.0 | Fully developed | 500-800°C | 500-1000 kW/m² |
| 4.0 | Flashover | >800°C | >1000 kW/m² |
| -1.0 | Inaccessible | N/A | (no oxygen, extreme conditions) |
| -2.0 | Permanent obstacle | N/A | (walls, barriers) |

**Fire Progression**: Cells ignite at 1.0, grow to 4.0 over 30-60 seconds, then decay back to 0.0 after fuel depletion.

---

## Architecture

### 1. EnvironmentalParameters

Dataclass containing all environmental conditions affecting fire spread.

```python
@dataclass
class EnvironmentalParameters:
    # Wind and airflow
    wind_speed: float = 1.5  # m/s (AGGRESSIVE: high wind)
    wind_direction: float = 0.0  # radians (0=east, π/2=north)
    ventilation_rate: float = 0.3  # air changes per hour

    # Atmospheric conditions
    oxygen_level: float = 21.0  # percentage
    temperature: float = 20.0  # Celsius
    humidity: float = 50.0  # percentage

    # Fuel properties
    fuel_density: float = 1.0  # relative availability
    fuel_moisture: float = 10.0  # percentage

    # Room characteristics
    ceiling_height: float = 2.7  # meters
    room_volume: float = 100.0  # cubic meters

    # Advanced physics (AGGRESSIVE TUNING)
    thermal_conductivity: float = 0.5  # heat transfer coefficient
    ignition_threshold: float = 0.2  # easier ignition
    burn_rate_modifier: float = 1.5  # 50% faster spread
```

**Aggressive Differences from Realistic Model**:
- `wind_speed`: 1.5 m/s (vs. 0.5 m/s realistic)
- `ignition_threshold`: 0.2 (vs. 0.5 realistic) - easier ignition
- `burn_rate_modifier`: 1.5 (vs. 1.0 realistic) - 50% faster

---

### 2. AdvancedFireModel

Main simulation engine implementing cellular automata fire spread.

#### Internal State

```python
class AdvancedFireModel:
    # Grid dimensions
    rows: int
    cols: int

    # Environmental parameters
    env: EnvironmentalParameters

    # Environmental maps (updated each step)
    oxygen_map: List[List[float]]  # oxygen percentage per cell
    temperature_map: List[List[float]]  # temperature in Celsius
    fuel_map: List[List[float]]  # remaining fuel (0.0-1.0)
    smoke_density: List[List[float]]  # smoke accumulation

    # Fire history tracking
    burn_time: List[List[float]]  # timesteps burned
    max_intensity_reached: List[List[float]]  # peak intensity

    # Precomputed wind effects
    wind_influence: List[List[Tuple[float, float]]]  # (x, y) vectors
```

#### Constructor

```python
def __init__(self, rows: int, cols: int,
             env_params: Optional[EnvironmentalParameters] = None)
```

**Parameters**:
- `rows`, `cols`: Grid dimensions (must match simulation grid)
- `env_params`: Custom environmental parameters (optional, defaults to aggressive settings)

**Initialization**:
- Creates environmental maps initialized to ambient conditions
- Precomputes wind influence vectors for efficiency
- Initializes fire history tracking arrays

---

## API Reference

### Primary Interface: simulate_step()

**The main method called by simulation.py each fire update cycle.**

```python
def simulate_step(self, current_state: List[List[float]]) -> Dict[str, float]
```

**Input**:
- `current_state`: 2D list representing current fire grid
  - Dimensions: `[rows][cols]`
  - Values: Fire intensity (0.0-4.0) or obstacle codes (-2, -1)

**Output**:
- Dictionary of changes in format: `{"x{col}y{row}": new_value}`
- Only includes cells that changed this step
- Example: `{"x5y3": 1.0, "x6y3": 2.5, "x4y4": 3.2}`

**Coordinate Format**:
- Keys use `"x{col}y{row}"` format (matches D* Lite coordinate system)
- To extract coordinates: `col = int(key.split('y')[0][1:])`, `row = int(key.split('y')[1])`

**Process Flow**:
1. Updates environmental conditions (oxygen consumption, heat transfer, smoke)
2. For each unburned cell (value = 0):
   - Calculates spread probability from neighbors
   - Randomly ignites if probability threshold met
   - Sets initial intensity to 1.0
3. For each burning cell (0 < value ≤ 4):
   - Progresses fire intensity based on oxygen/fuel
   - Applies decay if fuel depleted or oxygen starved
4. Returns only cells with significant changes (>0.01 difference)

---

### Statistics: get_simulation_statistics()

**Provides analytical data about current fire state.**

```python
def get_simulation_statistics(self) -> Dict[str, float]
```

**Output Dictionary**:
```python
{
    "oxygen_consumed_percent": float,  # Average O2 deficit across grid
    "co_concentration_ppm": float,  # Carbon monoxide estimate
    "average_temperature_rise": float,  # Mean temp above ambient
    "max_temperature_celsius": float,  # Hottest point in grid
    "total_smoke_density": float,  # Sum of smoke across all cells
    "fire_safety_index": float  # 0-100 safety score (lower = worse)
}
```

**Usage**: Called by `FireMonitor` for data collection and export.

---

### Factory Functions

#### create_fire_model()

**Convenience constructor with keyword arguments.**

```python
def create_fire_model(rows: int, cols: int, **env_kwargs) -> AdvancedFireModel
```

**Example**:
```python
model = create_fire_model(
    60, 60,
    wind_speed=2.0,
    wind_direction=0.785,  # 45 degrees
    humidity=20.0,
    fuel_density=1.2
)
```

#### simulate_fire_spread()

**Simple one-shot interface (not typically used in main simulation).**

```python
def simulate_fire_spread(fire_states: List[List[float]],
                        rows: int, cols: int,
                        **environmental_params) -> Dict[str, float]
```

**Note**: Creates new model instance each call - inefficient for continuous simulation.

---

## Integration with simulation.py

### 1. Initialization

The simulation creates the fire model based on configuration:

```python
# In EvacuationSimulation.__init__()
if config.fire_model_type == "aggressive":
    from fire_model_aggressive import create_fire_model
    self.fire_model = create_fire_model(
        config.map_rows,
        config.map_cols,
        wind_speed=1.5,  # Can be customized
        humidity=30.0
    )
```

**Configuration Parameter**: `"fire_model_type": "aggressive"` in JSON config.

### 2. Fire Update Cycle

Fire spreads periodically during simulation:

```python
# In EvacuationSimulation.run()
if timestep % fire_update_interval == 0:
    # Call fire model to get changes
    changes = self.fire_model.simulate_step(self.fire_map)

    # Apply changes to simulation's fire map
    for pos_key, new_value in changes.items():
        col = int(pos_key.split('y')[0][1:])
        row = int(pos_key.split('y')[1])
        self.fire_map[row][col] = new_value

    # Propagate changes to all agents
    self.update_environment(changes)
```

**Timing**:
- Default: `fire_update_interval = 4` timesteps
- With `timestep_duration = 0.5s`: Fire updates every 2 seconds
- Configurable via `fire_update_interval` in configuration JSON

### 3. Agent Graph Updates

Changes propagate to agent pathfinding graphs:

```python
# In EvacuationSimulation.update_environment()
for agent in self.agents:
    agent.update_graph(changes)

# In EvacuationAgent.update_graph()
for state, new_cost in changes.items():
    # Update terrain cost in D* Lite graph
    old_cost = self.graph.cells[state]
    self.graph.cells[state] = new_cost

    # Trigger D* Lite replanning if significant change
    if abs(new_cost - old_cost) > threshold:
        self.dstar.updateVertex(state)
```

**Cost Mapping**: Fire intensity directly affects pathfinding costs:
- Intensity 0.0 → Low cost (passable)
- Intensity 1.0-2.0 → Medium cost (agent avoids if possible)
- Intensity 3.0-4.0 → High cost (agent strongly avoids)
- Value -2 → Infinite cost (permanent obstacle)

### 4. Monitoring and Data Export

FireMonitor tracks changes for analysis:

```python
# In FireMonitor.monitor_step()
self.fire_history.append(copy.deepcopy(current_fire_map))

# Get environmental statistics from fire model
stats = fire_model.get_simulation_statistics()
self.oxygen_history.append(fire_model.oxygen_map)
self.temperature_history.append(fire_model.temperature_map)
self.smoke_history.append(fire_model.smoke_density)

# Export to JSON/CSV at end of simulation
monitor.export_data()
```

---

## Fire Physics Implementation

### Spread Probability Calculation

The aggressive model calculates ignition probability using multiple factors:

```python
def _calculate_spread_probability(self, current_state, row, col) -> float:
    # 1. Neighbor fire contribution (distance-weighted)
    for each burning neighbor:
        distance = sqrt((row - nr)^2 + (col - nc)^2)
        weight = 1.0 / distance
        contribution += neighbor_intensity * weight

    # 2. Wind direction bonus (30% boost if wind blows toward cell)
    if wind_blowing_toward_cell:
        direction_bonus = 1.0 + (wind_strength * 0.3)

    # 3. Base probability (AGGRESSIVE)
    base_prob = min(contribution * 0.08, 0.7)  # vs. 0.03, 0.5 realistic

    # 4. Environmental modifiers
    final_prob = base_prob *
                 oxygen_factor *       # Fire needs oxygen
                 fuel_factor *         # Fire needs fuel
                 moisture_penalty *    # Wet fuel resists
                 temp_bonus *          # Preheated surfaces ignite easier
                 smoke_penalty *       # Smoke reduces oxygen
                 wind_bonus *          # Wind accelerates spread
                 humidity_factor *     # Dry air accelerates
                 burn_rate_modifier    # Global speed multiplier (1.5 for aggressive)

    return min(final_prob, 0.95)  # Cap at 95%
```

**Aggressive Tuning**:
- Base multiplier: 0.08 (vs. 0.03 realistic) = **2.67× faster spread**
- Base cap: 0.7 (vs. 0.5 realistic) = **40% higher max probability**
- `burn_rate_modifier`: 1.5 = **Additional 50% speed boost**

### Fire Intensity Progression

Burning cells evolve through growth and decay phases:

```python
def _calculate_fire_progression(self, current_intensity, row, col) -> float:
    # Growth phase (1.0 → 4.0)
    if current_intensity < 4.0 and sufficient_oxygen_and_fuel:
        growth_rate = 0.2 * oxygen_factor * fuel_factor  # AGGRESSIVE
        return min(4.0, current_intensity + growth_rate)

    # Decay phase (triggers when...)
    if fuel_depleted or oxygen_starved or burn_duration > 40_timesteps:
        decay_rate = 0.15  # Slower decay = fire persists longer
        return max(0.0, current_intensity - decay_rate)

    return current_intensity  # Stable burning
```

**Growth Speed**:
- Realistic: 0.08 per step → 50 steps to flashover (25 seconds at 0.5s timesteps)
- Aggressive: 0.2 per step → **15 steps to flashover (7.5 seconds)**
- With fire_update_interval=4: **30-60 seconds to flashover** (as documented)

### Environmental Updates

Each simulation step updates environmental conditions:

```python
def _update_environmental_conditions(self, current_state):
    for each burning cell:
        # Oxygen consumption (AGGRESSIVE)
        oxygen_consumption = fire_intensity * 0.15  # vs. 0.05 realistic
        oxygen_map[cell] -= oxygen_consumption

        # Temperature increase (AGGRESSIVE)
        heat_production = fire_intensity * 12.0  # vs. 5.0 realistic
        temperature_map[cell] += heat_production

        # Smoke production
        smoke_production = fire_intensity * smoke_density_factor
        smoke_density[cell] += smoke_production

        # Fuel consumption
        fuel_consumption = fire_intensity * 0.02
        fuel_map[cell] -= fuel_consumption

        # Heat dissipation to neighbors
        heat_transfer = temp_difference * thermal_conductivity * 0.1
        neighbor_temperature += heat_transfer

    # Gradual oxygen replenishment through ventilation
    oxygen_map[cell] += ventilation_rate * 0.1

    # Smoke dissipation
    smoke_density[cell] -= ventilation_rate * 0.05
```

**Key Differences**:
- Oxygen consumption: **3× faster** (0.15 vs. 0.05)
- Heat production: **2.4× faster** (12.0 vs. 5.0)
- Creates rapid environmental degradation

---

## Usage Examples

### Basic Usage with Simulation

```python
from simulation import EvacuationSimulation, SimulationConfig
import json

# Load configuration with aggressive fire model
with open('example_configuration.json', 'r') as f:
    config_dict = json.load(f)
    config_dict['fire_model_type'] = 'aggressive'

config = SimulationConfig(**config_dict)
sim = EvacuationSimulation(config)

# Fire model automatically created and integrated
sim.run(show_visualization=True, use_pygame=True)

# Access fire statistics
stats = sim.fire_model.get_simulation_statistics()
print(f"Max temperature: {stats['max_temperature_celsius']:.1f}°C")
print(f"Safety index: {stats['fire_safety_index']:.1f}")
```

### Custom Environmental Parameters

```python
from fire_model_aggressive import create_fire_model, EnvironmentalParameters

# Create custom environment (e.g., windy, dry conditions)
custom_env = EnvironmentalParameters(
    wind_speed=2.5,           # Very high wind
    wind_direction=0.785,     # 45 degrees (northeast)
    humidity=15.0,            # Very dry
    fuel_density=1.5,         # High fuel load
    burn_rate_modifier=2.0    # Even more aggressive
)

fire_model = create_fire_model(60, 60, env_params=custom_env)

# Use in simulation loop
fire_state = [[0.0] * 60 for _ in range(60)]
fire_state[30][30] = 2.0  # Initial fire

for step in range(100):
    changes = fire_model.simulate_step(fire_state)

    # Apply changes
    for pos, value in changes.items():
        col = int(pos.split('y')[0][1:])
        row = int(pos.split('y')[1])
        fire_state[row][col] = value

    print(f"Step {step}: {len(changes)} cells changed")
```

### Analyzing Fire Spread Patterns

```python
import matplotlib.pyplot as plt
import numpy as np

# Run simulation and collect data
fire_history = []
for step in range(200):
    changes = fire_model.simulate_step(fire_state)
    # Apply changes...
    fire_history.append(np.array(fire_state))

# Visualize fire spread over time
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    step_idx = i * 25
    ax.imshow(fire_history[step_idx], cmap='hot', vmin=0, vmax=4)
    ax.set_title(f'Step {step_idx} ({step_idx * 0.5:.1f}s)')
    ax.axis('off')
plt.tight_layout()
plt.show()

# Calculate spread velocity
burned_cells = [(arr > 0).sum() for arr in fire_history]
spread_rate = np.diff(burned_cells)  # cells per timestep
print(f"Average spread: {np.mean(spread_rate):.2f} cells/step")
print(f"Peak spread: {np.max(spread_rate)} cells/step")
```

---

## Performance Characteristics

### Computational Complexity

- **Per Step**: O(rows × cols × k) where k ≈ 8 (neighbor checks)
- **Typical Grid**: 60×60 = 3600 cells → ~28,800 operations
- **Runtime**: ~5-15ms per fire update on modern CPU
- **Memory**: ~4-6 MB for 60×60 grid with all environmental maps

### Optimization Notes

1. **Wind Influence Precomputation**: Calculated once in `__init__()` rather than per-cell
2. **Sparse Change Dictionary**: Only returns modified cells (typically 5-15% of grid)
3. **Early Exit**: Skips probability calculation for already-burning cells
4. **Neighbor Caching**: 8-connectivity neighbor lists computed inline

### Scaling Guidelines

| Grid Size | Cells | Avg Time/Step | Memory |
|-----------|-------|---------------|--------|
| 30×30 | 900 | ~2ms | ~1 MB |
| 60×60 | 3,600 | ~8ms | ~5 MB |
| 100×100 | 10,000 | ~25ms | ~15 MB |
| 200×200 | 40,000 | ~120ms | ~60 MB |

**Recommendation**: For grids >100×100, consider reducing `fire_update_interval` or using realistic model.

---

## Differences from Realistic Model

| Aspect | Aggressive | Realistic | Ratio |
|--------|-----------|-----------|-------|
| **Time to flashover** | 30-60s | 3-6 min | **6-12× faster** |
| **Spread speed** | 0.3-0.5 m/s | 0.1-0.2 m/s | **2-3× faster** |
| **Ignition probability** | 40-60% | 15-30% | **2× higher** |
| **Burn duration** | 40+ steps | 20-30 steps | **1.3-2× longer** |
| **Wind speed default** | 1.5 m/s | 0.5 m/s | **3× stronger** |
| **Base spread multiplier** | 0.08 | 0.03 | **2.67× higher** |
| **Growth rate** | 0.2/step | 0.08/step | **2.5× faster** |
| **Oxygen consumption** | 0.15 | 0.05 | **3× faster** |
| **Heat production** | 12.0 | 5.0 | **2.4× higher** |

**Use Cases**:
- **Aggressive**: Algorithm stress-testing, worst-case analysis, competition scenarios
- **Realistic**: Evacuation planning, training, code compliance analysis

---

## Troubleshooting

### Issue: Fire spreads too quickly

**Solutions**:
1. Reduce `burn_rate_modifier` (default 1.5 → try 1.2)
2. Lower `wind_speed` (default 1.5 → try 1.0)
3. Increase `humidity` (default 50% → try 70%)
4. Increase `fuel_moisture` (default 10% → try 25%)
5. Switch to realistic model: `fire_model_type: "realistic"`

### Issue: Fire doesn't spread at all

**Check**:
1. Initial fire intensity > 0 (at least 1.0 for ignition)
2. Obstacles (-2) don't surround fire source
3. Sufficient oxygen (check `oxygen_map` values)
4. `fire_update_interval` not too large (should be ≤10)

### Issue: Out of memory errors

**Solutions**:
1. Reduce grid size in configuration
2. Decrease number of agents
3. Limit simulation duration
4. Disable environmental map tracking if not needed

### Issue: Changes dictionary empty

**Possible Causes**:
1. Fire already fully spread (all cells burned or blocked)
2. Fire extinguished (fuel depleted, no oxygen)
3. Change threshold too high (model only reports changes >0.01)
4. Current state contains only obstacles (-2, -1)

---

## API Compatibility

### Version Requirements

- **Python**: 3.7+ (uses dataclasses)
- **Dependencies**: `math`, `random`, `copy`, `typing` (all standard library)
- **Optional**: `scipy` (for MATLAB visualizer interpolation)

### Integration Checklist

When integrating aggressive fire model into your simulation:

- [ ] Import correct module: `from fire_model_aggressive import create_fire_model`
- [ ] Match grid dimensions: `model.rows == simulation.map_rows`
- [ ] Use correct coordinate format: `"x{col}y{row}"`
- [ ] Call `simulate_step()` with current fire state (not changes)
- [ ] Apply returned changes to your fire map
- [ ] Update agent graphs after fire updates
- [ ] Handle special values: -2 (obstacle), -1 (inaccessible)
- [ ] Check fire update interval appropriate for aggressive spread

---

## References

### Related Files

- [simulation.py](simulation.py) - Main simulation engine
- [fire_model_realistic.py](fire_model_realistic.py) - Realistic fire model
- [fire_monitor.py](fire_monitor.py) - Data collection and export
- [example_configuration.json](example_configuration.json) - Configuration format
- [CLAUDE.md](CLAUDE.md) - Project overview and architecture

### Key Functions in simulation.py

- `EvacuationSimulation.__init__()` - Line ~50: Fire model initialization
- `EvacuationSimulation.run()` - Line ~400-500: Fire update cycle
- `EvacuationSimulation.update_environment()` - Line ~350: Change propagation
- `EvacuationAgent.update_graph()` - Line ~200: D* Lite graph updates

### Configuration Parameters

```json
{
  "fire_model_type": "aggressive",
  "fire_update_interval": 4,
  "cell_size": 0.3,
  "timestep_duration": 0.5,
  "initial_fire_map": [[...]]
}
```

---

## Summary

The Aggressive Fire Model provides a worst-case fire spread simulation for stress-testing evacuation algorithms. Its API integrates seamlessly with the D* Lite pathfinding system, updating terrain costs dynamically as fire spreads.

**Key Takeaways**:
- Primary interface: `simulate_step(current_state)` → returns `{"x{col}y{row}": new_value}`
- Called every `fire_update_interval` timesteps by simulation
- 2-12× faster spread than realistic model for algorithm validation
- Tracks oxygen, temperature, smoke, and fuel for advanced analytics
- Fully compatible with existing simulation infrastructure

**For production evacuation planning, use [fire_model_realistic.py](fire_model_realistic.py) instead.**
