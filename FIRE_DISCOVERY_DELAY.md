# Fire Discovery Delay Feature

## Overview

The **fire discovery delay** feature allows you to simulate realistic fire detection time by running the fire spread model for a specified number of steps before agents start moving. This represents the time between when a fire starts and when occupants become aware of it.

## Motivation

In real-world scenarios, fires don't immediately trigger evacuation:
- Smoke detectors have detection delays (30 seconds to 2 minutes)
- Occupants may not immediately notice fire/smoke
- Alarm systems have notification delays
- Initial fire growth may be hidden (e.g., electrical fires inside walls)

By the time evacuation begins, the fire has already spread, creating more challenging and realistic scenarios.

## Configuration

Add `fire_discovery_delay` to your simulation configuration JSON:

```json
{
  "map_rows": 30,
  "map_cols": 40,
  "agent_num": 50,
  "fire_update_interval": 4,
  "fire_discovery_delay": 20,
  ...
}
```

**Parameter:**
- `fire_discovery_delay` (int, default: 0): Number of simulation steps where fire spreads but agents remain stationary
  - 0 = agents start moving immediately (no delay)
  - 20 = fire spreads for 20 steps before agents start evacuating
  - With default `timestep_duration=0.5s`, delay of 20 = 10 seconds of fire spread

## Implementation

The feature is implemented in **both** simulation variants through a unified configuration approach:

### 1. Original Simulation (`simulation.py`)

**Configuration:**
- Added `fire_discovery_delay: int = 0` to `SimulationConfig` (line 67)
- Loaded from JSON in `from_json()` method (line 98)
- Saved in `to_dict()` method (line 132)

**Runtime behavior (lines 1278-1298):**
```python
# Update fire model at specified interval (decoupled from agent movement)
if self.steps % self.config.fire_update_interval == 0:
    changes = self.update_fire()
    if changes:
        self.update_environment(changes)

# Only move agents after fire discovery delay has passed
if self.steps >= self.config.fire_discovery_delay:
    results = self.step()  # Normal agent movement
else:
    # Fire discovery delay: fire spreads but agents don't move
    results = []
```

### 2. Fast Simulation Phase 2 (`fast_simulation.py`)

**Configuration:**
- Added `fire_discovery_delay: int = 0` to `__init__()` parameter (line 65)
- Stored as instance variable (line 116)

**Runtime behavior (lines 169-214):**
```python
# Update fire periodically
if step > 0 and step % self.fire_update_interval == 0:
    old_grid = self.grid.copy()
    self.grid = self.fire.step()
    # ... update pathfinders ...

# Only move agents after fire discovery delay has passed
if step >= self.fire_discovery_delay:
    # Move agents (normal pathfinding)
    for i, agent in enumerate(self.agents):
        # ... movement logic ...
else:
    # Fire discovery delay: fire spreads but agents don't move
    # Still count active agents for early termination checks
    active_count = sum(1 for a in self.agents if a.status == 'active')
```

### 3. Monte Carlo Integration (`monte_carlo.py`)

Phase 2 simulations automatically pass `fire_discovery_delay` from config:

```python
sim = FastEvacuationSim(
    grid=fire_map,
    agent_starts=agent_starts,
    exits=exits,
    fire_starts=fire_starts,
    fire_update_interval=config_copy.fire_update_interval,
    fire_discovery_delay=config_copy.fire_discovery_delay,  # ← Passes through
    fire_spread_mode=fire_spread_mode
)
```

### 4. Phase 2 Visual Validation (`run_phase2_visual.py`)

Visual simulations also respect the delay from config:

```python
sim = FastEvacuationSim(
    grid=fire_map,
    agent_starts=agent_starts,
    exits=exits,
    fire_starts=fire_starts,
    fire_update_interval=config.fire_update_interval,
    fire_discovery_delay=config.fire_discovery_delay,  # ← Passes through
    fire_spread_mode=fire_spread_mode
)
```

## Usage Examples

### Example 1: No Delay (Immediate Response)

```json
{
  "fire_discovery_delay": 0,
  "fire_update_interval": 4,
  "timestep_duration": 0.5
}
```

- Step 0: Fire starts, agents begin moving
- Agents respond immediately (unrealistic but useful for baseline comparisons)

### Example 2: 10-Second Detection Delay

```json
{
  "fire_discovery_delay": 20,
  "fire_update_interval": 4,
  "timestep_duration": 0.5
}
```

- Steps 0-19: Fire spreads, agents stationary (10 seconds)
- Step 20: Agents start evacuating
- Realistic for smoke detector detection time

### Example 3: 2-Minute Hidden Fire

```json
{
  "fire_discovery_delay": 240,
  "fire_update_interval": 4,
  "timestep_duration": 0.5
}
```

- Steps 0-239: Fire grows undetected (2 minutes)
- Step 240: Evacuation begins
- Extreme scenario (e.g., electrical fire inside walls)

### Example 4: Realistic Commercial Building

```json
{
  "fire_discovery_delay": 60,
  "fire_update_interval": 4,
  "timestep_duration": 0.5,
  "fire_model_type": "realistic"
}
```

- 30-second detection delay (60 steps × 0.5s)
- Physics-aligned fire spread
- Realistic smoke detector + alarm notification time

## Interaction with Other Features

### Fire Update Interval

Fire still updates at `fire_update_interval` during the delay:

```json
{
  "fire_discovery_delay": 40,
  "fire_update_interval": 4
}
```

- Fire updates at steps: 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, ...
- Agents start moving at step 40
- Fire has spread for 10 update cycles (20 seconds) before evacuation

### Fire Spread Modes (Phase 2 only)

Works with all three fire spread modes:

```json
{
  "fire_discovery_delay": 20,
  "fire_spread_mode": "always_real"
}
```

- `"always_real"`: Stochastic spread throughout simulation
- `"real_then_simple"`: Stochastic spread until stable, then intensity growth only
- `"real_then_stop"`: Stochastic spread until stable, then static

Fire spreads according to the mode during the delay period.

### Visualization

Both pygame (run_phase2_visual.py) and MATLAB (run_with_matlab_viz.py) visualizations show:
- Fire spreading during delay period
- Agents standing still
- Step counter indicates when evacuation begins

## Design Philosophy

### Single Configuration Point

All simulation variants use `SimulationConfig.fire_discovery_delay`:
- Original simulation reads it from the config dataclass
- Fast simulation receives it as a constructor parameter
- Monte Carlo simulations pass it through from the loaded config
- No code duplication: each implementation has its own conditional check

### Decoupled Fire and Agent Updates

Fire updates and agent movement are independent:
- Fire can update without agent movement (during delay)
- Agents can move without fire updates (between `fire_update_interval` steps)
- This separation makes the delay implementation trivial

### Zero-Cost When Disabled

With default `fire_discovery_delay=0`:
- Conditional check: `if step >= 0` is always true
- No performance overhead
- Backward compatible with all existing configurations

## Testing

### Quick Test: Visual Validation

```bash
# Create test config with 10-second delay
cat > test_delay.json << EOF
{
  "map_rows": 20,
  "map_cols": 30,
  "agent_num": 10,
  "start_positions": ["x2y2", "x3y2", "x4y2"],
  "targets": ["x27y17"],
  "initial_fire_map": [[0, ...], ...],  # Add fire at x15y10
  "fire_discovery_delay": 20,
  "fire_update_interval": 4,
  "timestep_duration": 0.5
}
EOF

# Run with visualization
python run_phase2_visual.py --config test_delay.json
```

**Expected behavior:**
- Steps 0-19: Fire spreads, agents don't move
- Step 20: Agents start pathfinding and moving
- Visual confirmation that delay works

### Monte Carlo Test

```bash
# Test with parallel Monte Carlo runs
python monte_carlo.py --config test_delay.json --runs 100 --parallel --phase2
```

**Metrics to check:**
- Higher `average_fire_damage` with larger delays
- Lower `survival_rate` with larger delays
- Longer `avg_evacuation_time` (excludes delay period, measures evacuation after discovery)

## Performance Considerations

### Computational Cost

The delay adds minimal overhead:
- One integer comparison per step: `if step >= fire_discovery_delay`
- Fire updates continue as normal (no extra computation)
- Pathfinding updates still happen during delay (agents build/update graphs even if not moving)

### Memory Usage

No additional memory required:
- One integer stored per simulation instance
- Agent states remain unchanged during delay
- Fire grid continues updating in place

### Early Termination

Early termination checks still work during delay:
- Agents can die during delay if fire reaches their position
- `death_threshold` can trigger early termination even before agents move
- This is realistic: agents trapped at fire origin may die before they can react

## Common Patterns

### Sensitivity Analysis

Test how detection delay affects evacuation success:

```python
for delay in [0, 10, 20, 40, 60, 120]:
    config['fire_discovery_delay'] = delay
    # Run Monte Carlo simulations
    # Compare survival rates, evacuation times
```

### Detector Placement Optimization

Combine with fire placement randomization:

```python
# Vary fire start location AND detection delay
for fire_location in room_centers:
    for delay in detector_delays:
        # Run simulation
        # Find optimal detector placement to minimize average delay
```

### Worst-Case Scenario Testing

```python
config = {
    "fire_discovery_delay": 120,  # 1 minute delay
    "fire_model_type": "aggressive",  # Fast spread
    "fire_spread_mode": "always_real"  # Stochastic
}
# Stress test: fire spreads aggressively before evacuation starts
```

## Limitations

### Not Per-Agent

All agents discover fire simultaneously:
- Realistic for alarm system activation
- Less realistic for gradual awareness spread
- Future enhancement: per-agent awareness radius

### No Partial Awareness

Binary state: all agents unaware → all agents aware at step N:
- Doesn't model occupants near fire discovering it first
- Doesn't model knowledge spreading between agents
- Could be combined with communication_range for hybrid model

### Fixed Discovery Time

Discovery happens at fixed step regardless of fire size:
- Realistic for smoke detectors (detect smoke, not fire size)
- Less realistic for visual discovery (larger fires more noticeable)
- Could be enhanced with intensity-based triggers

## Summary

**What changed:**
- Added `fire_discovery_delay` to `SimulationConfig` dataclass
- Modified main simulation loops in both variants to skip agent movement during delay
- Updated Monte Carlo and visual wrappers to pass delay parameter

**What stayed the same:**
- Fire update logic (unchanged)
- Pathfinding algorithms (unchanged)
- All other configuration parameters (unchanged)
- No performance degradation when delay=0

**Single control point:**
- Set `fire_discovery_delay` in JSON config
- Works across all simulation variants
- No need to modify multiple files or classes
