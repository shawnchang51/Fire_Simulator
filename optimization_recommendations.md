# Optimization Recommendations for 200-Agent Simulation

## Profiling Results Summary (5 agents baseline)
- **Total time**: 3.8 seconds
- **CPU bottleneck**: D* Lite pathfinding (54% of time)
- **Memory peak**: 32.76 MB
- **Memory bottleneck**: Fire monitor history (~3.2 MB)

## Expected Impact at 200 Agents (40x scale)
- **Estimated runtime**: 2-5 minutes (without optimizations)
- **Estimated memory**: 200-400 MB
- **Main concerns**:
  - Environment updates broadcast to all 200 agents
  - Each agent runs independent pathfinding
  - Fire monitor history grows with timesteps

---

## Critical Configuration Changes

### 1. **Reduce Viewing Range** (High Impact)
```json
"viewing_range": 5  // Down from 10
```
**Why**: `scanForObstacles()` takes 0.144s with viewing_range=10. This is O(viewing_range²) per agent.
- Reduces obstacle scanning from ~100 cells to ~25 cells per scan
- With 200 agents, this saves significant time
- Trade-off: Agents discover fire/obstacles later

### 2. **Increase Fire Update Interval** (Medium Impact)
```json
"fire_update_interval": 4  // Up from 2 (now 2 seconds instead of 1 second)
```
**Why**: Each fire update triggers environment updates to ALL agents
- Reduces number of `update_environment()` calls by 50%
- Fire spread is still realistic (2 seconds is reasonable)
- Saves ~1 second per 40 timesteps

### 3. **Reduce Knowledge Sharing Frequency** (Medium Impact)
```json
"communication_range": 10.0,  // Down from 15.0
"sharing_interval": 10,       // Up from 5 (now every 5 seconds)
"sector_size": 10             // Match communication_range
```
**Why**: Knowledge sharing scales poorly with agent count
- Reduces graph merging operations by 50%
- Smaller communication range = fewer agents to check
- Trade-off: Slower information propagation

### 4. **Disable Environmental Factors** (Low-Medium Impact)
```json
"consider_env_factors": false
```
**Why**: Calculating temperature/smoke costs for pathfinding adds overhead
- Simplifies terrain cost calculations in `getTerrainCost()`
- Still realistic for basic evacuation modeling
- Re-enable if environmental realism is critical

### 5. **Increase Max Occupancy** (Low Impact, Prevents Gridlock)
```json
"max_occupancy": 2  // Up from 1
```
**Why**: With 200 agents, bottlenecks will form at doors/exits
- Allows some overlap to prevent stuck agents
- More realistic for crowded evacuations

---

## Code-Level Optimizations

### Priority 1: Optimize Environment Updates (Highest Impact)

**Current bottleneck**: `update_environment()` loops through ALL agents
```python
# simulation.py:944 - Current implementation
for agent in self.agents:
    if agent.status == 'active':
        agent.update_graph(changes)  # O(agents × changes)
```

**Optimization**: Only update agents near fire changes
```python
# Add spatial filtering
def update_environment(self, changes):
    """Update agent graphs with fire changes (spatially filtered)"""
    if not changes:
        return

    # Get bounding box of changes
    change_coords = [stateNameToCoords(pos) for pos in changes.keys()]
    min_x = min(c[0] for c in change_coords)
    max_x = max(c[0] for c in change_coords)
    min_y = min(c[1] for c in change_coords)
    max_y = max(c[1] for c in change_coords)

    # Add buffer for agent viewing range
    buffer = 15  # viewing_range + communication_range

    # Only update agents within or near the change region
    for agent in self.agents:
        if agent.status != 'active':
            continue

        agent_x, agent_y = stateNameToCoords(agent.current_position)

        # Check if agent is near changes
        if (min_x - buffer <= agent_x <= max_x + buffer and
            min_y - buffer <= agent_y <= max_y + buffer):
            agent.update_graph(changes)
```

**Expected speedup**: 5-10x for sparse fire spread

---

### Priority 2: Limit Fire Monitor History (Medium Memory Impact)

**Current issue**: Fire monitor stores full history unbounded
```python
# fire_monitor.py:71-76 - stores every timestep
self.oxygen_history.append(copy.deepcopy(self.fire_model.env_params.oxygen_map))
```

**Optimization**: Add history limit
```python
def monitor_step(self, changes):
    """Monitor a single simulation step with history limit"""
    # ... existing code ...

    # Add to history with limit
    MAX_HISTORY = 500  # Keep last 500 timesteps only

    if len(self.fire_history) >= MAX_HISTORY:
        self.fire_history.pop(0)
        self.oxygen_history.pop(0)
        self.temperature_history.pop(0)
        self.smoke_history.pop(0)
        self.fuel_history.pop(0)

    self.fire_history.append(copy.deepcopy(current_fire))
    # ... rest of appends ...
```

**Expected savings**: Memory stays bounded instead of growing linearly

---

### Priority 3: Cache Coordinate Conversions (Low Impact, Easy Win)

**Current overhead**: 2.24M string splits for coordinate parsing
```python
# utils.py:3 - called constantly
def stateNameToCoords(name):
    s = name.split('x')
    s = s[1].split('y')
    return int(s[0]), int(s[1])
```

**Optimization**: Add memoization
```python
from functools import lru_cache

@lru_cache(maxsize=10000)  # Cache up to 10k coordinate conversions
def stateNameToCoords(name):
    s = name.split('x')
    s = s[1].split('y')
    return int(s[0]), int(s[1])
```

**Expected speedup**: ~0.2 seconds saved

---

### Priority 4: Reduce D* Lite Update Frequency (Medium Impact)

**Current behavior**: D* Lite updates on every environment change
```python
# simulation.py:564 - called 61 times for 5 agents
def update_graph(self, changes):
    for position, value in changes.items():
        # ... update terrain ...
        self.graph.updateVertex(self.d_star_lite, u)  # Triggers replan
```

**Optimization**: Batch updates and only replan when significant
```python
def update_graph(self, changes, force_replan=False):
    """Update graph with changes, batch vertex updates"""
    updated_vertices = []

    for position, value in changes.items():
        # ... update terrain ...
        updated_vertices.append(u)

    # Only replan if changes affect current path or are significant
    if force_replan or self._changes_affect_path(updated_vertices):
        for u in updated_vertices:
            self.graph.updateVertex(self.d_star_lite, u)
```

---

## Profiling Commands for 200 Agents

Test with optimized configuration:
```bash
# Create test with 200 agents
python -c "
import json
from monte_carlo import replace_agents, replace_fire

with open('config_200_agents_optimized.json', 'r') as f:
    config = json.load(f)

# Generate random agent positions
config = replace_agents(config, 200)
config = replace_fire(config, 3)  # 3 fire sources

with open('test_200_agents.json', 'w') as f:
    json.dump(config, f, indent=2)
"

# Profile it
python profile_simulation.py  # Update to use test_200_agents.json
```

---

## Expected Performance After Optimizations

| Metric | Before | After Optimization | Improvement |
|--------|--------|-------------------|-------------|
| Runtime (estimated) | 150-200s | 30-60s | 3-5x faster |
| Peak Memory | 400+ MB | 150-200 MB | 2x less |
| Environment Update | O(n) every change | O(k) sparse | 5-10x faster |
| Fire Monitor Memory | Unbounded | Capped at 500 steps | Bounded |

---

## Testing Strategy

1. **Start small**: Test with 50 agents first
2. **Profile again**: Use `profile_simulation.py` to verify improvements
3. **Scale gradually**: 50 → 100 → 200 agents
4. **Monitor memory**: Watch for memory leaks with `tracemalloc`
5. **Disable visualization**: Set `use_pygame=False, use_matlab=False` for speed

---

## Quick Wins Checklist

- [ ] Reduce `viewing_range` to 5
- [ ] Increase `fire_update_interval` to 4
- [ ] Set `consider_env_factors: false`
- [ ] Add `@lru_cache` to `stateNameToCoords()`
- [ ] Limit fire monitor history to 500 timesteps
- [ ] Add spatial filtering to `update_environment()`
- [ ] Test with 50 agents first
- [ ] Profile to verify improvements

---

## When to Use Parallel Execution

For 200 agents × multiple runs:
```bash
# Monte Carlo with parallelization
python monte_carlo.py --runs 100 --parallel --processes 8
```

**Note**: Each parallel process will use ~150-200 MB memory, so limit processes based on available RAM.
