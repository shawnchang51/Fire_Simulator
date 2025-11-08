# Fire Simulator Profiling Report

**Test Configuration:**
- Map size: 60x60 (3600 cells)
- Agents: 5
- Fire model: aggressive
- Steps simulated: 200
- Peak memory: 90.48 MB
- Total time: ~68 seconds

---

## 🔥 TOP TIME BOTTLENECKS

### 1. **D* Lite Pathfinding - 19.5s (28.4%)**
- **Function:** `computeShortestPath` (d_star_lite.py)
- **Calls:** 2,615 times
- **Average:** 7.5ms per call
- **Issue:** Recomputing shortest paths frequently as environment changes

### 2. **Cost Calculations - 13.3s (19.4%)**
- **Function:** `GridWorld.cost` (grid.py:102)
- **Calls:** 226,668 times
- **Average:** 0.06ms per call
- **Issue:** Called excessively during pathfinding operations

### 3. **Fire Model Updates - 13.1s (19.1%)**
- **Function:** `update_fire` (fire_model_aggressive.py)
- **Calls:** 82 times
- **Average:** 160ms per call
- **Issue:** Nested loops over entire 60x60 grid, heat transfer calculations

### 4. **Obstacle Scanning - 10.9s (15.9%)**
- **Function:** `scanForObstacles` (d_star_lite.py)
- **Calls:** 524 times
- **Average:** 20.8ms per call
- **Issue:** Scanning viewing range for each agent movement

### 5. **Simulation Steps - 9.5s (13.9%)**
- **Function:** `step` (simulation.py)
- **Calls:** 200 times
- **Average:** 47.5ms per call
- **Note:** This is the main loop orchestrating all operations

---

## 💾 TOP MEMORY BOTTLENECKS

### 1. **Fire Model Heat Transfer - 6.1 MB**
- **Location:** `fire_model_aggressive.py:244`
- **Allocations:** 261,467 objects
- **Issue:** Creating temporary objects during heat transfer calculations
```python
self.temperature_map[ni][nj] += heat_transfer
```

### 2. **Fire Model Oxygen/Smoke - 10.4 MB**
- **Locations:**
  - `fire_model_aggressive.py:251` (5.2 MB) - Oxygen replenishment
  - `fire_model_aggressive.py:257` (5.2 MB) - Smoke dissipation
- **Allocations:** 220,910 objects each
- **Issue:** Similar to heat transfer - many small object allocations

### 3. **Fire Monitor History - 15.8 MB**
- **Location:** `fire_monitor.py:71-76`
- **Allocations:** ~60,000 objects (5 maps × 12,078 snapshots)
- **Issue:** Storing complete snapshots of ALL environmental maps every timestep:
  - Oxygen map (3600 cells)
  - Temperature map (3600 cells)
  - Smoke density map (3600 cells)
  - Fuel map (3600 cells)
  - Fire state map (3600 cells)
```python
oxygen_snapshot = [row[:] for row in self.model.oxygen_map]
temp_snapshot = [row[:] for row in self.model.temperature_map]
smoke_snapshot = [row[:] for row in self.model.smoke_density]
fuel_snapshot = [row[:] for row in self.model.fuel_map]
self.history['fire_states'].append([row[:] for row in fire_state])
```

### 4. **D* Lite Grid Graph - 3.0 MB**
- **Location:** `d_star_lite/grid.py:102-104`
- **Allocations:** 28,084 objects
- **Issue:** Building neighbor relationships and edge costs for entire grid

---

## 📊 Performance Breakdown

| Component | Time | % | Memory | % |
|-----------|------|---|--------|---|
| D* Lite Pathfinding | 19.5s | 28.4% | 3.0 MB | 3.3% |
| Fire Model Updates | 13.1s | 19.1% | 16.5 MB | 18.2% |
| Fire Monitor | - | - | 15.8 MB | 17.5% |
| Obstacle Scanning | 10.9s | 15.9% | - | - |
| Cost Calculations | 13.3s | 19.4% | - | - |
| Other | 18.7s | 27.2% | 55.2 MB | 61.0% |

---

## 🎯 OPTIMIZATION RECOMMENDATIONS

### High Priority (Biggest Impact)

#### 1. **Fire Monitor Lightweight Mode**
- **Current:** Stores 5 complete maps every timestep = 18K cells/step
- **Memory saved:** ~15.8 MB (17.5%)
- **Solution:** Already implemented! Use `lightweight_mode=True`
- **Impact:** ⭐⭐⭐⭐⭐

#### 2. **Reduce D* Lite Recalculations**
- **Current:** 2,615 calls to `computeShortestPath` in 200 steps
- **Time saved:** Potential 30-40%
- **Solutions:**
  - Cache paths when environment hasn't changed
  - Use larger tolerance for path invalidation
  - Only replan when fire is within X cells of current path
- **Impact:** ⭐⭐⭐⭐⭐

#### 3. **Optimize Fire Spread Calculations**
- **Current:** 160ms per update, creates 260K+ objects
- **Time saved:** 5-10s (7-15%)
- **Solutions:**
  - Use numpy arrays instead of list comprehensions
  - Pre-allocate neighbor arrays
  - Update only cells with active fire (sparse updates)
- **Impact:** ⭐⭐⭐⭐

### Medium Priority

#### 4. **Grid Cost Calculation Caching**
- **Current:** 226,668 calls in 200 steps
- **Time saved:** 3-5s (5-7%)
- **Solutions:**
  - Cache cost calculations for unchanged cells
  - Use dirty flagging for modified cells
- **Impact:** ⭐⭐⭐

#### 5. **Spatial Indexing for Obstacle Scanning**
- **Current:** 524 scans × 20ms = 10.9s
- **Time saved:** 3-5s (5-7%)
- **Solutions:**
  - Use quadtree or grid-based spatial indexing
  - Only scan when moving to new grid sectors
- **Impact:** ⭐⭐⭐

### Low Priority

#### 6. **Use Numba JIT Compilation**
- Add `@numba.jit` to hot loops in fire model
- Potential 2-3x speedup for fire calculations
- **Impact:** ⭐⭐

---

## 🚀 Quick Wins

### Immediate Actions (No Code Changes)

1. **Enable lightweight mode** in fire monitor:
   ```python
   fire_monitor = FireMonitor(fire_model, lightweight_mode=True)
   ```
   **Saves:** 15.8 MB memory (17.5%)

2. **Reduce fire_update_interval**:
   ```json
   "fire_update_interval": 8
   ```
   **Saves:** ~50% of fire update time (9-10s)

3. **Disable visualization** for batch runs:
   ```python
   sim.run(show_visualization=False, use_pygame=False)
   ```
   **Saves:** 1-2s overhead

---

## 📈 Scaling Considerations

### Memory Growth with Map Size

For a 60×60 map (3600 cells):
- Fire monitor: 15.8 MB
- Estimated for 100×100: **44 MB**
- Estimated for 200×200: **176 MB**

### Time Complexity

- **D* Lite:** O(N log N) where N = cells to update
- **Fire Model:** O(W × H) every update interval
- **Obstacle Scan:** O(V²) where V = viewing_range

### Recommendations for Large Maps

- **For maps > 100×100:** Enable lightweight_mode
- **For maps > 200×200:** Consider chunking/streaming fire updates
- **For many agents (>50):** Implement agent-level spatial indexing

---

## 🧪 Test Scenarios

| Scenario | Map Size | Agents | Memory | Time (200 steps) |
|----------|----------|--------|--------|------------------|
| Small | 30×30 | 3 | ~25 MB | ~15s |
| Current | 60×60 | 5 | ~90 MB | ~68s |
| Medium | 100×100 | 10 | ~250 MB | ~180s (est) |
| Large | 200×200 | 20 | ~1 GB | ~600s (est) |

---

## 📝 Profiling Commands

To reproduce this analysis:
```bash
python profile_simulator.py
```

To profile with different configurations:
```python
from simulation import SimulationConfig
import json

with open('your_config.json') as f:
    config = SimulationConfig.from_json(json.load(f))

# Run profiling
python profile_simulator.py
```

---

**Report Generated:** 2025-11-08
**Profiling Tool:** cProfile + tracemalloc
**Python Version:** 3.11.14
