# RL Training Optimization Roadmap

Performance optimization guide for using Monte Carlo simulations to train reinforcement learning agents for floor plan design.

## Executive Summary

**Goal**: Make simulations fast enough for RL training (100K-10M evaluations)

| Approach | Effort | Speedup | Per-Sim Time |
|----------|--------|---------|--------------|
| Current baseline | - | 1x | ~2.0s |
| Conservative (config tuning) | 1-2 days | 3-5x | ~0.4-0.6s |
| Moderate (architectural) | 1-2 weeks | 10-50x | ~0.04-0.2s |
| Aggressive (ML acceleration) | 1-2 months | 100-1000x | ~0.002-0.02s |

---

## Current Bottlenecks

From profiling analysis (`PERFORMANCE_RESULTS.md`):

| Component | Time % | Scaling Issue | Impact on RL |
|-----------|--------|---------------|--------------|
| D* Lite pathfinding | 54% | O(agents × grid × replans) | Critical |
| Environment updates | 20% | O(agents × fire_changes) | High |
| Fire simulation | 15% | O(grid_size) per update | Medium |
| Coordinate parsing | 5% | O(calls) - now cached | Low |
| I/O and visualization | 6% | O(data_size) | Eliminable |

### Why Current Design is Slow for RL

1. **D* Lite over-engineering**: Incremental replanning is great for single runs but wasteful when running millions of independent simulations
2. **String-based coordinates**: `"x12y9"` format adds overhead vs integer tuples
3. **Per-agent graph copies**: Each agent maintains full GridWorld (~1.8MB each)
4. **Full physics fire model**: Tracks oxygen, temperature, smoke, fuel - unnecessary for training
5. **Visualization overhead**: Even disabled, setup/teardown costs exist

---

## Phase 1: Conservative Optimizations (1-2 days, 3-5x speedup)

Configuration and minor code changes only. No architectural rewrites.

### 1.1 Training-Optimized Configuration

Create `configs/rl_training_config.json`:

```json
{
  "map_rows": 30,
  "map_cols": 30,
  "cell_size": 0.6,
  "timestep_duration": 0.5,
  "fire_update_interval": 8,
  "fire_model_type": "aggressive",
  "viewing_range": 2,
  "max_occupancy": 4,
  "agent_num": 5,
  "consider_env_factors": false,
  "communication_range": 0,
  "sharing_interval": 999999,
  "door_configs": []
}
```

**Rationale**:
- **Coarser grid (30×30 vs 60×60)**: 4x fewer cells = 4x faster pathfinding
- **Larger cells (0.6m vs 0.3m)**: Same physical space, fewer computations
- **Aggressive fire model**: Faster spread = shorter simulations
- **Reduced viewing_range (2 vs 5)**: 84% fewer cells scanned per agent
- **Disabled knowledge sharing**: Eliminates O(n²) agent interactions
- **No door graph**: Simpler pathfinding during training

### 1.2 Early Termination

Add to `simulation.py` in `EvacuationSimulation.run()`:

```python
def run(self, max_steps=1000, early_termination=True,
        stuck_threshold=0.5, min_active_agents=0):
    """
    Run simulation with optional early termination for RL training.

    Args:
        max_steps: Maximum simulation steps
        early_termination: Enable early stopping for bad designs
        stuck_threshold: Terminate if this fraction of agents stuck
        min_active_agents: Terminate if fewer agents active
    """
    for step in range(max_steps):
        # ... existing step logic ...

        if early_termination:
            active = sum(1 for a in self.agents if a.status == 'active')
            stuck = sum(1 for a in self.agents if a.status == 'stuck')
            dead = sum(1 for a in self.agents if a.status in ('trapped', 'dead'))

            # All agents resolved - no need to continue
            if active == 0:
                return self._compute_metrics(step, 'all_resolved')

            # Most agents stuck - bad floor plan design
            if stuck > len(self.agents) * stuck_threshold:
                return self._compute_metrics(step, 'mostly_stuck')

            # High casualty rate - bad design
            if dead > len(self.agents) * 0.3:
                return self._compute_metrics(step, 'high_casualties')

    return self._compute_metrics(max_steps, 'max_steps_reached')

def _compute_metrics(self, steps, termination_reason):
    """Compute RL reward metrics."""
    evacuated = sum(1 for a in self.agents if a.status == 'evacuated')
    stuck = sum(1 for a in self.agents if a.status == 'stuck')
    dead = sum(1 for a in self.agents if a.status in ('trapped', 'dead'))

    return {
        'steps': steps,
        'termination_reason': termination_reason,
        'evacuated': evacuated,
        'stuck': stuck,
        'dead': dead,
        'survival_rate': evacuated / len(self.agents),
        'avg_evacuation_time': steps,  # Simplified
        'reward': self._compute_reward(evacuated, stuck, dead, steps)
    }

def _compute_reward(self, evacuated, stuck, dead, steps):
    """Compute RL reward signal."""
    # Reward structure for floor plan design
    evacuation_bonus = evacuated * 10.0
    stuck_penalty = stuck * -5.0
    death_penalty = dead * -20.0
    time_penalty = steps * -0.01  # Encourage faster evacuation

    return evacuation_bonus + stuck_penalty + death_penalty + time_penalty
```

### 1.3 Disable All I/O

Create `rl_simulation.py` wrapper:

```python
"""Lightweight simulation wrapper for RL training."""

from simulation import EvacuationSimulation, SimulationConfig
import numpy as np

class RLSimulationWrapper:
    """Zero-overhead simulation for RL training."""

    def __init__(self, base_config_path='configs/rl_training_config.json'):
        import json
        with open(base_config_path) as f:
            self.base_config = json.load(f)

    def evaluate(self, floor_plan: np.ndarray,
                 agent_positions: list,
                 exit_positions: list,
                 fire_positions: list = None,
                 max_steps: int = 200) -> dict:
        """
        Evaluate a floor plan design.

        Args:
            floor_plan: 2D numpy array (-2=wall, 0=empty)
            agent_positions: List of (x, y) tuples for agent starts
            exit_positions: List of (x, y) tuples for exits
            fire_positions: Optional list of (x, y) tuples for fire starts
            max_steps: Maximum simulation steps

        Returns:
            Dict with evacuation metrics and reward
        """
        config = self._build_config(floor_plan, agent_positions,
                                    exit_positions, fire_positions)

        sim = EvacuationSimulation(config)

        # Run with all overhead disabled
        result = sim.run(
            max_steps=max_steps,
            show_visualization=False,
            use_pygame=False,
            use_matlab=False,
            early_termination=True
        )

        return result

    def _build_config(self, floor_plan, agents, exits, fires):
        """Build SimulationConfig from numpy arrays."""
        config_dict = self.base_config.copy()

        rows, cols = floor_plan.shape
        config_dict['map_rows'] = rows
        config_dict['map_cols'] = cols
        config_dict['agent_num'] = len(agents)
        config_dict['start_positions'] = [f'x{x}y{y}' for x, y in agents]
        config_dict['targets'] = [f'x{x}y{y}' for x, y in exits]

        # Convert numpy to list and add fires
        fire_map = floor_plan.tolist()
        if fires:
            for x, y in fires:
                if 0 <= y < rows and 0 <= x < cols:
                    fire_map[y][x] = 2.0
        config_dict['initial_fire_map'] = fire_map

        return SimulationConfig.from_json(config_dict)

    def batch_evaluate(self, floor_plans: list, num_workers: int = None) -> list:
        """Evaluate multiple floor plans in parallel."""
        from multiprocessing import Pool, cpu_count

        if num_workers is None:
            num_workers = cpu_count()

        with Pool(num_workers) as pool:
            results = pool.map(self._evaluate_single, floor_plans)

        return results

    def _evaluate_single(self, args):
        """Single evaluation for multiprocessing."""
        floor_plan, agents, exits, fires = args
        return self.evaluate(floor_plan, agents, exits, fires)
```

### 1.4 NumPy Fire Map

Modify `SimulationConfig.from_json()` to use numpy directly:

```python
@classmethod
def from_json(cls, json_data):
    # ... existing code ...

    # Convert fire map to numpy immediately
    fire_map = json_data.get('initial_fire_map',
                             [[0] * json_data['map_cols']] * json_data['map_rows'])
    if not isinstance(fire_map, np.ndarray):
        fire_map = np.array(fire_map, dtype=np.float32)

    config = cls(
        # ... other params ...
        initial_fire_map=fire_map,
    )
    return config
```

### Expected Results - Phase 1

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Time per simulation | 2.0s | 0.4-0.6s | 3-5x faster |
| Memory per simulation | 50MB | 30MB | 40% less |
| Simulations per hour | 1,800 | 6,000-9,000 | 3-5x more |

---

## Phase 2: Moderate Optimizations (1-2 weeks, 10-50x speedup)

Architectural improvements while staying in Python.

### 2.1 Fast A* Pathfinder (No Replanning)

Create `fast_pathfinder.py`:

```python
"""
Fast A* Pathfinder for RL Training
==================================

Replaces D* Lite with simple A* for training scenarios.
D* Lite's incremental replanning is unnecessary when:
- Running millions of independent simulations
- Fire positions are known at start (for RL reward computation)
- Approximate paths are acceptable during training

Performance: 10-20x faster than D* Lite for single queries
"""

import heapq
import numpy as np
from typing import List, Tuple, Optional, Set
from functools import lru_cache

class FastPathfinder:
    """
    Simple A* pathfinder optimized for RL training.

    Features:
    - No incremental replanning (faster for independent sims)
    - Integer coordinates (no string parsing)
    - Shared grid reference (no per-agent copies)
    - Optional path caching for repeated queries
    """

    # 8-directional movement
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]
    COSTS = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]

    def __init__(self, grid: np.ndarray, obstacle_value: float = -2):
        """
        Initialize pathfinder with grid.

        Args:
            grid: 2D numpy array where obstacle_value indicates walls
            obstacle_value: Value indicating impassable cells
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.obstacle_value = obstacle_value

        # Precompute obstacle mask for fast checking
        self.obstacles = (grid == obstacle_value)

    def update_grid(self, grid: np.ndarray):
        """Update grid (e.g., when fire spreads)."""
        self.grid = grid
        self.obstacles = (grid == self.obstacle_value) | (grid > 0)  # Include fire

    def find_path(self, start: Tuple[int, int],
                  goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path using A*.

        Args:
            start: (x, y) starting position
            goal: (x, y) goal position

        Returns:
            List of (x, y) positions from start to goal, or None if no path
        """
        if self._is_blocked(goal[0], goal[1]):
            return None

        if start == goal:
            return [start]

        # Priority queue: (f_score, counter, x, y)
        # Counter breaks ties to ensure FIFO behavior
        counter = 0
        open_set = [(self._heuristic(start, goal), counter, start[0], start[1])]

        came_from = {}
        g_score = {start: 0.0}

        while open_set:
            _, _, cx, cy = heapq.heappop(open_set)
            current = (cx, cy)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            for i, (dx, dy) in enumerate(self.DIRS):
                nx, ny = cx + dx, cy + dy

                if not self._is_valid(nx, ny):
                    continue

                # Calculate cost (higher for fire cells)
                move_cost = self.COSTS[i]
                cell_value = self.grid[ny, nx]
                if cell_value > 0:  # Fire
                    move_cost *= (1 + cell_value * 2)  # Fire avoidance

                tentative_g = g_score[current] + move_cost
                neighbor = (nx, ny)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, nx, ny))
                    came_from[neighbor] = current

        return None  # No path found

    def find_nearest_goal(self, start: Tuple[int, int],
                          goals: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find the nearest reachable goal from start."""
        best_goal = None
        best_distance = float('inf')

        for goal in goals:
            path = self.find_path(start, goal)
            if path and len(path) < best_distance:
                best_distance = len(path)
                best_goal = goal

        return best_goal

    def _is_valid(self, x: int, y: int) -> bool:
        """Check if position is valid and not blocked."""
        return (0 <= x < self.cols and
                0 <= y < self.rows and
                not self.obstacles[y, x])

    def _is_blocked(self, x: int, y: int) -> bool:
        """Check if position is blocked."""
        if not (0 <= x < self.cols and 0 <= y < self.rows):
            return True
        return self.obstacles[y, x]

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Chebyshev distance (allows diagonal movement)."""
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _reconstruct_path(self, came_from: dict,
                          current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dict."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


class BatchPathfinder:
    """
    Batch pathfinding for multiple agents.

    Optimizations:
    - Shared grid reference
    - Parallel path computation
    - Result caching
    """

    def __init__(self, grid: np.ndarray):
        self.pathfinder = FastPathfinder(grid)
        self._path_cache = {}

    def find_all_paths(self, agents: List[Tuple[int, int]],
                       goals: List[Tuple[int, int]]) -> List[Optional[List]]:
        """Find paths for all agents to nearest goals."""
        results = []
        for agent in agents:
            cache_key = (agent, tuple(goals))
            if cache_key in self._path_cache:
                results.append(self._path_cache[cache_key])
            else:
                nearest = self.pathfinder.find_nearest_goal(agent, goals)
                if nearest:
                    path = self.pathfinder.find_path(agent, nearest)
                else:
                    path = None
                self._path_cache[cache_key] = path
                results.append(path)
        return results

    def clear_cache(self):
        """Clear path cache (call when grid changes)."""
        self._path_cache.clear()

    def update_grid(self, grid: np.ndarray):
        """Update grid and clear cache."""
        self.pathfinder.update_grid(grid)
        self.clear_cache()
```

### 2.2 Vectorized Fire Model

Create `fast_fire.py`:

```python
"""
Vectorized Fire Model for RL Training
=====================================

Simplified cellular automata fire spread using NumPy operations.
Removes complex physics (oxygen, temperature, smoke) for speed.

Performance: 5-10x faster than AdvancedFireModel
"""

import numpy as np
from typing import Tuple, Optional

class FastFireModel:
    """
    Vectorized fire spread using NumPy convolution.

    Simplifications vs AdvancedFireModel:
    - No oxygen/temperature/smoke tracking
    - No fuel depletion
    - Fixed spread probabilities
    - No wind effects (can be added if needed)
    """

    # Spread kernel - probability of igniting neighbors
    SPREAD_KERNEL = np.array([
        [0.05, 0.15, 0.05],
        [0.15, 0.00, 0.15],
        [0.05, 0.15, 0.05]
    ], dtype=np.float32)

    def __init__(self, grid: np.ndarray,
                 spread_rate: float = 0.3,
                 intensity_growth: float = 0.5,
                 max_intensity: float = 4.0):
        """
        Initialize fire model.

        Args:
            grid: 2D array with initial fire state (-2=wall, 0=empty, >0=fire)
            spread_rate: Probability multiplier for fire spread
            intensity_growth: How fast fire intensity grows per step
            max_intensity: Maximum fire intensity
        """
        self.grid = grid.astype(np.float32)
        self.rows, self.cols = grid.shape
        self.spread_rate = spread_rate
        self.intensity_growth = intensity_growth
        self.max_intensity = max_intensity

        # Precompute obstacle mask
        self.walls = (grid == -2)

        # Random state for reproducibility
        self.rng = np.random.default_rng()

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def step(self) -> np.ndarray:
        """
        Advance fire by one step.

        Returns:
            Updated grid
        """
        # Get current fire cells (intensity > 0, < max)
        active_fire = (self.grid > 0) & (self.grid < self.max_intensity)

        # Grow existing fire intensity
        self.grid = np.where(
            active_fire,
            np.minimum(self.grid + self.intensity_growth, self.max_intensity),
            self.grid
        )

        # Calculate spread probability using convolution
        fire_mask = (self.grid > 0).astype(np.float32)
        spread_prob = self._convolve(fire_mask, self.SPREAD_KERNEL)
        spread_prob *= self.spread_rate

        # Determine which cells ignite
        random_vals = self.rng.random((self.rows, self.cols), dtype=np.float32)
        ignite = (random_vals < spread_prob) & (self.grid == 0) & ~self.walls

        # Ignite new cells
        self.grid = np.where(ignite, 1.0, self.grid)

        return self.grid

    def step_n(self, n: int) -> np.ndarray:
        """Advance fire by n steps."""
        for _ in range(n):
            self.step()
        return self.grid

    def get_fire_cells(self) -> np.ndarray:
        """Get coordinates of cells on fire."""
        fire_y, fire_x = np.where(self.grid > 0)
        return np.column_stack((fire_x, fire_y))

    def get_intensity(self, x: int, y: int) -> float:
        """Get fire intensity at position."""
        if 0 <= x < self.cols and 0 <= y < self.rows:
            return max(0, self.grid[y, x])
        return 0.0

    def _convolve(self, arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Manual convolution (faster than scipy for small kernels)."""
        pad = kernel.shape[0] // 2
        padded = np.pad(arr, pad, mode='constant', constant_values=0)

        result = np.zeros_like(arr)
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                result += kernel[i, j] * padded[i:i+self.rows, j:j+self.cols]

        return result


class DeterministicFireModel(FastFireModel):
    """
    Deterministic fire spread for consistent RL training.

    Instead of random ignition, uses threshold-based spread.
    Ensures same input always produces same output.
    """

    def __init__(self, grid: np.ndarray,
                 spread_threshold: float = 0.3,
                 intensity_growth: float = 0.5):
        super().__init__(grid, spread_threshold, intensity_growth)
        self.spread_threshold = spread_threshold

    def step(self) -> np.ndarray:
        """Deterministic fire step."""
        active_fire = (self.grid > 0) & (self.grid < self.max_intensity)

        # Grow existing fire
        self.grid = np.where(
            active_fire,
            np.minimum(self.grid + self.intensity_growth, self.max_intensity),
            self.grid
        )

        # Calculate spread probability
        fire_mask = (self.grid > 0).astype(np.float32)
        spread_prob = self._convolve(fire_mask, self.SPREAD_KERNEL)

        # Deterministic ignition based on threshold
        ignite = (spread_prob >= self.spread_threshold) & (self.grid == 0) & ~self.walls
        self.grid = np.where(ignite, 1.0, self.grid)

        return self.grid
```

### 2.3 Lightweight Simulation Engine

Create `fast_simulation.py`:

```python
"""
Lightweight Simulation for RL Training
======================================

Minimal simulation engine optimized for maximum throughput.
Removes all visualization, I/O, and unnecessary features.

Performance: 10-20x faster than full EvacuationSimulation
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from fast_pathfinder import FastPathfinder
from fast_fire import FastFireModel, DeterministicFireModel

@dataclass
class FastAgent:
    """Minimal agent state."""
    __slots__ = ['x', 'y', 'status', 'steps']
    x: int
    y: int
    status: str  # 'active', 'evacuated', 'stuck', 'dead'
    steps: int

@dataclass
class SimResult:
    """Simulation result for RL."""
    __slots__ = ['steps', 'evacuated', 'stuck', 'dead', 'survival_rate',
                 'avg_evacuation_time', 'reward', 'termination_reason']
    steps: int
    evacuated: int
    stuck: int
    dead: int
    survival_rate: float
    avg_evacuation_time: float
    reward: float
    termination_reason: str


class FastEvacuationSim:
    """
    Lightweight evacuation simulation for RL training.

    Optimizations:
    - Integer coordinates only
    - Shared grid (no per-agent copies)
    - Simple A* pathfinding (no D* Lite)
    - Vectorized fire spread
    - No visualization or I/O
    - Early termination for bad designs
    """

    def __init__(self,
                 grid: np.ndarray,
                 agent_starts: List[Tuple[int, int]],
                 exits: List[Tuple[int, int]],
                 fire_starts: List[Tuple[int, int]] = None,
                 deterministic_fire: bool = True,
                 fire_update_interval: int = 4):
        """
        Initialize simulation.

        Args:
            grid: 2D array (-2=wall, 0=empty)
            agent_starts: List of (x, y) agent starting positions
            exits: List of (x, y) exit positions
            fire_starts: Optional list of (x, y) initial fire positions
            deterministic_fire: Use deterministic fire spread
            fire_update_interval: Steps between fire updates
        """
        # Initialize grid with fire
        self.grid = grid.astype(np.float32)
        if fire_starts:
            for x, y in fire_starts:
                if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                    self.grid[y, x] = 2.0

        # Initialize agents
        self.agents = [FastAgent(x, y, 'active', 0) for x, y in agent_starts]
        self.exits = set(exits)
        self.exit_list = list(exits)

        # Initialize pathfinder and fire model
        self.pathfinder = FastPathfinder(self.grid)
        if deterministic_fire:
            self.fire = DeterministicFireModel(self.grid)
        else:
            self.fire = FastFireModel(self.grid)

        self.fire_update_interval = fire_update_interval
        self.step_count = 0

        # Precompute paths for all agents
        self._agent_paths = [None] * len(self.agents)
        self._recompute_paths()

    def _recompute_paths(self):
        """Recompute paths for all active agents."""
        self.pathfinder.update_grid(self.grid)
        for i, agent in enumerate(self.agents):
            if agent.status == 'active':
                path = self.pathfinder.find_path((agent.x, agent.y),
                                                  self._nearest_exit(agent.x, agent.y))
                self._agent_paths[i] = path

    def _nearest_exit(self, x: int, y: int) -> Tuple[int, int]:
        """Find nearest exit by Manhattan distance."""
        best = self.exit_list[0]
        best_dist = abs(x - best[0]) + abs(y - best[1])
        for ex in self.exit_list[1:]:
            dist = abs(x - ex[0]) + abs(y - ex[1])
            if dist < best_dist:
                best = ex
                best_dist = dist
        return best

    def run(self, max_steps: int = 200,
            stuck_threshold: float = 0.5,
            death_threshold: float = 0.3) -> SimResult:
        """
        Run simulation to completion.

        Args:
            max_steps: Maximum simulation steps
            stuck_threshold: Terminate if this fraction stuck
            death_threshold: Terminate if this fraction dead

        Returns:
            SimResult with metrics
        """
        evacuation_times = []

        for step in range(max_steps):
            self.step_count = step + 1

            # Update fire periodically
            if step > 0 and step % self.fire_update_interval == 0:
                self.grid = self.fire.step()
                self._recompute_paths()

            # Move agents
            active_count = 0
            for i, agent in enumerate(self.agents):
                if agent.status != 'active':
                    continue

                active_count += 1
                agent.steps += 1

                # Check if at exit
                if (agent.x, agent.y) in self.exits:
                    agent.status = 'evacuated'
                    evacuation_times.append(agent.steps)
                    continue

                # Check if in fire
                if self.grid[agent.y, agent.x] > 0:
                    agent.status = 'dead'
                    continue

                # Move along path
                path = self._agent_paths[i]
                if path and len(path) > 1:
                    next_pos = path[1]
                    # Check if next position is safe
                    if self.grid[next_pos[1], next_pos[0]] <= 0:
                        agent.x, agent.y = next_pos
                        self._agent_paths[i] = path[1:]
                    else:
                        # Path blocked, recompute
                        new_path = self.pathfinder.find_path(
                            (agent.x, agent.y),
                            self._nearest_exit(agent.x, agent.y))
                        if new_path:
                            self._agent_paths[i] = new_path
                        else:
                            agent.status = 'stuck'
                else:
                    agent.status = 'stuck'

            # Early termination checks
            if active_count == 0:
                return self._make_result(evacuation_times, 'all_resolved')

            stuck = sum(1 for a in self.agents if a.status == 'stuck')
            dead = sum(1 for a in self.agents if a.status == 'dead')

            if stuck > len(self.agents) * stuck_threshold:
                return self._make_result(evacuation_times, 'mostly_stuck')

            if dead > len(self.agents) * death_threshold:
                return self._make_result(evacuation_times, 'high_casualties')

        return self._make_result(evacuation_times, 'max_steps')

    def _make_result(self, evacuation_times: List[int],
                     reason: str) -> SimResult:
        """Create result object."""
        evacuated = sum(1 for a in self.agents if a.status == 'evacuated')
        stuck = sum(1 for a in self.agents if a.status == 'stuck')
        dead = sum(1 for a in self.agents if a.status == 'dead')
        total = len(self.agents)

        avg_time = np.mean(evacuation_times) if evacuation_times else self.step_count
        survival_rate = evacuated / total if total > 0 else 0

        # RL reward calculation
        reward = (
            evacuated * 10.0 +          # Bonus per evacuated
            stuck * -5.0 +              # Penalty per stuck
            dead * -20.0 +              # Heavy penalty per death
            self.step_count * -0.01     # Small time penalty
        )

        return SimResult(
            steps=self.step_count,
            evacuated=evacuated,
            stuck=stuck,
            dead=dead,
            survival_rate=survival_rate,
            avg_evacuation_time=avg_time,
            reward=reward,
            termination_reason=reason
        )


def evaluate_floor_plan(floor_plan: np.ndarray,
                        agent_positions: List[Tuple[int, int]],
                        exit_positions: List[Tuple[int, int]],
                        fire_positions: List[Tuple[int, int]] = None,
                        max_steps: int = 200,
                        seed: int = None) -> SimResult:
    """
    Evaluate a floor plan design.

    Convenience function for RL training.
    """
    if seed is not None:
        np.random.seed(seed)

    sim = FastEvacuationSim(
        grid=floor_plan,
        agent_starts=agent_positions,
        exits=exit_positions,
        fire_starts=fire_positions
    )

    return sim.run(max_steps=max_steps)


def batch_evaluate(scenarios: List[dict],
                   num_workers: int = None) -> List[SimResult]:
    """
    Evaluate multiple floor plans in parallel.

    Args:
        scenarios: List of dicts with keys:
            - floor_plan: np.ndarray
            - agent_positions: List[Tuple]
            - exit_positions: List[Tuple]
            - fire_positions: List[Tuple] (optional)
            - seed: int (optional)
        num_workers: Number of parallel workers

    Returns:
        List of SimResult objects
    """
    from multiprocessing import Pool, cpu_count

    if num_workers is None:
        num_workers = cpu_count()

    def eval_single(scenario):
        return evaluate_floor_plan(**scenario)

    with Pool(num_workers) as pool:
        results = pool.map(eval_single, scenarios)

    return results
```

### 2.4 RL Training Interface

Create `rl_interface.py`:

```python
"""
RL Training Interface
=====================

Clean interface for training RL agents to design floor plans.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from fast_simulation import FastEvacuationSim, SimResult, batch_evaluate

class FloorPlanEnv:
    """
    Gym-like environment for floor plan design.

    Action space: Place/remove walls, doors, exits
    State space: Current floor plan + metrics
    Reward: Based on evacuation performance
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (30, 30),
                 num_agents: int = 5,
                 num_fires: int = 2,
                 num_exits: int = 2,
                 max_walls: int = 100):
        """
        Initialize environment.

        Args:
            grid_size: (rows, cols) of the floor plan
            num_agents: Number of evacuation agents
            num_fires: Number of fire sources
            num_exits: Number of exits
            max_walls: Maximum walls that can be placed
        """
        self.rows, self.cols = grid_size
        self.num_agents = num_agents
        self.num_fires = num_fires
        self.num_exits = num_exits
        self.max_walls = max_walls

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset to empty floor plan."""
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.walls_placed = 0
        self.exits = []
        self.agent_starts = []
        self.fire_starts = []

        return self._get_state()

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take action in environment.

        Actions:
        - {'type': 'wall', 'x': int, 'y': int}: Place wall
        - {'type': 'exit', 'x': int, 'y': int}: Place exit
        - {'type': 'agent', 'x': int, 'y': int}: Place agent start
        - {'type': 'fire', 'x': int, 'y': int}: Place fire
        - {'type': 'remove', 'x': int, 'y': int}: Remove wall
        - {'type': 'evaluate'}: Run simulation and get reward

        Returns:
            state, reward, done, info
        """
        action_type = action['type']

        if action_type == 'evaluate':
            return self._evaluate()

        x, y = action.get('x', 0), action.get('y', 0)

        if not (0 <= x < self.cols and 0 <= y < self.rows):
            return self._get_state(), -1.0, False, {'error': 'out_of_bounds'}

        if action_type == 'wall':
            if self.walls_placed < self.max_walls and self.grid[y, x] == 0:
                self.grid[y, x] = -2
                self.walls_placed += 1

        elif action_type == 'exit':
            if self.grid[y, x] == 0:
                self.exits.append((x, y))

        elif action_type == 'agent':
            if self.grid[y, x] == 0 and len(self.agent_starts) < self.num_agents:
                self.agent_starts.append((x, y))

        elif action_type == 'fire':
            if self.grid[y, x] == 0 and len(self.fire_starts) < self.num_fires:
                self.fire_starts.append((x, y))

        elif action_type == 'remove':
            if self.grid[y, x] == -2:
                self.grid[y, x] = 0
                self.walls_placed -= 1

        return self._get_state(), 0.0, False, {}

    def _evaluate(self) -> Tuple[np.ndarray, float, bool, dict]:
        """Run simulation and return results."""
        # Validate setup
        if len(self.exits) == 0:
            return self._get_state(), -100.0, True, {'error': 'no_exits'}

        if len(self.agent_starts) == 0:
            # Random agent placement
            self._place_random_agents()

        if len(self.fire_starts) == 0:
            # Random fire placement
            self._place_random_fires()

        # Run simulation
        sim = FastEvacuationSim(
            grid=self.grid,
            agent_starts=self.agent_starts,
            exits=self.exits,
            fire_starts=self.fire_starts
        )

        result = sim.run(max_steps=200)

        info = {
            'evacuated': result.evacuated,
            'stuck': result.stuck,
            'dead': result.dead,
            'steps': result.steps,
            'survival_rate': result.survival_rate
        }

        return self._get_state(), result.reward, True, info

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        # Stack channels: walls, exits, agents, fires
        state = np.zeros((4, self.rows, self.cols), dtype=np.float32)
        state[0] = (self.grid == -2).astype(np.float32)  # Walls

        for x, y in self.exits:
            state[1, y, x] = 1.0
        for x, y in self.agent_starts:
            state[2, y, x] = 1.0
        for x, y in self.fire_starts:
            state[3, y, x] = 1.0

        return state

    def _place_random_agents(self):
        """Place agents randomly on empty cells."""
        empty = np.where(self.grid == 0)
        indices = np.random.choice(len(empty[0]),
                                   min(self.num_agents, len(empty[0])),
                                   replace=False)
        self.agent_starts = [(empty[1][i], empty[0][i]) for i in indices]

    def _place_random_fires(self):
        """Place fires randomly on empty cells."""
        empty = np.where(self.grid == 0)
        if len(empty[0]) > self.num_fires:
            indices = np.random.choice(len(empty[0]), self.num_fires, replace=False)
            self.fire_starts = [(empty[1][i], empty[0][i]) for i in indices]

    def render(self) -> str:
        """Simple text rendering."""
        chars = {-2: '#', 0: '.'}
        lines = []
        for y in range(self.rows):
            row = ''
            for x in range(self.cols):
                if (x, y) in self.exits:
                    row += 'E'
                elif (x, y) in self.agent_starts:
                    row += 'A'
                elif (x, y) in self.fire_starts:
                    row += 'F'
                else:
                    row += chars.get(self.grid[y, x], '.')
            lines.append(row)
        return '\n'.join(lines)
```

### Expected Results - Phase 2

| Metric | Phase 1 | Phase 2 | Improvement |
|--------|---------|---------|-------------|
| Time per simulation | 0.4-0.6s | 0.04-0.1s | 5-10x faster |
| Memory per simulation | 30MB | 10MB | 3x less |
| Simulations per hour | 6,000-9,000 | 36,000-90,000 | 6-10x more |

---

## Phase 3: Aggressive Optimizations (1-2 months, 100-1000x speedup)

JIT compilation, GPU acceleration, and ML surrogate models.

### 3.1 Numba JIT Compilation

Create `jit_pathfinder.py`:

```python
"""
JIT-Compiled Pathfinding
========================

Numba-accelerated A* for maximum single-thread performance.
10-50x faster than pure Python.
"""

from numba import jit, prange
import numpy as np

@jit(nopython=True, cache=True)
def astar_jit(grid: np.ndarray,
              start_x: int, start_y: int,
              goal_x: int, goal_y: int) -> float:
    """
    JIT-compiled A* pathfinding.

    Returns path length or inf if no path exists.
    """
    rows, cols = grid.shape
    INF = np.float32(1e9)

    # g_score array
    g_score = np.full((rows, cols), INF, dtype=np.float32)
    g_score[start_y, start_x] = 0.0

    # Open set as parallel arrays (faster than heap for small grids)
    max_open = rows * cols
    open_x = np.zeros(max_open, dtype=np.int32)
    open_y = np.zeros(max_open, dtype=np.int32)
    open_f = np.full(max_open, INF, dtype=np.float32)
    open_size = 1

    open_x[0] = start_x
    open_y[0] = start_y
    open_f[0] = max(abs(start_x - goal_x), abs(start_y - goal_y))

    # Direction arrays
    dx = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int32)
    dy = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int32)
    costs = np.array([1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414], dtype=np.float32)

    while open_size > 0:
        # Find minimum f
        min_idx = 0
        min_f = open_f[0]
        for i in range(1, open_size):
            if open_f[i] < min_f:
                min_f = open_f[i]
                min_idx = i

        cx = open_x[min_idx]
        cy = open_y[min_idx]

        # Check if goal reached
        if cx == goal_x and cy == goal_y:
            return g_score[goal_y, goal_x]

        # Remove from open set (swap with last)
        open_size -= 1
        open_x[min_idx] = open_x[open_size]
        open_y[min_idx] = open_y[open_size]
        open_f[min_idx] = open_f[open_size]

        # Expand neighbors
        for d in range(8):
            nx = cx + dx[d]
            ny = cy + dy[d]

            if 0 <= nx < cols and 0 <= ny < rows:
                if grid[ny, nx] >= 0:  # Not wall
                    tentative_g = g_score[cy, cx] + costs[d]

                    if tentative_g < g_score[ny, nx]:
                        g_score[ny, nx] = tentative_g
                        h = max(abs(nx - goal_x), abs(ny - goal_y))
                        f = tentative_g + h

                        # Add to open set
                        if open_size < max_open:
                            open_x[open_size] = nx
                            open_y[open_size] = ny
                            open_f[open_size] = f
                            open_size += 1

    return INF  # No path


@jit(nopython=True, parallel=True, cache=True)
def batch_astar(grids: np.ndarray,
                starts: np.ndarray,
                goals: np.ndarray) -> np.ndarray:
    """
    Parallel A* for batch of scenarios.

    Args:
        grids: (batch, rows, cols) array of grids
        starts: (batch, 2) array of start positions [x, y]
        goals: (batch, 2) array of goal positions [x, y]

    Returns:
        (batch,) array of path lengths
    """
    n = grids.shape[0]
    results = np.zeros(n, dtype=np.float32)

    for i in prange(n):
        results[i] = astar_jit(
            grids[i],
            starts[i, 0], starts[i, 1],
            goals[i, 0], goals[i, 1]
        )

    return results


@jit(nopython=True, cache=True)
def fire_step_jit(grid: np.ndarray,
                  walls: np.ndarray,
                  spread_rate: float = 0.3) -> np.ndarray:
    """
    JIT-compiled fire spread step.
    """
    rows, cols = grid.shape
    new_grid = grid.copy()

    # Grow existing fire
    for y in range(rows):
        for x in range(cols):
            if grid[y, x] > 0 and grid[y, x] < 4:
                new_grid[y, x] = min(grid[y, x] + 0.5, 4.0)

    # Spread to neighbors
    for y in range(rows):
        for x in range(cols):
            if grid[y, x] == 0 and not walls[y, x]:
                # Count fire neighbors
                fire_influence = 0.0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            if grid[ny, nx] > 0:
                                # Diagonal has less influence
                                weight = 0.1 if (abs(dx) + abs(dy) == 2) else 0.15
                                fire_influence += weight

                # Ignite based on influence
                if fire_influence * spread_rate > 0.3:
                    new_grid[y, x] = 1.0

    return new_grid


@jit(nopython=True, cache=True)
def simulate_evacuation_jit(grid: np.ndarray,
                            agent_x: np.ndarray,
                            agent_y: np.ndarray,
                            exit_x: np.ndarray,
                            exit_y: np.ndarray,
                            max_steps: int = 200,
                            fire_interval: int = 4) -> tuple:
    """
    Full JIT-compiled evacuation simulation.

    Returns: (evacuated, stuck, dead, steps)
    """
    rows, cols = grid.shape
    n_agents = len(agent_x)
    n_exits = len(exit_x)

    walls = (grid == -2)
    current_grid = grid.copy()

    # Agent state: 0=active, 1=evacuated, 2=stuck, 3=dead
    status = np.zeros(n_agents, dtype=np.int32)
    agent_steps = np.zeros(n_agents, dtype=np.int32)

    for step in range(max_steps):
        # Update fire
        if step > 0 and step % fire_interval == 0:
            current_grid = fire_step_jit(current_grid, walls)

        active_count = 0

        for i in range(n_agents):
            if status[i] != 0:
                continue

            active_count += 1
            agent_steps[i] += 1
            ax, ay = agent_x[i], agent_y[i]

            # Check if at exit
            for j in range(n_exits):
                if ax == exit_x[j] and ay == exit_y[j]:
                    status[i] = 1  # Evacuated
                    break

            if status[i] != 0:
                continue

            # Check if in fire
            if current_grid[ay, ax] > 0:
                status[i] = 3  # Dead
                continue

            # Find nearest exit and move toward it
            best_exit = 0
            best_dist = abs(ax - exit_x[0]) + abs(ay - exit_y[0])
            for j in range(1, n_exits):
                dist = abs(ax - exit_x[j]) + abs(ay - exit_y[j])
                if dist < best_dist:
                    best_dist = dist
                    best_exit = j

            # Simple greedy move toward exit
            tx, ty = exit_x[best_exit], exit_y[best_exit]

            # Try to move
            moved = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    if current_grid[ny, nx] >= 0 and current_grid[ny, nx] < 1:
                        # Check if this move gets us closer
                        old_dist = abs(ax - tx) + abs(ay - ty)
                        new_dist = abs(nx - tx) + abs(ny - ty)
                        if new_dist < old_dist:
                            agent_x[i] = nx
                            agent_y[i] = ny
                            moved = True
                            break

            if not moved:
                # Try any valid move
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = ax + dx, ay + dy
                    if 0 <= nx < cols and 0 <= ny < rows:
                        if current_grid[ny, nx] >= 0 and current_grid[ny, nx] < 1:
                            agent_x[i] = nx
                            agent_y[i] = ny
                            moved = True
                            break

                if not moved:
                    status[i] = 2  # Stuck

        if active_count == 0:
            break

    evacuated = np.sum(status == 1)
    stuck = np.sum(status == 2)
    dead = np.sum(status == 3)

    return (evacuated, stuck, dead, step + 1)
```

### 3.2 Neural Surrogate Model

Create `surrogate_model.py`:

```python
"""
Neural Surrogate Model for Floor Plan Evaluation
=================================================

Train a CNN to predict evacuation metrics directly from floor plan images.
1000x faster than simulation once trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict

class FloorPlanCNN(nn.Module):
    """
    CNN to predict evacuation metrics from floor plan.

    Input: (batch, 4, H, W) - channels: walls, exits, agents, fires
    Output: (batch, 4) - [evacuation_rate, avg_time, stuck_rate, death_rate]
    """

    def __init__(self, grid_size: int = 30):
        super().__init__()

        self.conv = nn.Sequential(
            # First block: 4 -> 32 channels
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second block: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third block: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # Output: (batch, 128, 4, 4)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 output metrics
            nn.Sigmoid()  # All outputs in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc(x)
        return x


class SurrogateTrainer:
    """
    Train surrogate model on simulation data.
    """

    def __init__(self, grid_size: int = 30, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = FloorPlanCNN(grid_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(batch_x)
            loss = self.criterion(pred, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                pred = self.model(batch_x)
                loss = self.criterion(pred, batch_y)
                total_loss += loss.item()

                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        return {
            'loss': total_loss / len(dataloader),
            'mae': np.mean(np.abs(preds - targets)),
            'correlation': np.corrcoef(preds.flatten(), targets.flatten())[0, 1]
        }

    def predict(self, floor_plan: np.ndarray) -> np.ndarray:
        """Predict metrics for single floor plan."""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(floor_plan).unsqueeze(0).to(self.device)
            pred = self.model(x)
            return pred.cpu().numpy()[0]

    def save(self, path: str):
        """Save model."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Load model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def generate_training_data(num_samples: int = 10000,
                           grid_size: int = 30,
                           num_agents: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data by running simulations.

    Returns:
        X: (num_samples, 4, grid_size, grid_size) floor plans
        y: (num_samples, 4) metrics [evac_rate, avg_time, stuck_rate, death_rate]
    """
    from fast_simulation import FastEvacuationSim

    X = np.zeros((num_samples, 4, grid_size, grid_size), dtype=np.float32)
    y = np.zeros((num_samples, 4), dtype=np.float32)

    for i in range(num_samples):
        # Generate random floor plan
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Random walls (10-30% of cells)
        num_walls = np.random.randint(grid_size * grid_size // 10,
                                       grid_size * grid_size // 3)
        wall_indices = np.random.choice(grid_size * grid_size, num_walls, replace=False)
        for idx in wall_indices:
            grid[idx // grid_size, idx % grid_size] = -2

        # Random exits (1-3)
        num_exits = np.random.randint(1, 4)
        empty_cells = np.where(grid == 0)
        exit_indices = np.random.choice(len(empty_cells[0]), num_exits, replace=False)
        exits = [(empty_cells[1][j], empty_cells[0][j]) for j in exit_indices]

        # Random agents
        empty_cells = np.where(grid == 0)
        agent_indices = np.random.choice(len(empty_cells[0]),
                                         min(num_agents, len(empty_cells[0])),
                                         replace=False)
        agents = [(empty_cells[1][j], empty_cells[0][j]) for j in agent_indices]

        # Random fires (1-2)
        empty_cells = np.where(grid == 0)
        num_fires = np.random.randint(1, 3)
        fire_indices = np.random.choice(len(empty_cells[0]),
                                        min(num_fires, len(empty_cells[0])),
                                        replace=False)
        fires = [(empty_cells[1][j], empty_cells[0][j]) for j in fire_indices]

        # Build input tensor
        X[i, 0] = (grid == -2).astype(np.float32)  # Walls
        for ex, ey in exits:
            X[i, 1, ey, ex] = 1.0  # Exits
        for ax, ay in agents:
            X[i, 2, ay, ax] = 1.0  # Agents
        for fx, fy in fires:
            X[i, 3, fy, fx] = 1.0  # Fires

        # Run simulation
        try:
            sim = FastEvacuationSim(grid, agents, exits, fires)
            result = sim.run(max_steps=200)

            total = len(agents)
            y[i, 0] = result.evacuated / total  # Evacuation rate
            y[i, 1] = result.steps / 200  # Normalized time
            y[i, 2] = result.stuck / total  # Stuck rate
            y[i, 3] = result.dead / total  # Death rate
        except:
            # Invalid configuration, use worst case
            y[i] = [0, 1, 0.5, 0.5]

        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")

    return X, y
```

### 3.3 Multi-Fidelity Evaluation

Create `multi_fidelity.py`:

```python
"""
Multi-Fidelity Floor Plan Evaluation
====================================

Use cheap approximations first, expensive simulations only for promising designs.
Reduces average evaluation time by 10-100x.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class EvalResult:
    """Multi-fidelity evaluation result."""
    reward: float
    metrics: Dict[str, float]
    fidelity_level: str
    eval_time: float

class MultiFidelityEvaluator:
    """
    Progressive floor plan evaluation.

    Levels:
    1. Heuristic score (0.0001s) - Quick rejection of obviously bad designs
    2. Neural surrogate (0.001s) - Learned approximation
    3. Fast simulation (0.05s) - Simplified physics
    4. Full simulation (0.5s) - Complete D* Lite + realistic fire
    """

    def __init__(self,
                 surrogate_model=None,
                 fast_sim_class=None,
                 full_sim_class=None):
        self.surrogate = surrogate_model
        self.fast_sim = fast_sim_class
        self.full_sim = full_sim_class

        # Thresholds for progressive evaluation
        self.heuristic_threshold = 0.3
        self.surrogate_threshold = 0.5
        self.fast_sim_threshold = 0.7

    def evaluate(self, floor_plan: np.ndarray,
                 agent_positions: list,
                 exit_positions: list,
                 fire_positions: list = None,
                 max_fidelity: str = 'full') -> EvalResult:
        """
        Evaluate floor plan with progressive fidelity.

        Args:
            floor_plan: 2D numpy array
            agent_positions: List of (x, y)
            exit_positions: List of (x, y)
            fire_positions: List of (x, y)
            max_fidelity: Maximum fidelity level to use

        Returns:
            EvalResult with metrics and fidelity level used
        """
        import time

        # Level 1: Heuristic score (~0.0001s)
        start = time.time()
        heuristic = self._heuristic_score(floor_plan, agent_positions,
                                          exit_positions, fire_positions)
        heuristic_time = time.time() - start

        if heuristic < self.heuristic_threshold:
            return EvalResult(
                reward=heuristic * 10 - 50,  # Negative reward for bad heuristic
                metrics={'heuristic': heuristic},
                fidelity_level='heuristic',
                eval_time=heuristic_time
            )

        if max_fidelity == 'heuristic':
            return EvalResult(
                reward=heuristic * 10,
                metrics={'heuristic': heuristic},
                fidelity_level='heuristic',
                eval_time=heuristic_time
            )

        # Level 2: Neural surrogate (~0.001s)
        if self.surrogate is not None:
            start = time.time()
            surrogate_pred = self._surrogate_predict(floor_plan, agent_positions,
                                                     exit_positions, fire_positions)
            surrogate_time = time.time() - start

            if surrogate_pred['survival_rate'] < self.surrogate_threshold:
                return EvalResult(
                    reward=self._compute_reward(surrogate_pred),
                    metrics=surrogate_pred,
                    fidelity_level='surrogate',
                    eval_time=heuristic_time + surrogate_time
                )

            if max_fidelity == 'surrogate':
                return EvalResult(
                    reward=self._compute_reward(surrogate_pred),
                    metrics=surrogate_pred,
                    fidelity_level='surrogate',
                    eval_time=heuristic_time + surrogate_time
                )

        # Level 3: Fast simulation (~0.05s)
        if self.fast_sim is not None:
            start = time.time()
            fast_result = self._run_fast_sim(floor_plan, agent_positions,
                                             exit_positions, fire_positions)
            fast_time = time.time() - start

            if fast_result['survival_rate'] < self.fast_sim_threshold:
                return EvalResult(
                    reward=self._compute_reward(fast_result),
                    metrics=fast_result,
                    fidelity_level='fast',
                    eval_time=heuristic_time + fast_time
                )

            if max_fidelity == 'fast':
                return EvalResult(
                    reward=self._compute_reward(fast_result),
                    metrics=fast_result,
                    fidelity_level='fast',
                    eval_time=heuristic_time + fast_time
                )

        # Level 4: Full simulation (~0.5s)
        if self.full_sim is not None:
            start = time.time()
            full_result = self._run_full_sim(floor_plan, agent_positions,
                                             exit_positions, fire_positions)
            full_time = time.time() - start

            return EvalResult(
                reward=self._compute_reward(full_result),
                metrics=full_result,
                fidelity_level='full',
                eval_time=heuristic_time + full_time
            )

        # Fallback to fast result or heuristic
        return EvalResult(
            reward=heuristic * 10,
            metrics={'heuristic': heuristic},
            fidelity_level='heuristic',
            eval_time=heuristic_time
        )

    def _heuristic_score(self, floor_plan, agents, exits, fires) -> float:
        """Quick heuristic evaluation."""
        score = 1.0

        # Penalize if no exits
        if not exits:
            return 0.0

        # Penalize if exits blocked
        for ex, ey in exits:
            if floor_plan[ey, ex] != 0:
                score -= 0.2

        # Check path existence (simple flood fill)
        for ax, ay in agents:
            if not self._path_exists(floor_plan, (ax, ay), exits[0]):
                score -= 0.3

        # Penalize high wall density
        wall_ratio = np.sum(floor_plan == -2) / floor_plan.size
        if wall_ratio > 0.4:
            score -= 0.2

        return max(0.0, score)

    def _path_exists(self, grid, start, goal) -> bool:
        """Simple BFS to check if path exists."""
        from collections import deque

        rows, cols = grid.shape
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < cols and 0 <= ny < rows and
                    (nx, ny) not in visited and grid[ny, nx] >= 0):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return False

    def _surrogate_predict(self, floor_plan, agents, exits, fires) -> dict:
        """Get surrogate model prediction."""
        # Build input tensor
        state = np.zeros((4, floor_plan.shape[0], floor_plan.shape[1]), dtype=np.float32)
        state[0] = (floor_plan == -2)
        for x, y in exits:
            state[1, y, x] = 1.0
        for x, y in agents:
            state[2, y, x] = 1.0
        if fires:
            for x, y in fires:
                state[3, y, x] = 1.0

        pred = self.surrogate.predict(state)
        return {
            'survival_rate': pred[0],
            'avg_time': pred[1] * 200,
            'stuck_rate': pred[2],
            'death_rate': pred[3]
        }

    def _run_fast_sim(self, floor_plan, agents, exits, fires) -> dict:
        """Run fast simulation."""
        sim = self.fast_sim(floor_plan, agents, exits, fires)
        result = sim.run(max_steps=200)
        return {
            'survival_rate': result.survival_rate,
            'avg_time': result.avg_evacuation_time,
            'stuck_rate': result.stuck / len(agents),
            'death_rate': result.dead / len(agents)
        }

    def _run_full_sim(self, floor_plan, agents, exits, fires) -> dict:
        """Run full simulation."""
        # Convert to full simulation format
        from simulation import EvacuationSimulation, SimulationConfig
        # ... implementation
        pass

    def _compute_reward(self, metrics: dict) -> float:
        """Compute RL reward from metrics."""
        return (
            metrics.get('survival_rate', 0) * 100 +
            metrics.get('stuck_rate', 0) * -30 +
            metrics.get('death_rate', 0) * -50 +
            (200 - metrics.get('avg_time', 200)) * 0.1
        )
```

### Expected Results - Phase 3

| Metric | Phase 2 | Phase 3 | Improvement |
|--------|---------|---------|-------------|
| Time per simulation | 0.04-0.1s | 0.001-0.02s | 5-50x faster |
| Simulations per hour | 36,000-90,000 | 180,000-3,600,000 | 5-40x more |
| GPU batch throughput | N/A | 10,000+ per second | New capability |

---

## Implementation Roadmap

### Week 1-2: Phase 1 (Conservative)
- [x] Create `configs/rl_training_config.json`
- [x] Add early termination to `simulation.py`
- [x] Create `rl_simulation.py` wrapper
- [x] Benchmark: Target 6,000+ sims/hour

### Week 3-4: Phase 2 (Moderate)
- [ ] Implement `fast_pathfinder.py`
- [ ] Implement `fast_fire.py`
- [ ] Implement `fast_simulation.py`
- [ ] Create `rl_interface.py`
- [ ] Benchmark: Target 50,000+ sims/hour

### Month 2: Phase 3 (Aggressive)
- [ ] Implement `jit_pathfinder.py` with Numba
- [ ] Generate training data for surrogate
- [ ] Train `surrogate_model.py`
- [ ] Implement `multi_fidelity.py`
- [ ] Benchmark: Target 500,000+ sims/hour

### Month 3: Integration & Tuning
- [ ] Integrate with RL framework (Stable Baselines3 / RLlib)
- [ ] Implement curriculum learning
- [ ] Tune surrogate model accuracy
- [ ] Production deployment

---

## Quick Start

### Immediate (Today)
```bash
# Use aggressive config
cp configs/rl_training_config.json configs/my_config.json
python simulation.py --config configs/my_config.json
```

### This Week
```python
# Use fast simulation wrapper
from rl_simulation import RLSimulationWrapper

env = RLSimulationWrapper()
result = env.evaluate(
    floor_plan=my_design,
    agent_positions=[(5, 5), (10, 10)],
    exit_positions=[(0, 15), (29, 15)]
)
print(f"Reward: {result['reward']}")
```

### This Month
```python
# Use multi-fidelity evaluation
from multi_fidelity import MultiFidelityEvaluator
from surrogate_model import SurrogateTrainer

# Train surrogate on 10k simulations
trainer = SurrogateTrainer()
# ... training code ...

evaluator = MultiFidelityEvaluator(
    surrogate_model=trainer,
    fast_sim_class=FastEvacuationSim
)

# Evaluate floor plan (uses cheapest sufficient fidelity)
result = evaluator.evaluate(floor_plan, agents, exits, fires)
print(f"Reward: {result.reward}, Fidelity: {result.fidelity_level}")
```

---

## Dependencies

### Phase 1 (No new dependencies)
- numpy
- existing simulation code

### Phase 2
```bash
pip install scipy  # For convolution (optional)
```

### Phase 3
```bash
pip install numba torch  # JIT + neural surrogate
pip install jax jaxlib   # Optional GPU acceleration
```

---

## FAQ

**Q: Which phase should I start with?**
A: Start with Phase 1 immediately (1-2 days). The config changes alone provide 3-5x speedup with zero code changes.

**Q: How accurate is the neural surrogate?**
A: With 10k training samples, expect 90-95% correlation with full simulation. For RL training, this is sufficient since the policy learns to optimize the surrogate reward.

**Q: Can I use GPU?**
A: Yes, Phase 3 supports GPU via PyTorch (surrogate model) and optionally JAX (vectorized fire/pathfinding). Batch evaluation of 1000+ floor plans is 100x faster on GPU.

**Q: What about determinism for reproducibility?**
A: Use `DeterministicFireModel` and set random seeds. The JIT-compiled version is fully deterministic.
