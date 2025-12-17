# AI-Guided Design Optimization Roadmap

Performance optimization guide for using Monte Carlo simulations to train a pairwise ranking model that accelerates door configuration search.

## Executive Summary

**Goal**: Make simulations fast enough for pairwise comparison labeling (10K-100K candidate evaluations)

| Approach | Effort | Speedup | Per-Sim Time |
|----------|--------|---------|--------------|
| Current baseline | - | 1x | ~2.0s |
| Conservative (config tuning) | 1-2 days | 3-5x | ~0.4-0.6s |
| Moderate (architectural) | 1-2 weeks | 10-50x | ~0.04-0.2s |
| Aggressive (ML acceleration) | 1-2 months | 100-1000x | ~0.002-0.02s |

---

## Current Bottlenecks

From profiling analysis (`PERFORMANCE_RESULTS.md`):

| Component | Time % | Scaling Issue | Impact on AI Training |
|-----------|--------|---------------|----------------------|
| D* Lite pathfinding | 54% | O(agents × grid × replans) | Critical |
| Environment updates | 20% | O(agents × fire_changes) | High |
| Fire simulation | 15% | O(grid_size) per update | Medium |
| Coordinate parsing | 5% | O(calls) - now cached | Low |
| I/O and visualization | 6% | O(data_size) | Eliminable |

### Why Current Design is Slow for Pairwise Labeling

1. **Need for batch evaluation**: Training requires k=3-5 Monte Carlo runs per candidate × N candidates per pair
2. **D* Lite overhead**: Incremental replanning unnecessary when running independent candidate evaluations
3. **Per-agent graph copies**: Each agent maintains full GridWorld (~1.8MB each)
4. **Full physics fire model**: Detailed physics tracking unnecessary for relative comparisons
5. **Visualization overhead**: Setup/teardown costs exist even when disabled
6. **Sequential labeling**: Cannot efficiently parallelize pairwise comparison generation

---

## Phase 1: Conservative Optimizations (1-2 days, 3-5x speedup)

Configuration and minor code changes only. No architectural rewrites.

### 1.1 Labeling-Optimized Configuration

Create `configs/ai_labeling_config.json`:

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

Create `ai_labeling_wrapper.py` wrapper:

```python
"""Lightweight simulation wrapper for pairwise comparison labeling."""

from simulation import EvacuationSimulation, SimulationConfig
import numpy as np

class AILabelingWrapper:
    """Zero-overhead simulation for generating pairwise labels."""

    def __init__(self, base_config_path='configs/ai_labeling_config.json'):
        import json
        with open(base_config_path) as f:
            self.base_config = json.load(f)

    def evaluate_candidate(self, floor_plan: np.ndarray,
                          door_config: list,
                          num_trials: int = 1,
                          seed: int = None) -> dict:
        """
        Evaluate a single door configuration candidate.

        Args:
            floor_plan: 2D numpy array (-2=wall, 0=empty)
            door_config: List of door dicts [{"id": "d1", "position": "x5y3", "type": "door"}, ...]
            num_trials: Number of Monte Carlo trials (k=3-5 for labeling)
            seed: Random seed for reproducibility

        Returns:
            Dict with averaged evacuation metrics
        """
        results = []
        for trial in range(num_trials):
            trial_seed = seed + trial if seed is not None else None
            config = self._build_config(floor_plan, door_config, trial_seed)

            sim = EvacuationSimulation(config)

            # Run with all overhead disabled
            result = sim.run(
                max_steps=200,
                show_visualization=False,
                use_pygame=False,
                use_matlab=False,
                early_termination=True
            )
            results.append(result)

        # Compute robust statistics (median/trimmed mean)
        return self._aggregate_results(results)

    def generate_pairwise_labels(self, floor_plan: np.ndarray,
                                 candidate_pairs: list,
                                 num_trials: int = 3,
                                 margin: float = 0.05) -> list:
        """
        Generate pairwise comparison labels for training.

        Args:
            floor_plan: Base floor plan (same for all candidates)
            candidate_pairs: List of (door_config_A, door_config_B) tuples
            num_trials: Monte Carlo trials per candidate (k=3-5)
            margin: Minimum difference to assign label (rejects ambiguous pairs)

        Returns:
            List of (config_A, config_B, label) where label=1 if A>B, 0 if B>A, None if ambiguous
        """
        labels = []
        for config_a, config_b in candidate_pairs:
            # Evaluate both candidates
            result_a = self.evaluate_candidate(floor_plan, config_a, num_trials)
            result_b = self.evaluate_candidate(floor_plan, config_b, num_trials)

            # Compare using robust metric (survival rate + time penalty)
            score_a = result_a['survival_rate'] - result_a['avg_evacuation_time'] / 1000
            score_b = result_b['survival_rate'] - result_b['avg_evacuation_time'] / 1000

            # Assign label with margin
            if score_a > score_b + margin:
                label = 1  # A is better
            elif score_b > score_a + margin:
                label = 0  # B is better
            else:
                label = None  # Ambiguous, discard

            labels.append((config_a, config_b, label, score_a, score_b))

        return labels

    def _build_config(self, floor_plan, door_config, seed):
        """Build SimulationConfig with door configuration."""
        config_dict = self.base_config.copy()

        rows, cols = floor_plan.shape
        config_dict['map_rows'] = rows
        config_dict['map_cols'] = cols
        config_dict['initial_fire_map'] = floor_plan.tolist()
        config_dict['door_configs'] = door_config

        if seed is not None:
            np.random.seed(seed)

        return SimulationConfig.from_json(config_dict)

    def _aggregate_results(self, results):
        """Aggregate multiple trial results using median."""
        evacuated = np.median([r['evacuated'] for r in results])
        stuck = np.median([r['stuck'] for r in results])
        dead = np.median([r['dead'] for r in results])
        steps = np.median([r['steps'] for r in results])
        total = len(results[0].get('agents', []))

        return {
            'evacuated': int(evacuated),
            'stuck': int(stuck),
            'dead': int(dead),
            'steps': int(steps),
            'survival_rate': evacuated / total if total > 0 else 0,
            'avg_evacuation_time': steps
        }

    def batch_evaluate(self, candidates: list, num_workers: int = None) -> list:
        """Evaluate multiple candidates in parallel."""
        from multiprocessing import Pool, cpu_count

        if num_workers is None:
            num_workers = cpu_count()

        with Pool(num_workers) as pool:
            results = pool.map(self._evaluate_single, candidates)

        return results

    def _evaluate_single(self, args):
        """Single evaluation for multiprocessing."""
        floor_plan, door_config, num_trials, seed = args
        return self.evaluate_candidate(floor_plan, door_config, num_trials, seed)
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

### 1.5 Candidate Generator (Completed)

Implemented `candidate_generator.py` for generating door configuration candidates:

**Features:**
- **Random placement**: Uniform sampling of valid wall positions
- **Rule-based strategies**:
  - `boundary_focused`: Prioritizes room boundaries for doors, perimeter for exits
  - `distributed`: Evenly distributes doors across floor plan grid sectors
  - `corner_exits`: Places exits in corners, doors on boundaries
- **Constraint validation**: Minimum spacing, connectivity checks
- **Room detection**: Uses connected components to identify separate rooms
- **Flexible exit placement**: Adaptive perimeter zone for exit positioning

**Usage:**
```python
from candidate_generator import generate_door_candidates

# Generate 100 candidates
candidates = generate_door_candidates(
    floor_plan=floor_plan,
    num_candidates=100,
    num_doors_range=(2, 5),
    num_exits_range=(1, 3),
    min_door_spacing=5,
    random_ratio=0.5,  # 50% random, 50% rule-based
    seed=42
)

# Each candidate is a list of door dicts
# [{"id": "d1", "position": "x15y20", "type": "door"}, ...]
```

**Integration with Pairwise Labeling:**
- Generates diverse candidate pool for evaluation
- Supports both random exploration and structured design rules
- Output format directly compatible with simulation config
- See `examples/candidate_generator_demo.py` for complete examples

---

## Phase 2: Moderate Optimizations (1-2 weeks, 10-50x speedup)

Architectural improvements while staying in Python and maintaining simulation accuracy.

**Key Philosophy**: Optimize the existing D* Lite algorithm rather than replacing it, since:
- Fire spreads dynamically during simulation (not known at start)
- D* Lite's incremental replanning is specifically designed for dynamic environments
- Replacing with A* would change agent behavior and invalidate training labels

### 2.1 Optimized D* Lite Pathfinder

Create `optimized_d_star_lite.py`:

```python
"""
Optimized D* Lite Pathfinder
============================

Performance optimizations for D* Lite while maintaining incremental replanning.
D* Lite is ESSENTIAL for dynamic fire environments because:
- Fire spreads continuously during simulation
- Agents discover changes incrementally within viewing_range
- Incremental replanning is faster than A* replanning from scratch

Performance target: 3-5x faster than current implementation
"""

import heapq
import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from functools import lru_cache
from collections import defaultdict

class OptimizedDStarLite:
    """
    Optimized D* Lite with performance improvements.

    Key Optimizations:
    1. Integer coordinates (no string parsing overhead)
    2. NumPy-backed priority queue and data structures
    3. Preallocated arrays for g/rhs values
    4. Efficient neighbor iteration
    5. Shared grid reference (no deep copies)
    """

    # 8-directional movement with costs
    DIRS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)], dtype=np.int32)
    COSTS = np.array([1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414], dtype=np.float32)

    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Initialize D* Lite pathfinder.

        Args:
            grid: 2D numpy array (-2=wall, 0=empty, >0=fire)
            start: (x, y) starting position
            goal: (x, y) goal position
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.start = start
        self.goal = goal

        # Preallocate cost arrays (much faster than dicts for grid-based access)
        self.g = np.full((self.rows, self.cols), np.inf, dtype=np.float32)
        self.rhs = np.full((self.rows, self.cols), np.inf, dtype=np.float32)

        # Priority queue: list of (key, (x, y))
        self.U = []
        self.counter = 0  # Tie-breaker

        # k_m for D* Lite
        self.k_m = 0

        # Initialize
        self.rhs[goal[1], goal[0]] = 0
        self._insert(goal, self._calculate_key(goal))

        # Precompute obstacle mask
        self._update_obstacles()

    def _update_obstacles(self):
        """Update obstacle mask from current grid."""
        self.obstacles = (self.grid == -2) | (self.grid > 3.0)  # Walls or intense fire

    def _calculate_key(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        """Calculate priority key for D* Lite."""
        x, y = pos
        g_val = self.g[y, x]
        rhs_val = self.rhs[y, x]
        min_val = min(g_val, rhs_val)

        h = self._heuristic(pos, self.start)
        return (min_val + h + self.k_m, min_val)

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Chebyshev distance heuristic."""
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _insert(self, pos: Tuple[int, int], key: Tuple[float, float]):
        """Insert node into priority queue."""
        self.counter += 1
        heapq.heappush(self.U, (key, self.counter, pos))

    def _pop(self) -> Tuple[int, int]:
        """Pop minimum key node from queue."""
        _, _, pos = heapq.heappop(self.U)
        return pos

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """Get valid neighbors with movement costs."""
        x, y = pos
        neighbors = []

        for i in range(8):
            nx = x + self.DIRS[i, 0]
            ny = y + self.DIRS[i, 1]

            if 0 <= nx < self.cols and 0 <= ny < self.rows:
                if not self.obstacles[ny, nx]:
                    # Base cost plus fire avoidance
                    cost = self.COSTS[i]
                    fire_intensity = self.grid[ny, nx]
                    if fire_intensity > 0:
                        cost *= (1.0 + fire_intensity * 2.0)
                    neighbors.append(((nx, ny), cost))

        return neighbors

    def _update_vertex(self, pos: Tuple[int, int]):
        """Update vertex according to D* Lite algorithm."""
        x, y = pos

        if pos != self.goal:
            # Update rhs value
            min_cost = np.inf
            for neighbor, edge_cost in self._get_neighbors(pos):
                nx, ny = neighbor
                cost = self.g[ny, nx] + edge_cost
                if cost < min_cost:
                    min_cost = cost
            self.rhs[y, x] = min_cost

        # Remove from queue if present (will be re-inserted if inconsistent)
        # Note: In practice, we check consistency when popping

        if self.g[y, x] != self.rhs[y, x]:
            self._insert(pos, self._calculate_key(pos))

    def compute_shortest_path(self):
        """Main D* Lite computation loop."""
        while self.U and (self.U[0][0] < self._calculate_key(self.start) or
                          self.rhs[self.start[1], self.start[0]] != self.g[self.start[1], self.start[0]]):

            k_old = self.U[0][0]
            u = self._pop()
            ux, uy = u

            k_new = self._calculate_key(u)

            if k_old < k_new:
                self._insert(u, k_new)
            elif self.g[uy, ux] > self.rhs[uy, ux]:
                self.g[uy, ux] = self.rhs[uy, ux]
                for neighbor, _ in self._get_neighbors(u):
                    self._update_vertex(neighbor)
            else:
                self.g[uy, ux] = np.inf
                self._update_vertex(u)
                for neighbor, _ in self._get_neighbors(u):
                    self._update_vertex(neighbor)

    def update_edge_costs(self, changed_cells: List[Tuple[int, int]]):
        """
        Update edge costs when environment changes.

        Args:
            changed_cells: List of (x, y) coordinates where grid changed
        """
        if not changed_cells:
            return

        # Update k_m for D* Lite
        self.k_m += self._heuristic(self.start, self.goal)

        # Update obstacle mask
        self._update_obstacles()

        # Update affected vertices
        affected = set()
        for cell in changed_cells:
            affected.add(cell)
            # Also add neighbors since their costs changed
            for neighbor, _ in self._get_neighbors(cell):
                affected.add(neighbor)

        for cell in affected:
            self._update_vertex(cell)

    def get_next_move(self) -> Optional[Tuple[int, int]]:
        """
        Get next move from current start position.

        Returns:
            Next position (x, y) or None if no path
        """
        # Recompute if needed
        self.compute_shortest_path()

        if self.g[self.start[1], self.start[0]] == np.inf:
            return None  # No path

        # Find neighbor with minimum g + cost
        best_neighbor = None
        best_cost = np.inf

        for neighbor, edge_cost in self._get_neighbors(self.start):
            nx, ny = neighbor
            cost = self.g[ny, nx] + edge_cost
            if cost < best_cost:
                best_cost = cost
                best_neighbor = neighbor

        return best_neighbor

    def move_start(self, new_start: Tuple[int, int]):
        """Move agent to new start position."""
        self.start = new_start


class SharedGridDStarLite:
    """
    Wrapper for multiple agents sharing a single grid.

    Optimizations:
    - Single shared grid (not copied per agent)
    - Batch environment updates
    - Spatial filtering for updates
    """

    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.agents = []  # List of OptimizedDStarLite instances

    def add_agent(self, start: Tuple[int, int], goal: Tuple[int, int]) -> OptimizedDStarLite:
        """Add new agent with D* Lite pathfinder."""
        agent = OptimizedDStarLite(self.grid, start, goal)
        self.agents.append(agent)
        return agent

    def update_environment(self, changed_cells: List[Tuple[int, int]],
                          spatial_filter: bool = True,
                          filter_radius: int = 10):
        """
        Update all agents with environment changes.

        Args:
            changed_cells: List of (x, y) where grid changed
            spatial_filter: Only update nearby agents
            filter_radius: Radius for spatial filtering
        """
        if not changed_cells:
            return

        if spatial_filter:
            # Only update agents near changes
            for agent in self.agents:
                # Check if agent is within radius of any change
                near_change = False
                for cx, cy in changed_cells:
                    dist = max(abs(agent.start[0] - cx), abs(agent.start[1] - cy))
                    if dist <= filter_radius:
                        near_change = True
                        break

                if near_change:
                    agent.update_edge_costs(changed_cells)
        else:
            # Update all agents
            for agent in self.agents:
                agent.update_edge_costs(changed_cells)
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
from optimized_d_star_lite import OptimizedDStarLite, SharedGridDStarLite
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
    - Optimized D* Lite pathfinding (maintains incremental replanning)
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

        # Initialize shared D* Lite pathfinder and fire model
        self.pathfinder_manager = SharedGridDStarLite(self.grid)
        self.agent_pathfinders = []
        for start in agent_starts:
            # Each agent gets its own D* Lite instance but shares grid
            nearest_exit = self._nearest_exit(start[0], start[1])
            pathfinder = self.pathfinder_manager.add_agent(start, nearest_exit)
            self.agent_pathfinders.append(pathfinder)

        if deterministic_fire:
            self.fire = DeterministicFireModel(self.grid)
        else:
            self.fire = FastFireModel(self.grid)

        self.fire_update_interval = fire_update_interval
        self.step_count = 0

    def _update_pathfinders(self, changed_cells: List[Tuple[int, int]]):
        """Update all pathfinders with environment changes."""
        if changed_cells:
            self.pathfinder_manager.update_environment(changed_cells, spatial_filter=True)

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
                old_grid = self.grid.copy()
                self.grid = self.fire.step()

                # Find changed cells for D* Lite updates
                changed_cells = []
                rows, cols = self.grid.shape
                for y in range(rows):
                    for x in range(cols):
                        if old_grid[y, x] != self.grid[y, x]:
                            changed_cells.append((x, y))

                self._update_pathfinders(changed_cells)

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

                # Use D* Lite to get next move
                pathfinder = self.agent_pathfinders[i]
                next_move = pathfinder.get_next_move()

                if next_move:
                    nx, ny = next_move
                    # Check if next position is safe
                    if self.grid[ny, nx] <= 0:
                        agent.x, agent.y = nx, ny
                        pathfinder.move_start((nx, ny))
                    else:
                        # Position became dangerous, D* Lite will replan
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

### 2.4 Pairwise Ranking Integration Interface

Create `pairwise_ranking_interface.py`:

```python
"""
Pairwise Ranking Integration Interface
======================================

Interface between scoring network and simulator for pairwise comparison labeling.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from fast_simulation import FastEvacuationSim, SimResult, batch_evaluate

class ScoringNetworkInterface:
    """
    Interface for scoring network to request candidate evaluations.

    Architecture:
    - Candidate Generator produces door configurations
    - Scoring Network predicts scalar scores
    - Simulator validates via Monte Carlo and generates pairwise labels
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (30, 30),
                 base_config: str = 'configs/ai_labeling_config.json',
                 num_trials_per_eval: int = 3):
        """
        Initialize interface.

        Args:
            grid_size: (rows, cols) of the floor plan
            base_config: Path to base simulation config
            num_trials_per_eval: Monte Carlo trials per candidate (k=3-5)
        """
        self.rows, self.cols = grid_size
        self.num_trials = num_trials_per_eval
        self.base_config = base_config

        # Load base config
        import json
        with open(base_config) as f:
            self.config_template = json.load(f)

    def generate_candidate_labels(self,
                                  floor_plan: np.ndarray,
                                  candidate_pool: List[Dict],
                                  num_pairs: int,
                                  pair_selection: str = 'mixed') -> List[Tuple]:
        """
        Generate pairwise comparison labels from candidate pool.

        Args:
            floor_plan: Base floor plan array
            candidate_pool: List of door configuration dicts
            num_pairs: Number of pairs to sample and label
            pair_selection: 'random', 'hard', or 'mixed' sampling strategy

        Returns:
            List of (config_A, config_B, label, score_A, score_B) tuples
        """
        # Sample pairs using specified strategy
        pairs = self._sample_pairs(candidate_pool, num_pairs, pair_selection)

        # Generate labels via simulator
        labels = []
        for config_a, config_b in pairs:
            result_a = self.evaluate_candidate(floor_plan, config_a)
            result_b = self.evaluate_candidate(floor_plan, config_b)

            # Compute simulator scores (survival rate - time penalty)
            score_a = self._compute_score(result_a)
            score_b = self._compute_score(result_b)

            # Assign pairwise label with margin
            margin = 0.05
            if score_a > score_b + margin:
                label = 1  # A > B
            elif score_b > score_a + margin:
                label = 0  # B > A
            else:
                label = None  # Ambiguous, discard

            if label is not None:
                labels.append((config_a, config_b, label, score_a, score_b))

        return labels

    def evaluate_candidate(self, floor_plan: np.ndarray, door_config: Dict) -> Dict:
        """
        Evaluate single candidate with k Monte Carlo trials.

        Args:
            floor_plan: 2D array (-2=wall, 0=empty)
            door_config: Door configuration dict

        Returns:
            Aggregated metrics dict
        """
        results = []
        for trial in range(self.num_trials):
            sim = self._build_simulation(floor_plan, door_config, seed=trial)
            result = sim.run(max_steps=200)
            results.append(result)

        # Return median statistics
        return self._aggregate_results(results)

    def batch_evaluate_topk(self,
                            floor_plan: np.ndarray,
                            candidates: List[Dict],
                            k: int) -> List[Tuple[Dict, float]]:
        """
        Evaluate top-k candidates selected by scoring network.

        Args:
            floor_plan: Base floor plan
            candidates: List of (door_config, predicted_score) tuples
            k: Number of top candidates to validate

        Returns:
            List of (door_config, simulator_score) for top-k
        """
        # Sort by predicted score and take top-k
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

        # Validate with full Monte Carlo
        validated = []
        for door_config, pred_score in sorted_candidates:
            result = self.evaluate_candidate(floor_plan, door_config)
            sim_score = self._compute_score(result)
            validated.append((door_config, sim_score))

        return validated

    def _sample_pairs(self, candidates, num_pairs, strategy):
        """Sample pairs using specified strategy."""
        import random

        pairs = []
        if strategy == 'random':
            # Pure random sampling
            for _ in range(num_pairs):
                a, b = random.sample(candidates, 2)
                pairs.append((a, b))

        elif strategy == 'hard':
            # Sample pairs with similar predicted scores (requires model predictions)
            # For now, default to random
            for _ in range(num_pairs):
                a, b = random.sample(candidates, 2)
                pairs.append((a, b))

        elif strategy == 'mixed':
            # 70% random, 30% hard pairs
            num_random = int(num_pairs * 0.7)
            num_hard = num_pairs - num_random

            for _ in range(num_random):
                a, b = random.sample(candidates, 2)
                pairs.append((a, b))

            for _ in range(num_hard):
                a, b = random.sample(candidates, 2)
                pairs.append((a, b))

        return pairs

    def _build_simulation(self, floor_plan, door_config, seed):
        """Build simulation from floor plan and door config."""
        from simulation import EvacuationSimulation, SimulationConfig
        import numpy as np

        config_dict = self.config_template.copy()
        config_dict['initial_fire_map'] = floor_plan.tolist()
        config_dict['door_configs'] = door_config

        if seed is not None:
            np.random.seed(seed)

        config = SimulationConfig.from_json(config_dict)
        return EvacuationSimulation(config)

    def _aggregate_results(self, results):
        """Aggregate trial results using median."""
        evacuated = np.median([r.evacuated for r in results])
        steps = np.median([r.steps for r in results])
        stuck = np.median([r.stuck for r in results])
        dead = np.median([r.dead for r in results])

        total = results[0].num_agents if hasattr(results[0], 'num_agents') else 1

        return {
            'evacuated': int(evacuated),
            'stuck': int(stuck),
            'dead': int(dead),
            'steps': int(steps),
            'survival_rate': evacuated / total if total > 0 else 0
        }

    def _compute_score(self, result):
        """Compute simulator score from metrics."""
        # Survival rate minus time penalty
        return result['survival_rate'] - (result['steps'] / 1000)

    def log_evaluation(self, floor_plan, door_config, model_score, sim_score):
        """Log candidate evaluation for later analysis."""
        # Log to file for correlation analysis and fine-tuning
        import json
        import time

        log_entry = {
            'timestamp': time.time(),
            'floor_plan_hash': hash(floor_plan.tobytes()),
            'door_config': door_config,
            'model_score': float(model_score),
            'sim_score': float(sim_score)
        }

        with open('candidate_evaluations.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
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

### 3.2 Scoring Network Integration Points

The scoring network (CNN + optional GNN) is trained externally using pairwise labels from the simulator. The simulator provides integration points:

```python
"""
Scoring Network Integration
============================

Integration points for external scoring network (trained on pairwise comparisons).
"""

import numpy as np
from typing import List, Dict, Tuple

class ScoringNetworkPlugin:
    """
    Plugin interface for scoring network inference.

    The scoring network is trained externally using pairwise labels
    generated by the simulator. This interface allows the network
    to be plugged into the search/inference pipeline.
    """

    def __init__(self, model_path: str = None, device: str = 'cpu'):
        """
        Initialize scoring network.

        Args:
            model_path: Path to trained model checkpoint
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load trained scoring network."""
        import torch

        # Load model architecture and weights
        # This is implemented by the ML team
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

    def score_candidate(self, floor_plan: np.ndarray, door_config: Dict) -> float:
        """
        Score a single door configuration candidate.

        Args:
            floor_plan: 2D array with walls
            door_config: Door configuration dict

        Returns:
            Scalar score (higher = better predicted performance)
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        import torch

        # Convert to model input format
        input_tensor = self._prepare_input(floor_plan, door_config)

        # Run inference
        with torch.no_grad():
            score = self.model(input_tensor)

        return float(score.item())

    def score_batch(self, floor_plan: np.ndarray, door_configs: List[Dict]) -> np.ndarray:
        """
        Score multiple candidates in batch (faster).

        Args:
            floor_plan: Base floor plan
            door_configs: List of door configurations

        Returns:
            Array of scores
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        import torch

        # Prepare batch
        inputs = torch.stack([
            self._prepare_input(floor_plan, cfg)
            for cfg in door_configs
        ])

        # Batch inference
        with torch.no_grad():
            scores = self.model(inputs)

        return scores.cpu().numpy()

    def _prepare_input(self, floor_plan, door_config):
        """Convert floor plan + door config to model input tensor."""
        import torch

        # Create 4-channel input: walls, doors, exits, connectivity
        # This format matches the training data
        rows, cols = floor_plan.shape
        input_tensor = torch.zeros((4, rows, cols), dtype=torch.float32)

        # Channel 0: Walls
        input_tensor[0] = torch.from_numpy((floor_plan == -2).astype(np.float32))

        # Channel 1: Doors
        for door in door_config:
            pos = door['position']  # "x5y3" format
            x = int(pos.split('y')[0][1:])
            y = int(pos.split('y')[1])
            input_tensor[1, y, x] = 1.0

        # Channel 2: Exits (from door types)
        for door in door_config:
            if door.get('type') == 'exit':
                pos = door['position']
                x = int(pos.split('y')[0][1:])
                y = int(pos.split('y')[1])
                input_tensor[2, y, x] = 1.0

        # Channel 3: Optional connectivity/adjacency features
        # (can be computed from door graph if using GNN fusion)

        return input_tensor.unsqueeze(0).to(self.device)


def generate_training_labels(simulator_interface,
                            floor_plans: List[np.ndarray],
                            candidates_per_plan: int = 50,
                            pairs_per_plan: int = 100,
                            output_path: str = 'pairwise_labels.jsonl') -> Dict:
    """
    Generate pairwise training labels for scoring network.

    Args:
        simulator_interface: ScoringNetworkInterface instance
        floor_plans: List of base floor plans
        candidates_per_plan: Number of door configs to generate per plan
        pairs_per_plan: Number of pairs to sample per plan
        output_path: Where to save labels

    Returns:
        Statistics about label generation
    """
    import json
    from candidate_generator import generate_door_candidates  # Implemented separately

    total_labels = 0
    ambiguous_count = 0

    with open(output_path, 'w') as f:
        for plan_idx, floor_plan in enumerate(floor_plans):
            # Generate candidate pool
            candidates = generate_door_candidates(
                floor_plan,
                num_candidates=candidates_per_plan
            )

            # Generate pairwise labels
            labels = simulator_interface.generate_candidate_labels(
                floor_plan,
                candidates,
                num_pairs=pairs_per_plan,
                pair_selection='mixed'
            )

            # Save labels
            for config_a, config_b, label, score_a, score_b in labels:
                if label is not None:
                    entry = {
                        'floor_plan_id': plan_idx,
                        'config_a': config_a,
                        'config_b': config_b,
                        'label': label,
                        'score_a': score_a,
                        'score_b': score_b
                    }
                    f.write(json.dumps(entry) + '\n')
                    total_labels += 1
                else:
                    ambiguous_count += 1

            print(f"Generated {total_labels} labels for floor plan {plan_idx + 1}/{len(floor_plans)}")

    return {
        'total_labels': total_labels,
        'ambiguous_discarded': ambiguous_count,
        'label_rate': total_labels / (total_labels + ambiguous_count)
    }
```

### 3.3 AI-Guided Search Pipeline

Create `ai_search_pipeline.py`:

```python
"""
AI-Guided Door Configuration Search
===================================

Use scoring network to narrow candidate pool before expensive simulator validation.
Reduces simulator calls by 10-100x while finding top designs.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Search result with AI and simulator metrics."""
    door_config: Dict
    model_score: float
    simulator_score: float
    rank_by_model: int
    rank_by_simulator: int
    simulator_metrics: Dict

class AIGuidedSearch:
    """
    AI-guided search for optimal door configurations.

    Pipeline:
    1. Generate large candidate pool (N_large = 1000-10000)
    2. Score all candidates with fast neural network
    3. Select top-k by model score (k = 10-50)
    4. Validate top-k with full Monte Carlo simulator
    5. Return validated top designs

    Speedup: Only k simulator calls instead of N_large
    """

    def __init__(self,
                 scoring_network,
                 simulator_interface,
                 top_k: int = 20):
        """
        Initialize search pipeline.

        Args:
            scoring_network: ScoringNetworkPlugin instance
            simulator_interface: ScoringNetworkInterface instance
            top_k: Number of candidates to validate with simulator
        """
        self.scorer = scoring_network
        self.simulator = simulator_interface
        self.top_k = top_k

    def search(self,
               floor_plan: np.ndarray,
               candidate_pool: List[Dict],
               log_results: bool = True) -> List[SearchResult]:
        """
        Search for best door configurations using AI pre-screening.

        Args:
            floor_plan: Base floor plan (walls)
            candidate_pool: Large pool of door configurations (N_large)
            log_results: Whether to log model vs simulator scores

        Returns:
            List of SearchResult objects sorted by simulator score
        """
        import time

        print(f"Searching {len(candidate_pool)} candidates with AI guidance...")

        # Step 1: Score all candidates with neural network (fast)
        start = time.time()
        model_scores = self.scorer.score_batch(floor_plan, candidate_pool)
        model_time = time.time() - start

        print(f"  Model scoring: {model_time:.2f}s ({len(candidate_pool)/model_time:.0f} candidates/sec)")

        # Step 2: Select top-k by model score
        top_k_indices = np.argsort(model_scores)[-self.top_k:][::-1]
        top_k_candidates = [(candidate_pool[i], model_scores[i]) for i in top_k_indices]

        print(f"  Selected top-{self.top_k} candidates for validation")

        # Step 3: Validate with full simulator
        start = time.time()
        validated_results = []

        for rank, (door_config, model_score) in enumerate(top_k_candidates):
            result = self.simulator.evaluate_candidate(floor_plan, door_config, num_trials=5)
            sim_score = result['survival_rate'] - result['steps'] / 1000

            search_result = SearchResult(
                door_config=door_config,
                model_score=float(model_score),
                simulator_score=float(sim_score),
                rank_by_model=rank + 1,
                rank_by_simulator=-1,  # Will be set after sorting
                simulator_metrics=result
            )
            validated_results.append(search_result)

            # Log for correlation analysis
            if log_results:
                self.simulator.log_evaluation(floor_plan, door_config, model_score, sim_score)

        sim_time = time.time() - start
        print(f"  Simulator validation: {sim_time:.2f}s ({self.top_k/sim_time:.1f} candidates/sec)")

        # Step 4: Sort by simulator score and assign ranks
        validated_results.sort(key=lambda x: x.simulator_score, reverse=True)
        for rank, result in enumerate(validated_results):
            result.rank_by_simulator = rank + 1

        # Print speedup statistics
        total_time = model_time + sim_time
        naive_time_estimate = len(candidate_pool) * (sim_time / self.top_k)
        speedup = naive_time_estimate / total_time

        print(f"\n  Total time: {total_time:.2f}s")
        print(f"  Estimated naive time: {naive_time_estimate:.1f}s")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  Sim calls saved: {len(candidate_pool) - self.top_k} ({100*(1-self.top_k/len(candidate_pool)):.1f}%)")

        return validated_results

    def evaluate_search_quality(self,
                               floor_plan: np.ndarray,
                               candidate_pool: List[Dict],
                               num_validate: int = 100) -> Dict:
        """
        Evaluate how well AI selects top candidates.

        Args:
            floor_plan: Base floor plan
            candidate_pool: Candidate pool
            num_validate: Number of random candidates to validate for comparison

        Returns:
            Quality metrics (Spearman correlation, top-k recall)
        """
        import time
        from scipy.stats import spearmanr

        # Score all with model
        model_scores = self.scorer.score_batch(floor_plan, candidate_pool)

        # Validate a random sample with simulator
        sample_indices = np.random.choice(len(candidate_pool), num_validate, replace=False)
        sim_scores = []

        for idx in sample_indices:
            result = self.simulator.evaluate_candidate(floor_plan, candidate_pool[idx], num_trials=3)
            sim_score = result['survival_rate'] - result['steps'] / 1000
            sim_scores.append(sim_score)

        sample_model_scores = model_scores[sample_indices]

        # Compute correlation
        spearman_corr, p_value = spearmanr(sample_model_scores, sim_scores)

        # Compute top-k recall
        true_top_k = set(np.argsort(sim_scores)[-self.top_k:])
        pred_top_k = set(np.argsort(sample_model_scores)[-self.top_k:])
        top_k_recall = len(true_top_k & pred_top_k) / self.top_k

        return {
            'spearman_correlation': spearman_corr,
            'p_value': p_value,
            'top_k_recall': top_k_recall,
            'num_validated': num_validate
        }
```

### Expected Results - Phase 3

| Metric | Phase 2 | Phase 3 | Improvement |
|--------|---------|---------|-------------|
| Time per simulation | 0.04-0.1s | 0.001-0.02s | 5-50x faster |
| Simulations per hour | 36,000-90,000 | 180,000-3,600,000 | 5-40x more |
| GPU batch throughput | N/A | 10,000+ per second | New capability |

---

## Implementation Roadmap

### Week 1-2: Phase 1 (Conservative - Simulator Optimization)
- [x] Create `configs/ai_labeling_config.json`
- [x] Add early termination to `simulation.py`
- [x] Create `ai_labeling_wrapper.py` with pairwise comparison methods
- [x] Benchmark: Target 6,000+ sims/hour for label generation
- [x] Implement candidate generator (random/rule-based door placement)

### Week 3-4: Phase 2 (Moderate - Fast Labeling Pipeline)
- [ ] Implement `optimized_d_star_lite.py` (optimized D* Lite maintaining incremental replanning)
- [ ] Implement `fast_fire.py` (vectorized fire model)
- [ ] Implement `fast_simulation.py` (lightweight eval with optimized D* Lite)
- [ ] Create `pairwise_ranking_interface.py`
- [ ] Generate initial training dataset (5K-10K pairs)
- [ ] Benchmark: Target 50,000+ sims/hour

### Week 5-6: Phase 3 (AI Integration - Model Training & Search)
- [ ] Train scoring network (CNN + optional GNN) on pairwise labels
  - ML team implements model architecture
  - Train with pairwise loss (logistic/hinge)
  - Monitor Spearman correlation and top-k recall
- [ ] Implement `scoring_network_plugin.py` (inference interface)
- [ ] Implement `ai_search_pipeline.py` (top-k selection + validation)
- [ ] Benchmark: Target 10-100x speedup via AI pre-screening

### Week 7-8: Evaluation & Dashboard
- [ ] Implement evaluation metrics (Spearman, top-k recall, sim calls saved)
- [ ] Create dashboard with:
  - Scatter plot: model score vs simulator score
  - Ranking metrics visualization
  - Case studies: random vs AI-selected layouts
- [ ] Generate "AI suggested" designs for demo
- [ ] Production deployment & documentation

---

## Quick Start

### Immediate (Today) - Simulator Optimization
```bash
# Use optimized config for fast labeling
cp configs/ai_labeling_config.json configs/my_config.json
python simulation.py --config configs/my_config.json
```

### Week 1-2 - Generate Pairwise Labels
```python
# Generate pairwise comparison labels for training
from ai_labeling_wrapper import AILabelingWrapper
from candidate_generator import generate_door_candidates  # To be implemented

# Initialize wrapper
labeler = AILabelingWrapper(base_config_path='configs/ai_labeling_config.json')

# Generate candidates for a floor plan
floor_plan = ...  # 2D numpy array
candidates = generate_door_candidates(floor_plan, num_candidates=100)

# Generate pairwise labels
pairs = [(candidates[i], candidates[j]) for i in range(0, 100, 2) for j in range(i+1, min(i+10, 100))]
labels = labeler.generate_pairwise_labels(floor_plan, pairs, num_trials=3, margin=0.05)

# Labels format: (config_a, config_b, label, score_a, score_b)
print(f"Generated {len([l for l in labels if l[2] is not None])} valid labels")
```

### Week 3-4 - Fast Labeling at Scale
```python
# Generate large training dataset efficiently
from pairwise_ranking_interface import ScoringNetworkInterface

interface = ScoringNetworkInterface(
    grid_size=(30, 30),
    base_config='configs/ai_labeling_config.json',
    num_trials_per_eval=3
)

# Generate many pairwise labels across multiple floor plans
floor_plans = [...]  # List of floor plan arrays
labels = generate_training_labels(
    interface,
    floor_plans,
    candidates_per_plan=50,
    pairs_per_plan=100,
    output_path='training_labels.jsonl'
)
print(f"Generated {labels['total_labels']} pairwise labels")
```

### Week 5+ - AI-Guided Search
```python
# Use trained scoring network to accelerate search
from scoring_network_plugin import ScoringNetworkPlugin
from ai_search_pipeline import AIGuidedSearch
from pairwise_ranking_interface import ScoringNetworkInterface

# Load trained model
scorer = ScoringNetworkPlugin(model_path='models/scoring_network.pt')
simulator = ScoringNetworkInterface()

# Create search pipeline
search = AIGuidedSearch(
    scoring_network=scorer,
    simulator_interface=simulator,
    top_k=20  # Only validate top-20 by model score
)

# Search for best door configuration
floor_plan = ...
candidates = generate_door_candidates(floor_plan, num_candidates=1000)
results = search.search(floor_plan, candidates)

# Best design found
best = results[0]
print(f"Best design: {best.door_config}")
print(f"Simulator score: {best.simulator_score:.3f}")
print(f"Validated only {len(results)} out of {len(candidates)} candidates")
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
A: Start with Phase 1 immediately (1-2 days). The config changes provide 3-5x speedup for label generation with zero architectural changes. This lets you start collecting pairwise labels right away.

**Q: How is this different from RL?**
A: This uses **pairwise ranking** instead of RL. The simulator generates ground-truth labels comparing pairs of designs (A > B?), and a CNN learns to predict relative quality. This is more robust to noisy simulator outputs and requires less training data than RL policy learning.

**Q: How much training data is needed?**
A: Start with 5K-10K pairwise comparisons across 10-20 floor plans. With proper data augmentation and mixed sampling (random + hard pairs), this provides good Spearman correlation (0.7-0.9). More data improves correlation but yields diminishing returns.

**Q: How accurate is the scoring network?**
A: With 10K pairwise labels, expect Spearman correlation of 0.7-0.85 between model scores and simulator scores. Top-k recall (finding true top designs) is typically 60-80% for k=20. This is sufficient for 10-50x search speedup.

**Q: Can I use GPU?**
A: Yes, the scoring network uses PyTorch and runs efficiently on GPU. Batch inference on 1000 candidates takes <1 second on GPU vs 2-3 seconds on CPU. The simulator still runs on CPU but only validates top-k candidates.

**Q: What about determinism for reproducibility?**
A: Use fixed random seeds for both training data generation and inference. The simulator with `DeterministicFireModel` and seeded numpy ensures reproducible labels. The trained network is deterministic at inference time.

**Q: How do I know if the AI is working?**
A: Monitor these metrics:
- **Spearman correlation**: Should be >0.7 after training on 10K pairs
- **Top-k recall**: Fraction of true top-20 designs found in model's top-20 (target: >60%)
- **Simulation calls saved**: Should reduce calls by 90-95% (only validate top-k of N_large)
- **Case studies**: Visual comparison of random vs AI-selected designs
