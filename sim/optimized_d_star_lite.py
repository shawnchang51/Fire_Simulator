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

        # Priority queue: list of (key, counter, (x, y))
        self.U = []
        self.counter = 0  # Tie-breaker

        # Version tracking to ignore stale queue entries
        self.version = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.current_version = 0

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
        """Insert node into priority queue with version tracking."""
        self.counter += 1
        self.current_version += 1
        x, y = pos
        self.version[y, x] = self.current_version
        heapq.heappush(self.U, (key, self.counter, pos, self.current_version))

    def _pop(self) -> Tuple[Tuple[int, int], int]:
        """Pop minimum key node from queue, returning position and version."""
        _, _, pos, version = heapq.heappop(self.U)
        return pos, version

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

    def compute_shortest_path(self, max_iterations=10000):
        """Main D* Lite computation loop with safety limit."""
        iterations = 0
        while (self.U and
               iterations < max_iterations and
               (self.U[0][0] < self._calculate_key(self.start) or
                self.rhs[self.start[1], self.start[0]] != self.g[self.start[1], self.start[0]])):

            k_old = self.U[0][0]
            u, u_version = self._pop()
            ux, uy = u

            # Skip stale entries
            if u_version != self.version[uy, ux]:
                continue

            iterations += 1
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

        if iterations >= max_iterations:
            # Safety limit reached - likely no path exists
            pass

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
