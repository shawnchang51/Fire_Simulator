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
from fast_fire import FastFireModel, DeterministicFireModel, FireSpreadMode

@dataclass
class FastAgent:
    """Minimal agent state."""
    __slots__ = ['x', 'y', 'status', 'steps', 'fire_damage']
    x: int
    y: int
    status: str  # 'active', 'evacuated', 'stuck', 'dead'
    steps: int
    fire_damage: float  # Accumulated fire exposure

@dataclass
class SimResult:
    """Simulation result for RL."""
    __slots__ = ['steps', 'evacuated', 'stuck', 'dead', 'survival_rate',
                 'avg_evacuation_time', 'reward', 'termination_reason',
                 'avg_fire_damage', 'agent_fire_exposures']
    steps: int
    evacuated: int
    stuck: int
    dead: int
    survival_rate: float
    avg_evacuation_time: float
    reward: float
    termination_reason: str
    avg_fire_damage: float
    agent_fire_exposures: List[float]  # Fire exposure for each agent


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
                 fire_update_interval: int = 4,
                 fire_discovery_delay: int = 0,
                 fire_spread_mode: str = 'always_real'):
        """
        Initialize simulation.

        Args:
            grid: 2D array (-2=wall, 0=empty)
            agent_starts: List of (x, y) agent starting positions
            exits: List of (x, y) exit positions
            fire_starts: Optional list of (x, y) initial fire positions
            deterministic_fire: Use deterministic fire spread (legacy, overridden by fire_spread_mode)
            fire_update_interval: Steps between fire updates
            fire_discovery_delay: Steps of fire-only propagation before agents start moving (discovery time)
            fire_spread_mode: Fire spread behavior - 'always_real', 'real_then_simple', or 'real_then_stop'
        """
        # Initialize grid with fire
        self.grid = grid.astype(np.float32)
        if fire_starts:
            for x, y in fire_starts:
                if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                    self.grid[y, x] = 2.0

        # Initialize agents with fire damage tracking
        self.agents = [FastAgent(x, y, 'active', 0, 0.0) for x, y in agent_starts]
        self.exits = set(exits)
        self.exit_list = list(exits)

        # Initialize shared D* Lite pathfinder
        self.pathfinder_manager = SharedGridDStarLite(self.grid)
        self.agent_pathfinders = []
        for start in agent_starts:
            # Each agent gets its own D* Lite instance but shares grid
            nearest_exit = self._nearest_exit(start[0], start[1])
            pathfinder = self.pathfinder_manager.add_agent(start, nearest_exit)
            # IMPORTANT: Compute initial path before simulation starts
            pathfinder.compute_shortest_path()
            self.agent_pathfinders.append(pathfinder)

        # Initialize fire model with spread mode support
        # Convert string to enum
        if isinstance(fire_spread_mode, str):
            fire_spread_mode = FireSpreadMode(fire_spread_mode)

        if deterministic_fire and fire_spread_mode == FireSpreadMode.ALWAYS_REAL:
            # Legacy behavior: deterministic_fire=True uses DeterministicFireModel
            self.fire = DeterministicFireModel(self.grid)
        else:
            # New behavior: use FastFireModel with spread mode
            self.fire = FastFireModel(self.grid, spread_mode=fire_spread_mode)

        self.fire_update_interval = fire_update_interval
        self.fire_discovery_delay = fire_discovery_delay
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

            # Only move agents after fire discovery delay has passed
            active_count = 0
            if step >= self.fire_discovery_delay:
                # Move agents
                for i, agent in enumerate(self.agents):
                    if agent.status != 'active':
                        continue

                    active_count += 1
                    agent.steps += 1

                    # Track fire damage at current position
                    fire_intensity = max(0, self.grid[agent.y, agent.x])
                    if fire_intensity > 0:
                        agent.fire_damage += fire_intensity

                    # Check if at exit
                    if (agent.x, agent.y) in self.exits:
                        agent.status = 'evacuated'
                        evacuation_times.append(agent.steps)
                        continue

                    # Check if in intense fire (death threshold)
                    if fire_intensity > 3.0:
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
            else:
                # Fire discovery delay: fire spreads but agents don't move
                # Still count active agents for early termination checks
                active_count = sum(1 for a in self.agents if a.status == 'active')

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

        # Survival = evacuated + stuck (not dead)
        survived = evacuated + stuck
        survival_rate = survived / total if total > 0 else 0

        avg_time = np.mean(evacuation_times) if evacuation_times else self.step_count

        # Calculate average fire damage across all agents
        fire_exposures = [a.fire_damage for a in self.agents]
        avg_fire_damage = np.mean(fire_exposures) if fire_exposures else 0.0

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
            termination_reason=reason,
            avg_fire_damage=avg_fire_damage,
            agent_fire_exposures=fire_exposures
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
