"""
Diversity Sampler for Training Data Generation

Ensures comprehensive coverage across all diversity dimensions:
- Floor plan sizes
- Agent counts and distributions
- Fire scenarios
- Exit configurations

Uses stratified sampling to guarantee representation of edge cases
and maintain balance across the parameter space.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Iterator
from dataclasses import dataclass, field
from collections import defaultdict
import itertools


@dataclass
class ScenarioConfig:
    """Configuration for a single simulation scenario"""
    floor_plan_id: int
    agent_count: int
    agent_distribution: str  # 'uniform', 'clustered', 'room_based'
    fire_count: int
    fire_positions: List[Tuple[int, int]]  # Will be sampled
    fire_spread_rate: float
    fire_discovery_delay: int
    exit_count: int
    exit_placement: str  # 'distributed', 'corners', 'random'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'floor_plan_id': self.floor_plan_id,
            'agent_count': self.agent_count,
            'agent_distribution': self.agent_distribution,
            'fire_count': self.fire_count,
            'fire_positions': self.fire_positions,
            'fire_spread_rate': self.fire_spread_rate,
            'fire_discovery_delay': self.fire_discovery_delay,
            'exit_count': self.exit_count,
            'exit_placement': self.exit_placement
        }


@dataclass
class DiversityBins:
    """Defines bins for each diversity dimension"""
    # Floor plan size bins (by total cells)
    size_bins: List[Tuple[int, int]] = field(default_factory=lambda: [
        (400, 900),      # 20x20 to 30x30 (small)
        (900, 1600),     # 30x30 to 40x40 (medium-small)
        (1600, 2500),    # 40x40 to 50x50 (medium)
        (2500, 4900),    # 50x50 to 70x70 (large)
        (4900, 6400),    # 70x70 to 80x80 (very large)
    ])

    # Agent count bins
    agent_bins: List[Tuple[int, int]] = field(default_factory=lambda: [
        (10, 25),        # sparse
        (25, 50),        # light
        (50, 100),       # moderate
        (100, 150),      # dense
        (150, 200),      # very dense
    ])

    # Fire count options
    fire_counts: List[int] = field(default_factory=lambda: [1, 2, 3])

    # Fire spread rate bins
    spread_rates: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.6])

    # Exit count options
    exit_counts: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    # Agent distribution strategies
    agent_distributions: List[str] = field(default_factory=lambda: [
        'uniform', 'clustered', 'room_based'
    ])

    # Fire discovery delay bins
    discovery_delays: List[int] = field(default_factory=lambda: [0, 5, 10, 20])


class DiversitySampler:
    """
    Samples simulation configurations to ensure diversity coverage.

    Key features:
    - Stratified sampling across all dimensions
    - Tracks coverage to identify gaps
    - Generates balanced scenario batches
    """

    def __init__(
        self,
        bins: Optional[DiversityBins] = None,
        seed: Optional[int] = None
    ):
        self.bins = bins or DiversityBins()
        self.rng = np.random.default_rng(seed)

        # Track coverage
        self.coverage: Dict[str, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))

    def sample_scenarios_for_plan(
        self,
        floor_plan_id: int,
        floor_plan_size: Tuple[int, int],
        num_scenarios: int = 20,
        passable_cells: int = 0
    ) -> List[ScenarioConfig]:
        """
        Sample diverse scenarios for a single floor plan.

        Args:
            floor_plan_id: ID of the floor plan
            floor_plan_size: (rows, cols) of the floor plan
            num_scenarios: Number of scenarios to generate
            passable_cells: Number of passable cells (for density calculations)

        Returns:
            List of ScenarioConfig objects
        """
        rows, cols = floor_plan_size
        total_cells = rows * cols

        if passable_cells == 0:
            passable_cells = int(total_cells * 0.6)  # Estimate

        scenarios = []

        # Determine appropriate agent range based on floor plan size
        max_agents = min(200, int(passable_cells * 0.15))  # Max 15% density
        min_agents = max(10, int(passable_cells * 0.01))   # Min 1% density

        for i in range(num_scenarios):
            # Sample each dimension
            agent_count = self._sample_from_range(
                min_agents, max_agents,
                self._get_underrepresented_bin('agent_count', self.bins.agent_bins)
            )

            agent_dist = self._sample_underrepresented(
                'agent_distribution',
                self.bins.agent_distributions
            )

            fire_count = self._sample_underrepresented(
                'fire_count',
                self.bins.fire_counts
            )

            spread_rate = self._sample_underrepresented(
                'spread_rate',
                self.bins.spread_rates
            )

            discovery_delay = self._sample_underrepresented(
                'discovery_delay',
                self.bins.discovery_delays
            )

            exit_count = self._sample_underrepresented(
                'exit_count',
                self.bins.exit_counts
            )

            exit_placement = self.rng.choice(['distributed', 'corners', 'random'])

            scenario = ScenarioConfig(
                floor_plan_id=floor_plan_id,
                agent_count=agent_count,
                agent_distribution=agent_dist,
                fire_count=fire_count,
                fire_positions=[],  # To be filled when placing fires
                fire_spread_rate=spread_rate,
                fire_discovery_delay=discovery_delay,
                exit_count=exit_count,
                exit_placement=exit_placement
            )

            scenarios.append(scenario)

            # Update coverage
            self._update_coverage(scenario, total_cells)

        return scenarios

    def sample_stratified_batch(
        self,
        floor_plans: List[Tuple[int, Tuple[int, int], int]],  # (id, size, passable_cells)
        total_scenarios: int
    ) -> List[ScenarioConfig]:
        """
        Sample scenarios across multiple floor plans with stratification.

        Ensures even coverage across:
        - Floor plan sizes
        - All other dimensions

        Args:
            floor_plans: List of (floor_plan_id, (rows, cols), passable_cells)
            total_scenarios: Total number of scenarios to generate

        Returns:
            List of ScenarioConfig objects
        """
        # Group floor plans by size bin
        size_groups: Dict[int, List] = defaultdict(list)
        for plan_id, size, passable in floor_plans:
            total = size[0] * size[1]
            bin_idx = self._get_size_bin_idx(total)
            size_groups[bin_idx].append((plan_id, size, passable))

        # Distribute scenarios across size bins
        scenarios_per_bin = total_scenarios // len(size_groups)
        remainder = total_scenarios % len(size_groups)

        all_scenarios = []

        for bin_idx, plans in size_groups.items():
            n_scenarios = scenarios_per_bin + (1 if bin_idx < remainder else 0)
            scenarios_per_plan = max(1, n_scenarios // len(plans))

            for plan_id, size, passable in plans:
                plan_scenarios = self.sample_scenarios_for_plan(
                    plan_id, size, scenarios_per_plan, passable
                )
                all_scenarios.extend(plan_scenarios)

        return all_scenarios

    def get_coverage_report(self) -> Dict[str, Dict[str, float]]:
        """
        Get coverage statistics for each dimension.

        Returns:
            Dict mapping dimension names to coverage percentages
        """
        report = {}

        for dim_name, counts in self.coverage.items():
            total = sum(counts.values())
            if total == 0:
                continue

            dim_report = {}
            for value, count in counts.items():
                dim_report[str(value)] = count / total

            report[dim_name] = dim_report

        return report

    def get_underrepresented_dimensions(self, threshold: float = 0.1) -> List[str]:
        """
        Identify dimensions with underrepresented bins.

        Args:
            threshold: Minimum expected proportion (e.g., 0.1 for 10%)

        Returns:
            List of dimension names with coverage issues
        """
        issues = []
        report = self.get_coverage_report()

        for dim_name, proportions in report.items():
            min_prop = min(proportions.values()) if proportions else 0
            if min_prop < threshold:
                issues.append(dim_name)

        return issues

    def _sample_from_range(
        self,
        min_val: int,
        max_val: int,
        target_bin: Optional[Tuple[int, int]] = None
    ) -> int:
        """Sample a value from range, optionally targeting a specific bin"""
        if target_bin is not None:
            # Intersect target bin with valid range
            low = max(min_val, target_bin[0])
            high = min(max_val, target_bin[1])
            if low < high:
                return int(self.rng.integers(low, high))

        return int(self.rng.integers(min_val, max_val))

    def _sample_underrepresented(
        self,
        dimension: str,
        options: List[Any]
    ) -> Any:
        """Sample from options, biasing toward underrepresented values"""
        counts = self.coverage[dimension]

        if not counts:
            # No data yet, sample uniformly
            return self.rng.choice(options)

        # Calculate inverse frequency weights
        total = sum(counts.values()) + len(options)  # Add smoothing
        weights = []
        for opt in options:
            count = counts.get(opt, 0) + 1  # Smoothing
            weights.append(1.0 / count)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        return self.rng.choice(options, p=weights)

    def _get_underrepresented_bin(
        self,
        dimension: str,
        bins: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """Get the most underrepresented bin for a dimension"""
        counts = self.coverage[dimension]

        if not counts:
            return self.rng.choice(bins)

        # Find bin with lowest count
        bin_counts = []
        for bin_range in bins:
            bin_key = f"{bin_range[0]}-{bin_range[1]}"
            bin_counts.append((bin_range, counts.get(bin_key, 0)))

        # Sort by count and pick from bottom half randomly
        bin_counts.sort(key=lambda x: x[1])
        bottom_half = bin_counts[:max(1, len(bin_counts) // 2)]

        return self.rng.choice([b[0] for b in bottom_half])

    def _get_size_bin_idx(self, total_cells: int) -> int:
        """Get the bin index for a floor plan size"""
        for idx, (low, high) in enumerate(self.bins.size_bins):
            if low <= total_cells < high:
                return idx
        return len(self.bins.size_bins) - 1

    def _update_coverage(self, scenario: ScenarioConfig, total_cells: int):
        """Update coverage tracking with new scenario"""
        # Size bin
        size_bin_idx = self._get_size_bin_idx(total_cells)
        size_range = self.bins.size_bins[size_bin_idx]
        self.coverage['size'][f"{size_range[0]}-{size_range[1]}"] += 1

        # Agent count bin
        for low, high in self.bins.agent_bins:
            if low <= scenario.agent_count < high:
                self.coverage['agent_count'][f"{low}-{high}"] += 1
                break

        # Other dimensions
        self.coverage['agent_distribution'][scenario.agent_distribution] += 1
        self.coverage['fire_count'][scenario.fire_count] += 1
        self.coverage['spread_rate'][scenario.fire_spread_rate] += 1
        self.coverage['discovery_delay'][scenario.fire_discovery_delay] += 1
        self.coverage['exit_count'][scenario.exit_count] += 1


class AgentPlacer:
    """Places agents in floor plans according to different strategies"""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def place_agents(
        self,
        grid: np.ndarray,
        num_agents: int,
        strategy: str = 'uniform',
        room_centers: Optional[List[Tuple[int, int]]] = None
    ) -> List[Tuple[int, int]]:
        """
        Place agents in the floor plan.

        Args:
            grid: Floor plan grid (-2 = wall, 0 = passable)
            num_agents: Number of agents to place
            strategy: 'uniform', 'clustered', or 'room_based'
            room_centers: List of room center positions (for room_based)

        Returns:
            List of (col, row) positions for agents
        """
        # Find all passable positions
        passable = np.argwhere(grid == 0)
        if len(passable) == 0:
            return []

        num_agents = min(num_agents, len(passable))

        if strategy == 'uniform':
            return self._place_uniform(passable, num_agents)
        elif strategy == 'clustered':
            return self._place_clustered(passable, num_agents)
        elif strategy == 'room_based' and room_centers:
            return self._place_room_based(passable, num_agents, room_centers)
        else:
            return self._place_uniform(passable, num_agents)

    def _place_uniform(
        self,
        passable: np.ndarray,
        num_agents: int
    ) -> List[Tuple[int, int]]:
        """Uniform random placement"""
        indices = self.rng.choice(len(passable), size=num_agents, replace=False)
        positions = passable[indices]
        return [(int(pos[1]), int(pos[0])) for pos in positions]  # (col, row)

    def _place_clustered(
        self,
        passable: np.ndarray,
        num_agents: int
    ) -> List[Tuple[int, int]]:
        """Place agents in clusters"""
        num_clusters = self.rng.integers(2, max(3, num_agents // 10))
        agents_per_cluster = num_agents // num_clusters

        positions = []

        # Select cluster centers
        center_indices = self.rng.choice(len(passable), size=num_clusters, replace=False)
        centers = passable[center_indices]

        for i, center in enumerate(centers):
            n = agents_per_cluster if i < num_clusters - 1 else num_agents - len(positions)

            # Find positions near this center
            distances = np.linalg.norm(passable - center, axis=1)
            nearby_indices = np.argsort(distances)[:n * 3]  # Get more than needed

            # Sample from nearby positions
            if len(nearby_indices) >= n:
                selected = self.rng.choice(nearby_indices, size=n, replace=False)
            else:
                selected = nearby_indices

            for idx in selected:
                pos = passable[idx]
                positions.append((int(pos[1]), int(pos[0])))

            if len(positions) >= num_agents:
                break

        return positions[:num_agents]

    def _place_room_based(
        self,
        passable: np.ndarray,
        num_agents: int,
        room_centers: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Distribute agents across rooms"""
        if not room_centers:
            return self._place_uniform(passable, num_agents)

        agents_per_room = num_agents // len(room_centers)
        positions = []

        for i, center in enumerate(room_centers):
            n = agents_per_room if i < len(room_centers) - 1 else num_agents - len(positions)

            # Find positions near this room center
            center_arr = np.array([center[1], center[0]])  # (row, col)
            distances = np.linalg.norm(passable - center_arr, axis=1)
            nearby_indices = np.argsort(distances)[:n * 2]

            if len(nearby_indices) >= n:
                selected = self.rng.choice(nearby_indices, size=n, replace=False)
            else:
                selected = nearby_indices

            for idx in selected:
                pos = passable[idx]
                positions.append((int(pos[1]), int(pos[0])))

        return positions[:num_agents]


class FirePlacer:
    """Places fire starting positions in floor plans"""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def place_fires(
        self,
        grid: np.ndarray,
        num_fires: int,
        exit_positions: List[Tuple[int, int]],
        strategy: str = 'varied'
    ) -> List[Tuple[int, int]]:
        """
        Place fire starting positions.

        Args:
            grid: Floor plan grid
            num_fires: Number of fire sources
            exit_positions: List of exit positions to consider blocking
            strategy: 'random', 'blocking', 'center', 'varied'

        Returns:
            List of (col, row) fire positions
        """
        passable = np.argwhere(grid == 0)
        if len(passable) == 0:
            return []

        if strategy == 'varied':
            strategy = self.rng.choice(['random', 'blocking', 'center'])

        if strategy == 'blocking' and exit_positions:
            return self._place_blocking(passable, num_fires, exit_positions)
        elif strategy == 'center':
            return self._place_center(passable, num_fires, grid.shape)
        else:
            return self._place_random(passable, num_fires)

    def _place_random(
        self,
        passable: np.ndarray,
        num_fires: int
    ) -> List[Tuple[int, int]]:
        """Random fire placement"""
        num_fires = min(num_fires, len(passable))
        indices = self.rng.choice(len(passable), size=num_fires, replace=False)
        positions = passable[indices]
        return [(int(pos[1]), int(pos[0])) for pos in positions]

    def _place_blocking(
        self,
        passable: np.ndarray,
        num_fires: int,
        exit_positions: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Place fires to potentially block exits"""
        positions = []

        for i in range(min(num_fires, len(exit_positions))):
            exit_pos = exit_positions[i % len(exit_positions)]
            exit_arr = np.array([exit_pos[1], exit_pos[0]])  # (row, col)

            # Find positions near but not at exit
            distances = np.linalg.norm(passable - exit_arr, axis=1)

            # Target positions 3-10 cells from exit
            valid_mask = (distances > 3) & (distances < 15)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) > 0:
                idx = self.rng.choice(valid_indices)
                pos = passable[idx]
                positions.append((int(pos[1]), int(pos[0])))

        # Fill remaining with random
        while len(positions) < num_fires:
            idx = self.rng.integers(len(passable))
            pos = passable[idx]
            new_pos = (int(pos[1]), int(pos[0]))
            if new_pos not in positions:
                positions.append(new_pos)

        return positions

    def _place_center(
        self,
        passable: np.ndarray,
        num_fires: int,
        grid_shape: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Place fires near center of the floor plan"""
        center = np.array([grid_shape[0] // 2, grid_shape[1] // 2])
        distances = np.linalg.norm(passable - center, axis=1)

        # Get positions in center third
        max_dist = min(grid_shape) // 3
        center_mask = distances < max_dist
        center_indices = np.where(center_mask)[0]

        if len(center_indices) >= num_fires:
            indices = self.rng.choice(center_indices, size=num_fires, replace=False)
        else:
            indices = self.rng.choice(len(passable), size=num_fires, replace=False)

        positions = passable[indices]
        return [(int(pos[1]), int(pos[0])) for pos in positions]


if __name__ == '__main__':
    # Test diversity sampler
    print("Testing DiversitySampler...")

    sampler = DiversitySampler(seed=42)

    # Simulate sampling for multiple floor plans
    test_plans = [
        (0, (30, 30), 600),
        (1, (40, 40), 1000),
        (2, (50, 50), 1500),
        (3, (60, 60), 2000),
        (4, (70, 70), 2800),
    ]

    all_scenarios = sampler.sample_stratified_batch(test_plans, total_scenarios=50)

    print(f"\nGenerated {len(all_scenarios)} scenarios")

    # Show coverage report
    report = sampler.get_coverage_report()
    print("\nCoverage Report:")
    for dim, proportions in report.items():
        print(f"\n  {dim}:")
        for value, prop in proportions.items():
            print(f"    {value}: {prop:.1%}")

    # Check for issues
    issues = sampler.get_underrepresented_dimensions(threshold=0.05)
    if issues:
        print(f"\nUnderrepresented dimensions: {issues}")
    else:
        print("\nAll dimensions have good coverage!")
