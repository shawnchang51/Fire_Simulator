"""
Lightweight Simulation Wrapper for RL Training
===============================================

Zero-overhead simulation for RL training with:
- No visualization
- No I/O operations
- Early termination for bad designs
- Clean numpy-based interface
"""

from simulation import EvacuationSimulation, SimulationConfig
import numpy as np
import json


class RLSimulationWrapper:
    """Zero-overhead simulation for RL training."""

    def __init__(self, base_config_path='configs/rl_training_config.json'):
        """
        Initialize wrapper with base configuration.

        Args:
            base_config_path: Path to base JSON configuration file
        """
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = json.load(f)

    def evaluate(self, floor_plan: np.ndarray = None,
                 agent_positions: list = None,
                 exit_positions: list = None,
                 fire_positions: list = None,
                 max_steps: int = 200,
                 config_overrides: dict = None,
                 use_door_graph: bool = False) -> dict:
        """
        Evaluate a floor plan design.

        Args:
            floor_plan: 2D numpy array (-2=wall, 0=empty). If None, uses config.
            agent_positions: List of (x, y) tuples for agent starts. If None, uses config.
            exit_positions: List of (x, y) tuples for exits. If None, uses config targets.
            fire_positions: Optional list of (x, y) tuples for fire starts
            max_steps: Maximum simulation steps (default: 200)
            config_overrides: Optional dict to override config parameters

        Returns:
            Dict with evacuation metrics and reward:
                - steps: Simulation steps taken
                - termination_reason: Why simulation ended
                - evacuated_agents: Number evacuated
                - stuck_count: Number stuck
                - dead_count: Number dead
                - survival_rate: Fraction evacuated
                - reward: RL reward signal
        """
        config = self._build_config(floor_plan, agent_positions,
                                    exit_positions, fire_positions,
                                    config_overrides, use_door_graph)

        sim = EvacuationSimulation(config, silent=True)

        # Run with all overhead disabled
        result = sim.run(
            max_steps=max_steps,
            show_visualization=False,
            use_pygame=False,
            use_matlab=False,
            early_termination=True,
            stuck_threshold=0.5,
            death_threshold=0.3
        )

        return result

    def _build_config(self, floor_plan, agents, exits, fires, config_overrides, use_door_graph=False):
        """Build SimulationConfig from numpy arrays and positions."""
        config_dict = self.base_config.copy()

        # Disable door graph for RL training (use simple D* Lite only)
        if not use_door_graph:
            config_dict['door_configs'] = None

        # Apply any config overrides first
        if config_overrides:
            config_dict.update(config_overrides)

        # Update with provided parameters
        if floor_plan is not None:
            rows, cols = floor_plan.shape
            config_dict['map_rows'] = rows
            config_dict['map_cols'] = cols

            # Convert numpy to list for fire map
            fire_map = floor_plan.tolist()
            if fires:
                for x, y in fires:
                    if 0 <= y < rows and 0 <= x < cols:
                        fire_map[y][x] = 2.0
            config_dict['initial_fire_map'] = fire_map
        else:
            # Just add fires to existing map
            if fires:
                rows = config_dict['map_rows']
                cols = config_dict['map_cols']
                fire_map = config_dict.get('initial_fire_map',
                                          [[0] * cols for _ in range(rows)])
                for x, y in fires:
                    if 0 <= y < rows and 0 <= x < cols:
                        fire_map[y][x] = 2.0
                config_dict['initial_fire_map'] = fire_map

        if agents is not None:
            config_dict['agent_num'] = len(agents)
            config_dict['start_positions'] = [f'x{x}y{y}' for x, y in agents]

        if exits is not None:
            config_dict['targets'] = [f'x{x}y{y}' for x, y in exits]

        return SimulationConfig.from_json(config_dict)

    def batch_evaluate(self, scenarios: list, num_workers: int = None) -> list:
        """
        Evaluate multiple floor plans in parallel.

        Args:
            scenarios: List of dicts with keys:
                - floor_plan: np.ndarray (optional)
                - agent_positions: List[Tuple] (optional)
                - exit_positions: List[Tuple] (optional)
                - fire_positions: List[Tuple] (optional)
                - max_steps: int (optional)
                - config_overrides: dict (optional)
            num_workers: Number of parallel workers (default: CPU count)

        Returns:
            List of result dicts
        """
        from multiprocessing import Pool, cpu_count

        if num_workers is None:
            num_workers = cpu_count()

        with Pool(num_workers) as pool:
            results = pool.map(self._evaluate_single, scenarios)

        return results

    def _evaluate_single(self, scenario):
        """Single evaluation for multiprocessing."""
        return self.evaluate(
            floor_plan=scenario.get('floor_plan'),
            agent_positions=scenario.get('agent_positions'),
            exit_positions=scenario.get('exit_positions'),
            fire_positions=scenario.get('fire_positions'),
            max_steps=scenario.get('max_steps', 200),
            config_overrides=scenario.get('config_overrides')
        )


# Convenience function for quick evaluation
def evaluate_floor_plan(floor_plan: np.ndarray,
                       agent_positions: list,
                       exit_positions: list,
                       fire_positions: list = None,
                       max_steps: int = 200,
                       config_path: str = 'configs/rl_training_config.json') -> dict:
    """
    Quick floor plan evaluation function.

    Args:
        floor_plan: 2D numpy array (-2=wall, 0=empty)
        agent_positions: List of (x, y) tuples for agent starts
        exit_positions: List of (x, y) tuples for exits
        fire_positions: Optional list of (x, y) tuples for fire starts
        max_steps: Maximum simulation steps
        config_path: Path to base configuration file

    Returns:
        Dict with evacuation metrics and reward
    """
    wrapper = RLSimulationWrapper(config_path)
    return wrapper.evaluate(floor_plan, agent_positions, exit_positions,
                           fire_positions, max_steps)


if __name__ == "__main__":
    # Example usage
    print("RL Simulation Wrapper - Example Usage")
    print("=" * 50)

    # Create a simple 30x30 floor plan
    floor_plan = np.zeros((30, 30), dtype=np.float32)

    # Add some walls (creating a room)
    floor_plan[10:20, 10] = -2  # Left wall
    floor_plan[10:20, 20] = -2  # Right wall
    floor_plan[10, 10:21] = -2  # Top wall
    floor_plan[20, 10:21] = -2  # Bottom wall
    floor_plan[15, 10] = 0      # Door in left wall

    # Define agents, exits, and fire
    agents = [(12, 12), (15, 15), (18, 18)]
    exits = [(29, 15)]
    fires = [(14, 14)]

    # Evaluate the floor plan
    wrapper = RLSimulationWrapper()
    result = wrapper.evaluate(floor_plan, agents, exits, fires, max_steps=200)

    print(f"\nResults:")
    print(f"  Steps: {result['steps']}")
    print(f"  Termination: {result.get('termination_reason', 'N/A')}")
    print(f"  Evacuated: {result['evacuated_agents']}/{len(agents)}")
    print(f"  Stuck: {result['stuck_count']}")
    print(f"  Dead: {result['dead_count']}")
    print(f"  Survival Rate: {result['survival_rate']:.1%}")
    print(f"  Reward: {result.get('reward', 'N/A'):.2f}")

    print("\nExample: Batch evaluation")
    print("-" * 50)

    # Create multiple scenarios
    scenarios = []
    for i in range(5):
        scenario = {
            'agent_positions': [(5, 5), (10, 10), (15, 15)],
            'exit_positions': [(29, 15)],
            'fire_positions': [(10 + i, 10 + i)],  # Different fire positions
            'max_steps': 200
        }
        scenarios.append(scenario)

    # Evaluate in parallel
    results = wrapper.batch_evaluate(scenarios, num_workers=4)

    print(f"\nEvaluated {len(results)} scenarios")
    for i, result in enumerate(results):
        print(f"  Scenario {i+1}: Reward={result.get('reward', 'N/A'):.2f}, "
              f"Evacuated={result['evacuated_agents']}, "
              f"Steps={result['steps']}")
