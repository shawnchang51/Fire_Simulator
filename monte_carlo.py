"""
Monte Carlo Evacuation Simulation
==================================

This module provides both serial and parallel Monte Carlo simulation capabilities
for the fire evacuation system. The parallel implementation utilizes all available
CPU cores for maximum performance.

Features:
- Random fire and agent placement for each simulation
- Serial execution for debugging and small runs
- Parallel execution using multiprocessing for large-scale simulations
- Reproducible results with random seed control
- Comprehensive statistics aggregation

Usage:
    # Serial mode (10 runs)
    python monte_carlo.py --runs 10

    # Parallel mode using all CPU cores
    python monte_carlo.py --runs 100 --parallel

    # Parallel mode with specific number of processes
    python monte_carlo.py --runs 50 --parallel --processes 4

Functions:
    replace_fire(config, num_fires): Randomly place fire on valid positions
    replace_agents(config, num_agents): Randomly place agents on valid positions
    run_monte_carlo_simulation(config, num_runs, seed): Serial execution
    run_monte_carlo_parallel(config, num_runs, seed, num_processes): Parallel execution
"""

from simulation import EvacuationSimulation, SimulationConfig
from d_star_lite.utils import coordsToStateName, stateNameToCoords
import argparse, json, os
import random
from collections import Counter
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import copy
from functools import partial

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

def replace_fire(config: SimulationConfig, num_fires: int=None) -> SimulationConfig:
    """
    Replace fire positions with random valid locations.
    Fire can't be placed on doors, exits, or obstacles.

    Args:
        config: Simulation configuration
        num_fires: Number of fire locations (if None, keeps original count)

    Returns:
        Updated configuration with new fire positions
    """
    if num_fires is None:
        # Count existing fires
        num_fires = sum(1 for row in config.initial_fire_map
                       for cell in row if cell > 0)

    # Create set of invalid positions (doors, exits, obstacles)
    invalid_positions = set()

    # Add door and exit positions
    if config.door_configs:
        for door in config.door_configs:
            col, row = stateNameToCoords(door['position'])
            invalid_positions.add((row, col))

    # Add obstacle positions (-2 in fire map)
    for row_idx, row in enumerate(config.initial_fire_map):
        for col_idx, cell in enumerate(row):
            if cell < 0:
                invalid_positions.add((row_idx, col_idx))

    # Find all valid positions (not obstacles, doors, or exits)
    valid_positions = []
    for row_idx in range(config.map_rows):
        for col_idx in range(config.map_cols):
            if (row_idx, col_idx) not in invalid_positions:
                valid_positions.append((row_idx, col_idx))

    # Create new fire map (copy obstacles from original)
    new_fire_map = [[config.initial_fire_map[r][c] if config.initial_fire_map[r][c] == -2 else 0
                     for c in range(config.map_cols)]
                    for r in range(config.map_rows)]

    # Randomly select fire positions
    if len(valid_positions) < num_fires:
        raise ValueError(f"Not enough valid positions for {num_fires} fires. Only {len(valid_positions)} available.")

    fire_positions = random.sample(valid_positions, num_fires)

    # Place fires (intensity 2.0 like in the example)
    for row_idx, col_idx in fire_positions:
        new_fire_map[row_idx][col_idx] = 2.0

    # Update config
    config.initial_fire_map = new_fire_map
    return config

def replace_agents(config: SimulationConfig, num_agents: int=None) -> SimulationConfig:
    """
    Replace agent starting positions with random valid locations.
    Agents can't be placed on fire, doors, exits, or obstacles.
    Also validates that each agent can reach an exit through the door graph.

    Args:
        config: Simulation configuration
        num_agents: Number of agents (if None, uses config.agent_num)

    Returns:
        Updated configuration with new agent starting positions
    """
    if num_agents is None:
        num_agents = config.agent_num

    # Create set of invalid positions (fire, doors, exits, obstacles)
    invalid_positions = set()

    # Add door and exit positions
    if config.door_configs:
        for door in config.door_configs:
            col, row = stateNameToCoords(door['position'])
            invalid_positions.add((row, col))

    # Add fire and obstacle positions
    for row_idx, row in enumerate(config.initial_fire_map):
        for col_idx, cell in enumerate(row):
            # Fire (positive values) or obstacles (<0)
            if cell != 0:
                invalid_positions.add((row_idx, col_idx))

    # Find all valid positions
    valid_positions = []
    for row_idx in range(config.map_rows):
        for col_idx in range(config.map_cols):
            if (row_idx, col_idx) not in invalid_positions:
                valid_positions.append((row_idx, col_idx))

    if len(valid_positions) < num_agents:
        raise ValueError(f"Not enough valid positions for {num_agents} agents. Only {len(valid_positions)} available.")

    # If using door graphs, validate that agents can reach exits
    if config.door_configs:
        from door_graph import build_door_graph, replan_path

        # Build door graph to test connectivity
        door_graph = build_door_graph(config.initial_fire_map, config.door_configs)

        # Filter positions to only those with valid paths to exits
        reachable_positions = []
        for row_idx, col_idx in valid_positions:
            pos = coordsToStateName(col_idx, row_idx)
            # Test if this position can reach an exit
            path = replan_path(door_graph, pos, config.initial_fire_map)
            if path is not None:
                reachable_positions.append((row_idx, col_idx))

        if len(reachable_positions) < num_agents:
            raise ValueError(
                f"Not enough reachable positions for {num_agents} agents. "
                f"Only {len(reachable_positions)} positions have valid paths to exits "
                f"(out of {len(valid_positions)} obstacle-free positions)."
            )

        # Use only reachable positions
        valid_positions = reachable_positions

    # Randomly select agent positions
    agent_positions = random.sample(valid_positions, num_agents)

    # Convert to state names format
    new_start_positions = [coordsToStateName(col_idx, row_idx) for row_idx, col_idx in agent_positions]

    # Update config
    config.start_positions = new_start_positions
    config.agent_num = num_agents
    return config


def run_monte_carlo_simulation(config: SimulationConfig, num_runs: int, random_seed: int = None) -> list:
    """
    Run multiple evacuation simulations and collect statistics.

    Args:
        config (SimulationConfig): Configuration for the simulation.
        num_runs (int): Number of simulation runs to perform.
    """

    random.seed(random_seed)
    results = []
    statistics = {}
    path_count = {}
    average_steps = 0
    average_fire_damage = 0
    average_peak_temp = 0
    average_avg_temp = 0
    evacuated_agents = 0
    survived_agents = 0

    # Use tqdm progress bar if available
    iterator = tqdm(range(num_runs), desc="Running simulations", unit="run") if TQDM_AVAILABLE else range(num_runs)

    for i in iterator:
        # Create a fresh copy of config for each simulation run
        run_config = copy.deepcopy(config)
        # Randomize fire and agent positions for THIS run
        run_config = replace_fire(run_config)
        run_config = replace_agents(run_config)

        sim = EvacuationSimulation(run_config, silent=True)
        result = sim.run(500, show_visualization=False, use_pygame=False, use_matlab=False)

        results.append(result)
        path_count = dict(Counter(path_count) + Counter(result['path_count']))
        average_steps = (average_steps * i + result['steps']) / (i + 1)
        average_fire_damage = (average_fire_damage * i + result['average_fire_damage']) / (i + 1)
        average_peak_temp = (average_peak_temp * i + result['average_peak_temp']) / (i + 1)
        average_avg_temp = (average_avg_temp * i + result['average_avg_temp']) / (i + 1)
        evacuated_agents += result['evacuated_agents']
        survived_agents += result['survived_agents']

        # Update progress bar with stats
        if TQDM_AVAILABLE:
            success_rate = (evacuated_agents / ((i + 1) * config.agent_num)) * 100
            iterator.set_postfix({
                'Success Rate': f'{success_rate:.1f}%',
                'Evacuated': evacuated_agents,
                'Avg Steps': f'{average_steps:.1f}'
            })
        else:
            # Simple print for fallback
            success_rate = (evacuated_agents / ((i + 1) * config.agent_num)) * 100
            print(f"Run {i+1}/{num_runs} | Success: {success_rate:.1f}% | Evacuated: {evacuated_agents}")

    statistics['path_count'] = path_count
    statistics['average_steps'] = average_steps
    statistics['average_fire_damage'] = average_fire_damage
    statistics['average_peak_temp'] = average_peak_temp
    statistics['average_avg_temp'] = average_avg_temp
    statistics['evacuated_agents'] = evacuated_agents
    statistics['survived_agents'] = survived_agents

    return results, statistics

def _run_single_simulation(args):
    """
    Worker function to run a single simulation (for parallel execution).

    Args:
        args: Tuple of (config, run_number, total_runs, seed)

    Returns:
        Dictionary with simulation results
    """
    config, run_number, total_runs, seed = args

    # Set random seed for this specific run (for reproducibility)
    if seed is not None:
        random.seed(seed + run_number)

    # Create a deep copy to avoid shared state between processes
    config_copy = copy.deepcopy(config)

    # Randomize fire and agent positions for THIS run
    config_copy = replace_fire(config_copy)
    config_copy = replace_agents(config_copy)

    sim = EvacuationSimulation(config_copy, silent=True)

    # Run without visualization for speed
    result = sim.run(
        max_steps=500,
        show_visualization=False,
        use_pygame=False,
        use_matlab=False
    )

    return result


def run_monte_carlo_parallel(config: SimulationConfig, num_runs: int, random_seed: int = None, num_processes: int = None) -> tuple:
    """
    Run multiple evacuation simulations in parallel using all available CPU cores.

    Args:
        config (SimulationConfig): Configuration for the simulation.
        num_runs (int): Number of simulation runs to perform.
        random_seed (int): Random seed for reproducibility (optional).
        num_processes (int): Number of processes to use (default: all CPU cores).

    Returns:
        Tuple of (results, statistics)
    """
    # Determine number of processes
    if num_processes is None:
        num_processes = cpu_count()

    print(f"\n{'='*60}")
    print(f"Running {num_runs} simulations in parallel using {num_processes} CPU cores")
    print(f"{'='*60}\n")

    # Prepare arguments for each simulation run
    sim_args = [(config, i, num_runs, random_seed) for i in range(num_runs)]

    # Run simulations in parallel with progress bar
    with Pool(processes=num_processes) as pool:
        if TQDM_AVAILABLE:
            results = list(tqdm(
                pool.imap(_run_single_simulation, sim_args),
                total=num_runs,
                desc="Running parallel simulations",
                unit="run"
            ))
        else:
            results = pool.map(_run_single_simulation, sim_args)
            print(f"Completed all {num_runs} simulations")

    print(f"\n{'='*60}")
    print(f"All {num_runs} simulations completed!")
    print(f"{'='*60}\n")

    # Aggregate statistics from all runs
    print("Aggregating results...")
    statistics = {}
    path_count = {}
    average_steps = 0
    average_fire_damage = 0
    average_peak_temp = 0
    average_avg_temp = 0
    evacuated_agents = 0
    survived_agents = 0

    for i, result in enumerate(results):
        path_count = dict(Counter(path_count) + Counter(result['path_count']))
        average_steps = (average_steps * i + result['steps']) / (i + 1)
        average_fire_damage = (average_fire_damage * i + result['average_fire_damage']) / (i + 1)
        average_peak_temp = (average_peak_temp * i + result['average_peak_temp']) / (i + 1)
        average_avg_temp = (average_avg_temp * i + result['average_avg_temp']) / (i + 1)
        evacuated_agents += result['evacuated_agents']
        survived_agents += result['survived_agents']

    # Calculate success rate
    total_agents = num_runs * config.agent_num
    success_rate = (evacuated_agents / total_agents) * 100 if total_agents > 0 else 0

    statistics['path_count'] = path_count
    statistics['average_steps'] = average_steps
    statistics['average_fire_damage'] = average_fire_damage
    statistics['average_peak_temp'] = average_peak_temp
    statistics['average_avg_temp'] = average_avg_temp
    statistics['evacuated_agents'] = evacuated_agents
    statistics['survived_agents'] = survived_agents
    statistics['success_rate'] = success_rate

    print(f"Overall Success Rate: {success_rate:.1f}% ({evacuated_agents}/{total_agents} agents evacuated)")

    return results, statistics

if __name__ == "__main__":
    # Required for Windows multiprocessing support
    mp.freeze_support()

    parser = argparse.ArgumentParser(
        description="Monte Carlo Evacuation Simulation",
        epilog="""
Examples:
  # Run 10 simulations in serial mode:
  python monte_carlo.py --runs 10

  # Run 50 simulations in parallel using all CPU cores:
  python monte_carlo.py --runs 50 --parallel

  # Run 100 simulations in parallel using 4 processes:
  python monte_carlo.py --runs 100 --parallel --processes 4

  # Use custom configuration file:
  python monte_carlo.py --config my_config.json --runs 20 --parallel
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="example_configuration.json",
        help="Path to configuration file (default: example_configuration.json)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of simulation runs (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run simulations in parallel using all CPU cores"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of processes for parallel execution (default: all CPU cores)"
    )
    args = parser.parse_args()

    # Load configuration
    json_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(json_path, 'r', encoding='utf-8') as f:
        json_config = json.load(f)

    sim_config = SimulationConfig.from_json(json_config)

    # Run simulations
    import time
    start_time = time.time()

    if args.parallel:
        print(f"Running in PARALLEL mode with {args.processes or cpu_count()} processes")
        results, statistics = run_monte_carlo_parallel(
            sim_config,
            num_runs=args.runs,
            random_seed=args.seed,
            num_processes=args.processes
        )
    else:
        print(f"Running in SERIAL mode")
        results, statistics = run_monte_carlo_simulation(
            sim_config,
            num_runs=args.runs,
            random_seed=args.seed
        )

    elapsed_time = time.time() - start_time

    # Print summary
    print(f"\n{'='*60}")
    print(f"MONTE CARLO SIMULATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total runs: {args.runs}")
    print(f"Mode: {'Parallel' if args.parallel else 'Serial'}")
    if args.parallel:
        print(f"Processes used: {args.processes or cpu_count()}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Average time per run: {elapsed_time/args.runs:.2f} seconds")
    print(f"\nStatistics:")
    print(f"  Average steps: {statistics['average_steps']:.2f}")
    print(f"  Average fire damage: {statistics['average_fire_damage']:.2f}")
    print(f"  Average peak temperature: {statistics['average_peak_temp']:.2f}")
    print(f"  Average temperature: {statistics['average_avg_temp']:.2f}")
    print(f"  Total evacuated agents: {statistics['evacuated_agents']}")
    print(f"  Total survived agents: {statistics['survived_agents']}")
    print(f"{'='*60}\n")