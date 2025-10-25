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
- Automatic file saving with full results, statistics, and human-readable summaries

Usage:
    # Serial mode (10 runs)
    python monte_carlo.py --runs 10

    # Parallel mode using all CPU cores
    python monte_carlo.py --runs 100 --parallel

    # Parallel mode with specific number of processes and custom output directory
    python monte_carlo.py --runs 50 --parallel --processes 4 --output ./my_results

Output:
    Creates ./monte_carlo_results/{config_name}_{timestamp}/
        - full_results.json      (complete simulation data)
        - summary.txt            (human-readable summary)
        - statistics.json        (aggregated statistics)
        - config_used.json       (configuration that was used)

Functions:
    replace_fire(config, num_fires): Randomly place fire on valid positions
    replace_agents(config, num_agents): Randomly place agents on valid positions
    run_monte_carlo_simulation(config, num_runs, seed): Serial execution
    run_monte_carlo_parallel(config, num_runs, seed, num_processes): Parallel execution
    save_comprehensive_results(config, results, statistics, ...): Save results to files
"""

from simulation import EvacuationSimulation, SimulationConfig
from d_star_lite.utils import coordsToStateName, stateNameToCoords
from distribution_analysis import compute_distributions, compute_per_run_distributions, print_distribution_summary
import argparse, json, os
import random
from collections import Counter
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import copy
from functools import partial
from pathlib import Path
from datetime import datetime
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

def replace_fire(config: SimulationConfig, num_fires: int=30) -> SimulationConfig:
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

    # Get actual dimensions from the fire map itself (not from config)
    actual_rows = len(config.initial_fire_map)
    # Handle jagged arrays by finding the maximum column count
    actual_cols = max(len(row) for row in config.initial_fire_map) if actual_rows > 0 else 0

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
    # Use actual fire map dimensions, not config dimensions
    valid_positions = []
    for row_idx in range(actual_rows):
        for col_idx in range(actual_cols):
            if (row_idx, col_idx) not in invalid_positions:
                valid_positions.append((row_idx, col_idx))

    # Create new fire map (copy obstacles from original)
    # Handle jagged arrays by checking bounds for each row
    new_fire_map = []
    for r in range(actual_rows):
        row_length = len(config.initial_fire_map[r])
        new_row = []
        for c in range(actual_cols):
            if c < row_length and config.initial_fire_map[r][c] == -2:
                new_row.append(-2)
            else:
                new_row.append(0)
        new_fire_map.append(new_row)

    # Randomly select fire positions
    if len(valid_positions) < num_fires:
        raise ValueError(f"Not enough valid positions for {num_fires} fires. Only {len(valid_positions)} available.")

    fire_positions = random.sample(valid_positions, num_fires)

    # Place fires (intensity 2.0 like in the example)
    for row_idx, col_idx in fire_positions:
        new_fire_map[row_idx][col_idx] = 2.0

    # Update config with new fire map AND ensure map dimensions match actual fire map
    config.initial_fire_map = new_fire_map
    config.map_rows = actual_rows
    config.map_cols = actual_cols
    return config

def replace_agents(config: SimulationConfig, num_agents: int=None) -> SimulationConfig:
    """
    Replace agent starting positions with random valid locations.
    Agents can't be placed on fire, doors, exits, or obstacles.
    Also validates that each agent can reach an exit through the door graph.

    Additionally, assigns random fearness values to each agent based on config.agent_fearness:
    - If agent_fearness has 2+ values: Random uniform between first two values
    - If agent_fearness has 1 value: Use that value for all agents
    - If agent_fearness is empty/None: Default to 1.0 for all agents

    Args:
        config: Simulation configuration
        num_agents: Number of agents (if None, uses config.agent_num)

    Returns:
        Updated configuration with new agent starting positions and fearness values

    Example:
        config.agent_fearness = [0.5, 1.5]  # Each agent gets random fear in [0.5, 1.5]
        config.agent_fearness = [1.0]       # All agents get fear = 1.0
        config.agent_fearness = None        # All agents get default fear = 1.0
    """
    if num_agents is None:
        num_agents = config.agent_num

    # Get actual dimensions from the fire map itself (not from config)
    actual_rows = len(config.initial_fire_map)
    # Handle jagged arrays by finding the maximum column count
    actual_cols = max(len(row) for row in config.initial_fire_map) if actual_rows > 0 else 0

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
    # Use actual fire map dimensions, not config dimensions
    valid_positions = []
    for row_idx in range(actual_rows):
        for col_idx in range(actual_cols):
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

    # Randomly assign fearness values between first two values in agent_fearness
    if config.agent_fearness and len(config.agent_fearness) >= 2:
        min_fear = min(config.agent_fearness[0], config.agent_fearness[1])
        max_fear = max(config.agent_fearness[0], config.agent_fearness[1])

        # Generate random fearness for each agent between min and max
        new_fearness = [random.uniform(min_fear, max_fear) for _ in range(num_agents)]
        config.agent_fearness = new_fearness
    elif config.agent_fearness and len(config.agent_fearness) == 1:
        # If only one value, use it for all agents
        config.agent_fearness = [config.agent_fearness[0]] * num_agents
    else:
        # No fearness specified, use default 1.0 for all
        config.agent_fearness = [1.0] * num_agents

    # Update config
    config.start_positions = new_start_positions
    config.agent_num = num_agents
    return config


def run_monte_carlo_simulation(config: SimulationConfig, num_runs: int) -> list:
    """
    Run multiple evacuation simulations and collect statistics.

    Args:
        config (SimulationConfig): Configuration for the simulation.
        num_runs (int): Number of simulation runs to perform.
    """
    results = []
    statistics = {}
    path_count = {}
    average_steps = 0
    average_fire_damage = 0
    average_peak_temp = 0
    average_avg_temp = 0
    evacuated_agents = 0
    survived_agents = 0

    # Collect all agent records for distribution analysis
    all_agent_records = []

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

        # Collect agent records from this run
        if 'agent_records' in result:
            all_agent_records.extend(result['agent_records'])

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

    # Compute per-agent distributions across all runs
    print("\nComputing per-agent distributions...")
    agent_distributions = compute_distributions(all_agent_records, num_bins=30, include_raw_values=False)
    statistics['agent_distributions'] = agent_distributions

    # Compute per-run distributions (distributions of run-level averages)
    print("Computing per-run distributions...")
    run_distributions = compute_per_run_distributions(results, num_bins=20)
    statistics['run_distributions'] = run_distributions

    # Print summary
    print_distribution_summary(agent_distributions)

    return results, statistics

def _run_single_simulation(args):
    """
    Worker function to run a single simulation (for parallel execution).

    Args:
        args: Tuple of (config, run_number, total_runs)

    Returns:
        Dictionary with simulation results (or error result if failed)
    """
    config, run_number, total_runs = args

    try:
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

    except Exception as e:
        # Return error result instead of crashing the worker process
        import traceback
        error_msg = f"Run {run_number + 1}/{total_runs} failed: {str(e)}"
        print(f"⚠️  {error_msg}")

        # Return a minimal result dictionary to avoid breaking aggregation
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'run_number': run_number,
            'path_count': {},
            'steps': 0,
            'average_fire_damage': 0.0,
            'average_peak_temp': 0.0,
            'average_avg_temp': 0.0,
            'evacuated_agents': 0,
            'survived_agents': 0
        }


def run_monte_carlo_parallel(config: SimulationConfig, num_runs: int, num_processes: int = None) -> tuple:
    """
    Run multiple evacuation simulations in parallel using all available CPU cores.

    Args:
        config (SimulationConfig): Configuration for the simulation.
        num_runs (int): Number of simulation runs to perform.
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
    sim_args = [(config, i, num_runs) for i in range(num_runs)]

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
    error_count = 0

    # Collect all agent records for distribution analysis
    all_agent_records = []

    for i, result in enumerate(results):
        # Skip error results in averaging but count them
        if result.get('status') == 'error':
            error_count += 1
            continue

        # Collect agent records from this run
        if 'agent_records' in result:
            all_agent_records.extend(result['agent_records'])

        path_count = dict(Counter(path_count) + Counter(result['path_count']))
        # Only count successful runs in averaging
        successful_runs = i + 1 - error_count
        if successful_runs > 0:
            average_steps = (average_steps * (successful_runs - 1) + result['steps']) / successful_runs
            average_fire_damage = (average_fire_damage * (successful_runs - 1) + result['average_fire_damage']) / successful_runs
            average_peak_temp = (average_peak_temp * (successful_runs - 1) + result['average_peak_temp']) / successful_runs
            average_avg_temp = (average_avg_temp * (successful_runs - 1) + result['average_avg_temp']) / successful_runs
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
    statistics['error_count'] = error_count
    statistics['successful_runs'] = num_runs - error_count

    if error_count > 0:
        print(f"⚠️  Warning: {error_count}/{num_runs} simulations failed with errors")
    print(f"Overall Success Rate: {success_rate:.1f}% ({evacuated_agents}/{total_agents} agents evacuated)")

    # Compute per-agent distributions across all runs
    print("\nComputing per-agent distributions...")
    agent_distributions = compute_distributions(all_agent_records, num_bins=30, include_raw_values=False)
    statistics['agent_distributions'] = agent_distributions

    # Compute per-run distributions (distributions of run-level averages)
    print("Computing per-run distributions...")
    # Filter out error results for run-level distributions
    successful_results = [r for r in results if r.get('status') != 'error']
    run_distributions = compute_per_run_distributions(successful_results, num_bins=20)
    statistics['run_distributions'] = run_distributions

    # Print summary
    print_distribution_summary(agent_distributions)

    return results, statistics


def save_comprehensive_results(
    config: SimulationConfig,
    results: list,
    statistics: dict,
    output_dir: Path,
    elapsed_time: float,
    num_runs: int,
    mode: str,
    num_processes: int = None
):
    """
    Save complete Monte Carlo results to multiple files for full traceability.

    Args:
        config: The configuration used
        results: List of individual simulation results
        statistics: Aggregated statistics
        output_dir: Directory to save results
        elapsed_time: Total time taken
        num_runs: Number of runs performed
        mode: 'serial' or 'parallel'
        num_processes: Number of processes used (if parallel)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Helper function to convert non-JSON-serializable keys (like tuples) to strings
    def convert_dict_keys_to_strings(obj):
        """Recursively convert dictionary keys to strings for JSON serialization."""
        if isinstance(obj, dict):
            return {str(k): convert_dict_keys_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_dict_keys_to_strings(item) for item in obj]
        else:
            return obj

    # 1. Save FULL results (every simulation run)
    full_results_path = output_dir / "full_results.json"
    full_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_runs": num_runs,
            "mode": mode,
            "num_processes": num_processes,
            "elapsed_time_seconds": elapsed_time,
            "time_per_run_seconds": elapsed_time / num_runs if num_runs > 0 else 0,
        },
        "configuration": config.to_dict(),
        "individual_runs": convert_dict_keys_to_strings(results),
        "aggregated_statistics": convert_dict_keys_to_strings(statistics)
    }

    with open(full_results_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=2)
    print(f"  ✓ Saved full results to: {full_results_path}")

    # 2. Save statistics only (smaller file)
    stats_path = output_dir / "statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(convert_dict_keys_to_strings(statistics), f, indent=2)
    print(f"  ✓ Saved statistics to: {stats_path}")

    # 3. Save configuration used
    config_path = output_dir / "config_used.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  ✓ Saved configuration to: {config_path}")

    # 4. Save human-readable summary
    summary_path = output_dir / "summary.txt"
    total_agents = num_runs * config.agent_num
    success_rate = (statistics['evacuated_agents'] / total_agents * 100) if total_agents > 0 else 0

    summary_text = f"""
{'='*80}
MONTE CARLO SIMULATION SUMMARY
{'='*80}

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION
-------------
Config File: {config_path.name}
Map Size: {config.map_rows} x {config.map_cols}
Cell Size: {config.cell_size}m
Timestep Duration: {config.timestep_duration}s
Fire Update Interval: {config.fire_update_interval} timesteps
Fire Model Type: {config.fire_model_type}
Agents per Simulation: {config.agent_num}
Viewing Range: {config.viewing_range} cells
Max Occupancy: {config.max_occupancy}
Number of Doors: {len(config.door_configs) if config.door_configs else 0}

EXECUTION
---------
Number of Runs: {num_runs}
Execution Mode: {mode.upper()}
Processes Used: {num_processes if num_processes else 'N/A'}
Total Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)
Time per Run: {elapsed_time/num_runs:.2f} seconds

RESULTS
-------
Total Agents: {total_agents}
Evacuated Agents: {statistics['evacuated_agents']}
Success Rate: {success_rate:.2f}%

Average Steps: {statistics['average_steps']:.2f}
Average Fire Damage: {statistics['average_fire_damage']:.4f}
Average Peak Temperature: {statistics['average_peak_temp']:.2f}°C
Average Temperature: {statistics['average_avg_temp']:.2f}°C

PATH STATISTICS
---------------
"""

    # Add path count details
    if statistics.get('path_count'):
        summary_text += "Most Common Paths:\n"
        sorted_paths = sorted(statistics['path_count'].items(), key=lambda x: x[1], reverse=True)
        for path, count in sorted_paths[:10]:  # Top 10 paths
            summary_text += f"  {path}: {count} times\n"

    # Add distribution summary
    summary_text += "\nPER-AGENT DISTRIBUTION SUMMARY\n"
    summary_text += "------------------------------\n"

    if statistics.get('agent_distributions'):
        agent_dist = statistics['agent_distributions']

        if '_summary' in agent_dist:
            summary_text += f"Total Agents Analyzed: {agent_dist['_summary'].get('total_agents', 'N/A')}\n"
            summary_text += f"Overall Survival Rate: {agent_dist['_summary'].get('survival_rate', 0) * 100:.2f}%\n\n"

            if 'status_counts' in agent_dist['_summary']:
                summary_text += "Agent Status Distribution:\n"
                for status, count in agent_dist['_summary']['status_counts'].items():
                    summary_text += f"  {status}: {count}\n"

        # Add metrics summary
        summary_text += "\nKey Metrics (Per-Agent):\n"
        for metric in ['steps', 'fire_damage', 'peak_temp', 'average_temp']:
            if metric in agent_dist and 'statistics' in agent_dist[metric]:
                stats = agent_dist[metric]['statistics']
                percentiles = agent_dist[metric].get('percentiles', {})
                summary_text += f"\n{metric.upper()}:\n"
                summary_text += f"  Mean: {stats.get('mean', 'N/A'):.2f}\n"
                summary_text += f"  Std Dev: {stats.get('std_dev', 'N/A'):.2f}\n"
                summary_text += f"  Range: [{stats.get('min', 'N/A'):.2f}, {stats.get('max', 'N/A'):.2f}]\n"
                summary_text += f"  Median (50th): {percentiles.get('p50', 'N/A'):.2f}\n"

    summary_text += f"\n{'='*80}\n"
    summary_text += "\nNOTE: Full distribution data (histograms, percentiles) saved in statistics.json\n"
    summary_text += f"{'='*80}\n"

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"  ✓ Saved summary to: {summary_path}")

    print(f"\n  📁 All results saved to: {output_dir}\n")


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

  # Specify custom output directory:
  python monte_carlo.py --runs 50 --parallel --output ./my_results

Output:
  Creates ./monte_carlo_results/{config_name}_{timestamp}/
      - full_results.json      (complete simulation data)
      - summary.txt            (human-readable summary)
      - statistics.json        (aggregated statistics)
      - config_used.json       (configuration that was used)
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
    parser.add_argument(
        "--output",
        type=str,
        default="./monte_carlo_results",
        help="Base directory for output files (default: ./monte_carlo_results)"
    )
    args = parser.parse_args()

    # Load configuration
    json_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(json_path, 'r', encoding='utf-8') as f:
        json_config = json.load(f)

    sim_config = SimulationConfig.from_json(json_config)

    # Run simulations
    start_time = time.time()

    if args.parallel:
        print(f"Running in PARALLEL mode with {args.processes or cpu_count()} processes")
        results, statistics = run_monte_carlo_parallel(
            sim_config,
            num_runs=args.runs,
            num_processes=args.processes
        )
    else:
        print(f"Running in SERIAL mode")
        results, statistics = run_monte_carlo_simulation(
            sim_config,
            num_runs=args.runs
        )

    elapsed_time = time.time() - start_time

    # Create output directory with timestamp and config name
    config_name = Path(args.config).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base_dir = Path(args.output)
    output_dir = output_base_dir / f"{config_name}_{timestamp}"

    # Save comprehensive results
    print(f"\n📊 Saving results...")
    mode = 'parallel' if args.parallel else 'serial'
    save_comprehensive_results(
        config=sim_config,
        results=results,
        statistics=statistics,
        output_dir=output_dir,
        elapsed_time=elapsed_time,
        num_runs=args.runs,
        mode=mode,
        num_processes=args.processes
    )

    # Print brief summary to console
    total_agents = args.runs * sim_config.agent_num
    success_rate = (statistics['evacuated_agents'] / total_agents * 100) if total_agents > 0 else 0

    print(f"{'='*60}")
    print(f"QUICK SUMMARY")
    print(f"{'='*60}")
    print(f"Total runs: {args.runs}")
    print(f"Mode: {'Parallel' if args.parallel else 'Serial'}")
    if args.parallel:
        print(f"Processes used: {args.processes or cpu_count()}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Average time per run: {elapsed_time/args.runs:.2f} seconds")
    print(f"\nSuccess Rate: {success_rate:.2f}%")
    print(f"Evacuated: {statistics['evacuated_agents']}/{total_agents} agents")
    if statistics.get('error_count', 0) > 0:
        print(f"⚠️  Failed runs: {statistics['error_count']}/{args.runs}")
    print(f"\n📁 Full results saved to: {output_dir}")
    print(f"{'='*60}\n")