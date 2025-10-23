"""
Monte Carlo Queue System
=========================

Run multiple Monte Carlo simulations sequentially from a queue of JSON configuration files.
Each run generates a comprehensive output file with all simulation data.

Usage:
    # Create a folder with JSON config files, then run:
    python monte_carlo_queue.py --queue-folder ./configs --runs 100 --parallel

    # Or specify individual files:
    python monte_carlo_queue.py --configs config1.json config2.json config3.json --runs 50

Output:
    Creates ./monte_carlo_results/{config_name}_{timestamp}/
        - full_results.json      (complete simulation data)
        - summary.txt            (human-readable summary)
        - statistics.json        (aggregated statistics)
        - config_used.json       (configuration that was used)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import time
from typing import List, Dict, Any

from simulation import SimulationConfig
from monte_carlo import run_monte_carlo_simulation, run_monte_carlo_parallel


def save_comprehensive_results(
    config: SimulationConfig,
    results: List[Dict],
    statistics: Dict,
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

    summary_text += f"\n{'='*80}\n"

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"  ✓ Saved summary to: {summary_path}")

    print(f"\n  📁 All results saved to: {output_dir}\n")


def process_config_file(
    config_path: Path,
    num_runs: int,
    random_seed: int,
    parallel: bool,
    num_processes: int,
    output_base_dir: Path,
    queue_index: int,
    total_configs: int
) -> Dict[str, Any]:
    """
    Process a single configuration file with Monte Carlo simulation.

    Args:
        queue_index: Index in the queue (1-based) for unique naming
        total_configs: Total number of configs in queue

    Returns:
        Dictionary with results and metadata
    """
    print(f"\n{'='*80}")
    print(f"Processing: {config_path.name}")
    print(f"{'='*80}\n")

    # Load configuration
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            json_config = json.load(f)
        config = SimulationConfig.from_json(json_config)
    except Exception as e:
        print(f"❌ ERROR loading config file {config_path}: {e}")
        return {
            'config_file': str(config_path),
            'status': 'error',
            'error': str(e)
        }

    # Create output directory with unique name
    config_name = config_path.stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Add zero-padded index to ensure uniqueness and sorting
    unique_id = f"{timestamp}_{queue_index:03d}"
    output_dir = output_base_dir / f"{config_name}_{unique_id}"

    # Handle rare case where directory already exists
    counter = 1
    while output_dir.exists():
        output_dir = output_base_dir / f"{config_name}_{unique_id}_dup{counter}"
        counter += 1

    # Run Monte Carlo
    start_time = time.time()
    try:
        if parallel:
            results, statistics = run_monte_carlo_parallel(
                config,
                num_runs=num_runs,
                random_seed=random_seed,
                num_processes=num_processes
            )
            mode = 'parallel'
        else:
            results, statistics = run_monte_carlo_simulation(
                config,
                num_runs=num_runs,
                random_seed=random_seed
            )
            mode = 'serial'
    except Exception as e:
        print(f"❌ ERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config_file': str(config_path),
            'status': 'error',
            'error': str(e)
        }

    elapsed_time = time.time() - start_time

    # Save comprehensive results
    print(f"\n📊 Saving results...")
    save_comprehensive_results(
        config=config,
        results=results,
        statistics=statistics,
        output_dir=output_dir,
        elapsed_time=elapsed_time,
        num_runs=num_runs,
        mode=mode,
        num_processes=num_processes
    )

    # Print summary
    total_agents = num_runs * config.agent_num
    success_rate = (statistics['evacuated_agents'] / total_agents * 100) if total_agents > 0 else 0

    print(f"✅ COMPLETED: {config_path.name}")
    print(f"   Success Rate: {success_rate:.2f}%")
    print(f"   Evacuated: {statistics['evacuated_agents']}/{total_agents}")
    print(f"   Time: {elapsed_time:.2f}s ({elapsed_time/num_runs:.2f}s per run)")

    return {
        'config_file': str(config_path),
        'status': 'success',
        'output_dir': str(output_dir),
        'elapsed_time': elapsed_time,
        'success_rate': success_rate,
        'evacuated_agents': statistics['evacuated_agents'],
        'total_agents': total_agents
    }


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Queue System - Run multiple configurations sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all JSON files in a folder:
  python monte_carlo_queue.py --queue-folder ./configs --runs 100 --parallel

  # Run specific config files:
  python monte_carlo_queue.py --configs config1.json config2.json --runs 50

  # Serial mode with custom output directory:
  python monte_carlo_queue.py --configs config.json --runs 20 --output ./my_results
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--queue-folder',
        type=str,
        help='Folder containing JSON configuration files to process'
    )
    input_group.add_argument(
        '--configs',
        type=str,
        nargs='+',
        help='List of specific JSON configuration files to process'
    )

    # Simulation parameters
    parser.add_argument(
        '--runs',
        type=int,
        default=100,
        help='Number of Monte Carlo runs per configuration (default: 100)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run simulations in parallel mode'
    )
    parser.add_argument(
        '--processes',
        type=int,
        default=None,
        help='Number of processes for parallel execution (default: all CPU cores)'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='./monte_carlo_results',
        help='Base directory for output files (default: ./monte_carlo_results)'
    )

    args = parser.parse_args()

    # Get list of config files
    if args.queue_folder:
        queue_folder = Path(args.queue_folder)
        if not queue_folder.exists():
            print(f"❌ ERROR: Queue folder not found: {queue_folder}")
            sys.exit(1)

        config_files = sorted(queue_folder.glob('*.json'))
        if not config_files:
            print(f"❌ ERROR: No JSON files found in {queue_folder}")
            sys.exit(1)
    else:
        config_files = [Path(f) for f in args.configs]
        for cf in config_files:
            if not cf.exists():
                print(f"❌ ERROR: Config file not found: {cf}")
                sys.exit(1)

    output_base_dir = Path(args.output)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Print queue summary
    print(f"\n{'='*80}")
    print(f"MONTE CARLO QUEUE SYSTEM")
    print(f"{'='*80}")
    print(f"\n📋 Queue Summary:")
    print(f"   Number of configurations: {len(config_files)}")
    print(f"   Runs per configuration: {args.runs}")
    print(f"   Total simulations: {len(config_files) * args.runs}")
    print(f"   Mode: {'PARALLEL' if args.parallel else 'SERIAL'}")
    if args.parallel and args.processes:
        print(f"   Processes: {args.processes}")
    print(f"   Output directory: {output_base_dir}")
    print(f"\n📁 Configuration files:")
    for i, cf in enumerate(config_files, 1):
        print(f"   {i}. {cf.name}")
    print()

    # Process queue
    queue_start_time = time.time()
    queue_start_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    queue_results = []

    for i, config_file in enumerate(config_files, 1):
        print(f"\n{'#'*80}")
        print(f"# Processing {i}/{len(config_files)}")
        print(f"{'#'*80}")

        result = process_config_file(
            config_path=config_file,
            num_runs=args.runs,
            random_seed=args.seed,
            parallel=args.parallel,
            num_processes=args.processes,
            output_base_dir=output_base_dir,
            queue_index=i,
            total_configs=len(config_files)
        )
        queue_results.append(result)

    queue_elapsed_time = time.time() - queue_start_time

    # Save queue summary (use start timestamp for consistency)
    queue_summary_path = output_base_dir / f"queue_summary_{queue_start_timestamp}.json"
    queue_summary = {
        'timestamp': datetime.now().isoformat(),
        'total_configs': len(config_files),
        'runs_per_config': args.runs,
        'total_simulations': len(config_files) * args.runs,
        'mode': 'parallel' if args.parallel else 'serial',
        'total_time_seconds': queue_elapsed_time,
        'results': queue_results
    }

    with open(queue_summary_path, 'w', encoding='utf-8') as f:
        json.dump(queue_summary, f, indent=2)

    # Print final summary
    print(f"\n{'='*80}")
    print(f"QUEUE COMPLETED")
    print(f"{'='*80}\n")
    print(f"Total Time: {queue_elapsed_time:.2f} seconds ({queue_elapsed_time/60:.2f} minutes)")
    print(f"Configurations Processed: {len(config_files)}")
    print(f"Total Simulations: {len(config_files) * args.runs}")
    print(f"\n📊 Results:")

    successful = sum(1 for r in queue_results if r['status'] == 'success')
    failed = sum(1 for r in queue_results if r['status'] == 'error')

    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")

    if successful > 0:
        avg_success_rate = sum(r.get('success_rate', 0) for r in queue_results if r['status'] == 'success') / successful
        print(f"   📈 Average Success Rate: {avg_success_rate:.2f}%")

    print(f"\n📁 Queue summary saved to: {queue_summary_path}")
    print(f"📁 All results in: {output_base_dir}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
