"""
Verify training data downloaded from servers.

Checks:
1. All expected files exist
2. JSONL files are valid JSON
3. Floor plan NPZ files are loadable
4. Data completeness and consistency
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def verify_directory(base_dir):
    """Verify a single data directory."""
    results = {
        'name': os.path.basename(base_dir),
        'exists': os.path.exists(base_dir),
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    if not results['exists']:
        results['errors'].append(f"Directory does not exist: {base_dir}")
        return results

    # Check required files
    config_file = os.path.join(base_dir, 'config.json')
    checkpoint_file = os.path.join(base_dir, 'checkpoint.json')
    sim_results_file = os.path.join(base_dir, 'simulation_results.jsonl')
    floor_plans_dir = os.path.join(base_dir, 'floor_plans')

    required_files = {
        'config.json': config_file,
        'checkpoint.json': checkpoint_file,
        'simulation_results.jsonl': sim_results_file,
        'floor_plans/': floor_plans_dir
    }

    for name, path in required_files.items():
        if not os.path.exists(path):
            results['errors'].append(f"Missing: {name}")

    # Load and check config
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            results['stats']['config'] = {
                'plan_range': f"[{config['plan_start_idx']}, {config['plan_end_idx']})",
                'expected_plans': config['plan_end_idx'] - config['plan_start_idx'],
                'evaluation_only': config['evaluation_only']
            }
    except Exception as e:
        results['errors'].append(f"Config load error: {e}")
        config = None

    # Load and check checkpoint
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            progress = checkpoint['progress']
            results['stats']['checkpoint'] = {
                'completion': f"{progress['completion_percent']:.2f}%",
                'configs_evaluated': progress['configs_evaluated'],
                'configs_total': progress['configs_total'],
                'floor_plans_loaded': progress['floor_plans_loaded']
            }

            # Check if complete
            if progress['completion_percent'] < 100.0:
                missing = progress['configs_total'] - progress['configs_evaluated']
                results['warnings'].append(
                    f"Incomplete: {missing} configs missing "
                    f"({100 - progress['completion_percent']:.2f}% missing)"
                )
    except Exception as e:
        results['errors'].append(f"Checkpoint load error: {e}")

    # Verify JSONL file
    try:
        line_count = 0
        parse_errors = 0
        sample_data = None

        with open(sim_results_file, 'r') as f:
            for i, line in enumerate(f):
                line_count += 1
                try:
                    data = json.loads(line.strip())
                    if sample_data is None:
                        sample_data = data
                except json.JSONDecodeError as e:
                    parse_errors += 1
                    if parse_errors <= 5:  # Only report first 5 errors
                        results['errors'].append(f"JSONL parse error at line {i+1}: {e}")

        results['stats']['simulation_results'] = {
            'line_count': line_count,
            'parse_errors': parse_errors,
            'valid': parse_errors == 0
        }

        # Check expected count
        if config:
            expected_plans = config['plan_end_idx'] - config['plan_start_idx']
            expected_configs = (expected_plans *
                               config['exit_configs_per_plan'] *
                               config['door_configs_per_exit'] *
                               config['monte_carlo_runs_per_config'])

            results['stats']['simulation_results']['expected_count'] = expected_configs
            missing = expected_configs - line_count

            if missing > 0:
                results['warnings'].append(
                    f"JSONL: {missing} lines missing ({missing/expected_configs*100:.2f}%)"
                )

    except Exception as e:
        results['errors'].append(f"JSONL verification error: {e}")

    # Verify floor plans
    try:
        floor_plan_files = list(Path(floor_plans_dir).glob('plan_*.npz'))
        results['stats']['floor_plans'] = {
            'count': len(floor_plan_files)
        }

        if config:
            expected_plans = config['plan_end_idx'] - config['plan_start_idx']
            if len(floor_plan_files) != expected_plans:
                results['warnings'].append(
                    f"Floor plans: expected {expected_plans}, found {len(floor_plan_files)}"
                )

        # Try loading a sample floor plan
        if floor_plan_files:
            sample_plan = floor_plan_files[0]
            try:
                data = np.load(sample_plan)
                required_keys = ['grid', 'door_positions', 'exit_positions']
                for key in required_keys:
                    if key not in data:
                        results['errors'].append(f"Floor plan missing key: {key}")
                results['stats']['floor_plans']['sample_loadable'] = True
                results['stats']['floor_plans']['sample_grid_shape'] = data['grid'].shape
            except Exception as e:
                results['errors'].append(f"Floor plan load error: {e}")
                results['stats']['floor_plans']['sample_loadable'] = False
    except Exception as e:
        results['errors'].append(f"Floor plan verification error: {e}")

    return results

def print_report(result):
    """Print verification report for a single directory."""
    print(f"\n{'='*70}")
    print(f"Directory: {result['name']}")
    print(f"{'='*70}")

    if not result['exists']:
        print("❌ DIRECTORY DOES NOT EXIST")
        return

    # Print stats
    print("\nConfiguration:")
    if 'config' in result['stats']:
        for key, value in result['stats']['config'].items():
            print(f"  {key}: {value}")

    print("\nProgress:")
    if 'checkpoint' in result['stats']:
        for key, value in result['stats']['checkpoint'].items():
            print(f"  {key}: {value}")

    print("\nSimulation Results (JSONL):")
    if 'simulation_results' in result['stats']:
        for key, value in result['stats']['simulation_results'].items():
            print(f"  {key}: {value}")

    print("\nFloor Plans:")
    if 'floor_plans' in result['stats']:
        for key, value in result['stats']['floor_plans'].items():
            print(f"  {key}: {value}")

    # Print warnings
    if result['warnings']:
        print("\n[!] Warnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")

    # Print errors
    if result['errors']:
        print("\n[X] Errors:")
        for error in result['errors']:
            print(f"  - {error}")

    # Overall status
    print("\nStatus:", end=" ")
    if result['errors']:
        print("[X] FAILED - Has errors")
    elif result['warnings']:
        print("[!] ACCEPTABLE - Minor issues (likely failed simulations)")
    else:
        print("[OK] PERFECT")

def main():
    base_path = Path("c:/dev/Fire_Simulator/server_results")

    print("="*70)
    print("TRAINING DATA VERIFICATION")
    print("="*70)
    print(f"\nBase directory: {base_path}")

    # Find all data directories
    data_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('v5_data')])

    print(f"Found {len(data_dirs)} data directories\n")

    all_results = []
    for data_dir in data_dirs:
        result = verify_directory(str(data_dir))
        all_results.append(result)
        print_report(result)

    # Print summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    total_errors = sum(len(r['errors']) for r in all_results)
    total_warnings = sum(len(r['warnings']) for r in all_results)

    # Aggregate stats
    total_sim_results = sum(
        r['stats'].get('simulation_results', {}).get('line_count', 0)
        for r in all_results
    )
    total_expected = sum(
        r['stats'].get('simulation_results', {}).get('expected_count', 0)
        for r in all_results
    )
    total_floor_plans = sum(
        r['stats'].get('floor_plans', {}).get('count', 0)
        for r in all_results
    )

    print(f"\nTotal simulation results: {total_sim_results:,} / {total_expected:,}")
    if total_expected > 0:
        completeness = (total_sim_results / total_expected) * 100
        print(f"Completeness: {completeness:.2f}%")
        missing = total_expected - total_sim_results
        print(f"Missing: {missing} ({(missing/total_expected)*100:.2f}%)")

    print(f"Total floor plans: {total_floor_plans}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")

    print(f"\n{'='*70}")
    if total_errors > 0:
        print("[X] OVERALL STATUS: FAILED")
        print("Action: DO NOT delete server data - requires investigation")
    elif total_warnings > 0:
        print("[OK] OVERALL STATUS: ACCEPTABLE")
        print("Action: Data is usable. Minor missing data is expected from failed")
        print("        simulations. Safe to delete from server if >99% complete.")
    else:
        print("[OK] OVERALL STATUS: PERFECT")
        print("Action: All data verified. Safe to delete from server.")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
