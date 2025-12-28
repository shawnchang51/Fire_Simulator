"""
Deep integrity check - sample random lines from JSONL files
and verify all required fields are present.
"""

import json
import random
import numpy as np
from pathlib import Path

def check_jsonl_integrity(jsonl_path, sample_size=100):
    """Check random sample of JSONL lines for data integrity."""
    errors = []
    warnings = []

    # Required fields in simulation results
    required_fields = [
        'floor_plan_id',
        'exit_config_id',
        'config_id',
        'config',
        'scenario_hash',
        'survival_rate',
        'avg_evacuation_time',
        'steps',
        'evacuated',
        'stuck',
        'dead',
        'avg_fire_damage',
        'score'
    ]

    # Load all lines
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)

    # Sample random lines
    sample_size = min(sample_size, total_lines)
    sample_indices = random.sample(range(total_lines), sample_size)

    valid_count = 0
    for idx in sample_indices:
        try:
            data = json.loads(lines[idx].strip())

            # Check required fields
            missing_fields = [f for f in required_fields if f not in data]
            if missing_fields:
                errors.append(f"Line {idx+1}: missing fields {missing_fields}")
                continue

            # Validate data types and ranges
            if not isinstance(data['survival_rate'], (int, float)) or not (0 <= data['survival_rate'] <= 1):
                warnings.append(f"Line {idx+1}: invalid survival_rate {data['survival_rate']}")

            if not isinstance(data['steps'], int) or data['steps'] < 0:
                warnings.append(f"Line {idx+1}: invalid steps {data['steps']}")

            if not isinstance(data['evacuated'], int) or data['evacuated'] < 0:
                warnings.append(f"Line {idx+1}: invalid evacuated {data['evacuated']}")

            valid_count += 1

        except json.JSONDecodeError as e:
            errors.append(f"Line {idx+1}: JSON parse error - {e}")
        except Exception as e:
            errors.append(f"Line {idx+1}: Unexpected error - {e}")

    return {
        'total_lines': total_lines,
        'sample_size': sample_size,
        'valid_count': valid_count,
        'errors': errors,
        'warnings': warnings
    }

def check_floor_plans(floor_plans_dir, sample_size=10):
    """Check random sample of floor plan NPZ files."""
    errors = []
    warnings = []

    floor_plan_files = list(Path(floor_plans_dir).glob('plan_*.npz'))
    total_plans = len(floor_plan_files)

    # Sample random floor plans
    sample_size = min(sample_size, total_plans)
    sample_files = random.sample(floor_plan_files, sample_size)

    valid_count = 0
    for plan_file in sample_files:
        try:
            data = np.load(plan_file)

            # Check required keys
            required_keys = ['grid', 'door_positions', 'exit_positions']
            missing_keys = [k for k in required_keys if k not in data]
            if missing_keys:
                errors.append(f"{plan_file.name}: missing keys {missing_keys}")
                continue

            # Validate grid
            grid = data['grid']
            if grid.ndim != 2:
                errors.append(f"{plan_file.name}: grid should be 2D, got {grid.ndim}D")
                continue

            if grid.size == 0:
                errors.append(f"{plan_file.name}: grid is empty")
                continue

            # Check door and exit positions
            doors = data['door_positions']
            exits = data['exit_positions']

            if len(doors) == 0:
                warnings.append(f"{plan_file.name}: no door positions")

            if len(exits) == 0:
                warnings.append(f"{plan_file.name}: no exit positions")

            valid_count += 1

        except Exception as e:
            errors.append(f"{plan_file.name}: {e}")

    return {
        'total_plans': total_plans,
        'sample_size': sample_size,
        'valid_count': valid_count,
        'errors': errors,
        'warnings': warnings
    }

def main():
    random.seed(42)
    base_path = Path("c:/dev/Fire_Simulator/server_results")

    print("="*70)
    print("DEEP INTEGRITY CHECK")
    print("="*70)
    print("Sampling random data to verify integrity...\n")

    data_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('v5_data')])

    all_good = True

    for data_dir in data_dirs:
        print(f"\n{data_dir.name}:")
        print("-" * 50)

        # Check JSONL
        jsonl_path = data_dir / 'simulation_results.jsonl'
        if jsonl_path.exists():
            print("  Checking JSONL (100 random samples)...")
            result = check_jsonl_integrity(jsonl_path, sample_size=100)
            print(f"    Total lines: {result['total_lines']:,}")
            print(f"    Sampled: {result['sample_size']}")
            print(f"    Valid: {result['valid_count']}/{result['sample_size']}")

            if result['errors']:
                print(f"    [X] Errors: {len(result['errors'])}")
                for error in result['errors'][:5]:  # Show first 5
                    print(f"        - {error}")
                all_good = False
            else:
                print("    [OK] No errors")

            if result['warnings']:
                print(f"    [!] Warnings: {len(result['warnings'])}")
                for warning in result['warnings'][:5]:  # Show first 5
                    print(f"        - {warning}")

        # Check floor plans
        floor_plans_dir = data_dir / 'floor_plans'
        if floor_plans_dir.exists():
            print("  Checking Floor Plans (10 random samples)...")
            result = check_floor_plans(floor_plans_dir, sample_size=10)
            print(f"    Total plans: {result['total_plans']}")
            print(f"    Sampled: {result['sample_size']}")
            print(f"    Valid: {result['valid_count']}/{result['sample_size']}")

            if result['errors']:
                print(f"    [X] Errors: {len(result['errors'])}")
                for error in result['errors'][:5]:  # Show first 5
                    print(f"        - {error}")
                all_good = False
            else:
                print("    [OK] No errors")

            if result['warnings']:
                print(f"    [!] Warnings: {len(result['warnings'])}")
                for warning in result['warnings'][:5]:  # Show first 5
                    print(f"        - {warning}")

    print("\n" + "="*70)
    if all_good:
        print("[OK] DEEP INTEGRITY CHECK PASSED")
        print("All sampled data is valid and properly formatted.")
    else:
        print("[X] DEEP INTEGRITY CHECK FAILED")
        print("Some data corruption detected. Review errors above.")
    print("="*70)

if __name__ == '__main__':
    main()
