"""
Test script for V5 Data Analyzer

Creates a small mock V5 dataset and runs the analyzer to demonstrate functionality.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np

from analyze_v5_data import V5DataAnalyzer


def create_mock_v5_dataset(output_dir: str, num_plans: int = 10):
    """Create a small mock V5 dataset for testing"""
    print(f"Creating mock V5 dataset in {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    # Create metadata
    metadata = {
        'version': 'v5',
        'description': 'Mock dataset for testing',
        'config': {
            'num_floor_plans': num_plans,
            'exit_configs_per_plan': 3,
            'door_configs_per_exit': 4,
            'monte_carlo_runs_per_config': 5,
            'same_exit_ratio': 0.70,
            'cross_exit_ratio': 0.20,
            'cross_plan_ratio': 0.10
        },
        'statistics': {
            'total_floor_plans': num_plans,
            'exit_configs_per_plan': 3,
            'door_configs_per_exit': 4,
            'total_configurations': num_plans * 3 * 4,
            'total_pairs': num_plans * 100
        }
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create simulation results
    results_file = os.path.join(output_dir, 'simulation_results.jsonl')
    all_results = []

    with open(results_file, 'w') as f:
        for plan_id in range(num_plans):
            for exit_id in range(3):  # 3 exit configs
                for config_id in range(4):  # 4 door configs per exit
                    # Random performance metrics
                    survival_rate = np.random.uniform(0.5, 0.95)
                    steps = int(np.random.uniform(50, 300))
                    evacuated = int(survival_rate * 50)
                    dead = 50 - evacuated

                    result = {
                        'floor_plan_id': plan_id,
                        'exit_config_id': exit_id,
                        'config_id': config_id,
                        'config': {
                            'door_config': [
                                {'id': f'e{i}', 'position': f'x{i*10}y0', 'type': 'exit'}
                                for i in range(2)
                            ]
                        },
                        'scenario': {'monte_carlo_runs': 5},
                        'survival_rate': survival_rate,
                        'avg_evacuation_time': steps * 0.5,
                        'steps': steps,
                        'evacuated': evacuated,
                        'stuck': 0,
                        'dead': dead,
                        'avg_fire_damage': np.random.uniform(0, 0.5),
                        'score': survival_rate - (steps / 1000) * 0.5
                    }

                    f.write(json.dumps(result) + '\n')
                    all_results.append(result)

    print(f"  Created {len(all_results)} simulation results")

    # Create pairs (same-exit, cross-exit, cross-plan)
    pairs_per_split = {
        'train': int(num_plans * 100 * 0.7),
        'val': int(num_plans * 100 * 0.15),
        'test': int(num_plans * 100 * 0.15)
    }

    for split, num_pairs in pairs_per_split.items():
        pairs_file = os.path.join(output_dir, f'{split}_pairs.jsonl')

        with open(pairs_file, 'w') as f:
            for i in range(num_pairs):
                # Determine pair type based on ratios
                rand = np.random.random()
                if rand < 0.70:
                    pair_type = 'same_exit'
                    # Same plan, same exit
                    plan_id = np.random.randint(0, num_plans)
                    exit_id = np.random.randint(0, 3)
                    result_a = [r for r in all_results
                               if r['floor_plan_id'] == plan_id and r['exit_config_id'] == exit_id][0]
                    result_b = [r for r in all_results
                               if r['floor_plan_id'] == plan_id and r['exit_config_id'] == exit_id][-1]
                elif rand < 0.90:
                    pair_type = 'cross_exit'
                    # Same plan, different exits
                    plan_id = np.random.randint(0, num_plans)
                    exit_a, exit_b = np.random.choice(3, size=2, replace=False)
                    result_a = [r for r in all_results
                               if r['floor_plan_id'] == plan_id and r['exit_config_id'] == exit_a][0]
                    result_b = [r for r in all_results
                               if r['floor_plan_id'] == plan_id and r['exit_config_id'] == exit_b][0]
                else:
                    pair_type = 'cross_plan'
                    # Different plans
                    plan_a, plan_b = np.random.choice(num_plans, size=2, replace=False)
                    result_a = [r for r in all_results if r['floor_plan_id'] == plan_a][0]
                    result_b = [r for r in all_results if r['floor_plan_id'] == plan_b][0]

                score_a = result_a['score']
                score_b = result_b['score']
                label = 1 if score_a > score_b else 0

                pair = {
                    'floor_plan_id_a': result_a['floor_plan_id'],
                    'floor_plan_id_b': result_b['floor_plan_id'],
                    'config_a': result_a['config'],
                    'config_b': result_b['config'],
                    'scenario_a': result_a['scenario'],
                    'scenario_b': result_b['scenario'],
                    'score_a': score_a,
                    'score_b': score_b,
                    'label': label,
                    'label_confidence': min(1.0, abs(score_a - score_b) / 0.3),
                    'pair_type': pair_type
                }

                f.write(json.dumps(pair) + '\n')

        print(f"  Created {num_pairs} {split} pairs")

    print(f"Mock dataset created successfully!")


def test_analyzer():
    """Test the V5 analyzer with mock data"""
    print("\n" + "=" * 80)
    print("Testing V5 Data Analyzer")
    print("=" * 80 + "\n")

    # Create temporary directory for mock data
    temp_dir = tempfile.mkdtemp(prefix='v5_test_')
    print(f"Using temporary directory: {temp_dir}\n")

    try:
        # Create mock dataset
        create_mock_v5_dataset(temp_dir, num_plans=10)

        # Run analyzer
        print("\nRunning analyzer...")
        analyzer = V5DataAnalyzer(temp_dir)

        # Load data
        analyzer.load_data()

        # Run analysis
        report = analyzer.analyze()

        # Print report
        report.print_summary()

        # Save report
        report_path = os.path.join(temp_dir, 'analysis_report.json')
        analyzer.save_report(report_path)
        print(f"\nReport saved to: {report_path}")

        # Try to generate plots (will skip if matplotlib not available)
        try:
            plot_dir = os.path.join(temp_dir, 'plots')
            analyzer.plot_distributions(plot_dir)
            print(f"Plots saved to: {plot_dir}")
        except Exception as e:
            print(f"Plot generation skipped: {e}")

        # Test validation result
        if report.is_valid():
            print("\n✓ Dataset validation PASSED")
        else:
            print("\n✗ Dataset validation FAILED")
            print(f"  Errors: {len(report.get_errors())}")
            print(f"  Warnings: {len(report.get_warnings())}")

        # Show some statistics
        print("\nKey Statistics:")
        print(f"  Total configurations: {report.total_door_configs:,}")
        print(f"  Total pairs: {report.total_pairs:,}")
        print(f"  Avg survival rate: {report.performance_stats['survival_rate']['mean']:.3f}")

        pair_dist = report.pair_stats.get('type_distribution', {})
        print(f"\nPair Distribution:")
        for ptype, ratio in pair_dist.items():
            print(f"  {ptype}: {ratio:.2%}")

    finally:
        # Clean up
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    test_analyzer()
