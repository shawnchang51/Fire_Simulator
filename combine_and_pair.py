"""
Combine data from distributed server results and run pairing phase.

This script:
1. Merges simulation results from all v5_data_* directories
2. Combines floor plans
3. Runs the pairing phase to generate training pairs
4. Creates train/val/test splits
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict
from dataclasses import asdict
import logging

# Import from generation script
from generate_training_data_v5 import (
    GenerationConfigV5,
    HierarchicalSimulationResult,
    HierarchicalPairConstructor,
    PairConstructor,
    PairwiseLabel,
    PairWriter,
    DataValidator,
    create_dataset_splits,
    NumpyEncoder
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def combine_server_results(
    server_results_dir: str = 'server_results',
    output_dir: str = 'training_data_v5_combined'
):
    """
    Combine distributed server results into a single dataset.

    Args:
        server_results_dir: Directory containing v5_data_* subdirectories
        output_dir: Output directory for combined dataset
    """

    logger.info("=" * 70)
    logger.info("COMBINING DISTRIBUTED SERVER RESULTS")
    logger.info("=" * 70)

    base_path = Path(server_results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Find all v5_data_* directories
    data_dirs = sorted([d for d in base_path.iterdir()
                       if d.is_dir() and d.name.startswith('v5_data')])

    logger.info(f"Found {len(data_dirs)} server result directories:")
    for d in data_dirs:
        logger.info(f"  - {d.name}")

    if not data_dirs:
        raise ValueError(f"No v5_data_* directories found in {server_results_dir}")

    # Load config from first directory (all should be identical except plan_start/end)
    config_path = data_dirs[0] / 'config.json'
    with open(config_path, 'r') as f:
        base_config = json.load(f)

    logger.info(f"\nBase configuration loaded from {data_dirs[0].name}")

    # Create combined config
    combined_config = GenerationConfigV5(
        resplan_pkl_path=base_config['resplan_pkl_path'],
        cell_size_m=base_config['cell_size_m'],
        num_floor_plans=base_config['num_floor_plans'],
        exit_configs_per_plan=base_config['exit_configs_per_plan'],
        door_configs_per_exit=base_config['door_configs_per_exit'],
        monte_carlo_runs_per_config=base_config['monte_carlo_runs_per_config'],
        same_exit_ratio=base_config['same_exit_ratio'],
        cross_exit_ratio=base_config['cross_exit_ratio'],
        cross_plan_ratio=base_config['cross_plan_ratio'],
        pairs_per_plan=base_config['pairs_per_plan'],
        workers=base_config['workers'],
        num_exits_range=tuple(base_config['num_exits_range']),
        num_doors_range=tuple(base_config['num_doors_range']),
        min_door_spacing=base_config['min_door_spacing'],
        min_exit_spacing=base_config['min_exit_spacing'],
        occupant_density_range=tuple(base_config['occupant_density_range']),
        max_steps=base_config['max_steps'],
        num_fires_range=tuple(base_config['num_fires_range']),
        fire_spread_rate_range=tuple(base_config['fire_spread_rate_range']),
        fire_intensity_growth_range=tuple(base_config['fire_intensity_growth_range']),
        fire_discovery_delay_range=tuple(base_config['fire_discovery_delay_range']),
        fire_damage_threshold=base_config['fire_damage_threshold'],
        output_dir=output_dir,
        checkpoint_interval=base_config['checkpoint_interval'],
        seed=base_config['seed'],
        plan_start_idx=None,  # Full range
        plan_end_idx=None,
        evaluation_only=False  # We're doing pairing now
    )

    # Save combined config
    config_output = output_path / 'config.json'
    with open(config_output, 'w') as f:
        json.dump(asdict(combined_config), f, indent=2)
    logger.info(f"Saved combined config to {config_output}")

    # Step 1: Combine simulation results
    logger.info("\n[Step 1/3] Combining simulation results...")
    combined_results_file = output_path / 'simulation_results.jsonl'
    total_lines = 0

    with open(combined_results_file, 'w') as outfile:
        for data_dir in data_dirs:
            sim_results_file = data_dir / 'simulation_results.jsonl'

            if not sim_results_file.exists():
                logger.warning(f"  Missing simulation_results.jsonl in {data_dir.name}")
                continue

            logger.info(f"  Processing {data_dir.name}...")
            lines_written = 0

            with open(sim_results_file, 'r') as infile:
                for line in infile:
                    outfile.write(line)
                    lines_written += 1

            total_lines += lines_written
            logger.info(f"    Added {lines_written:,} simulation results")

    logger.info(f"  Total simulation results: {total_lines:,}")

    # Step 2: Combine floor plans
    logger.info("\n[Step 2/3] Combining floor plans...")
    floor_plans_output = output_path / 'floor_plans'
    floor_plans_output.mkdir(exist_ok=True)
    total_plans = 0

    for data_dir in data_dirs:
        floor_plans_dir = data_dir / 'floor_plans'

        if not floor_plans_dir.exists():
            logger.warning(f"  Missing floor_plans/ in {data_dir.name}")
            continue

        logger.info(f"  Processing {data_dir.name}...")
        plans_copied = 0

        for plan_file in floor_plans_dir.glob('plan_*.npz'):
            dest_file = floor_plans_output / plan_file.name
            shutil.copy2(plan_file, dest_file)
            plans_copied += 1

        total_plans += plans_copied
        logger.info(f"    Copied {plans_copied} floor plans")

    logger.info(f"  Total floor plans: {total_plans}")

    # Step 3: Load simulation results for pairing
    logger.info("\n[Step 3/3] Loading simulation results for pairing...")
    all_results = []

    with open(combined_results_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())

                # Convert to HierarchicalSimulationResult
                result = HierarchicalSimulationResult(
                    floor_plan_id=data['floor_plan_id'],
                    exit_config_id=data['exit_config_id'],
                    config_id=data['config_id'],
                    config=data['config'],
                    scenario=data.get('scenario', {}),
                    survival_rate=data['survival_rate'],
                    avg_evacuation_time=data['avg_evacuation_time'],
                    steps=data['steps'],
                    evacuated=data['evacuated'],
                    stuck=data['stuck'],
                    dead=data['dead'],
                    avg_fire_damage=data['avg_fire_damage'],
                    scenario_hash=data.get('scenario_hash', '')
                )
                all_results.append(result)

            except Exception as e:
                logger.warning(f"  Error loading line {i+1}: {e}")

    logger.info(f"  Loaded {len(all_results):,} simulation results")

    # Run pairing phase
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING PAIRING PHASE")
    logger.info("=" * 70)

    # Initialize pair constructor
    pair_constructor = PairConstructor(seed=combined_config.seed)
    hierarchical_pair_constructor = HierarchicalPairConstructor(
        pair_constructor,
        same_exit_ratio=combined_config.same_exit_ratio,
        cross_exit_ratio=combined_config.cross_exit_ratio,
        cross_plan_ratio=combined_config.cross_plan_ratio
    )

    # Calculate expected number of floor plans from simulation results
    unique_plans = len(set(r.floor_plan_id for r in all_results))
    total_pairs = combined_config.pairs_per_plan * unique_plans

    logger.info(f"Unique floor plans: {unique_plans}")
    logger.info(f"Target total pairs: {total_pairs:,}")
    logger.info(f"  Same-exit: {int(total_pairs * combined_config.same_exit_ratio):,}")
    logger.info(f"  Cross-exit: {int(total_pairs * combined_config.cross_exit_ratio):,}")
    logger.info(f"  Cross-plan: {int(total_pairs * combined_config.cross_plan_ratio):,}")

    # Normalize scores by floor plan
    logger.info("\nNormalizing scores by floor plan...")
    all_results = pair_constructor.normalize_scores_by_plan(all_results)

    # Construct pairs
    logger.info("\nConstructing hierarchical pairs...")
    all_pairs = hierarchical_pair_constructor.construct_hierarchical_pairs(
        all_results,
        num_pairs=total_pairs
    )

    # Save raw pairs
    raw_pairs_file = output_path / 'raw_pairs.jsonl'
    with open(raw_pairs_file, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair.to_dict(), cls=NumpyEncoder) + '\n')

    logger.info(f"  Saved {len(all_pairs):,} raw pairs to {raw_pairs_file}")

    # Balance labels
    logger.info("\nBalancing labels...")
    all_pairs = pair_constructor.balance_labels(all_pairs)

    logger.info(f"  Balanced pairs: {len(all_pairs):,}")

    # Statistics
    stats = pair_constructor.get_pair_statistics(all_pairs)
    logger.info(f"  Label distribution: {stats.get('label_1_ratio', 0):.1%} positive")
    logger.info(f"  Avg score diff: {stats.get('avg_score_diff', 0):.3f}")

    # Pair type breakdown
    type_dist = stats.get('pair_type_distribution', {})
    for pair_type, count in type_dist.items():
        logger.info(f"  {pair_type}: {count:,} pairs ({count/len(all_pairs):.1%})")

    # Create splits
    logger.info("\n" + "=" * 70)
    logger.info("CREATING DATASET SPLITS")
    logger.info("=" * 70)

    pair_dicts = [p.to_dict() for p in all_pairs]
    train_pairs, val_pairs, test_pairs = create_dataset_splits(
        pair_dicts, train_ratio=0.7, val_ratio=0.15, seed=combined_config.seed
    )

    # Validate
    logger.info("\nValidating dataset...")
    validator = DataValidator()
    report = validator.validate_dataset(train_pairs, val_pairs, test_pairs)
    report.print_summary()

    # Save splits
    logger.info("\nSaving dataset splits...")
    writer = PairWriter(output_dir)
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in train_pairs], 'train_pairs.jsonl')
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in val_pairs], 'val_pairs.jsonl')
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in test_pairs], 'test_pairs.jsonl')

    logger.info(f"  Train: {len(train_pairs):,} pairs")
    logger.info(f"  Val: {len(val_pairs):,} pairs")
    logger.info(f"  Test: {len(test_pairs):,} pairs")

    # Save metadata
    from datetime import datetime
    metadata = {
        'description': 'Combined hierarchical exit + door configuration comparison',
        'version': 'v5_combined',
        'config': asdict(combined_config),
        'statistics': {
            'total_floor_plans': unique_plans,
            'exit_configs_per_plan': combined_config.exit_configs_per_plan,
            'door_configs_per_exit': combined_config.door_configs_per_exit,
            'total_configurations': len(all_results),
            'total_pairs': len(all_pairs),
            'train_pairs': len(train_pairs),
            'val_pairs': len(val_pairs),
            'test_pairs': len(test_pairs)
        },
        'source_directories': [d.name for d in data_dirs],
        'validation': report.to_dict(),
        'generated_at': datetime.now().isoformat()
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"  Metadata saved to metadata.json")

    logger.info("\n" + "=" * 70)
    logger.info("COMBINATION AND PAIRING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Total simulation results: {len(all_results):,}")
    logger.info(f"Total pairs: {len(all_pairs):,}")
    logger.info(f"  Train: {len(train_pairs):,}")
    logger.info(f"  Val: {len(val_pairs):,}")
    logger.info(f"  Test: {len(test_pairs):,}")
    logger.info("=" * 70)

    return output_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Combine distributed server results and run pairing phase'
    )
    parser.add_argument('--server-results-dir', type=str,
                       default='server_results',
                       help='Directory containing v5_data_* subdirectories')
    parser.add_argument('--output-dir', type=str,
                       default='training_data_v5_combined',
                       help='Output directory for combined dataset')

    args = parser.parse_args()

    output_dir = combine_server_results(
        server_results_dir=args.server_results_dir,
        output_dir=args.output_dir
    )

    print(f"\n✅ Combined training data saved to: {output_dir}")


if __name__ == '__main__':
    main()
