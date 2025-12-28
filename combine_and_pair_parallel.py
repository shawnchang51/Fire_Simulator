"""
Parallel version: Combine data from distributed server results and run pairing phase.

This script:
1. Merges simulation results from all v5_data_* directories
2. Combines floor plans
3. Runs PARALLELIZED pairing phase to generate training pairs
4. Creates train/val/test splits

Key optimization: Pair generation is parallelized across multiple workers.
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import asdict
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import random

# Import from generation script
from generate_training_data_v5 import (
    GenerationConfigV5,
    HierarchicalSimulationResult,
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


def generate_same_exit_pairs_worker(args: Tuple) -> List[Dict]:
    """Worker function to generate same-exit pairs for one group."""
    plan_id, exit_id, group_results, pairs_per_group, seed = args

    # Recreate pair constructor with worker-specific seed
    pair_constructor = PairConstructor(seed=seed + plan_id * 10000 + exit_id)

    group_pairs = pair_constructor.construct_pairs(
        group_results,
        num_pairs=pairs_per_group,
        strategy='mixed',
        within_plan_ratio=1.0
    )

    # Tag and convert to dicts
    return [p.to_dict() for p in group_pairs]


def generate_cross_exit_pairs_worker(args: Tuple) -> List[Dict]:
    """Worker function to generate cross-exit pairs for one plan."""
    plan_id, plan_results, num_pairs, margin, seed = args

    # Recreate pair constructor
    pair_constructor = PairConstructor(seed=seed + plan_id * 20000)

    # Group by exit config within this plan
    by_exit = defaultdict(list)
    for r in plan_results:
        by_exit[r.exit_config_id].append(r)

    # Need at least 2 different exit configs
    if len(by_exit) < 2:
        return []

    pairs = []
    exit_ids = list(by_exit.keys())
    attempts = 0
    max_attempts = num_pairs * 5

    rng = np.random.default_rng(seed + plan_id * 20000)

    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1

        # Pick two different exit configs
        exit_a, exit_b = rng.choice(exit_ids, size=2, replace=False)

        # Pick random result from each
        result_a = by_exit[exit_a][rng.integers(len(by_exit[exit_a]))]
        result_b = by_exit[exit_b][rng.integers(len(by_exit[exit_b]))]

        # Create pair
        pair = pair_constructor._create_pair(
            result_a, result_b, 'cross_exit', 'random'
        )

        if pair is not None:
            pairs.append(pair.to_dict())

    return pairs


def generate_cross_plan_pairs_worker(args: Tuple) -> List[Dict]:
    """Worker function to generate a batch of cross-plan pairs."""
    batch_id, by_plan, plan_ids, num_pairs, margin, seed = args

    # Recreate pair constructor
    pair_constructor = PairConstructor(seed=seed + batch_id * 30000)

    pairs = []
    attempts = 0
    max_attempts = num_pairs * 5

    rng = np.random.default_rng(seed + batch_id * 30000)

    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1

        # Pick two different plans
        plan_a, plan_b = rng.choice(plan_ids, size=2, replace=False)

        # Pick random result from each
        result_a = by_plan[plan_a][rng.integers(len(by_plan[plan_a]))]
        result_b = by_plan[plan_b][rng.integers(len(by_plan[plan_b]))]

        # Create pair
        pair = pair_constructor._create_pair(
            result_a, result_b, 'cross_plan', 'random'
        )

        if pair is not None:
            pairs.append(pair.to_dict())

    return pairs


def construct_hierarchical_pairs_parallel(
    results: List[HierarchicalSimulationResult],
    num_pairs: int,
    same_exit_ratio: float,
    cross_exit_ratio: float,
    cross_plan_ratio: float,
    seed: int,
    workers: int = 64
) -> List[PairwiseLabel]:
    """
    Parallel version of hierarchical pair construction.

    Args:
        results: List of simulation results
        num_pairs: Total number of pairs to generate
        same_exit_ratio: Ratio of same-exit pairs
        cross_exit_ratio: Ratio of cross-exit pairs
        cross_plan_ratio: Ratio of cross-plan pairs
        seed: Random seed
        workers: Number of parallel workers

    Returns:
        List of PairwiseLabel objects
    """
    logger.info(f"  Using {workers} parallel workers for pair generation")

    # Group results by plan and exit config
    by_plan_exit = defaultdict(list)
    by_plan = defaultdict(list)

    for result in results:
        key = (result.floor_plan_id, result.exit_config_id)
        by_plan_exit[key].append(result)
        by_plan[result.floor_plan_id].append(result)

    # Calculate pair counts
    num_same_exit = int(num_pairs * same_exit_ratio)
    num_cross_exit = int(num_pairs * cross_exit_ratio)
    num_cross_plan = num_pairs - num_same_exit - num_cross_exit

    all_pairs = []

    # 1. Same-exit pairs (parallel by group)
    logger.info(f"  Generating {num_same_exit:,} same-exit pairs in parallel...")
    groups = [(k, v) for k, v in by_plan_exit.items() if len(v) >= 2]

    if groups:
        pairs_per_group = max(1, num_same_exit // len(groups))

        tasks = [
            (plan_id, exit_id, group_results, pairs_per_group, seed)
            for (plan_id, exit_id), group_results in groups
        ]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(generate_same_exit_pairs_worker, task) for task in tasks]

            for future in as_completed(futures):
                try:
                    pairs_dicts = future.result()
                    # Tag as same-exit
                    for p in pairs_dicts:
                        p['pair_type'] = 'same_exit'
                    all_pairs.extend(pairs_dicts)
                except Exception as e:
                    logger.warning(f"Same-exit worker failed: {e}")

        logger.info(f"    Generated {len([p for p in all_pairs if p.get('pair_type') == 'same_exit']):,} same-exit pairs")

    # 2. Cross-exit pairs (parallel by plan)
    logger.info(f"  Generating {num_cross_exit:,} cross-exit pairs in parallel...")
    plans_with_multi_exits = {
        plan_id: plan_results
        for plan_id, plan_results in by_plan.items()
        if len(set(r.exit_config_id for r in plan_results)) >= 2
    }

    if plans_with_multi_exits:
        pairs_per_plan = max(1, num_cross_exit // len(plans_with_multi_exits))

        tasks = [
            (plan_id, plan_results, pairs_per_plan, 0.002, seed)
            for plan_id, plan_results in plans_with_multi_exits.items()
        ]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(generate_cross_exit_pairs_worker, task) for task in tasks]

            for future in as_completed(futures):
                try:
                    pairs_dicts = future.result()
                    all_pairs.extend(pairs_dicts)
                except Exception as e:
                    logger.warning(f"Cross-exit worker failed: {e}")

        logger.info(f"    Generated {len([p for p in all_pairs if p.get('pair_type') == 'cross_exit']):,} cross-exit pairs")

    # 3. Cross-plan pairs (parallel batches)
    logger.info(f"  Generating {num_cross_plan:,} cross-plan pairs in parallel...")
    plan_ids = list(by_plan.keys())

    if len(plan_ids) >= 2 and num_cross_plan > 0:
        # Split work into batches for parallel processing
        num_batches = min(workers, num_cross_plan // 100)
        pairs_per_batch = num_cross_plan // num_batches if num_batches > 0 else num_cross_plan

        tasks = [
            (batch_id, by_plan, plan_ids, pairs_per_batch, 0.002, seed)
            for batch_id in range(num_batches)
        ]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(generate_cross_plan_pairs_worker, task) for task in tasks]

            for future in as_completed(futures):
                try:
                    pairs_dicts = future.result()
                    all_pairs.extend(pairs_dicts)
                except Exception as e:
                    logger.warning(f"Cross-plan worker failed: {e}")

        logger.info(f"    Generated {len([p for p in all_pairs if p.get('pair_type') == 'cross_plan']):,} cross-plan pairs")

    # Convert back to PairwiseLabel objects and shuffle
    logger.info(f"  Converting {len(all_pairs):,} pair dicts to PairwiseLabel objects...")
    pair_objects = [PairwiseLabel.from_dict(p) for p in all_pairs]

    random.seed(seed)
    random.shuffle(pair_objects)

    logger.info(f"  Total pairs constructed: {len(pair_objects):,}")
    return pair_objects


def combine_server_results(
    server_results_dir: str = 'server_results',
    output_dir: str = 'training_data_v5_combined',
    workers: int = 64
):
    """
    Combine distributed server results and run PARALLEL pairing.

    Args:
        server_results_dir: Directory containing v5_data_* subdirectories
        output_dir: Output directory for combined dataset
        workers: Number of parallel workers for pair generation
    """

    logger.info("=" * 70)
    logger.info("COMBINING DISTRIBUTED SERVER RESULTS (PARALLEL)")
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

    # Load config from first directory
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
        workers=workers,
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
        plan_start_idx=None,
        plan_end_idx=None,
        evaluation_only=False
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
            if not dest_file.exists():  # Skip duplicates
                shutil.copy2(plan_file, dest_file)
                plans_copied += 1

        total_plans += plans_copied
        logger.info(f"    Copied {plans_copied} floor plans")

    logger.info(f"  Total floor plans: {total_plans}")

    # Step 3: Load simulation results
    logger.info("\n[Step 3/3] Loading simulation results for pairing...")
    all_results = []

    with open(combined_results_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())

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

    # Run PARALLEL pairing phase
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING PARALLEL PAIRING PHASE")
    logger.info("=" * 70)

    unique_plans = len(set(r.floor_plan_id for r in all_results))
    total_pairs = combined_config.pairs_per_plan * unique_plans

    logger.info(f"Unique floor plans: {unique_plans}")
    logger.info(f"Target total pairs: {total_pairs:,}")
    logger.info(f"  Same-exit: {int(total_pairs * combined_config.same_exit_ratio):,}")
    logger.info(f"  Cross-exit: {int(total_pairs * combined_config.cross_exit_ratio):,}")
    logger.info(f"  Cross-plan: {int(total_pairs * combined_config.cross_plan_ratio):,}")

    # Normalize scores by floor plan
    logger.info("\nNormalizing scores by floor plan...")
    pair_constructor = PairConstructor(seed=combined_config.seed)
    all_results = pair_constructor.normalize_scores_by_plan(all_results)

    # Construct pairs in PARALLEL
    logger.info("\nConstructing hierarchical pairs (PARALLEL)...")
    all_pairs = construct_hierarchical_pairs_parallel(
        all_results,
        num_pairs=total_pairs,
        same_exit_ratio=combined_config.same_exit_ratio,
        cross_exit_ratio=combined_config.cross_exit_ratio,
        cross_plan_ratio=combined_config.cross_plan_ratio,
        seed=combined_config.seed,
        workers=workers
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
        'description': 'Combined hierarchical exit + door configuration comparison (PARALLEL)',
        'version': 'v5_combined_parallel',
        'config': asdict(combined_config),
        'statistics': {
            'total_floor_plans': unique_plans,
            'exit_configs_per_plan': combined_config.exit_configs_per_plan,
            'door_configs_per_exit': combined_config.door_configs_per_exit,
            'total_configurations': len(all_results),
            'total_pairs': len(all_pairs),
            'train_pairs': len(train_pairs),
            'val_pairs': len(val_pairs),
            'test_pairs': len(test_pairs),
            'parallel_workers': workers
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
        description='Combine distributed server results and run PARALLEL pairing phase'
    )
    parser.add_argument('--server-results-dir', type=str,
                       default='server_results',
                       help='Directory containing v5_data_* subdirectories')
    parser.add_argument('--output-dir', type=str,
                       default='training_data_v5_combined',
                       help='Output directory for combined dataset')
    parser.add_argument('--workers', type=int,
                       default=64,
                       help='Number of parallel workers for pair generation')

    args = parser.parse_args()

    output_dir = combine_server_results(
        server_results_dir=args.server_results_dir,
        output_dir=args.output_dir,
        workers=args.workers
    )

    print(f"\n✅ Combined training data saved to: {output_dir}")


if __name__ == '__main__':
    main()
