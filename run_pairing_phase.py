"""
Run pairing phase on combined simulation results from distributed execution.

This script loads simulation_results.jsonl from multiple distributed workers,
reconstructs the data structures, and runs Phase 3-4:
- Phase 3: Normalize scores and construct hierarchical pairs
- Phase 4: Validate and split into train/val/test

Usage:
    python run_pairing_phase.py --input-file combined/simulation_results.jsonl --output-dir final_output --config-json combined/config.json
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from dataclasses import asdict
from typing import List, Dict
from collections import defaultdict

import numpy as np

# Import components from generate_training_data_v5
from generate_training_data_v5 import (
    GenerationConfigV5,
    HierarchicalSimulationResult,
    HierarchicalPairConstructor,
    NumpyEncoder
)
from pair_constructor import PairConstructor, PairwiseLabel, PairWriter
from data_validator import DataValidator, create_dataset_splits


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_simulation_results(jsonl_path: str) -> List[HierarchicalSimulationResult]:
    """
    Load simulation results from JSONL file and reconstruct HierarchicalSimulationResult objects.

    Args:
        jsonl_path: Path to simulation_results.jsonl file

    Returns:
        List of HierarchicalSimulationResult objects
    """
    logger.info(f"Loading simulation results from {jsonl_path}...")

    results = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)

                # Reconstruct HierarchicalSimulationResult
                result = HierarchicalSimulationResult(
                    floor_plan_id=data['floor_plan_id'],
                    exit_config_id=data['exit_config_id'],
                    config_id=data['config_id'],
                    config=data['config'],
                    scenario=data['scenario'],
                    survival_rate=data['survival_rate'],
                    avg_evacuation_time=data['avg_evacuation_time'],
                    steps=data['steps'],
                    evacuated=data['evacuated'],
                    stuck=data['stuck'],
                    dead=data['dead'],
                    avg_fire_damage=data['avg_fire_damage'],
                    scenario_hash=data.get('scenario_hash', '')
                )
                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue

    logger.info(f"Loaded {len(results)} simulation results")

    # Verify unique (floor_plan_id, exit_config_id, config_id, scenario_hash) tuples
    unique_keys = set()
    for r in results:
        key = (r.floor_plan_id, r.exit_config_id, r.config_id, r.scenario_hash)
        if key in unique_keys:
            logger.warning(f"Duplicate result found: {key}")
        unique_keys.add(key)

    logger.info(f"Found {len(unique_keys)} unique configurations")

    # Print statistics
    floor_plan_ids = set(r.floor_plan_id for r in results)
    logger.info(f"Floor plans: {len(floor_plan_ids)}")
    logger.info(f"Floor plan ID range: [{min(floor_plan_ids)}, {max(floor_plan_ids)}]")

    return results


def construct_hierarchical_pairs(
    results: List[HierarchicalSimulationResult],
    config: GenerationConfigV5
) -> List[PairwiseLabel]:
    """
    Run Phase 3: Normalize scores and construct hierarchical pairs.

    Args:
        results: List of simulation results
        config: Generation configuration

    Returns:
        List of PairwiseLabel objects
    """
    logger.info("\n[Phase 3] Constructing three-tier pairwise labels...")

    # Initialize components
    pair_constructor = PairConstructor(seed=config.seed)
    hierarchical_pair_constructor = HierarchicalPairConstructor(
        pair_constructor,
        same_exit_ratio=config.same_exit_ratio,
        cross_exit_ratio=config.cross_exit_ratio,
        cross_plan_ratio=config.cross_plan_ratio
    )

    # Calculate total pairs
    floor_plan_ids = set(r.floor_plan_id for r in results)
    total_pairs = config.pairs_per_plan * len(floor_plan_ids)

    logger.info(f"  Target total pairs: {total_pairs}")
    logger.info(f"  Same-exit: {int(total_pairs * config.same_exit_ratio)}")
    logger.info(f"  Cross-exit: {int(total_pairs * config.cross_exit_ratio)}")
    logger.info(f"  Cross-plan: {int(total_pairs * config.cross_plan_ratio)}")

    # Normalize scores by floor plan before pairing
    logger.info(f"  Normalizing scores by floor plan...")
    normalized_results = pair_constructor.normalize_scores_by_plan(results)

    # Construct pairs
    all_pairs = hierarchical_pair_constructor.construct_hierarchical_pairs(
        normalized_results,
        num_pairs=total_pairs
    )

    # Balance labels
    all_pairs = pair_constructor.balance_labels(all_pairs)

    logger.info(f"  Constructed {len(all_pairs)} pairwise labels")

    # Statistics
    stats = pair_constructor.get_pair_statistics(all_pairs)
    logger.info(f"  Label distribution: {stats.get('label_1_ratio', 0):.1%} positive")
    logger.info(f"  Avg score diff: {stats.get('avg_score_diff', 0):.3f}")

    # Pair type breakdown
    type_dist = stats.get('pair_type_distribution', {})
    for pair_type, count in type_dist.items():
        logger.info(f"  {pair_type}: {count} pairs ({count/len(all_pairs):.1%})")

    return all_pairs


def validate_and_save(
    pairs: List[PairwiseLabel],
    config: GenerationConfigV5,
    output_dir: str,
    num_floor_plans: int
):
    """
    Run Phase 4: Validate and save final splits to disk.

    Args:
        pairs: List of pairwise labels
        config: Generation configuration
        output_dir: Output directory
        num_floor_plans: Number of floor plans in dataset
    """
    logger.info("\n[Phase 4] Validating and saving...")

    os.makedirs(output_dir, exist_ok=True)

    # Convert to dicts
    pair_dicts = [p.to_dict() for p in pairs]

    # Save raw pairs first
    raw_pairs_file = os.path.join(output_dir, 'raw_pairs.jsonl')
    with open(raw_pairs_file, 'w') as f:
        for pair in pair_dicts:
            f.write(json.dumps(pair, cls=NumpyEncoder) + '\n')
    logger.info(f"  Raw pairs saved to {raw_pairs_file}")

    # Create splits
    train_pairs, val_pairs, test_pairs = create_dataset_splits(
        pair_dicts, train_ratio=0.7, val_ratio=0.15, seed=config.seed
    )

    # Validate
    validator = DataValidator()
    report = validator.validate_dataset(train_pairs, val_pairs, test_pairs)
    report.print_summary()

    # Save pairs
    writer = PairWriter(output_dir)
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in train_pairs], 'train_pairs.jsonl')
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in val_pairs], 'val_pairs.jsonl')
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in test_pairs], 'test_pairs.jsonl')

    logger.info(f"  Saved {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test pairs")

    # Save metadata
    metadata = {
        'description': 'Hierarchical exit + door configuration comparison (distributed execution)',
        'version': 'v5',
        'config': asdict(config),
        'statistics': {
            'total_floor_plans': num_floor_plans,
            'exit_configs_per_plan': config.exit_configs_per_plan,
            'door_configs_per_exit': config.door_configs_per_exit,
            'monte_carlo_runs_per_config': config.monte_carlo_runs_per_config,
            'total_pairs': len(pairs),
            'train_pairs': len(train_pairs),
            'val_pairs': len(val_pairs),
            'test_pairs': len(test_pairs)
        },
        'validation': report.to_dict(),
        'generated_at': datetime.now().isoformat()
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"  Metadata saved to metadata.json")


def main():
    parser = argparse.ArgumentParser(
        description='Run pairing phase on combined simulation results from distributed execution'
    )

    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to combined simulation_results.jsonl file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for final dataset')
    parser.add_argument('--config-json', type=str, required=True,
                        help='Path to config.json from evaluation phase')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Pairing Phase - Distributed Training Data Generation V5")
    logger.info("=" * 60)
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Config file: {args.config_json}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)

    # Load config
    logger.info("\nLoading configuration...")
    with open(args.config_json, 'r') as f:
        config_dict = json.load(f)

    # Convert lists to tuples for range parameters
    if 'num_exits_range' in config_dict and isinstance(config_dict['num_exits_range'], list):
        config_dict['num_exits_range'] = tuple(config_dict['num_exits_range'])
    if 'num_doors_range' in config_dict and isinstance(config_dict['num_doors_range'], list):
        config_dict['num_doors_range'] = tuple(config_dict['num_doors_range'])
    if 'occupant_density_range' in config_dict and isinstance(config_dict['occupant_density_range'], list):
        config_dict['occupant_density_range'] = tuple(config_dict['occupant_density_range'])
    if 'num_fires_range' in config_dict and isinstance(config_dict['num_fires_range'], list):
        config_dict['num_fires_range'] = tuple(config_dict['num_fires_range'])
    if 'fire_spread_rate_range' in config_dict and isinstance(config_dict['fire_spread_rate_range'], list):
        config_dict['fire_spread_rate_range'] = tuple(config_dict['fire_spread_rate_range'])
    if 'fire_intensity_growth_range' in config_dict and isinstance(config_dict['fire_intensity_growth_range'], list):
        config_dict['fire_intensity_growth_range'] = tuple(config_dict['fire_intensity_growth_range'])
    if 'fire_discovery_delay_range' in config_dict and isinstance(config_dict['fire_discovery_delay_range'], list):
        config_dict['fire_discovery_delay_range'] = tuple(config_dict['fire_discovery_delay_range'])

    config = GenerationConfigV5(**config_dict)
    logger.info(f"  Loaded config (seed: {config.seed})")

    # Load simulation results
    results = load_simulation_results(args.input_file)

    if not results:
        logger.error("No simulation results loaded. Exiting.")
        sys.exit(1)

    # Count unique floor plans
    num_floor_plans = len(set(r.floor_plan_id for r in results))

    # Run Phase 3: Construct pairs
    pairs = construct_hierarchical_pairs(results, config)

    # Run Phase 4: Validate and save
    validate_and_save(pairs, config, args.output_dir, num_floor_plans)

    logger.info("\n" + "=" * 60)
    logger.info("Pairing phase complete!")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
