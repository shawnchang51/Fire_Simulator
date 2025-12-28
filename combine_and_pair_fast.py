"""
Fast in-memory version: Combine data from distributed server results and run pairing phase.

Optimizations:
1. Load all data directly into RAM (no intermediate file writes)
2. Use orjson for fast JSON parsing (5-10x faster)
3. Single-threaded pair generation (memory operations are fast, no pickle overhead)
4. Minimal IO operations

Designed for high-memory machines (768GB+ RAM).
"""

import os
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict
import logging
from collections import defaultdict
import random
import time

# Try to use orjson for faster JSON parsing, fallback to standard json
try:
    import orjson
    def json_loads(s):
        return orjson.loads(s)
    def json_dumps(obj, indent=None):
        opts = orjson.OPT_SERIALIZE_NUMPY
        if indent:
            opts |= orjson.OPT_INDENT_2
        return orjson.dumps(obj, option=opts).decode('utf-8')
    USING_ORJSON = True
except ImportError:
    import json
    json_loads = json.loads
    def json_dumps(obj, indent=None):
        return json.dumps(obj, indent=indent, default=str)
    USING_ORJSON = False

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


def construct_hierarchical_pairs_fast(
    results: List[HierarchicalSimulationResult],
    num_pairs: int,
    same_exit_ratio: float,
    cross_exit_ratio: float,
    cross_plan_ratio: float,
    seed: int
) -> List[PairwiseLabel]:
    """
    Fast in-memory hierarchical pair construction.
    
    All operations happen in RAM - no pickle serialization overhead.
    """
    logger.info(f"  Constructing pairs in-memory (single-threaded, fast)")
    
    rng = np.random.default_rng(seed)
    pair_constructor = PairConstructor(seed=seed)
    
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
    
    # 1. Same-exit pairs
    logger.info(f"  Generating {num_same_exit:,} same-exit pairs...")
    start = time.time()
    
    groups = [(k, v) for k, v in by_plan_exit.items() if len(v) >= 2]
    if groups:
        pairs_per_group = max(1, num_same_exit // len(groups))
        
        for (plan_id, exit_id), group_results in groups:
            group_pairs = pair_constructor.construct_pairs(
                group_results,
                num_pairs=pairs_per_group,
                strategy='mixed',
                within_plan_ratio=1.0
            )
            for p in group_pairs:
                p.pair_type = 'same_exit'
            all_pairs.extend(group_pairs)
    
    logger.info(f"    Generated {len([p for p in all_pairs if p.pair_type == 'same_exit']):,} same-exit pairs in {time.time()-start:.1f}s")
    
    # 2. Cross-exit pairs
    logger.info(f"  Generating {num_cross_exit:,} cross-exit pairs...")
    start = time.time()
    
    plans_with_multi_exits = {
        plan_id: plan_results
        for plan_id, plan_results in by_plan.items()
        if len(set(r.exit_config_id for r in plan_results)) >= 2
    }
    
    cross_exit_count = 0
    if plans_with_multi_exits:
        pairs_per_plan = max(1, num_cross_exit // len(plans_with_multi_exits))
        
        for plan_id, plan_results in plans_with_multi_exits.items():
            # Group by exit config
            by_exit = defaultdict(list)
            for r in plan_results:
                by_exit[r.exit_config_id].append(r)
            
            exit_ids = list(by_exit.keys())
            attempts = 0
            max_attempts = pairs_per_plan * 5
            generated = 0
            
            while generated < pairs_per_plan and attempts < max_attempts:
                attempts += 1
                
                exit_a, exit_b = rng.choice(exit_ids, size=2, replace=False)
                result_a = by_exit[exit_a][rng.integers(len(by_exit[exit_a]))]
                result_b = by_exit[exit_b][rng.integers(len(by_exit[exit_b]))]
                
                # Create pair directly (bypass scenario_hash check for cross-exit)
                score_a = result_a.effective_score
                score_b = result_b.effective_score
                score_diff = score_a - score_b
                
                # Skip ambiguous pairs
                if abs(score_diff) < pair_constructor.margin:
                    continue
                
                label = 1 if score_diff > 0 else 0
                confidence = min(1.0, abs(score_diff) / 0.3)
                
                pair = PairwiseLabel(
                    floor_plan_id_a=result_a.floor_plan_id,
                    floor_plan_id_b=result_b.floor_plan_id,
                    config_a=result_a.config,
                    config_b=result_b.config,
                    scenario_a=result_a.scenario,
                    scenario_b=result_b.scenario,
                    score_a=result_a.score,
                    score_b=result_b.score,
                    label=label,
                    label_confidence=confidence,
                    pair_type='cross_exit_random'
                )
                all_pairs.append(pair)
                generated += 1
                cross_exit_count += 1
    
    logger.info(f"    Generated {cross_exit_count:,} cross-exit pairs in {time.time()-start:.1f}s")
    
    # 3. Cross-plan pairs
    logger.info(f"  Generating {num_cross_plan:,} cross-plan pairs...")
    start = time.time()
    
    plan_ids = list(by_plan.keys())
    cross_plan_count = 0
    
    if len(plan_ids) >= 2 and num_cross_plan > 0:
        attempts = 0
        max_attempts = num_cross_plan * 5
        
        while cross_plan_count < num_cross_plan and attempts < max_attempts:
            attempts += 1
            
            plan_a, plan_b = rng.choice(plan_ids, size=2, replace=False)
            result_a = by_plan[plan_a][rng.integers(len(by_plan[plan_a]))]
            result_b = by_plan[plan_b][rng.integers(len(by_plan[plan_b]))]
            
            pair = pair_constructor._create_pair(result_a, result_b, 'cross_plan', 'random')
            if pair is not None:
                all_pairs.append(pair)
                cross_plan_count += 1
    
    logger.info(f"    Generated {cross_plan_count:,} cross-plan pairs in {time.time()-start:.1f}s")
    
    # Shuffle
    random.seed(seed)
    random.shuffle(all_pairs)
    
    logger.info(f"  Total pairs constructed: {len(all_pairs):,}")
    return all_pairs


def combine_server_results_fast(
    server_results_dir: str = 'server_results',
    output_dir: str = 'training_data_v5_combined',
    save_combined_jsonl: bool = True
):
    """
    Fast in-memory combination of distributed server results.
    
    All data is loaded directly into RAM, processed, then written once.
    """
    total_start = time.time()
    
    logger.info("=" * 70)
    logger.info("COMBINING DISTRIBUTED SERVER RESULTS (FAST IN-MEMORY)")
    logger.info("=" * 70)
    logger.info(f"Using {'orjson (fast)' if USING_ORJSON else 'standard json (slower)'}")
    
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
        base_config = json_loads(f.read())
    
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
        workers=1,  # Not used in fast version
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
        f.write(json_dumps(asdict(combined_config), indent=2))
    logger.info(f"Saved combined config to {config_output}")
    
    # Step 1: Load ALL simulation results directly into memory
    logger.info("\n[Step 1/3] Loading all simulation results into RAM...")
    start = time.time()
    
    all_results = []
    raw_lines = []  # Keep raw lines if we need to save combined JSONL
    
    for data_dir in data_dirs:
        sim_results_file = data_dir / 'simulation_results.jsonl'
        
        if not sim_results_file.exists():
            logger.warning(f"  Missing simulation_results.jsonl in {data_dir.name}")
            continue
        
        logger.info(f"  Loading {data_dir.name}...")
        lines_loaded = 0
        
        with open(sim_results_file, 'r') as f:
            for line in f:
                try:
                    if save_combined_jsonl:
                        raw_lines.append(line)
                    
                    data = json_loads(line.strip())
                    
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
                    lines_loaded += 1
                    
                except Exception as e:
                    logger.warning(f"    Error loading line: {e}")
        
        logger.info(f"    Loaded {lines_loaded:,} results")
    
    logger.info(f"  Total: {len(all_results):,} results loaded in {time.time()-start:.1f}s")
    
    # Optionally save combined JSONL (for future reference)
    if save_combined_jsonl and raw_lines:
        logger.info("  Saving combined simulation_results.jsonl...")
        combined_results_file = output_path / 'simulation_results.jsonl'
        with open(combined_results_file, 'w') as f:
            f.writelines(raw_lines)
        logger.info(f"    Saved {len(raw_lines):,} lines")
        del raw_lines  # Free memory
    
    # Step 2: Copy floor plans
    logger.info("\n[Step 2/3] Copying floor plans...")
    start = time.time()
    
    floor_plans_output = output_path / 'floor_plans'
    floor_plans_output.mkdir(exist_ok=True)
    total_plans = 0
    
    for data_dir in data_dirs:
        floor_plans_dir = data_dir / 'floor_plans'
        
        if not floor_plans_dir.exists():
            logger.warning(f"  Missing floor_plans/ in {data_dir.name}")
            continue
        
        plans_copied = 0
        for plan_file in floor_plans_dir.glob('plan_*.npz'):
            dest_file = floor_plans_output / plan_file.name
            if not dest_file.exists():
                shutil.copy2(plan_file, dest_file)
                plans_copied += 1
        
        total_plans += plans_copied
        if plans_copied > 0:
            logger.info(f"  {data_dir.name}: {plans_copied} plans")
    
    logger.info(f"  Total: {total_plans} floor plans copied in {time.time()-start:.1f}s")
    
    # Step 3: Pair generation (in-memory)
    logger.info("\n[Step 3/3] Generating pairs...")
    logger.info("=" * 70)
    logger.info("RUNNING IN-MEMORY PAIRING PHASE")
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
    start = time.time()
    pair_constructor = PairConstructor(seed=combined_config.seed)
    all_results = pair_constructor.normalize_scores_by_plan(all_results)
    logger.info(f"  Normalized in {time.time()-start:.1f}s")
    
    # Construct pairs (fast in-memory)
    logger.info("\nConstructing hierarchical pairs (in-memory)...")
    start = time.time()
    all_pairs = construct_hierarchical_pairs_fast(
        all_results,
        num_pairs=total_pairs,
        same_exit_ratio=combined_config.same_exit_ratio,
        cross_exit_ratio=combined_config.cross_exit_ratio,
        cross_plan_ratio=combined_config.cross_plan_ratio,
        seed=combined_config.seed
    )
    logger.info(f"  Pair construction took {time.time()-start:.1f}s")
    
    # Save raw pairs
    logger.info("\nSaving raw pairs...")
    start = time.time()
    raw_pairs_file = output_path / 'raw_pairs.jsonl'
    with open(raw_pairs_file, 'w') as f:
        for pair in all_pairs:
            f.write(json_dumps(pair.to_dict()) + '\n')
    logger.info(f"  Saved {len(all_pairs):,} raw pairs in {time.time()-start:.1f}s")
    
    # Balance labels
    logger.info("\nBalancing labels...")
    start = time.time()
    all_pairs = pair_constructor.balance_labels(all_pairs)
    logger.info(f"  Balanced to {len(all_pairs):,} pairs in {time.time()-start:.1f}s")
    
    # Statistics
    stats = pair_constructor.get_pair_statistics(all_pairs)
    logger.info(f"  Label distribution: {stats.get('label_1_ratio', 0):.1%} positive")
    logger.info(f"  Avg score diff: {stats.get('avg_score_diff', 0):.3f}")
    
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
    start = time.time()
    writer = PairWriter(output_dir)
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in train_pairs], 'train_pairs.jsonl')
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in val_pairs], 'val_pairs.jsonl')
    writer.write_jsonl([PairwiseLabel.from_dict(p) for p in test_pairs], 'test_pairs.jsonl')
    
    logger.info(f"  Train: {len(train_pairs):,} pairs")
    logger.info(f"  Val: {len(val_pairs):,} pairs")
    logger.info(f"  Test: {len(test_pairs):,} pairs")
    logger.info(f"  Saved in {time.time()-start:.1f}s")
    
    # Save metadata
    from datetime import datetime
    metadata = {
        'description': 'Combined hierarchical exit + door configuration comparison (FAST IN-MEMORY)',
        'version': 'v5_combined_fast',
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
            'using_orjson': USING_ORJSON
        },
        'source_directories': [d.name for d in data_dirs],
        'validation': report.to_dict(),
        'generated_at': datetime.now().isoformat(),
        'total_time_seconds': time.time() - total_start
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        f.write(json_dumps(metadata, indent=2))
    
    logger.info(f"  Metadata saved to metadata.json")
    
    total_time = time.time() - total_start
    logger.info("\n" + "=" * 70)
    logger.info("COMBINATION AND PAIRING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Total simulation results: {len(all_results):,}")
    logger.info(f"Total pairs: {len(all_pairs):,}")
    logger.info(f"  Train: {len(train_pairs):,}")
    logger.info(f"  Val: {len(val_pairs):,}")
    logger.info(f"  Test: {len(test_pairs):,}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("=" * 70)
    
    return output_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fast in-memory combination of distributed server results'
    )
    parser.add_argument('--server-results-dir', type=str,
                       default='server_results',
                       help='Directory containing v5_data_* subdirectories')
    parser.add_argument('--output-dir', type=str,
                       default='training_data_v5_combined',
                       help='Output directory for combined dataset')
    parser.add_argument('--no-save-combined', action='store_true',
                       help='Skip saving combined simulation_results.jsonl (saves memory)')
    
    args = parser.parse_args()
    
    output_dir = combine_server_results_fast(
        server_results_dir=args.server_results_dir,
        output_dir=args.output_dir,
        save_combined_jsonl=not args.no_save_combined
    )
    
    print(f"\n✅ Combined training data saved to: {output_dir}")


if __name__ == '__main__':
    main()
