"""
Training Data Generation V3 - Using Monte Carlo Phase2

Correct approach:
1. Generate floor plan
2. Generate door/exit configs
3. Use monte_carlo.py phase2 for evaluation (handles agent/fire randomization)
4. Construct pairwise labels from monte carlo statistics

This properly leverages the optimized monte_carlo infrastructure.
"""

import os
import sys
import json
import time
import argparse
import logging
import tempfile
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import traceback

import numpy as np

# Local imports
from floor_plan_generator import FloorPlanGenerator, FloorPlanMetadata
from candidate_generator import CandidateGenerator
from pair_constructor import PairConstructor, SimulationResult, PairwiseLabel, PairWriter
from data_validator import DataValidator, create_dataset_splits

# Import monte carlo functionality
from monte_carlo import run_monte_carlo_parallel
from simulation import SimulationConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for training data generation"""
    num_floor_plans: int = 1000
    door_configs_per_plan: int = 30      # Different door placements per floor plan
    monte_carlo_runs_per_config: int = 10  # Monte Carlo runs per door config
    pairs_per_plan: int = 200
    workers: int = 8

    # Floor plan parameters
    size_range: Tuple[int, int] = (20, 80)

    # Door configuration parameters
    num_doors_range: Tuple[int, int] = (2, 5)
    num_exits_range: Tuple[int, int] = (1, 3)
    min_door_spacing: int = 3

    # Monte Carlo simulation parameters
    agent_count_range: Tuple[int, int] = (10, 50)  # Will be randomized per run
    max_steps: int = 500
    fire_spread_rate: float = 0.3
    fire_intensity_growth: float = 0.5
    fire_damage_threshold: float = 10.0

    # Output parameters
    output_dir: str = './training_data_v3'
    checkpoint_interval: int = 50

    # Random seed
    seed: int = 42


def evaluate_door_config_monte_carlo(args: Tuple) -> Dict[str, Any]:
    """
    Evaluate a single door configuration using monte carlo phase2.

    Args:
        args: (floor_plan_id, config_id, grid, door_config, mc_params)

    Returns:
        Dict with door config and monte carlo statistics
    """
    floor_plan_id, config_id, grid_data, door_config, mc_params = args

    grid = np.array(grid_data, dtype=np.float32)

    try:
        # Create a temporary config for this door configuration
        # Extract exits and doors
        exits = []
        doors = []

        for door in door_config:
            pos_str = door.get('position', '')
            if 'x' in pos_str and 'y' in pos_str:
                parts = pos_str.split('y')
                x = int(parts[0][1:])
                y = int(parts[1])

                door_dict = {
                    'id': door['id'],
                    'position': pos_str,
                    'type': door['type']
                }

                if door['type'] == 'exit':
                    exits.append(door_dict)
                else:
                    doors.append(door_dict)

        # Create simulation config
        rows, cols = grid.shape
        agent_count = np.random.randint(
            mc_params['agent_count_range'][0],
            mc_params['agent_count_range'][1]
        )

        config_dict = {
            'map_rows': rows,
            'map_cols': cols,
            'cell_size': 0.3,
            'timestep_duration': 0.5,
            'fire_update_interval': 2,
            'agent_num': agent_count,
            'max_occupancy': 2,
            'viewing_range': 10,
            'fire_spread_rate': mc_params['fire_spread_rate'],
            'fire_intensity_growth': mc_params['fire_intensity_growth'],
            'fire_damage_threshold': mc_params['fire_damage_threshold'],
            'fire_discovery_delay': 0,
            'start_positions': [],  # Will be randomized by monte carlo
            'targets': [exits[0]['position']] if exits else [f"x{cols-1}y{rows-1}"],
            'initial_fire_map': grid.tolist(),
            'door_configs': doors + exits
        }

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            temp_config_path = f.name

        try:
            # Load config
            config = SimulationConfig.from_file(temp_config_path)

            # Run monte carlo with phase2
            results, statistics = run_monte_carlo_parallel(
                config=config,
                num_runs=mc_params['monte_carlo_runs'],
                num_processes=1,  # Will be parallelized at higher level
                save_full_results=False,
                use_phase2=True,
                fire_spread_mode='always_real'
            )

            # Extract key statistics
            return {
                'floor_plan_id': floor_plan_id,
                'config_id': config_id,
                'door_config': door_config,
                'survival_rate': statistics['success_rate'] / 100.0,  # Convert to 0-1
                'avg_steps': statistics['average_steps'],
                'avg_fire_damage': statistics['average_fire_damage'],
                'evacuated': statistics['evacuated_agents'],
                'survived': statistics['survived_agents'],
                'error_count': statistics['error_count'],
                'num_runs': mc_params['monte_carlo_runs'],
                'statistics': statistics  # Keep full stats for analysis
            }

        finally:
            # Clean up temp file
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)

    except Exception as e:
        logger.error(f"Error evaluating config {config_id} for plan {floor_plan_id}: {e}")
        logger.error(traceback.format_exc())
        return {
            'floor_plan_id': floor_plan_id,
            'config_id': config_id,
            'error': str(e)
        }


class TrainingDataGeneratorV3:
    """
    V3: Leverages monte_carlo.py phase2 for robust evaluation.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

        # Initialize components
        self.floor_plan_generator = FloorPlanGenerator(seed=config.seed)
        self.pair_constructor = PairConstructor(seed=config.seed)
        self.validator = DataValidator()

        # Storage
        self.floor_plans: Dict[int, Tuple[np.ndarray, FloorPlanMetadata]] = {}
        self.all_results: List[SimulationResult] = []
        self.all_pairs: List[PairwiseLabel] = []

        os.makedirs(config.output_dir, exist_ok=True)

    def generate(self) -> str:
        """Run the full generation pipeline."""
        logger.info("="*60)
        logger.info("Training Data Generation V3 (Monte Carlo Phase2)")
        logger.info("="*60)
        logger.info(f"Floor plans: {self.config.num_floor_plans}")
        logger.info(f"Door configs per plan: {self.config.door_configs_per_plan}")
        logger.info(f"Monte Carlo runs per config: {self.config.monte_carlo_runs_per_config}")
        logger.info(f"Workers: {self.config.workers}")
        logger.info("="*60)

        # Phase 1: Generate floor plans
        logger.info("\n[Phase 1/4] Generating diverse floor plans...")
        self._generate_floor_plans()

        # Phase 2: Generate door configs and evaluate with monte carlo
        logger.info("\n[Phase 2/4] Evaluating door configs with Monte Carlo...")
        self._evaluate_door_configs_monte_carlo()

        # Phase 3: Construct pairwise labels
        logger.info("\n[Phase 3/4] Constructing pairwise labels...")
        self._construct_pairs()

        # Phase 4: Validate and save
        logger.info("\n[Phase 4/4] Validating and saving...")
        self._validate_and_save()

        logger.info("\n" + "="*60)
        logger.info("Generation complete!")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info("="*60)

        return self.config.output_dir

    def _generate_floor_plans(self):
        """Generate diverse floor plans"""
        for i in range(self.config.num_floor_plans):
            plans = self.floor_plan_generator.generate_batch(
                num_plans=1,
                size_range=self.config.size_range
            )

            if plans:
                grid, metadata = plans[0]
                self.floor_plans[i] = (grid, metadata)

            if (i + 1) % 100 == 0:
                logger.info(f"  Generated {i + 1}/{self.config.num_floor_plans} floor plans")

        logger.info(f"  Generated {len(self.floor_plans)} valid floor plans")

    def _evaluate_door_configs_monte_carlo(self):
        """
        Evaluate door configurations using monte carlo phase2.
        This properly randomizes agents/fire across runs.
        """
        tasks = []

        for plan_id, (grid, metadata) in self.floor_plans.items():
            # Generate door configurations for this floor plan
            candidate_gen = CandidateGenerator(
                floor_plan=grid,
                min_door_spacing=self.config.min_door_spacing,
                seed=self.config.seed + plan_id
            )

            door_configs = candidate_gen.generate_candidate_pool(
                num_candidates=self.config.door_configs_per_plan,
                num_doors_range=self.config.num_doors_range,
                num_exits_range=self.config.num_exits_range,
                random_ratio=0.5
            )

            # Create evaluation tasks
            mc_params = {
                'monte_carlo_runs': self.config.monte_carlo_runs_per_config,
                'agent_count_range': self.config.agent_count_range,
                'fire_spread_rate': self.config.fire_spread_rate,
                'fire_intensity_growth': self.config.fire_intensity_growth,
                'fire_damage_threshold': self.config.fire_damage_threshold
            }

            for config_id, door_config in enumerate(door_configs):
                tasks.append((plan_id, config_id, grid.tolist(), door_config, mc_params))

        logger.info(f"  Evaluating {len(tasks)} door configurations...")

        # Run evaluations in parallel
        completed = 0
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=self.config.workers) as executor:
            futures = {executor.submit(evaluate_door_config_monte_carlo, task): task[0]
                      for task in tasks}

            for future in as_completed(futures):
                try:
                    result = future.result()

                    if 'error' not in result:
                        # Convert to SimulationResult
                        sim_result = SimulationResult(
                            floor_plan_id=result['floor_plan_id'],
                            config_id=result['config_id'],
                            config={'door_config': result['door_config']},
                            scenario={'monte_carlo_runs': result['num_runs']},
                            survival_rate=result['survival_rate'],
                            avg_evacuation_time=result['avg_steps'] * 0.5,
                            steps=int(result['avg_steps']),
                            evacuated=result['evacuated'],
                            stuck=result['survived'] - result['evacuated'],
                            dead=0,  # Calculated from survival rate
                            avg_fire_damage=result['avg_fire_damage']
                        )
                        self.all_results.append(sim_result)

                    completed += 1

                    if completed % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        remaining = len(tasks) - completed
                        eta = remaining / rate if rate > 0 else 0
                        logger.info(f"  Completed {completed}/{len(tasks)} configs "
                                  f"({len(self.all_results)} valid) - ETA: {timedelta(seconds=int(eta))}")

                except Exception as e:
                    logger.error(f"Task failed: {e}")

        logger.info(f"  Evaluated {len(self.all_results)} door configurations")

    def _construct_pairs(self):
        """Construct pairwise labels within each floor plan"""
        by_plan = defaultdict(list)
        for result in self.all_results:
            by_plan[result.floor_plan_id].append(result)

        for plan_id, plan_results in by_plan.items():
            if len(plan_results) < 2:
                continue

            # All pairs within same floor plan
            pairs = self.pair_constructor.construct_pairs(
                plan_results,
                num_pairs=self.config.pairs_per_plan,
                strategy='mixed',
                within_plan_ratio=1.0
            )

            self.all_pairs.extend(pairs)

        # Balance labels
        self.all_pairs = self.pair_constructor.balance_labels(self.all_pairs)

        logger.info(f"  Constructed {len(self.all_pairs)} pairwise labels")
        stats = self.pair_constructor.get_pair_statistics(self.all_pairs)
        logger.info(f"  Label distribution: {stats.get('label_1_ratio', 0):.1%} positive")
        logger.info(f"  Avg score diff: {stats.get('avg_score_diff', 0):.3f}")

    def _validate_and_save(self):
        """Validate and save to disk"""
        pair_dicts = [p.to_dict() for p in self.all_pairs]

        # Create splits
        train_pairs, val_pairs, test_pairs = create_dataset_splits(
            pair_dicts, train_ratio=0.7, val_ratio=0.15, seed=self.config.seed
        )

        # Validate
        report = self.validator.validate_dataset(train_pairs, val_pairs, test_pairs)
        report.print_summary()

        # Save pairs
        writer = PairWriter(self.config.output_dir)
        writer.write_jsonl([PairwiseLabel.from_dict(p) for p in train_pairs], 'train_pairs.jsonl')
        writer.write_jsonl([PairwiseLabel.from_dict(p) for p in val_pairs], 'val_pairs.jsonl')
        writer.write_jsonl([PairwiseLabel.from_dict(p) for p in test_pairs], 'test_pairs.jsonl')

        logger.info(f"  Saved {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test pairs")

        # Save metadata
        metadata = {
            'description': 'Door placement comparison using Monte Carlo Phase2',
            'config': asdict(self.config),
            'statistics': {
                'total_floor_plans': len(self.floor_plans),
                'total_door_configs': len(self.all_results),
                'total_monte_carlo_runs': len(self.all_results) * self.config.monte_carlo_runs_per_config,
                'total_pairs': len(self.all_pairs),
                'train_pairs': len(train_pairs),
                'val_pairs': len(val_pairs),
                'test_pairs': len(test_pairs)
            },
            'validation': report.to_dict(),
            'generated_at': datetime.now().isoformat()
        }

        with open(os.path.join(self.config.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save floor plans
        self._save_floor_plans()

    def _save_floor_plans(self):
        """Save floor plans to disk"""
        floor_plans_dir = os.path.join(self.config.output_dir, 'floor_plans')
        os.makedirs(floor_plans_dir, exist_ok=True)

        for plan_id, (grid, metadata) in self.floor_plans.items():
            np.savez_compressed(
                os.path.join(floor_plans_dir, f'plan_{plan_id:05d}.npz'),
                grid=grid,
                size=np.array(metadata.size),
                room_count=metadata.room_count,
                method=metadata.generation_method
            )

        logger.info(f"  Saved {len(self.floor_plans)} floor plans")


def main():
    parser = argparse.ArgumentParser(
        description='Generate training data using Monte Carlo Phase2 (V3)'
    )

    parser.add_argument('--num-floor-plans', type=int, default=1000)
    parser.add_argument('--door-configs-per-plan', type=int, default=30)
    parser.add_argument('--monte-carlo-runs', type=int, default=10,
                       help='Monte Carlo runs per door config (randomizes agents/fire)')
    parser.add_argument('--pairs-per-plan', type=int, default=200)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output-dir', type=str, default='./training_data_v3')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    config = GenerationConfig(
        num_floor_plans=args.num_floor_plans,
        door_configs_per_plan=args.door_configs_per_plan,
        monte_carlo_runs_per_config=args.monte_carlo_runs,
        pairs_per_plan=args.pairs_per_plan,
        workers=args.workers,
        output_dir=args.output_dir,
        seed=args.seed
    )

    generator = TrainingDataGeneratorV3(config)
    output_dir = generator.generate()

    print(f"\nTraining data saved to: {output_dir}")
    print(f"\nTotal simulations: {config.num_floor_plans * config.door_configs_per_plan * config.monte_carlo_runs_per_config:,}")


if __name__ == '__main__':
    main()
