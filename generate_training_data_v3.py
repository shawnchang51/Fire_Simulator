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
    size_range: Tuple[int, int] = (50, 120)  # Larger buildings for meaningful door placement differences

    # Door configuration parameters
    num_doors_range: Tuple[int, int] = (2, 5)
    num_exits_range: Tuple[int, int] = (1, 3)
    min_door_spacing: int = 5  # More spacing to create challenging placements

    # Monte Carlo simulation parameters
    occupant_density_range: Tuple[float, float] = (0.05, 0.15)  # Higher density for congestion
    max_steps: int = 500

    # Fire parameter ranges (will be randomized per MC run)
    num_fires_range: Tuple[int, int] = (3, 7)  # More fires for challenge
    fire_spread_rate_range: Tuple[float, float] = (0.3, 0.8)  # More aggressive fire
    fire_intensity_growth_range: Tuple[float, float] = (0.5, 1.5)  # Faster intensity growth
    fire_discovery_delay_range: Tuple[int, int] = (5, 30)  # Longer delays to allow fire to spread

    # Fixed parameters (same baseline for all comparisons)
    fire_damage_threshold: float = 10.0  # Fixed threshold for consistent baseline

    # Output parameters
    output_dir: str = './training_data_v3'
    checkpoint_interval: int = 50

    # Random seed
    seed: int = 42


def evaluate_door_config_monte_carlo(args: Tuple) -> Dict[str, Any]:
    """
    Evaluate a single door configuration with randomized monte carlo runs.

    Each run randomizes:
    - Agent positions
    - Fire positions
    - Fire spread rate
    - Fire intensity growth
    - Fire discovery delay
    - Fire damage threshold

    Args:
        args: (floor_plan_id, config_id, grid, door_config, mc_params)

    Returns:
        Dict with door config and monte carlo statistics
    """
    floor_plan_id, config_id, grid_data, door_config, mc_params = args

    grid = np.array(grid_data, dtype=np.float32)

    try:
        from fast_simulation import FastEvacuationSim
        from fast_fire import FireSpreadMode

        # Extract exit positions from door config
        exit_positions = []
        for door in door_config:
            if door.get('type') == 'exit':
                pos_str = door.get('position', '')
                if 'x' in pos_str and 'y' in pos_str:
                    parts = pos_str.split('y')
                    x = int(parts[0][1:])
                    y = int(parts[1])
                    exit_positions.append((x, y))

        if not exit_positions:
            return {
                'floor_plan_id': floor_plan_id,
                'config_id': config_id,
                'error': 'No exits found in door config'
            }

        rows, cols = grid.shape

        # Find valid positions for agent/fire placement
        passable_positions = []
        for y in range(rows):
            for x in range(cols):
                if grid[y, x] == 0:  # Passable
                    passable_positions.append((x, y))

        if len(passable_positions) < 10:
            return {
                'floor_plan_id': floor_plan_id,
                'config_id': config_id,
                'error': 'Not enough passable positions'
            }

        # Run multiple monte carlo trials with randomization
        trial_results = []

        for run_idx in range(mc_params['monte_carlo_runs']):
            # Randomize occupant density and calculate agent count
            occupant_density = np.random.uniform(
                mc_params['occupant_density_range'][0],
                mc_params['occupant_density_range'][1]
            )
            agent_count = int(len(passable_positions) * occupant_density)
            agent_count = max(5, min(agent_count, len(passable_positions)))  # Clamp to valid range

            # Random agent positions
            agent_indices = np.random.choice(
                len(passable_positions),
                size=agent_count,
                replace=False
            )
            agent_positions = [passable_positions[i] for i in agent_indices]

            # Randomize fire count (2-5 fires for better differentiation)
            num_fires = np.random.randint(
                mc_params['num_fires_range'][0],
                mc_params['num_fires_range'][1] + 1
            )

            # Ensure fire doesn't overlap with agents
            available_for_fire = [pos for pos in passable_positions if pos not in agent_positions]
            if len(available_for_fire) < num_fires:
                num_fires = max(1, len(available_for_fire))

            fire_indices = np.random.choice(len(available_for_fire), size=num_fires, replace=False)
            fire_positions = [available_for_fire[i] for i in fire_indices]

            # Randomize fire spread parameters
            fire_spread_rate = np.random.uniform(
                mc_params['fire_spread_rate_range'][0],
                mc_params['fire_spread_rate_range'][1]
            )
            fire_intensity_growth = np.random.uniform(
                mc_params['fire_intensity_growth_range'][0],
                mc_params['fire_intensity_growth_range'][1]
            )
            fire_discovery_delay = np.random.randint(
                mc_params['fire_discovery_delay_range'][0],
                mc_params['fire_discovery_delay_range'][1] + 1
            )

            # Fixed threshold for consistent baseline
            fire_damage_threshold = mc_params['fire_damage_threshold']

            try:
                # Run Phase 2 simulation with randomized parameters
                sim = FastEvacuationSim(
                    grid=grid.copy(),
                    agent_starts=agent_positions,
                    exits=exit_positions,
                    fire_starts=fire_positions,
                    deterministic_fire=False,
                    fire_update_interval=2,
                    fire_discovery_delay=fire_discovery_delay,
                    fire_spread_mode='always_real',
                    fire_spread_rate=fire_spread_rate,
                    fire_intensity_growth=fire_intensity_growth,
                    fire_damage_threshold=fire_damage_threshold
                )

                result = sim.run(max_steps=mc_params['max_steps'])

                trial_results.append({
                    'survival_rate': result.survival_rate,
                    'steps': result.steps,
                    'evacuated': result.evacuated,
                    'dead': result.dead,
                    'avg_fire_damage': result.avg_fire_damage,
                    # Store randomized params for analysis
                    'occupant_density': occupant_density,
                    'agent_count': agent_count,
                    'num_fires': num_fires,
                    'fire_spread_rate': fire_spread_rate,
                    'fire_discovery_delay': fire_discovery_delay
                })

            except Exception as e:
                logger.warning(f"Trial {run_idx} failed for config {config_id}: {e}")
                continue

        if not trial_results:
            return {
                'floor_plan_id': floor_plan_id,
                'config_id': config_id,
                'error': 'All trials failed'
            }

        # Aggregate results (median for robustness)
        return {
            'floor_plan_id': floor_plan_id,
            'config_id': config_id,
            'door_config': door_config,
            'survival_rate': float(np.median([r['survival_rate'] for r in trial_results])),
            'avg_steps': float(np.median([r['steps'] for r in trial_results])),
            'avg_fire_damage': float(np.median([r['avg_fire_damage'] for r in trial_results])),
            'evacuated': int(np.median([r['evacuated'] for r in trial_results])),
            'dead': int(np.median([r['dead'] for r in trial_results])),
            'survived': int(np.median([r['evacuated'] for r in trial_results])),
            'num_runs': len(trial_results),
            'success_runs': len(trial_results),
            # Parameter diversity stats
            'occupant_density_range': [
                float(min(r['occupant_density'] for r in trial_results)),
                float(max(r['occupant_density'] for r in trial_results))
            ],
            'agent_count_range': [
                int(min(r['agent_count'] for r in trial_results)),
                int(max(r['agent_count'] for r in trial_results))
            ],
            'num_fires_range': [
                int(min(r['num_fires'] for r in trial_results)),
                int(max(r['num_fires'] for r in trial_results))
            ],
            'fire_spread_rate_range': [
                float(min(r['fire_spread_rate'] for r in trial_results)),
                float(max(r['fire_spread_rate'] for r in trial_results))
            ],
            'fire_delay_range': [
                int(min(r['fire_discovery_delay'] for r in trial_results)),
                int(max(r['fire_discovery_delay'] for r in trial_results))
            ]
        }

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
        floor_plans_dir = os.path.join(self.config.output_dir, 'floor_plans')
        os.makedirs(floor_plans_dir, exist_ok=True)

        for i in range(self.config.num_floor_plans):
            plans = self.floor_plan_generator.generate_batch(
                num_plans=1,
                size_range=self.config.size_range
            )

            if plans:
                grid, metadata = plans[0]
                self.floor_plans[i] = (grid, metadata)

                # Save floor plan immediately
                np.savez_compressed(
                    os.path.join(floor_plans_dir, f'plan_{i:05d}.npz'),
                    grid=grid,
                    size=np.array(metadata.size),
                    room_count=metadata.room_count,
                    method=metadata.generation_method
                )

            if (i + 1) % 100 == 0:
                logger.info(f"  Generated {i + 1}/{self.config.num_floor_plans} floor plans (saved to disk)")

        logger.info(f"  Generated {len(self.floor_plans)} valid floor plans")

    def _evaluate_door_configs_monte_carlo(self):
        """
        Evaluate door configurations using monte carlo phase2.
        This properly randomizes agents/fire across runs.
        """
        # Process floor plans one at a time to avoid memory leak from batching all tasks
        total_tasks = len(self.floor_plans) * self.config.door_configs_per_plan
        completed = 0
        start_time = time.time()
        results_file = os.path.join(self.config.output_dir, 'simulation_results.jsonl')

        logger.info(f"  Evaluating {total_tasks} door configurations...")

        # Monte Carlo parameters (shared across all evaluations)
        mc_params = {
            'monte_carlo_runs': self.config.monte_carlo_runs_per_config,
            'occupant_density_range': self.config.occupant_density_range,
            'num_fires_range': self.config.num_fires_range,
            'fire_spread_rate_range': self.config.fire_spread_rate_range,
            'fire_intensity_growth_range': self.config.fire_intensity_growth_range,
            'fire_discovery_delay_range': self.config.fire_discovery_delay_range,
            'fire_damage_threshold': self.config.fire_damage_threshold,  # Fixed
            'max_steps': self.config.max_steps
        }

        with ProcessPoolExecutor(max_workers=self.config.workers) as executor:
            # Process one floor plan at a time to avoid memory leak
            for plan_id, (grid, metadata) in self.floor_plans.items():
                plan_start_time = time.time()

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

                # Create tasks for this floor plan only
                grid_list = grid.tolist()
                tasks = [
                    (plan_id, config_id, grid_list, door_config, mc_params)
                    for config_id, door_config in enumerate(door_configs)
                ]

                # Submit tasks for this floor plan
                logger.info(f"  Floor plan {plan_id}: Submitting {len(tasks)} configs to {self.config.workers} workers")
                futures = {executor.submit(evaluate_door_config_monte_carlo, task): task
                          for task in tasks}

                plan_completed = 0

                # Process results for this floor plan
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

                            # Save result immediately to disk
                            with open(results_file, 'a') as f:
                                result_dict = {
                                    'floor_plan_id': sim_result.floor_plan_id,
                                    'config_id': sim_result.config_id,
                                    'config': sim_result.config,
                                    'scenario': sim_result.scenario,
                                    'survival_rate': sim_result.survival_rate,
                                    'avg_evacuation_time': sim_result.avg_evacuation_time,
                                    'steps': sim_result.steps,
                                    'evacuated': sim_result.evacuated,
                                    'stuck': sim_result.stuck,
                                    'dead': sim_result.dead,
                                    'avg_fire_damage': sim_result.avg_fire_damage,
                                    'score': sim_result.score
                                }
                                f.write(json.dumps(result_dict) + '\n')

                        completed += 1

                        if completed % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed
                            remaining = total_tasks - completed
                            eta = remaining / rate if rate > 0 else 0
                            logger.info(f"  Completed {completed}/{total_tasks} configs "
                                      f"({len(self.all_results)} valid) - ETA: {timedelta(seconds=int(eta))}")

                            # Save checkpoint metadata
                            if completed % self.config.checkpoint_interval == 0:
                                self._save_checkpoint_metadata(completed, total_tasks)

                    except Exception as e:
                        logger.error(f"Task failed: {e}")

                # Clear tasks list after processing this floor plan to free memory
                del tasks
                del grid_list

                # Log completion for this floor plan
                plan_elapsed = time.time() - plan_start_time
                plan_rate = len(door_configs) / plan_elapsed if plan_elapsed > 0 else 0
                logger.info(f"  Floor plan {plan_id} complete: {len(door_configs)} configs in {plan_elapsed:.1f}s ({plan_rate:.1f} configs/sec)")

                if (plan_id + 1) % 10 == 0:
                    logger.info(f"  Progress: {plan_id + 1}/{len(self.floor_plans)} floor plans processed")

        logger.info(f"  Evaluated {len(self.all_results)} door configurations")
        logger.info(f"  Results saved to {results_file}")

    def _construct_pairs(self):
        """Construct pairwise labels within each floor plan"""
        by_plan = defaultdict(list)
        for result in self.all_results:
            by_plan[result.floor_plan_id].append(result)

        logger.info(f"  Grouped results into {len(by_plan)} floor plans")

        # Save pairs incrementally
        raw_pairs_file = os.path.join(self.config.output_dir, 'raw_pairs.jsonl')
        pairs_saved = 0

        for plan_id, plan_results in by_plan.items():
            if len(plan_results) < 2:
                logger.debug(f"  Plan {plan_id}: Only {len(plan_results)} config(s), skipping")
                continue

            # Debug: Show score distribution for this floor plan
            scores = [r.score for r in plan_results]
            if (plan_id + 1) % 100 == 0:
                logger.info(f"  Plan {plan_id}: {len(plan_results)} configs, scores: "
                           f"min={min(scores):.4f}, max={max(scores):.4f}, "
                           f"range={max(scores)-min(scores):.4f}")

            # All pairs within same floor plan
            pairs = self.pair_constructor.construct_pairs(
                plan_results,
                num_pairs=self.config.pairs_per_plan,
                strategy='mixed',
                within_plan_ratio=1.0
            )

            if (plan_id + 1) % 100 == 0:
                logger.info(f"  Plan {plan_id}: Constructed {len(pairs)} pairs")

            self.all_pairs.extend(pairs)

            # Save pairs for this plan immediately
            with open(raw_pairs_file, 'a') as f:
                for pair in pairs:
                    f.write(json.dumps(pair.to_dict()) + '\n')
            pairs_saved += len(pairs)

            if (plan_id + 1) % self.config.checkpoint_interval == 0:
                logger.info(f"  Checkpoint: {pairs_saved} pairs saved to disk")

        # Balance labels
        self.all_pairs = self.pair_constructor.balance_labels(self.all_pairs)

        logger.info(f"  Constructed {len(self.all_pairs)} pairwise labels")
        logger.info(f"  Raw pairs saved to {raw_pairs_file}")
        stats = self.pair_constructor.get_pair_statistics(self.all_pairs)
        logger.info(f"  Label distribution: {stats.get('label_1_ratio', 0):.1%} positive")
        logger.info(f"  Avg score diff: {stats.get('avg_score_diff', 0):.3f}")

    def _validate_and_save(self):
        """Validate and save final splits to disk"""
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

        logger.info(f"  Metadata saved to metadata.json")
        logger.info(f"  Floor plans already saved to floor_plans/ (incremental)")

    def _save_checkpoint_metadata(self, completed: int, total: int):
        """Save progress checkpoint metadata"""
        checkpoint = {
            'checkpoint_time': datetime.now().isoformat(),
            'progress': {
                'floor_plans_generated': len(self.floor_plans),
                'configs_evaluated': len(self.all_results),
                'configs_completed': completed,
                'configs_total': total,
                'completion_percent': (completed / total * 100) if total > 0 else 0
            },
            'config': asdict(self.config)
        }

        checkpoint_file = os.path.join(self.config.output_dir, 'checkpoint.json')
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)


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
