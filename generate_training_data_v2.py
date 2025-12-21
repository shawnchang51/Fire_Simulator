"""
Training Data Generation - CORRECTED VERSION

Generates pairwise labels comparing DIFFERENT DOOR PLACEMENTS on the SAME floor plan.
Uses existing optimized components:
- candidate_generator.py: generates door/exit configurations
- fast_simulation.py: Phase 2 fast simulation with batch_evaluate()
- monte_carlo.py: optimized parallel evaluation

Key difference from v1:
- Varies DOOR/EXIT positions (what model learns)
- Keeps agents/fire consistent per scenario (controlled variables)
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import traceback

import numpy as np

# Local imports - use existing optimized code
from floor_plan_generator import FloorPlanGenerator, FloorPlanMetadata
from candidate_generator import CandidateGenerator
from fast_simulation import batch_evaluate, SimResult
from diversity_sampler import DiversitySampler, AgentPlacer, FirePlacer
from pair_constructor import PairConstructor, SimulationResult, PairwiseLabel, PairWriter
from data_validator import DataValidator, create_dataset_splits


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
    door_configs_per_plan: int = 30  # Different door placements per floor plan
    scenarios_per_config: int = 3    # Different agent/fire scenarios per door config
    trials_per_scenario: int = 3     # Monte Carlo trials per scenario
    pairs_per_plan: int = 200
    workers: int = 8

    # Floor plan parameters
    size_range: Tuple[int, int] = (20, 80)

    # Door configuration parameters
    num_doors_range: Tuple[int, int] = (2, 5)
    num_exits_range: Tuple[int, int] = (1, 3)
    min_door_spacing: int = 3

    # Simulation parameters
    max_steps: int = 500
    fire_spread_rate: float = 0.3

    # Output parameters
    output_dir: str = './training_data_v2'
    checkpoint_interval: int = 50

    # Random seed
    seed: int = 42


def evaluate_door_config_batch(args: Tuple) -> List[Dict[str, Any]]:
    """
    Evaluate a batch of door configurations on the same floor plan.
    Uses batch_evaluate() for parallel processing.

    Args:
        args: (floor_plan_id, grid, door_configs, scenarios, sim_params)

    Returns:
        List of evaluation results
    """
    floor_plan_id, grid_data, door_configs, scenarios, sim_params = args

    grid = np.array(grid_data, dtype=np.float32)
    results = []

    try:
        # For each door configuration
        for config_id, door_config in enumerate(door_configs):
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
                logger.warning(f"No exits found for config {config_id}")
                continue

            # Evaluate this door config on multiple scenarios
            scenario_results = []

            for scenario in scenarios:
                agent_positions = scenario['agent_positions']
                fire_positions = scenario['fire_positions']

                # Build batch for parallel trials
                trial_scenarios = []
                for trial in range(sim_params['trials_per_scenario']):
                    trial_scenarios.append({
                        'floor_plan': grid,
                        'agent_positions': agent_positions,
                        'exit_positions': exit_positions,
                        'fire_positions': fire_positions,
                        'config': {
                            'fire_spread_rate': scenario.get('fire_spread_rate', 0.3),
                            'fire_intensity_growth': 0.5,
                            'fire_damage_threshold': 10.0,
                            'fire_discovery_delay': scenario.get('fire_discovery_delay', 0),
                            'max_steps': sim_params['max_steps']
                        },
                        'seed': sim_params['seed'] + trial + config_id * 100
                    })

                # Batch evaluate trials in parallel (uses fast_simulation.batch_evaluate)
                trial_results = batch_evaluate(trial_scenarios, num_workers=None)

                # Aggregate trial results (median for robustness)
                valid_results = [r for r in trial_results if r.survival_rate >= 0]
                if valid_results:
                    scenario_results.append({
                        'survival_rate': np.median([r.survival_rate for r in valid_results]),
                        'steps': int(np.median([r.steps for r in valid_results])),
                        'evacuated': int(np.median([r.evacuated for r in valid_results])),
                        'dead': int(np.median([r.dead for r in valid_results])),
                        'avg_fire_damage': np.median([r.avg_fire_damage for r in valid_results])
                    })

            # Aggregate across scenarios (median)
            if scenario_results:
                aggregated = {
                    'floor_plan_id': floor_plan_id,
                    'config_id': config_id,
                    'door_config': door_config,
                    'survival_rate': np.median([s['survival_rate'] for s in scenario_results]),
                    'steps': int(np.median([s['steps'] for s in scenario_results])),
                    'evacuated': int(np.median([s['evacuated'] for s in scenario_results])),
                    'dead': int(np.median([s['dead'] for s in scenario_results])),
                    'avg_fire_damage': np.median([s['avg_fire_damage'] for s in scenario_results]),
                    'num_scenarios': len(scenario_results)
                }
                results.append(aggregated)

    except Exception as e:
        logger.error(f"Error evaluating floor plan {floor_plan_id}: {e}")
        logger.error(traceback.format_exc())

    return results


class TrainingDataGeneratorV2:
    """
    CORRECTED version: Compares different DOOR PLACEMENTS on the same floor plan.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

        # Initialize components
        self.floor_plan_generator = FloorPlanGenerator(seed=config.seed)
        self.agent_placer = AgentPlacer(seed=config.seed + 1)
        self.fire_placer = FirePlacer(seed=config.seed + 2)
        self.pair_constructor = PairConstructor(seed=config.seed + 3)
        self.validator = DataValidator()

        # Storage
        self.floor_plans: Dict[int, Tuple[np.ndarray, FloorPlanMetadata]] = {}
        self.all_results: List[SimulationResult] = []
        self.all_pairs: List[PairwiseLabel] = []

        os.makedirs(config.output_dir, exist_ok=True)

    def generate(self) -> str:
        """Run the full generation pipeline."""
        logger.info("="*60)
        logger.info("Training Data Generation V2 (Door Placement Comparison)")
        logger.info("="*60)
        logger.info(f"Floor plans: {self.config.num_floor_plans}")
        logger.info(f"Door configs per plan: {self.config.door_configs_per_plan}")
        logger.info(f"Scenarios per config: {self.config.scenarios_per_config}")
        logger.info(f"Trials per scenario: {self.config.trials_per_scenario}")
        logger.info(f"Workers: {self.config.workers}")
        logger.info("="*60)

        # Phase 1: Generate floor plans
        logger.info("\n[Phase 1/4] Generating diverse floor plans...")
        self._generate_floor_plans()

        # Phase 2: Generate door configurations and evaluate
        logger.info("\n[Phase 2/4] Generating door configs and running simulations...")
        self._generate_and_evaluate_door_configs()

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

    def _generate_and_evaluate_door_configs(self):
        """
        KEY FUNCTION: Generate different door placements for each floor plan and evaluate them.
        This is what the model will learn to rank.
        """
        batches = []

        for plan_id, (grid, metadata) in self.floor_plans.items():
            # Generate diverse door configurations for this floor plan
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

            # Generate agent/fire scenarios (keep consistent across door configs for fair comparison)
            passable_count = int(np.sum(grid == 0))
            agent_count = min(50, max(10, int(passable_count * 0.03)))  # 3% density

            scenarios = []
            for _ in range(self.config.scenarios_per_config):
                # Place agents
                agent_positions = self.agent_placer.place_agents(
                    grid, agent_count, 'uniform', metadata.room_centers
                )

                # Place fires
                fire_positions = self.fire_placer.place_fires(
                    grid, num_fires=1, exit_positions=[], strategy='random'
                )

                scenarios.append({
                    'agent_positions': agent_positions,
                    'fire_positions': fire_positions,
                    'fire_spread_rate': self.config.fire_spread_rate,
                    'fire_discovery_delay': 0
                })

            sim_params = {
                'trials_per_scenario': self.config.trials_per_scenario,
                'max_steps': self.config.max_steps,
                'seed': self.config.seed
            }

            batches.append((plan_id, grid.tolist(), door_configs, scenarios, sim_params))

        # Run evaluations in parallel
        logger.info(f"  Evaluating {len(batches)} floor plans with door configs...")
        completed = 0
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=self.config.workers) as executor:
            futures = {executor.submit(evaluate_door_config_batch, batch): batch[0]
                      for batch in batches}

            for future in as_completed(futures):
                plan_id = futures[future]
                try:
                    results = future.result()

                    # Convert to SimulationResult objects
                    for r in results:
                        sim_result = SimulationResult(
                            floor_plan_id=r['floor_plan_id'],
                            config_id=r['config_id'],
                            config={'door_config': r['door_config']},
                            scenario={'door_placement': 'varied'},
                            survival_rate=r['survival_rate'],
                            avg_evacuation_time=r['steps'] * 0.5,  # timestep_duration
                            steps=r['steps'],
                            evacuated=r['evacuated'],
                            stuck=0,
                            dead=r['dead'],
                            avg_fire_damage=r['avg_fire_damage']
                        )
                        self.all_results.append(sim_result)

                    completed += 1

                    if completed % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        remaining = len(batches) - completed
                        eta = remaining / rate if rate > 0 else 0
                        logger.info(f"  Completed {completed}/{len(batches)} plans "
                                  f"({len(self.all_results)} configs) - ETA: {timedelta(seconds=int(eta))}")

                except Exception as e:
                    logger.error(f"Batch for plan {plan_id} failed: {e}")

        logger.info(f"  Evaluated {len(self.all_results)} door configurations")

    def _construct_pairs(self):
        """Construct pairwise labels - WITHIN floor plan only"""
        by_plan = defaultdict(list)
        for result in self.all_results:
            by_plan[result.floor_plan_id].append(result)

        # Construct pairs WITHIN each floor plan
        for plan_id, plan_results in by_plan.items():
            if len(plan_results) < 2:
                continue

            # All pairs are within-plan (comparing door configs on SAME floor plan)
            pairs = self.pair_constructor.construct_pairs(
                plan_results,
                num_pairs=self.config.pairs_per_plan,
                strategy='mixed',
                within_plan_ratio=1.0  # 100% within-plan
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
            'description': 'Pairwise labels comparing DOOR PLACEMENTS on the same floor plan',
            'config': asdict(self.config),
            'statistics': {
                'total_floor_plans': len(self.floor_plans),
                'total_door_configs': len(self.all_results),
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
        description='Generate training data comparing door placements (CORRECTED VERSION)'
    )

    parser.add_argument('--num-floor-plans', type=int, default=1000)
    parser.add_argument('--door-configs-per-plan', type=int, default=30)
    parser.add_argument('--scenarios-per-config', type=int, default=3)
    parser.add_argument('--trials-per-scenario', type=int, default=3)
    parser.add_argument('--pairs-per-plan', type=int, default=200)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output-dir', type=str, default='./training_data_v2')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    config = GenerationConfig(
        num_floor_plans=args.num_floor_plans,
        door_configs_per_plan=args.door_configs_per_plan,
        scenarios_per_config=args.scenarios_per_config,
        trials_per_scenario=args.trials_per_scenario,
        pairs_per_plan=args.pairs_per_plan,
        workers=args.workers,
        output_dir=args.output_dir,
        seed=args.seed
    )

    generator = TrainingDataGeneratorV2(config)
    output_dir = generator.generate()

    print(f"\nTraining data saved to: {output_dir}")


if __name__ == '__main__':
    main()
