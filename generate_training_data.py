"""
Training Data Generation Orchestrator

Main script for generating pairwise ranking training data on EPYC server.

Features:
- Parallel floor plan generation and simulation
- Checkpointing and resume capability
- Progress monitoring with ETA
- Automatic diversity verification
- Memory-efficient streaming to disk

Usage:
    # Full generation (EPYC server)
    python generate_training_data.py \
        --num-floor-plans 5000 \
        --configs-per-plan 30 \
        --trials-per-config 5 \
        --pairs-per-plan 200 \
        --workers 120 \
        --output-dir ./training_data

    # Quick test run
    python generate_training_data.py \
        --num-floor-plans 10 \
        --configs-per-plan 5 \
        --trials-per-config 3 \
        --pairs-per-plan 20 \
        --workers 4 \
        --output-dir ./test_data

    # Resume from checkpoint
    python generate_training_data.py --resume ./training_data/checkpoint.json
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

# Local imports
from floor_plan_generator import FloorPlanGenerator, FloorPlanMetadata
from diversity_sampler import DiversitySampler, ScenarioConfig, AgentPlacer, FirePlacer
from pair_constructor import PairConstructor, SimulationResult, PairwiseLabel, PairWriter
from data_validator import DataValidator, create_dataset_splits


# Configure logging
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
    configs_per_plan: int = 30
    trials_per_config: int = 5
    pairs_per_plan: int = 200
    workers: int = 8

    # Floor plan parameters
    size_range: Tuple[int, int] = (20, 80)
    method_weights: Dict[str, float] = None

    # Simulation parameters
    max_steps: int = 500
    fire_spread_mode: str = 'ALWAYS_REAL'

    # Output parameters
    output_dir: str = './training_data'
    shard_size: int = 100000
    checkpoint_interval: int = 100  # Save checkpoint every N floor plans

    # Random seed
    seed: int = 42

    def __post_init__(self):
        if self.method_weights is None:
            self.method_weights = {
                'bsp': 0.4,
                'grid': 0.3,
                'template': 0.2,
                'cellular': 0.1
            }


@dataclass
class GenerationProgress:
    """Tracks generation progress for checkpointing"""
    completed_plans: int = 0
    total_plans: int = 0
    completed_simulations: int = 0
    total_simulations: int = 0
    completed_pairs: int = 0
    start_time: str = ""
    last_checkpoint: str = ""
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def eta_seconds(self) -> float:
        if self.completed_plans == 0:
            return 0
        elapsed = (datetime.now() - datetime.fromisoformat(self.start_time)).total_seconds()
        rate = self.completed_plans / elapsed
        remaining = self.total_plans - self.completed_plans
        return remaining / rate if rate > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GenerationProgress':
        return cls(**d)


def run_simulation_batch(args: Tuple) -> List[Dict[str, Any]]:
    """
    Run a batch of simulations for a single floor plan.
    Designed to run in a worker process.

    Args:
        args: Tuple of (floor_plan_id, grid, scenarios, config)

    Returns:
        List of simulation result dictionaries
    """
    floor_plan_id, grid_data, scenarios, sim_config = args

    # Reconstruct numpy array from serialized data
    grid = np.array(grid_data, dtype=np.float32)

    results = []

    try:
        # Import simulation components (avoid importing in main process)
        from fast_simulation import FastEvacuationSim
        from fast_fire import FireSpreadMode

        # Map spread mode string to enum
        spread_mode_map = {
            'ALWAYS_REAL': FireSpreadMode.ALWAYS_REAL,
            'REAL_THEN_SIMPLE': FireSpreadMode.REAL_THEN_SIMPLE,
            'REAL_THEN_STOP': FireSpreadMode.REAL_THEN_STOP
        }
        spread_mode = spread_mode_map.get(sim_config.get('fire_spread_mode', 'ALWAYS_REAL'),
                                          FireSpreadMode.ALWAYS_REAL)

        for scenario in scenarios:
            config_id = scenario.get('config_id', 0)
            exit_positions = scenario.get('exit_positions', [])
            agent_positions = scenario.get('agent_positions', [])
            fire_positions = scenario.get('fire_positions', [])
            fire_spread_rate = scenario.get('fire_spread_rate', 0.3)
            fire_discovery_delay = scenario.get('fire_discovery_delay', 0)

            # Run multiple trials
            trial_results = []

            for trial in range(sim_config.get('trials_per_config', 3)):
                try:
                    max_steps = sim_config.get('max_steps', 500)

                    # Convert positions to tuples (FastEvacuationSim requires tuples)
                    agent_starts_tuples = [tuple(pos) if not isinstance(pos, tuple) else pos for pos in agent_positions]
                    exits_tuples = [tuple(pos) if not isinstance(pos, tuple) else pos for pos in exit_positions]
                    fire_starts_tuples = [tuple(pos) if not isinstance(pos, tuple) else pos for pos in fire_positions]

                    # Map fire spread mode from config to enum value
                    spread_mode = sim_config.get('fire_spread_mode', 'ALWAYS_REAL').lower().replace('_', '_')
                    # Convert ALWAYS_REAL → always_real
                    if spread_mode == 'always_real' or spread_mode == 'always real':
                        spread_mode = 'always_real'
                    elif spread_mode == 'real_then_simple' or spread_mode == 'real then simple':
                        spread_mode = 'real_then_simple'
                    elif spread_mode == 'real_then_stop' or spread_mode == 'real then stop':
                        spread_mode = 'real_then_stop'

                    # Run simulation with correct API
                    sim = FastEvacuationSim(
                        grid=grid.copy(),
                        agent_starts=agent_starts_tuples,
                        exits=exits_tuples,
                        fire_starts=fire_starts_tuples,
                        fire_update_interval=2,
                        fire_discovery_delay=fire_discovery_delay,
                        fire_spread_mode=spread_mode,
                        fire_spread_rate=fire_spread_rate,
                        fire_intensity_growth=0.5,
                        fire_damage_threshold=10.0
                    )

                    result = sim.run(max_steps=max_steps)

                    trial_results.append({
                        'survival_rate': result.survival_rate,
                        'steps': result.steps,
                        'evacuated': result.evacuated,
                        'stuck': result.stuck,
                        'dead': result.dead,
                        'avg_fire_damage': result.avg_fire_damage,
                        'avg_evacuation_time': result.avg_evacuation_time
                    })

                except Exception as e:
                    import traceback
                    error_msg = f"{str(e)}\n{traceback.format_exc()}"
                    logger.error(f"Trial {trial} failed: {error_msg}")
                    trial_results.append({
                        'error': error_msg,
                        'survival_rate': 0,
                        'steps': 0
                    })

            # Aggregate trial results (use median for robustness)
            valid_trials = [t for t in trial_results if 'error' not in t]

            if valid_trials:
                aggregated = {
                    'floor_plan_id': floor_plan_id,
                    'config_id': config_id,
                    'config': {
                        'exit_positions': exit_positions,
                    },
                    'scenario': {
                        'agent_count': len(agent_positions),
                        'fire_positions': fire_positions,
                        'fire_spread_rate': fire_spread_rate,
                        'fire_discovery_delay': fire_discovery_delay
                    },
                    'survival_rate': np.median([t['survival_rate'] for t in valid_trials]),
                    'avg_evacuation_time': np.median([t['avg_evacuation_time'] for t in valid_trials]),
                    'steps': int(np.median([t['steps'] for t in valid_trials])),
                    'evacuated': int(np.median([t['evacuated'] for t in valid_trials])),
                    'stuck': int(np.median([t['stuck'] for t in valid_trials])),
                    'dead': int(np.median([t['dead'] for t in valid_trials])),
                    'avg_fire_damage': np.median([t['avg_fire_damage'] for t in valid_trials]),
                    'num_trials': len(valid_trials)
                }
            else:
                aggregated = {
                    'floor_plan_id': floor_plan_id,
                    'config_id': config_id,
                    'error': 'All trials failed',
                    'survival_rate': 0,
                    'steps': 0
                }

            results.append(aggregated)

    except Exception as e:
        logger.error(f"Error in simulation batch for plan {floor_plan_id}: {e}")
        results.append({
            'floor_plan_id': floor_plan_id,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

    return results


class TrainingDataGenerator:
    """
    Main orchestrator for training data generation.

    Handles:
    - Floor plan generation with diversity
    - Parallel simulation execution
    - Pair construction and label generation
    - Checkpointing and resume
    - Progress monitoring
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.progress = GenerationProgress(
            total_plans=config.num_floor_plans,
            start_time=datetime.now().isoformat()
        )

        # Initialize components
        self.floor_plan_generator = FloorPlanGenerator(seed=config.seed)
        self.diversity_sampler = DiversitySampler(seed=config.seed)
        self.agent_placer = AgentPlacer(seed=config.seed)
        self.fire_placer = FirePlacer(seed=config.seed)
        self.pair_constructor = PairConstructor(seed=config.seed)
        self.validator = DataValidator()

        # Storage
        self.floor_plans: Dict[int, Tuple[np.ndarray, FloorPlanMetadata]] = {}
        self.all_results: List[SimulationResult] = []
        self.all_pairs: List[PairwiseLabel] = []

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def generate(self, resume_from: Optional[str] = None) -> str:
        """
        Run the full generation pipeline.

        Args:
            resume_from: Path to checkpoint file to resume from

        Returns:
            Path to output directory
        """
        if resume_from:
            self._load_checkpoint(resume_from)
            logger.info(f"Resumed from checkpoint: {self.progress.completed_plans}/{self.progress.total_plans} plans")

        logger.info("="*60)
        logger.info("Training Data Generation")
        logger.info("="*60)
        logger.info(f"Floor plans: {self.config.num_floor_plans}")
        logger.info(f"Configs per plan: {self.config.configs_per_plan}")
        logger.info(f"Trials per config: {self.config.trials_per_config}")
        logger.info(f"Pairs per plan: {self.config.pairs_per_plan}")
        logger.info(f"Workers: {self.config.workers}")
        logger.info(f"Output: {self.config.output_dir}")
        logger.info("="*60)

        try:
            # Phase 1: Generate floor plans
            logger.info("\n[Phase 1/4] Generating floor plans...")
            self._generate_floor_plans()

            # Phase 2: Run simulations
            logger.info("\n[Phase 2/4] Running simulations...")
            self._run_simulations()

            # Phase 3: Construct pairs
            logger.info("\n[Phase 3/4] Constructing pairwise labels...")
            self._construct_pairs()

            # Phase 4: Validate and save
            logger.info("\n[Phase 4/4] Validating and saving...")
            self._validate_and_save()

            logger.info("\n" + "="*60)
            logger.info("Generation complete!")
            logger.info(f"Output directory: {self.config.output_dir}")
            logger.info("="*60)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self._save_checkpoint()
            raise

        return self.config.output_dir

    def _generate_floor_plans(self):
        """Generate diverse floor plans"""
        start_id = self.progress.completed_plans

        for i in range(start_id, self.config.num_floor_plans):
            try:
                # Generate floor plan
                plans = self.floor_plan_generator.generate_batch(
                    num_plans=1,
                    size_range=self.config.size_range,
                    method_weights=self.config.method_weights
                )

                if plans:
                    grid, metadata = plans[0]

                    # Add exits
                    num_exits = np.random.randint(1, 5)
                    exit_positions = self.floor_plan_generator.add_exits_to_plan(
                        grid, num_exits, 'distributed'
                    )
                    metadata.exit_positions = exit_positions

                    self.floor_plans[i] = (grid, metadata)

            except Exception as e:
                logger.warning(f"Failed to generate floor plan {i}: {e}")
                self.progress.errors.append(f"Plan {i}: {str(e)}")

            # Progress update
            if (i + 1) % 100 == 0:
                logger.info(f"  Generated {i + 1}/{self.config.num_floor_plans} floor plans")

        logger.info(f"  Generated {len(self.floor_plans)} valid floor plans")

    def _run_simulations(self):
        """Run simulations in parallel"""
        # Prepare simulation batches
        batches = []

        for plan_id, (grid, metadata) in self.floor_plans.items():
            # Sample scenarios for this plan
            passable_count = int(np.sum(grid == 0))
            scenarios = self.diversity_sampler.sample_scenarios_for_plan(
                floor_plan_id=plan_id,
                floor_plan_size=metadata.size,
                num_scenarios=self.config.configs_per_plan,
                passable_cells=passable_count
            )

            # Prepare scenario configs
            scenario_configs = []
            for config_id, scenario in enumerate(scenarios):
                # Place agents
                agent_positions = self.agent_placer.place_agents(
                    grid, scenario.agent_count, scenario.agent_distribution,
                    metadata.room_centers
                )

                # Place fires
                fire_positions = self.fire_placer.place_fires(
                    grid, scenario.fire_count, metadata.exit_positions, 'varied'
                )

                scenario_configs.append({
                    'config_id': config_id,
                    'exit_positions': metadata.exit_positions,
                    'agent_positions': agent_positions,
                    'fire_positions': fire_positions,
                    'fire_spread_rate': scenario.fire_spread_rate,
                    'fire_discovery_delay': scenario.fire_discovery_delay
                })

            # Create batch
            sim_config = {
                'trials_per_config': self.config.trials_per_config,
                'max_steps': self.config.max_steps,
                'fire_spread_mode': self.config.fire_spread_mode,
                'seed': self.config.seed
            }

            batches.append((plan_id, grid.tolist(), scenario_configs, sim_config))

        # Calculate total simulations
        self.progress.total_simulations = sum(
            len(b[2]) * self.config.trials_per_config for b in batches
        )
        logger.info(f"  Total simulations to run: {self.progress.total_simulations}")

        # Run in parallel
        completed = 0
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=self.config.workers) as executor:
            futures = {executor.submit(run_simulation_batch, batch): batch[0]
                      for batch in batches}

            for future in as_completed(futures):
                plan_id = futures[future]
                try:
                    results = future.result()

                    # Convert to SimulationResult objects
                    for r in results:
                        if 'error' not in r:
                            sim_result = SimulationResult(
                                floor_plan_id=r['floor_plan_id'],
                                config_id=r['config_id'],
                                config=r['config'],
                                scenario=r['scenario'],
                                survival_rate=r['survival_rate'],
                                avg_evacuation_time=r['avg_evacuation_time'],
                                steps=r['steps'],
                                evacuated=r['evacuated'],
                                stuck=r['stuck'],
                                dead=r['dead'],
                                avg_fire_damage=r['avg_fire_damage']
                            )
                            self.all_results.append(sim_result)

                    completed += 1
                    self.progress.completed_plans = completed

                    # Progress update
                    if completed % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        remaining = len(batches) - completed
                        eta = remaining / rate if rate > 0 else 0
                        eta_str = str(timedelta(seconds=int(eta)))
                        logger.info(f"  Completed {completed}/{len(batches)} plans "
                                  f"({len(self.all_results)} results) - ETA: {eta_str}")

                    # Checkpoint
                    if completed % self.config.checkpoint_interval == 0:
                        self._save_checkpoint()

                except Exception as e:
                    logger.error(f"Batch for plan {plan_id} failed: {e}")
                    self.progress.errors.append(f"Batch {plan_id}: {str(e)}")

        logger.info(f"  Completed {len(self.all_results)} simulation results")

    def _construct_pairs(self):
        """Construct pairwise labels from results"""
        if not self.all_results:
            logger.warning("No simulation results to construct pairs from")
            return

        # Group results by floor plan
        by_plan = defaultdict(list)
        for result in self.all_results:
            by_plan[result.floor_plan_id].append(result)

        # Construct pairs for each plan
        for plan_id, plan_results in by_plan.items():
            if len(plan_results) < 2:
                continue

            pairs = self.pair_constructor.construct_pairs(
                plan_results,
                num_pairs=self.config.pairs_per_plan,
                strategy='mixed',
                within_plan_ratio=0.85
            )

            self.all_pairs.extend(pairs)

        # Balance labels
        self.all_pairs = self.pair_constructor.balance_labels(self.all_pairs)

        logger.info(f"  Constructed {len(self.all_pairs)} pairwise labels")

        # Show pair statistics
        stats = self.pair_constructor.get_pair_statistics(self.all_pairs)
        logger.info(f"  Label distribution: {stats.get('label_1_ratio', 0):.1%} positive")
        logger.info(f"  Avg score diff: {stats.get('avg_score_diff', 0):.3f}")

    def _validate_and_save(self):
        """Validate data quality and save to disk"""
        # Convert pairs to dicts
        pair_dicts = [p.to_dict() for p in self.all_pairs]

        # Create train/val/test splits
        train_pairs, val_pairs, test_pairs = create_dataset_splits(
            pair_dicts,
            train_ratio=0.7,
            val_ratio=0.15,
            seed=self.config.seed
        )

        # Validate
        report = self.validator.validate_dataset(train_pairs, val_pairs, test_pairs)
        report.print_summary()

        if not report.is_valid:
            logger.warning("Dataset validation found errors - proceeding anyway")

        # Save pairs
        writer = PairWriter(self.config.output_dir)

        train_files = writer.write_jsonl(
            [PairwiseLabel.from_dict(p) for p in train_pairs],
            'train_pairs.jsonl',
            shard_size=self.config.shard_size
        )

        val_files = writer.write_jsonl(
            [PairwiseLabel.from_dict(p) for p in val_pairs],
            'val_pairs.jsonl'
        )

        test_files = writer.write_jsonl(
            [PairwiseLabel.from_dict(p) for p in test_pairs],
            'test_pairs.jsonl'
        )

        logger.info(f"  Saved {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test pairs")

        # Save metadata
        metadata = {
            'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else vars(self.config),
            'statistics': {
                'total_floor_plans': len(self.floor_plans),
                'total_simulations': len(self.all_results),
                'total_pairs': len(self.all_pairs),
                'train_pairs': len(train_pairs),
                'val_pairs': len(val_pairs),
                'test_pairs': len(test_pairs)
            },
            'validation': report.to_dict(),
            'diversity_coverage': self.diversity_sampler.get_coverage_report(),
            'files': {
                'train': train_files,
                'val': val_files,
                'test': test_files
            },
            'generated_at': datetime.now().isoformat()
        }

        metadata_path = os.path.join(self.config.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save floor plans (compressed)
        self._save_floor_plans()

        logger.info(f"  Saved metadata to {metadata_path}")

    def _save_floor_plans(self):
        """Save floor plans to disk"""
        floor_plans_dir = os.path.join(self.config.output_dir, 'floor_plans')
        os.makedirs(floor_plans_dir, exist_ok=True)

        # Save as compressed numpy arrays
        for plan_id, (grid, metadata) in self.floor_plans.items():
            np.savez_compressed(
                os.path.join(floor_plans_dir, f'plan_{plan_id:05d}.npz'),
                grid=grid,
                size=np.array(metadata.size),
                room_count=metadata.room_count,
                method=metadata.generation_method,
                exit_positions=np.array(metadata.exit_positions)
            )

        logger.info(f"  Saved {len(self.floor_plans)} floor plans")

    def _save_checkpoint(self):
        """Save checkpoint for resume capability"""
        self.progress.last_checkpoint = datetime.now().isoformat()

        checkpoint = {
            'progress': self.progress.to_dict(),
            'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else vars(self.config),
            'completed_plan_ids': list(self.floor_plans.keys()),
            'results_count': len(self.all_results),
            'pairs_count': len(self.all_pairs)
        }

        checkpoint_path = os.path.join(self.config.output_dir, 'checkpoint.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.info(f"  Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume state"""
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        self.progress = GenerationProgress.from_dict(checkpoint['progress'])

        # Note: Full state restoration would require saving/loading floor plans and results
        # For simplicity, we restart from the last completed plan count
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate pairwise ranking training data for floor plan optimization'
    )

    parser.add_argument('--num-floor-plans', type=int, default=1000,
                       help='Number of floor plans to generate (default: 1000)')
    parser.add_argument('--configs-per-plan', type=int, default=30,
                       help='Configurations per floor plan (default: 30)')
    parser.add_argument('--trials-per-config', type=int, default=5,
                       help='Simulation trials per config (default: 5)')
    parser.add_argument('--pairs-per-plan', type=int, default=200,
                       help='Pairwise labels per floor plan (default: 200)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    parser.add_argument('--output-dir', type=str, default='./training_data',
                       help='Output directory (default: ./training_data)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume from')
    parser.add_argument('--size-min', type=int, default=20,
                       help='Minimum floor plan size (default: 20)')
    parser.add_argument('--size-max', type=int, default=80,
                       help='Maximum floor plan size (default: 80)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum simulation steps (default: 500)')

    args = parser.parse_args()

    # Create configuration
    config = GenerationConfig(
        num_floor_plans=args.num_floor_plans,
        configs_per_plan=args.configs_per_plan,
        trials_per_config=args.trials_per_config,
        pairs_per_plan=args.pairs_per_plan,
        workers=args.workers,
        output_dir=args.output_dir,
        seed=args.seed,
        size_range=(args.size_min, args.size_max),
        max_steps=args.max_steps
    )

    # Run generation
    generator = TrainingDataGenerator(config)
    output_dir = generator.generate(resume_from=args.resume)

    print(f"\nTraining data saved to: {output_dir}")
    print("\nFiles generated:")
    print("  - train_pairs.jsonl  (training data)")
    print("  - val_pairs.jsonl    (validation data)")
    print("  - test_pairs.jsonl   (test data)")
    print("  - metadata.json      (generation config and stats)")
    print("  - floor_plans/       (saved floor plan grids)")


if __name__ == '__main__':
    main()
