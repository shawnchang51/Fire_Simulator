"""
Training Data Generation V5 - Hierarchical Exit + Door Configuration

Three-tier hierarchy:
1. Floor Plan (from ResPlan dataset)
2. Exit Configuration (M different exit placements per plan)
3. Door Configuration (N different door placements per exit config, keeping door count constant)

Three pair types for ranking:
- Same-exit pairs: Same plan, same exits, different door placements (most pairs)
- Cross-exit pairs: Same plan, different exits (second most)
- Cross-plan pairs: Different plans (least)
"""

import os
import sys
import json
import time
import argparse
import logging
import traceback
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_scenario(
    passable_positions: List[Tuple[int, int]],
    mc_params: Dict[str, Any],
    seed: int
) -> Dict[str, Any]:
    """
    Generate a single scenario (fire/agent positions and parameters).

    This scenario can be reused across multiple door configurations to ensure
    fair comparisons - all configs are evaluated under identical conditions.
    """
    rng = np.random.default_rng(seed)

    # Occupant density and agent count
    occupant_density = rng.uniform(
        mc_params['occupant_density_range'][0],
        mc_params['occupant_density_range'][1]
    )
    agent_count = int(len(passable_positions) * occupant_density)
    agent_count = max(5, min(agent_count, len(passable_positions)))

    # Random agent positions
    agent_indices = rng.choice(len(passable_positions), size=agent_count, replace=False)
    agent_positions = [passable_positions[i] for i in agent_indices]

    # Fire count
    num_fires = rng.integers(
        mc_params['num_fires_range'][0],
        mc_params['num_fires_range'][1] + 1
    )

    # Fire positions (avoid agents)
    available_for_fire = [pos for pos in passable_positions if pos not in agent_positions]
    if len(available_for_fire) < num_fires:
        num_fires = max(1, len(available_for_fire))

    fire_indices = rng.choice(len(available_for_fire), size=num_fires, replace=False)
    fire_positions = [available_for_fire[i] for i in fire_indices]

    # Fire parameters
    fire_spread_rate = rng.uniform(
        mc_params['fire_spread_rate_range'][0],
        mc_params['fire_spread_rate_range'][1]
    )
    fire_intensity_growth = rng.uniform(
        mc_params['fire_intensity_growth_range'][0],
        mc_params['fire_intensity_growth_range'][1]
    )
    fire_discovery_delay = rng.integers(
        mc_params['fire_discovery_delay_range'][0],
        mc_params['fire_discovery_delay_range'][1] + 1
    )

    # Convert all numpy types to native Python types for JSON serialization
    return {
        'agent_positions': [tuple(pos) for pos in agent_positions],
        'fire_positions': [tuple(pos) for pos in fire_positions],
        'fire_spread_rate': float(fire_spread_rate),
        'fire_intensity_growth': float(fire_intensity_growth),
        'fire_discovery_delay': int(fire_discovery_delay),
        'fire_damage_threshold': float(mc_params['fire_damage_threshold']),
        'occupant_density': float(occupant_density),
        'agent_count': int(agent_count),
        'num_fires': int(num_fires),
        'seed': int(seed)
    }


def compute_scenario_hash(scenario: Dict[str, Any]) -> str:
    """
    Compute a deterministic hash for a scenario.

    Used to match pairs that were evaluated under identical conditions.
    """
    # Create canonical representation
    canonical = json.dumps({
        'agent_positions': sorted(scenario['agent_positions']),
        'fire_positions': sorted(scenario['fire_positions']),
        'fire_spread_rate': round(scenario['fire_spread_rate'], 4),
        'fire_intensity_growth': round(scenario['fire_intensity_growth'], 4),
        'fire_discovery_delay': scenario['fire_discovery_delay'],
    }, sort_keys=True)

    return hashlib.md5(canonical.encode()).hexdigest()[:12]

# Local imports
from resplan_loader import (
    ResPlanLoader,
    ResPlanFloorPlan,
    apply_door_config,
    get_resplan_door_config
)
from candidate_generator import CandidateGenerator
from pair_constructor import PairConstructor, SimulationResult, PairwiseLabel, PairWriter
from data_validator import DataValidator, create_dataset_splits


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfigV5:
    """Configuration for V5 training data generation with hierarchical exit/door configs"""

    # ResPlan source
    resplan_pkl_path: str = 'ResPlan/ResPlan.pkl'
    cell_size_m: float = 0.3

    # Sampling
    num_floor_plans: int = 1000

    # Hierarchical configuration generation
    exit_configs_per_plan: int = 5          # M different exit configurations
    door_configs_per_exit: int = 6          # N different door placements per exit config
    monte_carlo_runs_per_config: int = 10   # Monte Carlo runs per door config

    # Pairing strategy (must sum to 1.0)
    # Cross-plan pairs are disabled by default to prevent distribution bias:
    # comparing scores across plans conflates plan difficulty with door quality
    same_exit_ratio: float = 0.75   # Same plan, same exits, different doors
    cross_exit_ratio: float = 0.25  # Same plan, different exits
    cross_plan_ratio: float = 0.0   # Different plans (disabled - causes bias)
    pairs_per_plan: int = 300       # Total pairs per floor plan

    # Workers
    workers: int = 8

    # Exit/Door configuration parameters
    num_exits_range: Tuple[int, int] = (1, 3)
    num_doors_range: Tuple[int, int] = (2, 6)  # Allow variable door count
    min_door_spacing: int = 3
    min_exit_spacing: int = 5

    # Monte Carlo simulation parameters
    occupant_density_range: Tuple[float, float] = (0.05, 0.15)
    max_steps: int = 500

    # Fire parameter ranges (randomized per MC run) - INCREASED DIFFICULTY
    num_fires_range: Tuple[int, int] = (15, 25)  # Was: (3, 7) - more fires
    fire_spread_rate_range: Tuple[float, float] = (0.5, 1.5)  # Was: (0.3, 0.8) - faster spread
    fire_intensity_growth_range: Tuple[float, float] = (0.8, 1.8)  # Was: (0.5, 1.5) - faster growth
    fire_discovery_delay_range: Tuple[int, int] = (10, 40)  # Was: (5, 30) - longer delays

    # Fixed parameters
    fire_damage_threshold: float = 10.0

    # Output parameters
    output_dir: str = './training_data_v5'
    checkpoint_interval: int = 50

    # Random seed
    seed: int = 42


@dataclass
class HierarchicalSimulationResult(SimulationResult):
    """Extended SimulationResult with exit_config_id for hierarchical pairing"""
    exit_config_id: int = 0

    # Note: scenario_hash is inherited from SimulationResult base class


def evaluate_door_config_with_scenario(args: Tuple) -> Dict[str, Any]:
    """
    Evaluate a single door configuration with a PRE-DEFINED scenario.

    This ensures that multiple door configurations are evaluated under identical
    conditions (same fire/agent positions), enabling valid pairwise comparisons.

    Args:
        args: (floor_plan_id, exit_config_id, config_id, base_grid_data, door_config, scenario, max_steps)

    Returns:
        Dict with door config, scenario_hash, and simulation metrics
    """
    floor_plan_id, exit_config_id, config_id, base_grid_data, door_config, scenario, max_steps = args

    base_grid = np.array(base_grid_data, dtype=np.float32)
    scenario_hash = compute_scenario_hash(scenario)

    try:
        from fast_simulation import FastEvacuationSim

        # Apply door configuration to base grid (open the doors)
        grid = apply_door_config(base_grid, door_config)

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
                'exit_config_id': exit_config_id,
                'config_id': config_id,
                'scenario_hash': scenario_hash,
                'error': 'No exits found in door config'
            }

        # Use the pre-defined scenario
        agent_positions = scenario['agent_positions']
        fire_positions = scenario['fire_positions']

        try:
            sim = FastEvacuationSim(
                grid=grid.copy(),
                agent_starts=agent_positions,
                exits=exit_positions,
                fire_starts=fire_positions,
                deterministic_fire=False,
                fire_update_interval=2,
                fire_discovery_delay=scenario['fire_discovery_delay'],
                fire_spread_mode='always_real',
                fire_spread_rate=scenario['fire_spread_rate'],
                fire_intensity_growth=scenario['fire_intensity_growth'],
                fire_damage_threshold=scenario['fire_damage_threshold']
            )

            result = sim.run(max_steps=max_steps)

            return {
                'floor_plan_id': floor_plan_id,
                'exit_config_id': exit_config_id,
                'config_id': config_id,
                'door_config': door_config,
                'scenario_hash': scenario_hash,
                'scenario': {
                    'agent_count': scenario['agent_count'],
                    'num_fires': scenario['num_fires'],
                    'fire_spread_rate': scenario['fire_spread_rate'],
                    'fire_discovery_delay': scenario['fire_discovery_delay']
                },
                'survival_rate': float(result.survival_rate),
                'avg_steps': float(result.steps),
                'avg_fire_damage': float(result.avg_fire_damage),
                'evacuated': int(result.evacuated),
                'dead': int(result.dead),
                'survived': int(result.evacuated),
                'num_runs': 1,
                'success_runs': 1
            }

        except Exception as e:
            return {
                'floor_plan_id': floor_plan_id,
                'exit_config_id': exit_config_id,
                'config_id': config_id,
                'scenario_hash': scenario_hash,
                'error': f'Simulation failed: {e}'
            }

    except Exception as e:
        return {
            'floor_plan_id': floor_plan_id,
            'exit_config_id': exit_config_id,
            'config_id': config_id,
            'scenario_hash': scenario_hash,
            'error': str(e)
        }


def evaluate_door_config_monte_carlo_v5(args: Tuple) -> Dict[str, Any]:
    """
    DEPRECATED: Use evaluate_door_config_with_scenario for scenario-consistent evaluation.

    Evaluate a single door configuration with randomized monte carlo runs.
    This function generates random scenarios internally, which prevents scenario matching.

    Args:
        args: (floor_plan_id, exit_config_id, config_id, base_grid_data, door_config, mc_params)

    Returns:
        Dict with door config, exit_config_id, and monte carlo statistics
    """
    floor_plan_id, exit_config_id, config_id, base_grid_data, door_config, mc_params = args

    base_grid = np.array(base_grid_data, dtype=np.float32)

    try:
        from fast_simulation import FastEvacuationSim

        # Apply door configuration to base grid (open the doors)
        grid = apply_door_config(base_grid, door_config)

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
                'exit_config_id': exit_config_id,
                'config_id': config_id,
                'error': 'No exits found in door config'
            }

        rows, cols = grid.shape

        # Find valid positions for agent/fire placement (passable cells)
        passable_positions = []
        for y in range(rows):
            for x in range(cols):
                if grid[y, x] == 0:  # Passable
                    passable_positions.append((x, y))

        if len(passable_positions) < 10:
            return {
                'floor_plan_id': floor_plan_id,
                'exit_config_id': exit_config_id,
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
            agent_count = max(5, min(agent_count, len(passable_positions)))

            # Random agent positions
            agent_indices = np.random.choice(
                len(passable_positions),
                size=agent_count,
                replace=False
            )
            agent_positions = [passable_positions[i] for i in agent_indices]

            # Randomize fire count
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
                'exit_config_id': exit_config_id,
                'config_id': config_id,
                'error': 'All trials failed'
            }

        # Aggregate results (median for robustness)
        return {
            'floor_plan_id': floor_plan_id,
            'exit_config_id': exit_config_id,
            'config_id': config_id,
            'door_config': door_config,
            'scenario_hash': '',  # No scenario hash - randomized per trial
            'survival_rate': float(np.median([r['survival_rate'] for r in trial_results])),
            'avg_steps': float(np.median([r['steps'] for r in trial_results])),
            'avg_fire_damage': float(np.median([r['avg_fire_damage'] for r in trial_results])),
            'evacuated': int(np.median([r['evacuated'] for r in trial_results])),
            'dead': int(np.median([r['dead'] for r in trial_results])),
            'survived': int(np.median([r['evacuated'] for r in trial_results])),
            'num_runs': len(trial_results),
            'success_runs': len(trial_results)
        }

    except Exception as e:
        logger.error(f"Error evaluating config {config_id} for plan {floor_plan_id}: {e}")
        logger.error(traceback.format_exc())
        return {
            'floor_plan_id': floor_plan_id,
            'exit_config_id': exit_config_id,
            'config_id': config_id,
            'error': str(e)
        }


class HierarchicalPairConstructor:
    """
    Constructs three types of pairwise labels:
    1. Same-exit: Same plan, same exit config, different door placements
    2. Cross-exit: Same plan, different exit configs
    3. Cross-plan: Different plans
    """

    def __init__(
        self,
        pair_constructor: PairConstructor,
        same_exit_ratio: float = 0.70,
        cross_exit_ratio: float = 0.20,
        cross_plan_ratio: float = 0.10
    ):
        assert abs(same_exit_ratio + cross_exit_ratio + cross_plan_ratio - 1.0) < 0.01, \
            "Ratios must sum to 1.0"

        self.pair_constructor = pair_constructor
        self.same_exit_ratio = same_exit_ratio
        self.cross_exit_ratio = cross_exit_ratio
        self.cross_plan_ratio = cross_plan_ratio

    def construct_hierarchical_pairs(
        self,
        results: List[HierarchicalSimulationResult],
        num_pairs: int
    ) -> List[PairwiseLabel]:
        """
        Construct three types of pairs with specified ratios.

        Args:
            results: List of hierarchical simulation results
            num_pairs: Total number of pairs to generate

        Returns:
            List of PairwiseLabel objects with appropriate pair_type tags
        """
        # Group results by plan and exit config
        by_plan_exit = defaultdict(list)
        by_plan = defaultdict(list)

        for result in results:
            key = (result.floor_plan_id, result.exit_config_id)
            by_plan_exit[key].append(result)
            by_plan[result.floor_plan_id].append(result)

        # Calculate pair counts
        num_same_exit = int(num_pairs * self.same_exit_ratio)
        num_cross_exit = int(num_pairs * self.cross_exit_ratio)
        num_cross_plan = num_pairs - num_same_exit - num_cross_exit

        all_pairs = []

        # 1. Same-exit pairs (same plan, same exit config, different doors)
        logger.info(f"  Generating {num_same_exit} same-exit pairs...")
        same_exit_pairs = self._construct_same_exit_pairs(by_plan_exit, num_same_exit)
        all_pairs.extend(same_exit_pairs)

        # 2. Cross-exit pairs (same plan, different exit configs)
        logger.info(f"  Generating {num_cross_exit} cross-exit pairs...")
        cross_exit_pairs = self._construct_cross_exit_pairs(by_plan, num_cross_exit)
        all_pairs.extend(cross_exit_pairs)

        # 3. Cross-plan pairs (different plans)
        logger.info(f"  Generating {num_cross_plan} cross-plan pairs...")
        cross_plan_pairs = self._construct_cross_plan_pairs(by_plan, num_cross_plan)
        all_pairs.extend(cross_plan_pairs)

        # Shuffle
        import random
        random.shuffle(all_pairs)

        logger.info(f"  Total pairs constructed: {len(all_pairs)}")
        return all_pairs

    def _construct_same_exit_pairs(
        self,
        by_plan_exit: Dict[Tuple[int, int], List[HierarchicalSimulationResult]],
        num_pairs: int
    ) -> List[PairwiseLabel]:
        """Construct pairs with same plan and same exit config"""
        pairs = []

        # Distribute pairs across plan-exit groups
        groups = [(k, v) for k, v in by_plan_exit.items() if len(v) >= 2]
        if not groups:
            logger.warning("  No valid groups for same-exit pairs")
            return pairs

        pairs_per_group = max(1, num_pairs // len(groups))

        for (plan_id, exit_id), group_results in groups:
            if len(group_results) < 2:
                continue

            # Use mixed strategy for variety
            group_pairs = self.pair_constructor.construct_pairs(
                group_results,
                num_pairs=pairs_per_group,
                strategy='mixed',
                within_plan_ratio=1.0
            )

            # Tag as same-exit pairs
            for pair in group_pairs:
                pair.pair_type = 'same_exit'

            pairs.extend(group_pairs)

            if len(pairs) >= num_pairs:
                break

        return pairs[:num_pairs]

    def _construct_cross_exit_pairs(
        self,
        by_plan: Dict[int, List[HierarchicalSimulationResult]],
        num_pairs: int
    ) -> List[PairwiseLabel]:
        """Construct pairs with same plan but different exit configs"""
        pairs = []

        for plan_id, plan_results in by_plan.items():
            # Group by exit config within this plan
            by_exit = defaultdict(list)
            for r in plan_results:
                by_exit[r.exit_config_id].append(r)

            # Need at least 2 different exit configs
            if len(by_exit) < 2:
                continue

            exit_ids = list(by_exit.keys())
            attempts = 0
            max_attempts = num_pairs * 3

            while len(pairs) < num_pairs and attempts < max_attempts:
                attempts += 1

                # Pick two different exit configs
                exit_a, exit_b = np.random.choice(exit_ids, size=2, replace=False)

                # Pick random result from each
                result_a = np.random.choice(by_exit[exit_a])
                result_b = np.random.choice(by_exit[exit_b])

                # Create pair
                pair = self.pair_constructor._create_pair(
                    result_a, result_b, 'cross_exit', 'random'
                )

                if pair is not None:
                    pairs.append(pair)

            if len(pairs) >= num_pairs:
                break

        return pairs[:num_pairs]

    def _construct_cross_plan_pairs(
        self,
        by_plan: Dict[int, List[HierarchicalSimulationResult]],
        num_pairs: int
    ) -> List[PairwiseLabel]:
        """Construct pairs with different plans"""
        pairs = []
        plan_ids = list(by_plan.keys())

        if len(plan_ids) < 2:
            logger.warning("  Not enough plans for cross-plan pairs")
            return pairs

        attempts = 0
        max_attempts = num_pairs * 5

        while len(pairs) < num_pairs and attempts < max_attempts:
            attempts += 1

            # Pick two different plans
            plan_a, plan_b = np.random.choice(plan_ids, size=2, replace=False)

            # Pick random result from each
            result_a = np.random.choice(by_plan[plan_a])
            result_b = np.random.choice(by_plan[plan_b])

            # Create pair
            pair = self.pair_constructor._create_pair(
                result_a, result_b, 'cross_plan', 'random'
            )

            if pair is not None:
                pairs.append(pair)

        return pairs[:num_pairs]


class TrainingDataGeneratorV5:
    """
    V5: Hierarchical exit + door configuration with three-tier pairing.

    Hierarchy:
    - Floor plans from ResPlan.pkl
    - Multiple exit configurations per plan
    - Multiple door configurations per exit config (keeping door count constant)
    """

    def __init__(self, config: GenerationConfigV5):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Initialize components
        self.resplan_loader = ResPlanLoader(
            config.resplan_pkl_path,
            cell_size_m=config.cell_size_m
        )
        self.pair_constructor = PairConstructor(seed=config.seed)
        self.hierarchical_pair_constructor = HierarchicalPairConstructor(
            self.pair_constructor,
            same_exit_ratio=config.same_exit_ratio,
            cross_exit_ratio=config.cross_exit_ratio,
            cross_plan_ratio=config.cross_plan_ratio
        )
        self.validator = DataValidator()

        # Storage
        self.floor_plans: Dict[int, ResPlanFloorPlan] = {}
        self.all_results: List[HierarchicalSimulationResult] = []
        self.all_pairs: List[PairwiseLabel] = []

        os.makedirs(config.output_dir, exist_ok=True)

    def generate(self) -> str:
        """Run the full generation pipeline."""
        logger.info("=" * 60)
        logger.info("Training Data Generation V5 (Hierarchical Exit+Door)")
        logger.info("=" * 60)
        logger.info(f"ResPlan source: {self.config.resplan_pkl_path}")
        logger.info(f"Floor plans to use: {self.config.num_floor_plans}")
        logger.info(f"Exit configs per plan: {self.config.exit_configs_per_plan}")
        logger.info(f"Door configs per exit: {self.config.door_configs_per_exit}")
        logger.info(f"Monte Carlo runs per config: {self.config.monte_carlo_runs_per_config}")
        logger.info(f"Pair ratios: same-exit={self.config.same_exit_ratio:.0%}, "
                   f"cross-exit={self.config.cross_exit_ratio:.0%}, "
                   f"cross-plan={self.config.cross_plan_ratio:.0%}")
        logger.info(f"Workers: {self.config.workers}")
        logger.info("=" * 60)

        # Phase 1: Load floor plans from ResPlan
        logger.info("\n[Phase 1/4] Loading ResPlan floor plans...")
        self._load_floor_plans()

        # Phase 2: Generate hierarchical configs and evaluate with monte carlo
        logger.info("\n[Phase 2/4] Evaluating hierarchical configs with Monte Carlo...")
        self._evaluate_hierarchical_configs()

        # Phase 3: Construct three-tier pairwise labels
        logger.info("\n[Phase 3/4] Constructing three-tier pairwise labels...")
        self._construct_hierarchical_pairs()

        # Phase 4: Validate and save
        logger.info("\n[Phase 4/4] Validating and saving...")
        self._validate_and_save()

        logger.info("\n" + "=" * 60)
        logger.info("Generation complete!")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info("=" * 60)

        return self.config.output_dir

    def _load_floor_plans(self):
        """Phase 1: Load and convert ResPlan floor plans"""
        floor_plans_dir = os.path.join(self.config.output_dir, 'floor_plans')
        os.makedirs(floor_plans_dir, exist_ok=True)

        # Load all plans
        plans = self.resplan_loader.load_all()
        logger.info(f"  Loaded {len(plans)} plans from ResPlan.pkl")

        # Random sampling
        target_count = self.config.num_floor_plans
        indices = list(range(len(plans)))
        self.rng.shuffle(indices)

        valid_plans = []
        processed = 0

        for idx in indices:
            plan = plans[idx]
            processed += 1

            try:
                fp = self.resplan_loader.convert_plan(plan, idx)
                if fp is None:
                    continue

                # Skip plans with no internal doors
                if len(fp.door_positions) < 1:
                    logger.debug(f"  Plan {idx} skipped: no internal doors")
                    continue

                valid_plans.append(fp)

                if len(valid_plans) >= target_count:
                    logger.info(f"  Collected {len(valid_plans)} valid plans (processed {processed})")
                    break

            except Exception as e:
                logger.warning(f"  Plan {idx} conversion failed: {e}")
                continue

            if processed % 100 == 0:
                logger.info(f"  Processed {processed} plans ({len(valid_plans)} valid, need {target_count})")

        logger.info(f"  Found {len(valid_plans)} valid plans with doors")

        # Store and save floor plans
        for i, fp in enumerate(valid_plans):
            self.floor_plans[i] = fp

            # Save floor plan to disk
            np.savez_compressed(
                os.path.join(floor_plans_dir, f'plan_{i:05d}.npz'),
                grid=fp.grid,
                door_positions=np.array(fp.door_positions),
                exit_positions=np.array(fp.exit_positions),
                resplan_index=fp.plan_index,
                metadata=fp.metadata
            )

        logger.info(f"  Saved {len(self.floor_plans)} floor plans to {floor_plans_dir}")

    def _generate_exit_configs(
        self,
        floor_plan: ResPlanFloorPlan,
        seed: int
    ) -> List[List[Dict]]:
        """
        Generate M different exit configurations for a floor plan.

        Config 0: Original ResPlan exit positions (baseline)
        Configs 1-M: Generated via CandidateGenerator

        Args:
            floor_plan: The ResPlanFloorPlan object
            seed: Random seed

        Returns:
            List of exit configs (each is a list of exit dicts, doors not included yet)
        """
        exit_configs = []

        # Config 0: Original ResPlan exit positions
        original_exits = [d for d in floor_plan.resplan_door_config if d['type'] == 'exit']
        exit_configs.append(original_exits)

        # Configs 1-M: Generate variations
        try:
            candidate_gen = CandidateGenerator(
                floor_plan=floor_plan.grid,
                min_door_spacing=self.config.min_exit_spacing,
                seed=seed
            )

            for i in range(1, self.config.exit_configs_per_plan):
                # Generate exit-only configuration
                num_exits = self.rng.integers(
                    self.config.num_exits_range[0],
                    self.config.num_exits_range[1] + 1
                )

                generated = candidate_gen.generate_candidate_pool(
                    num_candidates=1,
                    num_doors_range=(0, 0),  # No doors, just exits
                    num_exits_range=(num_exits, num_exits),
                    random_ratio=1.0
                )

                if generated:
                    # Extract only exits
                    exits_only = [d for d in generated[0] if d['type'] == 'exit']
                    exit_configs.append(exits_only)

        except Exception as e:
            logger.warning(f"Exit config generation failed: {e}")

        return exit_configs[:self.config.exit_configs_per_plan]

    def _generate_door_configs_for_exit(
        self,
        floor_plan: ResPlanFloorPlan,
        exit_config: List[Dict],
        num_doors: int,
        seed: int
    ) -> List[List[Dict]]:
        """
        Generate N different door placements for a given exit configuration.

        Keeps the number of doors constant (= num_doors from original plan).

        Config 0: Original ResPlan door positions (if exit config is also original)
        Configs 1-N: Generated variations with same door count

        Args:
            floor_plan: The ResPlanFloorPlan object
            exit_config: Exit configuration (list of exit dicts)
            num_doors: Number of internal doors to generate
            seed: Random seed

        Returns:
            List of full door configs (exits + doors)
        """
        door_configs = []

        # Config 0: Try using original doors if this is the original exit config
        original_exits = [d for d in floor_plan.resplan_door_config if d['type'] == 'exit']
        is_original_exit = (len(exit_config) == len(original_exits))

        if is_original_exit:
            # Use original door positions
            original_doors = [d for d in floor_plan.resplan_door_config if d['type'] == 'door']
            door_configs.append(exit_config + original_doors[:num_doors])

        # Generate additional door placements
        try:
            candidate_gen = CandidateGenerator(
                floor_plan=floor_plan.grid,
                min_door_spacing=self.config.min_door_spacing,
                seed=seed
            )

            num_to_generate = self.config.door_configs_per_exit - len(door_configs)

            generated = candidate_gen.generate_candidate_pool(
                num_candidates=num_to_generate,
                num_doors_range=self.config.num_doors_range,  # Variable door count
                num_exits_range=(0, 0),  # No exits, they're already in exit_config
                random_ratio=0.8
            )

            for gen_config in generated:
                # Combine exits with generated doors
                doors_only = [d for d in gen_config if d['type'] == 'door']
                full_config = exit_config + doors_only
                door_configs.append(full_config)

        except Exception as e:
            logger.warning(f"Door config generation failed: {e}")

        return door_configs[:self.config.door_configs_per_exit]

    def _evaluate_hierarchical_configs(self):
        """
        Evaluate hierarchical configurations: Plan → Exit → Doors

        Uses shared scenarios: all door configs within an (exit_config, scenario) group
        are evaluated under identical fire/agent conditions for valid pairwise comparison.
        """
        total_evals = (len(self.floor_plans) *
                       self.config.exit_configs_per_plan *
                       self.config.door_configs_per_exit *
                       self.config.monte_carlo_runs_per_config)
        completed = 0
        start_time = time.time()
        results_file = os.path.join(self.config.output_dir, 'simulation_results.jsonl')

        logger.info(f"  Total evaluations: {total_evals} "
                   f"({len(self.floor_plans)} plans × {self.config.exit_configs_per_plan} exits × "
                   f"{self.config.door_configs_per_exit} doors × {self.config.monte_carlo_runs_per_config} scenarios)")

        # Monte Carlo parameters for scenario generation
        mc_params = {
            'occupant_density_range': self.config.occupant_density_range,
            'num_fires_range': self.config.num_fires_range,
            'fire_spread_rate_range': self.config.fire_spread_rate_range,
            'fire_intensity_growth_range': self.config.fire_intensity_growth_range,
            'fire_discovery_delay_range': self.config.fire_discovery_delay_range,
            'fire_damage_threshold': self.config.fire_damage_threshold,
        }

        # Phase 1: Collect all tasks upfront (no executor yet)
        logger.info("  Phase 1: Generating all tasks...")
        all_tasks = []
        grid_cache = {}  # Cache grid lists to avoid repeated conversions

        for plan_id, floor_plan in self.floor_plans.items():
            num_doors = len(floor_plan.door_positions)

            exit_configs = self._generate_exit_configs(
                floor_plan,
                seed=self.config.seed + plan_id * 1000
            )

            grid_list = floor_plan.grid.tolist()
            grid_cache[plan_id] = grid_list

            # Find passable positions for scenario generation
            passable_positions = []
            for y in range(floor_plan.grid.shape[0]):
                for x in range(floor_plan.grid.shape[1]):
                    if floor_plan.grid[y, x] == 0:
                        passable_positions.append((x, y))

            for exit_id, exit_config in enumerate(exit_configs):
                door_configs = self._generate_door_configs_for_exit(
                    floor_plan,
                    exit_config,
                    num_doors=num_doors,
                    seed=self.config.seed + plan_id * 1000 + exit_id * 100
                )

                # Pre-generate scenarios for this (plan, exit_config)
                scenarios = []
                for scenario_idx in range(self.config.monte_carlo_runs_per_config):
                    scenario_seed = (self.config.seed + plan_id * 10000 +
                                    exit_id * 1000 + scenario_idx)
                    scenario = generate_scenario(passable_positions, mc_params, scenario_seed)
                    scenarios.append(scenario)

                # Create evaluation tasks
                for config_id, door_config in enumerate(door_configs):
                    for scenario in scenarios:
                        all_tasks.append((
                            plan_id,
                            exit_id,
                            config_id,
                            grid_list,
                            door_config,
                            scenario,
                            self.config.max_steps
                        ))

            if (plan_id + 1) % 50 == 0:
                logger.info(f"    Prepared {plan_id + 1}/{len(self.floor_plans)} floor plans...")

        logger.info(f"  Generated {len(all_tasks)} total tasks")

        # Phase 2: Submit all tasks and process results with batched I/O
        logger.info("  Phase 2: Executing evaluations...")
        WRITE_BATCH_SIZE = 500  # Flush to disk every N results
        result_buffer = []

        with ProcessPoolExecutor(max_workers=self.config.workers) as executor:
            # Submit ALL tasks at once
            futures = {
                executor.submit(evaluate_door_config_with_scenario, task): idx
                for idx, task in enumerate(all_tasks)
            }

            for future in as_completed(futures):
                try:
                    result = future.result()

                    if 'error' in result:
                        logger.warning(f"Evaluation error: {result.get('error')}")
                    else:
                        sim_result = HierarchicalSimulationResult(
                            floor_plan_id=result['floor_plan_id'],
                            exit_config_id=result['exit_config_id'],
                            config_id=result['config_id'],
                            config={'door_config': result['door_config']},
                            scenario=result.get('scenario', {}),
                            survival_rate=result['survival_rate'],
                            avg_evacuation_time=result['avg_steps'] * 0.5,
                            steps=int(result['avg_steps']),
                            evacuated=result['evacuated'],
                            stuck=result['survived'] - result['evacuated'],
                            dead=result['dead'],
                            avg_fire_damage=result['avg_fire_damage'],
                            scenario_hash=result.get('scenario_hash', '')
                        )
                        self.all_results.append(sim_result)

                        # Buffer result for batched writing
                        result_buffer.append({
                            'floor_plan_id': sim_result.floor_plan_id,
                            'exit_config_id': sim_result.exit_config_id,
                            'config_id': sim_result.config_id,
                            'config': sim_result.config,
                            'scenario': sim_result.scenario,
                            'scenario_hash': sim_result.scenario_hash,
                            'survival_rate': sim_result.survival_rate,
                            'avg_evacuation_time': sim_result.avg_evacuation_time,
                            'steps': sim_result.steps,
                            'evacuated': sim_result.evacuated,
                            'stuck': sim_result.stuck,
                            'dead': sim_result.dead,
                            'avg_fire_damage': sim_result.avg_fire_damage,
                            'score': sim_result.score
                        })

                        # Flush buffer when full
                        if len(result_buffer) >= WRITE_BATCH_SIZE:
                            with open(results_file, 'a') as f:
                                for res in result_buffer:
                                    f.write(json.dumps(res, cls=NumpyEncoder) + '\n')
                            result_buffer.clear()

                    completed += 1

                    if completed % 500 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        remaining = total_evals - completed
                        eta = remaining / rate if rate > 0 else 0
                        logger.info(
                            f"  Completed {completed}/{total_evals} evaluations "
                            f"({len(self.all_results)} valid) - {rate:.1f}/s - ETA: {timedelta(seconds=int(eta))}"
                        )

                        if completed % (self.config.checkpoint_interval * 10) == 0:
                            self._save_checkpoint_metadata(completed, total_evals)

                except Exception as e:
                    logger.error(f"Task failed: {e}")

        # Flush remaining buffered results
        if result_buffer:
            with open(results_file, 'a') as f:
                for res in result_buffer:
                    f.write(json.dumps(res, cls=NumpyEncoder) + '\n')
            result_buffer.clear()

        # Clean up grid cache
        grid_cache.clear()

        logger.info(f"  Evaluated {len(self.all_results)} configurations")
        logger.info(f"  Results saved to {results_file}")

    def _construct_hierarchical_pairs(self):
        """Construct three-tier pairwise labels"""
        total_pairs = self.config.pairs_per_plan * len(self.floor_plans)

        logger.info(f"  Target total pairs: {total_pairs}")
        logger.info(f"  Same-exit: {int(total_pairs * self.config.same_exit_ratio)}")
        logger.info(f"  Cross-exit: {int(total_pairs * self.config.cross_exit_ratio)}")
        logger.info(f"  Cross-plan: {int(total_pairs * self.config.cross_plan_ratio)}")

        # Normalize scores by floor plan before pairing
        # This makes margin threshold meaningful across plans with different difficulties
        logger.info(f"  Normalizing scores by floor plan...")
        self.all_results = self.pair_constructor.normalize_scores_by_plan(self.all_results)

        # Construct pairs
        self.all_pairs = self.hierarchical_pair_constructor.construct_hierarchical_pairs(
            self.all_results,
            num_pairs=total_pairs
        )

        # Save pairs incrementally
        raw_pairs_file = os.path.join(self.config.output_dir, 'raw_pairs.jsonl')
        with open(raw_pairs_file, 'w') as f:
            for pair in self.all_pairs:
                f.write(json.dumps(pair.to_dict(), cls=NumpyEncoder) + '\n')

        # Balance labels
        self.all_pairs = self.pair_constructor.balance_labels(self.all_pairs)

        logger.info(f"  Constructed {len(self.all_pairs)} pairwise labels")
        logger.info(f"  Raw pairs saved to {raw_pairs_file}")

        # Statistics
        stats = self.pair_constructor.get_pair_statistics(self.all_pairs)
        logger.info(f"  Label distribution: {stats.get('label_1_ratio', 0):.1%} positive")
        logger.info(f"  Avg score diff: {stats.get('avg_score_diff', 0):.3f}")

        # Pair type breakdown
        type_dist = stats.get('pair_type_distribution', {})
        for pair_type, count in type_dist.items():
            logger.info(f"  {pair_type}: {count} pairs ({count/len(self.all_pairs):.1%})")

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
            'description': 'Hierarchical exit + door configuration comparison',
            'version': 'v5',
            'config': asdict(self.config),
            'statistics': {
                'total_floor_plans': len(self.floor_plans),
                'exit_configs_per_plan': self.config.exit_configs_per_plan,
                'door_configs_per_exit': self.config.door_configs_per_exit,
                'total_configurations': len(self.all_results),
                'total_monte_carlo_runs': len(self.all_results) * self.config.monte_carlo_runs_per_config,
                'total_pairs': len(self.all_pairs),
                'train_pairs': len(train_pairs),
                'val_pairs': len(val_pairs),
                'test_pairs': len(test_pairs)
            },
            'resplan_info': {
                'source': self.config.resplan_pkl_path,
                'plan_count': self.resplan_loader.get_plan_count(),
                'plans_used': len(self.floor_plans)
            },
            'validation': report.to_dict(),
            'generated_at': datetime.now().isoformat()
        }

        with open(os.path.join(self.config.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"  Metadata saved to metadata.json")

    def _save_checkpoint_metadata(self, completed: int, total: int):
        """Save progress checkpoint metadata"""
        checkpoint = {
            'checkpoint_time': datetime.now().isoformat(),
            'progress': {
                'floor_plans_loaded': len(self.floor_plans),
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
        description='Generate training data with hierarchical exit/door configs (V5)'
    )

    parser.add_argument('--resplan-path', type=str, default='ResPlan/ResPlan.pkl',
                        help='Path to ResPlan.pkl file')
    parser.add_argument('--num-floor-plans', type=int, default=1000)
    parser.add_argument('--exit-configs-per-plan', type=int, default=5,
                        help='Number of exit configurations per floor plan')
    parser.add_argument('--door-configs-per-exit', type=int, default=6,
                        help='Number of door configurations per exit config')
    parser.add_argument('--monte-carlo-runs', type=int, default=10,
                        help='Monte Carlo runs per configuration')
    parser.add_argument('--pairs-per-plan', type=int, default=300,
                        help='Total pairs per floor plan')
    parser.add_argument('--same-exit-ratio', type=float, default=0.70)
    parser.add_argument('--cross-exit-ratio', type=float, default=0.20)
    parser.add_argument('--cross-plan-ratio', type=float, default=0.10)
    parser.add_argument('--num-doors-range', type=int, nargs=2, default=[2, 6],
                        help='Range of internal door counts (min, max)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output-dir', type=str, default='./training_data_v5')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    config = GenerationConfigV5(
        resplan_pkl_path=args.resplan_path,
        num_floor_plans=args.num_floor_plans,
        exit_configs_per_plan=args.exit_configs_per_plan,
        door_configs_per_exit=args.door_configs_per_exit,
        monte_carlo_runs_per_config=args.monte_carlo_runs,
        pairs_per_plan=args.pairs_per_plan,
        same_exit_ratio=args.same_exit_ratio,
        cross_exit_ratio=args.cross_exit_ratio,
        cross_plan_ratio=args.cross_plan_ratio,
        num_doors_range=tuple(args.num_doors_range),
        workers=args.workers,
        output_dir=args.output_dir,
        seed=args.seed
    )

    generator = TrainingDataGeneratorV5(config)
    output_dir = generator.generate()

    print(f"\nTraining data saved to: {output_dir}")
    total_sims = (config.num_floor_plans *
                  config.exit_configs_per_plan *
                  config.door_configs_per_exit *
                  config.monte_carlo_runs_per_config)
    print(f"Total simulations: {total_sims:,}")
    print(f"Total pairs: {config.pairs_per_plan * config.num_floor_plans:,}")


if __name__ == '__main__':
    main()
