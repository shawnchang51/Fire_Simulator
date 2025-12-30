"""
Pairwise Ranking Integration Interface
======================================

Interface between scoring network and simulator for pairwise comparison labeling.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from fast_simulation import FastEvacuationSim, SimResult, batch_evaluate

class ScoringNetworkInterface:
    """
    Interface for scoring network to request candidate evaluations.

    Architecture:
    - Candidate Generator produces door configurations
    - Scoring Network predicts scalar scores
    - Simulator validates via Monte Carlo and generates pairwise labels
    """

    def __init__(self,
                 grid_size: Tuple[int, int] = (30, 30),
                 base_config: str = 'configs/ai_labeling_config.json',
                 num_trials_per_eval: int = 3):
        """
        Initialize interface.

        Args:
            grid_size: (rows, cols) of the floor plan
            base_config: Path to base simulation config
            num_trials_per_eval: Monte Carlo trials per candidate (k=3-5)
        """
        self.rows, self.cols = grid_size
        self.num_trials = num_trials_per_eval
        self.base_config = base_config

        # Load base config
        import json
        import os
        if os.path.exists(base_config):
            with open(base_config) as f:
                self.config_template = json.load(f)
        else:
            # Use default config if file doesn't exist
            self.config_template = {
                'map_rows': grid_size[0],
                'map_cols': grid_size[1],
                'cell_size': 0.6,
                'timestep_duration': 0.5,
                'fire_update_interval': 8,
                'agent_num': 5,
                'max_occupancy': 4
            }

    def generate_candidate_labels(self,
                                  floor_plan: np.ndarray,
                                  candidate_pool: List[Dict],
                                  num_pairs: int,
                                  pair_selection: str = 'mixed') -> List[Tuple]:
        """
        Generate pairwise comparison labels from candidate pool.

        Args:
            floor_plan: Base floor plan array
            candidate_pool: List of door configuration dicts
            num_pairs: Number of pairs to sample and label
            pair_selection: 'random', 'hard', or 'mixed' sampling strategy

        Returns:
            List of (config_A, config_B, label, score_A, score_B) tuples
        """
        # Sample pairs using specified strategy
        pairs = self._sample_pairs(candidate_pool, num_pairs, pair_selection)

        # Generate labels via simulator
        labels = []
        for config_a, config_b in pairs:
            result_a = self.evaluate_candidate(floor_plan, config_a)
            result_b = self.evaluate_candidate(floor_plan, config_b)

            # Compute simulator scores (survival rate - time penalty)
            score_a = self._compute_score(result_a)
            score_b = self._compute_score(result_b)

            # Assign pairwise label with margin
            margin = 0.05
            if score_a > score_b + margin:
                label = 1  # A > B
            elif score_b > score_a + margin:
                label = 0  # B > A
            else:
                label = None  # Ambiguous, discard

            if label is not None:
                labels.append((config_a, config_b, label, score_a, score_b))

        return labels

    def evaluate_candidate(self, floor_plan: np.ndarray, door_config: Dict) -> Dict:
        """
        Evaluate single candidate with k Monte Carlo trials.

        Args:
            floor_plan: 2D array (-2=wall, 0=empty)
            door_config: Door configuration dict

        Returns:
            Aggregated metrics dict
        """
        results = []
        for trial in range(self.num_trials):
            sim = self._build_simulation(floor_plan, door_config, seed=trial)
            result = sim.run(max_steps=200)
            results.append(result)

        # Return median statistics
        return self._aggregate_results(results)

    def batch_evaluate_topk(self,
                            floor_plan: np.ndarray,
                            candidates: List[Dict],
                            k: int) -> List[Tuple[Dict, float]]:
        """
        Evaluate top-k candidates selected by scoring network.

        Args:
            floor_plan: Base floor plan
            candidates: List of (door_config, predicted_score) tuples
            k: Number of top candidates to validate

        Returns:
            List of (door_config, simulator_score) for top-k
        """
        # Sort by predicted score and take top-k
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

        # Validate with full Monte Carlo
        validated = []
        for door_config, pred_score in sorted_candidates:
            result = self.evaluate_candidate(floor_plan, door_config)
            sim_score = self._compute_score(result)
            validated.append((door_config, sim_score))

        return validated

    def _sample_pairs(self, candidates, num_pairs, strategy):
        """Sample pairs using specified strategy."""
        import random

        pairs = []
        if strategy == 'random':
            # Pure random sampling
            for _ in range(num_pairs):
                a, b = random.sample(candidates, 2)
                pairs.append((a, b))

        elif strategy == 'hard':
            # Sample pairs with similar predicted scores (requires model predictions)
            # For now, default to random
            for _ in range(num_pairs):
                a, b = random.sample(candidates, 2)
                pairs.append((a, b))

        elif strategy == 'mixed':
            # 70% random, 30% hard pairs
            num_random = int(num_pairs * 0.7)
            num_hard = num_pairs - num_random

            for _ in range(num_random):
                a, b = random.sample(candidates, 2)
                pairs.append((a, b))

            for _ in range(num_hard):
                a, b = random.sample(candidates, 2)
                pairs.append((a, b))

        return pairs

    def _build_simulation(self, floor_plan, door_config, seed):
        """Build simulation from floor plan and door config."""
        # Extract agent and exit positions from door config
        agent_positions = []
        exit_positions = []
        fire_positions = []

        # Parse door config to extract exits
        if isinstance(door_config, list):
            for door in door_config:
                pos_str = door.get('position', '')
                if 'x' in pos_str and 'y' in pos_str:
                    # Parse "x5y3" format
                    parts = pos_str.split('y')
                    x = int(parts[0][1:])
                    y = int(parts[1])

                    if door.get('type') == 'exit':
                        exit_positions.append((x, y))

        # If no exits found in door config, find exits from floor plan
        if not exit_positions:
            # Look for perimeter cells that aren't walls
            rows, cols = floor_plan.shape
            for x in range(cols):
                if floor_plan[0, x] != -2:  # Top edge
                    exit_positions.append((x, 0))
                if floor_plan[rows-1, x] != -2:  # Bottom edge
                    exit_positions.append((x, rows-1))
            for y in range(rows):
                if floor_plan[y, 0] != -2:  # Left edge
                    exit_positions.append((0, y))
                if floor_plan[y, cols-1] != -2:  # Right edge
                    exit_positions.append((cols-1, y))

        # Generate random agent positions if not specified
        if not agent_positions:
            num_agents = self.config_template.get('agent_num', 5)
            rows, cols = floor_plan.shape
            for _ in range(num_agents * 3):  # Try up to 3x to find valid positions
                if len(agent_positions) >= num_agents:
                    break
                x = np.random.randint(1, cols - 1)
                y = np.random.randint(1, rows - 1)
                if floor_plan[y, x] == 0:  # Empty cell
                    agent_positions.append((x, y))

        # Find fire positions from floor plan
        fire_y, fire_x = np.where(floor_plan > 0)
        fire_positions = list(zip(fire_x, fire_y))

        if seed is not None:
            np.random.seed(seed)

        sim = FastEvacuationSim(
            grid=floor_plan.copy(),
            agent_starts=agent_positions[:self.config_template.get('agent_num', 5)],
            exits=exit_positions[:max(1, len(exit_positions))],
            fire_starts=fire_positions if fire_positions else None,
            deterministic_fire=True,
            fire_update_interval=self.config_template.get('fire_update_interval', 4)
        )

        return sim

    def _aggregate_results(self, results):
        """Aggregate trial results using median."""
        evacuated = np.median([r.evacuated for r in results])
        steps = np.median([r.steps for r in results])
        stuck = np.median([r.stuck for r in results])
        dead = np.median([r.dead for r in results])

        total = results[0].evacuated + results[0].stuck + results[0].dead

        return {
            'evacuated': int(evacuated),
            'stuck': int(stuck),
            'dead': int(dead),
            'steps': int(steps),
            'survival_rate': evacuated / total if total > 0 else 0
        }

    def _compute_score(self, result):
        """Compute simulator score from metrics."""
        # Survival rate minus time penalty
        return result['survival_rate'] - (result['steps'] / 1000)

    def log_evaluation(self, floor_plan, door_config, model_score, sim_score):
        """Log candidate evaluation for later analysis."""
        # Log to file for correlation analysis and fine-tuning
        import json
        import time

        log_entry = {
            'timestamp': time.time(),
            'floor_plan_hash': hash(floor_plan.tobytes()),
            'door_config': door_config,
            'model_score': float(model_score),
            'sim_score': float(sim_score)
        }

        with open('candidate_evaluations.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


def generate_training_labels(simulator_interface,
                            floor_plans: List[np.ndarray],
                            candidates_per_plan: int = 50,
                            pairs_per_plan: int = 100,
                            output_path: str = 'pairwise_labels.jsonl') -> Dict:
    """
    Generate pairwise training labels for scoring network.

    Args:
        simulator_interface: ScoringNetworkInterface instance
        floor_plans: List of base floor plans
        candidates_per_plan: Number of door configs to generate per plan
        pairs_per_plan: Number of pairs to sample per plan
        output_path: Where to save labels

    Returns:
        Statistics about label generation
    """
    import json

    # Import candidate generator if available
    try:
        from candidate_generator import generate_door_candidates
    except ImportError:
        print("Warning: candidate_generator not found. Using dummy candidates.")
        def generate_door_candidates(floor_plan, num_candidates=50):
            # Return empty list as placeholder
            return [[] for _ in range(num_candidates)]

    total_labels = 0
    ambiguous_count = 0

    with open(output_path, 'w') as f:
        for plan_idx, floor_plan in enumerate(floor_plans):
            # Generate candidate pool
            candidates = generate_door_candidates(
                floor_plan,
                num_candidates=candidates_per_plan
            )

            # Generate pairwise labels
            labels = simulator_interface.generate_candidate_labels(
                floor_plan,
                candidates,
                num_pairs=pairs_per_plan,
                pair_selection='mixed'
            )

            # Save labels
            for config_a, config_b, label, score_a, score_b in labels:
                if label is not None:
                    entry = {
                        'floor_plan_id': plan_idx,
                        'config_a': config_a,
                        'config_b': config_b,
                        'label': label,
                        'score_a': score_a,
                        'score_b': score_b
                    }
                    f.write(json.dumps(entry) + '\n')
                    total_labels += 1
                else:
                    ambiguous_count += 1

            print(f"Generated {total_labels} labels for floor plan {plan_idx + 1}/{len(floor_plans)}")

    return {
        'total_labels': total_labels,
        'ambiguous_discarded': ambiguous_count,
        'label_rate': total_labels / (total_labels + ambiguous_count) if (total_labels + ambiguous_count) > 0 else 0
    }
