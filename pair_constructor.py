"""
Pair Constructor for Pairwise Ranking Training Data

Constructs pairwise comparisons from simulation results:
- Within-plan pairs: Same floor plan, different configurations
- Cross-plan pairs: Different floor plans (for transfer learning)
- Hard negative mining: Pairs with similar scores
- Balanced sampling: Equal positive/negative labels

Usage:
    constructor = PairConstructor(margin=0.05)
    pairs = constructor.construct_pairs(results, strategy='mixed')
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Iterator
from dataclasses import dataclass, field
from collections import defaultdict
import json
import random


@dataclass
class SimulationResult:
    """Result from a single simulation run"""
    floor_plan_id: int
    config_id: int
    config: Dict[str, Any]  # Door/exit configuration
    scenario: Dict[str, Any]  # Agent count, fire positions, etc.

    # Metrics
    survival_rate: float
    avg_evacuation_time: float
    steps: int
    evacuated: int
    stuck: int
    dead: int
    avg_fire_damage: float

    # Composite score for ranking
    @property
    def score(self) -> float:
        """Compute ranking score (higher is better)"""
        # Primary: survival rate (weight = 1.0)
        # Secondary: efficiency - fewer steps is better (weight = 1.0, INCREASED from 0.5)
        # Tertiary: low fire damage (weight = 0.5, INCREASED from 0.2)
        return (
            self.survival_rate * 1.0 -
            (self.steps / 1000) * 1.0 -
            self.avg_fire_damage * 0.5
        )


@dataclass
class PairwiseLabel:
    """A single pairwise comparison label"""
    floor_plan_id_a: int
    floor_plan_id_b: int
    config_a: Dict[str, Any]
    config_b: Dict[str, Any]
    scenario_a: Dict[str, Any]
    scenario_b: Dict[str, Any]
    score_a: float
    score_b: float
    label: int  # 1 if A > B, 0 if B > A
    label_confidence: float  # How confident (based on score difference)
    pair_type: str  # 'within_plan', 'cross_plan', 'hard_negative'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'floor_plan_id_a': self.floor_plan_id_a,
            'floor_plan_id_b': self.floor_plan_id_b,
            'config_a': self.config_a,
            'config_b': self.config_b,
            'scenario_a': self.scenario_a,
            'scenario_b': self.scenario_b,
            'score_a': self.score_a,
            'score_b': self.score_b,
            'label': self.label,
            'label_confidence': self.label_confidence,
            'pair_type': self.pair_type
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PairwiseLabel':
        return cls(**d)


class PairConstructor:
    """
    Constructs pairwise training labels from simulation results.

    Supports multiple pair construction strategies:
    - random: Uniform random pairing
    - hard_negative: Pairs with similar scores (harder to distinguish)
    - easy: Pairs with large score differences
    - mixed: Combination of strategies for robust training
    """

    def __init__(
        self,
        margin: float = 0.002,  # Reduced from 0.05 to capture smaller score differences
        hard_negative_threshold: float = 0.02,  # Adjusted proportionally
        easy_threshold: float = 0.05,  # Adjusted proportionally
        seed: Optional[int] = None
    ):
        """
        Args:
            margin: Minimum score difference to assign a label (avoid ambiguous pairs)
            hard_negative_threshold: Max score difference for hard negatives
            easy_threshold: Min score difference for easy pairs
            seed: Random seed
        """
        self.margin = margin
        self.hard_negative_threshold = hard_negative_threshold
        self.easy_threshold = easy_threshold
        self.rng = np.random.default_rng(seed)
        self.py_random = random.Random(seed)

    def construct_pairs(
        self,
        results: List[SimulationResult],
        num_pairs: int,
        strategy: str = 'mixed',
        within_plan_ratio: float = 0.8
    ) -> List[PairwiseLabel]:
        """
        Construct pairwise labels from simulation results.

        Args:
            results: List of simulation results
            num_pairs: Number of pairs to generate
            strategy: 'random', 'hard_negative', 'easy', or 'mixed'
            within_plan_ratio: Proportion of within-plan pairs (vs cross-plan)

        Returns:
            List of PairwiseLabel objects
        """
        # Group results by floor plan
        by_plan = defaultdict(list)
        for result in results:
            by_plan[result.floor_plan_id].append(result)

        # Calculate pair distribution
        num_within = int(num_pairs * within_plan_ratio)
        num_cross = num_pairs - num_within

        pairs = []

        # Generate within-plan pairs
        if num_within > 0:
            within_pairs = self._construct_within_plan_pairs(
                by_plan, num_within, strategy
            )
            pairs.extend(within_pairs)

        # Generate cross-plan pairs
        if num_cross > 0:
            cross_pairs = self._construct_cross_plan_pairs(
                by_plan, num_cross, strategy
            )
            pairs.extend(cross_pairs)

        # Shuffle pairs
        self.py_random.shuffle(pairs)

        return pairs

    def _construct_within_plan_pairs(
        self,
        by_plan: Dict[int, List[SimulationResult]],
        num_pairs: int,
        strategy: str
    ) -> List[PairwiseLabel]:
        """Construct pairs from same floor plan"""
        pairs = []

        # Distribute pairs across floor plans
        plan_ids = list(by_plan.keys())
        pairs_per_plan = max(1, num_pairs // len(plan_ids))

        for plan_id in plan_ids:
            plan_results = by_plan[plan_id]
            if len(plan_results) < 2:
                continue

            plan_pairs = self._sample_pairs_from_group(
                plan_results, pairs_per_plan, strategy, 'within_plan'
            )
            pairs.extend(plan_pairs)

            if len(pairs) >= num_pairs:
                break

        return pairs[:num_pairs]

    def _construct_cross_plan_pairs(
        self,
        by_plan: Dict[int, List[SimulationResult]],
        num_pairs: int,
        strategy: str
    ) -> List[PairwiseLabel]:
        """Construct pairs across different floor plans"""
        pairs = []
        plan_ids = list(by_plan.keys())

        if len(plan_ids) < 2:
            return pairs

        attempts = 0
        max_attempts = num_pairs * 10

        while len(pairs) < num_pairs and attempts < max_attempts:
            attempts += 1

            # Sample two different floor plans
            plan_id_a, plan_id_b = self.rng.choice(plan_ids, size=2, replace=False)

            # Sample one result from each
            result_a = self.py_random.choice(by_plan[plan_id_a])
            result_b = self.py_random.choice(by_plan[plan_id_b])

            # Check if valid pair
            pair = self._create_pair(result_a, result_b, 'cross_plan', strategy)
            if pair is not None:
                pairs.append(pair)

        return pairs

    def _sample_pairs_from_group(
        self,
        results: List[SimulationResult],
        num_pairs: int,
        strategy: str,
        pair_type: str
    ) -> List[PairwiseLabel]:
        """Sample pairs from a group of results"""
        pairs = []

        if strategy == 'mixed':
            # 50% random, 30% hard, 20% easy
            num_random = int(num_pairs * 0.5)
            num_hard = int(num_pairs * 0.3)
            num_easy = num_pairs - num_random - num_hard

            pairs.extend(self._sample_random_pairs(results, num_random, pair_type))
            pairs.extend(self._sample_hard_negative_pairs(results, num_hard, pair_type))
            pairs.extend(self._sample_easy_pairs(results, num_easy, pair_type))

        elif strategy == 'random':
            pairs = self._sample_random_pairs(results, num_pairs, pair_type)

        elif strategy == 'hard_negative':
            pairs = self._sample_hard_negative_pairs(results, num_pairs, pair_type)

        elif strategy == 'easy':
            pairs = self._sample_easy_pairs(results, num_pairs, pair_type)

        return pairs

    def _sample_random_pairs(
        self,
        results: List[SimulationResult],
        num_pairs: int,
        pair_type: str
    ) -> List[PairwiseLabel]:
        """Sample random pairs"""
        pairs = []
        attempts = 0
        rejected_below_margin = 0
        max_attempts = num_pairs * 5

        while len(pairs) < num_pairs and attempts < max_attempts:
            attempts += 1

            if len(results) < 2:
                break

            idx_a, idx_b = self.rng.choice(len(results), size=2, replace=False)
            score_diff = abs(results[idx_a].score - results[idx_b].score)
            pair = self._create_pair(results[idx_a], results[idx_b], pair_type, 'random')
            if pair is not None:
                pairs.append(pair)
            elif score_diff < self.margin:
                rejected_below_margin += 1

        if attempts > 0 and len(pairs) == 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"    Random pairs: {attempts} attempts, {len(pairs)} created, "
                          f"{rejected_below_margin} rejected (below margin {self.margin})")

        return pairs

    def _sample_hard_negative_pairs(
        self,
        results: List[SimulationResult],
        num_pairs: int,
        pair_type: str
    ) -> List[PairwiseLabel]:
        """Sample pairs with similar scores (hard negatives)"""
        pairs = []

        # Sort by score
        sorted_results = sorted(results, key=lambda r: r.score)

        attempts = 0
        max_attempts = num_pairs * 5

        while len(pairs) < num_pairs and attempts < max_attempts:
            attempts += 1

            # Pick a random result and find one with similar score
            idx = self.rng.integers(len(sorted_results))
            result_a = sorted_results[idx]

            # Look at neighbors in sorted order
            for offset in [1, -1, 2, -2]:
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(sorted_results):
                    result_b = sorted_results[neighbor_idx]
                    score_diff = abs(result_a.score - result_b.score)

                    if self.margin < score_diff < self.hard_negative_threshold:
                        pair = self._create_pair(result_a, result_b, pair_type, 'hard_negative')
                        if pair is not None:
                            pairs.append(pair)
                            break

        return pairs

    def _sample_easy_pairs(
        self,
        results: List[SimulationResult],
        num_pairs: int,
        pair_type: str
    ) -> List[PairwiseLabel]:
        """Sample pairs with large score differences (easy)"""
        pairs = []

        # Sort by score
        sorted_results = sorted(results, key=lambda r: r.score)
        n = len(sorted_results)

        if n < 2:
            return pairs

        # Calculate quartile boundaries, ensuring valid ranges
        # For small n, fall back to first/last elements
        bottom_end = max(1, n // 4)  # At least 1 element in bottom range
        top_start = min(n - 1, 3 * n // 4)  # At least 1 element in top range

        attempts = 0
        max_attempts = num_pairs * 5

        while len(pairs) < num_pairs and attempts < max_attempts:
            attempts += 1

            # Pick from top and bottom quarters
            top_idx = self.rng.integers(top_start, n)
            bottom_idx = self.rng.integers(0, bottom_end)

            result_a = sorted_results[top_idx]
            result_b = sorted_results[bottom_idx]

            score_diff = abs(result_a.score - result_b.score)
            if score_diff > self.easy_threshold:
                pair = self._create_pair(result_a, result_b, pair_type, 'easy')
                if pair is not None:
                    pairs.append(pair)

        return pairs

    def _create_pair(
        self,
        result_a: SimulationResult,
        result_b: SimulationResult,
        pair_type: str,
        selection_strategy: str
    ) -> Optional[PairwiseLabel]:
        """Create a pairwise label from two results"""
        score_diff = result_a.score - result_b.score

        # Skip ambiguous pairs
        if abs(score_diff) < self.margin:
            return None

        # Determine label
        label = 1 if score_diff > 0 else 0

        # Calculate confidence based on score difference
        confidence = min(1.0, abs(score_diff) / 0.3)

        return PairwiseLabel(
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
            pair_type=f"{pair_type}_{selection_strategy}"
        )

    def balance_labels(
        self,
        pairs: List[PairwiseLabel],
        target_ratio: float = 0.5
    ) -> List[PairwiseLabel]:
        """
        Balance the label distribution by undersampling the majority class.

        Args:
            pairs: List of pairwise labels
            target_ratio: Target proportion of label=1 (default 0.5 for balanced)

        Returns:
            Balanced list of pairs
        """
        label_1 = [p for p in pairs if p.label == 1]
        label_0 = [p for p in pairs if p.label == 0]

        current_ratio = len(label_1) / len(pairs) if pairs else 0.5

        if abs(current_ratio - target_ratio) < 0.05:
            return pairs  # Already balanced

        # Undersample majority class
        if len(label_1) > len(label_0):
            target_1 = int(len(label_0) / (1 - target_ratio) * target_ratio)
            label_1 = self.py_random.sample(label_1, min(target_1, len(label_1)))
        else:
            target_0 = int(len(label_1) / target_ratio * (1 - target_ratio))
            label_0 = self.py_random.sample(label_0, min(target_0, len(label_0)))

        balanced = label_1 + label_0
        self.py_random.shuffle(balanced)
        return balanced

    def get_pair_statistics(self, pairs: List[PairwiseLabel]) -> Dict[str, Any]:
        """Get statistics about the pair distribution"""
        if not pairs:
            return {}

        labels = [p.label for p in pairs]
        confidences = [p.label_confidence for p in pairs]
        pair_types = [p.pair_type for p in pairs]

        type_counts = defaultdict(int)
        for pt in pair_types:
            type_counts[pt] += 1

        score_diffs = [abs(p.score_a - p.score_b) for p in pairs]

        return {
            'total_pairs': len(pairs),
            'label_1_ratio': sum(labels) / len(labels),
            'avg_confidence': np.mean(confidences),
            'avg_score_diff': np.mean(score_diffs),
            'min_score_diff': np.min(score_diffs),
            'max_score_diff': np.max(score_diffs),
            'pair_type_distribution': dict(type_counts),
            'unique_floor_plans': len(set(
                [p.floor_plan_id_a for p in pairs] +
                [p.floor_plan_id_b for p in pairs]
            ))
        }


class PairWriter:
    """Writes pairwise labels to disk in various formats"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def write_jsonl(
        self,
        pairs: List[PairwiseLabel],
        filename: str = 'pairs.jsonl',
        shard_size: Optional[int] = None
    ) -> List[str]:
        """
        Write pairs to JSONL format.

        Args:
            pairs: List of pairwise labels
            filename: Output filename (or prefix if sharding)
            shard_size: If set, split into multiple shards

        Returns:
            List of written file paths
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        written_files = []

        if shard_size is None or len(pairs) <= shard_size:
            # Single file
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w') as f:
                for pair in pairs:
                    f.write(json.dumps(pair.to_dict()) + '\n')
            written_files.append(filepath)
        else:
            # Multiple shards
            base_name = filename.rsplit('.', 1)[0]
            ext = filename.rsplit('.', 1)[1] if '.' in filename else 'jsonl'

            for i in range(0, len(pairs), shard_size):
                shard_pairs = pairs[i:i + shard_size]
                shard_filename = f"{base_name}_shard_{i // shard_size:04d}.{ext}"
                filepath = os.path.join(self.output_dir, shard_filename)

                with open(filepath, 'w') as f:
                    for pair in shard_pairs:
                        f.write(json.dumps(pair.to_dict()) + '\n')

                written_files.append(filepath)

        return written_files

    def write_statistics(
        self,
        pairs: List[PairwiseLabel],
        constructor: PairConstructor,
        filename: str = 'pair_statistics.json'
    ):
        """Write pair statistics to JSON"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        stats = constructor.get_pair_statistics(pairs)

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)

        return filepath


class PairReader:
    """Reads pairwise labels from disk"""

    @staticmethod
    def read_jsonl(filepath: str) -> Iterator[PairwiseLabel]:
        """Read pairs from JSONL file"""
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    yield PairwiseLabel.from_dict(data)

    @staticmethod
    def read_jsonl_shards(filepaths: List[str]) -> Iterator[PairwiseLabel]:
        """Read pairs from multiple JSONL shards"""
        for filepath in filepaths:
            yield from PairReader.read_jsonl(filepath)


if __name__ == '__main__':
    # Test pair construction
    print("Testing PairConstructor...")

    # Create mock results
    results = []
    for plan_id in range(5):
        for config_id in range(20):
            # Random scores with some variance
            base_score = np.random.uniform(0.5, 0.95)
            result = SimulationResult(
                floor_plan_id=plan_id,
                config_id=config_id,
                config={'exits': [{'id': f'e{i}', 'position': f'x{i*10}y0'} for i in range(2)]},
                scenario={'agent_count': 50, 'fire_positions': [(20, 20)]},
                survival_rate=base_score,
                avg_evacuation_time=np.random.uniform(30, 100),
                steps=int(np.random.uniform(50, 200)),
                evacuated=int(base_score * 50),
                stuck=int((1 - base_score) * 25),
                dead=int((1 - base_score) * 25),
                avg_fire_damage=np.random.uniform(0, 0.5)
            )
            results.append(result)

    constructor = PairConstructor(margin=0.05, seed=42)

    # Test different strategies
    for strategy in ['random', 'hard_negative', 'easy', 'mixed']:
        pairs = constructor.construct_pairs(
            results,
            num_pairs=100,
            strategy=strategy,
            within_plan_ratio=0.8
        )

        stats = constructor.get_pair_statistics(pairs)
        print(f"\n{strategy.upper()} Strategy:")
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Label 1 ratio: {stats['label_1_ratio']:.2%}")
        print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
        print(f"  Avg score diff: {stats['avg_score_diff']:.3f}")
        print(f"  Pair types: {stats['pair_type_distribution']}")

    # Test balancing
    pairs = constructor.construct_pairs(results, num_pairs=200, strategy='mixed')
    balanced = constructor.balance_labels(pairs)
    balanced_stats = constructor.get_pair_statistics(balanced)
    print(f"\nAfter balancing: {balanced_stats['label_1_ratio']:.2%} label=1")
