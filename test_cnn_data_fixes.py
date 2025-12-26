"""
Tests for CNN training data pipeline fixes.

Validates:
- P0-1: No data leakage in train/val/test splits
- P0-2: Scenario hash matching for within-plan pairs
- P1-1: Cross-plan pairs disabled
- P1-2: Score normalization per floor plan
"""

import unittest
import numpy as np
from collections import defaultdict

from pair_constructor import PairConstructor, SimulationResult, PairwiseLabel
from data_validator import create_dataset_splits, DataValidator


class TestP01DataLeakageFix(unittest.TestCase):
    """P0-1: Verify no floor plan appears in multiple splits"""

    def test_no_cross_split_leakage(self):
        """Cross-plan pairs spanning splits should be filtered out"""
        # Create pairs where some cross different floor plans
        pairs = []
        for i in range(100):
            plan_a = i % 10  # Plans 0-9
            plan_b = (i + 1) % 10  # Often different from plan_a
            pairs.append({
                'floor_plan_id_a': plan_a,
                'floor_plan_id_b': plan_b,
                'score_a': 0.8,
                'score_b': 0.7,
                'label': 1,
                'pair_type': 'cross_plan' if plan_a != plan_b else 'within_plan'
            })

        train, val, test = create_dataset_splits(pairs, seed=42)

        # Extract plan IDs from each split
        def get_plan_ids(split_pairs):
            ids = set()
            for p in split_pairs:
                ids.add(p['floor_plan_id_a'])
                ids.add(p['floor_plan_id_b'])
            return ids

        train_plans = get_plan_ids(train)
        val_plans = get_plan_ids(val)
        test_plans = get_plan_ids(test)

        # Verify no overlap
        self.assertEqual(len(train_plans & val_plans), 0, "Train and val share floor plans")
        self.assertEqual(len(train_plans & test_plans), 0, "Train and test share floor plans")
        self.assertEqual(len(val_plans & test_plans), 0, "Val and test share floor plans")

    def test_cross_split_pairs_filtered(self):
        """Pairs where floor_plan_a and floor_plan_b are in different splits should be dropped"""
        # Create explicit cross-plan pairs
        pairs = []
        for plan_a in range(5):
            for plan_b in range(5, 10):
                pairs.append({
                    'floor_plan_id_a': plan_a,
                    'floor_plan_id_b': plan_b,
                    'score_a': 0.8,
                    'score_b': 0.7,
                    'label': 1,
                    'pair_type': 'cross_plan'
                })

        train, val, test = create_dataset_splits(pairs, seed=42)

        # All cross-plan pairs should be filtered since plans 0-4 and 5-9
        # will end up in different splits
        total_remaining = len(train) + len(val) + len(test)
        self.assertLess(total_remaining, len(pairs), "Some cross-split pairs should be filtered")


class TestP02ScenarioConsistency(unittest.TestCase):
    """P0-2: Verify scenario hash matching for within-plan pairs"""

    def test_scenario_hash_checked_for_within_plan(self):
        """Within-plan pairs with different scenario hashes should be rejected"""
        constructor = PairConstructor(margin=0.001, seed=42)

        result_a = SimulationResult(
            floor_plan_id=1,
            config_id=1,
            config={},
            scenario={},
            survival_rate=0.9,
            avg_evacuation_time=50,
            steps=100,
            evacuated=45,
            stuck=3,
            dead=2,
            avg_fire_damage=0.1,
            scenario_hash='abc123'
        )

        result_b = SimulationResult(
            floor_plan_id=1,  # Same plan
            config_id=2,
            config={},
            scenario={},
            survival_rate=0.7,
            avg_evacuation_time=60,
            steps=120,
            evacuated=35,
            stuck=5,
            dead=10,
            avg_fire_damage=0.3,
            scenario_hash='def456'  # Different scenario
        )

        # Should return None because scenario hashes don't match
        pair = constructor._create_pair(result_a, result_b, 'within_plan', 'random')
        self.assertIsNone(pair, "Within-plan pair with different scenarios should be rejected")

    def test_scenario_hash_match_creates_pair(self):
        """Within-plan pairs with matching scenario hashes should be accepted"""
        constructor = PairConstructor(margin=0.001, seed=42)

        result_a = SimulationResult(
            floor_plan_id=1,
            config_id=1,
            config={},
            scenario={},
            survival_rate=0.9,
            avg_evacuation_time=50,
            steps=100,
            evacuated=45,
            stuck=3,
            dead=2,
            avg_fire_damage=0.1,
            scenario_hash='abc123'
        )

        result_b = SimulationResult(
            floor_plan_id=1,  # Same plan
            config_id=2,
            config={},
            scenario={},
            survival_rate=0.7,
            avg_evacuation_time=60,
            steps=120,
            evacuated=35,
            stuck=5,
            dead=10,
            avg_fire_damage=0.3,
            scenario_hash='abc123'  # Same scenario
        )

        pair = constructor._create_pair(result_a, result_b, 'within_plan', 'random')
        self.assertIsNotNone(pair, "Within-plan pair with matching scenarios should be accepted")

    def test_cross_plan_ignores_scenario_hash(self):
        """Cross-plan pairs don't require scenario hash matching"""
        constructor = PairConstructor(margin=0.001, seed=42)

        result_a = SimulationResult(
            floor_plan_id=1,
            config_id=1,
            config={},
            scenario={},
            survival_rate=0.9,
            avg_evacuation_time=50,
            steps=100,
            evacuated=45,
            stuck=3,
            dead=2,
            avg_fire_damage=0.1,
            scenario_hash='abc123'
        )

        result_b = SimulationResult(
            floor_plan_id=2,  # Different plan
            config_id=1,
            config={},
            scenario={},
            survival_rate=0.7,
            avg_evacuation_time=60,
            steps=120,
            evacuated=35,
            stuck=5,
            dead=10,
            avg_fire_damage=0.3,
            scenario_hash='def456'  # Different scenario (ok for cross-plan)
        )

        pair = constructor._create_pair(result_a, result_b, 'cross_plan', 'random')
        self.assertIsNotNone(pair, "Cross-plan pair doesn't require scenario matching")


class TestP11CrossPlanDisabled(unittest.TestCase):
    """P1-1: Verify cross-plan pairs are disabled by default"""

    def test_default_cross_plan_ratio_is_zero(self):
        """Default config should have cross_plan_ratio=0.0"""
        from generate_training_data_v5 import GenerationConfigV5

        config = GenerationConfigV5()
        self.assertEqual(config.cross_plan_ratio, 0.0, "cross_plan_ratio should default to 0.0")
        self.assertAlmostEqual(
            config.same_exit_ratio + config.cross_exit_ratio, 1.0, places=5,
            msg="same_exit + cross_exit should sum to 1.0"
        )


class TestP12ScoreNormalization(unittest.TestCase):
    """P1-2: Verify per-plan z-score normalization"""

    def test_normalization_per_plan(self):
        """Scores should be normalized within each floor plan"""
        constructor = PairConstructor(seed=42)

        # Create results with different score ranges per plan
        results = []

        # Plan 0: high scores (0.8-0.9)
        for i in range(5):
            results.append(SimulationResult(
                floor_plan_id=0,
                config_id=i,
                config={},
                scenario={},
                survival_rate=0.8 + i * 0.02,
                avg_evacuation_time=50,
                steps=100,
                evacuated=40,
                stuck=5,
                dead=5,
                avg_fire_damage=0.1
            ))

        # Plan 1: low scores (0.5-0.6)
        for i in range(5):
            results.append(SimulationResult(
                floor_plan_id=1,
                config_id=i,
                config={},
                scenario={},
                survival_rate=0.5 + i * 0.02,
                avg_evacuation_time=80,
                steps=160,
                evacuated=25,
                stuck=10,
                dead=15,
                avg_fire_damage=0.4
            ))

        # Normalize
        normalized = constructor.normalize_scores_by_plan(results)

        # Check that each plan's normalized scores have mean ~0 and std ~1
        for plan_id in [0, 1]:
            plan_results = [r for r in normalized if r.floor_plan_id == plan_id]
            norm_scores = [r.normalized_score for r in plan_results]

            mean = np.mean(norm_scores)
            std = np.std(norm_scores)

            self.assertAlmostEqual(mean, 0.0, places=1,
                msg=f"Plan {plan_id} normalized mean should be ~0, got {mean}")
            self.assertAlmostEqual(std, 1.0, places=1,
                msg=f"Plan {plan_id} normalized std should be ~1, got {std}")

    def test_effective_score_uses_normalized(self):
        """effective_score should return normalized_score when available"""
        result = SimulationResult(
            floor_plan_id=0,
            config_id=0,
            config={},
            scenario={},
            survival_rate=0.8,
            avg_evacuation_time=50,
            steps=100,
            evacuated=40,
            stuck=5,
            dead=5,
            avg_fire_damage=0.1,
            normalized_score=1.5
        )

        self.assertEqual(result.effective_score, 1.5,
            "effective_score should use normalized_score")

    def test_effective_score_falls_back_to_raw(self):
        """effective_score should fall back to raw score if not normalized"""
        result = SimulationResult(
            floor_plan_id=0,
            config_id=0,
            config={},
            scenario={},
            survival_rate=0.8,
            avg_evacuation_time=50,
            steps=100,
            evacuated=40,
            stuck=5,
            dead=5,
            avg_fire_damage=0.1
        )

        self.assertEqual(result.effective_score, result.score,
            "effective_score should fall back to raw score")


class TestScenarioGeneration(unittest.TestCase):
    """Test scenario generation and hashing utilities"""

    def test_scenario_hash_deterministic(self):
        """Same scenario should produce same hash"""
        from generate_training_data_v5 import generate_scenario, compute_scenario_hash

        mc_params = {
            'occupant_density_range': (0.05, 0.15),
            'num_fires_range': (3, 7),
            'fire_spread_rate_range': (0.3, 0.8),
            'fire_intensity_growth_range': (0.5, 1.5),
            'fire_discovery_delay_range': (5, 30),
            'fire_damage_threshold': 10.0
        }

        passable = [(x, y) for x in range(10) for y in range(10)]

        scenario1 = generate_scenario(passable, mc_params, seed=42)
        scenario2 = generate_scenario(passable, mc_params, seed=42)

        hash1 = compute_scenario_hash(scenario1)
        hash2 = compute_scenario_hash(scenario2)

        self.assertEqual(hash1, hash2, "Same seed should produce same scenario hash")

    def test_different_seeds_different_hashes(self):
        """Different seeds should produce different hashes"""
        from generate_training_data_v5 import generate_scenario, compute_scenario_hash

        mc_params = {
            'occupant_density_range': (0.05, 0.15),
            'num_fires_range': (3, 7),
            'fire_spread_rate_range': (0.3, 0.8),
            'fire_intensity_growth_range': (0.5, 1.5),
            'fire_discovery_delay_range': (5, 30),
            'fire_damage_threshold': 10.0
        }

        passable = [(x, y) for x in range(10) for y in range(10)]

        scenario1 = generate_scenario(passable, mc_params, seed=42)
        scenario2 = generate_scenario(passable, mc_params, seed=43)

        hash1 = compute_scenario_hash(scenario1)
        hash2 = compute_scenario_hash(scenario2)

        self.assertNotEqual(hash1, hash2, "Different seeds should produce different hashes")


if __name__ == '__main__':
    unittest.main(verbosity=2)
