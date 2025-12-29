"""
Pairwise Dataset for Ranking Training

Loads pairwise labels from *_pairs.jsonl files and provides
(grid_a, scenario_a, grid_b, scenario_b, label, confidence) tuples.

Key Design:
    - Normalization stats computed from TRAIN split only (prevents leakage)
    - Efficient caching of floor plans in memory
    - Ground truth scores included for per-plan ranking evaluation
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import RankingConfig


def parse_position(pos_str: str) -> Tuple[int, int]:
    """
    Parse position string 'x16y0' -> (x=16, y=0).

    Args:
        pos_str: Position string in format 'xNyM'

    Returns:
        Tuple of (x, y) coordinates
    """
    match = re.match(r'x(\d+)y(\d+)', pos_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Invalid position format: {pos_str}")


def compute_scenario_stats(
    pairs_file: str,
    floor_plan_ids: Optional[Set[int]] = None
) -> Dict[str, List[float]]:
    """
    Compute mean and std for scenario parameters from pairs file.

    Args:
        pairs_file: Path to pairs JSONL file
        floor_plan_ids: Optional filter by floor plan IDs

    Returns:
        Dict with 'means' and 'stds' for scenario normalization
    """
    scenario_values = {
        'agent_count': [],
        'num_fires': [],
        'fire_spread_rate': [],
        'fire_discovery_delay': []
    }

    with open(pairs_file, 'r') as f:
        for line in f:
            pair = json.loads(line)

            # Filter by floor plan ID if specified
            if floor_plan_ids is not None:
                if pair['floor_plan_id_a'] not in floor_plan_ids:
                    continue

            # Collect scenario values from both A and B
            for suffix in ['a', 'b']:
                scenario = pair[f'scenario_{suffix}']
                scenario_values['agent_count'].append(scenario['agent_count'])
                scenario_values['num_fires'].append(scenario['num_fires'])
                scenario_values['fire_spread_rate'].append(scenario['fire_spread_rate'])
                scenario_values['fire_discovery_delay'].append(scenario['fire_discovery_delay'])

    # Compute means and stds
    means = [np.mean(scenario_values[k]) for k in
             ['agent_count', 'num_fires', 'fire_spread_rate', 'fire_discovery_delay']]
    stds = [np.std(scenario_values[k]) for k in
            ['agent_count', 'num_fires', 'fire_spread_rate', 'fire_discovery_delay']]

    # Prevent division by zero
    stds = [max(s, 1e-6) for s in stds]

    return {'means': means, 'stds': stds}


class PairwiseDataset(Dataset):
    """
    PyTorch Dataset for pairwise ranking training.

    Loads pairwise labels and provides grid/scenario tensors for both
    configurations in each pair.

    Attributes:
        pairs: List of pairwise label dictionaries
        floor_plans: Dict mapping floor_plan_id to grid data
        scenario_stats: Normalization statistics for scenarios
    """

    def __init__(
        self,
        pairs_file: str,
        floor_plans_dir: str,
        target_size: Tuple[int, int] = (96, 128),
        scenario_stats: Optional[Dict] = None,
        compute_stats: bool = False,
        max_pairs: Optional[int] = None
    ):
        """
        Initialize the dataset.

        Args:
            pairs_file: Path to pairs JSONL file
            floor_plans_dir: Path to floor_plans/ directory
            target_size: (H, W) target size for grid padding
            scenario_stats: Pre-computed normalization stats (from train set)
            compute_stats: If True, compute stats from this file
            max_pairs: Maximum number of pairs to load (for debugging)
        """
        self.target_size = target_size

        # Load pairs
        self.pairs = self._load_pairs(pairs_file, max_pairs)

        # Get unique floor plan IDs
        fp_ids = set()
        for pair in self.pairs:
            fp_ids.add(pair['floor_plan_id_a'])
            fp_ids.add(pair['floor_plan_id_b'])

        # Load floor plans
        self.floor_plans = self._load_floor_plans(floor_plans_dir, fp_ids)

        # Compute or use provided scenario stats
        if compute_stats:
            self.scenario_stats = compute_scenario_stats(pairs_file)
        elif scenario_stats is not None:
            self.scenario_stats = scenario_stats
        else:
            self.scenario_stats = None

    def _load_pairs(self, pairs_file: str, max_pairs: Optional[int]) -> List[Dict]:
        """Load pairs from JSONL file."""
        pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
                    if max_pairs is not None and len(pairs) >= max_pairs:
                        break
        return pairs

    def _load_floor_plans(
        self,
        floor_plans_dir: str,
        floor_plan_ids: Set[int]
    ) -> Dict[int, Dict]:
        """Load floor plan grids into memory."""
        plans = {}
        floor_plans_path = Path(floor_plans_dir)

        for plan_id in floor_plan_ids:
            plan_file = floor_plans_path / f"plan_{plan_id:05d}.npz"
            if plan_file.exists():
                data = np.load(plan_file, allow_pickle=True)
                plans[plan_id] = {
                    'grid': data['grid'],
                    'door_positions': data['door_positions'],
                    'exit_positions': data['exit_positions']
                }

        return plans

    def _encode_grid(self, floor_plan_id: int, config: Dict) -> torch.Tensor:
        """
        Create 4-channel tensor from grid and door/exit config.

        Channels:
            0: Wall mask (grid == -2)
            1: Passable mask (grid == 0)
            2: Door positions
            3: Exit positions

        Args:
            floor_plan_id: Floor plan ID
            config: Door/exit configuration dict

        Returns:
            Grid tensor of shape (4, H, W)
        """
        floor_plan = self.floor_plans[floor_plan_id]
        grid = floor_plan['grid']
        H, W = grid.shape
        tH, tW = self.target_size

        # Create 4-channel encoding
        encoded = np.zeros((4, tH, tW), dtype=np.float32)

        # Handle cases where grid is larger than target size - clip
        H_copy = min(H, tH)
        W_copy = min(W, tW)

        # Channel 0: Wall mask (including padding)
        encoded[0, :H_copy, :W_copy] = (grid[:H_copy, :W_copy] == -2).astype(np.float32)
        encoded[0, H_copy:, :] = 1.0  # Padding as walls
        encoded[0, :, W_copy:] = 1.0

        # Channel 1: Passable mask
        encoded[1, :H_copy, :W_copy] = (grid[:H_copy, :W_copy] == 0).astype(np.float32)

        # Channels 2 & 3: Doors and exits from config
        for item in config.get('door_config', []):
            x, y = parse_position(item['position'])
            if 0 <= y < tH and 0 <= x < tW:
                channel = 3 if item['type'] == 'exit' else 2
                encoded[channel, y, x] = 1.0

        return torch.from_numpy(encoded)

    def _normalize_scenario(self, scenario: Dict) -> torch.Tensor:
        """
        Normalize scenario parameters.

        Args:
            scenario: Scenario dictionary

        Returns:
            Normalized scenario tensor of shape (4,)
        """
        features = [
            scenario['agent_count'],
            scenario['num_fires'],
            scenario['fire_spread_rate'],
            scenario['fire_discovery_delay']
        ]

        if self.scenario_stats is not None:
            means = self.scenario_stats['means']
            stds = self.scenario_stats['stds']
            features = [(f - m) / s for f, m, s in zip(features, means, stds)]

        return torch.tensor(features, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a pairwise sample.

        Returns:
            Dict with keys:
                - grid_a, grid_b: Grid tensors (4, H, W)
                - scenario_a, scenario_b: Scenario tensors (4,)
                - label: Binary label (1 if A > B, else 0)
                - confidence: Label confidence [0, 1]
                - score_a, score_b: Ground truth scores (for evaluation)
                - floor_plan_id_a, floor_plan_id_b: Floor plan IDs
        """
        pair = self.pairs[idx]

        # Encode grids
        grid_a = self._encode_grid(pair['floor_plan_id_a'], pair['config_a'])
        grid_b = self._encode_grid(pair['floor_plan_id_b'], pair['config_b'])

        # Normalize scenarios
        scenario_a = self._normalize_scenario(pair['scenario_a'])
        scenario_b = self._normalize_scenario(pair['scenario_b'])

        return {
            'grid_a': grid_a,
            'scenario_a': scenario_a,
            'grid_b': grid_b,
            'scenario_b': scenario_b,
            'label': torch.tensor(pair['label'], dtype=torch.long),
            'confidence': torch.tensor(pair['label_confidence'], dtype=torch.float32),
            'score_a': torch.tensor(pair['score_a'], dtype=torch.float32),
            'score_b': torch.tensor(pair['score_b'], dtype=torch.float32),
            'floor_plan_id_a': pair['floor_plan_id_a'],
            'floor_plan_id_b': pair['floor_plan_id_b'],
        }


class SingleConfigDataset(Dataset):
    """
    Dataset for evaluating single configurations (for per-plan metrics).

    Loads unique configurations from simulation results for computing
    Kendall Tau, Spearman, NDCG per floor plan.
    """

    def __init__(
        self,
        simulation_results_file: str,
        floor_plans_dir: str,
        floor_plan_ids: Optional[Set[int]] = None,
        target_size: Tuple[int, int] = (96, 128),
        scenario_stats: Optional[Dict] = None,
        max_configs: Optional[int] = None
    ):
        """
        Initialize the dataset.

        Args:
            simulation_results_file: Path to simulation_results.jsonl
            floor_plans_dir: Path to floor_plans/ directory
            floor_plan_ids: Set of floor plan IDs to include
            target_size: (H, W) target size for grid padding
            scenario_stats: Normalization stats from training
            max_configs: Maximum configs to load (for debugging)
        """
        self.target_size = target_size
        self.scenario_stats = scenario_stats

        # Load records
        self.records = self._load_records(
            simulation_results_file, floor_plan_ids, max_configs
        )

        # Get unique floor plan IDs
        fp_ids = set(r['floor_plan_id'] for r in self.records)

        # Load floor plans
        self.floor_plans = self._load_floor_plans(floor_plans_dir, fp_ids)

    def _load_records(
        self,
        results_file: str,
        floor_plan_ids: Optional[Set[int]],
        max_configs: Optional[int]
    ) -> List[Dict]:
        """Load simulation records from JSONL."""
        records = []
        with open(results_file, 'r') as f:
            for line in f:
                record = json.loads(line)
                if floor_plan_ids is None or record['floor_plan_id'] in floor_plan_ids:
                    records.append(record)
                    if max_configs is not None and len(records) >= max_configs:
                        break
        return records

    def _load_floor_plans(
        self,
        floor_plans_dir: str,
        floor_plan_ids: Set[int]
    ) -> Dict[int, Dict]:
        """Load floor plan grids into memory."""
        plans = {}
        floor_plans_path = Path(floor_plans_dir)

        for plan_id in floor_plan_ids:
            plan_file = floor_plans_path / f"plan_{plan_id:05d}.npz"
            if plan_file.exists():
                data = np.load(plan_file, allow_pickle=True)
                plans[plan_id] = {
                    'grid': data['grid'],
                    'door_positions': data['door_positions'],
                    'exit_positions': data['exit_positions']
                }

        return plans

    def _encode_grid(self, floor_plan_id: int, config: Dict) -> torch.Tensor:
        """Create 4-channel tensor from grid and config."""
        floor_plan = self.floor_plans[floor_plan_id]
        grid = floor_plan['grid']
        H, W = grid.shape
        tH, tW = self.target_size

        encoded = np.zeros((4, tH, tW), dtype=np.float32)
        encoded[0, :H, :W] = (grid == -2).astype(np.float32)
        encoded[0, H:, :] = 1.0
        encoded[0, :, W:] = 1.0
        encoded[1, :H, :W] = (grid == 0).astype(np.float32)

        for item in config.get('door_config', []):
            x, y = parse_position(item['position'])
            if 0 <= y < tH and 0 <= x < tW:
                channel = 3 if item['type'] == 'exit' else 2
                encoded[channel, y, x] = 1.0

        return torch.from_numpy(encoded)

    def _normalize_scenario(self, scenario: Dict) -> torch.Tensor:
        """Normalize scenario parameters."""
        features = [
            scenario['agent_count'],
            scenario['num_fires'],
            scenario['fire_spread_rate'],
            scenario['fire_discovery_delay']
        ]

        if self.scenario_stats is not None:
            means = self.scenario_stats['means']
            stds = self.scenario_stats['stds']
            features = [(f - m) / s for f, m, s in zip(features, means, stds)]

        return torch.tensor(features, dtype=torch.float32)

    def _compute_score(self, record: Dict) -> float:
        """Compute ground truth ranking score."""
        # Same formula as pair_constructor.py
        survival = record['survival_rate']
        steps = record['steps']
        fire_damage = record['avg_fire_damage']
        return (0.7 * survival
                - 0.15 * ((steps - 11) / (70 - 11))
                - 0.15 * ((fire_damage - 0.0167) / (3.0111 - 0.0167))
                + 0.118) / 0.818

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]

        grid = self._encode_grid(record['floor_plan_id'], record['config'])
        scenario = self._normalize_scenario(record['scenario'])
        ground_truth_score = self._compute_score(record)

        return {
            'grid': grid,
            'scenario': scenario,
            'floor_plan_id': record['floor_plan_id'],
            'ground_truth_score': ground_truth_score,
        }


def create_pairwise_dataloaders(
    config: RankingConfig,
    compute_stats: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test dataloaders for pairwise ranking.

    Normalization statistics are computed from TRAINING split only
    to prevent data leakage.

    Args:
        config: Model configuration
        compute_stats: Whether to compute normalization stats

    Returns:
        Tuple of (train_loader, val_loader, test_loader, stats_dict)
    """
    data_dir = Path(config.data_dir)
    train_pairs_file = str(data_dir / "train_pairs.jsonl")
    val_pairs_file = str(data_dir / "val_pairs.jsonl")
    test_pairs_file = str(data_dir / "test_pairs.jsonl")

    # Compute stats from TRAINING data only
    scenario_stats = None
    if compute_stats:
        if config.scenario_means is None:
            scenario_stats = compute_scenario_stats(train_pairs_file)
        else:
            scenario_stats = {
                'means': config.scenario_means,
                'stds': config.scenario_stds
            }

    # Create datasets (all use train stats)
    train_dataset = PairwiseDataset(
        pairs_file=train_pairs_file,
        floor_plans_dir=config.floor_plans_dir,
        target_size=config.target_grid_size,
        scenario_stats=scenario_stats,
        compute_stats=False
    )

    val_dataset = PairwiseDataset(
        pairs_file=val_pairs_file,
        floor_plans_dir=config.floor_plans_dir,
        target_size=config.target_grid_size,
        scenario_stats=scenario_stats,  # Use TRAIN stats
        compute_stats=False
    )

    test_dataset = PairwiseDataset(
        pairs_file=test_pairs_file,
        floor_plans_dir=config.floor_plans_dir,
        target_size=config.target_grid_size,
        scenario_stats=scenario_stats,  # Use TRAIN stats
        compute_stats=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader, {'scenario_stats': scenario_stats}
