"""
Pairwise Dataset V2 for Enhanced Ranking Training

Extends V1 with:
- Hard negative mining support (precomputed hardness scores)
- Auxiliary task labels (survival_rate, steps, avg_fire_damage)
- get_hard_indices() for sampler integration

Key Design:
    - Normalization stats computed from TRAIN split only (prevents leakage)
    - Efficient caching of floor plans in memory
    - Ground truth scores included for per-plan ranking evaluation
    - Hardness = |score_a - score_b|, lower = harder to distinguish
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import RankingV2Config


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


class PairwiseDatasetV2(Dataset):
    """
    PyTorch Dataset V2 for pairwise ranking training.

    Enhanced features over V1:
    - Precomputed hardness scores for hard negative mining
    - Support for auxiliary task labels
    - get_hard_indices() method for sampler integration

    Attributes:
        pairs: List of pairwise label dictionaries
        floor_plans: Dict mapping floor_plan_id to grid data
        scenario_stats: Normalization statistics for scenarios
        augment: Whether to apply random shift augmentation (train only)
        augment_rotate90: Whether to apply random 90-degree rotation (train only)
        score_diffs: Precomputed |score_a - score_b| for each pair
        hardness_indices: Indices sorted by hardness (hardest first)
        include_auxiliary: Whether to include auxiliary task labels
        auxiliary_tasks: List of auxiliary tasks to include
    """

    def __init__(
        self,
        pairs_file: str,
        floor_plans_dir: str,
        target_size: Tuple[int, int] = (96, 128),
        scenario_stats: Optional[Dict] = None,
        compute_stats: bool = False,
        max_pairs: Optional[int] = None,
        augment: bool = False,
        augment_rotate90: bool = False,
        include_auxiliary: bool = True,
        auxiliary_tasks: Optional[List[str]] = None
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
            augment: If True, apply random shift augmentation (for training)
            augment_rotate90: If True, apply random 90-degree rotation (for training)
            include_auxiliary: If True, include auxiliary task labels
            auxiliary_tasks: List of auxiliary tasks (e.g., ["survival_rate", "steps"])
        """
        self.target_size = target_size
        self.augment = augment
        self.augment_rotate90 = augment_rotate90
        self.include_auxiliary = include_auxiliary
        self.auxiliary_tasks = auxiliary_tasks or ["survival_rate", "steps"]

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

        # NEW: Precompute hardness for hard negative mining
        self._precompute_hardness()

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

    def _precompute_hardness(self):
        """
        Precompute hardness scores for hard negative mining.

        Hardness = |score_a - score_b|
        Lower score difference = harder to distinguish = more valuable for training
        """
        self.score_diffs = []
        for pair in self.pairs:
            diff = abs(pair['score_a'] - pair['score_b'])
            self.score_diffs.append(diff)
        self.score_diffs = np.array(self.score_diffs)

        # Create hardness index (sorted by difficulty, hardest first)
        # Lower score_diff = harder, so we sort ascending
        self.hardness_indices = np.argsort(self.score_diffs)

    def get_hard_indices(self, threshold: float) -> np.ndarray:
        """
        Return indices of pairs with |score_diff| < threshold.

        These are "hard" pairs that are difficult to distinguish.

        Args:
            threshold: Maximum score difference to be considered "hard"

        Returns:
            Array of pair indices
        """
        return np.where(self.score_diffs < threshold)[0]

    def get_hardest_n(self, n: int) -> np.ndarray:
        """
        Return indices of the N hardest pairs.

        Args:
            n: Number of hardest pairs to return

        Returns:
            Array of pair indices (sorted by hardness)
        """
        return self.hardness_indices[:n]

    def get_hardness_percentile(self, percentile: float) -> float:
        """
        Get the score difference at a given percentile.

        Args:
            percentile: Percentile (0-100)

        Returns:
            Score difference threshold at that percentile
        """
        return np.percentile(self.score_diffs, percentile)

    def _encode_grid(self, floor_plan_id: int, config: Dict) -> torch.Tensor:
        """
        Create 5-channel tensor from grid and door/exit config.

        Channels:
            0: Wall mask (grid == -2)
            1: Passable mask (grid == 0)
            2: Door positions
            3: Exit positions
            4: Valid mask (1.0 for real grid, 0.0 for padding)

        Args:
            floor_plan_id: Floor plan ID
            config: Door/exit configuration dict

        Returns:
            Grid tensor of shape (5, H, W)
        """
        floor_plan = self.floor_plans[floor_plan_id]
        grid = floor_plan['grid']
        H, W = grid.shape
        tH, tW = self.target_size

        # Create 5-channel encoding with -1.0 padding
        encoded = np.full((5, tH, tW), -1.0, dtype=np.float32)

        # Handle cases where grid is larger than target size - clip
        H_copy = min(H, tH)
        W_copy = min(W, tW)

        # Channel 0: Wall mask (real grid only, padding is -1.0)
        encoded[0, :H_copy, :W_copy] = (grid[:H_copy, :W_copy] == -2).astype(np.float32)

        # Channel 1: Passable mask (real grid only, padding is -1.0)
        encoded[1, :H_copy, :W_copy] = (grid[:H_copy, :W_copy] == 0).astype(np.float32)

        # Channels 2 & 3: Initialize real grid area to 0.0, then place doors/exits
        encoded[2, :H_copy, :W_copy] = 0.0
        encoded[3, :H_copy, :W_copy] = 0.0
        for item in config.get('door_config', []):
            x, y = parse_position(item['position'])
            if 0 <= y < tH and 0 <= x < tW:
                channel = 3 if item['type'] == 'exit' else 2
                encoded[channel, y, x] = 1.0

        # Channel 4: Valid mask (1.0 for real grid, 0.0 for padding)
        encoded[4, :H_copy, :W_copy] = 1.0
        encoded[4, H_copy:, :] = 0.0
        encoded[4, :, W_copy:] = 0.0

        return torch.from_numpy(encoded)

    def _random_shift_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Apply random shift augmentation to grid (right and down only).

        Since floor plans are positioned at the top-left corner, we can only
        shift right and down without losing content. The shift amount is
        computed based on the valid mask (channel 4) to ensure no content
        goes out of bounds.

        Args:
            grid: Grid tensor of shape (5, H, W)

        Returns:
            Shifted grid tensor with padding filled appropriately
        """
        _, tH, tW = grid.shape
        valid_mask = grid[4]  # Channel 4 is the valid mask

        # Find the bounding box of valid content
        # valid_mask is 1.0 for real grid, 0.0 for padding
        valid_rows = (valid_mask.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
        valid_cols = (valid_mask.sum(dim=0) > 0).nonzero(as_tuple=True)[0]

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            return grid  # No valid content, return as-is

        # Content spans from (0, 0) to (max_row, max_col) since it's top-left aligned
        max_row = valid_rows[-1].item()
        max_col = valid_cols[-1].item()

        # Calculate maximum possible shift
        max_shift_down = tH - 1 - max_row
        max_shift_right = tW - 1 - max_col

        if max_shift_down <= 0 and max_shift_right <= 0:
            return grid  # No room to shift

        # Random shift amounts (can be 0)
        shift_down = np.random.randint(0, max_shift_down + 1) if max_shift_down > 0 else 0
        shift_right = np.random.randint(0, max_shift_right + 1) if max_shift_right > 0 else 0

        if shift_down == 0 and shift_right == 0:
            return grid  # No shift needed

        # Create new grid with padding values
        # Channels 0-3: -1.0 for padding, Channel 4: 0.0 for padding
        shifted = torch.full_like(grid, -1.0)
        shifted[4] = 0.0  # Valid mask padding is 0.0

        # Copy content to new position
        src_h = tH - shift_down
        src_w = tW - shift_right
        shifted[:, shift_down:, shift_right:] = grid[:, :src_h, :src_w]

        return shifted

    def _random_rotate_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Apply random 90-degree rotation augmentation to grid.

        Randomly rotates by 0°, 90°, 180°, or 270° and re-pads to target size.
        After rotation, the content is repositioned to the top-left corner
        and padded appropriately.

        Args:
            grid: Grid tensor of shape (5, H, W)

        Returns:
            Rotated grid tensor of shape (5, tH, tW) with correct padding
        """
        # Random rotation: 0, 1, 2, or 3 times 90 degrees (counter-clockwise)
        k = np.random.randint(0, 4)
        if k == 0:
            return grid  # No rotation

        _, tH, tW = grid.shape

        # Apply rotation (dims 1 and 2 are H and W)
        rotated = torch.rot90(grid, k=k, dims=(1, 2))

        # After rotation, dimensions may have swapped
        _, rH, rW = rotated.shape

        # If dimensions match target, we're done
        if rH == tH and rW == tW:
            return rotated

        # Otherwise, we need to re-pad/crop to target size
        # Create new grid with padding values
        result = torch.full((5, tH, tW), -1.0, dtype=grid.dtype)
        result[4] = 0.0  # Valid mask padding is 0.0

        # Find the valid content region in rotated grid
        valid_mask = rotated[4]
        valid_rows = (valid_mask > 0.5).any(dim=1).nonzero(as_tuple=True)[0]
        valid_cols = (valid_mask > 0.5).any(dim=0).nonzero(as_tuple=True)[0]

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            return result  # No valid content

        # Get bounding box of valid content
        min_row, max_row = valid_rows[0].item(), valid_rows[-1].item()
        min_col, max_col = valid_cols[0].item(), valid_cols[-1].item()
        content_h = max_row - min_row + 1
        content_w = max_col - min_col + 1

        # Crop content if it doesn't fit in target size
        copy_h = min(content_h, tH)
        copy_w = min(content_w, tW)

        # Copy valid content to top-left of result (same pattern as original encoding)
        result[:, :copy_h, :copy_w] = rotated[
            :,
            min_row:min_row + copy_h,
            min_col:min_col + copy_w
        ]

        return result

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

    def _get_auxiliary_from_pair(self, pair: Dict, suffix: str) -> Dict[str, float]:
        """
        Extract auxiliary task targets from pair data.

        The auxiliary labels may be stored directly in pair data
        or need to be computed from simulation results.

        Args:
            pair: Pair dictionary
            suffix: 'a' or 'b'

        Returns:
            Dict mapping task name to target value
        """
        aux = {}

        for task in self.auxiliary_tasks:
            key = f'{task}_{suffix}'
            if key in pair:
                # Direct storage in pair data
                aux[task] = pair[key]
            elif task == 'survival_rate':
                # Compute from score if not available
                # Note: This is an approximation; real data should have this
                aux[task] = pair.get(f'survival_rate_{suffix}', pair[f'score_{suffix}'])
            elif task == 'steps':
                aux[task] = pair.get(f'steps_{suffix}', 50.0)  # Default
            elif task == 'avg_fire_damage':
                aux[task] = pair.get(f'avg_fire_damage_{suffix}', 0.5)  # Default

        return aux

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a pairwise sample.

        Returns:
            Dict with keys:
                - grid_a, grid_b: Grid tensors (5, H, W)
                - scenario_a, scenario_b: Scenario tensors (4,)
                - label: Binary label (1 if A > B, else 0)
                - confidence: Label confidence [0, 1]
                - score_a, score_b: Ground truth scores (for evaluation)
                - score_diff: |score_a - score_b| (for mining analysis)
                - floor_plan_id_a, floor_plan_id_b: Floor plan IDs
                - [auxiliary task labels if include_auxiliary=True]
        """
        pair = self.pairs[idx]

        # Encode grids
        grid_a = self._encode_grid(pair['floor_plan_id_a'], pair['config_a'])
        grid_b = self._encode_grid(pair['floor_plan_id_b'], pair['config_b'])

        # Apply random 90-degree rotation augmentation (training only, independent for each grid)
        if self.augment_rotate90:
            grid_a = self._random_rotate_grid(grid_a)
            grid_b = self._random_rotate_grid(grid_b)

        # Apply random shift augmentation (training only, independent for each grid)
        if self.augment:
            grid_a = self._random_shift_grid(grid_a)
            grid_b = self._random_shift_grid(grid_b)

        # Normalize scenarios
        scenario_a = self._normalize_scenario(pair['scenario_a'])
        scenario_b = self._normalize_scenario(pair['scenario_b'])

        result = {
            'grid_a': grid_a,
            'scenario_a': scenario_a,
            'grid_b': grid_b,
            'scenario_b': scenario_b,
            'label': torch.tensor(pair['label'], dtype=torch.long),
            'confidence': torch.tensor(pair['label_confidence'], dtype=torch.float32),
            'score_a': torch.tensor(pair['score_a'], dtype=torch.float32),
            'score_b': torch.tensor(pair['score_b'], dtype=torch.float32),
            'score_diff': torch.tensor(self.score_diffs[idx], dtype=torch.float32),
            'floor_plan_id_a': pair['floor_plan_id_a'],
            'floor_plan_id_b': pair['floor_plan_id_b'],
        }

        # Add auxiliary task labels
        if self.include_auxiliary:
            aux_a = self._get_auxiliary_from_pair(pair, 'a')
            aux_b = self._get_auxiliary_from_pair(pair, 'b')

            for task in self.auxiliary_tasks:
                if task in aux_a:
                    result[f'{task}_a'] = torch.tensor(aux_a[task], dtype=torch.float32)
                if task in aux_b:
                    result[f'{task}_b'] = torch.tensor(aux_b[task], dtype=torch.float32)

        return result


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
        """Create 5-channel tensor from grid and config."""
        floor_plan = self.floor_plans[floor_plan_id]
        grid = floor_plan['grid']
        H, W = grid.shape
        tH, tW = self.target_size

        # Create 5-channel encoding with -1.0 padding
        encoded = np.full((5, tH, tW), -1.0, dtype=np.float32)

        # Handle cases where grid is larger than target size - clip
        H_copy = min(H, tH)
        W_copy = min(W, tW)

        # Channel 0: Wall mask (real grid only, padding is -1.0)
        encoded[0, :H_copy, :W_copy] = (grid[:H_copy, :W_copy] == -2).astype(np.float32)

        # Channel 1: Passable mask (real grid only, padding is -1.0)
        encoded[1, :H_copy, :W_copy] = (grid[:H_copy, :W_copy] == 0).astype(np.float32)

        # Channels 2 & 3: Initialize real grid area to 0.0, then place doors/exits
        encoded[2, :H_copy, :W_copy] = 0.0
        encoded[3, :H_copy, :W_copy] = 0.0
        for item in config.get('door_config', []):
            x, y = parse_position(item['position'])
            if 0 <= y < tH and 0 <= x < tW:
                channel = 3 if item['type'] == 'exit' else 2
                encoded[channel, y, x] = 1.0

        # Channel 4: Valid mask (1.0 for real grid, 0.0 for padding)
        encoded[4, :H_copy, :W_copy] = 1.0
        encoded[4, H_copy:, :] = 0.0
        encoded[4, :, W_copy:] = 0.0

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
            # Raw metrics for visualization and auxiliary evaluation
            'survival_rate': record['survival_rate'],
            'steps': record['steps'],
            'avg_fire_damage': record['avg_fire_damage'],
        }


def create_pairwise_dataloaders(
    config: RankingV2Config,
    compute_stats: bool = True,
    return_datasets: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test dataloaders for pairwise ranking V2.

    Normalization statistics are computed from TRAINING split only
    to prevent data leakage.

    Args:
        config: Model configuration
        compute_stats: Whether to compute normalization stats
        return_datasets: If True, also return dataset objects (for sampler access)

    Returns:
        Tuple of (train_loader, val_loader, test_loader, stats_dict)
        If return_datasets=True, stats_dict includes 'train_dataset'
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
    train_dataset = PairwiseDatasetV2(
        pairs_file=train_pairs_file,
        floor_plans_dir=config.floor_plans_dir,
        target_size=config.target_grid_size,
        scenario_stats=scenario_stats,
        compute_stats=False,
        augment=config.augment_shift,
        augment_rotate90=config.augment_rotate90,
        include_auxiliary=len(config.auxiliary_tasks) > 0,
        auxiliary_tasks=config.auxiliary_tasks
    )

    val_dataset = PairwiseDatasetV2(
        pairs_file=val_pairs_file,
        floor_plans_dir=config.floor_plans_dir,
        target_size=config.target_grid_size,
        scenario_stats=scenario_stats,
        compute_stats=False,
        include_auxiliary=len(config.auxiliary_tasks) > 0,
        auxiliary_tasks=config.auxiliary_tasks
    )

    test_dataset = PairwiseDatasetV2(
        pairs_file=test_pairs_file,
        floor_plans_dir=config.floor_plans_dir,
        target_size=config.target_grid_size,
        scenario_stats=scenario_stats,
        compute_stats=False,
        include_auxiliary=len(config.auxiliary_tasks) > 0,
        auxiliary_tasks=config.auxiliary_tasks
    )

    # Create dataloaders
    # Note: When using HardNegativeBatchSampler, set shuffle=False
    # and pass the custom batch_sampler instead
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # Will be overridden if using custom sampler
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0
    )

    stats_dict = {'scenario_stats': scenario_stats}
    if return_datasets:
        stats_dict['train_dataset'] = train_dataset
        stats_dict['val_dataset'] = val_dataset
        stats_dict['test_dataset'] = test_dataset

    return train_loader, val_loader, test_loader, stats_dict


def create_train_loader_with_sampler(
    dataset: PairwiseDatasetV2,
    batch_sampler,
    config: RankingV2Config
) -> DataLoader:
    """
    Create train dataloader with custom batch sampler (for hard negative mining).

    Args:
        dataset: Training dataset
        batch_sampler: Custom batch sampler (e.g., HardNegativeBatchSampler)
        config: Configuration

    Returns:
        DataLoader with custom batch sampling
    """
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0
    )
