"""
PyTorch Dataset for Fire Simulation data
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import ModelConfig


def parse_position(pos_str: str) -> Tuple[int, int]:
    """Parse position string 'x16y0' -> (x=16, y=0)."""
    match = re.match(r'x(\d+)y(\d+)', pos_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Invalid position format: {pos_str}")


def get_floor_plan_ids_from_pairs(pairs_file: str) -> Set[int]:
    """Extract unique floor plan IDs from a pairs JSONL file."""
    fp_ids = set()
    with open(pairs_file, 'r') as f:
        for line in f:
            pair = json.loads(line)
            fp_ids.add(pair['floor_plan_id_a'])
            fp_ids.add(pair['floor_plan_id_b'])
    return fp_ids


def compute_normalization_stats(
    simulation_results_file: str,
    floor_plan_ids: Optional[Set[int]] = None
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute mean and std for scenario and target variables.

    Args:
        simulation_results_file: Path to simulation_results.jsonl
        floor_plan_ids: Optional set of floor plan IDs to filter by

    Returns:
        Dict with 'scenario' and 'target' keys, each containing 'means' and 'stds'
    """
    scenario_values = {
        'agent_count': [],
        'num_fires': [],
        'fire_spread_rate': [],
        'fire_discovery_delay': []
    }
    target_values = {
        'survival_rate': [],
        'avg_evacuation_time': [],
        'steps': [],
        'avg_fire_damage': []
    }

    with open(simulation_results_file, 'r') as f:
        for line in f:
            record = json.loads(line)

            # Filter by floor plan ID if specified
            if floor_plan_ids is not None and record['floor_plan_id'] not in floor_plan_ids:
                continue

            # Scenario values
            scenario = record['scenario']
            scenario_values['agent_count'].append(scenario['agent_count'])
            scenario_values['num_fires'].append(scenario['num_fires'])
            scenario_values['fire_spread_rate'].append(scenario['fire_spread_rate'])
            scenario_values['fire_discovery_delay'].append(scenario['fire_discovery_delay'])

            # Target values
            target_values['survival_rate'].append(record['survival_rate'])
            target_values['avg_evacuation_time'].append(record['avg_evacuation_time'])
            target_values['steps'].append(record['steps'])
            target_values['avg_fire_damage'].append(record['avg_fire_damage'])

    # Compute means and stds
    scenario_means = [np.mean(scenario_values[k]) for k in ['agent_count', 'num_fires', 'fire_spread_rate', 'fire_discovery_delay']]
    scenario_stds = [np.std(scenario_values[k]) for k in ['agent_count', 'num_fires', 'fire_spread_rate', 'fire_discovery_delay']]

    target_means = [np.mean(target_values[k]) for k in ['survival_rate', 'avg_evacuation_time', 'steps', 'avg_fire_damage']]
    target_stds = [np.std(target_values[k]) for k in ['survival_rate', 'avg_evacuation_time', 'steps', 'avg_fire_damage']]

    # Prevent division by zero
    scenario_stds = [max(s, 1e-6) for s in scenario_stds]
    target_stds = [max(s, 1e-6) for s in target_stds]

    return {
        'scenario': {'means': scenario_means, 'stds': scenario_stds},
        'target': {'means': target_means, 'stds': target_stds}
    }


class FireSimulationDataset(Dataset):
    """
    PyTorch Dataset for fire simulation data.

    Loads floor plan grids and simulation results, encodes grids as 4-channel
    tensors, and normalizes scenario/target values.
    """

    def __init__(
        self,
        simulation_results_file: str,
        floor_plans_dir: str,
        floor_plan_ids: Optional[Set[int]] = None,
        target_size: Tuple[int, int] = (96, 128),
        scenario_stats: Optional[Dict] = None,
        target_stats: Optional[Dict] = None,
        max_plans: Optional[int] = None
    ):
        """
        Args:
            simulation_results_file: Path to simulation_results.jsonl
            floor_plans_dir: Path to floor_plans/ directory
            floor_plan_ids: Set of floor plan IDs to include (for train/val/test split)
            target_size: (H, W) target size for grid padding
            scenario_stats: Dict with 'means' and 'stds' for scenario normalization
            target_stats: Dict with 'means' and 'stds' for target normalization
            max_plans: Limit number of floor plans to load (None = all)
        """
        self.target_size = target_size
        self.scenario_stats = scenario_stats
        self.target_stats = target_stats

        # Load floor plans into memory
        self.floor_plans = self._load_floor_plans(floor_plans_dir, floor_plan_ids, max_plans)

        # Load simulation records
        self.records = self._load_records(simulation_results_file, set(self.floor_plans.keys()))

    def _load_floor_plans(
        self,
        floor_plans_dir: str,
        floor_plan_ids: Optional[Set[int]],
        max_plans: Optional[int]
    ) -> Dict[int, Dict]:
        """Load floor plan grids into memory."""
        plans = {}
        floor_plans_path = Path(floor_plans_dir)

        # Get list of available plan files
        plan_files = sorted(floor_plans_path.glob("plan_*.npz"))

        for plan_file in plan_files:
            # Extract plan ID from filename (e.g., plan_00123.npz -> 123)
            plan_id = int(plan_file.stem.split('_')[1])

            # Filter by floor plan IDs if specified
            if floor_plan_ids is not None and plan_id not in floor_plan_ids:
                continue

            # Check max_plans limit
            if max_plans is not None and len(plans) >= max_plans:
                break

            # Load plan data
            data = np.load(plan_file, allow_pickle=True)
            plans[plan_id] = {
                'grid': data['grid'],
                'door_positions': data['door_positions'],
                'exit_positions': data['exit_positions']
            }

        return plans

    def _load_records(
        self,
        simulation_results_file: str,
        valid_floor_plan_ids: Set[int]
    ) -> List[Dict]:
        """Load simulation records, filtering by floor plan IDs."""
        records = []

        with open(simulation_results_file, 'r') as f:
            for line in f:
                record = json.loads(line)
                if record['floor_plan_id'] in valid_floor_plan_ids:
                    records.append(record)

        return records

    def _encode_grid(self, grid: np.ndarray, config: Dict) -> torch.Tensor:
        """
        Create 4-channel tensor from grid and door/exit config.

        Channels:
            0: Wall mask (grid == -2)
            1: Passable mask (grid == 0)
            2: Door positions
            3: Exit positions
        """
        H, W = grid.shape
        tH, tW = self.target_size

        # Clip grid dimensions to target size if larger
        copyH = min(H, tH)
        copyW = min(W, tW)

        # Create 4-channel encoding
        encoded = np.zeros((4, tH, tW), dtype=np.float32)

        # Channel 0: Wall mask (including padding)
        encoded[0, :copyH, :copyW] = (grid[:copyH, :copyW] == -2).astype(np.float32)
        # Padding treated as walls
        encoded[0, copyH:, :] = 1.0
        encoded[0, :, copyW:] = 1.0

        # Channel 1: Passable mask
        encoded[1, :copyH, :copyW] = (grid[:copyH, :copyW] == 0).astype(np.float32)

        # Channels 2 & 3: Doors and exits from config
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

    def _normalize_targets(self, record: Dict) -> torch.Tensor:
        """Normalize target values."""
        targets = [
            record['survival_rate'],
            record['avg_evacuation_time'],
            record['steps'],
            record['avg_fire_damage']
        ]

        if self.target_stats is not None:
            means = self.target_stats['means']
            stds = self.target_stats['stds']
            targets = [(t - m) / s for t, m, s in zip(targets, means, stds)]

        return torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]

        # Get floor plan
        floor_plan = self.floor_plans[record['floor_plan_id']]

        # Encode grid with door/exit positions
        grid_tensor = self._encode_grid(floor_plan['grid'], record['config'])

        # Normalize scenario and targets
        scenario_tensor = self._normalize_scenario(record['scenario'])
        targets_tensor = self._normalize_targets(record)

        return {
            'grid': grid_tensor,           # Shape: (4, H, W)
            'scenario': scenario_tensor,   # Shape: (4,)
            'targets': targets_tensor      # Shape: (4,)
        }


def create_dataloaders(
    config: ModelConfig,
    compute_stats: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Model configuration
        compute_stats: Whether to compute normalization stats from training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader, stats_dict)
    """
    data_dir = Path(config.data_dir)

    # Get floor plan IDs for each split
    train_fp_ids = get_floor_plan_ids_from_pairs(str(data_dir / "train_pairs.jsonl"))
    val_fp_ids = get_floor_plan_ids_from_pairs(str(data_dir / "val_pairs.jsonl"))
    test_fp_ids = get_floor_plan_ids_from_pairs(str(data_dir / "test_pairs.jsonl"))

    # Compute normalization stats from training data
    stats = None
    scenario_stats = None
    target_stats = None

    if compute_stats:
        if config.scenario_means is None or config.target_means is None:
            stats = compute_normalization_stats(
                str(data_dir / "simulation_results.jsonl"),
                train_fp_ids
            )
            scenario_stats = stats['scenario']
            target_stats = stats['target']
        else:
            scenario_stats = {'means': config.scenario_means, 'stds': config.scenario_stds}
            target_stats = {'means': config.target_means, 'stds': config.target_stds}

    # Create datasets
    train_dataset = FireSimulationDataset(
        simulation_results_file=str(data_dir / "simulation_results.jsonl"),
        floor_plans_dir=config.floor_plans_dir,
        floor_plan_ids=train_fp_ids,
        target_size=config.target_grid_size,
        scenario_stats=scenario_stats,
        target_stats=target_stats,
        max_plans=config.max_plans
    )

    val_dataset = FireSimulationDataset(
        simulation_results_file=str(data_dir / "simulation_results.jsonl"),
        floor_plans_dir=config.floor_plans_dir,
        floor_plan_ids=val_fp_ids,
        target_size=config.target_grid_size,
        scenario_stats=scenario_stats,
        target_stats=target_stats,
        max_plans=config.max_plans
    )

    test_dataset = FireSimulationDataset(
        simulation_results_file=str(data_dir / "simulation_results.jsonl"),
        floor_plans_dir=config.floor_plans_dir,
        floor_plan_ids=test_fp_ids,
        target_size=config.target_grid_size,
        scenario_stats=scenario_stats,
        target_stats=target_stats,
        max_plans=config.max_plans
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

    return train_loader, val_loader, test_loader, {
        'scenario_stats': scenario_stats,
        'target_stats': target_stats
    }
