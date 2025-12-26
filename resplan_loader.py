"""
ResPlan Loader Module for Training Data Generation V4

Provides utilities to load ResPlan floor plans and convert them to
simulation-compatible grids with door position extraction.
"""

import os
import sys
import pickle
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

# Add ResPlan directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ResPlan'))

from resplan_utils import (
    normalize_keys,
    calculate_grid_size_for_plan,
    calculate_door_positions_in_grid,
    get_structural_plan,
    structural_plan_to_multilinestring,
    multilinestring_to_grid
)

logger = logging.getLogger(__name__)


@dataclass
class ResPlanFloorPlan:
    """Container for a converted ResPlan floor plan"""
    plan_index: int                              # Index in ResPlan.pkl
    grid: np.ndarray                             # Rasterized grid (-2=wall, 0=passable)
    door_positions: List[Tuple[int, int]]        # Internal door cell coords (x, y)
    exit_positions: List[Tuple[int, int]]        # front_door cell coords (x, y)
    resplan_door_config: List[Dict[str, Any]]    # Door config in CandidateGenerator format
    metadata: Dict[str, Any] = field(default_factory=dict)  # area, unitType, etc.


def load_resplan_dataset(pkl_path: str) -> List[Dict]:
    """
    Load ResPlan dataset from pickle file.

    Args:
        pkl_path: Path to ResPlan.pkl file

    Returns:
        List of plan dictionaries

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data format is invalid
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"ResPlan file not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        plans = pickle.load(f)

    if not isinstance(plans, (list, tuple)) or len(plans) == 0:
        raise ValueError("Invalid ResPlan data format. Expected non-empty list of plans.")

    logger.info(f"Loaded {len(plans)} plans from {pkl_path}")
    return plans


def normalize_plan(plan: Dict) -> Dict:
    """
    Normalize plan keys to handle typos in the dataset.

    Args:
        plan: Raw plan dictionary

    Returns:
        Plan with normalized keys
    """
    try:
        normalize_keys(plan)
    except Exception as e:
        logger.warning(f"normalize_keys failed: {e}")
    return plan


def plan_to_grid(plan: Dict, cell_size_m: float = 0.3) -> Optional[np.ndarray]:
    """
    Convert a ResPlan plan to a rasterized grid.

    The grid has:
    - 0 for passable cells (interior space)
    - -2 for walls, obstacles, and cells outside the building

    Note: Door cells are also marked as -2 (walls) by default.

    Args:
        plan: Normalized plan dictionary
        cell_size_m: Cell size in meters

    Returns:
        2D numpy array grid, or None if conversion fails
    """
    try:
        # Get structural plan (walls, doors, windows)
        structural = get_structural_plan(plan)

        # Convert to MultiLineString
        multilines = structural_plan_to_multilinestring(plan)

        if multilines is None or (hasattr(multilines, 'is_empty') and multilines.is_empty):
            logger.warning("structural_plan_to_multilinestring returned empty result")
            return None

        # Rasterize to grid
        grid = multilinestring_to_grid(
            multilines,
            plan,
            cell_size_m=cell_size_m,
            structural_value=-2,
            empty_value=0,
            add_border=True,
            mark_outside=True,
            remove_spurs=True,
            max_spur_length=2,
            filter_branch_endpoints=True,
            max_branch_length=2.0
        )

        return grid

    except Exception as e:
        logger.warning(f"plan_to_grid failed: {e}")
        return None


def extract_door_cells(plan: Dict, grid: np.ndarray,
                       cell_size_m: float = 0.3) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Extract door cell coordinates from a plan.

    Args:
        plan: Normalized plan dictionary
        grid: Rasterized grid (for bounds checking)
        cell_size_m: Cell size in meters

    Returns:
        Tuple of (door_positions, exit_positions) where each is a list of (x, y) tuples
    """
    try:
        positions = calculate_door_positions_in_grid(plan, cell_size_m)
    except Exception as e:
        logger.warning(f"calculate_door_positions_in_grid failed: {e}")
        return [], []

    door_cells = []
    for gx, gy in positions.get('door', []):
        # Round to integer cell coords
        cx, cy = int(round(gx)), int(round(gy))
        # Verify cell is within grid bounds
        if 0 <= cy < grid.shape[0] and 0 <= cx < grid.shape[1]:
            door_cells.append((cx, cy))

    exit_cells = []
    for gx, gy in positions.get('front_door', []):
        cx, cy = int(round(gx)), int(round(gy))
        if 0 <= cy < grid.shape[0] and 0 <= cx < grid.shape[1]:
            exit_cells.append((cx, cy))

    return door_cells, exit_cells


def get_resplan_door_config(door_positions: List[Tuple[int, int]],
                            exit_positions: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """
    Convert ResPlan door positions to CandidateGenerator format.

    Args:
        door_positions: List of (x, y) tuples for internal doors
        exit_positions: List of (x, y) tuples for exits

    Returns:
        List of door config dicts compatible with CandidateGenerator
    """
    config = []

    # Exits from front_door
    for i, (x, y) in enumerate(exit_positions):
        config.append({
            'id': f'e{i+1}',
            'position': f'x{x}y{y}',
            'type': 'exit'
        })

    # Internal doors
    for i, (x, y) in enumerate(door_positions):
        config.append({
            'id': f'd{i+1}',
            'position': f'x{x}y{y}',
            'type': 'door'
        })

    return config


def apply_door_config(base_grid: np.ndarray,
                      door_config: List[Dict[str, Any]]) -> np.ndarray:
    """
    Apply a door configuration to create simulation-ready grid.

    Takes a base grid where all doors are walls (-2) and sets
    the specified door cells to passable (0).

    Args:
        base_grid: Grid with no doors (all walls = -2)
        door_config: List of door dicts with 'position' as "x{col}y{row}"

    Returns:
        Grid with door cells set to 0 (passable)
    """
    grid = base_grid.copy()

    for door in door_config:
        pos_str = door.get('position', '')
        if not pos_str or 'x' not in pos_str or 'y' not in pos_str:
            continue

        try:
            parts = pos_str.split('y')
            x = int(parts[0][1:])  # col
            y = int(parts[1])      # row

            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                grid[y, x] = 0  # Open the door
        except (ValueError, IndexError) as e:
            logger.warning(f"Invalid door position format: {pos_str}")
            continue

    return grid


def extract_plan_metadata(plan: Dict) -> Dict[str, Any]:
    """
    Extract metadata from a plan dictionary.

    Args:
        plan: Plan dictionary

    Returns:
        Dictionary with extracted metadata
    """
    metadata = {}

    # Standard fields
    for key in ['area', 'net_area', 'wall_width', 'unitType']:
        if key in plan:
            metadata[key] = plan[key]

    # Count rooms by checking for common room keys
    room_keys = ['living', 'bedroom', 'bathroom', 'kitchen', 'balcony', 'dining']
    room_count = sum(1 for k in room_keys if k in plan and plan[k] is not None)
    metadata['room_count'] = room_count

    return metadata


def convert_plan(plan: Dict, plan_index: int,
                 cell_size_m: float = 0.3) -> Optional[ResPlanFloorPlan]:
    """
    Convert a single ResPlan plan to a ResPlanFloorPlan object.

    Args:
        plan: Raw plan dictionary
        plan_index: Index in the original dataset
        cell_size_m: Cell size in meters

    Returns:
        ResPlanFloorPlan object, or None if conversion fails
    """
    # Normalize keys
    plan = normalize_plan(plan)

    # Convert to grid
    grid = plan_to_grid(plan, cell_size_m)
    if grid is None:
        return None

    # Extract door positions
    door_positions, exit_positions = extract_door_cells(plan, grid, cell_size_m)

    # Create door config in CandidateGenerator format
    resplan_door_config = get_resplan_door_config(door_positions, exit_positions)

    # Extract metadata
    metadata = extract_plan_metadata(plan)
    metadata['grid_shape'] = grid.shape
    metadata['cell_size_m'] = cell_size_m

    return ResPlanFloorPlan(
        plan_index=plan_index,
        grid=grid,
        door_positions=door_positions,
        exit_positions=exit_positions,
        resplan_door_config=resplan_door_config,
        metadata=metadata
    )


class ResPlanLoader:
    """
    Loader class for ResPlan dataset with caching support.
    """

    def __init__(self, pkl_path: str, cell_size_m: float = 0.3):
        """
        Initialize the loader.

        Args:
            pkl_path: Path to ResPlan.pkl file
            cell_size_m: Cell size in meters for grid conversion
        """
        self.pkl_path = pkl_path
        self.cell_size_m = cell_size_m
        self._plans: Optional[List[Dict]] = None
        self._converted: Dict[int, ResPlanFloorPlan] = {}

    def load_all(self) -> List[Dict]:
        """Load all plans from the pickle file."""
        if self._plans is None:
            self._plans = load_resplan_dataset(self.pkl_path)
        return self._plans

    def get_plan_count(self) -> int:
        """Get the total number of plans in the dataset."""
        return len(self.load_all())

    def convert_plan(self, plan: Dict, plan_index: int) -> Optional[ResPlanFloorPlan]:
        """
        Convert a single plan with caching.

        Args:
            plan: Plan dictionary
            plan_index: Index of the plan

        Returns:
            ResPlanFloorPlan object or None if conversion fails
        """
        if plan_index in self._converted:
            return self._converted[plan_index]

        result = convert_plan(plan, plan_index, self.cell_size_m)
        if result is not None:
            self._converted[plan_index] = result

        return result

    def convert_all(self, min_doors: int = 1,
                    progress_callback=None) -> List[ResPlanFloorPlan]:
        """
        Convert all valid plans from the dataset.

        Args:
            min_doors: Minimum number of internal doors required
            progress_callback: Optional callback(current, total) for progress reporting

        Returns:
            List of successfully converted ResPlanFloorPlan objects
        """
        plans = self.load_all()
        valid_plans = []

        for idx, plan in enumerate(plans):
            if progress_callback:
                progress_callback(idx, len(plans))

            try:
                fp = self.convert_plan(plan, idx)
                if fp is None:
                    logger.debug(f"Plan {idx}: conversion failed")
                    continue

                if len(fp.door_positions) < min_doors:
                    logger.debug(f"Plan {idx}: only {len(fp.door_positions)} doors (need {min_doors})")
                    continue

                valid_plans.append(fp)

            except Exception as e:
                logger.warning(f"Plan {idx}: unexpected error: {e}")
                continue

        logger.info(f"Converted {len(valid_plans)}/{len(plans)} valid plans")
        return valid_plans

    def save_converted_plans(self, output_dir: str,
                             plans: List[ResPlanFloorPlan]) -> None:
        """
        Save converted plans to NPZ files.

        Args:
            output_dir: Output directory
            plans: List of ResPlanFloorPlan objects to save
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, fp in enumerate(plans):
            filepath = os.path.join(output_dir, f'plan_{i:05d}.npz')
            np.savez_compressed(
                filepath,
                grid=fp.grid,
                door_positions=np.array(fp.door_positions),
                exit_positions=np.array(fp.exit_positions),
                plan_index=fp.plan_index,
                metadata=fp.metadata
            )

        logger.info(f"Saved {len(plans)} plans to {output_dir}")


if __name__ == '__main__':
    # Test the loader
    logging.basicConfig(level=logging.INFO)

    pkl_path = 'ResPlan/ResPlan.pkl'
    if os.path.exists(pkl_path):
        loader = ResPlanLoader(pkl_path)
        plans = loader.convert_all(min_doors=1)

        print(f"\nConverted {len(plans)} valid plans")

        if plans:
            sample = plans[0]
            print(f"\nSample plan {sample.plan_index}:")
            print(f"  Grid shape: {sample.grid.shape}")
            print(f"  Door positions: {sample.door_positions}")
            print(f"  Exit positions: {sample.exit_positions}")
            print(f"  ResPlan door config: {sample.resplan_door_config}")
            print(f"  Metadata: {sample.metadata}")
    else:
        print(f"ResPlan file not found: {pkl_path}")
