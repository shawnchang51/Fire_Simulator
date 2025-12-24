"""
Convert ResPlan floor plans to NPZ format for fire evacuation simulation.

Output NPZ structure:
    - grid: 2D array (rows x cols) with -2=walls/outside, 0=passable interior
    - door_positions: Nx2 array of (row, col) for door passable cells
    - exit_positions: Mx2 array of (row, col) for exit passable cells
    - metadata: dict with plan info and dimensions

Usage:
    python resplan_to_npz.py --plan-index 0 --cell-size 0.3 --output plan_0.npz
    python resplan_to_npz.py --random --output random_plan.npz
"""

import pickle
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2

# Add ResPlan utilities
sys.path.insert(0, str(Path(__file__).parent / "ResPlan"))
from resplan_utils import normalize_keys, get_geometries, centroid

from shapely.geometry import Polygon, Point, box
from shapely import affinity


class ResPlanToNPZ:
    """Convert ResPlan vector floor plans to NPZ grid format without scaling."""

    def __init__(self, plan: Dict[str, Any], cell_size: float = 0.3):
        """
        Args:
            plan: ResPlan floor plan dictionary
            cell_size: Physical size of each grid cell in meters (default: 0.3m)

        Note: Walls are always 1 cell wide. Doors remain as walls in the grid
        (door positions are saved in metadata for post-processing).
        """
        self.plan = normalize_keys(plan)
        self.cell_size = cell_size
        self.wall_thickness = 1  # Always 1 cell wide

        # Compute grid dimensions from actual plan size
        self._compute_dimensions()

    def _compute_dimensions(self):
        """Compute grid dimensions from actual plan dimensions using net_area for scaling."""
        # Use 'inner' polygon to get the interior space bounds
        inner = self.plan.get('inner')
        if inner is None or inner.is_empty:
            raise ValueError("Plan has no 'inner' geometry to determine bounds")

        # Get bounds in ResPlan coordinate units (arbitrary units, not meters)
        x_min, y_min, x_max, y_max = inner.bounds
        width_units = x_max - x_min
        height_units = y_max - y_min

        # Get real-world area in square meters
        net_area_m2 = self.plan.get('net_area', 0)
        if net_area_m2 <= 0:
            raise ValueError(f"Plan has invalid or missing net_area: {net_area_m2}")

        # Calculate conversion factor from ResPlan units to meters
        # IMPORTANT: Use actual polygon area, not bounding box area
        # (inner could be L-shaped, irregular, etc.)
        resplan_area_units2 = inner.area  # Shapely polygon's actual area
        scale_factor = np.sqrt(net_area_m2 / resplan_area_units2)

        # Convert to meters
        width_meters = width_units * scale_factor
        height_meters = height_units * scale_factor

        # Calculate grid size based on real dimensions and cell size
        self.grid_cols = int(np.ceil(width_meters / self.cell_size))
        self.grid_rows = int(np.ceil(height_meters / self.cell_size))

        # Store bounds and scale for coordinate conversion
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.scale_factor = scale_factor  # ResPlan units → meters

        print(f"Plan dimensions (ResPlan units): {width_units:.2f} x {height_units:.2f}")
        print(f"Plan dimensions (meters): {width_meters:.2f} x {height_meters:.2f}")
        print(f"Net area: {net_area_m2:.2f} sqm (scale factor: {scale_factor:.4f})")
        print(f"Grid size: {self.grid_rows} x {self.grid_cols} cells ({self.cell_size}m per cell)")

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert ResPlan coordinates to grid (col, row)."""
        # First convert from ResPlan units to meters
        x_meters = (x - self.x_min) * self.scale_factor
        y_meters = (y - self.y_min) * self.scale_factor

        # Then convert meters to grid cells
        col = int(x_meters / self.cell_size)
        row = int(y_meters / self.cell_size)
        return (col, row)

    def _create_base_grid(self) -> np.ndarray:
        """
        Create base grid with -2 everywhere (walls/outside).

        Returns:
            2D array initialized to -2
        """
        return np.full((self.grid_rows, self.grid_cols), -2, dtype=np.float32)

    def _rasterize_polygon(self, polygon: Polygon, value: float,
                          grid: np.ndarray, line_thickness: int = 0) -> np.ndarray:
        """
        Rasterize a polygon onto the grid.

        Args:
            polygon: Shapely Polygon in world coordinates
            value: Value to fill (0 for passable, -2 for obstacle)
            grid: Target grid array
            line_thickness: 0 for filled polygon, >0 for outline only
        """
        # Convert polygon coordinates to grid coordinates
        coords = np.array(polygon.exterior.coords)
        grid_coords = np.zeros_like(coords, dtype=np.int32)

        for i, (x, y) in enumerate(coords):
            col, row = self._world_to_grid(x, y)
            grid_coords[i] = [col, row]

        # Rasterize
        if line_thickness > 0:
            cv2.polylines(grid, [grid_coords], isClosed=True,
                         color=value, thickness=line_thickness)
        else:
            cv2.fillPoly(grid, [grid_coords], color=value)

        # Handle holes (interiors)
        for interior in polygon.interiors:
            interior_coords = np.array(interior.coords)
            interior_grid = np.zeros((len(interior_coords), 2), dtype=np.int32)
            for i, (x, y) in enumerate(interior_coords):
                col, row = self._world_to_grid(x, y)
                interior_grid[i] = [col, row]

            # Holes should be the opposite value
            hole_value = -2 if value == 0 else 0
            cv2.fillPoly(grid, [interior_grid], color=hole_value)

        return grid

    def create_grid(self) -> np.ndarray:
        """
        Create the complete floor plan grid.

        Process:
        1. Start with all -2 (walls/outside)
        2. Fill 'inner' polygon with 0 (passable interior)
        3. Draw walls as -2 on top
        4. Doors will be handled separately

        Returns:
            2D numpy array with -2=walls/outside, 0=interior
        """
        # Start with all -2 (walls and outside)
        grid = self._create_base_grid()

        # Fill interior space with 0 (passable)
        inner = self.plan.get('inner')
        if inner is not None and not inner.is_empty:
            if isinstance(inner, Polygon):
                self._rasterize_polygon(inner, 0, grid)
            else:
                # Handle MultiPolygon
                from shapely.geometry import MultiPolygon
                if isinstance(inner, MultiPolygon):
                    for poly in inner.geoms:
                        self._rasterize_polygon(poly, 0, grid)

        # Draw walls as -2 (thick lines)
        wall_geom = self.plan.get('wall')
        if wall_geom is not None and not wall_geom.is_empty:
            wall_geoms = get_geometries(wall_geom)
            for wg in wall_geoms:
                if isinstance(wg, Polygon):
                    self._rasterize_polygon(wg, -2, grid)
                else:
                    # Handle LineString walls
                    from shapely.geometry import LineString
                    if isinstance(wg, LineString):
                        coords = np.array(wg.coords)
                        grid_coords = np.zeros((len(coords), 2), dtype=np.int32)
                        for i, (x, y) in enumerate(coords):
                            col, row = self._world_to_grid(x, y)
                            grid_coords[i] = [col, row]
                        cv2.polylines(grid, [grid_coords], isClosed=False,
                                    color=-2, thickness=self.wall_thickness)

        # Draw doors as walls (-2) - they'll be opened in post-processing
        door_geoms = get_geometries(self.plan.get('door'))
        for door_geom in door_geoms:
            from shapely.geometry import LineString
            if isinstance(door_geom, LineString):
                coords = np.array(door_geom.coords)
                grid_coords = np.zeros((len(coords), 2), dtype=np.int32)
                for i, (x, y) in enumerate(coords):
                    col, row = self._world_to_grid(x, y)
                    grid_coords[i] = [col, row]
                cv2.polylines(grid, [grid_coords], isClosed=False,
                            color=-2, thickness=self.wall_thickness)
            elif isinstance(door_geom, Polygon):
                self._rasterize_polygon(door_geom, -2, grid)

        # Draw front doors (exits) as walls (-2) - they'll be opened in post-processing
        front_door_geoms = get_geometries(self.plan.get('front_door'))
        for fd_geom in front_door_geoms:
            from shapely.geometry import LineString
            if isinstance(fd_geom, LineString):
                coords = np.array(fd_geom.coords)
                grid_coords = np.zeros((len(coords), 2), dtype=np.int32)
                for i, (x, y) in enumerate(coords):
                    col, row = self._world_to_grid(x, y)
                    grid_coords[i] = [col, row]
                cv2.polylines(grid, [grid_coords], isClosed=False,
                            color=-2, thickness=self.wall_thickness)
            elif isinstance(fd_geom, Polygon):
                self._rasterize_polygon(fd_geom, -2, grid)

        return grid

    def extract_doors(self) -> List[Tuple[int, int]]:
        """
        Extract door positions (for post-processing).

        Doors remain as walls (-2) in the grid.

        Returns:
            door_positions: List of (row, col) tuples
        """
        door_positions = []

        # Internal doors
        door_geoms = get_geometries(self.plan.get('door'))
        for door_geom in door_geoms:
            if door_geom.is_empty:
                continue

            # Just record position, don't modify grid
            c = centroid(door_geom) if hasattr(door_geom, 'centroid') else door_geom.centroid
            col, row = self._world_to_grid(c.x, c.y)

            # Validate
            if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                door_positions.append((row, col))

        return door_positions

    def extract_exits(self) -> List[Tuple[int, int]]:
        """
        Extract exit positions (front doors) for post-processing.

        Exits remain as walls (-2) in the grid.

        Returns:
            exit_positions: List of (row, col) tuples
        """
        exit_positions = []

        # Front doors
        front_door_geoms = get_geometries(self.plan.get('front_door'))
        for fd_geom in front_door_geoms:
            if fd_geom.is_empty:
                continue

            # Just record position, don't modify grid
            c = centroid(fd_geom) if hasattr(fd_geom, 'centroid') else fd_geom.centroid
            col, row = self._world_to_grid(c.x, c.y)

            # Validate
            if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                exit_positions.append((row, col))

        return exit_positions

    def convert(self) -> Dict[str, Any]:
        """
        Perform full conversion to NPZ format.

        Returns:
            Dictionary with:
                - grid: 2D array (doors/exits remain as walls -2)
                - door_positions: Nx2 array
                - exit_positions: Mx2 array
                - metadata: dict
        """
        # Create base grid (doors/exits are walls)
        print("Creating base grid...")
        grid = self.create_grid()

        # Extract door positions (don't modify grid)
        print("Extracting door positions...")
        door_positions = self.extract_doors()
        print(f"Found {len(door_positions)} doors")

        # Extract exit positions (don't modify grid)
        print("Extracting exit positions...")
        exit_positions = self.extract_exits()
        print(f"Found {len(exit_positions)} exits")

        # Prepare metadata
        metadata = {
            'plan_id': self.plan.get('id', -1),
            'unit_type': self.plan.get('unitType', 'unknown'),
            'net_area': self.plan.get('net_area', 0.0),
            'cell_size': self.cell_size,
            'grid_rows': self.grid_rows,
            'grid_cols': self.grid_cols,
            'scale_factor': self.scale_factor,  # ResPlan units → meters
            'world_bounds': {
                'x_min': float(self.x_min),
                'y_min': float(self.y_min),
                'x_max': float(self.x_max),
                'y_max': float(self.y_max)
            },
            'num_doors': len(door_positions),
            'num_exits': len(exit_positions)
        }

        return {
            'grid': grid,
            'door_positions': np.array(door_positions, dtype=np.int32),
            'exit_positions': np.array(exit_positions, dtype=np.int32),
            'metadata': metadata
        }

    def save_npz(self, output_path: str):
        """
        Convert and save to NPZ file.

        Args:
            output_path: Path to output .npz file
        """
        data = self.convert()

        # Save with compression
        np.savez_compressed(
            output_path,
            grid=data['grid'],
            door_positions=data['door_positions'],
            exit_positions=data['exit_positions'],
            **data['metadata']  # Unpack metadata as separate keys
        )

        print(f"\n[SUCCESS] Saved NPZ to: {output_path}")
        print(f"  Grid shape: {data['grid'].shape}")
        print(f"  Passable cells: {np.sum(data['grid'] == 0)}")
        print(f"  Wall cells: {np.sum(data['grid'] == -2)}")
        print(f"  Doors: {len(data['door_positions'])}")
        print(f"  Exits: {len(data['exit_positions'])}")


def load_resplan_dataset(pkl_path: str = "ResPlan/ResPlan.pkl") -> List[Dict[str, Any]]:
    """Load ResPlan dataset from pickle file."""
    with open(pkl_path, 'rb') as f:
        plans = pickle.load(f)

    for plan in plans:
        normalize_keys(plan)

    return plans


def load_npz(npz_path: str) -> Dict[str, Any]:
    """
    Load and parse NPZ file.

    Returns:
        Dictionary with grid, door_positions, exit_positions, and metadata
    """
    data = np.load(npz_path, allow_pickle=True)

    return {
        'grid': data['grid'],
        'door_positions': data['door_positions'],
        'exit_positions': data['exit_positions'],
        'metadata': {
            'plan_id': int(data['plan_id']),
            'unit_type': str(data['unit_type']),
            'net_area': float(data['net_area']),
            'cell_size': float(data['cell_size']),
            'grid_rows': int(data['grid_rows']),
            'grid_cols': int(data['grid_cols']),
            'scale_factor': float(data['scale_factor']),
            'world_bounds': data['world_bounds'].item(),
            'num_doors': int(data['num_doors']),
            'num_exits': int(data['num_exits'])
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert ResPlan floor plans to NPZ format"
    )
    parser.add_argument('--plan-index', type=int,
                       help='Index of plan to convert (0-17106)')
    parser.add_argument('--random', action='store_true',
                       help='Select random plan')
    parser.add_argument('--cell-size', type=float, default=0.3,
                       help='Cell size in meters (default: 0.3)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output NPZ file path')
    parser.add_argument('--pkl-path', type=str, default='ResPlan/ResPlan.pkl',
                       help='Path to ResPlan.pkl (default: ResPlan/ResPlan.pkl)')

    args = parser.parse_args()

    # Validate inputs
    if args.plan_index is None and not args.random:
        parser.error("Must specify either --plan-index or --random")

    # Load dataset
    print(f"Loading ResPlan dataset from {args.pkl_path}...")
    plans = load_resplan_dataset(args.pkl_path)
    print(f"Loaded {len(plans)} plans")

    # Select plan
    if args.random:
        plan_idx = np.random.randint(0, len(plans))
        print(f"Randomly selected plan index: {plan_idx}")
    else:
        plan_idx = args.plan_index
        if not 0 <= plan_idx < len(plans):
            parser.error(f"Plan index must be between 0 and {len(plans)-1}")

    plan = plans[plan_idx]
    print(f"\nConverting plan {plan_idx} (ID: {plan.get('id', 'unknown')})")
    print(f"Unit type: {plan.get('unitType', 'unknown')}")
    print(f"Net area: {plan.get('net_area', 0):.2f} square meters")

    # Convert and save
    try:
        converter = ResPlanToNPZ(
            plan,
            cell_size=args.cell_size
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        converter.save_npz(str(output_path))

    except Exception as e:
        print(f"\n[ERROR] Conversion failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
