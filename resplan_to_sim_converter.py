"""
Convert ResPlan floor plans to Fire Evacuation Simulator configurations.

Usage:
    python resplan_to_sim_converter.py --plan-index 0 --grid-size 100 --output config.json
    python resplan_to_sim_converter.py --random --num-agents 10 --output config.json
"""

import pickle
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Add ResPlan utilities
sys.path.insert(0, str(Path(__file__).parent / "ResPlan"))
from resplan_utils import (
    normalize_keys, get_geometries, centroid,
    geometry_to_mask, plan_to_graph
)


class ResPlanConverter:
    """Convert ResPlan vector floor plans to grid-based simulator configs."""

    def __init__(self, plan: Dict[str, Any], grid_size: int = 100,
                 cell_size: float = 0.3, wall_thickness: int = 2):
        """
        Args:
            plan: ResPlan floor plan dictionary
            grid_size: Target grid dimension (will create grid_size x grid_size)
            cell_size: Simulator cell size in meters (default: 0.3m)
            wall_thickness: Thickness of walls in grid cells (default: 2)
        """
        self.plan = normalize_keys(plan)
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.wall_thickness = wall_thickness

        # Compute scaling factor from ResPlan coordinates to grid
        self._compute_scale()

    def _compute_scale(self):
        """Compute scaling to fit plan into grid_size x grid_size."""
        # Get bounds from 'inner' polygon (overall footprint)
        inner = self.plan.get('inner')
        if inner is None or inner.is_empty:
            # Fallback: use wall bounds
            walls = self.plan.get('wall')
            if walls is None or walls.is_empty:
                raise ValueError("Plan has no 'inner' or 'wall' geometry to determine bounds")
            inner = walls

        x_min, y_min, x_max, y_max = inner.bounds
        width = x_max - x_min
        height = y_max - y_min
        max_dim = max(width, height)

        # Scale to fit in grid with 5% margin
        self.scale = (self.grid_size * 0.95) / max_dim
        self.offset_x = x_min
        self.offset_y = y_min

    def _to_grid_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Convert ResPlan coordinates to grid (col, row)."""
        col = int((x - self.offset_x) * self.scale)
        row = int((y - self.offset_y) * self.scale)
        return (col, row)

    def _to_position_string(self, col: int, row: int) -> str:
        """Convert grid coordinates to simulator position format 'x{col}y{row}'."""
        return f"x{col}y{row}"

    def create_fire_map(self) -> np.ndarray:
        """
        Create initial fire map with walls as obstacles (-2).

        Returns:
            2D numpy array (rows x cols) with:
                0.0 = passable
                -2 = wall/obstacle
        """
        # Create empty grid
        fire_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Rasterize walls to binary mask
        wall_geom = self.plan.get('wall')
        if wall_geom is not None and not wall_geom.is_empty:
            # Scale and translate wall geometry
            from shapely import affinity
            scaled_wall = affinity.scale(wall_geom,
                                        xfact=self.scale,
                                        yfact=self.scale,
                                        origin=(0, 0))
            scaled_wall = affinity.translate(scaled_wall,
                                            xoff=-self.offset_x * self.scale,
                                            yoff=-self.offset_y * self.scale)

            # Rasterize to mask
            wall_mask = geometry_to_mask(scaled_wall,
                                        shape=(self.grid_size, self.grid_size),
                                        line_thickness=self.wall_thickness)

            # Set walls as obstacles (-2)
            fire_map[wall_mask > 127] = -2

        return fire_map

    def extract_doors(self) -> List[Dict[str, str]]:
        """
        Extract door positions from ResPlan geometries.

        Returns:
            List of door configs: [{"id": "d1", "position": "x12y5", "type": "door"}, ...]
        """
        door_configs = []

        # Internal doors
        door_geoms = get_geometries(self.plan.get('door'))
        for i, door_geom in enumerate(door_geoms):
            if door_geom.is_empty:
                continue
            c = door_geom.centroid
            col, row = self._to_grid_coords(c.x, c.y)
            door_configs.append({
                "id": f"d{i}",
                "position": self._to_position_string(col, row),
                "type": "door"
            })

        # Front door (exit)
        front_door_geoms = get_geometries(self.plan.get('front_door'))
        for i, fd_geom in enumerate(front_door_geoms):
            if fd_geom.is_empty:
                continue
            c = centroid(fd_geom) if hasattr(fd_geom, 'centroid') else fd_geom.centroid
            col, row = self._to_grid_coords(c.x, c.y)
            door_configs.append({
                "id": f"exit{i}",
                "position": self._to_position_string(col, row),
                "type": "exit"
            })

        return door_configs

    def get_room_centroids(self, room_types: Optional[List[str]] = None) -> List[Tuple[int, int]]:
        """
        Get centroids of rooms for agent starting positions.

        Args:
            room_types: List of room types to consider (default: bedroom, living, kitchen)

        Returns:
            List of (col, row) tuples
        """
        if room_types is None:
            room_types = ['bedroom', 'living', 'kitchen']

        centroids = []
        for room_type in room_types:
            room_geoms = get_geometries(self.plan.get(room_type))
            for geom in room_geoms:
                if geom.is_empty:
                    continue
                c = centroid(geom)
                col, row = self._to_grid_coords(c.x, c.y)
                # Check if valid (not on wall)
                if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
                    centroids.append((col, row))

        return centroids

    def get_exits(self) -> List[Tuple[int, int]]:
        """Get exit positions (front doors)."""
        exits = []
        front_door_geoms = get_geometries(self.plan.get('front_door'))
        for fd_geom in front_door_geoms:
            if fd_geom.is_empty:
                continue
            c = centroid(fd_geom) if hasattr(fd_geom, 'centroid') else fd_geom.centroid
            col, row = self._to_grid_coords(c.x, c.y)
            if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
                exits.append((col, row))

        return exits

    def generate_config(self, num_agents: int = 10,
                       fire_locations: Optional[List[Tuple[int, int]]] = None,
                       fire_model_type: str = "realistic") -> Dict[str, Any]:
        """
        Generate complete simulation configuration.

        Args:
            num_agents: Number of agents to place
            fire_locations: List of (col, row) for initial fires. If None, places in kitchen
            fire_model_type: "realistic", "aggressive", or "default"

        Returns:
            Simulation config dictionary
        """
        # Get fire map
        fire_map = self.create_fire_map()

        # Place initial fires
        if fire_locations is None:
            # Default: start fire in kitchen
            kitchen_geoms = get_geometries(self.plan.get('kitchen'))
            if kitchen_geoms:
                c = centroid(kitchen_geoms[0])
                col, row = self._to_grid_coords(c.x, c.y)
                fire_locations = [(col, row)]
            else:
                # Fallback: center of map
                fire_locations = [(self.grid_size // 2, self.grid_size // 2)]

        for col, row in fire_locations:
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                fire_map[row, col] = 1.0  # Initial fire

        # Get agent starting positions
        room_centroids = self.get_room_centroids()
        if not room_centroids:
            raise ValueError("No valid room centroids found for agent placement")

        # Sample agent positions (with replacement if needed)
        np.random.seed(42)
        if len(room_centroids) >= num_agents:
            selected_positions = np.random.choice(len(room_centroids),
                                                 size=num_agents,
                                                 replace=False)
        else:
            selected_positions = np.random.choice(len(room_centroids),
                                                 size=num_agents,
                                                 replace=True)

        start_positions = [self._to_position_string(*room_centroids[i])
                          for i in selected_positions]

        # Get exits as targets
        exits = self.get_exits()
        if not exits:
            raise ValueError("No exits found in floor plan")

        targets = [self._to_position_string(*exits[0])]  # Use first exit

        # Get door configs
        door_configs = self.extract_doors()

        # Build config
        config = {
            "map_rows": self.grid_size,
            "map_cols": self.grid_size,
            "agent_num": num_agents,
            "start_positions": start_positions,
            "targets": targets,
            "initial_fire_map": fire_map.tolist(),
            "cell_size": self.cell_size,
            "timestep_duration": 0.5,
            "fire_update_interval": 4,
            "fire_model_type": fire_model_type,
            "door_configs": door_configs,
            "viewing_range": 5,
            "max_occupancy": 2,
            "communication_range": 15.0,
            "sharing_interval": 5,
            "consider_env_factors": False,
            "metadata": {
                "source": "ResPlan",
                "plan_id": self.plan.get('id', 'unknown'),
                "net_area": self.plan.get('net_area', 0),
                "unit_type": self.plan.get('unitType', 'unknown'),
                "scale_factor": self.scale,
                "original_bounds": {
                    "x_min": self.offset_x,
                    "y_min": self.offset_y,
                }
            }
        }

        return config


def load_resplan_dataset(pkl_path: str = "ResPlan/ResPlan.pkl") -> List[Dict[str, Any]]:
    """Load ResPlan dataset from pickle file."""
    with open(pkl_path, 'rb') as f:
        plans = pickle.load(f)

    # Normalize keys
    for plan in plans:
        normalize_keys(plan)

    return plans


def main():
    parser = argparse.ArgumentParser(description="Convert ResPlan floor plans to simulator configs")
    parser.add_argument('--plan-index', type=int, help='Index of plan to convert (0-17106)')
    parser.add_argument('--random', action='store_true', help='Select random plan')
    parser.add_argument('--grid-size', type=int, default=100,
                       help='Grid dimension (default: 100x100)')
    parser.add_argument('--num-agents', type=int, default=10,
                       help='Number of agents (default: 10)')
    parser.add_argument('--cell-size', type=float, default=0.3,
                       help='Cell size in meters (default: 0.3)')
    parser.add_argument('--fire-model', type=str, default='realistic',
                       choices=['realistic', 'aggressive', 'default'],
                       help='Fire model type (default: realistic)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON config file path')
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
    print(f"Converting plan {plan_idx} (ID: {plan.get('id', 'unknown')})")

    # Convert
    try:
        converter = ResPlanConverter(plan,
                                     grid_size=args.grid_size,
                                     cell_size=args.cell_size)
        config = converter.generate_config(num_agents=args.num_agents,
                                          fire_model_type=args.fire_model)

        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n[SUCCESS] Created config: {output_path}")
        print(f"  Grid size: {config['map_rows']}x{config['map_cols']}")
        print(f"  Agents: {config['agent_num']}")
        print(f"  Doors: {len(config['door_configs'])}")
        print(f"  Exits: {len([d for d in config['door_configs'] if d['type'] == 'exit'])}")

    except Exception as e:
        print(f"\n[ERROR] Error during conversion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
