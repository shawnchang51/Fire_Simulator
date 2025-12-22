"""
Floor Plan Generator for Training Data Diversity

Generates diverse floor plans using multiple procedural methods:
- BSP (Binary Space Partitioning): Regular office-like layouts
- Grid-Based: Controlled room variation
- Template-Based: Realistic architectural patterns
- Cellular Automata: Organic/irregular layouts

Usage:
    generator = FloorPlanGenerator(seed=42)
    plans = generator.generate_batch(
        num_plans=1000,
        size_range=(20, 80),
        method_weights={'bsp': 0.4, 'grid': 0.3, 'template': 0.2, 'cellular': 0.1}
    )
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import random
from collections import deque


class CellType(Enum):
    """Cell types in floor plan grid"""
    PASSABLE = 0
    WALL = -2
    EXIT = -3  # Temporary marker, converted to door_config


@dataclass
class FloorPlanMetadata:
    """Metadata about generated floor plan"""
    size: Tuple[int, int]
    room_count: int
    generation_method: str
    obstacle_density: float
    corridor_width: int
    exit_positions: List[Tuple[int, int]] = field(default_factory=list)
    room_centers: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class Room:
    """Represents a room in the floor plan"""
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height


class FloorPlanGenerator:
    """
    Generates diverse floor plans for training data.

    Ensures variety across:
    - Map sizes (20x20 to 80x80)
    - Room counts (1-12 rooms)
    - Layout styles (open, compartmentalized, corridor-based)
    - Obstacle densities (5%-25%)
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.py_random = random.Random(seed)

    def generate_batch(
        self,
        num_plans: int,
        size_range: Tuple[int, int] = (20, 80),
        method_weights: Optional[Dict[str, float]] = None,
        realism_ratio: float = 0.6
    ) -> List[Tuple[np.ndarray, FloorPlanMetadata]]:
        """
        Generate a batch of diverse floor plans.

        Args:
            num_plans: Number of floor plans to generate
            size_range: (min_size, max_size) for map dimensions
            method_weights: Dict mapping method names to weights
                           Default: {'bsp': 0.4, 'grid': 0.3, 'template': 0.2, 'cellular': 0.1}
            realism_ratio: Balance between realistic (0.0-1.0) and challenging plans
                          - 1.0 = all realistic (office, school, hospital templates)
                          - 0.0 = all challenging (cellular, complex BSP)
                          - 0.6 = 60% realistic, 40% challenging (default)

        Returns:
            List of (grid, metadata) tuples
        """
        if method_weights is None:
            # Adjust weights based on realism_ratio
            # Realistic: template (office, school, hospital), simple grid
            # Challenging: cellular, complex BSP
            method_weights = {
                'bsp': 0.25 + 0.15 * (1 - realism_ratio),      # 25-40%
                'grid': 0.20 + 0.10 * (1 - realism_ratio),     # 20-30%
                'template': 0.35 + 0.15 * realism_ratio,       # 35-50%
                'cellular': 0.10 + 0.20 * (1 - realism_ratio)  # 10-30%
            }

        methods = list(method_weights.keys())
        weights = [method_weights[m] for m in methods]

        plans = []
        for i in range(num_plans):
            # Sample size (bias toward medium sizes)
            size = self._sample_size(size_range)

            # Sample method
            method = self.py_random.choices(methods, weights=weights)[0]

            # Generate floor plan
            grid, metadata = self._generate_single(size, method)

            # Validate and retry if invalid
            attempts = 0
            while not self._validate_plan(grid, metadata) and attempts < 5:
                grid, metadata = self._generate_single(size, method)
                attempts += 1

            if self._validate_plan(grid, metadata):
                plans.append((grid, metadata))

        return plans

    def _sample_size(self, size_range: Tuple[int, int]) -> Tuple[int, int]:
        """Sample map size with bias toward medium sizes"""
        min_size, max_size = size_range

        # Use truncated normal for natural distribution
        mean = (min_size + max_size) / 2
        std = (max_size - min_size) / 4

        rows = int(np.clip(self.rng.normal(mean, std), min_size, max_size))
        cols = int(np.clip(self.rng.normal(mean, std), min_size, max_size))

        # Occasionally make non-square
        if self.rng.random() < 0.3:
            aspect = self.rng.uniform(0.6, 1.4)
            cols = int(np.clip(rows * aspect, min_size, max_size))

        return (rows, cols)

    def _generate_single(
        self,
        size: Tuple[int, int],
        method: str
    ) -> Tuple[np.ndarray, FloorPlanMetadata]:
        """Generate a single floor plan using specified method"""

        generators = {
            'bsp': self._generate_bsp,
            'grid': self._generate_grid,
            'template': self._generate_template,
            'cellular': self._generate_cellular
        }

        return generators[method](size)

    def _generate_bsp(
        self,
        size: Tuple[int, int]
    ) -> Tuple[np.ndarray, FloorPlanMetadata]:
        """
        Binary Space Partitioning - creates office-like layouts.
        Recursively divides space into rooms and connects them.
        """
        rows, cols = size
        grid = np.full((rows, cols), CellType.WALL.value, dtype=np.float32)

        # Parameters
        min_room_size = max(5, min(rows, cols) // 6)
        max_depth = self.rng.integers(2, 5)
        corridor_width = self.rng.choice([1, 2])

        rooms: List[Room] = []

        def split_space(x: int, y: int, w: int, h: int, depth: int):
            """Recursively split space into rooms"""
            # Need <= to ensure split position has valid range (low < high)
            if depth >= max_depth or w <= min_room_size * 2 or h <= min_room_size * 2:
                # Create room with margin for walls
                room = Room(x + 1, y + 1, w - 2, h - 2)
                if room.width >= 3 and room.height >= 3:
                    rooms.append(room)
                return

            # Decide split direction
            if w > h * 1.25:
                split_vertical = True
            elif h > w * 1.25:
                split_vertical = False
            else:
                split_vertical = self.rng.random() < 0.5

            if split_vertical:
                split_pos = self.rng.integers(min_room_size, w - min_room_size)
                split_space(x, y, split_pos, h, depth + 1)
                split_space(x + split_pos, y, w - split_pos, h, depth + 1)
            else:
                split_pos = self.rng.integers(min_room_size, h - min_room_size)
                split_space(x, y, w, split_pos, depth + 1)
                split_space(x, y + split_pos, w, h - split_pos, depth + 1)

        # Generate rooms
        split_space(0, 0, cols, rows, 0)

        # Carve out rooms
        for room in rooms:
            grid[room.y:room.y+room.height, room.x:room.x+room.width] = CellType.PASSABLE.value

        # Connect rooms with corridors
        self._connect_rooms_mst(grid, rooms, corridor_width)

        # Add internal partitions to large rooms
        self._add_room_partitions(grid, rooms)

        # Add structured obstacles for challenging navigation
        obstacle_density = self.rng.uniform(0.08, 0.18)
        self._add_structured_obstacles(grid, obstacle_density, 'office_building')

        # Add perimeter wall
        grid[0, :] = CellType.WALL.value
        grid[-1, :] = CellType.WALL.value
        grid[:, 0] = CellType.WALL.value
        grid[:, -1] = CellType.WALL.value

        metadata = FloorPlanMetadata(
            size=size,
            room_count=len(rooms),
            generation_method='bsp',
            obstacle_density=float(np.sum(grid == CellType.WALL.value)) / grid.size,
            corridor_width=corridor_width,
            room_centers=[room.center for room in rooms]
        )

        return grid, metadata

    def _generate_grid(
        self,
        size: Tuple[int, int]
    ) -> Tuple[np.ndarray, FloorPlanMetadata]:
        """
        Grid-based generation - divides into sectors and merges some.
        Good for controlled variation.
        """
        rows, cols = size
        grid = np.full((rows, cols), CellType.WALL.value, dtype=np.float32)

        # Scale grid divisions with map size for more complexity
        min_dim = min(rows, cols)
        min_divisions = max(2, min_dim // 15)  # At least 2, more for larger maps
        max_divisions = max(3, min_dim // 8)   # Scale up with size

        grid_rows = self.rng.integers(min_divisions, max_divisions + 1)
        grid_cols = self.rng.integers(min_divisions, max_divisions + 1)
        corridor_width = self.rng.choice([1, 2])

        cell_h = (rows - 2) // grid_rows
        cell_w = (cols - 2) // grid_cols

        rooms = []

        # Create grid of rooms
        for gr in range(grid_rows):
            for gc in range(grid_cols):
                y = 1 + gr * cell_h
                x = 1 + gc * cell_w
                h = cell_h - 1
                w = cell_w - 1

                if h >= 3 and w >= 3:
                    room = Room(x, y, w, h)
                    rooms.append(room)
                    # Carve room
                    grid[y:y+h, x:x+w] = CellType.PASSABLE.value

        # Randomly merge adjacent rooms (remove walls between them)
        merge_prob = self.rng.uniform(0.2, 0.5)
        for i, room1 in enumerate(rooms):
            for j, room2 in enumerate(rooms[i+1:], i+1):
                if self.rng.random() < merge_prob:
                    # Check if adjacent
                    if self._rooms_adjacent(room1, room2):
                        self._merge_rooms(grid, room1, room2, corridor_width)

        # Connect all rooms
        self._connect_rooms_mst(grid, rooms, corridor_width)

        # Add internal partitions to large rooms
        self._add_room_partitions(grid, rooms)

        # Add structured obstacles for challenging navigation
        obstacle_density = self.rng.uniform(0.08, 0.18)
        self._add_structured_obstacles(grid, obstacle_density, 'generic')

        metadata = FloorPlanMetadata(
            size=size,
            room_count=len(rooms),
            generation_method='grid',
            obstacle_density=float(np.sum(grid == CellType.WALL.value)) / grid.size,
            corridor_width=corridor_width,
            room_centers=[room.center for room in rooms]
        )

        return grid, metadata

    def _generate_template(
        self,
        size: Tuple[int, int]
    ) -> Tuple[np.ndarray, FloorPlanMetadata]:
        """
        Template-based generation - uses predefined room shapes.
        Creates more realistic architectural patterns.
        """
        rows, cols = size
        grid = np.full((rows, cols), CellType.WALL.value, dtype=np.float32)

        corridor_width = self.rng.choice([2, 3])

        # Choose template pattern - mix of realistic and varied layouts
        pattern = self.rng.choice([
            'corridor_central',   # Classic office hallway
            'office_building',    # Realistic office with cubicles
            'school_layout',      # Classrooms along corridor
            'warehouse',          # Open space with aisles
            'hospital_wing',      # Patient rooms with nurse station
            'l_shape',
            'u_shape',
            'open_office'
        ])

        rooms = []

        if pattern == 'corridor_central':
            # Central corridor with rooms on both sides
            corridor_y = rows // 2 - corridor_width // 2
            grid[corridor_y:corridor_y+corridor_width, 1:cols-1] = CellType.PASSABLE.value

            # Rooms on top
            room_count_top = self.rng.integers(2, 5)
            room_width = (cols - 2) // room_count_top
            for i in range(room_count_top):
                x = 1 + i * room_width
                y = 1
                h = corridor_y - 2
                w = room_width - 1
                if h >= 3 and w >= 3:
                    room = Room(x, y, w, h)
                    rooms.append(room)
                    grid[y:y+h, x:x+w] = CellType.PASSABLE.value
                    # Door to corridor
                    door_x = x + w // 2
                    grid[y+h:corridor_y+1, door_x:door_x+1] = CellType.PASSABLE.value

            # Rooms on bottom
            room_count_bottom = self.rng.integers(2, 5)
            room_width = (cols - 2) // room_count_bottom
            for i in range(room_count_bottom):
                x = 1 + i * room_width
                y = corridor_y + corridor_width + 1
                h = rows - y - 2
                w = room_width - 1
                if h >= 3 and w >= 3:
                    room = Room(x, y, w, h)
                    rooms.append(room)
                    grid[y:y+h, x:x+w] = CellType.PASSABLE.value
                    # Door to corridor
                    door_x = x + w // 2
                    grid[corridor_y+corridor_width-1:y+1, door_x:door_x+1] = CellType.PASSABLE.value

        elif pattern == 'l_shape':
            # L-shaped building
            # Vertical section
            v_width = cols // 3
            grid[1:rows-1, 1:v_width] = CellType.PASSABLE.value
            rooms.append(Room(1, 1, v_width-1, rows-2))

            # Horizontal section
            h_height = rows // 3
            grid[rows-h_height-1:rows-1, 1:cols-1] = CellType.PASSABLE.value
            rooms.append(Room(v_width, rows-h_height-1, cols-v_width-1, h_height))

        elif pattern == 'u_shape':
            # U-shaped building
            wing_width = cols // 4
            center_height = rows // 3

            # Left wing
            grid[1:rows-1, 1:wing_width+1] = CellType.PASSABLE.value
            rooms.append(Room(1, 1, wing_width, rows-2))

            # Right wing
            grid[1:rows-1, cols-wing_width-1:cols-1] = CellType.PASSABLE.value
            rooms.append(Room(cols-wing_width-1, 1, wing_width, rows-2))

            # Bottom connector
            grid[rows-center_height-1:rows-1, 1:cols-1] = CellType.PASSABLE.value

        elif pattern == 'office_building':
            # Realistic office: reception, meeting rooms, cubicle area, break room
            # Main corridor
            corridor_y = rows // 2 - corridor_width // 2
            grid[corridor_y:corridor_y+corridor_width, 1:cols-1] = CellType.PASSABLE.value

            # Reception area (larger room at entrance)
            reception_w = cols // 4
            grid[corridor_y-3:corridor_y+corridor_width+3, 1:reception_w] = CellType.PASSABLE.value
            rooms.append(Room(1, corridor_y-3, reception_w-1, corridor_width+6))

            # Meeting rooms (top side, fewer but larger)
            meeting_room_count = self.rng.integers(2, 4)
            available_width = cols - reception_w - 2
            meeting_w = available_width // meeting_room_count
            for i in range(meeting_room_count):
                x = reception_w + i * meeting_w
                y = 1
                h = corridor_y - 2
                w = meeting_w - 1
                if h >= 4 and w >= 4:
                    room = Room(x, y, w, h)
                    rooms.append(room)
                    grid[y:y+h, x:x+w] = CellType.PASSABLE.value
                    # Door
                    door_x = x + w // 2
                    grid[y+h:corridor_y+1, door_x] = CellType.PASSABLE.value

            # Cubicle area (bottom side, open with desk obstacles added later)
            cubicle_y = corridor_y + corridor_width + 1
            cubicle_h = rows - cubicle_y - 2
            cubicle_w = (cols - 2) * 2 // 3
            if cubicle_h >= 4:
                grid[cubicle_y:cubicle_y+cubicle_h, 1:1+cubicle_w] = CellType.PASSABLE.value
                rooms.append(Room(1, cubicle_y, cubicle_w, cubicle_h))
                # Connect to corridor
                grid[corridor_y:cubicle_y+1, cubicle_w//2] = CellType.PASSABLE.value

            # Break room (bottom right)
            break_x = 1 + cubicle_w + 1
            break_w = cols - break_x - 1
            if break_w >= 4 and cubicle_h >= 4:
                grid[cubicle_y:cubicle_y+cubicle_h, break_x:break_x+break_w] = CellType.PASSABLE.value
                rooms.append(Room(break_x, cubicle_y, break_w, cubicle_h))
                # Connect to corridor
                grid[corridor_y:cubicle_y+1, break_x+break_w//2] = CellType.PASSABLE.value

        elif pattern == 'school_layout':
            # School: classrooms along main corridor, larger rooms at ends
            corridor_y = rows // 2 - corridor_width // 2
            grid[corridor_y:corridor_y+corridor_width, 1:cols-1] = CellType.PASSABLE.value

            # Classrooms on both sides (uniform size)
            classroom_count = max(2, (cols - 4) // 8)
            classroom_w = (cols - 2) // classroom_count

            for i in range(classroom_count):
                x = 1 + i * classroom_w
                w = classroom_w - 1

                # Top classroom
                y_top = 1
                h_top = corridor_y - 2
                if h_top >= 4 and w >= 4:
                    room = Room(x, y_top, w, h_top)
                    rooms.append(room)
                    grid[y_top:y_top+h_top, x:x+w] = CellType.PASSABLE.value
                    # Door (offset to side like real classrooms)
                    door_x = x + 2
                    grid[y_top+h_top:corridor_y+1, door_x] = CellType.PASSABLE.value

                # Bottom classroom
                y_bot = corridor_y + corridor_width + 1
                h_bot = rows - y_bot - 1
                if h_bot >= 4 and w >= 4:
                    room = Room(x, y_bot, w, h_bot)
                    rooms.append(room)
                    grid[y_bot:y_bot+h_bot, x:x+w] = CellType.PASSABLE.value
                    # Door
                    door_x = x + 2
                    grid[corridor_y+corridor_width-1:y_bot+1, door_x] = CellType.PASSABLE.value

        elif pattern == 'warehouse':
            # Warehouse: open space with regular aisles
            grid[1:rows-1, 1:cols-1] = CellType.PASSABLE.value
            rooms.append(Room(1, 1, cols-2, rows-2))

            # Create shelf aisles (vertical walls with gaps)
            aisle_spacing = self.rng.integers(6, 10)
            shelf_depth = self.rng.integers(2, 4)

            for x in range(aisle_spacing, cols - aisle_spacing, aisle_spacing):
                for y in range(3, rows - 3):
                    # Leave gaps for cross-aisles
                    if y % (rows // 3) < 2:
                        continue
                    if x + shelf_depth < cols - 1:
                        grid[y, x:x+shelf_depth] = CellType.WALL.value

        elif pattern == 'hospital_wing':
            # Hospital: patient rooms on one side, nurse station, utility rooms
            corridor_y = rows // 3
            corridor_h = corridor_width + 1
            grid[corridor_y:corridor_y+corridor_h, 1:cols-1] = CellType.PASSABLE.value

            # Patient rooms (top, uniform small rooms)
            patient_room_w = max(4, (cols - 2) // 6)
            patient_room_count = (cols - 2) // patient_room_w

            for i in range(patient_room_count):
                x = 1 + i * patient_room_w
                y = 1
                h = corridor_y - 1
                w = patient_room_w - 1
                if h >= 3 and w >= 3:
                    room = Room(x, y, w, h)
                    rooms.append(room)
                    grid[y:y+h, x:x+w] = CellType.PASSABLE.value
                    # Door
                    door_x = x + w // 2
                    grid[y+h:corridor_y+1, door_x] = CellType.PASSABLE.value

            # Nurse station (central, below corridor)
            station_y = corridor_y + corridor_h
            station_h = (rows - station_y - 1) // 2
            station_w = cols // 3
            station_x = (cols - station_w) // 2
            if station_h >= 3:
                grid[station_y:station_y+station_h, station_x:station_x+station_w] = CellType.PASSABLE.value
                rooms.append(Room(station_x, station_y, station_w, station_h))
                # Connect to corridor
                grid[corridor_y+corridor_h-1:station_y+1, station_x+station_w//2] = CellType.PASSABLE.value

            # Utility rooms on sides (bottom)
            util_y = station_y + station_h + 1
            util_h = rows - util_y - 1
            if util_h >= 3:
                # Left utility
                util_w = station_x - 2
                if util_w >= 3:
                    grid[util_y:util_y+util_h, 1:1+util_w] = CellType.PASSABLE.value
                    rooms.append(Room(1, util_y, util_w, util_h))
                    grid[station_y:util_y+1, util_w//2] = CellType.PASSABLE.value

                # Right utility
                right_x = station_x + station_w + 1
                right_w = cols - right_x - 1
                if right_w >= 3:
                    grid[util_y:util_y+util_h, right_x:right_x+right_w] = CellType.PASSABLE.value
                    rooms.append(Room(right_x, util_y, right_w, util_h))
                    grid[station_y:util_y+1, right_x+right_w//2] = CellType.PASSABLE.value

        else:  # open_office
            # Large open space with scattered columns/obstacles
            grid[1:rows-1, 1:cols-1] = CellType.PASSABLE.value
            rooms.append(Room(1, 1, cols-2, rows-2))

            # Add structural columns
            num_columns = self.rng.integers(4, 12)
            for _ in range(num_columns):
                cx = self.rng.integers(3, cols-3)
                cy = self.rng.integers(3, rows-3)
                col_size = self.rng.integers(1, 3)
                grid[cy:cy+col_size, cx:cx+col_size] = CellType.WALL.value

        # Add structured obstacles (furniture) for realistic layouts
        # Use lower density for already-structured templates
        if pattern in ['warehouse', 'hospital_wing', 'school_layout']:
            obstacle_density = self.rng.uniform(0.03, 0.08)
        else:
            obstacle_density = self.rng.uniform(0.08, 0.15)
        self._add_structured_obstacles(grid, obstacle_density, pattern)

        metadata = FloorPlanMetadata(
            size=size,
            room_count=len(rooms),
            generation_method='template',
            obstacle_density=float(np.sum(grid == CellType.WALL.value)) / grid.size,
            corridor_width=corridor_width,
            room_centers=[room.center for room in rooms]
        )

        return grid, metadata

    def _generate_cellular(
        self,
        size: Tuple[int, int]
    ) -> Tuple[np.ndarray, FloorPlanMetadata]:
        """
        Cellular automata - creates organic, irregular layouts.
        Good for edge cases and unusual scenarios.
        """
        rows, cols = size

        # Initialize with random noise (higher fill = more walls = more challenging)
        fill_prob = self.rng.uniform(0.45, 0.55)
        grid = (self.rng.random((rows, cols)) < fill_prob).astype(np.float32)
        grid = np.where(grid > 0, CellType.WALL.value, CellType.PASSABLE.value)

        # Apply smoothing iterations
        iterations = self.rng.integers(3, 7)
        birth_threshold = 5
        death_threshold = 4

        for _ in range(iterations):
            new_grid = grid.copy()
            for y in range(1, rows-1):
                for x in range(1, cols-1):
                    neighbors = np.sum(grid[y-1:y+2, x-1:x+2] == CellType.WALL.value) - (1 if grid[y, x] == CellType.WALL.value else 0)

                    if grid[y, x] == CellType.WALL.value:
                        if neighbors < death_threshold:
                            new_grid[y, x] = CellType.PASSABLE.value
                    else:
                        if neighbors >= birth_threshold:
                            new_grid[y, x] = CellType.WALL.value
            grid = new_grid

        # Ensure perimeter walls
        grid[0, :] = CellType.WALL.value
        grid[-1, :] = CellType.WALL.value
        grid[:, 0] = CellType.WALL.value
        grid[:, -1] = CellType.WALL.value

        # Find and keep only the largest connected component
        grid = self._keep_largest_component(grid)

        # Count approximate "rooms" (connected open areas separated by narrow passages)
        room_count = self._estimate_room_count(grid)

        metadata = FloorPlanMetadata(
            size=size,
            room_count=room_count,
            generation_method='cellular',
            obstacle_density=float(np.sum(grid == CellType.WALL.value)) / grid.size,
            corridor_width=1,
            room_centers=[]
        )

        return grid, metadata

    def _connect_rooms_mst(
        self,
        grid: np.ndarray,
        rooms: List[Room],
        corridor_width: int
    ):
        """Connect rooms using minimum spanning tree approach"""
        if len(rooms) < 2:
            return

        # Build complete graph of room distances
        centers = [room.center for room in rooms]
        n = len(centers)

        # Prim's algorithm for MST
        connected = {0}
        edges = []

        while len(connected) < n:
            best_edge = None
            best_dist = float('inf')

            for i in connected:
                for j in range(n):
                    if j not in connected:
                        dist = abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1])
                        if dist < best_dist:
                            best_dist = dist
                            best_edge = (i, j)

            if best_edge:
                edges.append(best_edge)
                connected.add(best_edge[1])

        # Carve corridors for MST edges
        for i, j in edges:
            self._carve_corridor(grid, centers[i], centers[j], corridor_width)

    def _carve_corridor(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        width: int
    ):
        """Carve an L-shaped corridor between two points"""
        x1, y1 = start
        x2, y2 = end

        # Randomly choose to go horizontal-first or vertical-first
        if self.rng.random() < 0.5:
            # Horizontal then vertical
            for x in range(min(x1, x2), max(x1, x2) + 1):
                for w in range(width):
                    if 0 <= y1 + w < grid.shape[0]:
                        grid[y1 + w, x] = CellType.PASSABLE.value
            for y in range(min(y1, y2), max(y1, y2) + 1):
                for w in range(width):
                    if 0 <= x2 + w < grid.shape[1]:
                        grid[y, x2 + w] = CellType.PASSABLE.value
        else:
            # Vertical then horizontal
            for y in range(min(y1, y2), max(y1, y2) + 1):
                for w in range(width):
                    if 0 <= x1 + w < grid.shape[1]:
                        grid[y, x1 + w] = CellType.PASSABLE.value
            for x in range(min(x1, x2), max(x1, x2) + 1):
                for w in range(width):
                    if 0 <= y2 + w < grid.shape[0]:
                        grid[y2 + w, x] = CellType.PASSABLE.value

    def _rooms_adjacent(self, room1: Room, room2: Room) -> bool:
        """Check if two rooms are adjacent (share a wall)"""
        # Horizontal adjacency
        if room1.x + room1.width == room2.x or room2.x + room2.width == room1.x:
            y_overlap = min(room1.y + room1.height, room2.y + room2.height) - max(room1.y, room2.y)
            return y_overlap > 2

        # Vertical adjacency
        if room1.y + room1.height == room2.y or room2.y + room2.height == room1.y:
            x_overlap = min(room1.x + room1.width, room2.x + room2.width) - max(room1.x, room2.x)
            return x_overlap > 2

        return False

    def _add_room_partitions(self, grid: np.ndarray, rooms: List[Room]):
        """
        Add internal partitions to large rooms to create sub-spaces.
        Makes navigation more interesting and realistic.
        """
        min_area_for_partition = 80  # Only partition rooms larger than this

        for room in rooms:
            if room.area < min_area_for_partition:
                continue

            # Probability of adding partition increases with room size
            partition_prob = min(0.8, room.area / 200)
            if self.rng.random() > partition_prob:
                continue

            # Choose partition style
            if room.width > room.height * 1.5:
                # Wide room: vertical partition
                self._add_vertical_partition(grid, room)
            elif room.height > room.width * 1.5:
                # Tall room: horizontal partition
                self._add_horizontal_partition(grid, room)
            else:
                # Square-ish room: random or L-shaped partition
                if self.rng.random() < 0.5:
                    if self.rng.random() < 0.5:
                        self._add_vertical_partition(grid, room)
                    else:
                        self._add_horizontal_partition(grid, room)
                else:
                    self._add_l_partition(grid, room)

    def _add_vertical_partition(self, grid: np.ndarray, room: Room):
        """Add vertical partition wall with doorway"""
        # Position partition at 1/3 or 2/3 of room width
        offset_ratio = self.py_random.choice([1/3, 1/2, 2/3])
        px = room.x + int(room.width * offset_ratio)

        # Leave doorway (1-2 cells)
        door_size = self.rng.integers(1, 3)
        door_start = room.y + self.rng.integers(1, max(2, room.height - door_size - 1))

        for y in range(room.y, room.y + room.height):
            if door_start <= y < door_start + door_size:
                continue  # Doorway
            if 0 < y < grid.shape[0] - 1 and 0 < px < grid.shape[1] - 1:
                grid[y, px] = CellType.WALL.value

    def _add_horizontal_partition(self, grid: np.ndarray, room: Room):
        """Add horizontal partition wall with doorway"""
        offset_ratio = self.py_random.choice([1/3, 1/2, 2/3])
        py = room.y + int(room.height * offset_ratio)

        door_size = self.rng.integers(1, 3)
        door_start = room.x + self.rng.integers(1, max(2, room.width - door_size - 1))

        for x in range(room.x, room.x + room.width):
            if door_start <= x < door_start + door_size:
                continue  # Doorway
            if 0 < py < grid.shape[0] - 1 and 0 < x < grid.shape[1] - 1:
                grid[py, x] = CellType.WALL.value

    def _add_l_partition(self, grid: np.ndarray, room: Room):
        """Add L-shaped partition to create corner space"""
        # Choose corner
        corner = self.py_random.choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])

        # Partition size (1/3 to 1/2 of room)
        h_len = int(room.width * self.rng.uniform(0.3, 0.5))
        v_len = int(room.height * self.rng.uniform(0.3, 0.5))

        if corner == 'top_left':
            px, py = room.x + h_len, room.y
            # Vertical part
            for y in range(room.y, room.y + v_len):
                if 0 < y < grid.shape[0] - 1 and 0 < px < grid.shape[1] - 1:
                    grid[y, px] = CellType.WALL.value
            # Horizontal part (with gap)
            py = room.y + v_len
            for x in range(room.x, px):
                if x == room.x + h_len // 2:
                    continue  # Doorway
                if 0 < py < grid.shape[0] - 1 and 0 < x < grid.shape[1] - 1:
                    grid[py, x] = CellType.WALL.value

        elif corner == 'top_right':
            px = room.x + room.width - h_len
            for y in range(room.y, room.y + v_len):
                if 0 < y < grid.shape[0] - 1 and 0 < px < grid.shape[1] - 1:
                    grid[y, px] = CellType.WALL.value
            py = room.y + v_len
            for x in range(px + 1, room.x + room.width):
                if x == px + h_len // 2:
                    continue
                if 0 < py < grid.shape[0] - 1 and 0 < x < grid.shape[1] - 1:
                    grid[py, x] = CellType.WALL.value

        elif corner == 'bottom_left':
            px = room.x + h_len
            for y in range(room.y + room.height - v_len, room.y + room.height):
                if 0 < y < grid.shape[0] - 1 and 0 < px < grid.shape[1] - 1:
                    grid[y, px] = CellType.WALL.value
            py = room.y + room.height - v_len
            for x in range(room.x, px):
                if x == room.x + h_len // 2:
                    continue
                if 0 < py < grid.shape[0] - 1 and 0 < x < grid.shape[1] - 1:
                    grid[py, x] = CellType.WALL.value

        else:  # bottom_right
            px = room.x + room.width - h_len
            for y in range(room.y + room.height - v_len, room.y + room.height):
                if 0 < y < grid.shape[0] - 1 and 0 < px < grid.shape[1] - 1:
                    grid[y, px] = CellType.WALL.value
            py = room.y + room.height - v_len
            for x in range(px + 1, room.x + room.width):
                if x == px + h_len // 2:
                    continue
                if 0 < py < grid.shape[0] - 1 and 0 < x < grid.shape[1] - 1:
                    grid[py, x] = CellType.WALL.value

    def _merge_rooms(
        self,
        grid: np.ndarray,
        room1: Room,
        room2: Room,
        corridor_width: int
    ):
        """Remove wall between adjacent rooms"""
        # Find shared wall and create opening
        if room1.x + room1.width <= room2.x:
            # room1 is left of room2
            x = room1.x + room1.width
            y_start = max(room1.y, room2.y) + 1
            y_end = min(room1.y + room1.height, room2.y + room2.height) - 1
            for y in range(y_start, y_end):
                grid[y, x] = CellType.PASSABLE.value
        elif room2.x + room2.width <= room1.x:
            # room2 is left of room1
            x = room2.x + room2.width
            y_start = max(room1.y, room2.y) + 1
            y_end = min(room1.y + room1.height, room2.y + room2.height) - 1
            for y in range(y_start, y_end):
                grid[y, x] = CellType.PASSABLE.value
        elif room1.y + room1.height <= room2.y:
            # room1 is above room2
            y = room1.y + room1.height
            x_start = max(room1.x, room2.x) + 1
            x_end = min(room1.x + room1.width, room2.x + room2.width) - 1
            for x in range(x_start, x_end):
                grid[y, x] = CellType.PASSABLE.value
        elif room2.y + room2.height <= room1.y:
            # room2 is above room1
            y = room2.y + room2.height
            x_start = max(room1.x, room2.x) + 1
            x_end = min(room1.x + room1.width, room2.x + room2.width) - 1
            for x in range(x_start, x_end):
                grid[y, x] = CellType.PASSABLE.value

    def _add_obstacles(self, grid: np.ndarray, density: float):
        """Add random internal obstacles (furniture, columns) - legacy method"""
        # Delegate to structured obstacles for better quality
        self._add_structured_obstacles(grid, density, 'generic')

    def _add_structured_obstacles(self, grid: np.ndarray, density: float, context: str = 'generic'):
        """
        Add structured obstacles that create meaningful navigation challenges.

        Args:
            grid: Floor plan grid
            density: Target obstacle density (0.0-1.0)
            context: Layout context for appropriate furniture types
        """
        rows, cols = grid.shape
        passable_coords = np.argwhere(grid == CellType.PASSABLE.value)

        if len(passable_coords) == 0:
            return

        target_obstacles = int(len(passable_coords) * density)
        placed = 0
        max_attempts = target_obstacles * 3
        attempts = 0

        # Context-appropriate obstacle types
        if context in ['office_building', 'open_office']:
            obstacle_types = ['desk_cluster', 'desk_row', 'meeting_table', 'cubicle_wall']
        elif context == 'school_layout':
            obstacle_types = ['desk_row', 'desk_cluster', 'teacher_desk']
        elif context == 'warehouse':
            obstacle_types = ['shelf_unit', 'pallet']
        elif context == 'hospital_wing':
            obstacle_types = ['bed', 'equipment', 'desk']
        else:
            obstacle_types = ['wall_segment', 'furniture_block', 'l_shape', 'desk_row']

        while placed < target_obstacles and attempts < max_attempts:
            attempts += 1

            idx = self.rng.integers(len(passable_coords))
            y, x = passable_coords[idx]

            # Skip cells too close to edges
            if y < 3 or y > rows - 4 or x < 3 or x > cols - 4:
                continue

            obs_type = self.py_random.choice(obstacle_types)
            cells_placed = self._place_obstacle(grid, y, x, obs_type, rows, cols)
            placed += cells_placed

    def _place_obstacle(self, grid: np.ndarray, y: int, x: int, obs_type: str, rows: int, cols: int) -> int:
        """Place a single structured obstacle, return number of cells placed"""
        cells_placed = 0

        if obs_type == 'wall_segment':
            # Horizontal or vertical wall segment (3-6 cells) - creates chokepoints
            length = self.rng.integers(3, 7)
            horizontal = self.rng.random() < 0.5
            for i in range(length):
                ny, nx = (y, x + i) if horizontal else (y + i, x)
                if 0 < ny < rows - 1 and 0 < nx < cols - 1:
                    if grid[ny, nx] == CellType.PASSABLE.value:
                        grid[ny, nx] = CellType.WALL.value
                        cells_placed += 1

        elif obs_type == 'furniture_block':
            # 2x2 or 2x3 solid block (desk, table)
            bh, bw = self.py_random.choice([(2, 2), (2, 3), (3, 2)])
            for dy in range(bh):
                for dx in range(bw):
                    ny, nx = y + dy, x + dx
                    if 0 < ny < rows - 1 and 0 < nx < cols - 1:
                        if grid[ny, nx] == CellType.PASSABLE.value:
                            grid[ny, nx] = CellType.WALL.value
                            cells_placed += 1

        elif obs_type == 'l_shape':
            # L-shaped obstacle - blocks diagonal shortcuts
            pattern = self.py_random.choice([
                [(0, 0), (0, 1), (1, 0)],           # └
                [(0, 0), (0, 1), (1, 1)],           # ┘
                [(0, 0), (1, 0), (1, 1)],           # ┐
                [(0, 1), (1, 0), (1, 1)],           # ┌
                [(0, 0), (0, 1), (0, 2), (1, 0)],   # Larger └
                [(0, 0), (1, 0), (2, 0), (2, 1)],   # Larger ┐
            ])
            for dy, dx in pattern:
                ny, nx = y + dy, x + dx
                if 0 < ny < rows - 1 and 0 < nx < cols - 1:
                    if grid[ny, nx] == CellType.PASSABLE.value:
                        grid[ny, nx] = CellType.WALL.value
                        cells_placed += 1

        elif obs_type == 'desk_row':
            # Row of desks with gaps (realistic office)
            length = self.rng.integers(4, 8)
            horizontal = self.rng.random() < 0.5
            for i in range(length):
                if i % 3 == 2:  # Gap every 3rd cell for walkway
                    continue
                ny, nx = (y, x + i) if horizontal else (y + i, x)
                if 0 < ny < rows - 1 and 0 < nx < cols - 1:
                    if grid[ny, nx] == CellType.PASSABLE.value:
                        grid[ny, nx] = CellType.WALL.value
                        cells_placed += 1

        elif obs_type == 'desk_cluster':
            # 2x2 cluster of desks (4 people facing each other)
            for dy in range(2):
                for dx in range(3):
                    if dx == 1:  # Middle gap
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 < ny < rows - 1 and 0 < nx < cols - 1:
                        if grid[ny, nx] == CellType.PASSABLE.value:
                            grid[ny, nx] = CellType.WALL.value
                            cells_placed += 1

        elif obs_type == 'cubicle_wall':
            # Cubicle partition walls (T or + shape)
            pattern = self.py_random.choice([
                [(0, 0), (0, 1), (0, 2), (1, 1)],           # T shape
                [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],   # + shape
                [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],   # ├ shape
            ])
            for dy, dx in pattern:
                ny, nx = y + dy, x + dx
                if 0 < ny < rows - 1 and 0 < nx < cols - 1:
                    if grid[ny, nx] == CellType.PASSABLE.value:
                        grid[ny, nx] = CellType.WALL.value
                        cells_placed += 1

        elif obs_type == 'meeting_table':
            # Rectangular meeting table (2x4 or 2x5)
            length = self.rng.integers(4, 6)
            for dy in range(2):
                for dx in range(length):
                    ny, nx = y + dy, x + dx
                    if 0 < ny < rows - 1 and 0 < nx < cols - 1:
                        if grid[ny, nx] == CellType.PASSABLE.value:
                            grid[ny, nx] = CellType.WALL.value
                            cells_placed += 1

        elif obs_type == 'shelf_unit':
            # Warehouse shelf (long vertical structure)
            length = self.rng.integers(4, 8)
            width = self.rng.integers(1, 3)
            for dy in range(length):
                for dx in range(width):
                    ny, nx = y + dy, x + dx
                    if 0 < ny < rows - 1 and 0 < nx < cols - 1:
                        if grid[ny, nx] == CellType.PASSABLE.value:
                            grid[ny, nx] = CellType.WALL.value
                            cells_placed += 1

        elif obs_type == 'pallet':
            # Small pallet/box (2x2)
            for dy in range(2):
                for dx in range(2):
                    ny, nx = y + dy, x + dx
                    if 0 < ny < rows - 1 and 0 < nx < cols - 1:
                        if grid[ny, nx] == CellType.PASSABLE.value:
                            grid[ny, nx] = CellType.WALL.value
                            cells_placed += 1

        elif obs_type == 'bed':
            # Hospital bed (2x3)
            for dy in range(2):
                for dx in range(3):
                    ny, nx = y + dy, x + dx
                    if 0 < ny < rows - 1 and 0 < nx < cols - 1:
                        if grid[ny, nx] == CellType.PASSABLE.value:
                            grid[ny, nx] = CellType.WALL.value
                            cells_placed += 1

        elif obs_type in ['equipment', 'desk', 'teacher_desk']:
            # Generic 2x2 equipment/desk
            for dy in range(2):
                for dx in range(2):
                    ny, nx = y + dy, x + dx
                    if 0 < ny < rows - 1 and 0 < nx < cols - 1:
                        if grid[ny, nx] == CellType.PASSABLE.value:
                            grid[ny, nx] = CellType.WALL.value
                            cells_placed += 1

        return cells_placed

    def _keep_largest_component(self, grid: np.ndarray) -> np.ndarray:
        """Keep only the largest connected passable area"""
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)

        components = []

        for y in range(rows):
            for x in range(cols):
                if grid[y, x] == CellType.PASSABLE.value and not visited[y, x]:
                    # BFS to find component
                    component = []
                    queue = deque([(y, x)])
                    visited[y, x] = True

                    while queue:
                        cy, cx = queue.popleft()
                        component.append((cy, cx))

                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < rows and 0 <= nx < cols:
                                if grid[ny, nx] == CellType.PASSABLE.value and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    queue.append((ny, nx))

                    components.append(component)

        if not components:
            return grid

        # Keep largest component, fill others with walls
        largest = max(components, key=len)
        largest_set = set(largest)

        for y in range(rows):
            for x in range(cols):
                if grid[y, x] == CellType.PASSABLE.value and (y, x) not in largest_set:
                    grid[y, x] = CellType.WALL.value

        return grid

    def _estimate_room_count(self, grid: np.ndarray) -> int:
        """Estimate number of distinct rooms in cellular automata output"""
        # Simple heuristic: count large open areas
        rows, cols = grid.shape
        passable_count = np.sum(grid == CellType.PASSABLE.value)

        if passable_count < 50:
            return 1

        # Rough estimate based on area
        avg_room_size = 100
        return max(1, int(passable_count / avg_room_size))

    def _validate_plan(self, grid: np.ndarray, metadata: FloorPlanMetadata) -> bool:
        """Validate that floor plan is usable for simulation"""
        rows, cols = grid.shape

        # Check minimum passable area (at least 20% of grid)
        passable = np.sum(grid == CellType.PASSABLE.value)
        if passable < 0.15 * rows * cols:
            return False

        # Check that there's a connected passable region
        if passable < 20:  # Need at least 20 passable cells
            return False

        # Check perimeter has walls
        if not (np.all(grid[0, :] == CellType.WALL.value) and
                np.all(grid[-1, :] == CellType.WALL.value) and
                np.all(grid[:, 0] == CellType.WALL.value) and
                np.all(grid[:, -1] == CellType.WALL.value)):
            return False

        return True

    def add_exits_to_plan(
        self,
        grid: np.ndarray,
        num_exits: int = 2,
        placement: str = 'distributed'
    ) -> List[Tuple[int, int]]:
        """
        Add exit positions to floor plan perimeter.

        Args:
            grid: Floor plan grid (modified in place)
            num_exits: Number of exits to add
            placement: 'distributed', 'corners', or 'random'

        Returns:
            List of exit positions (col, row)
        """
        rows, cols = grid.shape

        # Find valid perimeter positions (adjacent to passable interior)
        valid_positions = []

        # Top and bottom edges
        for x in range(1, cols - 1):
            if grid[1, x] == CellType.PASSABLE.value:
                valid_positions.append((x, 0, 'top'))
            if grid[rows-2, x] == CellType.PASSABLE.value:
                valid_positions.append((x, rows-1, 'bottom'))

        # Left and right edges
        for y in range(1, rows - 1):
            if grid[y, 1] == CellType.PASSABLE.value:
                valid_positions.append((0, y, 'left'))
            if grid[y, cols-2] == CellType.PASSABLE.value:
                valid_positions.append((cols-1, y, 'right'))

        if len(valid_positions) < num_exits:
            num_exits = len(valid_positions)

        exits = []

        if placement == 'corners':
            # Prefer corner positions
            corner_regions = {
                'top_left': [(x, y, e) for x, y, e in valid_positions if x < cols//3 and y < rows//3],
                'top_right': [(x, y, e) for x, y, e in valid_positions if x > 2*cols//3 and y < rows//3],
                'bottom_left': [(x, y, e) for x, y, e in valid_positions if x < cols//3 and y > 2*rows//3],
                'bottom_right': [(x, y, e) for x, y, e in valid_positions if x > 2*cols//3 and y > 2*rows//3],
            }

            for region in corner_regions.values():
                if region and len(exits) < num_exits:
                    pos = self.py_random.choice(region)
                    exits.append((pos[0], pos[1]))
                    valid_positions.remove(pos)

        elif placement == 'distributed':
            # Distribute exits evenly around perimeter
            edges = {'top': [], 'bottom': [], 'left': [], 'right': []}
            for pos in valid_positions:
                edges[pos[2]].append(pos)

            # Round-robin from each edge
            edge_order = ['top', 'right', 'bottom', 'left']
            idx = 0
            while len(exits) < num_exits:
                edge = edge_order[idx % 4]
                if edges[edge]:
                    pos = self.py_random.choice(edges[edge])
                    exits.append((pos[0], pos[1]))
                    edges[edge].remove(pos)
                idx += 1
                if all(len(e) == 0 for e in edges.values()):
                    break

        else:  # random
            sampled = self.py_random.sample(valid_positions, min(num_exits, len(valid_positions)))
            exits = [(pos[0], pos[1]) for pos in sampled]

        # Mark exits as passable (create openings in perimeter)
        for x, y in exits:
            grid[y, x] = CellType.PASSABLE.value

        return exits


def generate_diverse_plans(
    num_plans: int = 1000,
    seed: int = 42,
    size_range: Tuple[int, int] = (20, 80),
    realism_ratio: float = 0.6
) -> List[Tuple[np.ndarray, FloorPlanMetadata]]:
    """
    Convenience function to generate diverse floor plans.

    Args:
        num_plans: Number of plans to generate
        seed: Random seed for reproducibility
        size_range: (min_size, max_size) for map dimensions
        realism_ratio: Balance between realistic and challenging plans (0.0-1.0)
                      - 1.0 = mostly realistic building layouts
                      - 0.0 = mostly challenging/irregular layouts
                      - 0.6 = balanced mix (default)

    Returns:
        List of (grid, metadata) tuples
    """
    generator = FloorPlanGenerator(seed=seed)
    return generator.generate_batch(num_plans, size_range, realism_ratio=realism_ratio)


if __name__ == '__main__':
    # Test generation
    print("=" * 60)
    print("Floor Plan Generator - Test Suite")
    print("=" * 60)

    generator = FloorPlanGenerator(seed=42)

    # Test with different realism ratios
    print("\n--- Testing with realism_ratio=0.8 (mostly realistic) ---")
    plans_realistic = generator.generate_batch(5, size_range=(35, 50), realism_ratio=0.8)

    for i, (grid, meta) in enumerate(plans_realistic):
        print(f"\nPlan {i+1}: {meta.generation_method}")
        print(f"  Size: {meta.size}, Rooms: {meta.room_count}")
        print(f"  Obstacle density: {meta.obstacle_density:.1%}")

        # Show preview
        preview_rows = min(12, grid.shape[0])
        preview_cols = min(25, grid.shape[1])
        print(f"  Preview ({preview_rows}x{preview_cols}):")
        for row in grid[:preview_rows, :preview_cols]:
            print("    " + "".join(['#' if c == -2 else '.' for c in row]))

    print("\n--- Testing with realism_ratio=0.2 (mostly challenging) ---")
    generator2 = FloorPlanGenerator(seed=123)
    plans_challenging = generator2.generate_batch(5, size_range=(35, 50), realism_ratio=0.2)

    for i, (grid, meta) in enumerate(plans_challenging):
        print(f"\nPlan {i+1}: {meta.generation_method}")
        print(f"  Size: {meta.size}, Rooms: {meta.room_count}")
        print(f"  Obstacle density: {meta.obstacle_density:.1%}")

        preview_rows = min(12, grid.shape[0])
        preview_cols = min(25, grid.shape[1])
        print(f"  Preview ({preview_rows}x{preview_cols}):")
        for row in grid[:preview_rows, :preview_cols]:
            print("    " + "".join(['#' if c == -2 else '.' for c in row]))

    # Method distribution summary
    print("\n--- Method Distribution Summary ---")
    generator3 = FloorPlanGenerator(seed=999)
    test_plans = generator3.generate_batch(100, size_range=(30, 60), realism_ratio=0.6)
    method_counts = {}
    for _, meta in test_plans:
        method_counts[meta.generation_method] = method_counts.get(meta.generation_method, 0) + 1
    print("100 plans with realism_ratio=0.6:")
    for method, count in sorted(method_counts.items()):
        print(f"  {method}: {count}%")
