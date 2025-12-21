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
        method_weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[np.ndarray, FloorPlanMetadata]]:
        """
        Generate a batch of diverse floor plans.

        Args:
            num_plans: Number of floor plans to generate
            size_range: (min_size, max_size) for map dimensions
            method_weights: Dict mapping method names to weights
                           Default: {'bsp': 0.4, 'grid': 0.3, 'template': 0.2, 'cellular': 0.1}

        Returns:
            List of (grid, metadata) tuples
        """
        if method_weights is None:
            method_weights = {
                'bsp': 0.4,
                'grid': 0.3,
                'template': 0.2,
                'cellular': 0.1
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

        # Add some internal obstacles (furniture)
        obstacle_density = self.rng.uniform(0.02, 0.08)
        self._add_obstacles(grid, obstacle_density)

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

        # Grid divisions
        grid_rows = self.rng.integers(2, 5)
        grid_cols = self.rng.integers(2, 5)
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

        # Add obstacles
        obstacle_density = self.rng.uniform(0.03, 0.10)
        self._add_obstacles(grid, obstacle_density)

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

        # Choose template pattern
        pattern = self.rng.choice(['corridor_central', 'l_shape', 'u_shape', 'open_office'])

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

        # Add internal obstacles
        obstacle_density = self.rng.uniform(0.05, 0.15)
        self._add_obstacles(grid, obstacle_density)

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

        # Initialize with random noise
        fill_prob = self.rng.uniform(0.40, 0.50)
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
        """Add random internal obstacles (furniture, columns)"""
        rows, cols = grid.shape
        passable = np.argwhere(grid == CellType.PASSABLE.value)

        num_obstacles = int(len(passable) * density)

        for _ in range(num_obstacles):
            if len(passable) == 0:
                break

            idx = self.rng.integers(len(passable))
            y, x = passable[idx]

            # Small obstacle (1-2 cells)
            size = self.rng.choice([1, 1, 1, 2])
            for dy in range(size):
                for dx in range(size):
                    ny, nx = y + dy, x + dx
                    if 0 < ny < rows - 1 and 0 < nx < cols - 1:
                        if grid[ny, nx] == CellType.PASSABLE.value:
                            grid[ny, nx] = CellType.WALL.value

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
    size_range: Tuple[int, int] = (20, 80)
) -> List[Tuple[np.ndarray, FloorPlanMetadata]]:
    """
    Convenience function to generate diverse floor plans.

    Args:
        num_plans: Number of plans to generate
        seed: Random seed for reproducibility
        size_range: (min_size, max_size) for map dimensions

    Returns:
        List of (grid, metadata) tuples
    """
    generator = FloorPlanGenerator(seed=seed)
    return generator.generate_batch(num_plans, size_range)


if __name__ == '__main__':
    # Test generation
    print("Generating sample floor plans...")

    generator = FloorPlanGenerator(seed=42)
    plans = generator.generate_batch(10, size_range=(30, 60))

    for i, (grid, meta) in enumerate(plans):
        print(f"\nPlan {i+1}:")
        print(f"  Size: {meta.size}")
        print(f"  Rooms: {meta.room_count}")
        print(f"  Method: {meta.generation_method}")
        print(f"  Obstacle density: {meta.obstacle_density:.2%}")

        # Add exits
        exits = generator.add_exits_to_plan(grid, num_exits=2)
        print(f"  Exits: {exits}")

        # Show small preview
        preview = grid[:10, :10]
        print("  Preview (10x10):")
        for row in preview:
            print("    " + "".join(['#' if c == -2 else '.' for c in row]))
