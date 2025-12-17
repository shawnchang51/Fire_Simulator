"""
Candidate Generator for AI-Guided Design Optimization
======================================================

Generates door configuration candidates for pairwise comparison labeling.
Supports both random and rule-based placement strategies.

Part of Phase 1: Conservative Optimizations
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import random
from scipy.ndimage import label as connected_components


class CandidateGenerator:
    """
    Generate door configuration candidates using random and rule-based strategies.

    Supports:
    - Random placement on walls
    - Rule-based placement (on room boundaries, exits near edges)
    - Constraint validation (minimum spacing, connectivity)
    """

    def __init__(self,
                 floor_plan: np.ndarray,
                 min_door_spacing: int = 3,
                 wall_value: float = -2,
                 seed: Optional[int] = None):
        """
        Initialize candidate generator.

        Args:
            floor_plan: 2D numpy array (-2=wall, 0=empty, >0=fire)
            min_door_spacing: Minimum cells between doors
            wall_value: Value indicating walls
            seed: Random seed for reproducibility
        """
        self.floor_plan = floor_plan.copy()
        self.rows, self.cols = floor_plan.shape
        self.min_door_spacing = min_door_spacing
        self.wall_value = wall_value

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Precompute valid wall positions (walls adjacent to passable cells)
        self.valid_wall_positions = self._find_valid_wall_positions()

        # Identify rooms using connected components
        self.rooms = self._identify_rooms()

        # Find room boundaries (walls between rooms)
        self.room_boundaries = self._find_room_boundaries()

        # Find perimeter positions (walls on building edge)
        self.perimeter_positions = self._find_perimeter_positions()

    def _find_valid_wall_positions(self) -> List[Tuple[int, int]]:
        """
        Find wall positions that are adjacent to passable cells.
        These are valid door placement locations.
        """
        valid_positions = []

        for y in range(self.rows):
            for x in range(self.cols):
                if self.floor_plan[y, x] == self.wall_value:
                    # Check if adjacent to at least 2 passable cells
                    adjacent_passable = 0
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.rows and 0 <= nx < self.cols:
                            if self.floor_plan[ny, nx] >= 0:  # Passable
                                adjacent_passable += 1

                    # Valid door location: wall with 2+ passable neighbors
                    if adjacent_passable >= 2:
                        valid_positions.append((x, y))

        return valid_positions

    def _identify_rooms(self) -> List[Set[Tuple[int, int]]]:
        """
        Identify separate rooms using connected components.
        Returns list of sets, each containing (x, y) coordinates of a room.
        """
        # Create binary mask of passable cells
        passable = (self.floor_plan >= 0).astype(int)

        # Find connected components (rooms)
        labeled, num_rooms = connected_components(passable)

        rooms = []
        for room_id in range(1, num_rooms + 1):
            room_coords = set()
            coords = np.argwhere(labeled == room_id)
            for y, x in coords:
                room_coords.add((x, y))
            if len(room_coords) > 10:  # Ignore tiny spaces
                rooms.append(room_coords)

        return rooms

    def _find_room_boundaries(self) -> List[Tuple[int, int]]:
        """
        Find walls that separate different rooms (interior walls).
        These are high-priority locations for doors.
        """
        boundaries = []

        for y in range(1, self.rows - 1):
            for x in range(1, self.cols - 1):
                if self.floor_plan[y, x] == self.wall_value:
                    # Check if wall separates two different rooms
                    adjacent_rooms = set()
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if self.floor_plan[ny, nx] >= 0:
                            # Find which room this cell belongs to
                            for room_id, room in enumerate(self.rooms):
                                if (nx, ny) in room:
                                    adjacent_rooms.add(room_id)
                                    break

                    # Wall between different rooms
                    if len(adjacent_rooms) >= 2:
                        boundaries.append((x, y))

        return boundaries

    def _find_perimeter_positions(self) -> List[Tuple[int, int]]:
        """
        Find walls on the building perimeter (exits).
        """
        perimeter = []

        for x, y in self.valid_wall_positions:
            # Check if on or near edge
            if x <= 2 or x >= self.cols - 3 or y <= 2 or y >= self.rows - 3:
                perimeter.append((x, y))

        return perimeter

    def _check_spacing_constraint(self,
                                   position: Tuple[int, int],
                                   existing_positions: List[Tuple[int, int]]) -> bool:
        """
        Check if position maintains minimum spacing from existing doors.
        """
        x, y = position
        for ex, ey in existing_positions:
            dist = abs(x - ex) + abs(y - ey)  # Manhattan distance
            if dist < self.min_door_spacing:
                return False
        return True

    def generate_random_candidate(self,
                                   num_doors: int = 3,
                                   num_exits: int = 2) -> List[Dict]:
        """
        Generate a random door configuration.

        Args:
            num_doors: Number of internal doors
            num_exits: Number of exits

        Returns:
            List of door configuration dicts
        """
        candidate = []
        placed_positions = []
        door_counter = 0
        exit_counter = 0

        # Place exits first (on perimeter)
        attempts = 0
        while exit_counter < num_exits and attempts < 1000:
            if not self.perimeter_positions:
                break

            pos = random.choice(self.perimeter_positions)

            if self._check_spacing_constraint(pos, placed_positions):
                candidate.append({
                    "id": f"e{exit_counter + 1}",
                    "position": f"x{pos[0]}y{pos[1]}",
                    "type": "exit"
                })
                placed_positions.append(pos)
                exit_counter += 1

            attempts += 1

        # Place internal doors
        attempts = 0
        while door_counter < num_doors and attempts < 1000:
            if not self.valid_wall_positions:
                break

            pos = random.choice(self.valid_wall_positions)

            if self._check_spacing_constraint(pos, placed_positions):
                candidate.append({
                    "id": f"d{door_counter + 1}",
                    "position": f"x{pos[0]}y{pos[1]}",
                    "type": "door"
                })
                placed_positions.append(pos)
                door_counter += 1

            attempts += 1

        return candidate

    def generate_rule_based_candidate(self,
                                       num_doors: int = 3,
                                       num_exits: int = 2,
                                       strategy: str = 'boundary_focused') -> List[Dict]:
        """
        Generate rule-based door configuration.

        Args:
            num_doors: Number of internal doors
            num_exits: Number of exits
            strategy: 'boundary_focused', 'distributed', or 'corner_exits'

        Returns:
            List of door configuration dicts
        """
        candidate = []
        placed_positions = []

        if strategy == 'boundary_focused':
            # Prioritize room boundaries for doors
            candidate = self._place_boundary_focused(num_doors, num_exits, placed_positions)

        elif strategy == 'distributed':
            # Distribute doors evenly across floor plan
            candidate = self._place_distributed(num_doors, num_exits, placed_positions)

        elif strategy == 'corner_exits':
            # Place exits in corners, doors on boundaries
            candidate = self._place_corner_exits(num_doors, num_exits, placed_positions)

        return candidate

    def _place_boundary_focused(self,
                                 num_doors: int,
                                 num_exits: int,
                                 placed_positions: List) -> List[Dict]:
        """Place doors prioritizing room boundaries."""
        candidate = []
        door_counter = 0
        exit_counter = 0

        # Place exits on perimeter
        perimeter_sample = random.sample(
            self.perimeter_positions,
            min(num_exits, len(self.perimeter_positions))
        )

        for pos in perimeter_sample:
            if self._check_spacing_constraint(pos, placed_positions):
                candidate.append({
                    "id": f"e{exit_counter + 1}",
                    "position": f"x{pos[0]}y{pos[1]}",
                    "type": "exit"
                })
                placed_positions.append(pos)
                exit_counter += 1
                if exit_counter >= num_exits:
                    break

        # Place doors on room boundaries
        boundary_sample = random.sample(
            self.room_boundaries,
            min(num_doors, len(self.room_boundaries))
        )

        for pos in boundary_sample:
            if self._check_spacing_constraint(pos, placed_positions):
                candidate.append({
                    "id": f"d{door_counter + 1}",
                    "position": f"x{pos[0]}y{pos[1]}",
                    "type": "door"
                })
                placed_positions.append(pos)
                door_counter += 1
                if door_counter >= num_doors:
                    break

        return candidate

    def _place_distributed(self,
                           num_doors: int,
                           num_exits: int,
                           placed_positions: List) -> List[Dict]:
        """Distribute doors evenly across the floor plan."""
        candidate = []

        # Divide floor plan into grid sectors
        num_sectors = num_doors + num_exits
        sector_rows = int(np.sqrt(num_sectors))
        sector_cols = (num_sectors + sector_rows - 1) // sector_rows

        sector_height = self.rows // sector_rows
        sector_width = self.cols // sector_cols

        door_counter = 0
        exit_counter = 0

        for i in range(num_sectors):
            sector_y = (i // sector_cols) * sector_height
            sector_x = (i % sector_cols) * sector_width

            # Find valid positions in this sector
            sector_positions = [
                pos for pos in self.valid_wall_positions
                if sector_x <= pos[0] < sector_x + sector_width
                and sector_y <= pos[1] < sector_y + sector_height
            ]

            if not sector_positions:
                continue

            pos = random.choice(sector_positions)

            if self._check_spacing_constraint(pos, placed_positions):
                # First sectors are exits, rest are doors
                if exit_counter < num_exits and pos in self.perimeter_positions:
                    candidate.append({
                        "id": f"e{exit_counter + 1}",
                        "position": f"x{pos[0]}y{pos[1]}",
                        "type": "exit"
                    })
                    exit_counter += 1
                elif door_counter < num_doors:
                    candidate.append({
                        "id": f"d{door_counter + 1}",
                        "position": f"x{pos[0]}y{pos[1]}",
                        "type": "door"
                    })
                    door_counter += 1

                placed_positions.append(pos)

        return candidate

    def _place_corner_exits(self,
                            num_doors: int,
                            num_exits: int,
                            placed_positions: List) -> List[Dict]:
        """Place exits in corners, doors on boundaries."""
        candidate = []

        # Find corner positions (within distance from corners)
        corners = [
            (x, y) for x, y in self.perimeter_positions
            if (x < 5 or x > self.cols - 6) and (y < 5 or y > self.rows - 6)
        ]

        exit_counter = 0
        for pos in random.sample(corners, min(num_exits, len(corners))):
            if self._check_spacing_constraint(pos, placed_positions):
                candidate.append({
                    "id": f"e{exit_counter + 1}",
                    "position": f"x{pos[0]}y{pos[1]}",
                    "type": "exit"
                })
                placed_positions.append(pos)
                exit_counter += 1
                if exit_counter >= num_exits:
                    break

        # Fill remaining exits from perimeter if needed
        if exit_counter < num_exits:
            for pos in self.perimeter_positions:
                if pos not in placed_positions and self._check_spacing_constraint(pos, placed_positions):
                    candidate.append({
                        "id": f"e{exit_counter + 1}",
                        "position": f"x{pos[0]}y{pos[1]}",
                        "type": "exit"
                    })
                    placed_positions.append(pos)
                    exit_counter += 1
                    if exit_counter >= num_exits:
                        break

        # Place doors on room boundaries
        door_counter = 0
        for pos in random.sample(self.room_boundaries, min(num_doors, len(self.room_boundaries))):
            if self._check_spacing_constraint(pos, placed_positions):
                candidate.append({
                    "id": f"d{door_counter + 1}",
                    "position": f"x{pos[0]}y{pos[1]}",
                    "type": "door"
                })
                placed_positions.append(pos)
                door_counter += 1
                if door_counter >= num_doors:
                    break

        return candidate

    def generate_candidate_pool(self,
                                num_candidates: int,
                                num_doors_range: Tuple[int, int] = (2, 5),
                                num_exits_range: Tuple[int, int] = (1, 3),
                                random_ratio: float = 0.5) -> List[List[Dict]]:
        """
        Generate a pool of diverse door configuration candidates.

        Args:
            num_candidates: Number of candidates to generate
            num_doors_range: (min, max) number of internal doors
            num_exits_range: (min, max) number of exits
            random_ratio: Ratio of random vs rule-based (0.5 = 50% random)

        Returns:
            List of door configuration candidates
        """
        candidates = []
        strategies = ['boundary_focused', 'distributed', 'corner_exits']

        for i in range(num_candidates):
            num_doors = random.randint(*num_doors_range)
            num_exits = random.randint(*num_exits_range)

            # Randomly choose random or rule-based
            if random.random() < random_ratio:
                candidate = self.generate_random_candidate(num_doors, num_exits)
            else:
                strategy = random.choice(strategies)
                candidate = self.generate_rule_based_candidate(
                    num_doors, num_exits, strategy
                )

            if candidate:  # Only add non-empty candidates
                candidates.append(candidate)

        return candidates


def generate_door_candidates(floor_plan: np.ndarray,
                              num_candidates: int,
                              num_doors_range: Tuple[int, int] = (2, 5),
                              num_exits_range: Tuple[int, int] = (1, 3),
                              min_door_spacing: int = 3,
                              random_ratio: float = 0.5,
                              seed: Optional[int] = None) -> List[List[Dict]]:
    """
    Convenience function to generate door configuration candidates.

    Args:
        floor_plan: 2D numpy array (-2=wall, 0=empty)
        num_candidates: Number of candidates to generate
        num_doors_range: (min, max) internal doors per candidate
        num_exits_range: (min, max) exits per candidate
        min_door_spacing: Minimum spacing between doors (Manhattan distance)
        random_ratio: Ratio of random vs rule-based placement (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        List of door configuration candidates, each a list of door dicts

    Example:
        >>> floor_plan = np.array([[...]])  # Your floor plan
        >>> candidates = generate_door_candidates(
        ...     floor_plan,
        ...     num_candidates=50,
        ...     num_doors_range=(2, 4),
        ...     num_exits_range=(1, 2),
        ...     seed=42
        ... )
        >>> print(f"Generated {len(candidates)} candidates")
        >>> print(f"First candidate: {candidates[0]}")
    """
    generator = CandidateGenerator(
        floor_plan=floor_plan,
        min_door_spacing=min_door_spacing,
        seed=seed
    )

    return generator.generate_candidate_pool(
        num_candidates=num_candidates,
        num_doors_range=num_doors_range,
        num_exits_range=num_exits_range,
        random_ratio=random_ratio
    )


if __name__ == "__main__":
    # Demo usage
    print("Door Configuration Candidate Generator Demo")
    print("=" * 50)

    # Create a simple test floor plan
    test_plan = np.zeros((30, 30))

    # Add walls
    test_plan[0:3, :] = -2  # Top border
    test_plan[-3:, :] = -2  # Bottom border
    test_plan[:, 0:3] = -2  # Left border
    test_plan[:, -3:] = -2  # Right border

    # Add internal wall
    test_plan[10:20, 14:16] = -2

    print(f"Floor plan shape: {test_plan.shape}")
    print(f"Walls: {np.sum(test_plan == -2)} cells")
    print(f"Passable: {np.sum(test_plan >= 0)} cells")

    # Generate candidates
    candidates = generate_door_candidates(
        floor_plan=test_plan,
        num_candidates=10,
        num_doors_range=(2, 4),
        num_exits_range=(1, 2),
        min_door_spacing=5,
        random_ratio=0.5,
        seed=42
    )

    print(f"\nGenerated {len(candidates)} candidates")

    # Show first 3 candidates
    for i, candidate in enumerate(candidates[:3]):
        print(f"\nCandidate {i + 1}:")
        for door in candidate:
            print(f"  {door}")
