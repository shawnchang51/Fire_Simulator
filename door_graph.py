"""
Door/Exit Graph for Hierarchical Pathfinding
============================================
Implements two-tier navigation system:
- High-level: Door/exit graph (this file)
- Low-level: D* Lite on grid (existing system)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from d_star_lite.utils import stateNameToCoords
from collections import deque
import heapq


@dataclass
class DoorNode:
    """Represents a door or exit in the building"""
    id: str
    position: str  # "x15y3" format
    type: str  # 'door' or 'exit'


@dataclass
class DoorEdge:
    """Connection between two doors in the same room"""
    door_a: str
    door_b: str
    base_distance: float  # initial BFS distance
    current_weight: float  # dynamically updated


class DoorGraph:
    """Graph of doors and exits with connectivity"""

    def __init__(self):
        self.nodes: Dict[str, DoorNode] = {}
        self.edges: Dict[Tuple[str, str], DoorEdge] = {}
        self.adjacency: Dict[str, List[str]] = {}
        # Cache for get_connected_nodes results
        self._connected_cache: Dict[str, List[Tuple[str, float]]] = {}
        self._fire_mean_cache: Dict[str, float] = {}

    def __str__(self):
        return f"DoorGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"

    def add_node(self, node: DoorNode):
        """Add a door/exit node"""
        self.nodes[node.id] = node
        self.adjacency[node.id] = []

    def add_edge(self, door_a_id: str, door_b_id: str, distance: float):
        """Add bidirectional edge"""
        edge = DoorEdge(door_a_id, door_b_id, distance, distance)
        self.edges[(door_a_id, door_b_id)] = edge
        self.edges[(door_b_id, door_a_id)] = edge
        self.adjacency[door_a_id].append(door_b_id)
        self.adjacency[door_b_id].append(door_a_id)

    def get_weight(self, door_a: str, door_b: str) -> float:
        """Get current edge weight"""
        edge = self.edges.get((door_a, door_b))
        return edge.current_weight if edge else float('inf')

    def update_edge_weight(self, door_a: str, door_b: str, new_weight: float):
        """Update weight for both directions"""
        if (door_a, door_b) in self.edges:
            self.edges[(door_a, door_b)].current_weight = new_weight
        if (door_b, door_a) in self.edges:
            self.edges[(door_b, door_a)].current_weight = new_weight

    def get_connected_nodes_cached(self, grid, position: str) -> List[Tuple[str, float]]:
        """
        Get connected nodes with caching.

        Args:
            grid: 2D array representing the environment
            position: Starting position in "x{col}y{row}" format

        Returns:
            List of tuples (position_string, distance) for all connected doors
        """
        if position in self._connected_cache:
            return self._connected_cache[position], self._fire_mean_cache.get(position, 0.0)

        # Compute and cache
        door_positions = [node.position for node in self.nodes.values()]
        result, fire_mean = get_connected_nodes(grid, position, obstacle_value=-2,
                                     door_positions=door_positions)
        print(f"\033[32mCaching connected nodes for position {position}: {result} (type: {type(result)})\033[0m")
        self._connected_cache[position] = result
        self._fire_mean_cache[position] = fire_mean
        return result, fire_mean

    def clear_cache(self):
        """Clear the entire cache (call when grid changes due to fire/obstacles)"""
        self._connected_cache.clear()
        self._fire_mean_cache.clear()

    def invalidate_cache_region(self, changed_positions: List[str]):
        """
        Invalidate cache for positions affected by fire/obstacles.

        Args:
            changed_positions: List of position strings where fire/obstacles appeared
        """
        # Simple approach: Clear entire cache when anything changes
        # This is conservative but safe
        self._connected_cache.clear()
        self._fire_mean_cache.clear()

    def invalidate_cache_position(self, position: str):
        """
        Invalidate cache for a specific position.

        Args:
            position: Position string to invalidate
        """
        if position in self._connected_cache:
            del self._connected_cache[position]
        if position in self._fire_mean_cache:
            del self._fire_mean_cache[position]


def bfs_with_blocked_cells(grid, start_pos, goal_pos, blocked_positions):
    """
    BFS that treats blocked_positions as obstacles.
    Returns path length or None if unreachable.
    """
    # Convert to tuples for hashing
    start_pos = tuple(start_pos) if isinstance(start_pos, list) else start_pos
    goal_pos = tuple(goal_pos) if isinstance(goal_pos, list) else goal_pos

    if start_pos == goal_pos:
        return 0

    rows = len(grid)
    cols = len(grid[0])
    # Convert lists to tuples for hashing
    blocked_set = set(tuple(pos) if isinstance(pos, list) else pos for pos in blocked_positions)

    queue = deque([(start_pos[0], start_pos[1], 0)])
    visited = {start_pos}

    # 8-directional movement
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    while queue:
        x, y, dist = queue.popleft()

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Bounds check
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue

            # Skip visited
            if (nx, ny) in visited:
                continue

            # Skip walls (-2) and blocked doors
            if grid[ny][nx] == -2 or (nx, ny) in blocked_set:
                continue

            # Found goal
            if (nx, ny) == goal_pos:
                return dist + 1

            visited.add((nx, ny))
            queue.append((nx, ny, dist + 1))

    return None  # Unreachable


def compute_connectivity(grid, door_nodes: List[DoorNode]) -> List[Tuple[str, str, float]]:
    """
    Compute which doors are in same room (reachable without passing other doors).
    Returns list of (door_a_id, door_b_id, distance) tuples.
    """
    edges = []
    door_positions = {d.id: stateNameToCoords(d.position) for d in door_nodes}

    for i, door_i in enumerate(door_nodes):
        for door_j in door_nodes[i+1:]:
            pos_i = door_positions[door_i.id]
            pos_j = door_positions[door_j.id]

            # Block all OTHER doors
            blocked = [
                door_positions[d.id]
                for d in door_nodes
                if d.id not in [door_i.id, door_j.id]
            ]

            # Try to reach door_j from door_i
            distance = bfs_with_blocked_cells(grid, pos_i, pos_j, blocked)
            print(f"Computed connectivity between {door_i.id} and {door_j.id}: distance={distance}")

            if distance is not None:
                edges.append((door_i.id, door_j.id, float(distance)))

    return edges


def build_door_graph(grid, door_configs: List[dict]) -> DoorGraph:
    """
    Build complete door graph from configuration.

    Args:
        grid: 2D fire map
        door_configs: list of {"id": "d1", "position": "x15y3", "type": "door"}

    Returns: DoorGraph with nodes and edges
    """
    graph = DoorGraph()

    # Add nodes
    door_nodes = []
    for config in door_configs:
        node = DoorNode(
            id=config['id'],
            position=config['position'],
            type=config['type']
        )
        graph.add_node(node)
        door_nodes.append(node)

    # Compute connectivity
    edges = compute_connectivity(grid, door_nodes)

    # Add edges
    for door_a, door_b, distance in edges:
        graph.add_edge(door_a, door_b, distance)

    print(f"[DoorGraph] Built graph: {len(graph.nodes)} nodes, {len(edges)} edges")

    return graph


def dijkstra(graph: DoorGraph, start_door_id: str, goal_door_id: str) -> Optional[List[str]]:
    """
    Find shortest path using current edge weights.
    Returns list of door IDs or None if no path.
    """
    if start_door_id not in graph.nodes or goal_door_id not in graph.nodes:
        return None

    if start_door_id == goal_door_id:
        return [start_door_id]

    # Priority queue: (cost, door_id)
    pq = [(0.0, start_door_id)]
    distances = {start_door_id: 0.0}
    previous = {start_door_id: None}
    visited = set()

    while pq:
        current_dist, current_id = heapq.heappop(pq)

        if current_id in visited:
            continue

        visited.add(current_id)

        # Found goal
        if current_id == goal_door_id:
            # Reconstruct path
            path = []
            node = goal_door_id
            while node is not None:
                path.append(node)
                node = previous[node]
            return list(reversed(path))

        # Explore neighbors
        for neighbor_id in graph.adjacency[current_id]:
            if neighbor_id in visited:
                continue

            edge_weight = graph.get_weight(current_id, neighbor_id)
            new_dist = current_dist + edge_weight

            if neighbor_id not in distances or new_dist < distances[neighbor_id]:
                distances[neighbor_id] = new_dist
                previous[neighbor_id] = current_id
                heapq.heappush(pq, (new_dist, neighbor_id))

    return None  # No path found


def find_nearest_door(position: str, graph: DoorGraph) -> Optional[str]:
    """Find nearest door/exit to given position"""
    if not graph.nodes:
        return None

    pos = stateNameToCoords(position)
    min_dist = float('inf')
    nearest = None

    for door_id, door_node in graph.nodes.items():
        door_pos = stateNameToCoords(door_node.position)
        dist = ((pos[0] - door_pos[0])**2 + (pos[1] - door_pos[1])**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest = door_id

    return nearest


def find_nearest_exit(position: str, graph: DoorGraph) -> Optional[str]:
    """Find nearest exit to given position"""
    pos = stateNameToCoords(position)
    min_dist = float('inf')
    nearest = None

    for door_id, door_node in graph.nodes.items():
        if door_node.type != 'exit':
            continue
        door_pos = stateNameToCoords(door_node.position)
        dist = ((pos[0] - door_pos[0])**2 + (pos[1] - door_pos[1])**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest = door_id

    return nearest


def find_door_id_by_position(graph: DoorGraph, position: str) -> Optional[str]:
    """
    Find door ID that has the given position.

    Args:
        graph: DoorGraph to search
        position: Position string in "x{col}y{row}" format

    Returns:
        Door ID if found, None otherwise
    """
    for door_id, node in graph.nodes.items():
        if node.position == position:
            return door_id
    return None


def get_connected_nodes(grid, position: str, obstacle_value=-2, door_positions: List[str] = None) -> Tuple[List[Tuple[str, float]], float]:
    """
    Find all doors connected to the given position (in the same room).

    Uses flood-fill BFS to find all reachable doors without crossing obstacles.

    Args:
        grid: 2D array representing the environment
        position: Starting position in "x{col}y{row}" format (e.g., "x15y3")
        obstacle_value: Value in grid that represents obstacles (default: -2 for walls)
        door_positions: Optional list of door position strings to treat as boundaries

    Returns:
        List of tuples (position_string, distance) where:
        - position_string: "x{col}y{row}" format for each connected node
        - distance: BFS distance from starting position (diagonal moves count as 1)

    Example:
        >>> connected = get_connected_nodes(grid, "x10y5")
        >>> print(f"Found {len(connected)} connected cells")
        >>> print(connected[:5])  # First 5 cells with distances
        [('x10y5', 0.0), ('x10y6', 1.0), ('x11y5', 1.0), ('x9y5', 1.0), ('x10y4', 1.0)]
    """
    rows = len(grid)
    cols = len(grid[0])

    # Parse starting position
    start_coords = stateNameToCoords(position)
    start_x, start_y = start_coords

    # Validate starting position
    if not (0 <= start_x < cols and 0 <= start_y < rows):
        return []

    # If starting position is an obstacle, return empty list
    if grid[start_y][start_x] == obstacle_value:
        return []

    # BFS to find all connected cells with distances
    # Queue stores (x, y, distance)
    queue = deque([(start_x, start_y, 0)])
    visited = {(start_x, start_y)}
    connected_nodes = []

    # 8-directional movement (same as used in GridWorld)
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    connected_cells_num = 0
    fire_mean = 0.0

    while queue:
        x, y, dist = queue.popleft()

        connected_cells_num += 1
        fire_mean = (fire_mean * (connected_cells_num - 1) + grid[y][x]) / connected_cells_num if grid[y][x] >= 0 else fire_mean

        # Add current position to results (but don't expand beyond doors)
        if door_positions and f"x{x}y{y}" in door_positions:
            connected_nodes.append((f"x{x}y{y}", float(dist)))
            visited.add((x, y))
            continue  # Skip doors if specified (don't explore beyond them)

        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Bounds check
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue

            # Skip if already visited
            if (nx, ny) in visited:
                continue

            # Skip obstacles
            if grid[ny][nx] == obstacle_value:
                continue

            # Mark as visited and add to queue with incremented distance
            visited.add((nx, ny))
            queue.append((nx, ny, dist + 1))

    return connected_nodes, fire_mean


def update_room_edge_weights(grid, graph: DoorGraph, position: str, estimated_fire_value: float):
    """
    Update edge weights based on fire spread from given position.

    Args:
        grid: 2D fire map
        graph: DoorGraph to update
        position: Position where fire spread occurred
        estimated_fire_value: Additional weight penalty for fire
    """
    # Use cached version for better performance
    connected_nodes, _ = graph.get_connected_nodes_cached(grid, position)

    for i, (pos, dist) in enumerate(connected_nodes):
        door_id = find_door_id_by_position(graph, pos)
        if not door_id:
            continue
        for pos2, dist2 in connected_nodes[i+1:]:
            door_id2 = find_door_id_by_position(graph, pos2)
            if not door_id2:
                continue
            edge = graph.edges.get((door_id, door_id2))
            if not edge:
                continue
            graph.update_edge_weight(door_id, door_id2, estimated_fire_value + edge.base_distance)


def replan_path(graph: DoorGraph, start_pos: str, grid) -> Optional[List[str]]:
    """
    Find shortest path from start_pos to any exit in the door graph.

    Uses hybrid approach:
    - Step 0: Grid-level BFS to find connected doors from start_pos
    - Step 1+: Dijkstra on door graph to find shortest path to any exit

    Args:
        graph: DoorGraph containing doors and exits
        start_pos: Starting position in "x{col}y{row}" format
        grid: 2D array representing the environment

    Returns:
        List of door IDs forming the path to an exit, or None if no exit reachable

    Example:
        >>> path = replan_path(graph, "x10y5", grid)
        >>> print(path)
        ['door1', 'door3', 'exit2']
    """
    # STEP 0: Special initialization - bridge from grid to door graph
    # Check if start position is already at a door
    start_door_id = find_door_id_by_position(graph, start_pos)

    if start_door_id:
        # Start is already at a door
        if graph.nodes[start_door_id].type == 'exit':
            return [start_door_id]  # Already at exit
        # Initialize with this door at distance 0
        pq = [(0.0, start_door_id, [start_door_id])]
    else:
        # Start is not at a door - find connected doors via grid-level BFS
        connected_nodes, _ = graph.get_connected_nodes_cached(grid, start_pos)

        if not connected_nodes:
            print(f"\033[32mNo doors connected to start position {start_pos}\033[0m")
            return None  # No doors reachable from start position
        else:
            print(f"\033[32mFound {len(connected_nodes)} doors connected to start position {start_pos}\033[0m")

        # Initialize priority queue with all connected doors
        pq = []
        print(f"\033[32mConnected nodes:{connected_nodes}\033[0m")
        for door_pos, dist_to_door in connected_nodes:
            door_id = find_door_id_by_position(graph, door_pos)
            print(f"\033[32mConnected door at {door_pos} with ID {door_id} at distance {dist_to_door}\033[0m")
            if door_id:
                heapq.heappush(pq, (dist_to_door, door_id, [door_id]))

        if not pq:
            print(f"\033[32mNo valid doors found in connected nodes for start position {start_pos}\033[0m")
            return None  # No valid doors found in connected nodes

    # STEP 1+: Standard Dijkstra on door graph
    visited = set()

    while pq:
        current_dist, current_door, path = heapq.heappop(pq)

        # Skip if already visited
        if current_door in visited:
            continue
        visited.add(current_door)

        # Goal check: Is this an exit?
        if graph.nodes[current_door].type == 'exit':
            return path  # Found shortest path to an exit!

        # Explore neighbors via door graph edges
        for neighbor_door in graph.adjacency[current_door]:
            if neighbor_door in visited:
                continue

            # Get edge weight (may be dynamically updated due to fire)
            edge_weight = graph.get_weight(current_door, neighbor_door)

            # Skip if edge is blocked (infinite weight)
            if edge_weight == float('inf'):
                continue

            new_dist = current_dist + edge_weight
            new_path = path + [neighbor_door]

            heapq.heappush(pq, (new_dist, neighbor_door, new_path))

    print(f"\033[32mNo exit reachable from start position {start_pos}\033[0m")
    return None  # No exit reachable