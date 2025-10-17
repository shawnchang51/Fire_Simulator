"""
Door/Exit Graph for Hierarchical Pathfinding
============================================
Implements two-tier navigation system:
- High-level: Door/exit graph (this file)
- Low-level: D* Lite on grid (existing system)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
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