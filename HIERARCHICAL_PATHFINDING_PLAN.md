# Hierarchical Pathfinding Implementation Plan

## Overview
Implement a two-tier pathfinding system where agents use:
1. **High-level planning**: Door/exit graph with Dijkstra's algorithm
2. **Low-level navigation**: Existing D* Lite for cell-level movement

**Key Design Principles:**
- Static graph topology (computed once at startup)
- Lazy weight updates (only when entering new rooms)
- Human-like decision making (reassess at doorways, not every step)

---

## 1. NEW FILE: `door_graph.py`

### 1.1 Data Structures

```python
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
    door_a: str  # door id
    door_b: str  # door id
    base_distance: float  # initial BFS distance
    current_weight: float  # dynamically updated based on fire

class DoorGraph:
    """Graph of doors and exits with connectivity"""

    def __init__(self):
        self.nodes: Dict[str, DoorNode] = {}
        self.edges: Dict[Tuple[str, str], DoorEdge] = {}
        self.adjacency: Dict[str, List[str]] = {}  # door_id -> [neighbor_ids]

    def add_node(self, node: DoorNode):
        """Add a door/exit node to the graph"""
        self.nodes[node.id] = node
        self.adjacency[node.id] = []

    def add_edge(self, door_a_id: str, door_b_id: str, distance: float):
        """Add bidirectional edge between two doors"""
        edge = DoorEdge(door_a_id, door_b_id, distance, distance)
        self.edges[(door_a_id, door_b_id)] = edge
        self.edges[(door_b_id, door_a_id)] = edge
        self.adjacency[door_a_id].append(door_b_id)
        self.adjacency[door_b_id].append(door_a_id)

    def get_weight(self, door_a: str, door_b: str) -> float:
        """Get current weight of edge"""
        edge = self.edges.get((door_a, door_b))
        return edge.current_weight if edge else float('inf')

    def update_edge_weight(self, door_a: str, door_b: str, new_weight: float):
        """Update weight for both directions"""
        if (door_a, door_b) in self.edges:
            self.edges[(door_a, door_b)].current_weight = new_weight
        if (door_b, door_a) in self.edges:
            self.edges[(door_b, door_a)].current_weight = new_weight
```

### 1.2 BFS Connectivity Algorithm

```python
def bfs_with_blocked_cells(grid, start_pos, goal_pos, blocked_positions):
    """
    BFS that treats blocked_positions as obstacles.
    Returns path length or None if unreachable.

    Args:
        grid: 2D fire map array
        start_pos: (x, y) tuple
        goal_pos: (x, y) tuple
        blocked_positions: list of (x, y) tuples to treat as walls
    """
    if start_pos == goal_pos:
        return 0

    rows = len(grid)
    cols = len(grid[0])
    blocked_set = set(blocked_positions)

    queue = deque([(start_pos[0], start_pos[1], 0)])  # (x, y, distance)
    visited = {start_pos}

    # 8-directional movement
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    while queue:
        x, y, dist = queue.popleft()

        # Check all neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Bounds check
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue

            # Skip visited
            if (nx, ny) in visited:
                continue

            # Skip walls and blocked doors
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
    Compute which doors can reach each other in the same room.
    Two doors are connected if you can walk between them without passing other doors.

    Returns: list of (door_a_id, door_b_id, distance) tuples
    """
    edges = []
    door_positions = {d.id: stateNameToCoords(d.position) for d in door_nodes}

    for i, door_i in enumerate(door_nodes):
        for door_j in door_nodes[i+1:]:
            # Get positions
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

    print(f"[DoorGraph] Built graph with {len(graph.nodes)} nodes and {len(edges)} edges")

    return graph
```

### 1.3 Dijkstra Pathfinding

```python
def dijkstra(graph: DoorGraph, start_door_id: str, goal_door_id: str) -> Optional[List[str]]:
    """
    Find shortest path from start to goal using current edge weights.

    Returns: list of door IDs to visit, or None if no path
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
```

### 1.4 Utility Functions

```python
def find_nearest_door(position: str, graph: DoorGraph) -> str:
    """Find nearest door/exit to given position"""
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

def find_nearest_exit(position: str, graph: DoorGraph) -> str:
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
```

---

## 2. MODIFICATIONS: `simulation.py`

### 2.1 Import New Module

**Location:** Line 36 (after other imports)

```python
from door_graph import DoorGraph, build_door_graph, dijkstra, find_nearest_door, find_nearest_exit
```

### 2.2 Update SimulationConfig

**Location:** Line 52-67

**Add new field:**
```python
@dataclass
class SimulationConfig:
    # ... existing fields ...
    doors_and_exits: list[dict] = None  # NEW: List of door/exit configs
```

**Update from_json method (line 69-95):**
```python
@classmethod
def from_json(cls, json_data):
    config = cls(
        # ... existing fields ...
        doors_and_exits=json_data.get('doors_and_exits', [])  # NEW
    )
    return config
```

### 2.3 Update EvacuationAgent Class

**Location:** Line 97-268

**Add to __init__ (line 98):**
```python
def __init__(self, id: int, start:str, occupancy, max_occupancy, map_rows, map_cols,
             targets: list[str], viewing_range=5, fire_fearness=1.0,
             door_graph=None):  # NEW parameter
    # ... existing initialization ...

    # NEW: Hierarchical navigation
    self.door_graph = door_graph
    self.planned_door_route = []  # List of door IDs to visit
    self.current_door_idx = 0
    self.last_room_update_door = None  # Track when we last updated weights
    self.using_hierarchical = door_graph is not None and len(door_graph.nodes) > 0
```

**Add new method after __init__:**
```python
def plan_route_to_exit(self, exit_target):
    """Plan high-level route through doors to reach exit"""
    if not self.using_hierarchical:
        return

    nearest_door = find_nearest_door(self.s_current, self.door_graph)
    nearest_exit = find_nearest_exit(exit_target, self.door_graph)

    if nearest_door and nearest_exit:
        self.planned_door_route = dijkstra(self.door_graph, nearest_door, nearest_exit)
        if self.planned_door_route:
            print(f"Agent {self.id}: Planned route {' -> '.join(self.planned_door_route)}")
        else:
            print(f"Agent {self.id}: No route found, using direct navigation")
            self.using_hierarchical = False
    else:
        self.using_hierarchical = False

def on_enter_room(self, door_id):
    """Called when agent passes through a door - update weights and replan if needed"""
    if door_id == self.last_room_update_door:
        return  # Already updated from this door

    print(f"Agent {self.id}: Entering room via door {door_id}")

    # Update weights for all edges from this door
    for neighbor_id in self.door_graph.adjacency[door_id]:
        # Sample path between doors
        path = self.sample_path_between_doors(door_id, neighbor_id)
        danger = self.evaluate_path_danger(path)

        # Update weight
        edge = self.door_graph.edges.get((door_id, neighbor_id))
        if edge:
            base = edge.base_distance
            new_weight = base * (1.0 + danger * 10.0)  # Scale danger

            # Block if too dangerous
            if danger > 0.5:  # Threshold
                new_weight = float('inf')

            self.door_graph.update_edge_weight(door_id, neighbor_id, new_weight)

    # Replan if current route is now blocked
    if self.should_replan(door_id):
        print(f"Agent {self.id}: Replanning due to fire")
        self.replan_from_door(door_id)

    self.last_room_update_door = door_id

def sample_path_between_doors(self, door_a_id, door_b_id):
    """Get representative path cells between two doors"""
    pos_a = stateNameToCoords(self.door_graph.nodes[door_a_id].position)
    pos_b = stateNameToCoords(self.door_graph.nodes[door_b_id].position)

    # Bresenham line algorithm for sampling
    cells = []
    x0, y0 = pos_a
    x1, y1 = pos_b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return cells

def evaluate_path_danger(self, path):
    """Evaluate fire danger along a path"""
    if not path:
        return 0.0

    total_danger = 0.0
    for (x, y) in path:
        if 0 <= y < len(self.graph.cells) and 0 <= x < len(self.graph.cells[y]):
            cell_value = self.graph.cells[y][x]
            if cell_value > 0:  # Fire present
                total_danger += cell_value
            elif cell_value == -1:  # Blocked
                return 1.0  # Maximum danger

    return total_danger / len(path)

def should_replan(self, current_door_id):
    """Check if we need to replan route"""
    if self.current_door_idx >= len(self.planned_door_route) - 1:
        return False

    next_door = self.planned_door_route[self.current_door_idx + 1]
    weight = self.door_graph.get_weight(current_door_id, next_door)

    return weight == float('inf')

def replan_from_door(self, door_id):
    """Replan route from current door"""
    # Find where we are in the route
    try:
        current_idx = self.planned_door_route.index(door_id)
        self.current_door_idx = current_idx
    except ValueError:
        current_idx = self.current_door_idx

    # Get goal (last door in route should be exit)
    goal_door = self.planned_door_route[-1]

    # Replan
    new_route = dijkstra(self.door_graph, door_id, goal_door)
    if new_route:
        self.planned_door_route = new_route
        self.current_door_idx = 0
        print(f"Agent {self.id}: New route {' -> '.join(new_route)}")
    else:
        print(f"Agent {self.id}: No alternative route found!")
```

**Modify move() method (line 185-250):**

```python
def move(self):
    coord_current = stateNameToCoords(self.s_current)
    if(self.graph.cells[coord_current[1]][coord_current[0]] < 0):
        print(f"Warning: Agent {self.id} starting on an obstacle at {self.s_current}!")
        return 'stuck'
    else:
        self.fire_damage += self.graph.cells[coord_current[1]][coord_current[0]]

    try:
        # NEW: Check if using hierarchical navigation
        if self.using_hierarchical and self.planned_door_route:
            # Check if we reached a door
            if self.current_door_idx < len(self.planned_door_route):
                target_door_id = self.planned_door_route[self.current_door_idx]
                target_door_pos = self.door_graph.nodes[target_door_id].position

                # Check if we reached this door
                if self.s_current == target_door_pos:
                    self.on_enter_room(target_door_id)
                    self.current_door_idx += 1

                    # Check if reached final exit
                    if self.current_door_idx >= len(self.planned_door_route):
                        if self.door_graph.nodes[target_door_id].type == 'exit':
                            coord_current = stateNameToCoords(self.s_current)
                            self.occupancy[coord_current[1]][coord_current[0]] -= 1
                            return 'Evacuated'

                    # Set next door as goal
                    if self.current_door_idx < len(self.planned_door_route):
                        next_door_pos = self.door_graph.nodes[
                            self.planned_door_route[self.current_door_idx]
                        ].position
                        self.graph.setGoal(next_door_pos)
                        self.queue, self.k_m = self.graph.reset_for_new_planning()
                        self.graph, self.queue, self.k_m = initDStarLite(
                            self.graph, self.queue, self.s_current, next_door_pos, self.k_m
                        )

        # Continue with normal D* Lite movement (existing code)
        if self.s_current in self.position_history[-3:]:
            try:
                scanForObstacles(self.graph, self.queue, self.s_current, self.VIEWING_RANGE * 2, self.k_m)
                computeShortestPath(self.graph, self.queue, self.s_current, self.k_m)
            except Exception as e:
                print(f"Warning: Failed to perform cycle detection rescan for agent {self.id}: {e}")

        try:
            self.s_new, self.k_m = moveAndRescan(self.graph, self.queue, self.s_current,
                                                  self.VIEWING_RANGE, self.k_m,
                                                  self.occupancy, self.max_occupancy)
        except Exception as e:
            print(f"Error: Failed to move agent {self.id} from {self.s_current}: {e}")
            return f'Movement Error: {e}'

        self.position_history.append(self.s_current)
        if len(self.position_history) > 5:
            self.position_history.pop(0)

        # Handle old target system (for non-hierarchical mode)
        if not self.using_hierarchical:
            if self.s_new == 'goal' or self.s_new == self.target:
                # ... existing target handling code ...
                pass

        if self.s_new == 'stuck':
            self.occupancy[coord_current[1]][coord_current[0]] -= 1
            return 'stuck'
        else:
            coord_current = stateNameToCoords(self.s_current)
            self.occupancy[coord_current[1]][coord_current[0]] -= 1
            coord_new = stateNameToCoords(self.s_new)
            self.occupancy[coord_new[1]][coord_new[0]] += 1
            self.s_current = self.s_new
            return None

    except Exception as e:
        print(f"Critical error in move() for agent {self.id}: {e}")
        return f'Critical Movement Error: {e}'
```

### 2.4 Update EvacuationSimulation Class

**Location:** Line 269-564

**Modify __init__ (line 270-324):**

```python
def __init__(self, config: SimulationConfig):
    # ... existing initialization ...

    # NEW: Build door graph if doors are configured
    self.door_graph = None
    if config.doors_and_exits and len(config.doors_and_exits) > 0:
        print("Building door/exit graph...")
        self.door_graph = build_door_graph(
            config.initial_fire_map,
            config.doors_and_exits
        )

    # Initialize agents
    self.agents = []
    for i in range(self.agent_num):
        start_pos = config.start_positions[i]
        fearness = 1.0
        if config.agent_fearness and i < len(config.agent_fearness):
            fearness = config.agent_fearness[i]

        try:
            agent = EvacuationAgent(
                i, start_pos, self.occupancy, self.max_occupancy,
                self.map_rows, self.map_cols, self.targets,
                self.viewing_range, fire_fearness=fearness,
                door_graph=self.door_graph  # NEW: Pass door graph
            )

            # NEW: Plan initial route if using hierarchical navigation
            if agent.using_hierarchical:
                agent.plan_route_to_exit(self.targets[-1])  # Use last target as exit

            self.agents.append(agent)
        except Exception as e:
            print(f"Failed to initialize agent {i}: {e}")
            raise
```

---

## 3. MODIFICATIONS: `pygame_visualizer.py`

### 3.1 Add Door/Exit Colors

**Location:** Line 60-73 (colors dict)

```python
self.colors = {
    # ... existing colors ...
    'door': (139, 69, 19),  # Brown for doors
    'exit': (0, 255, 0),    # Bright green for exits
}
```

### 3.2 Add draw_doors_exits Method

**Location:** After draw_targets() method (line 169)

```python
def draw_doors_exits(self, door_graph):
    """Draw doors and exits on the grid"""
    if not door_graph:
        return

    for door_id, door_node in door_graph.nodes.items():
        try:
            coords = stateNameToCoords(door_node.position)
            if 0 <= coords[0] < self.map_cols and 0 <= coords[1] < self.map_rows:
                pixel_x, pixel_y = self.coord_to_pixel(coords[0], coords[1])

                # Choose color and symbol based on type
                if door_node.type == 'exit':
                    color = self.colors['exit']
                    symbol = "EXIT"
                else:
                    color = self.colors['door']
                    symbol = "D"

                # Draw as square
                rect_size = self.cell_size // 2
                rect = pygame.Rect(
                    pixel_x - rect_size // 2,
                    pixel_y - rect_size // 2,
                    rect_size,
                    rect_size
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)  # Black border

                # Draw label
                text = self.small_font.render(symbol, True, (255, 255, 255))
                text_rect = text.get_rect(center=(pixel_x, pixel_y))
                self.screen.blit(text, text_rect)
        except:
            pass
```

### 3.3 Update update_display Method

**Location:** Line 260-268

```python
def update_display(self, step, agents, targets, status, reached_targets=None, door_graph=None):
    """Update the entire display"""
    self.draw_grid(agents)
    self.draw_doors_exits(door_graph)  # NEW: Draw doors/exits
    self.draw_targets(targets, reached_targets)
    self.draw_agents(agents)
    self.draw_info_panel(step, status, agents)

    pygame.display.flip()
    return self.handle_events()
```

### 3.4 Update Calls in simulation.py

**Location:** simulation.py lines 486, 507, 539, 542

**Change from:**
```python
visualizer.update_display(self.steps, self.agents, self.targets, status)
```

**To:**
```python
visualizer.update_display(self.steps, self.agents, self.targets, status,
                          reached_targets, self.door_graph)
```

---

## 4. MODIFICATIONS: `visual_configurator.py`

### 4.1 Add Door/Exit Lists

**Location:** Line 26-31 (after targets list)

```python
# Lists for positions and targets
self.start_positions = []
self.targets = []
self.fire_positions = []
self.obstacle_positions = []
self.doors = []  # NEW: List of {"id": "d1", "position": "x5y5", "type": "door"}
self.exits = []  # NEW: List of {"id": "exit1", "position": "x50y50", "type": "exit"}
```

### 4.2 Add Door/Exit to Tool Selection

**Location:** Line 186 (tools list)

```python
tools = [
    ("Agent Start", "agent"),
    ("Target", "target"),
    ("Fire", "fire"),
    ("Obstacle", "obstacle"),
    ("Door", "door"),      # NEW
    ("Exit", "exit"),      # NEW
    ("Erase", "erase")
]
```

### 4.3 Add Door/Exit Section to UI

**Location:** After Fire Configuration Section (line 151)

```python
# Doors/Exits Configuration Section
doors_frame = ttk.LabelFrame(scrollable_frame, text="Doors and Exits", padding="10")
doors_frame.pack(fill=tk.X, pady=(0, 10))

door_control_frame = ttk.Frame(doors_frame)
door_control_frame.pack(fill=tk.X, pady=(0, 5))

ttk.Button(door_control_frame, text="Clear All Doors/Exits",
           command=self.clear_doors_exits).pack(side=tk.LEFT, padx=(0, 5))

self.doors_exits_listbox = tk.Listbox(doors_frame, height=4)
self.doors_exits_listbox.pack(fill=tk.X, pady=5)
```

### 4.4 Handle Door/Exit Clicks

**Location:** Line 280-324 (handle_cell_click method)

**Add after fire handling:**
```python
elif self.current_tool == "door":
    # Add door
    door_id = f"d{len(self.doors) + 1}"
    door = {"id": door_id, "position": position, "type": "door"}
    self.doors.append(door)
    self.refresh_doors_exits_listbox()

elif self.current_tool == "exit":
    # Add exit
    exit_id = f"exit{len(self.exits) + 1}"
    exit_obj = {"id": exit_id, "position": position, "type": "exit"}
    self.exits.append(exit_obj)
    self.refresh_doors_exits_listbox()

elif self.current_tool == "erase":
    # Remove any items at this position
    self.start_positions = [pos for pos in self.start_positions if pos != position]
    self.targets = [pos for pos in self.targets if pos != position]
    self.fire_positions = [f for f in self.fire_positions if f[0] != position]
    self.obstacle_positions = [pos for pos in self.obstacle_positions if pos != position]
    self.doors = [d for d in self.doors if d['position'] != position]  # NEW
    self.exits = [e for e in self.exits if e['position'] != position]  # NEW

    self.refresh_all_listboxes()
    self.config_vars['agent_num'].set(len(self.start_positions))
```

### 4.5 Add Helper Methods

**Location:** After refresh_fire_listbox (line 346)

```python
def refresh_doors_exits_listbox(self):
    """Refresh doors/exits listbox"""
    self.doors_exits_listbox.delete(0, tk.END)
    for door in self.doors:
        self.doors_exits_listbox.insert(tk.END, f"Door {door['id']}: {door['position']}")
    for exit_obj in self.exits:
        self.doors_exits_listbox.insert(tk.END, f"Exit {exit_obj['id']}: {exit_obj['position']}")

def clear_doors_exits(self):
    """Clear all doors and exits"""
    self.doors.clear()
    self.exits.clear()
    self.refresh_doors_exits_listbox()
    self.refresh_map()
```

### 4.6 Update clear_map Method

**Location:** Line 347-356

```python
def clear_map(self):
    """Clear all items from the map"""
    self.start_positions.clear()
    self.targets.clear()
    self.fire_positions.clear()
    self.obstacle_positions.clear()
    self.doors.clear()         # NEW
    self.exits.clear()         # NEW
    self.config_vars['agent_num'].set(0)
    self.refresh_all_listboxes()
    self.refresh_doors_exits_listbox()  # NEW
    self.create_initial_walls()
    self.refresh_map()
```

### 4.7 Update refresh_map Method

**Location:** Line 375-418

**Add after drawing obstacles (line 417):**
```python
# Draw doors
for door in self.doors:
    self.draw_cell(door['position'], "brown", "🚪")

# Draw exits
for exit_obj in self.exits:
    self.draw_cell(exit_obj['position'], "lightgreen", "🚪")
```

### 4.8 Update save_config Method

**Location:** Line 568-625

**Add to config_data dict (line 599):**
```python
config_data = {
    # ... existing fields ...
    'doors_and_exits': self.doors + self.exits  # NEW: Combine into single list
}
```

### 4.9 Update load_config Method

**Location:** Line 627-673

**Add after loading fire (line 668):**
```python
# Load doors and exits
self.doors.clear()
self.exits.clear()
doors_exits = config_data.get('doors_and_exits', [])
for item in doors_exits:
    if item['type'] == 'door':
        self.doors.append(item)
    elif item['type'] == 'exit':
        self.exits.append(item)
self.refresh_doors_exits_listbox()
```

---

## 5. UPDATE: `example_configuration.json`

**Location:** Add new field to configuration

```json
{
  "_comment": "Fire Evacuation Simulation Configuration",
  "map_rows": 60,
  "map_cols": 60,

  "doors_and_exits": [
    {"id": "d1", "position": "x15y15", "type": "door"},
    {"id": "d2", "position": "x30y30", "type": "door"},
    {"id": "d3", "position": "x15y45", "type": "door"},
    {"id": "exit1", "position": "x6y6", "type": "exit"},
    {"id": "exit2", "position": "x51y51", "type": "exit"}
  ],

  "max_occupancy": 1,
  "start_positions": [...],
  "targets": [...],
  "initial_fire_map": [[...]]
}
```

---

## 6. IMPLEMENTATION SEQUENCE

### Phase 1: Core Infrastructure (30-60 min)
1. ✅ Create `door_graph.py` with data structures
2. ✅ Implement BFS connectivity algorithm
3. ✅ Implement Dijkstra pathfinding
4. ✅ Add utility functions (find_nearest_door, etc.)
5. ✅ Test graph building with simple example

### Phase 2: Simulation Integration (45-90 min)
1. ✅ Update SimulationConfig to accept doors_and_exits
2. ✅ Modify EvacuationAgent.__init__ to accept door_graph
3. ✅ Add hierarchical planning methods to agent
4. ✅ Modify agent.move() to use door waypoints
5. ✅ Update EvacuationSimulation.__init__ to build graph
6. ✅ Test with simple 2-door scenario

### Phase 3: Visualization (30-45 min)
1. ✅ Update pygame_visualizer.py to draw doors/exits
2. ✅ Update visual_configurator.py to support door placement
3. ✅ Test visual configurator door placement
4. ✅ Update example_configuration.json with doors

### Phase 4: Testing & Refinement (60-120 min)
1. ✅ Test connectivity algorithm correctness
2. ✅ Test weight updates when entering rooms
3. ✅ Test replanning when doors blocked by fire
4. ✅ Compare performance: hierarchical vs. direct navigation
5. ✅ Tune danger thresholds and weight formulas

---

## 7. TESTING SCENARIOS

### Test 1: Two-Room Scenario
```
Room 1 (left): Agent starts here
Door 1: Connection between rooms
Room 2 (right): Exit here
Fire: Starts in Room 1

Expected: Agent plans route through Door1 to Exit,
          updates weights in Room1, continues through Door1
```

### Test 2: Fire Blocks Planned Route
```
Room layout: A -> Door1 -> B -> Door2 -> Exit
                      ↓
                    Door3 -> C -> Exit2

Initial plan: A -> Door1 -> Door2 -> Exit
Fire blocks Door1
Expected: Replan to A -> Door3 -> Exit2
```

### Test 3: No Doors (Backward Compatibility)
```
Config has no doors_and_exits field
Expected: Agent uses existing D* Lite to navigate to targets
```

---

## 8. PERFORMANCE CONSIDERATIONS

1. **Graph Building**: O(N²) BFS for N doors - acceptable for <50 doors
2. **Dijkstra**: O(E log V) per planning - fast for <100 doors
3. **Weight Updates**: Only at room entry - minimal overhead
4. **Memory**: Door graph << grid graph, negligible impact

---

## 9. FUTURE ENHANCEMENTS (Optional)

1. **Automatic Door Detection**: Heuristic to find narrow passages
2. **Dynamic Edge Creation**: Handle doors blocked by debris
3. **Multi-floor Support**: Stairs as special door types
4. **Cooperative Planning**: Agents share weight updates
5. **Risk-aware Planning**: Balance distance vs. fire danger explicitly

---

## END OF PLAN
