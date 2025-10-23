"""
Fire Evacuation Simulation with D* Lite Pathfinding
====================================================

SPATIAL/TEMPORAL MODEL ASSUMPTIONS:
- Default cell size: 0.3m × 0.3m (configurable via cell_size parameter)
- Default timestep: 0.5 seconds (configurable via timestep_duration parameter)
- Agent movement: 1 cell per timestep (0.6 m/s at default settings)
- Fire updates: Configurable interval (default every 4 timesteps = 2 seconds for realistic model)

COORDINATE SYSTEM:
- Grid positions: "x{col}y{row}" format (e.g., "x12y9")
- Grid access: grid[row][col] or grid[y][x]

FIRE MODEL OPTIONS:
- "realistic": Physics-aligned model (3-6 min to flashover, 0.1-0.2 m/s spread)
- "aggressive": Stress-testing model (30-60 sec to flashover, 0.3-0.5 m/s spread)
- "default": Original model from fire_model_float.py

Author: Fire Evacuation Simulation System
"""

import pretty_errors
import time
import os
from d_star_lite.grid import GridWorld
from d_star_lite.utils import stateNameToCoords
from d_star_lite.d_star_lite import initDStarLite, moveAndRescan, set_OBS_VAL, scanForObstacles, computeShortestPath
from dataclasses import dataclass
from door_graph import build_door_graph, replan_path, find_door_id_by_position, update_room_edge_weights, DoorGraph
import json
from fire_monitor import FireMonitor
from datetime import datetime
import argparse
import copy
import numpy as np

try:
    from pygame_visualizer import EvacuationVisualizer
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Pygame not available. Run 'pip install pygame' to enable graphical visualization.")

try:
    from matlab_visualizer import create_matlab_visualizer
    MATLAB_VISUALIZER_AVAILABLE = True
except ImportError:
    MATLAB_VISUALIZER_AVAILABLE = False
    print("MATLAB-style visualizer not available. Install scipy: pip install scipy")


@dataclass
class SimulationConfig:
    map_rows: int
    map_cols: int
    max_occupancy: int
    start_positions: list[str]
    initial_fire_map: list[list[float]]
    agent_num: int
    viewing_range: int = 5
    # New spatial/temporal parameters
    cell_size: float = 0.3  # meters per cell
    timestep_duration: float = 0.5  # seconds per timestep
    fire_update_interval: int = 4  # update fire every N timesteps (default: 4 timesteps = 2 seconds)
    fire_model_type: str = "realistic"  # "realistic", "aggressive", or "default"
    agent_fearness: list[float] = None  # fearness multiplier per agent (default: 1.0 for all)
    door_configs: list[dict] = None  # Optional door configurations
    consider_env_factors: bool = False  # Whether agents consider temperature and smoke in pathfinding
    wall_preference: float = 0.0  # Wall-following preference: 0=no preference, higher values=stronger wall-following
    # Knowledge sharing parameters
    communication_range: float = 15.0  # Distance in cells within which agents can share door graph knowledge
    sharing_interval: int = 5  # Share knowledge every N timesteps (default: 5 timesteps = 2.5 seconds)
    sector_size: int = None  # Size of spatial index sectors (auto-calculated if None)

    @classmethod
    def from_json(cls, json_data):
        config = cls(
            map_rows=json_data['map_rows'],
            map_cols=json_data['map_cols'],
            max_occupancy=json_data['max_occupancy'],
            start_positions=json_data['start_positions'],
            initial_fire_map=json_data.get('initial_fire_map', [[0 for _ in range(json_data['map_cols'])] for _ in range(json_data['map_rows'])]),
            agent_num=json_data['agent_num'],
            viewing_range=json_data.get('viewing_range', 5),
            cell_size=json_data.get('cell_size', 0.3),
            timestep_duration=json_data.get('timestep_duration', 0.5),
            fire_update_interval=json_data.get('fire_update_interval', 4),
            fire_model_type=json_data.get('fire_model_type', 'realistic'),
            agent_fearness=json_data.get('agent_fearness', []),
            door_configs=json_data.get('door_configs', []),
            consider_env_factors=json_data.get('consider_env_factors', False),
            wall_preference=json_data.get('wall_preference', 0.0),
            communication_range=json_data.get('communication_range', 15.0),
            sharing_interval=json_data.get('sharing_interval', 5),
            sector_size=json_data.get('sector_size', None)
        )

        # Auto-scale viewing_range based on cell_size if it seems too small
        # Assume 3m is a reasonable viewing distance for fire/smoke conditions
        if config.cell_size < 0.5 and config.viewing_range < 8:
            recommended_range = int(3.0 / config.cell_size)
            if config.viewing_range < recommended_range:
                print(f"Info: Auto-scaling viewing_range from {config.viewing_range} to {recommended_range} based on cell_size={config.cell_size}m")
                config.viewing_range = recommended_range

        return config

class EvacuationAgent():
    def __init__(self, id: int, start:str, occupancy, max_occupancy, map_rows, map_cols, viewing_range=10, fire_fearness=1.0, base_door_graph: DoorGraph=None, fire_model=None, consider_env_factors=False, wall_distance_map=None, wall_preference=0.0, initial_fire_map=None):
        try:
            self.id = id
            self.start = start
            self.occupancy = occupancy
            self.max_occupancy = max_occupancy
            self.VIEWING_RANGE = viewing_range
            self.fire_damage = 0
            self.fire_fearness = fire_fearness
            self.door_graph = copy.deepcopy(base_door_graph)
            self.door_path = []
            self.average_temp = 0.0
            self.peak_temp = 0.0
            self.total_steps = 0
            self.fire_model = fire_model
            self.consider_env_factors = consider_env_factors
            self.wall_distance_map = wall_distance_map
            self.wall_preference = wall_preference
            self.initial_fire_map = initial_fire_map

            try:
                self.graph = GridWorld(map_cols, map_rows, fire_fearness=fire_fearness)
            except Exception as e:
                raise ValueError(f"Agent {id}: Failed to create GridWorld with dimensions {map_cols}x{map_rows}: {e}")

            path = replan_path(self.door_graph, start, self.initial_fire_map if self.initial_fire_map is not None else self.graph.cells)
            # print(f"\033[31mAgent {id} initial door graph: {self.door_graph}\033[0m")
            # print(f"\033[31mAgent {id} initial door path from {start}: {path}\033[0m")
            if path is None:
                raise ValueError(f"Agent {id}: No valid door path found from start position {start}")

            self.targetidx = 0
            self.targets = [self.door_graph.nodes[node].position for node in path]
            self.target = self.targets[self.targetidx]
            # door_path will be populated as agent reaches each door/exit

            try:
                start_coords = stateNameToCoords(start)
                self.occupancy[start_coords[1]][start_coords[0]] += 1
            except Exception as e:
                raise ValueError(f"Agent {id}: Invalid start position '{start}': {e}")

            # Validate coordinates are within bounds
            if (start_coords[1] < 0 or start_coords[1] >= map_rows or
                start_coords[0] < 0 or start_coords[0] >= map_cols):
                raise ValueError(f"Agent {id}: Start position {start} ({start_coords}) is out of bounds for {map_rows}x{map_cols} grid")

            try:
                self.graph.cells[start_coords[1]][start_coords[0]] = 0
            except IndexError as e:
                raise ValueError(f"Agent {id}: Cannot access grid cell at {start_coords}: {e}")

            self.k_m = 0
            self.queue = []
            self.queue_set = set()  # Track queue membership for O(1) lookups

            try:
                # Set start and goal positions on the graph before initializing D* Lite
                self.graph.setStart(start)
                self.graph.setGoal(self.target)
                self.graph, self.queue, self.k_m, self.queue_set = initDStarLite(self.graph, self.queue, start, self.target, self.k_m)
            except Exception as e:
                raise ValueError(f"Agent {id}: Failed to initialize D* Lite from {start} to {self.target}: {e}")

            self.s_current = start
            self.position_history = []

        except Exception as e:
            print(f"Critical error initializing agent {id}: {e}")
            raise

    def estimate_fire(self, current_location=None, fire_fearness=None):
        _, fire_mean = self.door_graph.get_connected_nodes_cached(self.graph.cells, current_location if current_location is not None else self.s_current)
        return fire_mean*fire_fearness if fire_fearness is not None else fire_mean*self.fire_fearness

    def calculate_environmental_cost(self, row, col):
        """
        Calculate combined environmental cost considering fire intensity, temperature, smoke, and wall preference.
        Returns a cost value that will be used to update the graph cells.
        """
        # Get base fire intensity from current cell value
        base_fire = max(0, self.graph.cells[row][col]) if self.graph.cells[row][col] >= 0 else self.graph.cells[row][col]

        # Start with base cost
        combined_cost = base_fire

        # Add wall preference cost (always applied if wall_preference > 0)
        if self.wall_preference > 0 and self.wall_distance_map is not None and base_fire >= 0:
            state = f"x{col}y{row}"
            wall_dist = self.wall_distance_map.get(state, 0)
            wall_penalty = max(0, wall_dist - 1) * self.wall_preference
            combined_cost += wall_penalty

        # Add environmental factors if enabled
        if not self.consider_env_factors or self.fire_model is None:
            # Return early if not considering environmental factors
            return combined_cost if base_fire >= 0 else base_fire

        # Get temperature and smoke from fire model
        try:
            temperature = self.fire_model.temperature_map[row][col]
            smoke = self.fire_model.smoke_density[row][col]
        except (AttributeError, IndexError):
            # If fire model doesn't have these attributes, just return current cost
            return combined_cost if base_fire >= 0 else base_fire

        # Temperature contribution: normalize temperature above ambient (20°C)
        # High temps (>100°C) should significantly increase cost
        temp_cost = 0.0
        if temperature > 20.0:
            # Scale: 20-100°C -> 0-1.6, 100-200°C -> 1.6-3.6, etc.
            temp_cost = (temperature - 20.0) / 50.0

        # Smoke contribution: smoke density is typically 0-1 range
        # High smoke reduces visibility and breathability
        smoke_cost = smoke * 2.0  # Amplify smoke impact

        # Add temperature and smoke to combined cost
        # If cell is an obstacle (negative value), preserve it
        if base_fire < 0:
            return base_fire  # Keep obstacles as-is
        else:
            combined_cost += temp_cost + smoke_cost
            return combined_cost

    def update_lazy_graph(self, current_location=None):
        try:
            fire_estimate = self.estimate_fire(current_location, self.fire_fearness)
            update_room_edge_weights(self.graph.cells, self.door_graph, current_location, fire_estimate)
        except Exception as e:
            print(f"Error: Agent {self.id} failed to update lazy graph at {self.s_current}: {e}")
            raise
    
    def share_with_nearby(self, nearby_agents):
        """
        Share door graph knowledge with nearby agents.

        Uses conservative merging strategy: takes maximum edge weight to represent
        the most cautious estimate of path danger.

        Args:
            nearby_agents (list): List of EvacuationAgent objects within communication range
        """
        if self.door_graph is None:
            return

        for other_agent in nearby_agents:
            # Skip self (all agents in list are active - simulation removes inactive ones)
            if other_agent.id == self.id:
                continue

            # Skip agents without door graphs
            if other_agent.door_graph is None:
                continue

            # Merge knowledge bidirectionally
            # print(f"merging graph of agent {self.id} and agent {other_agent.id}")
            self._merge_door_graph_edges(other_agent.door_graph)
            other_agent._merge_door_graph_edges(self.door_graph)

    def _merge_door_graph_edges(self, other_door_graph):
        """
        Merge edge weights from another agent's door graph into this agent's graph.

        Uses conservative strategy: takes maximum weight (most dangerous observed path)
        to ensure agents avoid known hazards.

        Args:
            other_door_graph (DoorGraph): Another agent's door graph to merge from
        """
        if self.door_graph is None or other_door_graph is None:
            return

        # Merge edge weights
        for node_id in self.door_graph.nodes:
            if node_id not in other_door_graph.edges:
                continue

            for neighbor_id, other_weight in other_door_graph.edges[node_id].items():
                if neighbor_id not in self.door_graph.edges[node_id]:
                    # New edge discovered by other agent
                    self.door_graph.edges[node_id][neighbor_id] = other_weight
                else:
                    # Take maximum weight (conservative approach)
                    current_weight = self.door_graph.edges[node_id][neighbor_id]
                    self.door_graph.edges[node_id][neighbor_id] = max(current_weight, other_weight)

        # Invalidate cache after merging new knowledge
        self.door_graph.clear_cache()

    def replan_door_path(self, current_location):
        try:
            path = replan_path(self.door_graph, current_location, self.graph.cells)
            if path is None:
                print(f"Warning: Agent {self.id} could not find a new door path from {current_location}")
                return 'No Door Path'

            self.targets = [self.door_graph.nodes[node].position for node in path]
            self.targetidx = 0
            self.target = self.targets[self.targetidx]

            try:
                self.graph.setStart(self.s_current)
                self.graph.setGoal(self.target)
                # Reinitialize D* Lite for the new target
                self.queue, self.k_m = self.graph.reset_for_new_planning()
                self.graph, self.queue, self.k_m, self.queue_set = initDStarLite(self.graph, self.queue, self.s_current, self.target, self.k_m)
            except Exception as e:
                print(f"Error: Agent {self.id} failed to re-initialize D* Lite after door replanning: {e}")
                return 'Replanning Error'

            return None

        except Exception as e:
            print(f"Critical error in replan_door_path() for agent {self.id}: {e}")
            return f'Critical Replanning Error: {e}'

    def set_next_target(self, current_location, replan=False):
        try:
            # Record the door that was just reached (before incrementing targetidx)
            self.door_path.append(current_location)

            self.targetidx += 1
            current_id = find_door_id_by_position(self.door_graph, current_location)
            current_type = self.door_graph.nodes[current_id].type if current_id in self.door_graph.nodes else None
            if self.targetidx >= len(self.targets) or current_type == 'exit':
                return 'Evacuated'
            elif replan:
                x, y = stateNameToCoords(current_location)
                lx, ly = stateNameToCoords(self.position_history[-1]) if self.position_history else (None, None)
                x = x + x - lx if lx is not None else x
                y = y + y - ly if ly is not None else y
                not_door_location = f'x{x}y{y}'
                self.update_lazy_graph(not_door_location)
                result = self.replan_door_path(not_door_location)
                if result is not None:
                    return result
            else:
                try:
                    self.target = self.targets[self.targetidx]
                except IndexError as e:
                    print(f"Error: Agent {self.id} target index {self.targetidx} out of bounds for targets list of length {len(self.targets)}: {e}")
                    return 'Target Index Error'

                try:
                    self.graph.setStart(current_location)
                except Exception as e:
                    print(f"Error: Agent {self.id} failed to set start position to {current_location}: {e}")
                    return 'Start Position Error'

                try:
                    self.graph.setGoal(self.target)
                except Exception as e:
                    print(f"Error: Agent {self.id} failed to set goal to {self.target}: {e}")
                    return 'Goal Position Error'

                return None

        except Exception as e:
            print(f"Critical error in set_next_target() for agent {self.id}: {e}")
            return f'Critical Target Setting Error: {e}'
    
    def move(self):
        self.total_steps += 1
        coord_current = stateNameToCoords(self.s_current)

        # Validate coordinates are within bounds
        if (coord_current[1] < 0 or coord_current[1] >= len(self.graph.cells) or
            coord_current[0] < 0 or coord_current[0] >= len(self.graph.cells[0])):
            print(f"Error: Agent {self.id} at invalid position {self.s_current} -> coords {coord_current}")
            return 'stuck'

        if(self.graph.cells[coord_current[1]][coord_current[0]] < 0):
            print(f"Warning: Agent {self.id} starting on an obstacle at {self.s_current}!")
            return 'stuck'
        else:
            self.fire_damage += self.graph.cells[coord_current[1]][coord_current[0]]

            # Safely access temperature map with bounds checking
            try:
                temp = self.fire_model.temperature_map[coord_current[1]][coord_current[0]]
                self.average_temp = (self.average_temp * (self.total_steps - 1) + temp) / self.total_steps
                self.peak_temp = max(self.peak_temp, temp)
            except (IndexError, AttributeError) as e:
                # If temperature map doesn't exist or is wrong size, use default
                print(f"Warning: Agent {self.id} cannot access temperature at {coord_current}: {e}")
                # Continue without updating temperature stats
                    
        try:
            if self.s_current in self.position_history[-3:]:  # Check last 3 positions
                # print(f"Cycle detected at {self.s_current}! Forcing rescan...")
                # Force a more aggressive rescan to break the cycle
                try:
                    scanForObstacles(self.graph, self.queue, self.queue_set, self.s_current, self.VIEWING_RANGE * 2, self.k_m)
                    computeShortestPath(self.graph, self.queue, self.queue_set, self.s_current, self.k_m)
                except Exception as e:
                    print(f"Warning: Failed to perform cycle detection rescan for agent {self.id}: {e}")
                    # Continue with normal movement even if rescan fails

            try:
                self.s_new, self.k_m = moveAndRescan(self.graph, self.queue, self.queue_set, self.s_current, self.VIEWING_RANGE, self.k_m, self.occupancy, self.max_occupancy)
            except Exception as e:
                print(f"Error: Failed to move agent {self.id} from {self.s_current}: {e}")
                return f'Movement Error: {e}'

            self.position_history.append(self.s_current)
            if len(self.position_history) > 5:
                self.position_history.pop(0)

            # Check if agent reached target door (either on it or adjacent to it within 2 cells)
            target_reached = False
            coord_target = stateNameToCoords(self.target)

            if self.s_new == 'goal' or self.s_new == self.target:
                target_reached = True
            else:
                # Check if close to target (within 2 cells in 8-directions)
                coord_new = stateNameToCoords(self.s_new) if self.s_new != 'stuck' else None
                if coord_new:
                    dx = abs(coord_new[0] - coord_target[0])
                    dy = abs(coord_new[1] - coord_target[1])
                    if dx <= 2 and dy <= 2:  # Within 2 cells
                        target_reached = True

            if target_reached:
                # Move to new position if not stuck
                if self.s_new != 'stuck' and self.s_new != self.s_current:
                    coord_current = stateNameToCoords(self.s_current)
                    coord_new = stateNameToCoords(self.s_new)

                    # Bounds check before updating occupancy
                    if (0 <= coord_current[1] < len(self.occupancy) and
                        0 <= coord_current[0] < len(self.occupancy[0])):
                        self.occupancy[coord_current[1]][coord_current[0]] -= 1
                    else:
                        print(f"Warning: Agent {self.id} cannot update occupancy at invalid position {coord_current}")

                    if (0 <= coord_new[1] < len(self.occupancy) and
                        0 <= coord_new[0] < len(self.occupancy[0])):
                        self.occupancy[coord_new[1]][coord_new[0]] += 1
                    else:
                        print(f"Warning: Agent {self.id} cannot update occupancy at invalid position {coord_new}")

                    self.s_current = self.s_new

                self.position_history.append(self.s_current)
                # Use the door position for replanning, not current position
                result = self.set_next_target(self.target, replan=True)
                if result == 'Evacuated':
                    coord_current = stateNameToCoords(self.s_current)
                    if (0 <= coord_current[1] < len(self.occupancy) and
                        0 <= coord_current[0] < len(self.occupancy[0])):
                        self.occupancy[coord_current[1]][coord_current[0]] -= 1
                    return 'Evacuated'
                try:
                    self.queue, self.k_m = self.graph.reset_for_new_planning()
                    self.graph, self.queue, self.k_m, self.queue_set = initDStarLite(self.graph, self.queue, self.s_current, self.target, self.k_m)
                except Exception as e:
                    print(f"Error: Failed to initialize new path for agent {self.id} to target {self.target}: {e}")
                    return f'Path Planning Error: {e}'

                return 'New Target Set'
            elif self.s_new == 'stuck':
                if (0 <= coord_current[1] < len(self.occupancy) and
                    0 <= coord_current[0] < len(self.occupancy[0])):
                    self.occupancy[coord_current[1]][coord_current[0]] -= 1
                return 'stuck'
            else:
                coord_current = stateNameToCoords(self.s_current)
                coord_new = stateNameToCoords(self.s_new)

                # Bounds check before updating occupancy
                if (0 <= coord_current[1] < len(self.occupancy) and
                    0 <= coord_current[0] < len(self.occupancy[0])):
                    self.occupancy[coord_current[1]][coord_current[0]] -= 1
                else:
                    print(f"Warning: Agent {self.id} cannot update occupancy at invalid position {coord_current}")

                if (0 <= coord_new[1] < len(self.occupancy) and
                    0 <= coord_new[0] < len(self.occupancy[0])):
                    self.occupancy[coord_new[1]][coord_new[0]] += 1
                else:
                    print(f"Warning: Agent {self.id} cannot update occupancy at invalid position {coord_new}")

                self.s_current = self.s_new
                return None

        except Exception as e:
            print(f"Critical error in move() for agent {self.id}: {e}")
            return f'Critical Movement Error: {e}'
        
    def update_graph(self, changes):
        for(coord, value) in changes.items():
            try:
                x, y = stateNameToCoords(coord)
                if 0 <= x < self.graph.x_dim and 0 <= y < self.graph.y_dim:
                    self.graph.cells[y][x] = value
                else:
                    print(f"Warning: Agent {self.id} received out-of-bounds update for {coord} ({x},{y})")
            except Exception as e:
                print(f"Warning: Agent {self.id} failed to process graph update for {coord}: {e}")

        # If considering environmental factors OR wall preference, update all cells with costs
        if (self.consider_env_factors and self.fire_model is not None) or self.wall_preference > 0:
            try:
                for y in range(self.graph.y_dim):
                    for x in range(self.graph.x_dim):
                        # Calculate and apply environmental cost (includes wall preference)
                        env_cost = self.calculate_environmental_cost(y, x)
                        self.graph.cells[y][x] = env_cost
            except Exception as e:
                print(f"Warning: Agent {self.id} failed to apply environmental costs: {e}")

        self.graph.updateGraphFromTerrain()
        self.graph.reset_for_new_planning()
        self.graph, self.queue, self.k_m, self.queue_set = initDStarLite(self.graph, self.queue, self.s_current, self.targets[self.targetidx], self.k_m)
        self.graph.setStart(self.s_current)
        self.graph.setGoal(self.targets[self.targetidx])

class EvacuationSimulation():
    def _create_wall_distance_map(self):
        """
        Create a map of distances to nearest wall/obstacle using BFS.
        Returns a dictionary mapping state names to their distance from the nearest wall.
        """
        from collections import deque

        dist_map = {}
        queue = deque()

        # Initialize with all walls/obstacles (value = -2)
        for row in range(self.map_rows):
            for col in range(self.map_cols):
                state = f"x{col}y{row}"
                if self.shared_fire_map[row][col] == -2:
                    dist_map[state] = 0
                    queue.append((state, 0))

        # BFS to compute distances from walls
        while queue:
            state, dist = queue.popleft()
            x, y = stateNameToCoords(state)

            # Check 4-connected neighbors (only orthogonal, not diagonal)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                neighbor = f"x{nx}y{ny}"
                if 0 <= nx < self.map_cols and 0 <= ny < self.map_rows:
                    if neighbor not in dist_map:
                        dist_map[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

        return dist_map

    def __init__(self, config: SimulationConfig, silent: bool = False):
        # Set the obstacle value to match fire (-2)
        set_OBS_VAL(-2)
        self.evacuated_agents = []
        self.progress = {i: 0 for i in range(config.agent_num)}
        self.config = config  # Store config for fire update interval
        self.silent = silent  # Control print output

        self.path_count = dict()
        self.average_fire_damage = 0.0
        self.survived_agents = 0

        # Initialize fire model based on selected type
        if config.fire_model_type == "realistic":
            from fire_model_realistic import create_fire_model
            self.model = create_fire_model(rows=config.map_rows, cols=config.map_cols)
            if not silent:
                print(f"Using REALISTIC fire model (update interval: every {config.fire_update_interval} timesteps = {config.fire_update_interval * config.timestep_duration}s)")
        elif config.fire_model_type == "aggressive":
            from fire_model_aggressive import create_fire_model
            self.model = create_fire_model(rows=config.map_rows, cols=config.map_cols)
            if not silent:
                print(f"Using AGGRESSIVE fire model (update interval: every {config.fire_update_interval} timesteps = {config.fire_update_interval * config.timestep_duration}s)")
        else:
            # Default fire model
            self.model = create_fire_model(rows=config.map_rows, cols=config.map_cols, wind_speed=1.0)
            if not silent:
                print(f"Using DEFAULT fire model (update interval: every {config.fire_update_interval} timesteps = {config.fire_update_interval * config.timestep_duration}s)")

        self.monitor = FireMonitor(self.model)

        try:
            self.shared_fire_map = config.initial_fire_map if hasattr(config, 'initial_fire_map') else [[0 for _ in range(config.map_cols)] for _ in range(config.map_rows)]
            self.map_rows = config.map_rows
            self.map_cols = config.map_cols
            self.max_occupancy = config.max_occupancy
            self.agent_num = config.agent_num
            self.viewing_range = config.viewing_range
            self.door_configs = config.door_configs if config.door_configs else []
            self.occupancy = [0] * self.map_rows
            for i in range(self.map_rows):
                self.occupancy[i] = [0] * self.map_cols

            if not self.occupancy or len(self.occupancy) != self.map_rows or any(len(row) != self.map_cols for row in self.occupancy):
                raise ValueError("Occupancy grid dimensions do not match specified map dimensions")

            # Create wall distance map for wall preference pathfinding
            self.wall_distance_map = self._create_wall_distance_map()
            if not silent:
                print(f"Wall distance map created with {len(self.wall_distance_map)} cells.")

            self.agents: list[EvacuationAgent] = []
            self.base_door_graph = build_door_graph(self.shared_fire_map, self.door_configs)
            for i in range(self.agent_num):
                start_pos = config.start_positions[i]  # Example start positions; modify as needed
                # Get fearness for this agent (default to 1.0 if not specified)
                fearness = 1.0
                if config.agent_fearness and i < len(config.agent_fearness):
                    fearness = config.agent_fearness[i]
                try:
                    agent = EvacuationAgent(
                        i, start_pos, self.occupancy, self.max_occupancy,
                        self.map_rows, self.map_cols, self.viewing_range,
                        fire_fearness=fearness,
                        base_door_graph=self.base_door_graph,
                        fire_model=self.model,
                        consider_env_factors=config.consider_env_factors,
                        wall_distance_map=self.wall_distance_map,
                        wall_preference=config.wall_preference,
                        initial_fire_map=self.shared_fire_map
                    )
                    self.agents.append(agent)
                except Exception as e:
                    print(f"Failed to initialize agent {i}: {e}")
                    raise

        except Exception as e:
            print(f"Critical error initializing simulation: {e}")
            raise

        # Initialize spatial index for knowledge sharing
        from spatial_index import SpatialIndex

        # Auto-calculate sector_size if not provided
        sector_size = config.sector_size if config.sector_size is not None else int(config.communication_range)

        self.spatial_index = SpatialIndex(
            map_rows=config.map_rows,
            map_cols=config.map_cols,
            sector_size=sector_size
        )

        self.communication_range = config.communication_range
        self.sharing_interval = config.sharing_interval

        if not silent:
            print(f"Spatial index initialized with sector_size={sector_size}, communication_range={config.communication_range} cells")

        # try:
        #     self.anim = RealtimeGridAnimator(initial_grid=[[0.0 for _ in range(self.map_cols)] for _ in range(self.map_rows)])
        # except Exception as e:
        #     print(f"Warning: Failed to initialize real-time grid animators: {e}")
        #     self.anim = None

    def check_agent_survival(self, agent: EvacuationAgent) -> bool:
        # Define survival criteria based on fire damage and temperature thresholds
        MAX_FIRE_DAMAGE = 100.0  # Example threshold
        MAX_PEAK_TEMP = 150.0    # Example threshold in Celsius
        MAX_AVERAGE_TEMP = 100.0   # Example threshold in Celsius

        survived = agent.fire_damage < MAX_FIRE_DAMAGE and agent.peak_temp < MAX_PEAK_TEMP and agent.average_temp < MAX_AVERAGE_TEMP
        if not self.silent:
            if survived:
                print(f"Agent {agent.id} survived evacuation with fire damage {agent.fire_damage:.2f}, peak temp {agent.peak_temp:.2f}C, and average temp {agent.average_temp:.2f}C.")
            else:
                print(f"Agent {agent.id} did NOT survive evacuation! Fire damage: {agent.fire_damage:.2f}, Peak temp: {agent.peak_temp:.2f}C, Average temp: {agent.average_temp:.2f}C.")
        return survived

    def step(self):
        results = []
        for agent in self.agents:
            try:
                result = agent.move()
                results.append((agent.id, result))
                if result == 'Evacuated':
                    self.evacuated_agents.append(agent.id)
                    if not self.silent:
                        print(f"Agent {agent.id} has evacuated successfully.")
                    survived = self.check_agent_survival(agent)
                    if survived:
                        self.survived_agents += 1
                    self.path_count[tuple(agent.door_path)] = self.path_count.get(tuple(agent.door_path), 0) + 1
                    # Update average fire damage and temperatures
                    if len(self.evacuated_agents) == 1:
                        self.average_fire_damage = agent.fire_damage
                        self.average_peak_temp = agent.peak_temp
                        self.average_avg_temp = agent.average_temp
                    else:
                        self.average_fire_damage = self.average_fire_damage*(len(self.evacuated_agents)-1)/len(self.evacuated_agents) + agent.fire_damage/len(self.evacuated_agents)
                        self.average_peak_temp = self.average_peak_temp*(len(self.evacuated_agents)-1)/len(self.evacuated_agents) + agent.peak_temp/len(self.evacuated_agents)
                        self.average_avg_temp = self.average_avg_temp*(len(self.evacuated_agents)-1)/len(self.evacuated_agents) + agent.average_temp/len(self.evacuated_agents)
                    # Remove the agent from the simulation
                    self.agents.remove(agent)

                elif result == 'New Target Set':
                    if not self.silent:
                        print(f"Agent {agent.id} reached target and is setting new target {agent.target}.")
                    self.progress[agent.id] += 1
                elif result == 'stuck':
                    if not self.silent:
                        print(f"Agent {agent.id} is stuck at {agent.s_current}.")
                    self.agents.remove(agent)
                elif result is not None:
                    if not self.silent:
                        print(f"Agent {agent.id} encountered an issue: {result}")

            except Exception as e:
                print(f"Critical error moving agent {agent.id}: {e}")
                results.append((agent.id, f'Critical Movement Error: {e}'))
        return results

    def status(self):
        done = len(self.agents) == 0
        return done, {
            'total_agents': self.agent_num,
            'evacuated_agents': len(self.evacuated_agents),
            'remaining_agents': len(self.agents),
            'progress': self.progress
        }

    def visualize(self):
        """Display a simple grid visualization showing agent positions and targets"""
        # Create a grid representation
        grid = [['.' for _ in range(self.map_cols)] for _ in range(self.map_rows)]

        # Mark targets with 'T' and number them
        for i, target in enumerate(self.targets):
            try:
                coords = stateNameToCoords(target)
                if 0 <= coords[0] < self.map_cols and 0 <= coords[1] < self.map_rows:
                    grid[coords[1]][coords[0]] = f'T{i+1}'
            except:
                pass

        # Mark agent positions with their ID
        for agent in self.agents:
            try:
                coords = stateNameToCoords(agent.s_current)
                if 0 <= coords[0] < self.map_cols and 0 <= coords[1] < self.map_rows:
                    # If there's already a target here, show both
                    if grid[coords[1]][coords[0]].startswith('T'):
                        grid[coords[1]][coords[0]] = f'A{agent.id}+{grid[coords[1]][coords[0]]}'
                    else:
                        grid[coords[1]][coords[0]] = f'A{agent.id}'
            except:
                pass

        # Print the grid
        print("\n" + "="*50)
        print("EVACUATION SIMULATION VISUALIZATION")
        print("="*50)
        print("Legend: A# = Agent ID, T# = Target #, . = Empty")
        print("-" * (self.map_cols * 4 + 1))

        for row in grid:
            print("|", end="")
            for cell in row:
                # Pad cells to consistent width
                print(f"{cell:^3}", end="|")
            print()
            print("-" * (self.map_cols * 4 + 1))

        # Show agent targets
        print("\nAgent Status:")
        for agent in self.agents:
            print(f"  Agent {agent.id}: Position {agent.s_current}, Target {agent.target} (#{agent.targetidx + 1})")
        print()
    
    def update_environment(self, changes):
        for (coord, value) in changes.items():
            x, y = stateNameToCoords(coord)
            if 0 <= x < self.map_cols and 0 <= y < self.map_rows:
                self.shared_fire_map[y][x] = value
            else:
                print(f"Warning: Received out-of-bounds environment update for {coord} ({x},{y})")
        
        for agent in self.agents:
            try:
                agent.update_graph(changes)
            except Exception as e:
                print(f"Warning: Agent {agent.id} failed to update graph: {e}")

    def update_fire(self, show_visualization=False):
        # Conservative fire generation for testing color-coded visualization
        self.fire_data = self.monitor.monitor_step(self.shared_fire_map)
        if show_visualization:
            self.visualize_snapshot(self.fire_data['environmental_snapshot'])
        return self.fire_data['changes']

    def visualize_snapshot(self, fire_data):
        if self.anim:
            self.anim.update(fire_data['oxygen_map'])
        

    def run(self, max_steps=1000, show_visualization=False, use_pygame=False, use_matlab=False) -> dict:
        '''
        Run the evacuation simulation until all agents have evacuated or max_steps is reached.
        Args:
            max_steps (int): Maximum number of simulation steps to run.
            show_visualization (bool): Whether to show text-based visualization in the console.
            use_pygame (bool): Whether to use pygame for visualization if available.
            use_matlab (bool): Whether to use MATLAB-style visualization if available.
        Returns:
            dict: Summary of simulation results including path_count, steps, average_fire_damage,
                  average_peak_temp, average_avg_temp, evacuated_agents, survived_agents.
        '''
        self.simulation_results= dict()       
        self.steps = 0
        visualizer = None
        reached_targets = set()

        init_changes = {}
        #inital fire update
        for y in range(self.map_rows):
            for x in range(self.map_cols):
                if self.shared_fire_map[y][x] != 0:
                    coord = f'x{x}y{y}'
                    init_changes[coord] = self.shared_fire_map[y][x]
        self.update_environment(init_changes)

        # Initialize visualizer based on preference
        if use_matlab and MATLAB_VISUALIZER_AVAILABLE:
            visualizer = create_matlab_visualizer(
                self.map_rows,
                self.map_cols,
                self.model,
                trajectory_length=10
            )
            print("Using MATLAB-style visualization with environmental data.")
            print("Use checkboxes to toggle: Temperature, Oxygen, Smoke, Fuel, Fire, Trajectories")
        elif use_pygame and PYGAME_AVAILABLE:
            visualizer = EvacuationVisualizer(self.map_rows, self.map_cols, cell_size=30)
            print("Using pygame visualization. Close window or press ESC to quit.")
        elif use_matlab and not MATLAB_VISUALIZER_AVAILABLE:
            print("MATLAB-style visualizer not available. Install scipy: pip install scipy")
            print("Falling back to pygame or text visualization.")
            if PYGAME_AVAILABLE:
                visualizer = EvacuationVisualizer(self.map_rows, self.map_cols, cell_size=30)
                print("Using pygame visualization. Close window or press ESC to quit.")
        elif use_pygame and not PYGAME_AVAILABLE:
            print("Pygame not available, falling back to text visualization.")

        # Show initial state
        if show_visualization and not visualizer:
            print(f"\n=== STEP 0 (Initial State) ===")
            self.visualize()
        elif visualizer:
            done, status = self.status()
            # MATLAB visualizer needs fire state, pygame doesn't
            if use_matlab and MATLAB_VISUALIZER_AVAILABLE:
                fire_state_array = np.array(self.shared_fire_map)
                if not visualizer.update_display(self.steps, self.agents, self.door_configs, fire_state_array, status):
                    visualizer.close()
                    return
            else:
                if not visualizer.update_display(self.steps, self.agents, self.door_configs, status, reached_targets):
                    visualizer.close()
                    return

        while True:
            self.steps += 1
            done, status = self.status()

            if done:
                if not self.silent:
                    print("All agents have evacuated or are stuck. Simulation complete.")
                if show_visualization and not visualizer:
                    self.visualize()
                elif visualizer:
                    # Show final state for a few seconds
                    if use_matlab and MATLAB_VISUALIZER_AVAILABLE:
                        fire_state_array = np.array(self.shared_fire_map)
                        visualizer.update_display(self.steps, self.agents, self.door_configs, fire_state_array, status)
                    else:
                        visualizer.update_display(self.steps, self.agents, self.door_configs, status, reached_targets)
                    time.sleep(2)
                break
            elif self.steps >= max_steps:
                if not self.silent:
                    print("Maximum steps reached. Ending simulation.")
                break

            results = self.step()

            # Track reached targets for visualization
            # for agent_id, result in results:
            #     if result == 'New Target Set':
            #         # Find the agent and add their previous target to reached targets
            #         for agent in self.agents:
            #             if agent.id == agent_id and agent.targetidx > 0:
            #                 prev_target = self.targets[agent.targetidx - 1]
            #                 reached_targets.add(prev_target)

            # Update fire model at specified interval (decoupled from agent movement)
            if self.steps % self.config.fire_update_interval == 0:
                changes = self.update_fire()
                if changes:
                    self.update_environment(changes)

            # Share door graph knowledge between nearby agents at specified interval
            if self.steps % self.sharing_interval == 0:
                # Update spatial index with current agent positions - O(n)
                self.spatial_index.update(self.agents)

                # Each agent shares knowledge with nearby agents - O(n * k) where k << n
                for agent in self.agents:
                    # All agents in the list are active (evacuated/stuck agents are removed)
                    if agent.door_graph is not None:
                        nearby_agents = self.spatial_index.get_nearby_agents(
                            agent.s_current,
                            self.agents,
                            self.communication_range
                        )
                        agent.share_with_nearby(nearby_agents)

            # Handle visualization updates
            if visualizer:
                if use_matlab and MATLAB_VISUALIZER_AVAILABLE:
                    fire_state_array = np.array(self.shared_fire_map)
                    if not visualizer.update_display(self.steps, self.agents, self.door_configs, fire_state_array, status):
                        break  # User closed window
                else:
                    if not visualizer.update_display(self.steps, self.agents, self.door_configs, status, reached_targets):
                        break  # User closed window
                visualizer.wait_for_next_frame(fps=10)  # 10 FPS for better visibility
            else:
                # Show text visualization every few steps or when significant events occur
                show_this_step = False
                for agent_id, result in results:
                    if result in ['Evacuated', 'New Target Set', 'stuck'] or result is not None:
                        show_this_step = True
                        break

                if show_visualization and (show_this_step or self.steps % 5 == 0):
                    print(f"\n=== STEP {self.steps} ===")
                    self.visualize()

                # Print status if not in silent mode
                if not self.silent:
                    print(f"Status: {status}")

        # Create data directory if it doesn't exist (safe for parallel execution)
        data_dir = "./data"
        os.makedirs(data_dir, exist_ok=True)

        # Generate unique filename with microseconds to avoid conflicts in parallel execution
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.monitor.save_monitoring_data(f"{data_dir}/evacuation_simulation_data_{timestamp}.json", silent=self.silent)
        self.monitor.export_csv_data(f"{data_dir}/evacuation_simulation_data_{timestamp}.csv", silent=self.silent)

        if visualizer:
            visualizer.close()
        
        self.simulation_results['path_count'] = self.path_count
        self.simulation_results['steps'] = self.steps
        self.simulation_results['average_fire_damage'] = self.average_fire_damage
        self.simulation_results['average_peak_temp'] = self.average_peak_temp
        self.simulation_results['average_avg_temp'] = self.average_avg_temp
        self.simulation_results['evacuated_agents'] = len(self.evacuated_agents)
        self.simulation_results['survived_agents'] = self.survived_agents
        
        return self.simulation_results

if __name__ == "__main__":
    # Example configuration

    initial_fire_map = [[0 for _ in range(20)] for _ in range(20)]
    initial_fire_map[9][9] = 0.4  # Start fire at bottom-left corner

    # config = SimulationConfig(
    #     map_rows=20,
    #     map_cols=20,
    #     max_occupancy=2,
    #     targets=['x9y0', 'x19y13', 'x2y17', 'x16y3', 'x8y19'],
    #     agent_num=5,
    #     viewing_range=3,
    #     start_positions=['x0y19', 'x7y1', 'x13y5', 'x10y16', 'x19y10'],
    #     initial_fire_map=initial_fire_map
    # )

    parser = argparse.ArgumentParser(description="Read configuration file (JSON)")
    parser.add_argument(
        "--config",
        type=str,
        default="example_configuration.json",
        help="Path to configuration file (default: example_configuration.json)"
    )
    args = parser.parse_args()

    json_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(json_path, 'r', encoding='utf-8') as f:
        json_config = json.load(f)
    config = SimulationConfig.from_json(json_config)

    simulation = EvacuationSimulation(config)
    simulation_results = simulation.run(
        max_steps=500, 
        show_visualization=False, 
        use_pygame=False, 
        use_matlab=True
    )

    # Print the path count for each agent
    for agent_path, count in simulation_results['path_count'].items():
        print(f"Agent path: {agent_path}, Count: {count}")