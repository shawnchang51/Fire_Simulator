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

import random
import os
from d_star_lite.grid import GridWorld
from d_star_lite.utils import stateNameToCoords
from d_star_lite.d_star_lite import initDStarLite, moveAndRescan, set_OBS_VAL, scanForObstacles, computeShortestPath
from dataclasses import dataclass
from fire_model_float import simulate_fire_spread
import json
from fire_model_float import create_fire_model
from fire_monitor import FireMonitor
from snapshot_ainmator import RealtimeGridAnimator
from datetime import datetime
import argparse

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
    targets: list[str]
    initial_fire_map: list[list[float]]
    agent_num: int
    viewing_range: int = 5
    # New spatial/temporal parameters
    cell_size: float = 0.3  # meters per cell
    timestep_duration: float = 0.5  # seconds per timestep
    fire_update_interval: int = 4  # update fire every N timesteps (default: 4 timesteps = 2 seconds)
    fire_model_type: str = "realistic"  # "realistic", "aggressive", or "default"

    @classmethod
    def from_json(cls, json_data):
        config = cls(
            map_rows=json_data['map_rows'],
            map_cols=json_data['map_cols'],
            max_occupancy=json_data['max_occupancy'],
            start_positions=json_data['start_positions'],
            targets=json_data['targets'],
            initial_fire_map=json_data.get('initial_fire_map', [[0 for _ in range(json_data['map_cols'])] for _ in range(json_data['map_rows'])]),
            agent_num=json_data['agent_num'],
            viewing_range=json_data.get('viewing_range', 5),
            cell_size=json_data.get('cell_size', 0.3),
            timestep_duration=json_data.get('timestep_duration', 0.5),
            fire_update_interval=json_data.get('fire_update_interval', 4),
            fire_model_type=json_data.get('fire_model_type', 'realistic')
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
    def __init__(self, id: int, start:str, occupancy, max_occupancy, map_rows, map_cols, targets: list[str], viewing_range=5):
        try:
            self.id = id
            self.start = start
            self.occupancy = occupancy
            self.max_occupancy = max_occupancy
            self.VIEWING_RANGE = viewing_range
            self.fire_damage = 0

            # Validate inputs
            if not targets or len(targets) == 0:
                raise ValueError(f"Agent {id}: targets list cannot be empty")

            try:
                self.graph = GridWorld(map_rows, map_cols)
            except Exception as e:
                raise ValueError(f"Agent {id}: Failed to create GridWorld with dimensions {map_rows}x{map_cols}: {e}")

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

            self.targetidx = 0
            self.targets = targets
            self.target = targets[self.targetidx]

            self.k_m = 0
            self.queue = []

            try:
                # Set start and goal positions on the graph before initializing D* Lite
                self.graph.setStart(start)
                self.graph.setGoal(self.target)
                self.graph, self.queue, self.k_m = initDStarLite(self.graph, self.queue, start, self.target, self.k_m)
            except Exception as e:
                raise ValueError(f"Agent {id}: Failed to initialize D* Lite from {start} to {self.target}: {e}")

            self.s_current = start
            self.position_history = []

        except Exception as e:
            print(f"Critical error initializing agent {id}: {e}")
            raise

    def set_next_target(self, current_location):
        try:
            self.targetidx += 1
            if self.targetidx >= len(self.targets):
                return 'Evacuated'
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
        coord_current = stateNameToCoords(self.s_current)
        if(self.graph.cells[coord_current[1]][coord_current[0]] < 0):
            print(f"Warning: Agent {self.id} starting on an obstacle at {self.s_current}!")
            return 'stuck'
        else:
            self.fire_damage += self.graph.cells[coord_current[1]][coord_current[0]]
        
        try:
            if self.s_current in self.position_history[-3:]:  # Check last 3 positions
                # print(f"Cycle detected at {self.s_current}! Forcing rescan...")
                # Force a more aggressive rescan to break the cycle
                try:
                    scanForObstacles(self.graph, self.queue, self.s_current, self.VIEWING_RANGE * 2, self.k_m)
                    computeShortestPath(self.graph, self.queue, self.s_current, self.k_m)
                except Exception as e:
                    print(f"Warning: Failed to perform cycle detection rescan for agent {self.id}: {e}")
                    # Continue with normal movement even if rescan fails

            try:
                self.s_new, self.k_m = moveAndRescan(self.graph, self.queue, self.s_current, self.VIEWING_RANGE, self.k_m, self.occupancy, self.max_occupancy)
            except Exception as e:
                print(f"Error: Failed to move agent {self.id} from {self.s_current}: {e}")
                return f'Movement Error: {e}'

            self.position_history.append(self.s_current)
            if len(self.position_history) > 5:
                self.position_history.pop(0)

            if self.s_new == 'goal' or self.s_new == self.target:
                coord_current = stateNameToCoords(self.s_current)
                self.occupancy[coord_current[1]][coord_current[0]] -= 1
                coord_target = stateNameToCoords(self.target)
                self.occupancy[coord_target[1]][coord_target[0]] += 1

                self.s_current = self.target

                self.position_history.append(self.s_current)
                result = self.set_next_target(self.s_current)
                if result == 'Evacuated':
                    coord_target = stateNameToCoords(self.target)
                    self.occupancy[coord_target[1]][coord_target[0]] -= 1
                    return 'Evacuated'
                try:
                    self.queue, self.k_m = self.graph.reset_for_new_planning()
                    self.graph, self.queue, self.k_m = initDStarLite(self.graph, self.queue, self.s_current, self.target, self.k_m)
                except Exception as e:
                    print(f"Error: Failed to initialize new path for agent {self.id} to target {self.target}: {e}")
                    return f'Path Planning Error: {e}'

                return 'New Target Set'
            elif self.s_new == 'stuck':
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

        self.graph.updateGraphFromTerrain()
        self.graph.reset_for_new_planning()
        self.graph, self.queue, self.k_m = initDStarLite(self.graph, self.queue, self.s_current, self.targets[self.targetidx], self.k_m)
        self.graph.setStart(self.s_current)
        self.graph.setGoal(self.targets[self.targetidx])

class EvacuationSimulation():
    def __init__(self, config: SimulationConfig):
        # Set the obstacle value to match fire (-1)
        set_OBS_VAL(-1)
        self.evacuated_agents = []
        self.progress = {i: 0 for i in range(config.agent_num)}
        self.config = config  # Store config for fire update interval

        # Initialize fire model based on selected type
        if config.fire_model_type == "realistic":
            from fire_model_realistic import create_fire_model
            self.model = create_fire_model(rows=config.map_rows, cols=config.map_cols)
            print(f"Using REALISTIC fire model (update interval: every {config.fire_update_interval} timesteps = {config.fire_update_interval * config.timestep_duration}s)")
        elif config.fire_model_type == "aggressive":
            from fire_model_aggressive import create_fire_model
            self.model = create_fire_model(rows=config.map_rows, cols=config.map_cols)
            print(f"Using AGGRESSIVE fire model (update interval: every {config.fire_update_interval} timesteps = {config.fire_update_interval * config.timestep_duration}s)")
        else:
            # Default fire model
            self.model = create_fire_model(rows=config.map_rows, cols=config.map_cols, wind_speed=1.0)
            print(f"Using DEFAULT fire model (update interval: every {config.fire_update_interval} timesteps = {config.fire_update_interval * config.timestep_duration}s)")

        self.monitor = FireMonitor(self.model)

        try:
            self.shared_fire_map = config.initial_fire_map if hasattr(config, 'initial_fire_map') else [[0 for _ in range(config.map_cols)] for _ in range(config.map_rows)]
            self.map_rows = config.map_rows
            self.map_cols = config.map_cols
            self.max_occupancy = config.max_occupancy
            self.targets = config.targets
            self.agent_num = config.agent_num
            self.viewing_range = config.viewing_range
            self.occupancy = [0] * self.map_rows
            for i in range(self.map_rows):
                self.occupancy[i] = [0] * self.map_cols

            if not self.occupancy or len(self.occupancy) != self.map_rows or any(len(row) != self.map_cols for row in self.occupancy):
                raise ValueError("Occupancy grid dimensions do not match specified map dimensions")

            self.agents = []
            for i in range(self.agent_num):
                start_pos = config.start_positions[i]  # Example start positions; modify as needed
                try:
                    agent = EvacuationAgent(i, start_pos, self.occupancy, self.max_occupancy, self.map_rows, self.map_cols, self.targets, self.viewing_range)
                    self.agents.append(agent)
                except Exception as e:
                    print(f"Failed to initialize agent {i}: {e}")
                    raise

        except Exception as e:
            print(f"Critical error initializing simulation: {e}")
            raise

        # try:
        #     self.anim = RealtimeGridAnimator(initial_grid=[[0.0 for _ in range(self.map_cols)] for _ in range(self.map_rows)])
        # except Exception as e:
        #     print(f"Warning: Failed to initialize real-time grid animators: {e}")
        #     self.anim = None

    def step(self):
        results = []
        for agent in self.agents:
            try:
                result = agent.move()
                results.append((agent.id, result))
                if result == 'Evacuated':
                    self.evacuated_agents.append(agent.id)
                    print(f"Agent {agent.id} has evacuated successfully.")
                    # Remove the agent from the simulation
                    self.agents.remove(agent)
                elif result == 'New Target Set':
                    print(f"Agent {agent.id} reached target and is setting new target {agent.target}.")
                    self.progress[agent.id] += 1
                elif result == 'stuck':
                    print(f"Agent {agent.id} is stuck at {agent.s_current}.")
                    self.agents.remove(agent)
                elif result is not None:
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
        

    def run(self, max_steps=1000, show_visualization=True, use_pygame=True, use_matlab=False):
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
                import numpy as np
                fire_state_array = np.array(self.shared_fire_map)
                if not visualizer.update_display(self.steps, self.agents, self.targets, fire_state_array, status):
                    visualizer.close()
                    return
            else:
                if not visualizer.update_display(self.steps, self.agents, self.targets, status, reached_targets):
                    visualizer.close()
                    return

        while True:
            self.steps += 1
            done, status = self.status()

            if done:
                print("All agents have evacuated or are stuck. Simulation complete.")
                if show_visualization and not visualizer:
                    self.visualize()
                elif visualizer:
                    # Show final state for a few seconds
                    if use_matlab and MATLAB_VISUALIZER_AVAILABLE:
                        import numpy as np
                        fire_state_array = np.array(self.shared_fire_map)
                        visualizer.update_display(self.steps, self.agents, self.targets, fire_state_array, status)
                    else:
                        visualizer.update_display(self.steps, self.agents, self.targets, status, reached_targets)
                    import time
                    time.sleep(2)
                break
            elif self.steps >= max_steps:
                print("Maximum steps reached. Ending simulation.")
                break

            results = self.step()

            # Track reached targets for visualization
            for agent_id, result in results:
                if result == 'New Target Set':
                    # Find the agent and add their previous target to reached targets
                    for agent in self.agents:
                        if agent.id == agent_id and agent.targetidx > 0:
                            prev_target = self.targets[agent.targetidx - 1]
                            reached_targets.add(prev_target)

            # Update fire model at specified interval (decoupled from agent movement)
            if self.steps % self.config.fire_update_interval == 0:
                changes = self.update_fire()
                if changes:
                    self.update_environment(changes)

            # Handle visualization updates
            if visualizer:
                if use_matlab and MATLAB_VISUALIZER_AVAILABLE:
                    import numpy as np
                    fire_state_array = np.array(self.shared_fire_map)
                    if not visualizer.update_display(self.steps, self.agents, self.targets, fire_state_array, status):
                        break  # User closed window
                else:
                    if not visualizer.update_display(self.steps, self.agents, self.targets, status, reached_targets):
                        break  # User closed window
                visualizer.wait_for_next_frame(fps=5)  # 5 FPS for better visibility
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

                # For demonstration, we will just print the status
                print(f"Status: {status}")

        self.monitor.save_monitoring_data(f"./data/evacuation_simulation_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
        self.monitor.export_csv_data(f"./data/evacuation_simulation_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

        if visualizer:
            visualizer.close()

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
    simulation.run(
        max_steps=500, 
        show_visualization=False, 
        use_pygame=False, 
        use_matlab=True
        )