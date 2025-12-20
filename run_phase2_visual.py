"""
Phase 2 Simulation with Visual Validation
==========================================

Runs the optimized Phase 2 simulation (FastEvacuationSim) with pygame visualization
for validation purposes.

This allows you to visually validate that the Phase 2 optimized simulation maintains
the same behavior as the original simulation.

NOTE: Phase 2 simulation does not track environmental data (temperature, oxygen, smoke)
for performance reasons, so MATLAB-style visualization is not supported. Use pygame
visualization instead, which shows fire spread and agent movement.

Usage:
    # Run with pygame visualization (default)
    python run_phase2_visual.py

    # With custom configuration
    python run_phase2_visual.py --config example_configuration.json

    # Run faster with reduced fire update interval
    python run_phase2_visual.py --fire-interval 8
"""

import argparse
import json
import numpy as np
from pathlib import Path
from simulation import SimulationConfig
from d_star_lite.utils import stateNameToCoords
from fast_simulation import FastEvacuationSim, FastAgent
import time

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


class VisualPhase2Wrapper:
    """
    Wrapper around FastEvacuationSim that adds visualization support.
    Adapts the fast simulation to work with the original visualizers.
    """

    def __init__(self, sim: FastEvacuationSim, config: SimulationConfig):
        """
        Initialize wrapper.

        Args:
            sim: FastEvacuationSim instance
            config: Original SimulationConfig for map dimensions
        """
        self.sim = sim
        self.config = config
        self.map_rows = config.map_rows
        self.map_cols = config.map_cols
        self.steps = 0

    def _convert_agents_to_original_format(self):
        """
        Convert FastAgent instances to format expected by visualizers.

        Returns:
            List of agent-like objects with position, status, and trajectory
        """
        class VisualAgent:
            def __init__(self, fast_agent: FastAgent, agent_id: int, grid_wrapper):
                self.id = int(agent_id)  # Ensure it's an integer
                # FastAgent uses (x, y) coordinates
                col, row = fast_agent.x, fast_agent.y
                self.position = f"x{col}y{row}"
                self.s_current = f"x{col}y{row}"  # For MATLAB visualizer
                self.status = fast_agent.status
                self.fire_damage = fast_agent.fire_damage
                # Trajectory tracking (for visualization only)
                self.trajectory = []
                # Grid reference for pygame visualizer
                self.graph = grid_wrapper
                # Target index (Phase 2 doesn't use waypoints, just exits)
                self.targetidx = 0

        # Create a grid wrapper that has a 'cells' attribute
        class GridWrapper:
            def __init__(self, grid_array):
                self.cells = grid_array

        grid_wrapper = GridWrapper(self.sim.grid)

        agents = []
        for i, fa in enumerate(self.sim.agents):
            agent = VisualAgent(fa, i, grid_wrapper)
            agents.append(agent)

        return agents

    def status(self):
        """Get simulation status (done, and status dict/string)."""
        active = sum(1 for a in self.sim.agents if a.status == 'active')
        evacuated = sum(1 for a in self.sim.agents if a.status == 'evacuated')
        stuck = sum(1 for a in self.sim.agents if a.status == 'stuck')
        dead = sum(1 for a in self.sim.agents if a.status == 'dead')

        total = len(self.sim.agents)
        done = (active == 0)

        # Return both string (for MATLAB) and dict (for pygame)
        status_dict = {
            'evacuated_agents': evacuated,
            'total_agents': total,
            'active': active,
            'stuck': stuck,
            'dead': dead
        }
        status_str = f"Step {self.steps} | Active: {active} | Evacuated: {evacuated} | Stuck: {stuck} | Dead: {dead}"

        return done, status_dict, status_str

    def run_with_visualization(self, max_steps=500, use_pygame=True, use_matlab=False, fps=10):
        """
        Run Phase 2 simulation with visualization.

        Args:
            max_steps: Maximum simulation steps
            use_pygame: Use pygame visualization
            use_matlab: Use MATLAB-style visualization (overrides pygame)

        Returns:
            SimResult from FastEvacuationSim
        """
        visualizer = None

        # Auto-scale cell size based on map dimensions
        # Target window size: ~1200x900 pixels
        max_width = 1200
        max_height = 800  # Leave room for info panel (100px)
        cell_size = min(max_width // self.map_cols, max_height // self.map_rows)
        # Clamp to reasonable range
        cell_size = max(5, min(60, cell_size))

        print(f"Map size: {self.map_rows}×{self.map_cols}, using cell size: {cell_size}px")

        # Initialize visualizer based on preference
        if use_matlab and MATLAB_VISUALIZER_AVAILABLE:
            # Phase 2 doesn't track environmental data, so MATLAB viz isn't suitable
            print("WARNING: MATLAB-style visualization requires environmental data (temperature, oxygen, smoke).")
            print("Phase 2 simulation doesn't track this data for performance.")
            print("Falling back to pygame visualization which shows fire and agent positions.")
            if PYGAME_AVAILABLE:
                visualizer = EvacuationVisualizer(self.map_rows, self.map_cols, cell_size=cell_size)
                print("Using pygame visualization. Close window or press ESC to quit.")
            else:
                print("ERROR: Pygame not available. Cannot visualize Phase 2 simulation.")
                return None
        elif use_pygame and PYGAME_AVAILABLE:
            visualizer = EvacuationVisualizer(self.map_rows, self.map_cols, cell_size=cell_size)
            print("Using pygame visualization. Close window or press ESC to quit.")
        elif use_matlab and not MATLAB_VISUALIZER_AVAILABLE:
            print("MATLAB-style visualizer not available. Install scipy: pip install scipy")
            print("Falling back to pygame visualization.")
            if PYGAME_AVAILABLE:
                visualizer = EvacuationVisualizer(self.map_rows, self.map_cols, cell_size=cell_size)
                print("Using pygame visualization. Close window or press ESC to quit.")
        elif use_pygame and not PYGAME_AVAILABLE:
            print("Pygame not available, running without visualization.")
            visualizer = None

        if not visualizer:
            print("WARNING: No visualizer available! Running in headless mode.")
            print("Install pygame (pip install pygame) or scipy (pip install scipy) for visualization.")
            # Fall back to headless run
            return self.sim.run(max_steps=max_steps)

        # Show initial state
        visual_agents = self._convert_agents_to_original_format()
        done, status_dict, status_str = self.status()

        # Pygame visualizer for Phase 2 (MATLAB not supported due to missing environmental data)
        # Extract exit positions as targets
        targets = [f"x{x}y{y}" for x, y in self.sim.exits]
        visualizer.update_display(
            self.steps,
            visual_agents,
            targets,
            status_dict
        )
        # Respect display frame rate if available
        try:
            visualizer.wait_for_next_frame(fps=fps)
        except Exception:
            pass

        # Main simulation loop with visualization
        evacuation_times = []
        termination_reason = "max_steps"

        for step in range(max_steps):
            self.steps = step + 1
            self.sim.step_count = step + 1

            # Update fire periodically
            if step > 0 and step % self.sim.fire_update_interval == 0:
                old_grid = self.sim.grid.copy()
                self.sim.grid = self.sim.fire.step()

                # Find changed cells for D* Lite updates
                changed_cells = []
                rows, cols = self.sim.grid.shape
                for r in range(rows):
                    for c in range(cols):
                        if old_grid[r, c] != self.sim.grid[r, c]:
                            changed_cells.append((c, r))

                # Debug: Show fire spread activity
                if len(changed_cells) > 0:
                    print(f"Step {step}: Fire spreading to {len(changed_cells)} cells")
                elif step > 10:  # Only after initial setup
                    print(f"Step {step}: Fire stopped spreading (performance boost active)")

                # Update pathfinders
                self.sim._update_pathfinders(changed_cells)

            # Only move agents after fire discovery delay has passed
            if step >= self.sim.fire_discovery_delay:
                # Move all agents
                for i, agent in enumerate(self.sim.agents):
                    if agent.status != 'active':
                        continue

                    # Track fire damage BEFORE moving
                    cell_value = self.sim.grid[agent.y, agent.x]
                    if cell_value > 0:
                        agent.fire_damage += cell_value

                    # Get next position from pathfinder
                    pathfinder = self.sim.agent_pathfinders[i]
                    next_pos = pathfinder.get_next_move()

                    if next_pos is None:
                        agent.status = 'stuck'
                        continue

                    # Check if reached exit
                    if next_pos in self.sim.exits:
                        agent.status = 'evacuated'
                        agent.steps = step + 1
                        evacuation_times.append(agent.steps)
                        continue

                    # Move agent
                    agent.x, agent.y = next_pos
                    agent.steps = step + 1

                    # Update pathfinder position
                    pathfinder.move_start(next_pos)
            else:
                # Fire discovery delay: fire spreads but agents don't move
                # Still track fire damage at current positions
                for agent in self.sim.agents:
                    if agent.status == 'active':
                        cell_value = self.sim.grid[agent.y, agent.x]
                        if cell_value > 0:
                            agent.fire_damage += cell_value
                        # Check if fire kills agent during delay
                        if cell_value > 3.0:
                            agent.status = 'dead'

            # Check termination conditions
            active = sum(1 for a in self.sim.agents if a.status == 'active')
            evacuated = sum(1 for a in self.sim.agents if a.status == 'evacuated')
            stuck = sum(1 for a in self.sim.agents if a.status == 'stuck')
            dead = sum(1 for a in self.sim.agents if a.status == 'dead')

            # Update visualization (pygame only for Phase 2)
            visual_agents = self._convert_agents_to_original_format()
            done, status_dict, status_str = self.status()

            # Pygame visualizer expects (step, agents, targets, status)
            targets = [f"x{x}y{y}" for x, y in self.sim.exits]
            if not visualizer.update_display(
                self.steps,
                visual_agents,
                targets,
                status_dict
            ):
                break  # User closed window

            # Throttle frame rate to keep visualization consistent
            try:
                visualizer.wait_for_next_frame(fps=fps)
            except Exception:
                pass

            # Check termination
            if active == 0:
                termination_reason = "all_evacuated" if evacuated == len(self.sim.agents) else "no_active"
                break

        # Show final state for a moment
        time.sleep(1.0)

        if visualizer:
            visualizer.close()

        # Calculate results
        total = len(self.sim.agents)
        evacuated = sum(1 for a in self.sim.agents if a.status == 'evacuated')
        stuck = sum(1 for a in self.sim.agents if a.status == 'stuck')
        dead = sum(1 for a in self.sim.agents if a.status == 'dead')

        survival_rate = (evacuated + stuck) / total if total > 0 else 0
        avg_evacuation_time = sum(evacuation_times) / len(evacuation_times) if evacuation_times else 0

        # Calculate reward (same as original)
        reward = evacuated * 100.0 - stuck * 10.0 - dead * 20.0 - self.steps * 0.01

        # Calculate fire damage metrics
        fire_exposures = [a.fire_damage for a in self.sim.agents]
        avg_fire_damage = sum(fire_exposures) / len(fire_exposures) if fire_exposures else 0.0

        # Create result
        from fast_simulation import SimResult
        result = SimResult(
            steps=self.steps,
            evacuated=evacuated,
            stuck=stuck,
            dead=dead,
            survival_rate=survival_rate,
            avg_evacuation_time=avg_evacuation_time,
            reward=reward,
            termination_reason=termination_reason,
            avg_fire_damage=avg_fire_damage,
            agent_fire_exposures=fire_exposures
        )

        return result


def load_config_and_create_sim(config_path: str, fire_spread_mode: str = 'always_real') -> tuple:
    """
    Load configuration and create Phase 2 simulation.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Tuple of (FastEvacuationSim, SimulationConfig)
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        json_config = json.load(f)

    config = SimulationConfig.from_json(json_config)

    # Convert to Phase 2 format
    fire_map = np.array(config.initial_fire_map, dtype=np.float32)

    # Extract agent positions
    agent_starts = []
    for pos_str in config.start_positions:
        col, row = stateNameToCoords(pos_str)
        agent_starts.append((col, row))

    # Extract exits from door configs
    exits = []
    fire_starts = []
    if config.door_configs:
        for door in config.door_configs:
            col, row = stateNameToCoords(door['position'])
            if door.get('type') == 'exit':
                exits.append((col, row))

    # Find fire positions
    for row_idx, row in enumerate(fire_map):
        for col_idx, val in enumerate(row):
            if val > 0:
                fire_starts.append((col_idx, row_idx))

    # Create Phase 2 simulation
    sim = FastEvacuationSim(
        grid=fire_map,
        agent_starts=agent_starts,
        exits=exits if exits else [(fire_map.shape[1]-1, fire_map.shape[0]-1)],
        fire_starts=fire_starts if fire_starts else None,
        deterministic_fire=False,  # Use stochastic fire for realistic spread
        fire_update_interval=config.fire_update_interval,
        fire_discovery_delay=config.fire_discovery_delay,
        fire_spread_mode=fire_spread_mode
    )

    return sim, config


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 2 optimized simulation with pygame visualization for validation.\n"
                    "Cell size auto-scales based on map dimensions to fit on screen.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="example_configuration.json",
        help="Path to configuration file (default: example_configuration.json)"
    )
    parser.add_argument(
        "--matlab",
        action="store_true",
        help="Use MATLAB-style visualization (default: pygame)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum simulation steps (default: 500)"
    )
    parser.add_argument(
        "--fire-interval",
        type=int,
        default=None,
        help="Fire update interval in timesteps (default: use config value)"
    )
    parser.add_argument(
        "--fire-spread-mode",
        type=str,
        default="always_real",
        choices=["always_real", "real_then_simple", "real_then_stop"],
        help="Fire spread mode: 'always_real' (continuous stochastic spread, most realistic), "
             "'real_then_simple' (stochastic spread until stable, then intensity growth only), "
             "'real_then_stop' (stochastic spread until stable, then completely static) (default: always_real)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frame rate for pygame visualization (frames per second, default: 10)"
    )

    args = parser.parse_args()

    print("="*60)
    print("Phase 2 Visual Validation")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Visualization: {'MATLAB-style' if args.matlab else 'Pygame'}")
    print(f"Max steps: {args.max_steps}")
    print(f"FPS: {args.fps}")
    print(f"Fire spread mode: {args.fire_spread_mode}")
    print("="*60)
    print()

    # Load configuration and create simulation
    sim, config = load_config_and_create_sim(args.config, fire_spread_mode=args.fire_spread_mode)

    # Override fire interval if specified
    if args.fire_interval is not None:
        sim.fire_update_interval = args.fire_interval
        print(f"Using custom fire update interval: {args.fire_interval} timesteps")

    # Create wrapper with visualization support
    wrapper = VisualPhase2Wrapper(sim, config)

    # Run with visualization
    print("\nStarting simulation with visualization...")
    print("Close the visualization window to stop.\n")

    result = wrapper.run_with_visualization(
        max_steps=args.max_steps,
        use_pygame=(not args.matlab),
        use_matlab=args.matlab,
        fps=args.fps
    )

    if result:
        # Print results
        print("\n" + "="*60)
        print("SIMULATION RESULTS")
        print("="*60)
        print(f"Total steps: {result.steps}")
        print(f"Evacuated: {result.evacuated}")
        print(f"Stuck: {result.stuck}")
        print(f"Dead: {result.dead}")
        print(f"Survival rate: {result.survival_rate*100:.2f}%")
        print(f"Avg evacuation time: {result.avg_evacuation_time:.2f} steps")
        print(f"Avg fire damage: {result.avg_fire_damage:.4f}")
        print(f"Reward: {result.reward:.2f}")
        print(f"Termination reason: {result.termination_reason}")
        print("="*60)


if __name__ == "__main__":
    main()
