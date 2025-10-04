"""
MATLAB-based Advanced Visualization for Fire Evacuation Simulation
==================================================================

Provides interactive visualization of environmental data using MATLAB engine:
- Temperature maps with interpolation
- Oxygen level maps with interpolation
- Smoke density maps
- Fuel consumption maps
- Fire intensity overlay
- Agent trajectories with offset to avoid overlap
- Interactive checkboxes to toggle data layers
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive GUI
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from scipy.interpolate import griddata
from d_star_lite.utils import stateNameToCoords


class MatlabStyleVisualizer:
    """
    Advanced visualizer for fire evacuation simulation with environmental data.
    Uses matplotlib with MATLAB-style interpolation and interactive controls.
    """

    def __init__(self, rows: int, cols: int, fire_model, trajectory_length: int = 10):
        """
        Initialize the MATLAB-style visualizer.

        Args:
            rows: Number of grid rows
            cols: Number of grid columns
            fire_model: Fire model instance for accessing environmental data
            trajectory_length: Number of past positions to show for each agent
        """
        self.rows = rows
        self.cols = cols
        self.fire_model = fire_model
        self.trajectory_length = trajectory_length

        # Track agent trajectories
        self.agent_trajectories = {}  # {agent_id: deque of (x, y) positions}

        # Setup figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.setup_layout()

        # Data visibility toggles
        self.show_temperature = True
        self.show_oxygen = False
        self.show_smoke = False
        self.show_fuel = False
        self.show_fire = True
        self.show_trajectories = True

        # Create interactive checkboxes
        self.setup_checkboxes()

        # Colormap configurations
        self.temp_cmap = plt.cm.hot
        self.oxygen_cmap = plt.cm.Blues_r
        self.smoke_cmap = plt.cm.gray
        self.fuel_cmap = plt.cm.Greens

        # Agent colors (distinct colors for different agents)
        self.agent_colors = plt.cm.tab10(np.linspace(0, 1, 10))

        plt.ion()  # Interactive mode

    def setup_layout(self):
        """Create the layout with main plot and checkbox area"""
        # Main visualization area
        self.ax_main = plt.subplot2grid((10, 10), (0, 0), colspan=8, rowspan=10)

        # Checkbox area
        self.ax_checks = plt.subplot2grid((10, 10), (0, 8), colspan=2, rowspan=3)
        self.ax_checks.axis('off')

        # Colorbar area
        self.ax_colorbar = plt.subplot2grid((10, 10), (3, 8), colspan=2, rowspan=6)

        # Stats area
        self.ax_stats = plt.subplot2grid((10, 10), (9, 8), colspan=2, rowspan=1)
        self.ax_stats.axis('off')

    def setup_checkboxes(self):
        """Create interactive checkboxes for toggling visualization layers"""
        labels = [
            'Temperature',
            'Oxygen',
            'Smoke',
            'Fuel',
            'Fire',
            'Trajectories'
        ]

        # Initial visibility states
        visibility = [
            self.show_temperature,
            self.show_oxygen,
            self.show_smoke,
            self.show_fuel,
            self.show_fire,
            self.show_trajectories
        ]

        # Create checkboxes
        self.check_buttons = CheckButtons(
            self.ax_checks,
            labels,
            visibility
        )

        # Connect callback
        self.check_buttons.on_clicked(self.on_checkbox_clicked)

    def on_checkbox_clicked(self, label):
        """Handle checkbox toggle events"""
        if label == 'Temperature':
            self.show_temperature = not self.show_temperature
        elif label == 'Oxygen':
            self.show_oxygen = not self.show_oxygen
        elif label == 'Smoke':
            self.show_smoke = not self.show_smoke
        elif label == 'Fuel':
            self.show_fuel = not self.show_fuel
        elif label == 'Fire':
            self.show_fire = not self.show_fire
        elif label == 'Trajectories':
            self.show_trajectories = not self.show_trajectories

    def interpolate_data(self, data: np.ndarray, scale_factor: int = 4) -> np.ndarray:
        """
        Interpolate grid data for smooth visualization using cubic interpolation.

        Args:
            data: 2D array of data values
            scale_factor: Interpolation upscaling factor

        Returns:
            Interpolated 2D array
        """
        rows, cols = data.shape

        # Create original grid points
        x_orig = np.arange(cols)
        y_orig = np.arange(rows)
        X_orig, Y_orig = np.meshgrid(x_orig, y_orig)

        # Create fine grid for interpolation
        x_fine = np.linspace(0, cols - 1, cols * scale_factor)
        y_fine = np.linspace(0, rows - 1, rows * scale_factor)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

        # Flatten original data
        points = np.column_stack([X_orig.ravel(), Y_orig.ravel()])
        values = data.ravel()

        # Interpolate using cubic method
        Z_fine = griddata(points, values, (X_fine, Y_fine), method='cubic')

        # Fill any NaN values with nearest neighbor
        if np.any(np.isnan(Z_fine)):
            Z_nearest = griddata(points, values, (X_fine, Y_fine), method='nearest')
            Z_fine = np.where(np.isnan(Z_fine), Z_nearest, Z_fine)

        return Z_fine

    def update_agent_trajectories(self, agents: List[Any]):
        """
        Update tracked trajectories for all agents.

        Args:
            agents: List of EvacuationAgent instances
        """
        current_agent_ids = {agent.id for agent in agents}

        # Remove trajectories for evacuated/stuck agents
        ids_to_remove = [aid for aid in self.agent_trajectories if aid not in current_agent_ids]
        for aid in ids_to_remove:
            del self.agent_trajectories[aid]

        # Update current agent positions
        for agent in agents:
            if agent.id not in self.agent_trajectories:
                self.agent_trajectories[agent.id] = deque(maxlen=self.trajectory_length)

            # Get current position
            try:
                coords = stateNameToCoords(agent.s_current)
                self.agent_trajectories[agent.id].append(coords)
            except:
                pass

    def draw_agent_trajectories(self):
        """
        Draw agent trajectories with small offsets to avoid overlapping paths.
        Uses different colors for each agent.
        """
        if not self.show_trajectories:
            return

        for agent_id, trajectory in self.agent_trajectories.items():
            if len(trajectory) < 2:
                continue

            # Get color for this agent
            color = self.agent_colors[agent_id % len(self.agent_colors)]

            # Add small offset based on agent_id to avoid complete overlap
            offset_x = (agent_id % 3 - 1) * 0.15
            offset_y = (agent_id // 3 % 3 - 1) * 0.15

            # Convert trajectory to arrays
            xs = [pos[0] + offset_x for pos in trajectory]
            ys = [pos[1] + offset_y for pos in trajectory]

            # Draw trajectory line with decreasing opacity
            for i in range(len(xs) - 1):
                alpha = 0.3 + 0.7 * (i / len(xs))  # Fade older positions
                self.ax_main.plot(
                    [xs[i], xs[i+1]],
                    [ys[i], ys[i+1]],
                    color=color,
                    linewidth=2,
                    alpha=alpha,
                    zorder=10
                )

            # Draw agent current position as a circle
            if len(trajectory) > 0:
                self.ax_main.scatter(
                    xs[-1], ys[-1],
                    s=200,
                    c=[color],
                    marker='o',
                    edgecolors='white',
                    linewidths=2,
                    zorder=15,
                    alpha=0.9
                )

                # Add agent ID label
                self.ax_main.text(
                    xs[-1], ys[-1],
                    str(agent_id),
                    color='white',
                    fontsize=10,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    zorder=16
                )

    def update_display(self, step: int, agents: List[Any], targets: List[str],
                      fire_state: np.ndarray, status: Dict[str, Any]):
        """
        Update the visualization display.

        Args:
            step: Current simulation step
            agents: List of active agents
            targets: List of target positions
            fire_state: Current fire state grid
            status: Simulation status dictionary
        """
        # Update agent trajectories
        self.update_agent_trajectories(agents)

        # Clear main axis
        self.ax_main.clear()
        self.ax_colorbar.clear()

        # Determine which data to display (priority order)
        primary_data = None
        primary_label = ""
        primary_cmap = None
        vmin, vmax = 0, 1

        if self.show_temperature:
            temp_data = np.array(self.fire_model.temperature_map)
            primary_data = self.interpolate_data(temp_data)
            primary_label = "Temperature (°C)"
            primary_cmap = self.temp_cmap
            vmin, vmax = self.fire_model.env.temperature, np.max(temp_data)
        elif self.show_oxygen:
            oxygen_data = np.array(self.fire_model.oxygen_map)
            primary_data = self.interpolate_data(oxygen_data)
            primary_label = "Oxygen Level (%)"
            primary_cmap = self.oxygen_cmap
            vmin, vmax = 0, 21
        elif self.show_smoke:
            smoke_data = np.array(self.fire_model.smoke_density)
            primary_data = self.interpolate_data(smoke_data)
            primary_label = "Smoke Density"
            primary_cmap = self.smoke_cmap
            vmin, vmax = 0, np.max(smoke_data) if np.max(smoke_data) > 0 else 1
        elif self.show_fuel:
            fuel_data = np.array(self.fire_model.fuel_map)
            primary_data = self.interpolate_data(fuel_data)
            primary_label = "Fuel Remaining"
            primary_cmap = self.fuel_cmap
            vmin, vmax = 0, 1

        # Display primary data layer
        if primary_data is not None:
            extent = [0, self.cols, self.rows, 0]
            im = self.ax_main.imshow(
                primary_data,
                cmap=primary_cmap,
                interpolation='bilinear',
                extent=extent,
                alpha=0.8,
                vmin=vmin,
                vmax=vmax,
                aspect='auto'
            )

            # Add colorbar
            cbar = plt.colorbar(im, cax=self.ax_colorbar)
            cbar.set_label(primary_label, rotation=270, labelpad=20)

        # Overlay fire intensity if enabled
        if self.show_fire:
            fire_data = np.array(fire_state)
            # Mask non-fire cells
            fire_masked = np.ma.masked_where(
                (fire_data <= 0) | (fire_data == -2),
                fire_data
            )

            if primary_data is None:
                # Fire is the only layer
                extent = [0, self.cols, self.rows, 0]
                im = self.ax_main.imshow(
                    fire_masked,
                    cmap=plt.cm.hot,
                    interpolation='bilinear',
                    extent=extent,
                    alpha=0.9,
                    vmin=0,
                    vmax=4,
                    aspect='auto'
                )
                cbar = plt.colorbar(im, cax=self.ax_colorbar)
                cbar.set_label("Fire Intensity", rotation=270, labelpad=20)
            else:
                # Fire as overlay
                fire_interp = self.interpolate_data(fire_masked.filled(0))
                fire_interp_masked = np.ma.masked_where(fire_interp < 0.1, fire_interp)

                extent = [0, self.cols, self.rows, 0]
                self.ax_main.imshow(
                    fire_interp_masked,
                    cmap=plt.cm.YlOrRd,
                    interpolation='bilinear',
                    extent=extent,
                    alpha=0.6,
                    vmin=0,
                    vmax=4,
                    aspect='auto'
                )

        # Draw obstacles (walls)
        fire_array = np.array(fire_state)
        for i in range(self.rows):
            for j in range(self.cols):
                if fire_array[i, j] == -2:
                    self.ax_main.add_patch(
                        plt.Rectangle(
                            (j, i), 1, 1,
                            facecolor='gray',
                            edgecolor='black',
                            linewidth=1,
                            alpha=0.8,
                            zorder=5
                        )
                    )

        # Draw targets
        for idx, target in enumerate(targets):
            try:
                tx, ty = stateNameToCoords(target)
                self.ax_main.scatter(
                    tx, ty,
                    s=300,
                    c='lime',
                    marker='*',
                    edgecolors='black',
                    linewidths=2,
                    zorder=12,
                    label=f'Target {idx+1}' if idx == 0 else ''
                )
            except:
                pass

        # Draw agent trajectories
        self.draw_agent_trajectories()

        # Configure axes
        self.ax_main.set_xlim(0, self.cols)
        self.ax_main.set_ylim(self.rows, 0)
        self.ax_main.set_xlabel('Column (X)', fontsize=12)
        self.ax_main.set_ylabel('Row (Y)', fontsize=12)
        self.ax_main.set_title(
            f'Fire Evacuation Simulation - Step {step}',
            fontsize=14,
            fontweight='bold'
        )
        self.ax_main.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Update stats display
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        stats_text = (
            f"Active Agents: {status['remaining_agents']}\n"
            f"Evacuated: {status['evacuated_agents']}/{status['total_agents']}"
        )
        self.ax_stats.text(
            0.5, 0.5, stats_text,
            ha='center', va='center',
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        return True

    def wait_for_next_frame(self, fps: int = 5):
        """
        Wait for next frame based on desired FPS.

        Args:
            fps: Frames per second
        """
        import time
        time.sleep(1.0 / fps)

    def close(self):
        """Close the visualization"""
        plt.ioff()
        plt.close(self.fig)


# Convenience function for creating visualizer
def create_matlab_visualizer(rows: int, cols: int, fire_model,
                             trajectory_length: int = 10) -> MatlabStyleVisualizer:
    """
    Create a MATLAB-style visualizer instance.

    Args:
        rows: Grid rows
        cols: Grid columns
        fire_model: Fire model instance
        trajectory_length: Number of past positions to track

    Returns:
        MatlabStyleVisualizer instance
    """
    return MatlabStyleVisualizer(rows, cols, fire_model, trajectory_length)
