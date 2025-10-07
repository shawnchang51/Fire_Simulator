import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
from simulation import SimulationConfig, EvacuationSimulation

class VisualConfigurator:
    def __init__(self, root):
        self.root = root
        self.root.title("Evacuation Simulation Configurator")
        self.root.geometry("1600x1000")

        # Configuration variables - updated defaults for 0.3m cell model
        self.config_vars = {
            'map_rows': tk.IntVar(value=60),
            'map_cols': tk.IntVar(value=60),
            'max_occupancy': tk.IntVar(value=1),
            'agent_num': tk.IntVar(value=5),
            'viewing_range': tk.IntVar(value=10),
            'cell_size': tk.DoubleVar(value=0.3),
            'timestep_duration': tk.DoubleVar(value=0.5),
            'fire_update_interval': tk.IntVar(value=4),
            'fire_model_type': tk.StringVar(value='realistic')
        }

        # Lists for positions and targets
        self.start_positions = []
        self.targets = []
        self.fire_positions = []
        self.obstacle_positions = []

        # Interactive map variables - smaller cell size for larger grids
        self.cell_size = 10  # Reduced from 20 for 60x60 grids
        self.map_canvas = None
        self.current_tool = "agent"  # "agent", "target", "fire", "obstacle", "erase"

        self.create_widgets()

    def create_widgets(self):
        # Create main container with paned window for resizable layout
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side - Configuration panel
        config_panel = ttk.Frame(main_paned)
        main_paned.add(config_panel, weight=1)

        # Right side - Interactive Map
        map_panel = ttk.Frame(main_paned)
        main_paned.add(map_panel, weight=2)

        # Create configuration scrollable area
        canvas = tk.Canvas(config_panel)
        scrollbar = ttk.Scrollbar(config_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create interactive map
        self.create_interactive_map(map_panel)

        # Basic Configuration Section
        config_frame = ttk.LabelFrame(scrollable_frame, text="Basic Configuration", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))

        # Map dimensions
        ttk.Label(config_frame, text="Map Rows:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(config_frame, from_=5, to=100, textvariable=self.config_vars['map_rows'], width=10).grid(row=0, column=1, padx=(5, 20), pady=2)

        ttk.Label(config_frame, text="Map Columns:").grid(row=0, column=2, sticky=tk.W, pady=2)
        ttk.Spinbox(config_frame, from_=5, to=100, textvariable=self.config_vars['map_cols'], width=10).grid(row=0, column=3, padx=(5, 0), pady=2)

        # Agent configuration
        ttk.Label(config_frame, text="Number of Agents:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(config_frame, from_=1, to=50, textvariable=self.config_vars['agent_num'], width=10).grid(row=1, column=1, padx=(5, 20), pady=2)

        ttk.Label(config_frame, text="Max Occupancy:").grid(row=1, column=2, sticky=tk.W, pady=2)
        ttk.Spinbox(config_frame, from_=1, to=10, textvariable=self.config_vars['max_occupancy'], width=10).grid(row=1, column=3, padx=(5, 0), pady=2)

        ttk.Label(config_frame, text="Viewing Range:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(config_frame, from_=1, to=20, textvariable=self.config_vars['viewing_range'], width=10).grid(row=2, column=1, padx=(5, 0), pady=2)

        # Physics/Temporal Configuration Section
        physics_frame = ttk.LabelFrame(scrollable_frame, text="Physics & Fire Model Configuration", padding="10")
        physics_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(physics_frame, text="Cell Size (m):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(physics_frame, from_=0.1, to=2.0, increment=0.1, textvariable=self.config_vars['cell_size'], width=10).grid(row=0, column=1, padx=(5, 20), pady=2)

        ttk.Label(physics_frame, text="Timestep (s):").grid(row=0, column=2, sticky=tk.W, pady=2)
        ttk.Spinbox(physics_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.config_vars['timestep_duration'], width=10).grid(row=0, column=3, padx=(5, 0), pady=2)

        ttk.Label(physics_frame, text="Fire Update Interval:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(physics_frame, from_=1, to=20, textvariable=self.config_vars['fire_update_interval'], width=10).grid(row=1, column=1, padx=(5, 20), pady=2)

        ttk.Label(physics_frame, text="Fire Model:").grid(row=1, column=2, sticky=tk.W, pady=2)
        fire_model_combo = ttk.Combobox(physics_frame, textvariable=self.config_vars['fire_model_type'],
                                        values=['realistic', 'aggressive', 'default'], state='readonly', width=12)
        fire_model_combo.grid(row=1, column=3, padx=(5, 0), pady=2)

        # Info label for fire models
        info_text = "Realistic: 3-6min flashover | Aggressive: 30-60s flashover | Default: Original"
        ttk.Label(physics_frame, text=info_text, font=('Arial', 8), foreground='gray').grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))

        # Start Positions Section
        positions_frame = ttk.LabelFrame(scrollable_frame, text="Agent Start Positions", padding="10")
        positions_frame.pack(fill=tk.X, pady=(0, 10))

        pos_control_frame = ttk.Frame(positions_frame)
        pos_control_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(pos_control_frame, text="Add Random Positions", command=self.add_random_positions).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(pos_control_frame, text="Clear Positions", command=self.clear_positions).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(pos_control_frame, text="Add Manual Position", command=self.add_manual_position).pack(side=tk.LEFT)

        self.positions_listbox = tk.Listbox(positions_frame, height=5)
        self.positions_listbox.pack(fill=tk.X, pady=5)

        # Targets Section
        targets_frame = ttk.LabelFrame(scrollable_frame, text="Target Positions", padding="10")
        targets_frame.pack(fill=tk.X, pady=(0, 10))

        target_control_frame = ttk.Frame(targets_frame)
        target_control_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(target_control_frame, text="Add Random Targets", command=self.add_random_targets).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(target_control_frame, text="Clear Targets", command=self.clear_targets).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(target_control_frame, text="Add Manual Target", command=self.add_manual_target).pack(side=tk.LEFT)

        self.targets_listbox = tk.Listbox(targets_frame, height=5)
        self.targets_listbox.pack(fill=tk.X, pady=5)

        # Fire Configuration Section
        fire_frame = ttk.LabelFrame(scrollable_frame, text="Initial Fire Configuration", padding="10")
        fire_frame.pack(fill=tk.X, pady=(0, 10))

        fire_control_frame = ttk.Frame(fire_frame)
        fire_control_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(fire_control_frame, text="Add Random Fire", command=self.add_random_fire).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(fire_control_frame, text="Clear Fire", command=self.clear_fire).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(fire_control_frame, text="Add Manual Fire", command=self.add_manual_fire).pack(side=tk.LEFT)

        self.fire_listbox = tk.Listbox(fire_frame, height=4)
        self.fire_listbox.pack(fill=tk.X, pady=5)

        # Actions Section
        actions_frame = ttk.LabelFrame(scrollable_frame, text="Actions", padding="10")
        actions_frame.pack(fill=tk.X, pady=(0, 10))

        button_frame = ttk.Frame(actions_frame)
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="Load Configuration", command=self.load_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Save Configuration", command=self.save_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Run Simulation", command=self.run_simulation).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Validate Config", command=self.validate_config).pack(side=tk.LEFT)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def create_interactive_map(self, parent):
        # Map control frame
        control_frame = ttk.LabelFrame(parent, text="Interactive Map", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Tool selection
        tool_frame = ttk.Frame(control_frame)
        tool_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(tool_frame, text="Drawing Tool:").pack(side=tk.LEFT, padx=(0, 10))

        self.tool_var = tk.StringVar(value="agent")
        tools = [("Agent Start", "agent"), ("Target", "target"), ("Fire", "fire"), ("Obstacle", "obstacle"), ("Erase", "erase")]

        for text, value in tools:
            ttk.Radiobutton(tool_frame, text=text, variable=self.tool_var, value=value,
                           command=self.on_tool_change).pack(side=tk.LEFT, padx=(0, 10))

        # Map controls
        map_control_frame = ttk.Frame(control_frame)
        map_control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(map_control_frame, text="Clear Map", command=self.clear_map).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(map_control_frame, text="Refresh Map", command=self.refresh_map).pack(side=tk.LEFT, padx=(0, 5))

        # Add map dimension update button
        ttk.Button(map_control_frame, text="Update Map Size", command=self.update_map_size).pack(side=tk.LEFT, padx=(0, 5))

        # Zoom control for larger grids
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(zoom_frame, text="Canvas Cell Size (Zoom):").pack(side=tk.LEFT, padx=(0, 10))
        self.zoom_var = tk.IntVar(value=self.cell_size)
        zoom_scale = ttk.Scale(zoom_frame, from_=5, to=30, orient=tk.HORIZONTAL,
                              variable=self.zoom_var, command=self.on_zoom_change)
        zoom_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.zoom_label = ttk.Label(zoom_frame, text=f"{self.cell_size}px")
        self.zoom_label.pack(side=tk.LEFT)

        # Map canvas frame with scrollbars
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas with scrollbars
        self.map_canvas = tk.Canvas(canvas_frame, bg="white", scrollregion=(0, 0, 0, 0))

        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.map_canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.map_canvas.yview)

        self.map_canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        # Pack scrollbars and canvas
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.map_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind events
        self.map_canvas.bind("<Button-1>", self.on_map_click)
        self.map_canvas.bind("<B1-Motion>", self.on_map_drag)

        # Legend
        legend_frame = ttk.LabelFrame(parent, text="Legend", padding="5")
        legend_frame.pack(fill=tk.X, pady=(10, 0))

        legend_text = "🟦 Agent Start  🎯 Target  🔥 Fire  ⬛ Obstacle (-2)  ⬜ Empty"
        ttk.Label(legend_frame, text=legend_text).pack()

        # Initialize map with walls
        self.create_initial_walls()
        self.refresh_map()

    def on_tool_change(self):
        self.current_tool = self.tool_var.get()

    def on_zoom_change(self, value):
        """Handle zoom slider changes"""
        self.cell_size = int(float(value))
        self.zoom_label.config(text=f"{self.cell_size}px")
        self.refresh_map()

    def update_map_size(self):
        """Update map canvas size when dimensions change"""
        self.create_initial_walls()
        self.refresh_map()

    def on_map_click(self, event):
        """Handle mouse clicks on the map"""
        # Convert screen coordinates to grid coordinates
        canvas_x = self.map_canvas.canvasx(event.x)
        canvas_y = self.map_canvas.canvasy(event.y)

        grid_x = int(canvas_x // self.cell_size)
        grid_y = int(canvas_y // self.cell_size)

        # Check bounds
        rows = self.config_vars['map_rows'].get()
        cols = self.config_vars['map_cols'].get()

        if 0 <= grid_x < cols and 0 <= grid_y < rows:
            self.handle_cell_click(grid_x, grid_y)

    def on_map_drag(self, event):
        """Handle mouse drag on the map"""
        self.on_map_click(event)

    def handle_cell_click(self, x, y):
        """Handle clicking on a specific cell"""
        position = f'x{x}y{y}'

        if self.current_tool == "agent":
            # Add agent start position
            if position not in self.start_positions:
                self.start_positions.append(position)
                self.positions_listbox.insert(tk.END, f"Agent {len(self.start_positions)}: {position}")
                # Update agent number if needed
                if len(self.start_positions) > self.config_vars['agent_num'].get():
                    self.config_vars['agent_num'].set(len(self.start_positions))

        elif self.current_tool == "target":
            # Add target position
            if position not in self.targets:
                self.targets.append(position)
                self.targets_listbox.insert(tk.END, f"Target {len(self.targets)}: {position}")

        elif self.current_tool == "fire":
            # Add fire position with default intensity 2.0 (growth phase - spreads effectively)
            fire_tuple = (position, 2.0)
            # Remove existing fire at this position
            self.fire_positions = [f for f in self.fire_positions if f[0] != position]
            self.fire_positions.append(fire_tuple)
            self.refresh_fire_listbox()

        elif self.current_tool == "obstacle":
            # Add obstacle position
            if position not in self.obstacle_positions:
                self.obstacle_positions.append(position)

        elif self.current_tool == "erase":
            # Remove any items at this position
            self.start_positions = [pos for pos in self.start_positions if pos != position]
            self.targets = [pos for pos in self.targets if pos != position]
            self.fire_positions = [f for f in self.fire_positions if f[0] != position]
            self.obstacle_positions = [pos for pos in self.obstacle_positions if pos != position]

            self.refresh_all_listboxes()
            # Update agent number
            self.config_vars['agent_num'].set(len(self.start_positions))

        # Refresh the map display
        self.refresh_map()

    def refresh_all_listboxes(self):
        """Refresh all listboxes after changes"""
        # Refresh positions
        self.positions_listbox.delete(0, tk.END)
        for i, pos in enumerate(self.start_positions):
            self.positions_listbox.insert(tk.END, f"Agent {i+1}: {pos}")

        # Refresh targets
        self.targets_listbox.delete(0, tk.END)
        for i, target in enumerate(self.targets):
            self.targets_listbox.insert(tk.END, f"Target {i+1}: {target}")

        # Refresh fire
        self.refresh_fire_listbox()

    def refresh_fire_listbox(self):
        """Refresh fire listbox"""
        self.fire_listbox.delete(0, tk.END)
        for i, (pos, intensity) in enumerate(self.fire_positions):
            self.fire_listbox.insert(tk.END, f"Fire {i+1}: {pos} (intensity: {intensity})")

    def clear_map(self):
        """Clear all items from the map"""
        self.start_positions.clear()
        self.targets.clear()
        self.fire_positions.clear()
        self.obstacle_positions.clear()
        self.config_vars['agent_num'].set(0)
        self.refresh_all_listboxes()
        self.create_initial_walls()
        self.refresh_map()

    def create_initial_walls(self):
        """Create walls (obstacles) around the perimeter of the map"""
        rows = self.config_vars['map_rows'].get()
        cols = self.config_vars['map_cols'].get()

        self.obstacle_positions.clear()

        # Top and bottom walls
        for x in range(cols):
            self.obstacle_positions.append(f'x{x}y0')  # Top wall
            self.obstacle_positions.append(f'x{x}y{rows-1}')  # Bottom wall

        # Left and right walls (excluding corners to avoid duplicates)
        for y in range(1, rows-1):
            self.obstacle_positions.append(f'x0y{y}')  # Left wall
            self.obstacle_positions.append(f'x{cols-1}y{y}')  # Right wall

    def refresh_map(self):
        """Refresh the map display"""
        if not self.map_canvas:
            return

        # Clear canvas
        self.map_canvas.delete("all")

        rows = self.config_vars['map_rows'].get()
        cols = self.config_vars['map_cols'].get()

        # Update canvas size and scroll region
        canvas_width = cols * self.cell_size
        canvas_height = rows * self.cell_size
        self.map_canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))

        # Draw grid
        for i in range(cols + 1):
            x = i * self.cell_size
            self.map_canvas.create_line(x, 0, x, canvas_height, fill="lightgray", width=1)

        for i in range(rows + 1):
            y = i * self.cell_size
            self.map_canvas.create_line(0, y, canvas_width, y, fill="lightgray", width=1)

        # Draw agents
        for pos in self.start_positions:
            self.draw_cell(pos, "lightblue", "🟦")

        # Draw targets
        for pos in self.targets:
            self.draw_cell(pos, "lightgreen", "🎯")

        # Draw fire
        for pos, intensity in self.fire_positions:
            # Color intensity based on fire strength
            red_intensity = int(255 * intensity)
            color = f"#{red_intensity:02x}4040"
            self.draw_cell(pos, color, "🔥")

        # Draw obstacles
        for pos in self.obstacle_positions:
            self.draw_cell(pos, "black", "⬛")

    def draw_cell(self, position, color, symbol):
        """Draw a cell on the map"""
        try:
            x_part, y_part = position.split('y')
            x = int(x_part[1:])
            y = int(y_part)

            x1 = x * self.cell_size
            y1 = y * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size

            # Draw colored rectangle
            self.map_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black", width=1)

            # Draw symbol in center
            center_x = x1 + self.cell_size // 2
            center_y = y1 + self.cell_size // 2
            self.map_canvas.create_text(center_x, center_y, text=symbol, font=("Arial", 8))

        except (ValueError, IndexError):
            pass

    def add_random_positions(self):
        num_agents = self.config_vars['agent_num'].get()
        rows = self.config_vars['map_rows'].get()
        cols = self.config_vars['map_cols'].get()

        self.start_positions.clear()
        self.positions_listbox.delete(0, tk.END)

        for i in range(num_agents):
            x = random.randint(0, cols - 1)
            y = random.randint(0, rows - 1)
            position = f'x{x}y{y}'
            self.start_positions.append(position)
            self.positions_listbox.insert(tk.END, f"Agent {i+1}: {position}")

        self.refresh_map()

    def clear_positions(self):
        self.start_positions.clear()
        self.positions_listbox.delete(0, tk.END)
        self.refresh_map()

    def add_manual_position(self):
        dialog = PositionDialog(self.root, self.config_vars['map_rows'].get(), self.config_vars['map_cols'].get())
        if dialog.result:
            self.start_positions.append(dialog.result)
            self.positions_listbox.insert(tk.END, f"Agent {len(self.start_positions)}: {dialog.result}")
            self.refresh_map()

    def add_random_targets(self):
        rows = self.config_vars['map_rows'].get()
        cols = self.config_vars['map_cols'].get()

        self.targets.clear()
        self.targets_listbox.delete(0, tk.END)

        # Add 3-7 random targets
        num_targets = random.randint(3, 7)
        for i in range(num_targets):
            x = random.randint(0, cols - 1)
            y = random.randint(0, rows - 1)
            target = f'x{x}y{y}'
            self.targets.append(target)
            self.targets_listbox.insert(tk.END, f"Target {i+1}: {target}")

        self.refresh_map()

    def clear_targets(self):
        self.targets.clear()
        self.targets_listbox.delete(0, tk.END)
        self.refresh_map()

    def add_manual_target(self):
        dialog = PositionDialog(self.root, self.config_vars['map_rows'].get(), self.config_vars['map_cols'].get())
        if dialog.result:
            self.targets.append(dialog.result)
            self.targets_listbox.insert(tk.END, f"Target {len(self.targets)}: {dialog.result}")
            self.refresh_map()

    def add_random_fire(self):
        rows = self.config_vars['map_rows'].get()
        cols = self.config_vars['map_cols'].get()

        self.fire_positions.clear()
        self.fire_listbox.delete(0, tk.END)

        # Add 1-3 random fire locations
        num_fires = random.randint(1, 3)
        for i in range(num_fires):
            x = random.randint(0, cols - 1)
            y = random.randint(0, rows - 1)
            intensity = round(random.uniform(0.3, 0.8), 2)
            fire_info = (f'x{x}y{y}', intensity)
            self.fire_positions.append(fire_info)
            self.fire_listbox.insert(tk.END, f"Fire {i+1}: {fire_info[0]} (intensity: {fire_info[1]})")

        self.refresh_map()

    def clear_fire(self):
        self.fire_positions.clear()
        self.fire_listbox.delete(0, tk.END)
        self.refresh_map()

    def add_manual_fire(self):
        dialog = FireDialog(self.root, self.config_vars['map_rows'].get(), self.config_vars['map_cols'].get())
        if dialog.result:
            self.fire_positions.append(dialog.result)
            self.fire_listbox.insert(tk.END, f"Fire {len(self.fire_positions)}: {dialog.result[0]} (intensity: {dialog.result[1]})")
            self.refresh_map()

    def validate_config(self):
        errors = []

        # Check if we have enough start positions
        if len(self.start_positions) != self.config_vars['agent_num'].get():
            errors.append(f"Need {self.config_vars['agent_num'].get()} start positions, but have {len(self.start_positions)}")

        # Check if we have at least one target
        if len(self.targets) == 0:
            errors.append("At least one target position is required")

        # Check for valid positions
        rows = self.config_vars['map_rows'].get()
        cols = self.config_vars['map_cols'].get()

        for pos in self.start_positions + self.targets:
            if not self._is_valid_position(pos, rows, cols):
                errors.append(f"Invalid position: {pos}")

        if errors:
            messagebox.showerror("Configuration Errors", "\n".join(errors))
            return False
        else:
            messagebox.showinfo("Validation", "Configuration is valid!")
            return True

    def _is_valid_position(self, pos, rows, cols):
        try:
            # Extract x and y from format 'x#y#'
            x_part, y_part = pos.split('y')
            x = int(x_part[1:])  # Remove 'x' prefix
            y = int(y_part)
            return 0 <= x < cols and 0 <= y < rows
        except:
            return False

    def save_config(self):
        if not self.validate_config():
            return

        # Create initial fire map with obstacles
        rows = self.config_vars['map_rows'].get()
        cols = self.config_vars['map_cols'].get()
        initial_fire_map = [[0 for _ in range(cols)] for _ in range(rows)]

        # Add obstacles to the map with value -2
        for pos in self.obstacle_positions:
            try:
                x_part, y_part = pos.split('y')
                x = int(x_part[1:])
                y = int(y_part)
                if 0 <= x < cols and 0 <= y < rows:
                    initial_fire_map[y][x] = -2
            except:
                continue

        # Add fire to the map
        for pos, intensity in self.fire_positions:
            try:
                x_part, y_part = pos.split('y')
                x = int(x_part[1:])
                y = int(y_part)
                if 0 <= x < cols and 0 <= y < rows:
                    initial_fire_map[y][x] = intensity
            except:
                continue

        config_data = {
            'map_rows': self.config_vars['map_rows'].get(),
            'map_cols': self.config_vars['map_cols'].get(),
            'max_occupancy': self.config_vars['max_occupancy'].get(),
            'start_positions': self.start_positions,
            'targets': self.targets,
            'initial_fire_map': initial_fire_map,
            'agent_num': self.config_vars['agent_num'].get(),
            'viewing_range': self.config_vars['viewing_range'].get(),
            'cell_size': self.config_vars['cell_size'].get(),
            'timestep_duration': self.config_vars['timestep_duration'].get(),
            'fire_update_interval': self.config_vars['fire_update_interval'].get(),
            'fire_model_type': self.config_vars['fire_model_type'].get()
        }

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(config_data, f, indent=2)
                messagebox.showinfo("Success", f"Configuration saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def load_config(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'r') as f:
                    config_data = json.load(f)

                # Load basic configuration
                for key, var in self.config_vars.items():
                    if key in config_data:
                        var.set(config_data[key])

                # Load start positions
                self.start_positions = config_data.get('start_positions', [])
                self.positions_listbox.delete(0, tk.END)
                for i, pos in enumerate(self.start_positions):
                    self.positions_listbox.insert(tk.END, f"Agent {i+1}: {pos}")

                # Load targets
                self.targets = config_data.get('targets', [])
                self.targets_listbox.delete(0, tk.END)
                for i, target in enumerate(self.targets):
                    self.targets_listbox.insert(tk.END, f"Target {i+1}: {target}")

                # Load fire positions and obstacles
                self.fire_positions.clear()
                self.obstacle_positions.clear()
                self.fire_listbox.delete(0, tk.END)
                initial_fire_map = config_data.get('initial_fire_map', [])
                fire_count = 0
                for y, row in enumerate(initial_fire_map):
                    for x, intensity in enumerate(row):
                        pos = f'x{x}y{y}'
                        if intensity > 0:
                            fire_count += 1
                            self.fire_positions.append((pos, intensity))
                            self.fire_listbox.insert(tk.END, f"Fire {fire_count}: {pos} (intensity: {intensity})")
                        elif intensity == -2:
                            self.obstacle_positions.append(pos)

                messagebox.showinfo("Success", f"Configuration loaded from {filename}")
                self.refresh_map()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def run_simulation(self):
        if not self.validate_config():
            return

        try:
            # Create initial fire map with obstacles
            rows = self.config_vars['map_rows'].get()
            cols = self.config_vars['map_cols'].get()
            initial_fire_map = [[0 for _ in range(cols)] for _ in range(rows)]

            # Add obstacles to the map with value -2
            for pos in self.obstacle_positions:
                try:
                    x_part, y_part = pos.split('y')
                    x = int(x_part[1:])
                    y = int(y_part)
                    if 0 <= x < cols and 0 <= y < rows:
                        initial_fire_map[y][x] = -2
                except:
                    continue

            # Add fire to the map
            for pos, intensity in self.fire_positions:
                try:
                    x_part, y_part = pos.split('y')
                    x = int(x_part[1:])
                    y = int(y_part)
                    if 0 <= x < cols and 0 <= y < rows:
                        initial_fire_map[y][x] = intensity
                except:
                    continue

            # Create configuration with new physics parameters
            config = SimulationConfig(
                map_rows=self.config_vars['map_rows'].get(),
                map_cols=self.config_vars['map_cols'].get(),
                max_occupancy=self.config_vars['max_occupancy'].get(),
                start_positions=self.start_positions,
                targets=self.targets,
                initial_fire_map=initial_fire_map,
                agent_num=self.config_vars['agent_num'].get(),
                viewing_range=self.config_vars['viewing_range'].get(),
                cell_size=self.config_vars['cell_size'].get(),
                timestep_duration=self.config_vars['timestep_duration'].get(),
                fire_update_interval=self.config_vars['fire_update_interval'].get(),
                fire_model_type=self.config_vars['fire_model_type'].get()
            )

            # Run simulation
            simulation = EvacuationSimulation(config)

            # Create a new window to ask about visualization options
            self._run_simulation_with_options(simulation)

        except Exception as e:
            messagebox.showerror("Simulation Error", f"Failed to run simulation: {e}")

    def _run_simulation_with_options(self, simulation):
        # Create options dialog
        options_window = tk.Toplevel(self.root)
        options_window.title("Simulation Options")
        options_window.geometry("350x300")
        options_window.transient(self.root)
        options_window.grab_set()

        # Make sure it appears on top
        options_window.lift()
        options_window.focus_force()

        # Center the window on screen
        options_window.update_idletasks()
        width = options_window.winfo_width()
        height = options_window.winfo_height()
        x = (options_window.winfo_screenwidth() // 2) - (width // 2)
        y = (options_window.winfo_screenheight() // 2) - (height // 2)
        options_window.geometry(f"{width}x{height}+{x}+{y}")

        frame = ttk.Frame(options_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Choose visualization mode:").pack(pady=(0, 20))

        use_pygame = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Use Pygame visualization (if available)", variable=use_pygame).pack(anchor=tk.W, pady=5)

        show_text = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Show text visualization", variable=show_text).pack(anchor=tk.W, pady=5)

        use_matlab = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Run MATLAB simulation", variable=use_matlab).pack(anchor=tk.W, pady=5)

        max_steps = tk.IntVar(value=500)
        ttk.Label(frame, text="Max steps:").pack(anchor=tk.W, pady=(10, 0))
        ttk.Spinbox(frame, from_=50, to=2000, textvariable=max_steps, width=10).pack(anchor=tk.W, pady=5)

        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=(20, 0))

        def start_simulation():
            options_window.destroy()
            try:
                simulation.run(
                    max_steps=max_steps.get(),
                    show_visualization=show_text.get(),
                    use_pygame=use_pygame.get(),
                    use_matlab=use_matlab.get()
                )
                messagebox.showinfo("Simulation Complete", "Simulation finished successfully!")
            except Exception as e:
                messagebox.showerror("Simulation Error", f"Simulation failed: {e}")

        def cancel():
            options_window.destroy()

        ttk.Button(button_frame, text="Start", command=start_simulation).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT)

class PositionDialog:
    def __init__(self, parent, max_rows, max_cols):
        self.result = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Add Position")
        self.dialog.geometry("250x150")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))

        frame = ttk.Frame(self.dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="X coordinate:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.x_var = tk.IntVar(value=0)
        ttk.Spinbox(frame, from_=0, to=max_cols-1, textvariable=self.x_var, width=10).grid(row=0, column=1, padx=(5, 0), pady=2)

        ttk.Label(frame, text="Y coordinate:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.y_var = tk.IntVar(value=0)
        ttk.Spinbox(frame, from_=0, to=max_rows-1, textvariable=self.y_var, width=10).grid(row=1, column=1, padx=(5, 0), pady=2)

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(20, 0))

        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.LEFT)

        self.dialog.wait_window()

    def ok_clicked(self):
        self.result = f'x{self.x_var.get()}y{self.y_var.get()}'
        self.dialog.destroy()

    def cancel_clicked(self):
        self.dialog.destroy()

class FireDialog:
    def __init__(self, parent, max_rows, max_cols):
        self.result = None

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Add Fire")
        self.dialog.geometry("250x180")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))

        frame = ttk.Frame(self.dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="X coordinate:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.x_var = tk.IntVar(value=0)
        ttk.Spinbox(frame, from_=0, to=max_cols-1, textvariable=self.x_var, width=10).grid(row=0, column=1, padx=(5, 0), pady=2)

        ttk.Label(frame, text="Y coordinate:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.y_var = tk.IntVar(value=0)
        ttk.Spinbox(frame, from_=0, to=max_rows-1, textvariable=self.y_var, width=10).grid(row=1, column=1, padx=(5, 0), pady=2)

        ttk.Label(frame, text="Fire intensity:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.intensity_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(frame, from_=0.1, to=1.0, increment=0.1, textvariable=self.intensity_var, width=10).grid(row=2, column=1, padx=(5, 0), pady=2)

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(20, 0))

        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side=tk.LEFT)

        self.dialog.wait_window()

    def ok_clicked(self):
        position = f'x{self.x_var.get()}y{self.y_var.get()}'
        intensity = round(self.intensity_var.get(), 2)
        self.result = (position, intensity)
        self.dialog.destroy()

    def cancel_clicked(self):
        self.dialog.destroy()

def main():
    root = tk.Tk()
    app = VisualConfigurator(root)
    root.mainloop()

if __name__ == "__main__":
    main()