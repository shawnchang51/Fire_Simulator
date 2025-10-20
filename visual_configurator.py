"""
Visual Configurator for Fire Evacuation Simulation
===================================================
Interactive GUI tool for configuring and running evacuation simulations.

Features:
- Visual grid editor with click-to-place elements
- Configuration panel for all simulation parameters
- Load/Save JSON configurations
- Validation system
- Direct simulation runner
- Undo/redo functionality

Author: Fire Evacuation Simulation System
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import os
from typing import Dict, List, Tuple, Optional
import copy


# Color scheme
COLORS = {
    'bg': '#f0f0f0',
    'panel_bg': '#ffffff',
    'grid_bg': '#ffffff',
    'grid_line': '#e0e0e0',
    'hover': '#ffffcc',
    'agent_start': '#4285f4',      # Blue
    'door': '#34a853',              # Green
    'exit': '#fbbc04',              # Yellow/Gold
    'obstacle': '#2d2d2d',          # Dark gray
    'fire': '#ea4335',              # Red
    'fire_light': '#ff6b6b',        # Light red
    'passable': '#ffffff',          # White
    'selected_tool': '#c8e6c9',     # Light green
}


class VisualConfigurator:
    """Main application class for the visual configurator."""

    def __init__(self, root):
        self.root = root
        self.root.title("Fire Evacuation Simulator - Visual Configurator")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLORS['bg'])

        # Data structures
        self.grid_rows = 60
        self.grid_cols = 60
        self.cell_pixel_size = 12  # pixels per grid cell
        self.grid_data = self.create_initial_grid()
        self.agent_starts = []  # List of (col, row) tuples
        self.doors = []         # List of {"id": str, "position": "xCyR", "type": "door"}
        self.exits = []         # List of {"id": str, "position": "xCyR", "type": "exit"}

        # Configuration parameters
        self.config_params = {
            'cell_size': tk.DoubleVar(value=0.3),
            'timestep_duration': tk.DoubleVar(value=0.5),
            'fire_update_interval': tk.IntVar(value=2),
            'max_occupancy': tk.IntVar(value=1),
            'fire_model_type': tk.StringVar(value='realistic'),
            'viewing_range': tk.IntVar(value=10),
            'agent_fearness_bulk': tk.DoubleVar(value=1.0),
        }
        self.agent_fearness_values = []  # Per-agent fearness values

        # UI state
        self.current_tool = 'agent_start'
        self.hover_cell = None
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.zoom_level = 1.0

        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo = 50

        # Create UI
        self.create_menu()
        self.create_status_bar()  # Create status bar first
        self.create_main_layout()
        self.create_toolbar()
        self.create_grid_canvas()
        self.create_config_panel()

        # Keyboard shortcuts
        self.root.bind('<Control-z>', lambda e: self.undo())
        self.root.bind('<Control-y>', lambda e: self.redo())
        self.root.bind('<Control-s>', lambda e: self.save_config())
        self.root.bind('<Control-o>', lambda e: self.load_config())

        # Initialize with empty grid
        self.save_state_for_undo()
        self.update_grid_display()

    def create_initial_grid(self):
        """Create initial grid with obstacles on the border."""
        grid = [[0 for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]

        # Set outermost cells as obstacles (-2)
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Top row, bottom row, left column, or right column
                if row == 0 or row == self.grid_rows - 1 or col == 0 or col == self.grid_cols - 1:
                    grid[row][col] = -2

        return grid

    def create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_config, accelerator="Ctrl+N")
        file_menu.add_command(label="Load...", command=self.load_config, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_config, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_config_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear Grid", command=self.clear_grid)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Validate Configuration", command=self.validate_config)
        tools_menu.add_command(label="Auto-compute Parameters", command=self.auto_compute_params)

        # Run menu
        run_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Run", menu=run_menu)
        run_menu.add_command(label="Run Simulation...", command=self.run_simulation_dialog)

    def create_main_layout(self):
        """Create main layout with grid and config panel."""
        # Main container
        main_container = tk.Frame(self.root, bg=COLORS['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left side: Grid canvas
        self.left_frame = tk.Frame(main_container, bg=COLORS['panel_bg'], relief=tk.SUNKEN, bd=2)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Right side: Configuration panel
        self.right_frame = tk.Frame(main_container, bg=COLORS['panel_bg'], relief=tk.SUNKEN, bd=2, width=350)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        self.right_frame.pack_propagate(False)

    def create_toolbar(self):
        """Create toolbar with drawing tools."""
        toolbar = tk.Frame(self.left_frame, bg=COLORS['panel_bg'], relief=tk.RAISED, bd=1)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(toolbar, text="Drawing Tools:", bg=COLORS['panel_bg'], font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)

        # Tool buttons
        self.tool_buttons = {}
        tools = [
            ('agent_start', 'Agent Start', COLORS['agent_start']),
            ('door', 'Door', COLORS['door']),
            ('exit', 'Exit', COLORS['exit']),
            ('obstacle', 'Obstacle', COLORS['obstacle']),
            ('fire', 'Fire', COLORS['fire']),
            ('edit', 'Edit', '#9c27b0'),  # Purple for edit tool
            ('erase', 'Erase', COLORS['passable']),
        ]

        for tool_id, label, color in tools:
            btn = tk.Button(
                toolbar,
                text=label,
                bg=color,
                fg='white' if tool_id not in ['erase'] else 'black',
                activebackground=color,
                relief=tk.RAISED,
                bd=2,
                padx=10,
                pady=5,
                command=lambda t=tool_id: self.select_tool(t)
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.tool_buttons[tool_id] = btn

        # Highlight current tool
        self.select_tool('agent_start')

    def create_grid_canvas(self):
        """Create scrollable grid canvas."""
        # Canvas container with scrollbars
        canvas_container = tk.Frame(self.left_frame, bg=COLORS['panel_bg'])
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollbars
        v_scroll = tk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        h_scroll = tk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas
        self.canvas = tk.Canvas(
            canvas_container,
            bg=COLORS['grid_bg'],
            scrollregion=(0, 0, self.grid_cols * self.cell_pixel_size, self.grid_rows * self.cell_pixel_size),
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set,
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scroll.config(command=self.canvas.yview)
        h_scroll.config(command=self.canvas.xview)

        # Mouse events
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.bind('<Button-3>', self.on_canvas_right_click)
        self.canvas.bind('<Motion>', self.on_canvas_hover)
        self.canvas.bind('<Leave>', lambda e: self.on_canvas_hover(None))

        # Zoom controls
        zoom_frame = tk.Frame(self.left_frame, bg=COLORS['panel_bg'])
        zoom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        tk.Label(zoom_frame, text="Zoom:", bg=COLORS['panel_bg']).pack(side=tk.LEFT, padx=5)
        tk.Button(zoom_frame, text="-", command=self.zoom_out, width=3).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="+", command=self.zoom_in, width=3).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="Fit", command=self.zoom_fit, width=5).pack(side=tk.LEFT, padx=5)

        # Grid info
        self.grid_info_label = tk.Label(
            zoom_frame,
            text=f"Grid: {self.grid_cols}x{self.grid_rows}",
            bg=COLORS['panel_bg'],
            font=('Arial', 9)
        )
        self.grid_info_label.pack(side=tk.RIGHT, padx=5)

    def create_config_panel(self):
        """Create configuration panel with tabs."""
        # Title
        title_label = tk.Label(
            self.right_frame,
            text="Configuration",
            bg=COLORS['panel_bg'],
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=10)

        # Notebook for tabs
        notebook = ttk.Notebook(self.right_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Global Settings Tab
        global_tab = tk.Frame(notebook, bg=COLORS['panel_bg'])
        notebook.add(global_tab, text="Global")
        self.create_global_settings(global_tab)

        # Agent Settings Tab
        agent_tab = tk.Frame(notebook, bg=COLORS['panel_bg'])
        notebook.add(agent_tab, text="Agents")
        self.create_agent_settings(agent_tab)

        # Map Settings Tab
        map_tab = tk.Frame(notebook, bg=COLORS['panel_bg'])
        notebook.add(map_tab, text="Map")
        self.create_map_settings(map_tab)

        # Action buttons
        self.create_action_buttons()

    def create_global_settings(self, parent):
        """Create global settings inputs."""
        container = tk.Frame(parent, bg=COLORS['panel_bg'])
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        row = 0

        # Cell size
        tk.Label(container, text="Cell Size (m):", bg=COLORS['panel_bg']).grid(row=row, column=0, sticky='w', pady=5)
        tk.Spinbox(
            container,
            from_=0.1,
            to=2.0,
            increment=0.1,
            textvariable=self.config_params['cell_size'],
            width=15
        ).grid(row=row, column=1, pady=5)
        row += 1

        # Timestep duration
        tk.Label(container, text="Timestep (s):", bg=COLORS['panel_bg']).grid(row=row, column=0, sticky='w', pady=5)
        tk.Spinbox(
            container,
            from_=0.1,
            to=2.0,
            increment=0.1,
            textvariable=self.config_params['timestep_duration'],
            width=15
        ).grid(row=row, column=1, pady=5)
        row += 1

        # Fire update interval
        tk.Label(container, text="Fire Update Interval:", bg=COLORS['panel_bg']).grid(row=row, column=0, sticky='w', pady=5)
        tk.Spinbox(
            container,
            from_=1,
            to=20,
            increment=1,
            textvariable=self.config_params['fire_update_interval'],
            width=15
        ).grid(row=row, column=1, pady=5)
        row += 1

        # Max occupancy
        tk.Label(container, text="Max Occupancy:", bg=COLORS['panel_bg']).grid(row=row, column=0, sticky='w', pady=5)
        tk.Spinbox(
            container,
            from_=1,
            to=10,
            increment=1,
            textvariable=self.config_params['max_occupancy'],
            width=15
        ).grid(row=row, column=1, pady=5)
        row += 1

        # Fire model type
        tk.Label(container, text="Fire Model:", bg=COLORS['panel_bg']).grid(row=row, column=0, sticky='w', pady=5)
        fire_model_combo = ttk.Combobox(
            container,
            textvariable=self.config_params['fire_model_type'],
            values=['realistic', 'aggressive', 'default'],
            state='readonly',
            width=13
        )
        fire_model_combo.grid(row=row, column=1, pady=5)
        row += 1

    def create_agent_settings(self, parent):
        """Create agent settings inputs."""
        container = tk.Frame(parent, bg=COLORS['panel_bg'])
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        row = 0

        # Viewing range
        tk.Label(container, text="Viewing Range:", bg=COLORS['panel_bg']).grid(row=row, column=0, sticky='w', pady=5)
        tk.Spinbox(
            container,
            from_=1,
            to=30,
            increment=1,
            textvariable=self.config_params['viewing_range'],
            width=15
        ).grid(row=row, column=1, pady=5)
        row += 1

        # Agent count (read-only)
        tk.Label(container, text="Agent Count:", bg=COLORS['panel_bg']).grid(row=row, column=0, sticky='w', pady=5)
        self.agent_count_label = tk.Label(container, text="0", bg=COLORS['panel_bg'], font=('Arial', 10, 'bold'))
        self.agent_count_label.grid(row=row, column=1, sticky='w', pady=5)
        row += 1

        # Bulk fearness
        tk.Label(container, text="Bulk Fearness:", bg=COLORS['panel_bg']).grid(row=row, column=0, sticky='w', pady=5)
        bulk_frame = tk.Frame(container, bg=COLORS['panel_bg'])
        bulk_frame.grid(row=row, column=1, sticky='w', pady=5)
        tk.Spinbox(
            bulk_frame,
            from_=0.1,
            to=10.0,
            increment=0.1,
            textvariable=self.config_params['agent_fearness_bulk'],
            width=10
        ).pack(side=tk.LEFT)
        tk.Button(
            bulk_frame,
            text="Apply",
            command=self.apply_bulk_fearness,
            bg='#4285f4',
            fg='white',
            padx=5
        ).pack(side=tk.LEFT, padx=5)
        row += 1

        # Per-agent fearness
        tk.Label(container, text="Per-Agent Fearness:", bg=COLORS['panel_bg'], font=('Arial', 9, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', pady=(10, 5)
        )
        row += 1

        # Scrollable frame for fearness values
        fearness_container = tk.Frame(container, bg=COLORS['panel_bg'])
        fearness_container.grid(row=row, column=0, columnspan=2, sticky='nsew', pady=5)
        container.rowconfigure(row, weight=1)

        self.fearness_canvas = tk.Canvas(fearness_container, bg=COLORS['panel_bg'], height=200, highlightthickness=0)
        fearness_scroll = tk.Scrollbar(fearness_container, orient=tk.VERTICAL, command=self.fearness_canvas.yview)
        self.fearness_frame = tk.Frame(self.fearness_canvas, bg=COLORS['panel_bg'])

        self.fearness_canvas.create_window((0, 0), window=self.fearness_frame, anchor='nw')
        self.fearness_canvas.configure(yscrollcommand=fearness_scroll.set)

        self.fearness_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fearness_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.fearness_frame.bind('<Configure>', lambda e: self.fearness_canvas.configure(scrollregion=self.fearness_canvas.bbox('all')))

    def create_map_settings(self, parent):
        """Create map settings."""
        container = tk.Frame(parent, bg=COLORS['panel_bg'])
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        row = 0

        # Grid dimensions (auto-computed)
        tk.Label(container, text="Grid Dimensions:", bg=COLORS['panel_bg'], font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', pady=5
        )
        row += 1

        tk.Label(container, text="Rows:", bg=COLORS['panel_bg']).grid(row=row, column=0, sticky='w', pady=5)
        self.rows_label = tk.Label(container, text=str(self.grid_rows), bg=COLORS['panel_bg'], font=('Arial', 10, 'bold'))
        self.rows_label.grid(row=row, column=1, sticky='w', pady=5)
        row += 1

        tk.Label(container, text="Columns:", bg=COLORS['panel_bg']).grid(row=row, column=0, sticky='w', pady=5)
        self.cols_label = tk.Label(container, text=str(self.grid_cols), bg=COLORS['panel_bg'], font=('Arial', 10, 'bold'))
        self.cols_label.grid(row=row, column=1, sticky='w', pady=5)
        row += 1

        # Resize grid button
        tk.Button(
            container,
            text="Resize Grid...",
            command=self.resize_grid_dialog,
            bg='#4285f4',
            fg='white',
            padx=10,
            pady=5
        ).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1

        # Statistics
        tk.Label(container, text="Statistics:", bg=COLORS['panel_bg'], font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', pady=(10, 5)
        )
        row += 1

        self.stats_text = scrolledtext.ScrolledText(container, height=10, width=30, bg='#f9f9f9', state='disabled')
        self.stats_text.grid(row=row, column=0, columnspan=2, sticky='nsew', pady=5)
        container.rowconfigure(row, weight=1)

    def create_action_buttons(self):
        """Create action buttons at bottom of config panel."""
        button_frame = tk.Frame(self.right_frame, bg=COLORS['panel_bg'])
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Validate button
        tk.Button(
            button_frame,
            text="Validate",
            command=self.validate_config,
            bg='#34a853',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=8
        ).pack(fill=tk.X, pady=2)

        # Run simulation button
        tk.Button(
            button_frame,
            text="Run Simulation",
            command=self.run_simulation_dialog,
            bg='#ea4335',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=8
        ).pack(fill=tk.X, pady=2)

    def create_status_bar(self):
        """Create status bar at bottom."""
        self.status_bar = tk.Label(
            self.root,
            text="Ready. Click to place elements on the grid.",
            bg='#e0e0e0',
            anchor='w',
            relief=tk.SUNKEN,
            padx=5
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ==================== Tool Selection ====================

    def select_tool(self, tool_id):
        """Select a drawing tool."""
        self.current_tool = tool_id

        # Update button appearance
        for tid, btn in self.tool_buttons.items():
            if tid == tool_id:
                btn.config(relief=tk.SUNKEN, bd=3)
            else:
                btn.config(relief=tk.RAISED, bd=2)

        # Update status
        tool_names = {
            'agent_start': 'Agent Start Position',
            'door': 'Door',
            'exit': 'Exit',
            'obstacle': 'Obstacle',
            'fire': 'Fire',
            'edit': 'Edit Cell',
            'erase': 'Eraser'
        }
        self.update_status(f"Selected tool: {tool_names.get(tool_id, tool_id)}")

    # ==================== Grid Display ====================

    def update_grid_display(self):
        """Redraw the entire grid."""
        self.canvas.delete('all')

        # Draw grid cells
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x1 = col * self.cell_pixel_size
                y1 = row * self.cell_pixel_size
                x2 = x1 + self.cell_pixel_size
                y2 = y1 + self.cell_pixel_size

                # Determine cell color
                cell_value = self.grid_data[row][col]
                color = self.get_cell_color(cell_value)

                # Draw cell
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline=COLORS['grid_line'],
                    tags=f'cell_{col}_{row}'
                )

        # Draw agent starts
        for idx, (col, row) in enumerate(self.agent_starts):
            x = col * self.cell_pixel_size + self.cell_pixel_size / 2
            y = row * self.cell_pixel_size + self.cell_pixel_size / 2
            r = self.cell_pixel_size / 2 - 1

            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=COLORS['agent_start'],
                outline='white',
                width=1,
                tags='agent_start'
            )
            self.canvas.create_text(
                x, y,
                text=str(idx),
                fill='white',
                font=('Arial', int(self.cell_pixel_size * 0.6), 'bold'),
                tags='agent_start'
            )

        # Draw doors
        for door in self.doors:
            col, row = self.parse_position(door['position'])
            x = col * self.cell_pixel_size + self.cell_pixel_size / 2
            y = row * self.cell_pixel_size + self.cell_pixel_size / 2
            r = self.cell_pixel_size / 2 - 1

            self.canvas.create_rectangle(
                x - r, y - r, x + r, y + r,
                fill=COLORS['door'],
                outline='white',
                width=2,
                tags='door'
            )
            self.canvas.create_text(
                x, y,
                text='D',
                fill='white',
                font=('Arial', int(self.cell_pixel_size * 0.7), 'bold'),
                tags='door'
            )

        # Draw exits
        for exit_item in self.exits:
            col, row = self.parse_position(exit_item['position'])
            x = col * self.cell_pixel_size + self.cell_pixel_size / 2
            y = row * self.cell_pixel_size + self.cell_pixel_size / 2
            r = self.cell_pixel_size / 2 - 1

            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=COLORS['exit'],
                outline='white',
                width=2,
                tags='exit'
            )
            self.canvas.create_text(
                x, y,
                text='E',
                fill='white',
                font=('Arial', int(self.cell_pixel_size * 0.7), 'bold'),
                tags='exit'
            )

        # Update statistics
        self.update_statistics()
        self.update_agent_count()

    def get_cell_color(self, value):
        """Get color for a cell based on its value."""
        if value == -2:
            return COLORS['obstacle']
        elif value > 0:
            # Fire intensity (gradient)
            intensity = min(value / 4.0, 1.0)
            return self.interpolate_color(COLORS['fire_light'], COLORS['fire'], intensity)
        else:
            return COLORS['passable']

    def interpolate_color(self, color1, color2, t):
        """Interpolate between two colors."""
        # Simple hex color interpolation
        c1 = [int(color1[i:i+2], 16) for i in (1, 3, 5)]
        c2 = [int(color2[i:i+2], 16) for i in (1, 3, 5)]
        c = [int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3)]
        return f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}'

    # ==================== Mouse Events ====================

    def on_canvas_click(self, event):
        """Handle left-click on canvas."""
        col, row = self.get_grid_coords(event.x, event.y)
        if col is None or row is None:
            return

        # Edit tool doesn't need undo save here (done in edit dialog)
        if self.current_tool == 'edit':
            self.edit_cell(col, row)
            return

        self.save_state_for_undo()

        if self.current_tool == 'agent_start':
            self.place_agent_start(col, row)
        elif self.current_tool == 'door':
            self.place_door(col, row)
        elif self.current_tool == 'exit':
            self.place_exit(col, row)
        elif self.current_tool == 'obstacle':
            self.place_obstacle(col, row)
        elif self.current_tool == 'fire':
            self.place_fire(col, row)
        elif self.current_tool == 'erase':
            self.erase_cell(col, row)

        self.update_grid_display()

    def on_canvas_right_click(self, event):
        """Handle right-click (delete)."""
        col, row = self.get_grid_coords(event.x, event.y)
        if col is None or row is None:
            return

        self.save_state_for_undo()
        self.erase_cell(col, row)
        self.update_grid_display()

    def on_canvas_hover(self, event):
        """Handle mouse hover over canvas."""
        if event is None:
            self.hover_cell = None
            self.update_status("Ready")
            return

        col, row = self.get_grid_coords(event.x, event.y)
        if col is not None and row is not None:
            self.hover_cell = (col, row)
            self.update_status(f"Cell: ({col}, {row}) | Value: {self.grid_data[row][col]}")
        else:
            self.hover_cell = None

    def get_grid_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to grid coordinates."""
        # Adjust for scroll position
        canvas_x = self.canvas.canvasx(canvas_x)
        canvas_y = self.canvas.canvasy(canvas_y)

        col = int(canvas_x // self.cell_pixel_size)
        row = int(canvas_y // self.cell_pixel_size)

        if 0 <= col < self.grid_cols and 0 <= row < self.grid_rows:
            return col, row
        return None, None

    # ==================== Placement Functions ====================

    def place_agent_start(self, col, row):
        """Place an agent start position."""
        pos = (col, row)
        if pos not in self.agent_starts:
            self.agent_starts.append(pos)
            # Clear any grid value at this position
            self.grid_data[row][col] = 0

    def place_door(self, col, row):
        """Place a door."""
        position = f"x{col}y{row}"
        # Check if door already exists at this position
        if not any(d['position'] == position for d in self.doors):
            door_id = f"d{len(self.doors) + 1}"
            self.doors.append({"id": door_id, "position": position, "type": "door"})
            # Clear any grid value at this position
            self.grid_data[row][col] = 0

    def place_exit(self, col, row):
        """Place an exit."""
        position = f"x{col}y{row}"
        # Check if exit already exists at this position
        if not any(e['position'] == position for e in self.exits):
            exit_id = f"e{len(self.exits) + 1}"
            self.exits.append({"id": exit_id, "position": position, "type": "exit"})
            # Clear any grid value at this position
            self.grid_data[row][col] = 0

    def place_obstacle(self, col, row):
        """Place an obstacle."""
        self.grid_data[row][col] = -2
        # Remove any agent/door/exit at this position
        self.remove_elements_at(col, row)

    def place_fire(self, col, row):
        """Place fire."""
        self.grid_data[row][col] = 2.0  # Medium fire intensity
        # Remove any agent/door/exit at this position
        self.remove_elements_at(col, row)

    def erase_cell(self, col, row):
        """Erase everything at a cell."""
        self.grid_data[row][col] = 0
        self.remove_elements_at(col, row)

    def remove_elements_at(self, col, row):
        """Remove agent starts, doors, and exits at a position."""
        pos = (col, row)
        if pos in self.agent_starts:
            self.agent_starts.remove(pos)

        position = f"x{col}y{row}"
        self.doors = [d for d in self.doors if d['position'] != position]
        self.exits = [e for e in self.exits if e['position'] != position]

    def edit_cell(self, col, row):
        """Open edit dialog for a cell."""
        position = f"x{col}y{row}"
        pos = (col, row)

        # Determine what's at this cell
        is_agent = pos in self.agent_starts
        is_door = any(d['position'] == position for d in self.doors)
        is_exit = any(e['position'] == position for e in self.exits)
        cell_value = self.grid_data[row][col]
        is_fire = cell_value > 0
        is_obstacle = cell_value == -2

        # If cell is completely empty, show info
        if not (is_agent or is_door or is_exit or is_fire or is_obstacle):
            messagebox.showinfo("Edit Cell", f"Cell ({col}, {row}) is empty.\n\nUse other tools to place elements first.")
            return

        # Create edit dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit Cell ({col}, {row})")
        dialog.geometry("400x500")
        dialog.transient(self.root)
        dialog.grab_set()

        # Title
        tk.Label(
            dialog,
            text=f"Edit Cell ({col}, {row})",
            font=('Arial', 14, 'bold')
        ).pack(pady=10)

        # Container for edit options
        container = tk.Frame(dialog)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Variables to store edited values
        edited_values = {}

        # Edit agent fearness if agent is here
        if is_agent:
            agent_idx = self.agent_starts.index(pos)

            tk.Label(
                container,
                text=f"Agent {agent_idx} Start Position",
                font=('Arial', 11, 'bold'),
                fg=COLORS['agent_start']
            ).pack(anchor='w', pady=(10, 5))

            # Fearness value
            fearness_frame = tk.Frame(container)
            fearness_frame.pack(anchor='w', pady=5)

            tk.Label(fearness_frame, text="Fearness:").pack(side=tk.LEFT, padx=(0, 5))

            current_fearness = 1.0
            if agent_idx < len(self.agent_fearness_values):
                current_fearness = self.agent_fearness_values[agent_idx]

            fearness_var = tk.DoubleVar(value=current_fearness)
            tk.Spinbox(
                fearness_frame,
                from_=0.1,
                to=10.0,
                increment=0.1,
                textvariable=fearness_var,
                width=10
            ).pack(side=tk.LEFT)

            tk.Label(
                fearness_frame,
                text="(0.1 - 10.0)",
                font=('Arial', 8),
                fg='gray'
            ).pack(side=tk.LEFT, padx=5)

            edited_values['agent_fearness'] = (agent_idx, fearness_var)

        # Edit fire intensity if fire is here
        if is_fire:
            tk.Label(
                container,
                text="Fire",
                font=('Arial', 11, 'bold'),
                fg=COLORS['fire']
            ).pack(anchor='w', pady=(10, 5))

            # Fire intensity
            intensity_frame = tk.Frame(container)
            intensity_frame.pack(anchor='w', pady=5)

            tk.Label(intensity_frame, text="Intensity:").pack(side=tk.LEFT, padx=(0, 5))

            intensity_var = tk.DoubleVar(value=cell_value)
            tk.Scale(
                intensity_frame,
                from_=0.1,
                to=4.0,
                resolution=0.1,
                variable=intensity_var,
                orient=tk.HORIZONTAL,
                length=200
            ).pack(side=tk.LEFT)

            tk.Label(
                intensity_frame,
                text="(0.1 - 4.0)",
                font=('Arial', 8),
                fg='gray'
            ).pack(side=tk.LEFT, padx=5)

            # Info about intensity levels
            tk.Label(
                container,
                text="1.0 = Ignition | 2.0 = Growth | 3.0 = Developed | 4.0 = Flashover",
                font=('Arial', 8),
                fg='gray'
            ).pack(anchor='w', pady=(0, 5))

            edited_values['fire_intensity'] = intensity_var

        # Edit door ID
        if is_door:
            door = next(d for d in self.doors if d['position'] == position)
            tk.Label(
                container,
                text="Door",
                font=('Arial', 11, 'bold'),
                fg=COLORS['door']
            ).pack(anchor='w', pady=(10, 5))

            # Door ID
            id_frame = tk.Frame(container)
            id_frame.pack(anchor='w', pady=5)

            tk.Label(id_frame, text="ID:").pack(side=tk.LEFT, padx=(0, 5))

            door_id_var = tk.StringVar(value=door['id'])
            tk.Entry(
                id_frame,
                textvariable=door_id_var,
                width=15
            ).pack(side=tk.LEFT)

            tk.Label(
                id_frame,
                text="(e.g., d1, d2, door_main)",
                font=('Arial', 8),
                fg='gray'
            ).pack(side=tk.LEFT, padx=5)

            edited_values['door_id'] = (door, door_id_var)

        # Edit exit ID
        if is_exit:
            exit_item = next(e for e in self.exits if e['position'] == position)
            tk.Label(
                container,
                text="Exit",
                font=('Arial', 11, 'bold'),
                fg=COLORS['exit']
            ).pack(anchor='w', pady=(10, 5))

            # Exit ID
            id_frame = tk.Frame(container)
            id_frame.pack(anchor='w', pady=5)

            tk.Label(id_frame, text="ID:").pack(side=tk.LEFT, padx=(0, 5))

            exit_id_var = tk.StringVar(value=exit_item['id'])
            tk.Entry(
                id_frame,
                textvariable=exit_id_var,
                width=15
            ).pack(side=tk.LEFT)

            tk.Label(
                id_frame,
                text="(e.g., e1, e2, exit_main)",
                font=('Arial', 8),
                fg='gray'
            ).pack(side=tk.LEFT, padx=5)

            edited_values['exit_id'] = (exit_item, exit_id_var)

        if is_obstacle:
            tk.Label(
                container,
                text="Obstacle",
                font=('Arial', 11, 'bold'),
                fg=COLORS['obstacle']
            ).pack(anchor='w', pady=(10, 5))

            tk.Label(
                container,
                text="No editable properties for obstacles.",
                font=('Arial', 9),
                fg='gray'
            ).pack(anchor='w', pady=5)

        # Apply button
        def apply_changes():
            self.save_state_for_undo()

            # Apply agent fearness
            if 'agent_fearness' in edited_values:
                agent_idx, fearness_var = edited_values['agent_fearness']
                new_fearness = fearness_var.get()

                # Ensure fearness list is large enough
                while len(self.agent_fearness_values) <= agent_idx:
                    self.agent_fearness_values.append(1.0)

                self.agent_fearness_values[agent_idx] = new_fearness
                self.update_fearness_inputs()

            # Apply fire intensity
            if 'fire_intensity' in edited_values:
                intensity_var = edited_values['fire_intensity']
                new_intensity = intensity_var.get()
                self.grid_data[row][col] = new_intensity

            # Apply door ID
            if 'door_id' in edited_values:
                door, door_id_var = edited_values['door_id']
                new_id = door_id_var.get().strip()

                if new_id:  # Only update if not empty
                    # Check for duplicate IDs
                    existing_ids = [d['id'] for d in self.doors if d != door] + [e['id'] for e in self.exits]
                    if new_id in existing_ids:
                        messagebox.showwarning("Duplicate ID", f"ID '{new_id}' already exists. Please use a unique ID.")
                        return
                    door['id'] = new_id
                else:
                    messagebox.showwarning("Invalid ID", "Door ID cannot be empty.")
                    return

            # Apply exit ID
            if 'exit_id' in edited_values:
                exit_item, exit_id_var = edited_values['exit_id']
                new_id = exit_id_var.get().strip()

                if new_id:  # Only update if not empty
                    # Check for duplicate IDs
                    existing_ids = [d['id'] for d in self.doors] + [e['id'] for e in self.exits if e != exit_item]
                    if new_id in existing_ids:
                        messagebox.showwarning("Duplicate ID", f"ID '{new_id}' already exists. Please use a unique ID.")
                        return
                    exit_item['id'] = new_id
                else:
                    messagebox.showwarning("Invalid ID", "Exit ID cannot be empty.")
                    return

            self.update_grid_display()
            dialog.destroy()
            self.update_status(f"Cell ({col}, {row}) updated")

        tk.Button(
            dialog,
            text="Apply Changes",
            command=apply_changes,
            bg='#4285f4',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20,
            pady=10
        ).pack(pady=20)

        # Cancel button
        tk.Button(
            dialog,
            text="Cancel",
            command=dialog.destroy,
            padx=20,
            pady=5
        ).pack(pady=(0, 10))

    # ==================== Undo/Redo ====================

    def save_state_for_undo(self):
        """Save current state to undo stack."""
        state = {
            'grid_data': copy.deepcopy(self.grid_data),
            'agent_starts': copy.deepcopy(self.agent_starts),
            'doors': copy.deepcopy(self.doors),
            'exits': copy.deepcopy(self.exits),
        }
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

        # Clear redo stack when new action is performed
        self.redo_stack.clear()

    def undo(self):
        """Undo last action."""
        if not self.undo_stack:
            self.update_status("Nothing to undo")
            return

        # Save current state to redo stack
        current_state = {
            'grid_data': copy.deepcopy(self.grid_data),
            'agent_starts': copy.deepcopy(self.agent_starts),
            'doors': copy.deepcopy(self.doors),
            'exits': copy.deepcopy(self.exits),
        }
        self.redo_stack.append(current_state)

        # Restore previous state
        state = self.undo_stack.pop()
        self.grid_data = state['grid_data']
        self.agent_starts = state['agent_starts']
        self.doors = state['doors']
        self.exits = state['exits']

        self.update_grid_display()
        self.update_status("Undo successful")

    def redo(self):
        """Redo last undone action."""
        if not self.redo_stack:
            self.update_status("Nothing to redo")
            return

        # Save current state to undo stack
        self.save_state_for_undo()

        # Restore redo state
        state = self.redo_stack.pop()
        self.grid_data = state['grid_data']
        self.agent_starts = state['agent_starts']
        self.doors = state['doors']
        self.exits = state['exits']

        self.update_grid_display()
        self.update_status("Redo successful")

    # ==================== Zoom ====================

    def zoom_in(self):
        """Zoom in the grid."""
        self.cell_pixel_size = min(30, int(self.cell_pixel_size * 1.2))
        self.update_canvas_size()
        self.update_grid_display()

    def zoom_out(self):
        """Zoom out the grid."""
        self.cell_pixel_size = max(5, int(self.cell_pixel_size * 0.8))
        self.update_canvas_size()
        self.update_grid_display()

    def zoom_fit(self):
        """Fit grid to canvas."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            zoom_x = canvas_width / self.grid_cols
            zoom_y = canvas_height / self.grid_rows
            self.cell_pixel_size = max(5, min(30, int(min(zoom_x, zoom_y))))
            self.update_canvas_size()
            self.update_grid_display()

    def update_canvas_size(self):
        """Update canvas scroll region after zoom."""
        self.canvas.config(
            scrollregion=(0, 0, self.grid_cols * self.cell_pixel_size, self.grid_rows * self.cell_pixel_size)
        )

    # ==================== Configuration Management ====================

    def new_config(self):
        """Create a new configuration."""
        if messagebox.askyesno("New Configuration", "Clear all data and start fresh?"):
            self.grid_data = self.create_initial_grid()
            self.agent_starts = []
            self.doors = []
            self.exits = []
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.save_state_for_undo()
            self.update_grid_display()
            self.update_status("New configuration created")

    def load_config(self):
        """Load configuration from JSON file."""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            defaultextension=".json"
        )

        if not filename:
            return

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Load grid dimensions
            self.grid_rows = config.get('map_rows', 60)
            self.grid_cols = config.get('map_cols', 60)

            # Load fire map
            self.grid_data = config.get('initial_fire_map', [[0 for _ in range(self.grid_cols)] for _ in range(self.grid_rows)])

            # Load agent starts
            self.agent_starts = [self.parse_position(pos) for pos in config.get('start_positions', [])]

            # Load doors and exits
            door_configs = config.get('door_configs', [])
            self.doors = [d for d in door_configs if d.get('type') == 'door']
            self.exits = [e for e in door_configs if e.get('type') == 'exit']

            # Load parameters
            self.config_params['cell_size'].set(config.get('cell_size', 0.3))
            self.config_params['timestep_duration'].set(config.get('timestep_duration', 0.5))
            self.config_params['fire_update_interval'].set(config.get('fire_update_interval', 2))
            self.config_params['max_occupancy'].set(config.get('max_occupancy', 1))
            self.config_params['fire_model_type'].set(config.get('fire_model_type', 'realistic'))
            self.config_params['viewing_range'].set(config.get('viewing_range', 10))

            # Load agent fearness
            self.agent_fearness_values = config.get('agent_fearness', [])

            # Update UI
            self.rows_label.config(text=str(self.grid_rows))
            self.cols_label.config(text=str(self.grid_cols))
            self.grid_info_label.config(text=f"Grid: {self.grid_cols}x{self.grid_rows}")
            self.update_canvas_size()
            self.update_grid_display()
            self.update_fearness_inputs()

            self.update_status(f"Loaded configuration from {os.path.basename(filename)}")
            messagebox.showinfo("Success", "Configuration loaded successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{str(e)}")

    def save_config(self):
        """Save configuration to default file."""
        filename = "example_configuration.json"
        self.save_config_to_file(filename)

    def save_config_as(self):
        """Save configuration to a new file."""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration As",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            defaultextension=".json"
        )

        if filename:
            self.save_config_to_file(filename)

    def save_config_to_file(self, filename):
        """Save current configuration to a JSON file."""
        try:
            # Build configuration dictionary
            config = {
                "_comment": "Fire Evacuation Simulation Configuration",
                "map_rows": self.grid_rows,
                "map_cols": self.grid_cols,
                "max_occupancy": self.config_params['max_occupancy'].get(),
                "cell_size": self.config_params['cell_size'].get(),
                "timestep_duration": self.config_params['timestep_duration'].get(),
                "fire_update_interval": self.config_params['fire_update_interval'].get(),
                "fire_model_type": self.config_params['fire_model_type'].get(),
                "start_positions": [self.format_position(col, row) for col, row in self.agent_starts],
                "door_configs": self.doors + self.exits,
                "initial_fire_map": self.grid_data,
                "agent_fearness": self.agent_fearness_values if self.agent_fearness_values else [1.0] * len(self.agent_starts),
                "agent_num": len(self.agent_starts),
                "viewing_range": self.config_params['viewing_range'].get()
            }

            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)

            self.update_status(f"Saved configuration to {os.path.basename(filename)}")
            messagebox.showinfo("Success", f"Configuration saved to:\n{filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration:\n{str(e)}")

    # ==================== Validation ====================

    def validate_config(self):
        """Validate current configuration."""
        issues = []

        # Check agent starts
        if not self.agent_starts:
            issues.append("- No agent start positions defined")

        # Check exits
        if not self.exits:
            issues.append("- No exits defined")

        # Check for negative parameters
        if self.config_params['cell_size'].get() <= 0:
            issues.append("- Cell size must be positive")

        if self.config_params['timestep_duration'].get() <= 0:
            issues.append("- Timestep duration must be positive")

        if self.config_params['fire_update_interval'].get() <= 0:
            issues.append("- Fire update interval must be positive")

        if self.config_params['max_occupancy'].get() <= 0:
            issues.append("- Max occupancy must be positive")

        if self.config_params['viewing_range'].get() <= 0:
            issues.append("- Viewing range must be positive")

        # Display results
        if issues:
            msg = "Validation Failed:\n\n" + "\n".join(issues)
            messagebox.showwarning("Validation", msg)
            self.update_status("Validation failed")
        else:
            messagebox.showinfo("Validation", "Configuration is valid!")
            self.update_status("Validation passed")

    # ==================== Auto-compute ====================

    def auto_compute_params(self):
        """Auto-compute derived parameters."""
        # Update agent count and fearness
        self.update_agent_count()
        self.update_fearness_inputs()

        messagebox.showinfo("Auto-compute", "Parameters auto-computed successfully!")
        self.update_status("Parameters auto-computed")

    def update_agent_count(self):
        """Update agent count display."""
        count = len(self.agent_starts)
        self.agent_count_label.config(text=str(count))

    def update_fearness_inputs(self):
        """Update per-agent fearness input fields."""
        # Clear existing fearness widgets
        for widget in self.fearness_frame.winfo_children():
            widget.destroy()

        # Ensure fearness values list matches agent count
        agent_count = len(self.agent_starts)
        while len(self.agent_fearness_values) < agent_count:
            self.agent_fearness_values.append(1.0)
        while len(self.agent_fearness_values) > agent_count:
            self.agent_fearness_values.pop()

        # Create input fields for each agent
        for i in range(agent_count):
            frame = tk.Frame(self.fearness_frame, bg=COLORS['panel_bg'])
            frame.pack(fill=tk.X, pady=2)

            tk.Label(frame, text=f"Agent {i}:", bg=COLORS['panel_bg'], width=8, anchor='w').pack(side=tk.LEFT)

            var = tk.DoubleVar(value=self.agent_fearness_values[i])
            spinbox = tk.Spinbox(
                frame,
                from_=0.1,
                to=10.0,
                increment=0.1,
                textvariable=var,
                width=10,
                command=lambda idx=i, v=var: self.update_fearness_value(idx, v)
            )
            spinbox.pack(side=tk.LEFT, padx=5)

            # Also bind to FocusOut to catch manual edits
            spinbox.bind('<FocusOut>', lambda e, idx=i, v=var: self.update_fearness_value(idx, v))

    def update_fearness_value(self, index, var):
        """Update fearness value for a specific agent."""
        try:
            value = var.get()
            if 0 <= index < len(self.agent_fearness_values):
                self.agent_fearness_values[index] = value
        except:
            pass

    def apply_bulk_fearness(self):
        """Apply bulk fearness value to all agents."""
        bulk_value = self.config_params['agent_fearness_bulk'].get()
        self.agent_fearness_values = [bulk_value] * len(self.agent_starts)
        self.update_fearness_inputs()
        self.update_status(f"Applied fearness {bulk_value} to all agents")

    # ==================== Grid Resize ====================

    def resize_grid_dialog(self):
        """Show dialog to resize grid."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Resize Grid")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="New Grid Dimensions:").pack(pady=10)

        frame = tk.Frame(dialog)
        frame.pack(pady=5)

        tk.Label(frame, text="Rows:").grid(row=0, column=0, padx=5, pady=5)
        rows_var = tk.IntVar(value=self.grid_rows)
        tk.Spinbox(frame, from_=10, to=200, textvariable=rows_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(frame, text="Columns:").grid(row=1, column=0, padx=5, pady=5)
        cols_var = tk.IntVar(value=self.grid_cols)
        tk.Spinbox(frame, from_=10, to=200, textvariable=cols_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        def apply_resize():
            new_rows = rows_var.get()
            new_cols = cols_var.get()

            # Create new grid with border obstacles
            new_grid = [[0 for _ in range(new_cols)] for _ in range(new_rows)]

            # Set border as obstacles
            for row in range(new_rows):
                for col in range(new_cols):
                    if row == 0 or row == new_rows - 1 or col == 0 or col == new_cols - 1:
                        new_grid[row][col] = -2

            # Copy existing interior data (skip borders)
            for row in range(1, min(self.grid_rows - 1, new_rows - 1)):
                for col in range(1, min(self.grid_cols - 1, new_cols - 1)):
                    new_grid[row][col] = self.grid_data[row][col]

            # Remove out-of-bounds elements
            self.agent_starts = [(c, r) for c, r in self.agent_starts if c < new_cols and r < new_rows]
            self.doors = [d for d in self.doors if self.is_position_valid(d['position'], new_cols, new_rows)]
            self.exits = [e for e in self.exits if self.is_position_valid(e['position'], new_cols, new_rows)]

            # Update grid
            self.grid_rows = new_rows
            self.grid_cols = new_cols
            self.grid_data = new_grid

            # Update UI
            self.rows_label.config(text=str(self.grid_rows))
            self.cols_label.config(text=str(self.grid_cols))
            self.grid_info_label.config(text=f"Grid: {self.grid_cols}x{self.grid_rows}")
            self.update_canvas_size()
            self.update_grid_display()

            dialog.destroy()
            self.update_status(f"Grid resized to {new_cols}x{new_rows}")

        tk.Button(dialog, text="Apply", command=apply_resize, bg='#4285f4', fg='white', padx=20).pack(pady=10)

    def is_position_valid(self, position, max_cols, max_rows):
        """Check if a position is within grid bounds."""
        col, row = self.parse_position(position)
        return 0 <= col < max_cols and 0 <= row < max_rows

    # ==================== Clear Grid ====================

    def clear_grid(self):
        """Clear all grid data."""
        if messagebox.askyesno("Clear Grid", "Clear all elements from the grid?"):
            self.save_state_for_undo()
            self.grid_data = self.create_initial_grid()
            self.agent_starts = []
            self.doors = []
            self.exits = []
            self.update_grid_display()
            self.update_status("Grid cleared")

    # ==================== Statistics ====================

    def update_statistics(self):
        """Update statistics display."""
        stats = []
        stats.append(f"Grid Size: {self.grid_cols} x {self.grid_rows}")
        stats.append(f"Agent Starts: {len(self.agent_starts)}")
        stats.append(f"Doors: {len(self.doors)}")
        stats.append(f"Exits: {len(self.exits)}")

        # Count obstacles and fire cells
        obstacle_count = sum(1 for row in self.grid_data for cell in row if cell == -2)
        fire_count = sum(1 for row in self.grid_data for cell in row if cell > 0)

        stats.append(f"Obstacles: {obstacle_count}")
        stats.append(f"Fire Cells: {fire_count}")

        # Physical dimensions
        physical_width = self.grid_cols * self.config_params['cell_size'].get()
        physical_height = self.grid_rows * self.config_params['cell_size'].get()
        stats.append(f"\nPhysical Size:")
        stats.append(f"  {physical_width:.1f}m x {physical_height:.1f}m")

        # Update text widget
        self.stats_text.config(state='normal')
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert('1.0', '\n'.join(stats))
        self.stats_text.config(state='disabled')

    # ==================== Simulation Runner ====================

    def run_simulation_dialog(self):
        """Show dialog to configure and run simulation."""
        # First validate
        issues = []
        if not self.agent_starts:
            issues.append("No agent start positions")
        if not self.exits:
            issues.append("No exits defined")

        if issues:
            messagebox.showerror("Cannot Run", "Fix these issues first:\n- " + "\n- ".join(issues))
            return

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Run Simulation")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Simulation Settings", font=('Arial', 12, 'bold')).pack(pady=10)

        frame = tk.Frame(dialog)
        frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        # Max steps
        tk.Label(frame, text="Max Steps:").grid(row=0, column=0, sticky='w', pady=5)
        max_steps_var = tk.IntVar(value=500)
        tk.Spinbox(frame, from_=10, to=5000, increment=10, textvariable=max_steps_var, width=15).grid(row=0, column=1, pady=5)

        # Visualization options
        tk.Label(frame, text="Visualization:").grid(row=1, column=0, sticky='w', pady=5)
        viz_var = tk.StringVar(value='pygame')
        viz_frame = tk.Frame(frame)
        viz_frame.grid(row=1, column=1, sticky='w', pady=5)
        tk.Radiobutton(viz_frame, text="Pygame", variable=viz_var, value='pygame').pack(anchor='w')
        tk.Radiobutton(viz_frame, text="MATLAB-style", variable=viz_var, value='matlab').pack(anchor='w')
        tk.Radiobutton(viz_frame, text="Text", variable=viz_var, value='text').pack(anchor='w')
        tk.Radiobutton(viz_frame, text="None", variable=viz_var, value='none').pack(anchor='w')

        def run_sim():
            # Save config first
            temp_config = "temp_visual_config.json"
            self.save_config_to_file(temp_config)

            # Prepare run command
            max_steps = max_steps_var.get()
            viz_type = viz_var.get()

            # Build run parameters
            use_pygame = viz_type == 'pygame'
            use_matlab = viz_type == 'matlab'
            show_viz = viz_type == 'text'

            dialog.destroy()

            # Import and run simulation
            try:
                from simulation import SimulationConfig, EvacuationSimulation
                import json

                with open(temp_config, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                config = SimulationConfig.from_json(config_data)
                sim = EvacuationSimulation(config)

                self.update_status("Running simulation...")
                self.root.update()

                sim.run(
                    max_steps=max_steps,
                    show_visualization=show_viz,
                    use_pygame=use_pygame,
                    use_matlab=use_matlab
                )

                self.update_status("Simulation completed")
                messagebox.showinfo("Complete", "Simulation finished successfully!")

            except Exception as e:
                messagebox.showerror("Simulation Error", f"Error running simulation:\n{str(e)}")
                self.update_status("Simulation failed")

        tk.Button(dialog, text="Run Simulation", command=run_sim, bg='#ea4335', fg='white', padx=20, pady=10).pack(pady=20)

    # ==================== Utility Functions ====================

    def parse_position(self, position: str) -> Tuple[int, int]:
        """Parse position string 'xCyR' to (col, row)."""
        # Format: "x12y9" -> (12, 9)
        parts = position.replace('x', '').split('y')
        return int(parts[0]), int(parts[1])

    def format_position(self, col: int, row: int) -> str:
        """Format (col, row) to position string 'xCyR'."""
        return f"x{col}y{row}"

    def update_status(self, message: str):
        """Update status bar message."""
        self.status_bar.config(text=message)


def main():
    """Main entry point."""
    root = tk.Tk()
    app = VisualConfigurator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
