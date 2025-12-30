import pygame
import sys
from d_star_lite.utils import stateNameToCoords

class Gradient:
    def __init__(self):
        pass

    def clamp(self, x, a, b):
        return max(a, min(b, x))

    def lerp(self, a, b, t):
        return a + (b - a) * t

    def interp_color(self, c1, c2, t):
        return (
            int(self.lerp(c1[0], c2[0], t)),
            int(self.lerp(c1[1], c2[1], t)),
            int(self.lerp(c1[2], c2[2], t)),
        )

    def fire_color_gradient(self, value):
        """
        value: number (expected 0..4). returns (r,g,b) as ints 0..255.
        分段節點（從冷暗到白熱）：可自行調整顏色陣列。
        """
        v = self.clamp(float(value), 0.0, 4.0) / 4.0  # 正規化到 0..1

        # 節點位置與顏色（位置是 0..1）——你可以改這些 RGB 來微調火焰風格
        stops = [
            (0.00, (10, 10, 12)),     # 0: 幾乎黑（未燃）
            (0.20, (120, 10, 5)),     # 深紅
            (0.45, (220, 40, 0)),     # 紅橙
            (0.70, (255, 140, 20)),   # 橙黃
            (1.00, (255, 245, 200)),  # 黃白（白熱）
        ]

        # 找到 v 落在哪兩個節點之間，線性插值
        for i in range(len(stops) - 1):
            p0, c0 = stops[i]
            p1, c1 = stops[i+1]
            if v <= p1:
                t = 0.0 if p1 == p0 else (v - p0) / (p1 - p0)
                return self.interp_color(c0, c1, t)

        return stops[-1][1]  # 理論上不會到這

class EvacuationVisualizer:
    def __init__(self, map_rows, map_cols, cell_size=60):
        pygame.init()

        self.map_rows = map_rows
        self.map_cols = map_cols
        self.cell_size = cell_size

        # Calculate window size
        self.width = map_cols * cell_size
        self.height = map_rows * cell_size + 100  # Extra space for info panel

        # Colors
        self.colors = {
            'background': (240, 240, 240),
            'grid': (200, 200, 200),
            'agent': (0, 100, 255),
            'target': (255, 100, 100),
            'target_reached': (100, 255, 100),
            'text': (0, 0, 0),
            'panel': (220, 220, 220),
            'cell_clear': (255, 255, 255),     # White for clear cells (value 0)
            'cell_difficult': (255, 255, 200), # Light yellow for difficult terrain (value 1+)
            'cell_fire': (255, 100, 100),      # Red for fire/obstacles (value < 0)
            'cell_impassable': (150, 0, 0)     # Dark red for impassable (value <= -5)
        }

        # Agent colors for different agents
        self.agent_colors = [
            (0, 100, 255),   # Blue
            (255, 150, 0),   # Orange
            (150, 0, 255),   # Purple
            (255, 0, 150),   # Pink
            (0, 200, 100),   # Green
        ]

        # Initialize pygame
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Evacuation Simulation")

        # Font for text
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        self.clock = pygame.time.Clock()

    def coord_to_pixel(self, x, y):
        """Convert grid coordinates to pixel coordinates"""
        pixel_x = x * self.cell_size + self.cell_size // 2
        pixel_y = y * self.cell_size + self.cell_size // 2
        return pixel_x, pixel_y

    def get_cell_color(self, cell_value):
        """Get color based on cell value"""
        if cell_value <= -5:
            return self.colors['cell_impassable']
        elif cell_value < 0:
            return self.colors['cell_fire']
        elif cell_value == 0:
            return self.colors['cell_clear']
        else:
            gradient = Gradient()
            return gradient.fire_color_gradient(cell_value)

    def draw_grid(self, agents):
        """Draw the grid with color-coded cells based on terrain values"""
        self.screen.fill(self.colors['background'])

        # Draw cells with colors based on their values
        if agents:
            grid = agents[0].graph
            for x in range(self.map_cols):
                for y in range(self.map_rows):
                    if 0 <= y < len(grid.cells) and 0 <= x < len(grid.cells[y]):
                        cell_value = grid.cells[y][x]
                        cell_color = self.get_cell_color(cell_value)

                        # Draw cell rectangle
                        cell_rect = pygame.Rect(
                            x * self.cell_size,
                            y * self.cell_size,
                            self.cell_size,
                            self.cell_size
                        )
                        pygame.draw.rect(self.screen, cell_color, cell_rect)

        # Draw grid lines on top
        for x in range(self.map_cols + 1):
            start_pos = (x * self.cell_size, 0)
            end_pos = (x * self.cell_size, self.map_rows * self.cell_size)
            pygame.draw.line(self.screen, self.colors['grid'], start_pos, end_pos, 1)

        for y in range(self.map_rows + 1):
            start_pos = (0, y * self.cell_size)
            end_pos = (self.map_cols * self.cell_size, y * self.cell_size)
            pygame.draw.line(self.screen, self.colors['grid'], start_pos, end_pos, 1)


    def draw_targets(self, targets, reached_targets=None):
        """Draw evacuation targets"""
        if reached_targets is None:
            reached_targets = set()

        for i, target in enumerate(targets):
            try:
                coords = stateNameToCoords(target)
                if 0 <= coords[0] < self.map_cols and 0 <= coords[1] < self.map_rows:
                    pixel_x, pixel_y = self.coord_to_pixel(coords[0], coords[1])

                    # Choose color based on whether target has been reached
                    color = self.colors['target_reached'] if target in reached_targets else self.colors['target']

                    # Draw target as a circle
                    pygame.draw.circle(self.screen, color, (pixel_x, pixel_y), self.cell_size // 3, 3)

                    # Draw target number
                    text = self.font.render(f"T{i+1}", True, self.colors['text'])
                    text_rect = text.get_rect(center=(pixel_x, pixel_y))
                    self.screen.blit(text, text_rect)
            except:
                pass

    def draw_agents(self, agents):
        """Draw agents at their current positions"""
        for agent in agents:
            try:
                coords = stateNameToCoords(agent.s_current)
                if 0 <= coords[0] < self.map_cols and 0 <= coords[1] < self.map_rows:
                    pixel_x, pixel_y = self.coord_to_pixel(coords[0], coords[1])

                    # Get agent color
                    color = self.agent_colors[agent.id % len(self.agent_colors)]

                    # Draw agent as a filled circle
                    pygame.draw.circle(self.screen, color, (pixel_x, pixel_y), self.cell_size // 4)

                    # Draw agent ID
                    text = self.small_font.render(f"A{agent.id}", True, (255, 255, 255))
                    text_rect = text.get_rect(center=(pixel_x, pixel_y))
                    self.screen.blit(text, text_rect)
            except:
                pass

    def draw_info_panel(self, step, status, agents):
        """Draw information panel at the bottom"""
        panel_y = self.map_rows * self.cell_size
        panel_rect = pygame.Rect(0, panel_y, self.width, 100)
        pygame.draw.rect(self.screen, self.colors['panel'], panel_rect)

        # Draw step counter
        step_text = self.font.render(f"Step: {step}", True, self.colors['text'])
        self.screen.blit(step_text, (10, panel_y + 10))

        # Draw evacuation status
        status_text = self.font.render(
            f"Evacuated: {status['evacuated_agents']}/{status['total_agents']}",
            True, self.colors['text']
        )
        self.screen.blit(status_text, (10, panel_y + 35))

        # Count different terrain types
        terrain_counts = {'clear': 0, 'difficult': 0, 'fire': 0, 'impassable': 0}
        if agents:
            grid = agents[0].graph
            for x in range(self.map_cols):
                for y in range(self.map_rows):
                    if 0 <= y < len(grid.cells) and 0 <= x < len(grid.cells[y]):
                        cell_value = grid.cells[y][x]
                        if cell_value <= -5:
                            terrain_counts['impassable'] += 1
                        elif cell_value < 0:
                            terrain_counts['fire'] += 1
                        elif cell_value == 0:
                            terrain_counts['clear'] += 1
                        else:
                            terrain_counts['difficult'] += 1

        # Draw terrain info
        total_obstacles = terrain_counts['fire'] + terrain_counts['impassable']
        terrain_text = self.font.render(f"Obstacles: {total_obstacles}", True, self.colors['text'])
        self.screen.blit(terrain_text, (200, panel_y + 10))

        # Draw agent information
        if agents:
            agent_info = "Agents: "
            for i, agent in enumerate(agents[:3]):  # Show first 3 agents to avoid overcrowding
                target_num = agent.targetidx + 1
                agent_info += f"A{agent.id}→T{target_num} "

            if len(agents) > 3:
                agent_info += f"(+{len(agents)-3} more)"

            info_text = self.small_font.render(agent_info, True, self.colors['text'])
            self.screen.blit(info_text, (10, panel_y + 60))

        # Draw controls
        controls_text = self.small_font.render("Press SPACE to pause, ESC to quit", True, self.colors['text'])
        self.screen.blit(controls_text, (self.width - 250, panel_y + 75))

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    # Pause functionality could be implemented here
                    pass
        return True

    def update_display(self, step, agents, targets, status, reached_targets=None):
        """Update the entire display"""
        self.draw_grid(agents)  # Draw color-coded grid based on cell values
        self.draw_targets(targets, reached_targets)
        self.draw_agents(agents)
        self.draw_info_panel(step, status, agents)

        pygame.display.flip()
        return self.handle_events()

    def wait_for_next_frame(self, fps=10):
        """Control frame rate"""
        self.clock.tick(fps)

    def close(self):
        """Clean up pygame"""
        pygame.quit()