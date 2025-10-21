"""
Spatial indexing for efficient proximity queries in evacuation simulation.

Uses grid-based partitioning to reduce agent-agent proximity checks from O(n²) to O(n).
Agents are organized into sectors, and proximity queries only check nearby sectors.
"""

import math
from d_star_lite.utils import stateNameToCoords


class SpatialIndex:
    """
    Efficiently find nearby agents using grid-based spatial partitioning.

    The map is divided into sectors of configurable size. When querying for nearby
    agents, only agents in the same sector and neighboring sectors are checked,
    significantly reducing computational cost for large agent populations.

    Time Complexity:
    - update(): O(n) where n is number of agents
    - get_nearby_agents(): O(k) where k is agents in nearby sectors (k << n typically)

    Attributes:
        sector_size (int): Size of each sector in grid cells
        map_rows (int): Total map height
        map_cols (int): Total map width
        sectors (dict): Maps (sector_x, sector_y) -> list of agent indices
    """

    def __init__(self, map_rows, map_cols, sector_size=10):
        """
        Initialize spatial index.

        Args:
            map_rows (int): Map height in cells
            map_cols (int): Map width in cells
            sector_size (int): Size of each sector in cells. Should be approximately
                              equal to communication_range for optimal performance.
        """
        self.sector_size = sector_size
        self.map_rows = map_rows
        self.map_cols = map_cols
        self.sectors = {}  # {(sector_x, sector_y): [agent_indices]}

    def _get_sector(self, pos):
        """
        Convert grid position to sector coordinates.

        Args:
            pos (str or tuple): Either "x{col}y{row}" string or (col, row) tuple

        Returns:
            tuple: (sector_x, sector_y) coordinates
        """
        if isinstance(pos, str):
            col, row = stateNameToCoords(pos)
        else:
            col, row = pos

        return (col // self.sector_size, row // self.sector_size)

    def _get_nearby_sectors(self, sector, radius=1):
        """
        Get all sectors within given radius (including diagonals).

        Args:
            sector (tuple): Center sector (sector_x, sector_y)
            radius (int): How many sectors away to include

        Returns:
            list: List of (sector_x, sector_y) tuples
        """
        sx, sy = sector
        nearby = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nearby.append((sx + dx, sy + dy))
        return nearby

    def update(self, agents):
        """
        Rebuild spatial index from current agent positions.

        Should be called periodically (e.g., every few timesteps) to keep index current.

        Args:
            agents (list): List of EvacuationAgent objects (only active agents)

        Time Complexity: O(n) where n is number of agents
        """
        self.sectors.clear()

        for i, agent in enumerate(agents):
            # All agents in the list are considered active
            # (evacuated/stuck agents are removed from the list by simulation)
            sector = self._get_sector(agent.s_current)

            if sector not in self.sectors:
                self.sectors[sector] = []

            self.sectors[sector].append(i)

    def get_nearby_agents(self, agent_pos, agents, max_distance):
        """
        Get all agents within max_distance of given position.

        Uses sector-based filtering to avoid checking all agents, then verifies
        actual Euclidean distance for agents in candidate sectors.

        Args:
            agent_pos (str): Position in "x{col}y{row}" format
            agents (list): List of all EvacuationAgent objects
            max_distance (float): Maximum distance in grid cells

        Returns:
            list: List of EvacuationAgent objects within max_distance

        Time Complexity: O(k) where k is agents in nearby sectors
        """
        sector = self._get_sector(agent_pos)

        # Calculate how many sectors we need to check based on max_distance
        # Add 1 to be safe with edge cases at sector boundaries
        sector_radius = int(math.ceil(max_distance / self.sector_size)) + 1
        nearby_sectors = self._get_nearby_sectors(sector, sector_radius)

        # Collect candidate agents from nearby sectors
        candidate_indices = []
        for s in nearby_sectors:
            if s in self.sectors:
                candidate_indices.extend(self.sectors[s])

        # Filter by actual Euclidean distance
        my_pos = stateNameToCoords(agent_pos)
        nearby_agents = []

        for agent_idx in candidate_indices:
            agent = agents[agent_idx]

            # All agents in the list are active (simulation removes inactive ones)
            other_pos = stateNameToCoords(agent.s_current)
            distance = math.sqrt((my_pos[0] - other_pos[0])**2 +
                               (my_pos[1] - other_pos[1])**2)

            if distance <= max_distance:
                nearby_agents.append(agent)

        return nearby_agents

    def get_sector_count(self):
        """
        Get number of currently occupied sectors.

        Returns:
            int: Number of sectors with at least one agent
        """
        return len(self.sectors)

    def get_stats(self):
        """
        Get statistics about current spatial index state.

        Returns:
            dict: Statistics including sector count, avg agents per sector, etc.
        """
        if not self.sectors:
            return {
                'occupied_sectors': 0,
                'total_agents': 0,
                'avg_agents_per_sector': 0,
                'max_agents_in_sector': 0
            }

        sector_sizes = [len(agents) for agents in self.sectors.values()]

        return {
            'occupied_sectors': len(self.sectors),
            'total_agents': sum(sector_sizes),
            'avg_agents_per_sector': sum(sector_sizes) / len(sector_sizes),
            'max_agents_in_sector': max(sector_sizes)
        }
