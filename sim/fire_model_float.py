"""
Advanced Indoor Fire Spread Model using Cellular Automata
========================================================

A sophisticated fire simulation model incorporating realistic physics for indoor environments.
Perfect for science fair presentations and emergency evacuation planning.

Fire States:
- 0: Clear/Unburned
- 1-4: Fire intensity levels (1=ignition, 4=peak intensity)
- -1: Inaccessible areas (no oxygen, extreme conditions)
- -2: Permanent obstacles (walls, barriers)

Author: Advanced Fire Modeling System
"""

import math
import random
import copy
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EnvironmentalParameters:
    """Environmental conditions affecting fire spread"""
    # Wind and airflow
    wind_speed: float = 0.5  # m/s (0-5 typical indoor range)
    wind_direction: float = 0.0  # radians (0 = east, π/2 = north)
    ventilation_rate: float = 0.3  # air changes per hour

    # Atmospheric conditions
    oxygen_level: float = 21.0  # percentage (normal: 21%)
    temperature: float = 20.0  # Celsius
    humidity: float = 50.0  # percentage

    # Fuel and material properties
    fuel_density: float = 1.0  # relative fuel availability
    fuel_moisture: float = 10.0  # percentage

    # Room characteristics
    ceiling_height: float = 2.7  # meters
    room_volume: float = 100.0  # cubic meters

    # Advanced physics parameters
    thermal_conductivity: float = 0.5  # heat transfer coefficient
    ignition_threshold: float = 0.3  # probability threshold for ignition
    burn_rate_modifier: float = 1.0  # overall fire spread speed

    # Fancy science fair parameters
    carbon_monoxide_production: float = 0.1  # CO production rate
    smoke_density_factor: float = 0.2  # smoke generation
    radiant_heat_factor: float = 0.8  # heat radiation efficiency


class AdvancedFireModel:
    """
    Sophisticated indoor fire spread simulation using cellular automata
    with realistic physics and environmental factors.
    """

    def __init__(self, rows: int, cols: int, env_params: Optional[EnvironmentalParameters] = None):
        self.rows = rows
        self.cols = cols
        self.env = env_params or EnvironmentalParameters()

        # Advanced simulation state
        self.oxygen_map = [[self.env.oxygen_level for _ in range(cols)] for _ in range(rows)]
        self.temperature_map = [[self.env.temperature for _ in range(cols)] for _ in range(rows)]
        self.fuel_map = [[self.env.fuel_density for _ in range(cols)] for _ in range(rows)]
        self.smoke_density = [[0.0 for _ in range(cols)] for _ in range(rows)]

        # Fire history for realistic burning patterns
        self.burn_time = [[0.0 for _ in range(cols)] for _ in range(rows)]
        self.max_intensity_reached = [[0.0 for _ in range(cols)] for _ in range(rows)]

        # Precompute wind effects for efficiency
        self.wind_influence = self._calculate_wind_influence()

    def _calculate_wind_influence(self) -> List[List[Tuple[float, float]]]:
        """Calculate wind direction influence on each cell"""
        wind_map = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                # Wind creates directional bias in fire spread
                wind_x = math.cos(self.env.wind_direction) * self.env.wind_speed
                wind_y = math.sin(self.env.wind_direction) * self.env.wind_speed
                row.append((wind_x, wind_y))
            wind_map.append(row)
        return wind_map

    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (8-connectivity)"""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
        return neighbors

    def _calculate_spread_probability(self, current_state: List[List[float]],
                                    row: int, col: int) -> float:
        """
        Calculate probability of fire spreading to this cell based on
        realistic fire physics and environmental conditions.
        """
        if current_state[row][col] != 0:  # Already burning or obstacle
            return 0.0

        # Base probability from neighboring fire intensity
        neighbors = self._get_neighbors(row, col)
        neighbor_fire_sum = 0.0
        max_neighbor_intensity = 0.0

        for nr, nc in neighbors:
            neighbor_state = current_state[nr][nc]
            if 0 < neighbor_state <= 4:  # Burning neighbor
                # Distance-weighted contribution
                distance = math.sqrt((row - nr)**2 + (col - nc)**2)
                weight = 1.0 / distance  # Closer neighbors have more influence

                # Wind direction bonus
                wind_x, wind_y = self.wind_influence[nr][nc]
                direction_bonus = 1.0
                if wind_x != 0 or wind_y != 0:
                    # Calculate if wind is blowing toward this cell
                    dx, dy = col - nc, row - nr
                    dot_product = (wind_x * dx + wind_y * dy)
                    if dot_product > 0:  # Wind favors this direction
                        direction_bonus = 1.0 + (dot_product * 0.3)

                contribution = neighbor_state * weight * direction_bonus
                neighbor_fire_sum += contribution
                max_neighbor_intensity = max(max_neighbor_intensity, neighbor_state)

        # Base spread probability
        base_prob = min(neighbor_fire_sum * 0.1, 0.8)

        # Environmental modifications

        # 1. Oxygen availability (critical factor)
        oxygen_factor = min(self.oxygen_map[row][col] / 16.0, 1.0)  # Below 16% O2, fire struggles
        if self.oxygen_map[row][col] < 12.0:  # Critical oxygen level
            oxygen_factor *= 0.1

        # 2. Fuel availability and moisture
        fuel_factor = self.fuel_map[row][col]
        moisture_penalty = max(0.1, 1.0 - self.env.fuel_moisture / 50.0)

        # 3. Temperature preheating effect
        temp_bonus = 1.0
        if self.temperature_map[row][col] > 100.0:  # Preheated surface
            temp_bonus = 1.0 + (self.temperature_map[row][col] - 100.0) / 200.0

        # 4. Smoke density reduces oxygen availability
        smoke_penalty = max(0.3, 1.0 - self.smoke_density[row][col])

        # 5. Wind speed enhancement
        wind_bonus = 1.0 + (self.env.wind_speed * 0.2)

        # 6. Humidity effect
        humidity_factor = max(0.5, 1.0 - self.env.humidity / 200.0)

        # Combined probability with all factors
        final_prob = (base_prob *
                     oxygen_factor *
                     fuel_factor *
                     moisture_penalty *
                     temp_bonus *
                     smoke_penalty *
                     wind_bonus *
                     humidity_factor *
                     self.env.burn_rate_modifier)

        return min(final_prob, 0.95)  # Cap at 95% to maintain some randomness

    def _update_environmental_conditions(self, current_state: List[List[float]]) -> None:
        """Update environmental conditions based on current fire state"""

        for i in range(self.rows):
            for j in range(self.cols):
                fire_intensity = current_state[i][j]

                if 0 < fire_intensity <= 4:
                    # Fire consumes oxygen
                    oxygen_consumption = fire_intensity * 0.5
                    self.oxygen_map[i][j] = max(0, self.oxygen_map[i][j] - oxygen_consumption)

                    # Fire increases temperature
                    heat_production = fire_intensity * 50.0
                    self.temperature_map[i][j] += heat_production

                    # Fire produces smoke
                    smoke_production = fire_intensity * self.env.smoke_density_factor
                    self.smoke_density[i][j] += smoke_production

                    # Fire consumes fuel
                    fuel_consumption = fire_intensity * 0.02
                    self.fuel_map[i][j] = max(0, self.fuel_map[i][j] - fuel_consumption)

                    # Update burn time
                    self.burn_time[i][j] += 1.0
                    self.max_intensity_reached[i][j] = max(
                        self.max_intensity_reached[i][j], fire_intensity
                    )

                # Heat dissipation to neighbors
                if self.temperature_map[i][j] > self.env.temperature:
                    neighbors = self._get_neighbors(i, j)
                    for ni, nj in neighbors:
                        if current_state[ni][nj] != -2:  # Not a wall
                            heat_transfer = (self.temperature_map[i][j] -
                                           self.temperature_map[ni][nj]) * self.env.thermal_conductivity * 0.1
                            self.temperature_map[ni][nj] += heat_transfer

                # Gradual oxygen replenishment through ventilation
                if self.oxygen_map[i][j] < self.env.oxygen_level:
                    replenishment = self.env.ventilation_rate * 0.1
                    self.oxygen_map[i][j] = min(
                        self.env.oxygen_level,
                        self.oxygen_map[i][j] + replenishment
                    )

                # Smoke dissipation
                if self.smoke_density[i][j] > 0:
                    dissipation = self.env.ventilation_rate * 0.05
                    self.smoke_density[i][j] = max(0, self.smoke_density[i][j] - dissipation)

    def _calculate_fire_progression(self, current_intensity: float, row: int, col: int) -> float:
        """Calculate how fire intensity changes over time in a cell"""

        # Fire growth phase (1 -> 4)
        if current_intensity < 4.0:
            # Growth rate depends on available oxygen and fuel
            oxygen_factor = min(self.oxygen_map[row][col] / 18.0, 1.0)
            fuel_factor = min(self.fuel_map[row][col], 1.0)

            if oxygen_factor > 0.7 and fuel_factor > 0.1:
                growth_rate = 0.3 * oxygen_factor * fuel_factor
                return min(4.0, current_intensity + growth_rate)

        # Decay phase (fire burns out)
        burn_duration = self.burn_time[row][col]

        # Fuel depletion causes decay
        if self.fuel_map[row][col] < 0.1:
            decay_rate = 0.4
        # Oxygen starvation causes decay
        elif self.oxygen_map[row][col] < 10.0:
            decay_rate = 0.3
        # Natural burnout after extended burning
        elif burn_duration > 20.0:
            decay_rate = 0.2
        else:
            return current_intensity

        return max(0.0, current_intensity - decay_rate)

    def simulate_step(self, current_state: List[List[float]]) -> Dict[str, float]:
        """
        Simulate one time step of fire spread and return changes.

        Args:
            current_state: Current fire state grid

        Returns:
            Dictionary of changes in format {"x{col}y{row}": new_value}
        """
        changes = {}
        new_state = copy.deepcopy(current_state)

        # Update environmental conditions first
        self._update_environmental_conditions(current_state)

        # Process each cell
        for i in range(self.rows):
            for j in range(self.cols):
                current_value = current_state[i][j]

                # Skip obstacles and inaccessible areas
                if current_value == -2 or current_value == -1:
                    continue

                # Handle unburned cells (potential ignition)
                if current_value == 0:
                    spread_prob = self._calculate_spread_probability(current_state, i, j)

                    # Add some randomness for realistic fire behavior
                    if random.random() < spread_prob:
                        new_value = 1.0  # Ignition!
                        changes[f"x{j}y{i}"] = new_value
                        new_state[i][j] = new_value

                # Handle burning cells (intensity progression)
                elif 0 < current_value <= 4:
                    new_intensity = self._calculate_fire_progression(current_value, i, j)

                    # Only record change if significant
                    if abs(new_intensity - current_value) > 0.01:
                        changes[f"x{j}y{i}"] = new_intensity
                        new_state[i][j] = new_intensity

        return changes

    def get_simulation_statistics(self) -> Dict[str, float]:
        """Return advanced statistics for science fair presentation"""
        total_oxygen_consumed = 0.0
        total_co_produced = 0.0
        total_heat_generated = 0.0
        max_temperature = 0.0
        total_smoke = 0.0

        for i in range(self.rows):
            for j in range(self.cols):
                oxygen_deficit = self.env.oxygen_level - self.oxygen_map[i][j]
                total_oxygen_consumed += max(0, oxygen_deficit)

                if self.burn_time[i][j] > 0:
                    total_co_produced += (self.burn_time[i][j] *
                                        self.env.carbon_monoxide_production)

                total_heat_generated += max(0, self.temperature_map[i][j] - self.env.temperature)
                max_temperature = max(max_temperature, self.temperature_map[i][j])
                total_smoke += self.smoke_density[i][j]

        return {
            "oxygen_consumed_percent": total_oxygen_consumed / (self.rows * self.cols) * 100,
            "co_concentration_ppm": total_co_produced * 100,  # Simplified conversion
            "average_temperature_rise": total_heat_generated / (self.rows * self.cols),
            "max_temperature_celsius": max_temperature,
            "total_smoke_density": total_smoke,
            "fire_safety_index": max(0, 100 - total_smoke - (total_co_produced * 10))
        }


def create_fire_model(rows: int, cols: int, **env_kwargs) -> AdvancedFireModel:
    """
    Factory function to create a fire model with custom environmental parameters.

    Example usage:
        model = create_fire_model(20, 30, wind_speed=1.2, humidity=30.0)
        changes = model.simulate_step(current_fire_state)
    """
    env_params = EnvironmentalParameters(**env_kwargs)
    return AdvancedFireModel(rows, cols, env_params)


# Convenience function for basic usage
def simulate_fire_spread(fire_states: List[List[float]], rows: int, cols: int,
                        **environmental_params) -> Dict[str, float]:
    """
    Simple interface for fire spread simulation.

    Args:
        fire_states: Current fire state grid (List[List[float]])
        rows: Number of rows
        cols: Number of columns
        **environmental_params: Environmental parameters (wind_speed, humidity, etc.)

    Returns:
        Dictionary of changes {"x{col}y{row}": new_fire_state}
    """
    model = create_fire_model(rows, cols, **environmental_params)
    return model.simulate_step(fire_states)


# Example demonstration for science fair
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower():
        print("🔥 Advanced Indoor Fire Simulation Model 🔥")
    else:
        print("*** Advanced Indoor Fire Simulation Model ***")
    print("=" * 50)

    # Create a sample room with obstacles
    rows, cols = 10, 15
    fire_state = [[0.0 for _ in range(cols)] for _ in range(rows)]

    # Add walls (obstacles)
    for i in range(rows):
        fire_state[i][0] = -2  # Left wall
        fire_state[i][cols-1] = -2  # Right wall
    for j in range(cols):
        fire_state[0][j] = -2  # Top wall
        fire_state[rows-1][j] = -2  # Bottom wall

    # Add some internal obstacles
    fire_state[5][7] = -2  # Pillar
    fire_state[3][10] = -1  # Inaccessible area

    # Start a fire in one corner
    fire_state[2][2] = 2.0

    # Create model with windy conditions
    model = create_fire_model(rows, cols,
                            wind_speed=1.5,
                            wind_direction=0.785,  # 45 degrees
                            humidity=25.0,
                            fuel_density=1.2)

    print(f"Initial fire state: Fire started at position (2,2)")
    print(f"Environmental conditions:")
    print(f"  Wind: {model.env.wind_speed} m/s at {model.env.wind_direction:.2f} radians")
    print(f"  Humidity: {model.env.humidity}%")
    print(f"  Fuel density: {model.env.fuel_density}")

    # Simulate several steps
    for step in range(5):
        changes = model.simulate_step(fire_state)

        if changes:
            print(f"\nStep {step + 1}: {len(changes)} cells changed")
            for pos, new_val in changes.items():
                x = int(pos.split('y')[0][1:])
                y = int(pos.split('y')[1])
                fire_state[y][x] = new_val
                print(f"  {pos}: {new_val:.2f}")

        # Show statistics
        stats = model.get_simulation_statistics()
        print(f"  Fire Safety Index: {stats['fire_safety_index']:.1f}")
        print(f"  Max Temperature: {stats['max_temperature_celsius']:.1f}°C")

    print("\nScience Fair Highlights:")
    print("• Realistic physics with oxygen consumption")
    print("• Wind effects on fire spread direction")
    print("• Temperature and smoke modeling")
    print("• Carbon monoxide production tracking")
    print("• Fuel depletion and fire decay")
    print("• Environmental factor interactions")