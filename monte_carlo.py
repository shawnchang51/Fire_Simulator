from simulation import EvacuationSimulation, SimulationConfig
from d_star_lite.utils import coordsToStateName, stateNameToCoords
import argparse, json, os
import random
from collections import Counter

def replace_fire(config: SimulationConfig, num_fires: int=None) -> SimulationConfig:
    """
    Replace fire positions with random valid locations.
    Fire can't be placed on doors, exits, or obstacles.

    Args:
        config: Simulation configuration
        num_fires: Number of fire locations (if None, keeps original count)

    Returns:
        Updated configuration with new fire positions
    """
    if num_fires is None:
        # Count existing fires
        num_fires = sum(1 for row in config.initial_fire_map
                       for cell in row if cell > 0)

    # Create set of invalid positions (doors, exits, obstacles)
    invalid_positions = set()

    # Add door and exit positions
    if config.door_configs:
        for door in config.door_configs:
            col, row = stateNameToCoords(door['position'])
            invalid_positions.add((row, col))

    # Add obstacle positions (-2 in fire map)
    for row_idx, row in enumerate(config.initial_fire_map):
        for col_idx, cell in enumerate(row):
            if cell < 0:
                invalid_positions.add((row_idx, col_idx))

    # Find all valid positions (not obstacles, doors, or exits)
    valid_positions = []
    for row_idx in range(config.map_rows):
        for col_idx in range(config.map_cols):
            if (row_idx, col_idx) not in invalid_positions:
                valid_positions.append((row_idx, col_idx))

    # Create new fire map (copy obstacles from original)
    new_fire_map = [[config.initial_fire_map[r][c] if config.initial_fire_map[r][c] == -2 else 0
                     for c in range(config.map_cols)]
                    for r in range(config.map_rows)]

    # Randomly select fire positions
    if len(valid_positions) < num_fires:
        raise ValueError(f"Not enough valid positions for {num_fires} fires. Only {len(valid_positions)} available.")

    fire_positions = random.sample(valid_positions, num_fires)

    # Place fires (intensity 2.0 like in the example)
    for row_idx, col_idx in fire_positions:
        new_fire_map[row_idx][col_idx] = 2.0

    # Update config
    config.initial_fire_map = new_fire_map
    return config

def replace_agents(config: SimulationConfig, num_agents: int=None) -> SimulationConfig:
    """
    Replace agent starting positions with random valid locations.
    Agents can't be placed on fire, doors, exits, or obstacles.

    Args:
        config: Simulation configuration
        num_agents: Number of agents (if None, uses config.agent_num)

    Returns:
        Updated configuration with new agent starting positions
    """
    if num_agents is None:
        num_agents = config.agent_num

    # Create set of invalid positions (fire, doors, exits, obstacles)
    invalid_positions = set()

    # Add door and exit positions
    if config.door_configs:
        for door in config.door_configs:
            col, row = stateNameToCoords(door['position'])
            invalid_positions.add((row, col))

    # Add fire and obstacle positions
    for row_idx, row in enumerate(config.initial_fire_map):
        for col_idx, cell in enumerate(row):
            # Fire (positive values) or obstacles (<0)
            if cell != 0:
                invalid_positions.add((row_idx, col_idx))

    # Find all valid positions
    valid_positions = []
    for row_idx in range(config.map_rows):
        for col_idx in range(config.map_cols):
            if (row_idx, col_idx) not in invalid_positions:
                valid_positions.append((row_idx, col_idx))

    # Randomly select agent positions
    if len(valid_positions) < num_agents:
        raise ValueError(f"Not enough valid positions for {num_agents} agents. Only {len(valid_positions)} available.")

    agent_positions = random.sample(valid_positions, num_agents)

    # Convert to state names format
    new_start_positions = [coordsToStateName(col_idx, row_idx) for row_idx, col_idx in agent_positions]

    # Update config
    config.start_positions = new_start_positions
    config.agent_num = num_agents
    return config


def run_monte_carlo_simulation(config: SimulationConfig, num_runs: int, random_seed: int = None) -> list:
    """
    Run multiple evacuation simulations and collect statistics.

    Args:
        config (SimulationConfig): Configuration for the simulation.
        num_runs (int): Number of simulation runs to perform.
    """

    config = replace_fire(config)
    config = replace_agents(config)

    random.seed(random_seed)
    results = []
    statistics = {}
    path_count = {}
    average_steps = 0
    average_fire_damage = 0
    average_peak_temp = 0
    average_avg_temp = 0
    evacuated_agents = 0
    survived_agents = 0
    for i in range(num_runs):
        print(f"Starting simulation run {i + 1}/{num_runs}")
        
        sim = EvacuationSimulation(config)
        result = sim.run(500)
        # path_count, steps, average_fire_damage, average_peak_temp, average_avg_temp, evacuated_agents, survived_agents
        results.append(result)
        path_count = dict(Counter(path_count) + Counter(result['path_count']))
        average_steps = (average_steps * i + result['steps']) / (i + 1)
        average_fire_damage = (average_fire_damage * i + result['average_fire_damage']) / (i + 1)
        average_peak_temp = (average_peak_temp * i + result['average_peak_temp']) / (i + 1)
        average_avg_temp = (average_avg_temp * i + result['average_avg_temp']) / (i + 1)
        evacuated_agents += result['evacuated_agents']
        survived_agents += result['survived_agents']
        print(f"Completed simulation run {i + 1}/{num_runs}\n")
    
    statistics['path_count'] = path_count
    statistics['average_steps'] = average_steps
    statistics['average_fire_damage'] = average_fire_damage
    statistics['average_peak_temp'] = average_peak_temp
    statistics['average_avg_temp'] = average_avg_temp
    statistics['evacuated_agents'] = evacuated_agents
    statistics['survived_agents'] = survived_agents

    return results, statistics

if __name__ == "__main__":
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

    sim_config = SimulationConfig.from_json(json_config)
    results, statistics = run_monte_carlo_simulation(sim_config, num_runs=10, random_seed=42)