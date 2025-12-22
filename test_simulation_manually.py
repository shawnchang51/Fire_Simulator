"""
Manually test if simulations work with generated floor plans
"""

import numpy as np
from fast_simulation import FastEvacuationSim
from fast_fire import FireSpreadMode

# Load a generated plan
data = np.load('test_training_data_v2/floor_plans/plan_00004.npz')
grid = data['grid']
exit_positions = [tuple(pos) for pos in data['exit_positions']]  # Convert to tuples

print(f"Grid shape: {grid.shape}")
print(f"Exits: {exit_positions}")
print(f"Grid min/max: {grid.min()}, {grid.max()}")

# Test placing agents
passable_cells = np.argwhere(grid == 0)
print(f"Passable cells: {len(passable_cells)}")

if len(passable_cells) == 0:
    print("ERROR: No passable cells!")
    exit(1)

# Place a few agents randomly
num_agents = 10
agent_indices = np.random.choice(len(passable_cells), min(num_agents, len(passable_cells)), replace=False)
agent_positions = [(int(passable_cells[i][1]), int(passable_cells[i][0])) for i in agent_indices]  # (col, row) format as tuples

print(f"Agent positions (first 3): {agent_positions[:3]}")

# Place fire - need (col, row) format as tuple
fire_cell = passable_cells[0]  # (row, col) from argwhere
fire_positions = [(int(fire_cell[1]), int(fire_cell[0]))]  # Convert to (col, row) tuple
print(f"Fire position: {fire_positions}")

# Create simulation config
config = {
    'map_rows': grid.shape[0],
    'map_cols': grid.shape[1],
    'cell_size': 0.3,
    'timestep_duration': 0.5,
    'fire_update_interval': 2,
    'agent_num': len(agent_positions),
    'max_occupancy': 2,
    'viewing_range': 10,
    'fire_spread_rate': 0.3,
    'fire_intensity_growth': 0.5,
    'fire_damage_threshold': 10.0,
    'fire_discovery_delay': 0,
    'max_steps': 200
}

print("\nTrying to create simulation...")
try:
    sim = FastEvacuationSim(
        grid=grid.copy(),
        agent_starts=agent_positions,
        exits=exit_positions,
        fire_starts=fire_positions,
        fire_update_interval=2,
        fire_discovery_delay=0,
        fire_spread_mode='always_real',
        fire_spread_rate=0.3,
        fire_intensity_growth=0.5,
        fire_damage_threshold=10.0
    )
    print("Simulation created successfully!")

    print("\nRunning simulation...")
    result = sim.run(max_steps=config['max_steps'])

    print(f"\nResult:")
    print(f"  Survival rate: {result.survival_rate:.1%}")
    print(f"  Steps: {result.steps}")
    print(f"  Evacuated: {result.evacuated}")
    print(f"  Dead: {result.dead}")
    print(f"  Stuck: {result.stuck}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
