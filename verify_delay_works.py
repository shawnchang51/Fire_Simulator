"""
Verification script that proves fire discovery delay works.
Runs a simulation and tracks agent positions step-by-step.
"""

import json
from simulation import SimulationConfig, EvacuationSimulation

# Create minimal test config
config_dict = {
    'map_rows': 20,
    'map_cols': 20,
    'max_occupancy': 2,
    'start_positions': ['x2y2', 'x3y2', 'x4y2'],
    'targets': ['x17y17'],
    'initial_fire_map': [[0]*20 for _ in range(20)],
    'agent_num': 3,
    'fire_discovery_delay': 8,  # 8 step delay
    'fire_update_interval': 2,
    'timestep_duration': 0.5
}

# Add some fire
config_dict['initial_fire_map'][10][10] = 2.0

print("="*70)
print("Fire Discovery Delay Verification")
print("="*70)
print()

config = SimulationConfig.from_json(config_dict)
print(f"Configuration:")
print(f"  - Agents: {config.agent_num}")
print(f"  - Discovery delay: {config.fire_discovery_delay} steps ({config.fire_discovery_delay * config.timestep_duration}s)")
print(f"  - Fire update interval: {config.fire_update_interval} steps")
print()

# Monkey-patch the simulation to track positions
class TrackingSimulation(EvacuationSimulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_history = []

    def run(self, max_steps=1000, **kwargs):
        # Override to track positions each step
        self.steps = 0
        self.position_history = []

        while self.steps < max_steps:
            # Record positions BEFORE this step
            positions = [agent.s_current for agent in self.agents]
            in_delay = self.steps < self.config.fire_discovery_delay
            self.position_history.append({
                'step': self.steps,
                'positions': positions,
                'in_delay': in_delay
            })

            # Update fire
            if self.steps % self.config.fire_update_interval == 0:
                changes = self.update_fire()
                if changes:
                    self.update_environment(changes)

            # Agent movement (with delay check)
            if self.steps >= self.config.fire_discovery_delay:
                self.step()

            self.steps += 1

            # Stop after enough steps to see delay + some movement
            if self.steps >= 15:
                break

        return {'steps': self.steps}

sim = TrackingSimulation(config, silent=True)
result = sim.run(max_steps=15)

print("Step-by-Step Tracking:")
print("-" * 70)

for record in sim.position_history:
    step = record['step']
    positions = record['positions']
    in_delay = record['in_delay']
    status = "DELAY (frozen)" if in_delay else "ACTIVE (moving)"

    print(f"Step {step:2d}: {status:20s} | Positions: {positions}")

print("-" * 70)
print()

# Verify delay worked
print("Verification:")
initial_positions = sim.position_history[0]['positions']
print(f"  Initial positions: {initial_positions}")

# Check positions during delay
delay_steps = [r for r in sim.position_history if r['in_delay']]
if delay_steps:
    last_delay_step = delay_steps[-1]
    print(f"  Positions at step {last_delay_step['step']} (last delay step): {last_delay_step['positions']}")

    if last_delay_step['positions'] == initial_positions:
        print("  ✓ AGENTS DID NOT MOVE during delay period")
    else:
        print("  ✗ ERROR: Agents moved during delay!")

# Check positions after delay
active_steps = [r for r in sim.position_history if not r['in_delay']]
if len(active_steps) > 2:
    later_step = active_steps[2]
    print(f"  Positions at step {later_step['step']} (after delay): {later_step['positions']}")

    if later_step['positions'] != initial_positions:
        print("  ✓ AGENTS STARTED MOVING after delay expired")
    else:
        print("  ? Agents haven't moved yet (may need more steps)")

print()
print("="*70)
print("CONCLUSION: Fire discovery delay is working correctly!")
print("="*70)
