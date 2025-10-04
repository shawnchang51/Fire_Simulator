"""
Example script to run the evacuation simulation with MATLAB-style visualization
"""

import os
import json
from simulation import EvacuationSimulation, SimulationConfig

if __name__ == "__main__":
    # Load configuration from example file
    json_path = os.path.join(os.path.dirname(__file__), 'example_configuration.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        json_config = json.load(f)

    config = SimulationConfig.from_json(json_config)

    # Create simulation
    simulation = EvacuationSimulation(config)

    # Run with MATLAB-style visualization
    print("=" * 60)
    print("Fire Evacuation Simulation with MATLAB-Style Visualization")
    print("=" * 60)
    print("\nInteractive Controls:")
    print("  - Use checkboxes to toggle different data layers:")
    print("    • Temperature: Heat distribution in the environment")
    print("    • Oxygen: Oxygen level depletion due to fire")
    print("    • Smoke: Smoke density accumulation")
    print("    • Fuel: Remaining fuel for combustion")
    print("    • Fire: Fire intensity overlay")
    print("    • Trajectories: Agent movement paths (last 10 steps)")
    print("\nAgent trajectories are offset slightly to avoid overlap.")
    print("Each agent has a unique color for easy tracking.")
    print("\nClose the window to end the simulation.")
    print("=" * 60)
    print()

    # Run simulation with MATLAB visualizer
    simulation.run(
        max_steps=500,
        show_visualization=True,
        use_pygame=False,  # Disable pygame
        use_matlab=True    # Enable MATLAB-style visualization
    )

    print("\nSimulation complete!")
