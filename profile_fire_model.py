"""
Profile the fire simulation to identify performance bottlenecks
"""

import cProfile
import pstats
import io
import json
import os
from simulation import EvacuationSimulation, SimulationConfig

def profile_simulation():
    """Run a short simulation with profiling enabled"""

    # Load configuration
    json_path = os.path.join(os.path.dirname(__file__), "example_configuration.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_config = json.load(f)
    config = SimulationConfig.from_json(json_config)

    # Create profiler
    profiler = cProfile.Profile()

    # Run simulation with profiling
    print("Starting profiled simulation...")
    profiler.enable()

    simulation = EvacuationSimulation(config)
    simulation.run(
        max_steps=50,  # Short run to identify bottlenecks
        show_visualization=False,
        use_pygame=False,
        use_matlab=False
    )

    profiler.disable()

    # Print statistics
    print("\n" + "="*80)
    print("PROFILING RESULTS - Top 30 functions by cumulative time")
    print("="*80)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

    print("\n" + "="*80)
    print("PROFILING RESULTS - Top 30 functions by total time")
    print("="*80)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())

    # Save detailed stats to file
    profiler.dump_stats('fire_simulation_profile.prof')
    print("\nDetailed profiling data saved to: fire_simulation_profile.prof")
    print("You can analyze it with: python -m pstats fire_simulation_profile.prof")

if __name__ == "__main__":
    profile_simulation()
