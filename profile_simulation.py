"""
Profile the evacuation simulation for CPU time and memory usage.
"""
import cProfile
import pstats
import tracemalloc
import io
import json
from simulation import EvacuationSimulation, SimulationConfig

def profile_time():
    """Profile CPU time usage."""
    print("=" * 80)
    print("PROFILING CPU TIME")
    print("=" * 80)

    # Load configuration
    with open('example_configuration.json', 'r') as f:
        config_dict = json.load(f)
    # Remove comment fields
    config_dict = {k: v for k, v in config_dict.items() if not k.startswith('_')}
    config = SimulationConfig(**config_dict)

    # Create profiler
    profiler = cProfile.Profile()

    # Run simulation with profiling
    profiler.enable()
    sim = EvacuationSimulation(config)
    sim.run(show_visualization=False, use_pygame=False, use_matlab=False)
    profiler.disable()

    # Analyze results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')

    print("\nTop 30 functions by CUMULATIVE time:")
    print("-" * 80)
    ps.print_stats(30)
    print(s.getvalue())

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('tottime')

    print("\n" + "=" * 80)
    print("Top 30 functions by TOTAL time (time spent in function itself):")
    print("-" * 80)
    ps.print_stats(30)
    print(s.getvalue())

    return profiler

def profile_memory():
    """Profile memory usage."""
    print("\n" + "=" * 80)
    print("PROFILING MEMORY USAGE")
    print("=" * 80)

    # Load configuration
    with open('example_configuration.json', 'r') as f:
        config_dict = json.load(f)
    # Remove comment fields
    config_dict = {k: v for k, v in config_dict.items() if not k.startswith('_')}
    config = SimulationConfig(**config_dict)

    # Start memory tracking
    tracemalloc.start()

    # Take snapshot before simulation
    snapshot_before = tracemalloc.take_snapshot()

    # Run simulation
    sim = EvacuationSimulation(config)
    sim.run(show_visualization=False, use_pygame=False, use_matlab=False)

    # Take snapshot after simulation
    snapshot_after = tracemalloc.take_snapshot()

    # Get current memory usage
    current, peak = tracemalloc.get_traced_memory()

    print(f"\nCurrent memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

    # Compare snapshots to find biggest memory allocations
    print("\n" + "-" * 80)
    print("Top 20 memory allocations (difference from start):")
    print("-" * 80)

    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    for stat in top_stats[:20]:
        print(f"{stat}")

    print("\n" + "-" * 80)
    print("Top 20 memory allocations (by size):")
    print("-" * 80)

    top_stats = snapshot_after.statistics('lineno')
    for stat in top_stats[:20]:
        print(f"{stat}")

    tracemalloc.stop()

if __name__ == '__main__':
    # Profile CPU time
    profile_time()

    # Profile memory
    profile_memory()

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
