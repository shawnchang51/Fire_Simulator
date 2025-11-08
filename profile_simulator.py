"""
Profiling script for Fire Evacuation Simulator
Analyzes both CPU time and memory usage to identify bottlenecks
"""

import cProfile
import pstats
import io
import tracemalloc
import json
from simulation import EvacuationSimulation, SimulationConfig
import sys


def profile_time(config, max_steps=200):
    """Profile CPU time usage"""
    print("=" * 70)
    print("TIME PROFILING")
    print("=" * 70)

    profiler = cProfile.Profile()
    profiler.enable()

    sim = EvacuationSimulation(config)
    sim.run(max_steps=max_steps, show_visualization=False, use_pygame=False, use_matlab=False)

    profiler.disable()

    # Print statistics
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions

    print(s.getvalue())

    # Also print by total time
    print("\n" + "=" * 70)
    print("TOP FUNCTIONS BY TOTAL TIME")
    print("=" * 70)
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())

    return profiler


def profile_memory(config, max_steps=200):
    """Profile memory usage"""
    print("\n" + "=" * 70)
    print("MEMORY PROFILING")
    print("=" * 70)

    tracemalloc.start()

    # Take initial snapshot
    snapshot1 = tracemalloc.take_snapshot()

    sim = EvacuationSimulation(config)

    # Snapshot after initialization
    snapshot2 = tracemalloc.take_snapshot()

    sim.run(max_steps=max_steps, show_visualization=False, use_pygame=False, use_matlab=False)

    # Snapshot after simulation
    snapshot3 = tracemalloc.take_snapshot()

    # Get current memory usage
    current, peak = tracemalloc.get_traced_memory()

    print(f"\nCurrent memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

    # Analyze initialization
    print("\n" + "-" * 70)
    print("MEMORY ALLOCATION DURING INITIALIZATION")
    print("-" * 70)
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    for stat in top_stats[:15]:
        print(stat)

    # Analyze simulation run
    print("\n" + "-" * 70)
    print("MEMORY ALLOCATION DURING SIMULATION RUN")
    print("-" * 70)
    top_stats = snapshot3.compare_to(snapshot2, 'lineno')
    for stat in top_stats[:15]:
        print(stat)

    # Top memory consumers overall
    print("\n" + "-" * 70)
    print("TOP MEMORY CONSUMERS (OVERALL)")
    print("-" * 70)
    top_stats = snapshot3.statistics('lineno')
    for stat in top_stats[:20]:
        print(stat)

    tracemalloc.stop()

    return current, peak


def profile_step_by_step(config, num_steps=50):
    """Profile memory usage step by step to find what grows over time"""
    print("\n" + "=" * 70)
    print("STEP-BY-STEP MEMORY GROWTH ANALYSIS")
    print("=" * 70)

    tracemalloc.start()

    sim = EvacuationSimulation(config)

    memory_samples = []
    for step in range(num_steps):
        current, peak = tracemalloc.get_traced_memory()
        memory_samples.append({
            'step': step,
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024
        })

        # Run one step
        sim.step_count += 1
        sim._run_single_step()

        if step % 10 == 0:
            print(f"Step {step:3d}: {current / 1024 / 1024:6.2f} MB (peak: {peak / 1024 / 1024:6.2f} MB)")

    tracemalloc.stop()

    # Calculate memory growth rate
    if len(memory_samples) > 1:
        growth_rate = (memory_samples[-1]['current_mb'] - memory_samples[0]['current_mb']) / num_steps
        print(f"\nMemory growth rate: {growth_rate:.4f} MB per step")
        print(f"Estimated memory for 1000 steps: {memory_samples[0]['current_mb'] + growth_rate * 1000:.2f} MB")

    return memory_samples


if __name__ == "__main__":
    # Load configuration
    config_file = "example_configuration.json"

    print(f"Loading configuration from {config_file}...")
    with open(config_file, 'r') as f:
        config_data = json.load(f)

    config = SimulationConfig.from_json(config_data)

    print(f"Configuration loaded:")
    print(f"  Map size: {config.map_rows}x{config.map_cols}")
    print(f"  Agents: {config.agent_num}")
    print(f"  Fire model: {config.fire_model_type}")
    print(f"  Doors: {len(config.door_configs) if config.door_configs else 0}")
    print()

    # Run profiling
    max_steps = 200  # Adjust this for longer/shorter profiling

    # Time profiling
    profile_time(config, max_steps=max_steps)

    # Memory profiling
    current, peak = profile_memory(config, max_steps=max_steps)

    # Step-by-step analysis
    memory_samples = profile_step_by_step(config, num_steps=50)

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Steps simulated: {max_steps}")
    print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
    print(f"  Final memory: {current / 1024 / 1024:.2f} MB")
