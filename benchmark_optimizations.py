"""
Benchmark script to compare original vs optimized simulator

Measures:
1. Execution time improvements
2. Memory usage reductions
3. Performance metrics (steps/sec, fire updates/sec)
"""

import time
import tracemalloc
import json
import sys
import copy


def benchmark_fire_model(config_file="example_configuration.json", steps=200):
    """Benchmark fire model - original vs optimized"""
    print("=" * 70)
    print("FIRE MODEL BENCHMARK")
    print("=" * 70)

    with open(config_file, 'r') as f:
        config_data = json.load(f)

    rows = config_data['map_rows']
    cols = config_data['map_cols']
    initial_fire = config_data.get('initial_fire_map', [[0]*cols for _ in range(rows)])

    # Set up some initial fires
    fire_state = copy.deepcopy(initial_fire)
    fire_count = sum(1 for row in fire_state for cell in row if 0 < cell <= 4)
    if fire_count == 0:
        # Add some fires for testing
        fire_state[10][10] = 2.0
        fire_state[20][20] = 3.0
        fire_state[30][30] = 2.5

    print(f"Map size: {rows}x{cols}")
    print(f"Initial fires: {fire_count if fire_count > 0 else 3}")
    print()

    # Benchmark ORIGINAL fire model
    print("Testing ORIGINAL fire model...")
    try:
        from fire_model_aggressive import AdvancedFireModel as OriginalModel

        tracemalloc.start()
        start_time = time.time()

        model_orig = OriginalModel(rows, cols)

        # Run updates
        current_state = copy.deepcopy(fire_state)
        for i in range(steps // 4):  # Fire updates every 4 steps
            model_orig.simulate_step(current_state)

        elapsed_orig = time.time() - start_time
        current_mem_orig, peak_mem_orig = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"  Time: {elapsed_orig:.2f}s")
        print(f"  Peak memory: {peak_mem_orig / 1024 / 1024:.2f} MB")
        print(f"  Updates/sec: {(steps // 4) / elapsed_orig:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        elapsed_orig = None
        peak_mem_orig = None

    # Benchmark OPTIMIZED fire model
    print("\nTesting OPTIMIZED fire model...")
    try:
        from fire_model_aggressive_optimized import AdvancedFireModel as OptimizedModel

        tracemalloc.start()
        start_time = time.time()

        model_opt = OptimizedModel(rows, cols)

        # Run updates
        current_state = copy.deepcopy(fire_state)
        for i in range(steps // 4):
            model_opt.simulate_step(current_state)

        elapsed_opt = time.time() - start_time
        current_mem_opt, peak_mem_opt = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"  Time: {elapsed_opt:.2f}s")
        print(f"  Peak memory: {peak_mem_opt / 1024 / 1024:.2f} MB")
        print(f"  Updates/sec: {(steps // 4) / elapsed_opt:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        elapsed_opt = None
        peak_mem_opt = None

    # Calculate improvements
    print("\n" + "-" * 70)
    print("FIRE MODEL IMPROVEMENTS")
    print("-" * 70)
    if elapsed_orig and elapsed_opt:
        speedup = elapsed_orig / elapsed_opt
        time_saved = elapsed_orig - elapsed_opt
        time_saved_pct = (time_saved / elapsed_orig) * 100
        print(f"Speedup: {speedup:.2f}x")
        print(f"Time saved: {time_saved:.2f}s ({time_saved_pct:.1f}%)")

    if peak_mem_orig and peak_mem_opt:
        mem_saved = (peak_mem_orig - peak_mem_opt) / 1024 / 1024
        mem_saved_pct = (mem_saved / (peak_mem_orig / 1024 / 1024)) * 100
        print(f"Memory saved: {mem_saved:.2f} MB ({mem_saved_pct:.1f}%)")

    return {
        'original_time': elapsed_orig,
        'optimized_time': elapsed_opt,
        'original_memory': peak_mem_orig,
        'optimized_memory': peak_mem_opt
    }


def benchmark_full_simulation(config_file="example_configuration.json", steps=200):
    """Benchmark full simulation - original vs optimized"""
    print("\n" + "=" * 70)
    print("FULL SIMULATION BENCHMARK")
    print("=" * 70)

    with open(config_file, 'r') as f:
        config_data = json.load(f)

    print(f"Map size: {config_data['map_rows']}x{config_data['map_cols']}")
    print(f"Agents: {config_data['agent_num']}")
    print(f"Steps: {steps}")
    print()

    # Benchmark ORIGINAL simulation
    print("Testing ORIGINAL simulation...")
    try:
        # Import with original fire model
        import importlib
        import fire_model_aggressive
        importlib.reload(fire_model_aggressive)

        from simulation import EvacuationSimulation, SimulationConfig

        config = SimulationConfig.from_json(config_data)

        tracemalloc.start()
        start_time = time.time()

        sim = EvacuationSimulation(config)
        result = sim.run(max_steps=steps, show_visualization=False,
                        use_pygame=False, use_matlab=False)

        elapsed_orig = time.time() - start_time
        current_mem_orig, peak_mem_orig = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"  Time: {elapsed_orig:.2f}s")
        print(f"  Peak memory: {peak_mem_orig / 1024 / 1024:.2f} MB")
        print(f"  Steps/sec: {steps / elapsed_orig:.2f}")
        if result:
            print(f"  Evacuated: {result.get('evacuated_agents', 0)}/{result.get('total_agents', 0)}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        elapsed_orig = None
        peak_mem_orig = None

    # For optimized, we would need to modify the simulation.py to use optimized modules
    # For now, let's just show the fire model improvements
    print("\nNOTE: Full simulation optimization requires modifying simulation.py")
    print("      to import fire_model_aggressive_optimized and grid_optimized")

    return {
        'original_time': elapsed_orig,
        'original_memory': peak_mem_orig
    }


def benchmark_grid_operations():
    """Benchmark D* Lite grid operations"""
    print("\n" + "=" * 70)
    print("D* LITE GRID BENCHMARK")
    print("=" * 70)

    rows, cols = 60, 60
    print(f"Grid size: {rows}x{cols}")
    print()

    # Benchmark ORIGINAL grid
    print("Testing ORIGINAL grid...")
    try:
        from d_star_lite.grid import GridWorld as OriginalGrid

        start_time = time.time()
        grid_orig = OriginalGrid(cols, rows, connect8=True, fire_fearness=1.0)

        # Simulate terrain updates
        for i in range(100):
            grid_orig.cells[i % rows][i % cols] = float(i % 4)
            grid_orig.updateGraphFromTerrain()

        elapsed_orig = time.time() - start_time
        print(f"  Time: {elapsed_orig:.2f}s")
        print(f"  Updates/sec: {100 / elapsed_orig:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        elapsed_orig = None

    # Benchmark OPTIMIZED grid
    print("\nTesting OPTIMIZED grid...")
    try:
        from d_star_lite.grid_optimized import GridWorld as OptimizedGrid

        start_time = time.time()
        grid_opt = OptimizedGrid(cols, rows, connect8=True, fire_fearness=1.0)

        # Simulate terrain updates
        for i in range(100):
            grid_opt.setCellValue(i % rows, i % cols, float(i % 4))
            grid_opt.updateGraphFromTerrain()

        elapsed_opt = time.time() - start_time
        print(f"  Time: {elapsed_opt:.2f}s")
        print(f"  Updates/sec: {100 / elapsed_opt:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        elapsed_opt = None

    # Calculate improvements
    print("\n" + "-" * 70)
    print("GRID IMPROVEMENTS")
    print("-" * 70)
    if elapsed_orig and elapsed_opt:
        speedup = elapsed_orig / elapsed_opt
        time_saved_pct = ((elapsed_orig - elapsed_opt) / elapsed_orig) * 100
        print(f"Speedup: {speedup:.2f}x")
        print(f"Time saved: {time_saved_pct:.1f}%")

    return {
        'original_time': elapsed_orig,
        'optimized_time': elapsed_opt
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FIRE SIMULATOR OPTIMIZATION BENCHMARK")
    print("=" * 70)
    print("\nComparing original vs optimized implementations\n")

    # Run benchmarks
    fire_results = benchmark_fire_model(steps=200)
    grid_results = benchmark_grid_operations()
    sim_results = benchmark_full_simulation(steps=200)

    # Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    print("\nFire Model:")
    if fire_results.get('original_time') and fire_results.get('optimized_time'):
        speedup = fire_results['original_time'] / fire_results['optimized_time']
        print(f"  Speedup: {speedup:.2f}x faster")
        print(f"  Time: {fire_results['original_time']:.2f}s → {fire_results['optimized_time']:.2f}s")

    if fire_results.get('original_memory') and fire_results.get('optimized_memory'):
        mem_orig_mb = fire_results['original_memory'] / 1024 / 1024
        mem_opt_mb = fire_results['optimized_memory'] / 1024 / 1024
        print(f"  Memory: {mem_orig_mb:.2f} MB → {mem_opt_mb:.2f} MB")

    print("\nD* Lite Grid:")
    if grid_results.get('original_time') and grid_results.get('optimized_time'):
        speedup = grid_results['original_time'] / grid_results['optimized_time']
        print(f"  Speedup: {speedup:.2f}x faster")
        print(f"  Time: {grid_results['original_time']:.2f}s → {grid_results['optimized_time']:.2f}s")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
