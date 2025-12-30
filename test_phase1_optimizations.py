"""
Test and Benchmark Phase 1 RL Optimizations
===========================================

Tests the Phase 1 conservative optimizations:
1. Training-optimized configuration
2. Early termination logic
3. RL simulation wrapper
4. NumPy fire map conversion
"""

import time
import numpy as np
from rl_simulation import RLSimulationWrapper, evaluate_floor_plan
from simulation import EvacuationSimulation, SimulationConfig
import json


def test_early_termination():
    """Test early termination logic with a bad design."""
    print("\n" + "="*70)
    print("TEST 1: Early Termination Logic")
    print("="*70)

    # Create a scenario where agents will get stuck (no path to exit)
    floor_plan = np.zeros((30, 30), dtype=np.float32)

    # Create a box that traps agents
    floor_plan[10:20, 10:20] = -2  # Walls
    floor_plan[14:16, 14:16] = 0   # Clear center

    # Place agents inside the box (trapped)
    agents = [(14, 14), (15, 15)]

    # Place exit outside the box
    exits = [(29, 15)]

    # No fire initially
    fires = []

    wrapper = RLSimulationWrapper()

    start_time = time.time()
    result = wrapper.evaluate(floor_plan, agents, exits, fires, max_steps=500)
    elapsed = time.time() - start_time

    print(f"\nResults (should terminate early due to stuck agents):")
    print(f"  Elapsed time: {elapsed:.3f}s")
    print(f"  Steps taken: {result['steps']}")
    print(f"  Termination reason: {result.get('termination_reason', 'N/A')}")
    print(f"  Evacuated: {result['evacuated_agents']}/{len(agents)}")
    print(f"  Stuck: {result['stuck_count']}")
    print(f"  Reward: {result.get('reward', 'N/A'):.2f}")

    # Verify early termination worked
    assert result['steps'] < 500, "Should terminate before max_steps"
    assert result.get('termination_reason') in ['mostly_stuck', 'all_resolved'], \
        f"Expected early termination, got {result.get('termination_reason')}"

    print("  [OK] Early termination test PASSED")
    return elapsed


def test_rl_wrapper_basic():
    """Test basic RL wrapper functionality."""
    print("\n" + "="*70)
    print("TEST 2: RL Wrapper Basic Functionality")
    print("="*70)

    # Create a simple passable floor plan
    floor_plan = np.zeros((30, 30), dtype=np.float32)

    # Add minimal walls
    floor_plan[0, :] = -2  # Top wall
    floor_plan[29, :] = -2  # Bottom wall
    floor_plan[:, 0] = -2  # Left wall
    floor_plan[:, 29] = -2  # Right wall

    agents = [(5, 5), (10, 10), (15, 15)]
    exits = [(28, 28)]
    fires = [(12, 12)]

    # Override door_configs to None to avoid door graph issues
    wrapper = RLSimulationWrapper()

    start_time = time.time()
    result = wrapper.evaluate(floor_plan, agents, exits, fires, max_steps=200)
    elapsed = time.time() - start_time

    print(f"\nResults:")
    print(f"  Elapsed time: {elapsed:.3f}s")
    print(f"  Steps: {result['steps']}")
    print(f"  Termination: {result.get('termination_reason', 'N/A')}")
    print(f"  Evacuated: {result['evacuated_agents']}/{len(agents)}")
    print(f"  Survival rate: {result['survival_rate']:.1%}")
    print(f"  Reward: {result.get('reward', 'N/A'):.2f}")

    # Verify result structure
    assert 'steps' in result
    assert 'evacuated_agents' in result
    assert 'survival_rate' in result
    assert 'reward' in result
    assert 'termination_reason' in result

    print("  [OK] RL wrapper test PASSED")
    return elapsed


def test_numpy_fire_map():
    """Test that numpy fire maps are handled correctly."""
    print("\n" + "="*70)
    print("TEST 3: NumPy Fire Map Handling")
    print("="*70)

    # Load the RL training config
    with open('configs/rl_training_config.json', 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # Test 1: List-based fire map (traditional)
    config1 = SimulationConfig.from_json(config_dict)
    assert isinstance(config1.initial_fire_map, np.ndarray), \
        "Fire map should be converted to numpy array"
    print("  [OK] List-based fire map converted to numpy")

    # Test 2: NumPy-based fire map (new)
    config_dict2 = config_dict.copy()
    config_dict2['initial_fire_map'] = np.array(config_dict['initial_fire_map'], dtype=np.float32)
    config2 = SimulationConfig.from_json(config_dict2)
    assert isinstance(config2.initial_fire_map, np.ndarray), \
        "Fire map should remain as numpy array"
    print("  [OK] NumPy fire map preserved")

    # Verify they're equivalent
    assert np.array_equal(config1.initial_fire_map, config2.initial_fire_map), \
        "Both approaches should produce identical fire maps"
    print("  [OK] Both methods produce identical results")

    print("  [OK] NumPy fire map test PASSED")


def benchmark_phase1_speedup():
    """Benchmark speedup from Phase 1 optimizations."""
    print("\n" + "="*70)
    print("BENCHMARK: Phase 1 Speedup Comparison")
    print("="*70)

    # Create test scenario
    floor_plan = np.zeros((30, 30), dtype=np.float32)
    floor_plan[0, :] = -2
    floor_plan[29, :] = -2
    floor_plan[:, 0] = -2
    floor_plan[:, 29] = -2

    agents = [(5, 5), (10, 10), (15, 15), (20, 20), (25, 25)]
    exits = [(28, 15)]
    fires = [(12, 12)]

    # Test with Phase 1 optimizations (RL wrapper)
    print("\n1. With Phase 1 Optimizations (RL wrapper):")
    wrapper = RLSimulationWrapper()

    times_optimized = []
    for i in range(5):
        start = time.time()
        result = wrapper.evaluate(floor_plan, agents, exits, fires, max_steps=200)
        elapsed = time.time() - start
        times_optimized.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.3f}s (steps={result['steps']}, "
              f"evacuated={result['evacuated_agents']})")

    avg_optimized = np.mean(times_optimized)
    std_optimized = np.std(times_optimized)
    print(f"   Average: {avg_optimized:.3f}s ± {std_optimized:.3f}s")

    # Test without optimizations (standard simulation)
    print("\n2. Without Phase 1 Optimizations (standard config):")

    # Load standard config for comparison
    with open('example_configuration.json', 'r', encoding='utf-8') as f:
        standard_config = json.load(f)

    # Adjust to match test scenario
    standard_config['map_rows'] = 30
    standard_config['map_cols'] = 30
    standard_config['agent_num'] = 5
    standard_config['start_positions'] = [f'x{x}y{y}' for x, y in agents]
    standard_config['targets'] = [f'x{x}y{y}' for x, y in exits]
    standard_config['door_configs'] = None  # No door configs for fair comparison

    # Create fire map with fire
    fire_map = [[0.0] * 30 for _ in range(30)]
    for fx, fy in fires:
        fire_map[fy][fx] = 2.0
    standard_config['initial_fire_map'] = fire_map

    config_std = SimulationConfig.from_json(standard_config)

    times_standard = []
    for i in range(5):
        start = time.time()
        sim = EvacuationSimulation(config_std, silent=True)
        result = sim.run(max_steps=200, show_visualization=False,
                        use_pygame=False, use_matlab=False)
        elapsed = time.time() - start
        times_standard.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.3f}s (steps={result['steps']}, "
              f"evacuated={result['evacuated_agents']})")

    avg_standard = np.mean(times_standard)
    std_standard = np.std(times_standard)
    print(f"   Average: {avg_standard:.3f}s ± {std_standard:.3f}s")

    # Calculate speedup
    speedup = avg_standard / avg_optimized if avg_optimized > 0 else 0

    print("\n" + "="*70)
    print("SPEEDUP ANALYSIS")
    print("="*70)
    print(f"  Standard simulation:  {avg_standard:.3f}s")
    print(f"  Phase 1 optimized:    {avg_optimized:.3f}s")
    print(f"  Speedup:              {speedup:.2f}x")
    print(f"  Time saved:           {avg_standard - avg_optimized:.3f}s ({(1-avg_optimized/avg_standard)*100:.1f}%)")

    if speedup >= 2.0:
        print(f"  [OK] Achieved target speedup (>= 2x)!")
    else:
        print(f"  [WARN] Below target speedup (expected >= 2x)")

    return speedup


def test_batch_evaluation():
    """Test batch evaluation capability."""
    print("\n" + "="*70)
    print("TEST 4: Batch Evaluation")
    print("="*70)

    wrapper = RLSimulationWrapper()

    # Create 10 test scenarios
    scenarios = []
    for i in range(10):
        scenario = {
            'agent_positions': [(5, 5), (10, 10), (15, 15)],
            'exit_positions': [(29, 15)],
            'fire_positions': [(10 + i, 10)],  # Different fire positions
            'max_steps': 200
        }
        scenarios.append(scenario)

    print(f"\nEvaluating {len(scenarios)} scenarios in parallel...")
    start = time.time()
    results = wrapper.batch_evaluate(scenarios, num_workers=4)
    elapsed = time.time() - start

    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Time per scenario: {elapsed/len(scenarios):.3f}s")
    print(f"  Throughput: {len(scenarios)/elapsed:.1f} scenarios/second")

    # Verify all results
    assert len(results) == len(scenarios), "Should have result for each scenario"
    for i, result in enumerate(results):
        assert 'reward' in result, f"Scenario {i} missing reward"
        print(f"  Scenario {i+1}: Reward={result['reward']:.2f}, "
              f"Evacuated={result['evacuated_agents']}, Steps={result['steps']}")

    print("  [OK] Batch evaluation test PASSED")
    return elapsed


def main():
    """Run all Phase 1 tests and benchmarks."""
    print("="*70)
    print("PHASE 1 RL OPTIMIZATION TESTS")
    print("="*70)
    print("\nTesting conservative optimizations (1-2 days, 3-5x speedup target)")

    try:
        # Run all tests
        test_numpy_fire_map()
        test_rl_wrapper_basic()
        test_early_termination()
        test_batch_evaluation()

        # Run benchmark
        speedup = benchmark_phase1_speedup()

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("  [OK] All tests PASSED")
        print(f"  [OK] Measured speedup: {speedup:.2f}x")

        if speedup >= 3.0:
            print("  [OK] EXCELLENT: Exceeded target speedup (3-5x)")
        elif speedup >= 2.0:
            print("  [OK] GOOD: Approaching target speedup")
        else:
            print("  [WARN] Note: Speedup lower than expected")
            print("    (Config optimizations may show bigger gains on larger grids/agents)")

        print("\n  Phase 1 optimizations implemented successfully!")
        print("="*70)

    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
