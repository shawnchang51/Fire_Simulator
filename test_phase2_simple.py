"""
Simple Phase 2 Test
====================

Quick test to verify Phase 2 components work correctly.
"""

import numpy as np
import time

def test_optimized_dstar():
    """Test OptimizedDStarLite basic functionality."""
    print("Testing OptimizedDStarLite...")
    from optimized_d_star_lite import OptimizedDStarLite

    # Create simple grid
    grid = np.zeros((10, 10), dtype=np.float32)
    grid[0, :] = -2  # Top wall
    grid[-1, :] = -2  # Bottom wall
    grid[:, 0] = -2  # Left wall
    grid[:, -1] = -2  # Right wall

    start = (1, 1)
    goal = (8, 8)

    pathfinder = OptimizedDStarLite(grid, start, goal)
    pathfinder.compute_shortest_path()
    next_move = pathfinder.get_next_move()

    print(f"  Start: {start}, Goal: {goal}")
    print(f"  Next move: {next_move}")
    print(f"  ✓ OptimizedDStarLite works!")


def test_fast_fire():
    """Test FastFireModel basic functionality."""
    print("\nTesting FastFireModel...")
    from fast_fire import FastFireModel, DeterministicFireModel

    # Create simple grid
    grid = np.zeros((10, 10), dtype=np.float32)
    grid[5, 5] = 2.0  # Initial fire

    fire = FastFireModel(grid.copy())
    fire.set_seed(42)

    initial_fire = len(fire.get_fire_cells())
    fire.step_n(10)
    final_fire = len(fire.get_fire_cells())

    print(f"  Initial fire cells: {initial_fire}")
    print(f"  After 10 steps: {final_fire}")
    print(f"  ✓ FastFireModel works!")

    # Test deterministic
    fire_det = DeterministicFireModel(grid.copy())
    fire_det.step_n(10)
    det_fire = len(fire_det.get_fire_cells())
    print(f"  Deterministic fire cells: {det_fire}")
    print(f"  ✓ DeterministicFireModel works!")


def test_fast_simulation():
    """Test FastEvacuationSim basic functionality."""
    print("\nTesting FastEvacuationSim...")
    from fast_simulation import FastEvacuationSim

    # Create simple grid
    grid = np.zeros((20, 20), dtype=np.float32)
    grid[0, :] = -2
    grid[-1, :] = -2
    grid[:, 0] = -2
    grid[:, -1] = -2

    agent_starts = [(5, 5), (10, 10)]
    exits = [(18, 18)]
    fire_starts = [(10, 5)]

    start_time = time.time()
    sim = FastEvacuationSim(
        grid=grid,
        agent_starts=agent_starts,
        exits=exits,
        fire_starts=fire_starts,
        deterministic_fire=True
    )

    result = sim.run(max_steps=100)
    elapsed = time.time() - start_time

    print(f"  Simulation time: {elapsed*1000:.2f}ms")
    print(f"  Steps: {result.steps}")
    print(f"  Evacuated: {result.evacuated}")
    print(f"  Survival rate: {result.survival_rate:.2%}")
    print(f"  Termination: {result.termination_reason}")
    print(f"  ✓ FastEvacuationSim works!")


def test_pairwise_interface():
    """Test pairwise ranking interface."""
    print("\nTesting ScoringNetworkInterface...")
    from pairwise_ranking_interface import ScoringNetworkInterface

    interface = ScoringNetworkInterface(
        grid_size=(20, 20),
        num_trials_per_eval=2
    )

    # Create simple floor plan
    floor_plan = np.zeros((20, 20), dtype=np.float32)
    floor_plan[0, :] = -2
    floor_plan[-1, :] = -2
    floor_plan[:, 0] = -2
    floor_plan[:, -1] = -2
    floor_plan[10, 5] = 2.0  # Fire

    # Dummy door config
    door_config = [
        {'id': 'e1', 'position': 'x18y18', 'type': 'exit'}
    ]

    start_time = time.time()
    result = interface.evaluate_candidate(floor_plan, door_config)
    elapsed = time.time() - start_time

    print(f"  Evaluation time: {elapsed*1000:.2f}ms")
    print(f"  Survival rate: {result['survival_rate']:.2%}")
    print(f"  ✓ ScoringNetworkInterface works!")


if __name__ == '__main__':
    print("="*60)
    print("PHASE 2 SIMPLE FUNCTIONALITY TEST")
    print("="*60)

    try:
        test_optimized_dstar()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_fast_fire()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_fast_simulation()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_pairwise_interface()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE!")
    print("="*60)
