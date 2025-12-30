"""
Phase 2 Performance Benchmark
==============================

Test and benchmark the Phase 2 optimized components:
- OptimizedDStarLite vs original D* Lite
- FastFireModel vs AdvancedFireModel
- FastEvacuationSim vs EvacuationSimulation

Target: 10-50x speedup over Phase 1
"""

import time
import numpy as np
from typing import List, Tuple
import sys

def create_test_floor_plan(size: int = 30) -> np.ndarray:
    """Create a test floor plan with walls and fire."""
    grid = np.zeros((size, size), dtype=np.float32)

    # Add perimeter walls
    grid[0, :] = -2
    grid[-1, :] = -2
    grid[:, 0] = -2
    grid[:, -1] = -2

    # Add some internal walls
    grid[size//2, 5:size-5] = -2

    # Add initial fire
    grid[size//4, size//4] = 2.0

    return grid


def benchmark_optimized_dstar_lite():
    """Benchmark OptimizedDStarLite pathfinding."""
    print("\n" + "="*60)
    print("Benchmarking OptimizedDStarLite")
    print("="*60)

    from optimized_d_star_lite import OptimizedDStarLite, SharedGridDStarLite

    grid = create_test_floor_plan(30)
    start = (5, 5)
    goal = (25, 25)

    # Test single pathfinder
    print("\nSingle agent pathfinding:")
    pathfinder = OptimizedDStarLite(grid, start, goal)

    start_time = time.time()
    pathfinder.compute_shortest_path()
    next_move = pathfinder.get_next_move()
    elapsed = time.time() - start_time

    print(f"  Initial path computation: {elapsed*1000:.2f}ms")
    print(f"  Next move: {next_move}")

    # Test replanning after environment change
    changed_cells = [(15, 15), (15, 16), (16, 15)]
    for x, y in changed_cells:
        grid[y, x] = 2.0

    start_time = time.time()
    pathfinder.update_edge_costs(changed_cells)
    next_move = pathfinder.get_next_move()
    elapsed = time.time() - start_time

    print(f"  Incremental replan: {elapsed*1000:.2f}ms")
    print(f"  New next move: {next_move}")

    # Test multi-agent shared grid
    print("\nMulti-agent shared grid (10 agents):")
    grid = create_test_floor_plan(30)
    manager = SharedGridDStarLite(grid)

    agents = []
    for i in range(10):
        start = (5 + i, 5)
        goal = (25, 25)
        agent = manager.add_agent(start, goal)
        agents.append(agent)

    start_time = time.time()
    for agent in agents:
        agent.compute_shortest_path()
    elapsed = time.time() - start_time

    print(f"  10 agents initial computation: {elapsed*1000:.2f}ms")
    print(f"  Average per agent: {elapsed*100:.2f}ms")

    # Test batch environment update
    changed_cells = [(15, 15), (15, 16), (16, 15)]
    start_time = time.time()
    manager.update_environment(changed_cells, spatial_filter=True)
    elapsed = time.time() - start_time

    print(f"  Batch update with spatial filter: {elapsed*1000:.2f}ms")


def benchmark_fast_fire():
    """Benchmark FastFireModel."""
    print("\n" + "="*60)
    print("Benchmarking FastFireModel")
    print("="*60)

    from fast_fire import FastFireModel, DeterministicFireModel

    grid = create_test_floor_plan(50)

    # Test FastFireModel
    print("\nFastFireModel (stochastic, 50x50 grid):")
    fire = FastFireModel(grid.copy())

    start_time = time.time()
    for _ in range(100):
        fire.step()
    elapsed = time.time() - start_time

    print(f"  100 steps: {elapsed*1000:.2f}ms")
    print(f"  Average per step: {elapsed*10:.2f}ms")
    print(f"  Fire cells: {len(fire.get_fire_cells())}")

    # Test DeterministicFireModel
    print("\nDeterministicFireModel (deterministic, 50x50 grid):")
    fire_det = DeterministicFireModel(grid.copy())

    start_time = time.time()
    for _ in range(100):
        fire_det.step()
    elapsed = time.time() - start_time

    print(f"  100 steps: {elapsed*1000:.2f}ms")
    print(f"  Average per step: {elapsed*10:.2f}ms")
    print(f"  Fire cells: {len(fire_det.get_fire_cells())}")


def benchmark_fast_simulation():
    """Benchmark FastEvacuationSim."""
    print("\n" + "="*60)
    print("Benchmarking FastEvacuationSim")
    print("="*60)

    from fast_simulation import FastEvacuationSim, evaluate_floor_plan

    grid = create_test_floor_plan(30)

    # Generate random agent starts
    agent_starts = [(10, 10), (12, 10), (10, 12), (15, 15), (20, 20)]
    exits = [(28, 28), (28, 1), (1, 28)]
    fire_starts = [(15, 15)]

    print("\nSingle simulation (5 agents, 30x30 grid):")
    start_time = time.time()

    sim = FastEvacuationSim(
        grid=grid.copy(),
        agent_starts=agent_starts,
        exits=exits,
        fire_starts=fire_starts,
        deterministic_fire=True,
        fire_update_interval=4
    )

    result = sim.run(max_steps=200)
    elapsed = time.time() - start_time

    print(f"  Time: {elapsed*1000:.2f}ms")
    print(f"  Steps: {result.steps}")
    print(f"  Evacuated: {result.evacuated}")
    print(f"  Stuck: {result.stuck}")
    print(f"  Dead: {result.dead}")
    print(f"  Survival rate: {result.survival_rate:.2%}")
    print(f"  Reward: {result.reward:.2f}")
    print(f"  Termination: {result.termination_reason}")

    # Benchmark multiple runs
    print("\n10 simulations (Monte Carlo):")
    start_time = time.time()

    results = []
    for seed in range(10):
        result = evaluate_floor_plan(
            floor_plan=grid.copy(),
            agent_positions=agent_starts,
            exit_positions=exits,
            fire_positions=fire_starts,
            max_steps=200,
            seed=seed
        )
        results.append(result)

    elapsed = time.time() - start_time

    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Average per sim: {elapsed*100:.2f}ms")
    print(f"  Throughput: {10/elapsed:.1f} sims/sec")
    print(f"  Estimated hourly: {int(10/elapsed * 3600)} sims")

    avg_survival = np.mean([r.survival_rate for r in results])
    print(f"  Average survival rate: {avg_survival:.2%}")


def benchmark_pairwise_interface():
    """Benchmark pairwise ranking interface."""
    print("\n" + "="*60)
    print("Benchmarking Pairwise Ranking Interface")
    print("="*60)

    from pairwise_ranking_interface import ScoringNetworkInterface

    interface = ScoringNetworkInterface(
        grid_size=(30, 30),
        num_trials_per_eval=3
    )

    floor_plan = create_test_floor_plan(30)

    # Create dummy door configs
    door_configs = [
        [{'id': 'd1', 'position': 'x5y15', 'type': 'door'},
         {'id': 'e1', 'position': 'x28y28', 'type': 'exit'}],
        [{'id': 'd1', 'position': 'x10y15', 'type': 'door'},
         {'id': 'e1', 'position': 'x28y1', 'type': 'exit'}],
        [{'id': 'd1', 'position': 'x15y10', 'type': 'door'},
         {'id': 'e1', 'position': 'x1y28', 'type': 'exit'}],
    ]

    print("\nEvaluating single candidate (3 Monte Carlo trials):")
    start_time = time.time()
    result = interface.evaluate_candidate(floor_plan, door_configs[0])
    elapsed = time.time() - start_time

    print(f"  Time: {elapsed*1000:.2f}ms")
    print(f"  Survival rate: {result['survival_rate']:.2%}")
    print(f"  Steps: {result['steps']}")

    print("\nGenerating pairwise labels (3 candidates, 3 pairs):")
    start_time = time.time()
    labels = interface.generate_candidate_labels(
        floor_plan=floor_plan,
        candidate_pool=door_configs,
        num_pairs=3,
        pair_selection='random'
    )
    elapsed = time.time() - start_time

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Labels generated: {len(labels)}")
    print(f"  Average per pair: {elapsed/3:.2f}s")

    if labels:
        print(f"\n  Sample label:")
        config_a, config_b, label, score_a, score_b = labels[0]
        print(f"    A score: {score_a:.4f}")
        print(f"    B score: {score_b:.4f}")
        print(f"    Label: {label} ({'A > B' if label == 1 else 'B > A'})")


def run_full_benchmark():
    """Run complete Phase 2 benchmark suite."""
    print("\n" + "="*60)
    print("PHASE 2 PERFORMANCE BENCHMARK")
    print("="*60)
    print("\nTarget: 10-50x speedup over Phase 1")
    print("Expected time per sim: 0.04-0.1s (40-100ms)")
    print("Expected throughput: 36,000-90,000 sims/hour")

    try:
        benchmark_optimized_dstar_lite()
    except Exception as e:
        print(f"\nERROR in OptimizedDStarLite benchmark: {e}")
        import traceback
        traceback.print_exc()

    try:
        benchmark_fast_fire()
    except Exception as e:
        print(f"\nERROR in FastFireModel benchmark: {e}")
        import traceback
        traceback.print_exc()

    try:
        benchmark_fast_simulation()
    except Exception as e:
        print(f"\nERROR in FastEvacuationSim benchmark: {e}")
        import traceback
        traceback.print_exc()

    try:
        benchmark_pairwise_interface()
    except Exception as e:
        print(f"\nERROR in Pairwise Interface benchmark: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Compare results to Phase 1 baseline")
    print("2. If performance targets met, proceed to Phase 3 (AI integration)")
    print("3. If not, profile bottlenecks and optimize further")


if __name__ == '__main__':
    run_full_benchmark()
