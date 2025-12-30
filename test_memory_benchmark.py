"""Memory benchmark test for the optimized simulation."""
import tracemalloc
import gc

def run_memory_test():
    """Test memory usage with a larger simulation."""
    print("=" * 60)
    print("Memory Benchmark Test")
    print("=" * 60)
    
    # Start memory tracking
    tracemalloc.start()
    gc.collect()
    
    from simulation import SimulationConfig, EvacuationSimulation
    
    # Create larger simulation
    rows, cols = 80, 80
    num_agents = 100
    
    initial_fire_map = [[0 for _ in range(cols)] for _ in range(rows)]
    initial_fire_map[25][25] = 0.3  # Start fire in center
    
    # Generate start positions in a grid pattern
    start_positions = []
    for i in range(num_agents):
        x = (i % 10) * 4 + 2
        y = (i // 10) * 8 + 2
        if x < cols and y < rows:
            start_positions.append(f'x{x}y{y}')
    
    config = SimulationConfig(
        map_rows=rows,
        map_cols=cols,
        max_occupancy=3,
        agent_num=len(start_positions),
        viewing_range=5,
        start_positions=start_positions,
        door_configs=[
            {'type': 'exit', 'id': 'exit1', 'position': 'x25y0', 'connected_ids': []},
            {'type': 'door', 'id': 'door1', 'position': 'x25y15', 'connected_ids': ['exit1']},
            {'type': 'door', 'id': 'door2', 'position': 'x10y25', 'connected_ids': ['door1']},
            {'type': 'door', 'id': 'door3', 'position': 'x40y25', 'connected_ids': ['door1']},
        ],
        initial_fire_map=initial_fire_map
    )
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"After config creation: {current/1024/1024:.2f} MB current, {peak/1024/1024:.2f} MB peak")
    
    # Create simulation
    sim = EvacuationSimulation(config, silent=True)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"After simulation init: {current/1024/1024:.2f} MB current, {peak/1024/1024:.2f} MB peak")
    
    # Run simulation
    result = sim.run(max_steps=200)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"After simulation run:  {current/1024/1024:.2f} MB current, {peak/1024/1024:.2f} MB peak")
    
    tracemalloc.stop()
    
    print(f"\nResults:")
    print(f"  - Steps: {result['steps']}")
    print(f"  - Evacuated: {result['evacuated_agents']}/{len(start_positions)}")
    print(f"  - Survived: {result['survived_agents']}")
    
    # Estimate per-agent memory
    per_agent_mb = peak / 1024 / 1024 / len(start_positions)
    print(f"\nEstimated memory per agent: {per_agent_mb:.3f} MB")
    print(f"Projected 300 agents: {per_agent_mb * 300:.1f} MB")
    
    return True


if __name__ == "__main__":
    run_memory_test()
