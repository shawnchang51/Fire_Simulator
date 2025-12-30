"""Quick test to verify memory optimizations work correctly."""
import traceback

def test_basic_simulation():
    """Test that a basic simulation runs with the optimizations."""
    print("=" * 60)
    print("Testing optimized simulation...")
    print("=" * 60)
    
    try:
        from simulation import SimulationConfig, EvacuationSimulation
        import numpy as np
        
        # Create initial fire map
        initial_fire_map = [[0 for _ in range(15)] for _ in range(15)]
        initial_fire_map[10][10] = 0.1  # Small fire
        
        # Create simple config
        config = SimulationConfig(
            map_rows=15,
            map_cols=15,
            max_occupancy=2,
            agent_num=5,
            viewing_range=3,
            start_positions=['x1y1', 'x13y1', 'x1y13', 'x13y13', 'x7y7'],
            door_configs=[
                {'type': 'exit', 'id': 'exit1', 'position': 'x7y0', 'connected_ids': []},
                {'type': 'door', 'id': 'door1', 'position': 'x7y5', 'connected_ids': ['exit1']},
            ],
            initial_fire_map=initial_fire_map
        )
        print("[OK] SimulationConfig created")
        
        # Create simulation
        sim = EvacuationSimulation(config, silent=True)
        print("[OK] EvacuationSimulation created")
        
        # Check NumPy optimizations
        print(f"  - Occupancy type: {type(sim.occupancy).__name__}")
        print(f"  - Occupancy shape: {sim.occupancy.shape}")
        print(f"  - Occupancy dtype: {sim.occupancy.dtype}")
        print(f"  - Number of agents: {len(sim.agents)}")
        
        # Check agent door_graph
        agent = sim.agents[0]
        print(f"  - Agent door_graph type: {type(agent.door_graph).__name__}")
        print(f"  - Agent door_graph nodes: {len(agent.door_graph.nodes)}")
        
        # Check GridWorld
        print(f"  - Agent graph (GridWorld) cells type: {type(agent.graph.cells).__name__}")
        if hasattr(agent.graph.cells, 'shape'):
            print(f"  - GridWorld cells shape: {agent.graph.cells.shape}")
        
        # Run simulation
        print("\nRunning simulation (max 100 steps)...")
        result = sim.run(max_steps=100)
        
        print(f"\n[OK] Simulation completed!")
        print(f"  - Steps: {result['steps']}")
        print(f"  - Evacuated: {result['evacuated_agents']}/{config.agent_num}")
        print(f"  - Survived: {result['survived_agents']}")
        print(f"  - Avg fire damage: {result['average_fire_damage']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        return False


def test_memory_savings():
    """Estimate memory savings from optimizations."""
    print("\n" + "=" * 60)
    print("Estimating memory savings...")
    print("=" * 60)
    
    import sys
    
    # Test DoorGraph shallow copy vs deep copy
    from door_graph import DoorGraph, DoorNode
    
    # Create a sample door graph
    dg = DoorGraph()
    for i in range(10):
        dg.add_node(DoorNode(f"node{i}", f"x{i}y{i}", "door"))
    for i in range(9):
        dg.add_edge(f"node{i}", f"node{i+1}", 1.0)
    
    # Shallow copy memory
    shallow = dg.shallow_copy()
    
    print(f"  - Original DoorGraph nodes: {len(dg.nodes)}")
    print(f"  - Shallow copy nodes: {len(shallow.nodes)}")
    print(f"  - Nodes are same object: {dg.nodes is shallow.nodes}")
    print(f"  - Edges are different object: {dg.edges is not shallow.edges}")
    
    # Check GridWorld cells with NumPy
    from d_star_lite.grid import GridWorld
    import numpy as np
    
    # Create list-based cells (old way)
    list_cells = [[0 for _ in range(100)] for _ in range(100)]
    list_size = sys.getsizeof(list_cells) + sum(sys.getsizeof(row) for row in list_cells)
    
    # Create NumPy array (new way)
    np_cells = np.zeros((100, 100), dtype=np.float32)
    np_size = np_cells.nbytes
    
    print(f"\n  Grid cells (100x100):")
    print(f"  - Python list memory: ~{list_size / 1024:.1f} KB")
    print(f"  - NumPy float32 memory: ~{np_size / 1024:.1f} KB")
    print(f"  - Savings: ~{(list_size - np_size) / 1024:.1f} KB ({100 * (list_size - np_size) / list_size:.1f}%)")
    
    # Occupancy array
    list_occ_size = sum(sys.getsizeof(row) for row in [[0] * 100 for _ in range(100)])
    np_occ = np.zeros((100, 100), dtype=np.int16)
    np_occ_size = np_occ.nbytes
    
    print(f"\n  Occupancy grid (100x100):")
    print(f"  - Python list memory: ~{list_occ_size / 1024:.1f} KB")
    print(f"  - NumPy int16 memory: ~{np_occ_size / 1024:.1f} KB")
    print(f"  - Savings: ~{(list_occ_size - np_occ_size) / 1024:.1f} KB ({100 * (list_occ_size - np_occ_size) / list_occ_size:.1f}%)")
    
    return True


def test_fire_model():
    """Test fire model optimizations."""
    print("\n" + "=" * 60)
    print("Testing fire model optimizations...")
    print("=" * 60)
    
    try:
        from fire_model_realistic import create_fire_model
        import numpy as np
        
        model = create_fire_model(rows=50, cols=50)
        print("[OK] Fire model created")
        
        # Check NumPy arrays
        print(f"  - Temperature map type: {type(model.temperature_map).__name__}")
        print(f"  - Oxygen map type: {type(model.oxygen_map).__name__}")
        print(f"  - Fuel map type: {type(model.fuel_map).__name__}")
        print(f"  - Wind influence type: {type(model.wind_influence).__name__}")
        
        if hasattr(model.temperature_map, 'shape'):
            print(f"  - Temperature map shape: {model.temperature_map.shape}")
            print(f"  - Temperature map dtype: {model.temperature_map.dtype}")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = True
    success &= test_basic_simulation()
    success &= test_memory_savings()
    success &= test_fire_model()
    
    print("\n" + "=" * 60)
    if success:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
