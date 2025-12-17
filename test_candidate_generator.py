"""
Test candidate generator with realistic floor plan from example configuration.
"""

import json
import numpy as np
from candidate_generator import generate_door_candidates, CandidateGenerator


def test_with_example_config():
    """Test candidate generator with example configuration floor plan."""
    print("Testing Candidate Generator with Example Configuration")
    print("=" * 60)

    # Load example configuration
    with open('example_configuration.json', 'r') as f:
        config = json.load(f)

    floor_plan = np.array(config['initial_fire_map'], dtype=np.float32)

    print(f"Floor plan shape: {floor_plan.shape}")
    print(f"Walls: {np.sum(floor_plan == -2)} cells")
    print(f"Passable: {np.sum(floor_plan >= 0)} cells")
    print()

    # Initialize generator to inspect analysis
    generator = CandidateGenerator(floor_plan, min_door_spacing=5, seed=42)

    print(f"Valid wall positions: {len(generator.valid_wall_positions)}")
    print(f"Rooms identified: {len(generator.rooms)}")
    print(f"Room boundaries: {len(generator.room_boundaries)}")
    print(f"Perimeter positions: {len(generator.perimeter_positions)}")
    print()

    # Test different generation strategies
    print("Testing generation strategies:")
    print("-" * 60)

    # 1. Random generation
    print("\n1. Random Candidate:")
    random_candidate = generator.generate_random_candidate(num_doors=3, num_exits=2)
    for door in random_candidate:
        print(f"   {door}")

    # 2. Boundary-focused
    print("\n2. Boundary-Focused Candidate:")
    boundary_candidate = generator.generate_rule_based_candidate(
        num_doors=3, num_exits=2, strategy='boundary_focused'
    )
    for door in boundary_candidate:
        print(f"   {door}")

    # 3. Distributed
    print("\n3. Distributed Candidate:")
    distributed_candidate = generator.generate_rule_based_candidate(
        num_doors=3, num_exits=2, strategy='distributed'
    )
    for door in distributed_candidate:
        print(f"   {door}")

    # 4. Corner exits
    print("\n4. Corner Exits Candidate:")
    corner_candidate = generator.generate_rule_based_candidate(
        num_doors=3, num_exits=2, strategy='corner_exits'
    )
    for door in corner_candidate:
        print(f"   {door}")

    # Generate a pool of candidates
    print("\n" + "=" * 60)
    print("Generating candidate pool...")

    candidates = generate_door_candidates(
        floor_plan=floor_plan,
        num_candidates=20,
        num_doors_range=(2, 5),
        num_exits_range=(1, 3),
        min_door_spacing=5,
        random_ratio=0.5,
        seed=42
    )

    print(f"Generated {len(candidates)} candidates")

    # Statistics
    door_counts = [len([d for d in c if d['type'] == 'door']) for c in candidates]
    exit_counts = [len([d for d in c if d['type'] == 'exit']) for c in candidates]

    print(f"\nCandidate statistics:")
    print(f"  Doors per candidate: min={min(door_counts)}, max={max(door_counts)}, avg={np.mean(door_counts):.1f}")
    print(f"  Exits per candidate: min={min(exit_counts)}, max={max(exit_counts)}, avg={np.mean(exit_counts):.1f}")

    # Show diversity: check unique positions
    all_positions = set()
    for candidate in candidates:
        for door in candidate:
            all_positions.add(door['position'])

    print(f"  Unique door positions used: {len(all_positions)}")

    # Show first 3 candidates in detail
    print("\nFirst 3 candidates in detail:")
    for i, candidate in enumerate(candidates[:3]):
        print(f"\nCandidate {i + 1} ({len(candidate)} total):")
        for door in candidate:
            print(f"  {door}")

    print("\n" + "=" * 60)
    print("Candidate generator test completed successfully!")


if __name__ == "__main__":
    test_with_example_config()
