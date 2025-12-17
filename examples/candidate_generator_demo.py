"""
Candidate Generator Demo for AI-Guided Design Optimization
===========================================================

Demonstrates how to use the candidate generator for generating
door configuration candidates for pairwise comparison labeling.

This is part of Phase 1: Conservative Optimizations
"""

import json
import numpy as np
import sys
sys.path.append('..')
from candidate_generator import generate_door_candidates, CandidateGenerator


def demo_basic_usage():
    """Basic usage: Generate candidates from a floor plan."""
    print("=" * 70)
    print("DEMO 1: Basic Candidate Generation")
    print("=" * 70)

    # Load example floor plan
    with open('../example_configuration.json', 'r') as f:
        config = json.load(f)

    floor_plan = np.array(config['initial_fire_map'], dtype=np.float32)

    # Generate 50 candidates
    candidates = generate_door_candidates(
        floor_plan=floor_plan,
        num_candidates=50,
        num_doors_range=(2, 4),
        num_exits_range=(1, 2),
        min_door_spacing=5,
        random_ratio=0.5,
        seed=42
    )

    print(f"Generated {len(candidates)} candidates")
    print(f"\nExample candidate:")
    for door in candidates[0]:
        print(f"  {door}")

    print("\n")


def demo_advanced_usage():
    """Advanced usage: Control generation strategies."""
    print("=" * 70)
    print("DEMO 2: Advanced - Control Generation Strategies")
    print("=" * 70)

    # Create a simple test floor plan
    test_plan = np.zeros((40, 40))

    # Add perimeter walls
    test_plan[0:3, :] = -2
    test_plan[-3:, :] = -2
    test_plan[:, 0:3] = -2
    test_plan[:, -3:] = -2

    # Add some internal walls to create rooms
    test_plan[15:25, 18:20] = -2  # Vertical wall
    test_plan[18:20, 10:30] = -2  # Horizontal wall

    # Initialize generator
    generator = CandidateGenerator(test_plan, min_door_spacing=5, seed=42)

    print(f"Floor plan analysis:")
    print(f"  Valid wall positions: {len(generator.valid_wall_positions)}")
    print(f"  Rooms identified: {len(generator.rooms)}")
    print(f"  Room boundaries: {len(generator.room_boundaries)}")
    print(f"  Perimeter positions: {len(generator.perimeter_positions)}")

    # Generate candidates with different strategies
    strategies = ['boundary_focused', 'distributed', 'corner_exits']

    print("\nGeneration strategies:")
    for strategy in strategies:
        candidate = generator.generate_rule_based_candidate(
            num_doors=3,
            num_exits=2,
            strategy=strategy
        )
        print(f"\n  {strategy}: {len(candidate)} doors/exits")
        for door in candidate:
            print(f"    {door}")

    print("\n")


def demo_for_pairwise_labeling():
    """Demo: Generate candidate pairs for pairwise comparison labeling."""
    print("=" * 70)
    print("DEMO 3: Generate Candidates for Pairwise Labeling")
    print("=" * 70)

    # Load floor plan
    with open('../example_configuration.json', 'r') as f:
        config = json.load(f)

    floor_plan = np.array(config['initial_fire_map'], dtype=np.float32)

    # Generate large candidate pool for pairwise labeling
    num_candidates = 100
    candidates = generate_door_candidates(
        floor_plan=floor_plan,
        num_candidates=num_candidates,
        num_doors_range=(2, 5),
        num_exits_range=(1, 3),
        min_door_spacing=5,
        random_ratio=0.5,
        seed=42
    )

    print(f"Generated {len(candidates)} candidates for labeling")

    # Generate candidate pairs for pairwise comparison
    # Strategy: Sample pairs for evaluation
    import random
    random.seed(42)

    num_pairs = 50  # Number of pairs to label
    pairs = []

    for _ in range(num_pairs):
        if len(candidates) >= 2:
            pair = random.sample(candidates, 2)
            pairs.append((pair[0], pair[1]))

    print(f"Generated {len(pairs)} candidate pairs for pairwise labeling")

    # Show first pair
    print(f"\nExample pair:")
    print(f"  Candidate A: {len(pairs[0][0])} doors/exits")
    for door in pairs[0][0]:
        print(f"    {door}")
    print(f"  Candidate B: {len(pairs[0][1])} doors/exits")
    for door in pairs[0][1]:
        print(f"    {door}")

    print("\nThese pairs can now be evaluated with the simulator")
    print("to generate pairwise comparison labels for training.")

    print("\n")


def demo_save_candidates():
    """Demo: Save candidates to JSON for later use."""
    print("=" * 70)
    print("DEMO 4: Save Candidates to JSON")
    print("=" * 70)

    # Load floor plan
    with open('../example_configuration.json', 'r') as f:
        config = json.load(f)

    floor_plan = np.array(config['initial_fire_map'], dtype=np.float32)

    # Generate candidates
    candidates = generate_door_candidates(
        floor_plan=floor_plan,
        num_candidates=20,
        num_doors_range=(2, 4),
        num_exits_range=(1, 2),
        seed=42
    )

    # Save to JSON
    output_file = 'generated_candidates.json'
    with open(output_file, 'w') as f:
        json.dump({
            'floor_plan_shape': list(floor_plan.shape),
            'num_candidates': len(candidates),
            'candidates': candidates
        }, f, indent=2)

    print(f"Saved {len(candidates)} candidates to {output_file}")

    # Load and verify
    with open(output_file, 'r') as f:
        loaded = json.load(f)

    print(f"Verified: Loaded {loaded['num_candidates']} candidates")
    print(f"First candidate: {loaded['candidates'][0]}")

    print("\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CANDIDATE GENERATOR DEMONSTRATION")
    print("Part of Phase 1: AI-Guided Design Optimization")
    print("=" * 70 + "\n")

    demo_basic_usage()
    demo_advanced_usage()
    demo_for_pairwise_labeling()
    demo_save_candidates()

    print("=" * 70)
    print("All demos completed successfully!")
    print("=" * 70)
