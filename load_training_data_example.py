"""
Example: How to load complete training data for your AI model

This shows how to load floor plan grids + door/exit configs + scenarios together.
"""

import json
import numpy as np
from resplan_loader import ResPlanLoader

def load_complete_training_data(pairs_file: str, resplan_pkl: str):
    """
    Load training pairs with their corresponding floor plan grids.

    Args:
        pairs_file: Path to train_pairs.jsonl/val_pairs.jsonl/test_pairs.jsonl
        resplan_pkl: Path to ResPlan.pkl

    Returns:
        List of complete training examples with floor plan grids
    """
    # 1. Load the floor plans from ResPlan.pkl
    print("Loading floor plans from ResPlan.pkl...")
    loader = ResPlanLoader(resplan_pkl, cell_size_m=0.3)
    all_floor_plans = loader.convert_all(min_doors=1)

    # Create a lookup dict: plan_index -> ResPlanFloorPlan
    floor_plan_lookup = {fp.plan_index: fp for fp in all_floor_plans}
    print(f"Loaded {len(floor_plan_lookup)} floor plans")

    # 2. Load the training pairs
    print(f"\nLoading training pairs from {pairs_file}...")
    training_examples = []

    with open(pairs_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            pair = json.loads(line)

            # Get floor plan IDs
            fp_id_a = pair['floor_plan_id_a']
            fp_id_b = pair['floor_plan_id_b']

            # Get the actual floor plan grids
            floor_plan_a = floor_plan_lookup[fp_id_a]
            floor_plan_b = floor_plan_lookup[fp_id_b]

            # Combine everything into a complete training example
            example = {
                # Floor plan A (grid + layout)
                'floor_plan_grid_a': floor_plan_a.grid,  # np.ndarray shape (H, W)
                'floor_plan_id_a': fp_id_a,

                # Configuration A (exits + doors)
                'exits_a': [d for d in pair['config_a']['door_config'] if d['type'] == 'exit'],
                'doors_a': [d for d in pair['config_a']['door_config'] if d['type'] == 'door'],

                # Scenario A
                'scenario_a': pair['scenario_a'],

                # Floor plan B (grid + layout)
                'floor_plan_grid_b': floor_plan_b.grid,  # np.ndarray shape (H, W)
                'floor_plan_id_b': fp_id_b,

                # Configuration B (exits + doors)
                'exits_b': [d for d in pair['config_b']['door_config'] if d['type'] == 'exit'],
                'doors_b': [d for d in pair['config_b']['door_config'] if d['type'] == 'door'],

                # Scenario B
                'scenario_b': pair['scenario_b'],

                # Label (which config is better)
                'label': pair['label'],  # 0 = config_a worse, 1 = config_a better
                'pair_type': pair['pair_type']
            }

            training_examples.append(example)

            if line_num % 100 == 0:
                print(f"  Loaded {line_num} pairs...")

    print(f"\nTotal: {len(training_examples)} complete training examples")
    return training_examples


def example_usage():
    """Example of how to use the loaded data"""

    # Load training data
    train_data = load_complete_training_data(
        pairs_file='test_output/train_pairs.jsonl',
        resplan_pkl='./ResPlan/ResPlan.pkl'
    )

    # Show what you get for the first example
    example = train_data[0]

    print("\n" + "="*60)
    print("EXAMPLE TRAINING SAMPLE:")
    print("="*60)

    print(f"\nFloor Plan A:")
    print(f"  Grid shape: {example['floor_plan_grid_a'].shape}")
    print(f"  Exits: {len(example['exits_a'])} - {example['exits_a']}")
    print(f"  Doors: {len(example['doors_a'])} - {example['doors_a']}")
    print(f"  Scenario: {example['scenario_a']}")

    print(f"\nFloor Plan B:")
    print(f"  Grid shape: {example['floor_plan_grid_b'].shape}")
    print(f"  Exits: {len(example['exits_b'])} - {example['exits_b']}")
    print(f"  Doors: {len(example['doors_b'])} - {example['doors_b']}")
    print(f"  Scenario: {example['scenario_b']}")

    print(f"\nLabel: {example['label']} (0=A worse, 1=A better)")
    print(f"Pair type: {example['pair_type']}")

    print("\n" + "="*60)
    print("YOUR MODEL INPUTS:")
    print("="*60)
    print("For each configuration A and B, you have:")
    print("  1. Floor plan grid (numpy array)")
    print("  2. Exit positions (list of {id, position, type})")
    print("  3. Door positions (list of {id, position, type})")
    print("  4. Scenario parameters (agent_count, fires, etc.)")
    print("\nYour model should predict which config is better!")


if __name__ == '__main__':
    example_usage()
