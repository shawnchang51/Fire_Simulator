"""Quick example showing what data you have for training"""

import json

# Load one training pair
with open('test_output/train_pairs.jsonl', 'r') as f:
    example = json.loads(f.readline())

print("="*70)
print("TRAINING DATA FORMAT")
print("="*70)

print("\n[CONFIGURATION A]")
print(f"Floor Plan ID: {example['floor_plan_id_a']}")
print(f"\nDoor Config (exits + doors combined):")
for item in example['config_a']['door_config']:
    print(f"  {item['type']:5s} - {item['id']:3s} at {item['position']}")

print(f"\nScenario A:")
for k, v in example['scenario_a'].items():
    print(f"  {k}: {v}")

print("\n[CONFIGURATION B]")
print(f"Floor Plan ID: {example['floor_plan_id_b']}")
print(f"\nDoor Config (exits + doors combined):")
for item in example['config_b']['door_config']:
    print(f"  {item['type']:5s} - {item['id']:3s} at {item['position']}")

print(f"\nScenario B:")
for k, v in example['scenario_b'].items():
    print(f"  {k}: {v}")

print("\n[LABEL]")
print(f"Label: {example['label']} (0 = A is worse, 1 = A is better)")
print(f"Pair type: {example['pair_type']}")
print(f"Confidence: {example['label_confidence']:.3f}")

print("\n" + "="*70)
print("TO GET FLOOR PLAN GRIDS:")
print("="*70)
print("""
Use ResPlanLoader to load the actual floor plan grids:

    from resplan_loader import ResPlanLoader

    loader = ResPlanLoader('./ResPlan/ResPlan.pkl', cell_size_m=0.3)
    floor_plans = loader.convert_all(min_doors=1)

    # Get grid for floor_plan_id
    floor_plan = [fp for fp in floor_plans if fp.plan_index == floor_plan_id][0]
    grid = floor_plan.grid  # numpy array (H, W)

Then combine the grid with the door_config and scenario to feed your model!
""")

print("\n" + "="*70)
print("YOUR MODEL SHOULD COMPARE:")
print("="*70)
print("""
Input A: (floor_plan_grid, exits, doors, scenario)
Input B: (floor_plan_grid, exits, doors, scenario)

Output: Which configuration is better? (0 or 1)
""")
