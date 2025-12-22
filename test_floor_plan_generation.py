"""
Test script to demonstrate floor plan generation as used in training data pipeline.
Mimics how generate_training_data.py generates floor plans.
"""

import numpy as np
from floor_plan_generator import FloorPlanGenerator

def visualize_plan(grid, metadata, exit_positions):
    """Create ASCII visualization of floor plan"""
    rows, cols = grid.shape

    # Create visualization grid
    vis = []
    for row in grid:
        vis_row = []
        for cell in row:
            if cell == -2:
                vis_row.append('#')
            else:
                vis_row.append('.')
        vis.append(vis_row)

    # Mark exits with 'E'
    for x, y in exit_positions:
        if 0 <= y < rows and 0 <= x < cols:
            vis[y][x] = 'E'

    return vis


def test_generation_pipeline():
    """Test floor plan generation as done in training data pipeline"""
    print("="*70)
    print("Floor Plan Generation Test - Training Data Pipeline Style")
    print("="*70)

    # Initialize generator with seed (like generate_training_data.py does)
    seed = 42
    generator = FloorPlanGenerator(seed=seed)

    # Test different realism ratios
    test_configs = [
        {'name': 'High Realism (80%)', 'realism_ratio': 0.8, 'count': 3},
        {'name': 'Balanced (60%)', 'realism_ratio': 0.6, 'count': 3},
        {'name': 'High Challenge (20%)', 'realism_ratio': 0.2, 'count': 3},
    ]

    for config in test_configs:
        print(f"\n{'='*70}")
        print(f"{config['name']} - realism_ratio={config['realism_ratio']}")
        print('='*70)

        # Generate plans (mimics generate_training_data.py lines 369-385)
        plans = generator.generate_batch(
            num_plans=config['count'],
            size_range=(30, 50),
            realism_ratio=config['realism_ratio']
        )

        for i, (grid, metadata) in enumerate(plans, 1):
            # Add exits (like training pipeline does)
            num_exits = np.random.randint(1, 5)  # 1-4 exits
            exit_positions = generator.add_exits_to_plan(
                grid, num_exits, placement='distributed'
            )

            # Store exit positions in metadata (like pipeline does)
            metadata.exit_positions = exit_positions

            # Print plan info
            print(f"\n--- Plan {i}: {metadata.generation_method} ---")
            print(f"Size: {metadata.size[0]}x{metadata.size[1]} ({metadata.size[0] * metadata.size[1]} cells)")
            print(f"Rooms: {metadata.room_count}")
            print(f"Exits: {len(exit_positions)}")
            print(f"Obstacle density: {metadata.obstacle_density:.1%}")

            # Calculate additional statistics
            passable = np.sum(grid == 0)
            total = grid.size
            print(f"Passable area: {passable}/{total} cells ({100*passable/total:.1f}%)")

            # Show visualization (first 15x30 or full if smaller)
            vis = visualize_plan(grid, metadata, exit_positions)
            display_rows = min(15, len(vis))
            display_cols = min(35, len(vis[0]))

            print(f"\nVisualization ({display_rows}x{display_cols}):")
            print("  (# = wall, . = passable, E = exit)")
            for row in vis[:display_rows]:
                print("  " + "".join(row[:display_cols]))

            if len(vis) > display_rows or len(vis[0]) > display_cols:
                print("  ... (truncated)")


def test_method_distribution():
    """Test that method distribution follows realism_ratio"""
    print("\n" + "="*70)
    print("Method Distribution Analysis")
    print("="*70)

    generator = FloorPlanGenerator(seed=999)

    test_ratios = [0.2, 0.6, 0.8]

    for ratio in test_ratios:
        print(f"\n--- realism_ratio = {ratio:.1f} ---")
        plans = generator.generate_batch(
            num_plans=50,
            size_range=(30, 50),
            realism_ratio=ratio
        )

        method_counts = {}
        for _, meta in plans:
            method = meta.generation_method
            method_counts[method] = method_counts.get(method, 0) + 1

        print(f"Generated {len(plans)} plans:")
        for method in sorted(method_counts.keys()):
            count = method_counts[method]
            pct = 100 * count / len(plans)
            print(f"  {method:12s}: {count:2d} plans ({pct:5.1f}%)")

        # Show realistic vs challenging split
        realistic_methods = {'template'}  # template has realistic layouts
        realistic_count = sum(method_counts.get(m, 0) for m in realistic_methods)
        print(f"\n  Realistic templates: {realistic_count}/{len(plans)} ({100*realistic_count/len(plans):.1f}%)")


def test_specific_templates():
    """Test specific realistic templates that training would use"""
    print("\n" + "="*70)
    print("Specific Template Examples (for training realism)")
    print("="*70)

    generator = FloorPlanGenerator(seed=123)

    # Generate with high realism to get template examples
    plans = generator.generate_batch(
        num_plans=10,
        size_range=(40, 45),
        realism_ratio=0.9  # High probability of templates
    )

    template_plans = [(g, m) for g, m in plans if m.generation_method == 'template']

    print(f"\nFound {len(template_plans)} template-based plans:")

    for i, (grid, meta) in enumerate(template_plans[:3], 1):  # Show first 3
        num_exits = 2
        exit_positions = generator.add_exits_to_plan(grid, num_exits, 'distributed')

        print(f"\n--- Template Plan {i} ---")
        print(f"Size: {meta.size[0]}x{meta.size[1]}")
        print(f"Rooms: {meta.room_count}")
        print(f"Exits: {len(exit_positions)}")

        vis = visualize_plan(grid, meta, exit_positions)
        display_rows = min(12, len(vis))
        display_cols = min(30, len(vis[0]))

        print(f"\nLayout preview:")
        for row in vis[:display_rows]:
            print("  " + "".join(row[:display_cols]))


if __name__ == '__main__':
    # Run tests
    test_generation_pipeline()
    test_method_distribution()
    test_specific_templates()

    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)
