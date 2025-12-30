"""
Generate sample floor plans and save as .npz files for visual inspection.
"""

import os
import numpy as np
from floor_plan_generator import FloorPlanGenerator

def save_floor_plans():
    """Generate and save floor plans with different realism ratios"""

    # Create output directory
    output_dir = './sample_floor_plans'
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("Generating Sample Floor Plans")
    print("="*70)

    generator = FloorPlanGenerator(seed=42)

    # Configuration for different types
    configs = [
        {
            'name': 'realistic',
            'realism_ratio': 0.9,
            'count': 5,
            'description': 'Realistic building layouts (office, school, hospital)'
        },
        {
            'name': 'balanced',
            'realism_ratio': 0.6,
            'count': 5,
            'description': 'Balanced mix of realistic and challenging'
        },
        {
            'name': 'challenging',
            'realism_ratio': 0.2,
            'count': 5,
            'description': 'Challenging layouts (complex mazes, irregular)'
        }
    ]

    total_saved = 0

    for config in configs:
        print(f"\n--- Generating {config['name']} plans ({config['description']}) ---")

        # Generate plans
        plans = generator.generate_batch(
            num_plans=config['count'],
            size_range=(35, 50),
            realism_ratio=config['realism_ratio']
        )

        for i, (grid, metadata) in enumerate(plans, 1):
            # Add exits
            num_exits = np.random.randint(2, 5)
            exit_positions = generator.add_exits_to_plan(
                grid, num_exits, placement='distributed'
            )

            # Create filename with descriptive info
            filename = f"{config['name']}_{i:02d}_{metadata.generation_method}_{metadata.size[0]}x{metadata.size[1]}.npz"
            filepath = os.path.join(output_dir, filename)

            # Save as compressed npz
            np.savez_compressed(
                filepath,
                grid=grid,
                size=np.array(metadata.size),
                room_count=metadata.room_count,
                generation_method=metadata.generation_method,
                obstacle_density=metadata.obstacle_density,
                exit_positions=np.array(exit_positions),
                room_centers=np.array(metadata.room_centers) if metadata.room_centers else np.array([])
            )

            # Calculate stats
            passable = np.sum(grid == 0)
            total_cells = grid.size

            print(f"  [{i}] {filename}")
            print(f"      Method: {metadata.generation_method}, "
                  f"Size: {metadata.size[0]}x{metadata.size[1]}, "
                  f"Rooms: {metadata.room_count}, "
                  f"Exits: {len(exit_positions)}")
            print(f"      Passable: {100*passable/total_cells:.1f}%, "
                  f"Obstacles: {metadata.obstacle_density:.1%}")

            total_saved += 1

    print(f"\n{'='*70}")
    print(f"Saved {total_saved} floor plans to: {output_dir}")
    print(f"{'='*70}")

    # Create a simple index file
    index_path = os.path.join(output_dir, 'README.txt')
    with open(index_path, 'w') as f:
        f.write("Floor Plan Samples\n")
        f.write("="*70 + "\n\n")
        f.write("Generated floor plans with different realism ratios:\n\n")
        f.write("realistic_*.npz - Realistic building layouts (offices, schools)\n")
        f.write("                  Good for learning real evacuation patterns\n\n")
        f.write("balanced_*.npz  - Mix of realistic and challenging layouts\n")
        f.write("                  Default training configuration\n\n")
        f.write("challenging_*.npz - Complex mazes and irregular layouts\n")
        f.write("                    Tests edge cases and difficult scenarios\n\n")
        f.write("="*70 + "\n\n")
        f.write("File Format:\n")
        f.write("  Each .npz file contains:\n")
        f.write("    - grid: 2D array (0=passable, -2=wall)\n")
        f.write("    - size: (rows, cols)\n")
        f.write("    - room_count: number of rooms\n")
        f.write("    - generation_method: bsp/grid/template/cellular\n")
        f.write("    - exit_positions: array of (col, row) tuples\n")
        f.write("    - room_centers: array of room center positions\n\n")
        f.write("To visualize, use: python visualize_npz.py <filename>\n")

    print(f"\nCreated index file: {index_path}")
    print("\nYou can now:")
    print("  1. Use npz_visualizer.py to view these files")
    print("  2. Load them with: data = np.load('sample_floor_plans/realistic_01_*.npz')")


if __name__ == '__main__':
    save_floor_plans()
