"""
Verify floor plan generation quality - check room counts and diversity
"""

from floor_plan_generator import FloorPlanGenerator

def verify_quality():
    print("="*70)
    print("Floor Plan Quality Verification")
    print("="*70)

    generator = FloorPlanGenerator(seed=999)

    # Generate 50 plans with balanced settings
    plans = generator.generate_batch(
        num_plans=50,
        size_range=(35, 50),
        realism_ratio=0.6
    )

    print(f"\nGenerated {len(plans)} valid floor plans")
    print("\n" + "="*70)
    print("Room Count Distribution")
    print("="*70)

    # Analyze room counts
    room_counts = {}
    for _, meta in plans:
        count = meta.room_count
        room_counts[count] = room_counts.get(count, 0) + 1

    for count in sorted(room_counts.keys()):
        freq = room_counts[count]
        pct = 100 * freq / len(plans)
        bar = "#" * int(pct / 2)
        print(f"  {count:2d} rooms: {freq:2d} plans ({pct:5.1f}%) {bar}")

    # Single-room analysis
    single_room = room_counts.get(1, 0)
    print(f"\n  Single-room plans: {single_room}/{len(plans)} ({100*single_room/len(plans):.1f}%)")
    print(f"  Multi-room plans: {len(plans)-single_room}/{len(plans)} ({100*(len(plans)-single_room)/len(plans):.1f}%)")

    # Method distribution
    print("\n" + "="*70)
    print("Method Distribution")
    print("="*70)

    method_counts = {}
    for _, meta in plans:
        method = meta.generation_method
        method_counts[method] = method_counts.get(method, 0) + 1

    for method in sorted(method_counts.keys()):
        count = method_counts[method]
        pct = 100 * count / len(plans)
        bar = "#" * int(pct / 2)
        print(f"  {method:12s}: {count:2d} plans ({pct:5.1f}%) {bar}")

    # Obstacle density analysis
    print("\n" + "="*70)
    print("Obstacle Density Statistics")
    print("="*70)

    densities = [meta.obstacle_density for _, meta in plans]
    avg_density = sum(densities) / len(densities)
    min_density = min(densities)
    max_density = max(densities)

    print(f"  Average: {avg_density:.1%}")
    print(f"  Range: {min_density:.1%} - {max_density:.1%}")

    # Passable area analysis
    print("\n" + "="*70)
    print("Passable Area Statistics")
    print("="*70)

    import numpy as np
    passable_pcts = []
    for grid, meta in plans:
        passable = np.sum(grid == 0)
        total = grid.size
        passable_pcts.append(100 * passable / total)

    avg_passable = sum(passable_pcts) / len(passable_pcts)
    min_passable = min(passable_pcts)
    max_passable = max(passable_pcts)

    print(f"  Average: {avg_passable:.1f}%")
    print(f"  Range: {min_passable:.1f}% - {max_passable:.1f}%")

    print("\n" + "="*70)
    print("Quality Assessment")
    print("="*70)

    # Quality checks
    checks = {
        "Single-room plans < 10%": single_room / len(plans) < 0.10,
        "Multi-room plans > 90%": (len(plans) - single_room) / len(plans) > 0.90,
        "Cellular < 10%": method_counts.get('cellular', 0) / len(plans) < 0.10,
        "Template 35-60%": 0.35 <= method_counts.get('template', 0) / len(plans) <= 0.60,
        "BSP 25-55%": 0.25 <= method_counts.get('bsp', 0) / len(plans) <= 0.55,
        "Passable area 50-80%": 50 <= avg_passable <= 80,
    }

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    all_passed = all(checks.values())
    print(f"\n{'All checks passed!' if all_passed else 'Some checks failed'}")


if __name__ == '__main__':
    verify_quality()
