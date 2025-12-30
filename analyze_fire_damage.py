"""
Analyze Per-Agent Fire Damage from Monte Carlo Results
=======================================================

Analyzes individual agent fire exposures from Phase 2 simulations.
"""

import json
import numpy as np
import sys
from pathlib import Path

def analyze_fire_damage(results_dir: str):
    """
    Analyze per-agent fire damage from Monte Carlo results.

    Args:
        results_dir: Path to monte_carlo_results directory
    """
    stats_file = Path(results_dir) / "statistics.json"

    if not stats_file.exists():
        print(f"Error: {stats_file} not found")
        return

    with open(stats_file) as f:
        stats = json.load(f)

    print("\n" + "="*70)
    print("FIRE DAMAGE ANALYSIS")
    print("="*70)

    # Check if we have agent fire exposures data
    has_agent_data = False
    all_exposures = []

    # Try to load from full results if available
    full_results_file = Path(results_dir) / "full_results.json"
    if full_results_file.exists():
        with open(full_results_file) as f:
            full_data = json.load(f)

        if 'individual_runs' in full_data:
            for run in full_data['individual_runs']:
                # Check both Phase 2 and original keys
                exposures_key = None
                if '_phase2_agent_fire_exposures' in run:
                    exposures_key = '_phase2_agent_fire_exposures'
                elif 'agent_fire_exposures' in run:
                    exposures_key = 'agent_fire_exposures'

                if exposures_key:
                    all_exposures.extend(run[exposures_key])
                    has_agent_data = True

    if not has_agent_data:
        print("\nNo per-agent fire damage data found.")
        print("This might be because:")
        print("  1. Used --no-full-results flag (per-agent data not saved)")
        print("  2. Using original simulation (not Phase 2)")
        print("\nTo get per-agent data, run with Phase 2 and without --no-full-results:")
        print("  python monte_carlo.py --runs 100 --phase2 --parallel")
        print("\nShowing aggregate statistics only:")
    else:
        print(f"\nTotal agents analyzed: {len(all_exposures)}")

    # Show aggregate statistics
    print(f"\nAggregate Statistics:")
    print(f"  Average fire damage: {stats.get('average_fire_damage', 0):.4f}")

    if has_agent_data and all_exposures:
        exposures = np.array(all_exposures)

        print(f"\nPer-Agent Fire Damage Distribution:")
        print(f"  Mean:   {np.mean(exposures):.4f}")
        print(f"  Median: {np.median(exposures):.4f}")
        print(f"  Std:    {np.std(exposures):.4f}")
        print(f"  Min:    {np.min(exposures):.4f}")
        print(f"  Max:    {np.max(exposures):.4f}")

        # Percentiles
        print(f"\nPercentiles:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  {p}th: {np.percentile(exposures, p):.4f}")

        # Histogram
        print(f"\nFire Damage Histogram:")
        hist, bins = np.histogram(exposures, bins=10)
        for i in range(len(hist)):
            bar_length = int(hist[i] / max(hist) * 40)
            print(f"  {bins[i]:6.2f} - {bins[i+1]:6.2f}: {'█' * bar_length} ({hist[i]})")

        # Categories
        no_damage = np.sum(exposures == 0)
        low_damage = np.sum((exposures > 0) & (exposures < 1))
        med_damage = np.sum((exposures >= 1) & (exposures < 5))
        high_damage = np.sum(exposures >= 5)

        print(f"\nFire Damage Categories:")
        print(f"  No damage (0.0):      {no_damage:5d} ({no_damage/len(exposures)*100:5.1f}%)")
        print(f"  Low damage (0-1):     {low_damage:5d} ({low_damage/len(exposures)*100:5.1f}%)")
        print(f"  Medium damage (1-5):  {med_damage:5d} ({med_damage/len(exposures)*100:5.1f}%)")
        print(f"  High damage (5+):     {high_damage:5d} ({high_damage/len(exposures)*100:5.1f}%)")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_fire_damage.py <results_directory>")
        print("\nExample:")
        print("  python analyze_fire_damage.py ./monte_carlo_results/lowd_normal_20251218_082400")
        sys.exit(1)

    analyze_fire_damage(sys.argv[1])
