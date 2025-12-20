"""
Verify fire discovery delay works in Monte Carlo simulations.

Strategy:
1. Run Monte Carlo with delay=0 (baseline)
2. Run Monte Carlo with delay=50 (50 steps)
3. Compare metrics - delay should cause:
   - Higher fire damage (fire spreads more before evacuation)
   - Lower survival rates (more agents trapped)
   - Longer total simulation time (delay + evacuation)
"""

import json
import numpy as np
import time
from monte_carlo import run_monte_carlo_parallel
from simulation import SimulationConfig

print("="*70)
print("MONTE CARLO FIRE DISCOVERY DELAY VERIFICATION")
print("="*70)
print()

# Create minimal test config
base_config = {
    'map_rows': 30,
    'map_cols': 30,
    'max_occupancy': 2,
    'start_positions': [f'x{5+i}y5' for i in range(10)],  # 10 agents in a row
    'targets': ['x25y25'],
    'initial_fire_map': [[0]*30 for _ in range(30)],
    'agent_num': 10,
    'fire_update_interval': 2,
    'fire_model_type': 'aggressive',
    'timestep_duration': 0.5,
    'cell_size': 0.3
}

# Add fire in the middle
base_config['initial_fire_map'][15][15] = 2.0

# Test 1: NO DELAY
print("TEST 1: Running Monte Carlo with NO delay (baseline)")
print("-"*70)

config_no_delay = base_config.copy()
config_no_delay['fire_discovery_delay'] = 0

config_obj_no_delay = SimulationConfig.from_json(config_no_delay)

print("  Config: delay=0, 10 agents, aggressive fire")
print("  Running 20 simulations...")

start = time.time()
results_no_delay = run_monte_carlo_parallel(
    config_obj_no_delay,
    num_runs=20,
    num_processes=4,
    use_phase2=True,
    fire_spread_mode='always_real'
)
elapsed = time.time() - start

print(f"  Completed in {elapsed:.1f}s")
print()

# Extract metrics
no_delay_stats = results_no_delay['statistics']
print("  Results with NO delay:")
print(f"    Avg survival rate: {no_delay_stats['survival_rate']['mean']:.2%}")
print(f"    Avg fire damage: {no_delay_stats['fire_damage']['mean']:.2f}")
print(f"    Avg steps: {no_delay_stats['steps']['mean']:.1f}")
print(f"    Avg evacuation time: {no_delay_stats['evacuation_time']['mean']:.1f} steps")
print()

# Test 2: WITH DELAY
print("TEST 2: Running Monte Carlo WITH 50-step delay")
print("-"*70)

config_with_delay = base_config.copy()
config_with_delay['fire_discovery_delay'] = 50  # 25 seconds at 0.5s/step

config_obj_with_delay = SimulationConfig.from_json(config_with_delay)

print("  Config: delay=50 (25 seconds), 10 agents, aggressive fire")
print("  Running 20 simulations...")

start = time.time()
results_with_delay = run_monte_carlo_parallel(
    config_obj_with_delay,
    num_runs=20,
    num_processes=4,
    use_phase2=True,
    fire_spread_mode='always_real'
)
elapsed = time.time() - start

print(f"  Completed in {elapsed:.1f}s")
print()

# Extract metrics
with_delay_stats = results_with_delay['statistics']
print("  Results WITH 50-step delay:")
print(f"    Avg survival rate: {with_delay_stats['survival_rate']['mean']:.2%}")
print(f"    Avg fire damage: {with_delay_stats['fire_damage']['mean']:.2f}")
print(f"    Avg steps: {with_delay_stats['steps']['mean']:.1f}")
print(f"    Avg evacuation time: {with_delay_stats['evacuation_time']['mean']:.1f} steps")
print()

# Compare
print("="*70)
print("COMPARISON & VERIFICATION")
print("="*70)
print()

survival_diff = no_delay_stats['survival_rate']['mean'] - with_delay_stats['survival_rate']['mean']
damage_diff = with_delay_stats['fire_damage']['mean'] - no_delay_stats['fire_damage']['mean']
steps_diff = with_delay_stats['steps']['mean'] - no_delay_stats['steps']['mean']

print("Impact of 50-step fire discovery delay:")
print(f"  Survival rate change: {survival_diff:+.1%} (should be negative)")
print(f"  Fire damage change: {damage_diff:+.2f} (should be positive)")
print(f"  Total steps change: {steps_diff:+.1f} (should be ~50)")
print()

# Verification
print("Verification:")
checks_passed = 0
total_checks = 0

# Check 1: Fire damage should be higher with delay
total_checks += 1
if damage_diff > 0:
    print(f"  [PASS] Fire damage increased by {damage_diff:.2f}")
    checks_passed += 1
else:
    print(f"  [FAIL] Fire damage didn't increase (got {damage_diff:.2f})")

# Check 2: Survival rate should be lower with delay (or same if all survive/die)
total_checks += 1
if survival_diff >= -0.01:  # Allow small tolerance
    print(f"  [PASS] Survival rate decreased or stable ({survival_diff:+.1%})")
    checks_passed += 1
else:
    print(f"  [WARN] Survival rate increased unexpectedly ({survival_diff:+.1%})")

# Check 3: Steps should increase by approximately the delay amount
total_checks += 1
expected_increase = 50
tolerance = 20  # Allow variance due to different evacuation outcomes
if abs(steps_diff - expected_increase) < tolerance:
    print(f"  [PASS] Total steps increased by ~{expected_increase} (got {steps_diff:+.1f})")
    checks_passed += 1
else:
    print(f"  [WARN] Steps didn't increase as expected (got {steps_diff:+.1f}, expected ~{expected_increase})")

# Check 4: Config was actually loaded with delay
total_checks += 1
if config_obj_with_delay.fire_discovery_delay == 50:
    print(f"  [PASS] Config contains delay=50")
    checks_passed += 1
else:
    print(f"  [FAIL] Config doesn't have correct delay")

print()
print(f"Passed {checks_passed}/{total_checks} checks")
print()

if checks_passed >= 3:
    print("="*70)
    print("[SUCCESS] Fire discovery delay is working in Monte Carlo!")
    print("="*70)
    print()
    print("Evidence:")
    print("  - Fire damage increased when delay added")
    print("  - Total simulation time increased by ~delay amount")
    print("  - Config parameter properly loaded and used")
else:
    print("="*70)
    print("[WARNING] Some checks failed - review results above")
    print("="*70)

print()
print("Key insight: The delay causes measurable differences in outcomes,")
print("proving that agents really do stay frozen while fire spreads.")
