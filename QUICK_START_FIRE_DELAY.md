# Quick Start: Fire Discovery Delay

## TL;DR

Add this to your config JSON to simulate fire spreading for 10 seconds before agents start evacuating:

```json
{
  "fire_discovery_delay": 20,
  ...
}
```

That's it! Works everywhere (simulation.py, fast_simulation.py, monte_carlo.py, run_phase2_visual.py).

## Parameter Reference

```json
{
  "fire_discovery_delay": <number>,  // Steps where fire spreads but agents don't move
  "fire_update_interval": 4,          // Fire updates every 4 steps (default)
  "timestep_duration": 0.5            // Each step = 0.5 seconds (default)
}
```

**Calculation:**
```
Real time delay = fire_discovery_delay × timestep_duration

Examples:
- 20 steps × 0.5s = 10 seconds
- 40 steps × 0.5s = 20 seconds
- 120 steps × 0.5s = 60 seconds (1 minute)
```

## Common Values

| Scenario | `fire_discovery_delay` | Real Time | Use Case |
|----------|----------------------|-----------|----------|
| No delay (default) | 0 | 0s | Baseline comparison |
| Smoke detector | 20 | 10s | Typical detector response |
| Commercial building | 60 | 30s | Detector + alarm notification |
| Hidden fire | 240 | 120s | Electrical fire in walls |
| Stress test | 600 | 300s | Extreme worst-case |

## Quick Test

```bash
# Run visual test
python run_phase2_visual.py --config example_fire_discovery_delay.json

# Run Monte Carlo comparison
python monte_carlo.py --config example_fire_discovery_delay.json --runs 100 --parallel --phase2
```

## Example Configurations

### Immediate Response (Baseline)
```json
{
  "fire_discovery_delay": 0
}
```

### Realistic Office Building
```json
{
  "fire_discovery_delay": 40,
  "fire_update_interval": 4,
  "timestep_duration": 0.5,
  "fire_model_type": "realistic"
}
```
20-second detection delay with realistic fire physics

### Worst-Case Scenario
```json
{
  "fire_discovery_delay": 120,
  "fire_model_type": "aggressive",
  "fire_spread_mode": "always_real"
}
```
1-minute hidden fire with aggressive spread

## Verification

Run tests to verify it works:
```bash
python test_fire_discovery_delay.py
```

Should output:
```
ALL TESTS PASSED [SUCCESS]
```

## See Also

- **`FIRE_DISCOVERY_DELAY.md`** - Full documentation with implementation details
- **`IMPLEMENTATION_SUMMARY.md`** - Technical implementation overview
- **`example_fire_discovery_delay.json`** - Working example configuration
