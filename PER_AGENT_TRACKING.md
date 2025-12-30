# Per-Agent Fire Damage Tracking

## ✅ Yes, Phase 2 Tracks Every Agent's Fire Damage!

### Memory Cost: **Negligible**

| Scenario | Memory Usage |
|----------|--------------|
| 100 runs × 200 agents | 156 KB |
| 500 runs × 200 agents | 781 KB |
| 1,000 runs × 200 agents | **1.5 MB** |
| 5,000 runs × 200 agents | 7.6 MB |

**Verdict:** It's so cheap we just keep it by default! 🎉

### What's Tracked Per Agent

Each agent gets:
- **Fire damage** (accumulated fire exposure over time)
- **Final status** (evacuated/stuck/dead)
- **Steps taken**

### How to Access Per-Agent Data

#### Method 1: Run with Full Results (Recommended)

```bash
# Saves per-agent data to full_results.json
python monte_carlo.py --runs 100 --config ./configs/lowd_normal.json --phase2 --parallel

# Analyze it
python analyze_fire_damage.py ./monte_carlo_results/lowd_normal_TIMESTAMP
```

#### Method 2: Even Works with --no-full-results!

```bash
# Still saves agent_fire_exposures (only ~1.5MB for 1000 runs)
python monte_carlo.py --runs 1000 --config ./configs/lowd_normal.json --phase2 --parallel --no-full-results
```

The `agent_fire_exposures` array is included even in minimal results because it's so small!

## 📊 Example Analysis Output

```
======================================================================
FIRE DAMAGE ANALYSIS
======================================================================

Total agents analyzed: 400

Aggregate Statistics:
  Average fire damage: 0.0100

Per-Agent Fire Damage Distribution:
  Mean:   0.0100
  Median: 0.0000
  Std:    0.1411
  Min:    0.0000
  Max:    2.0000

Percentiles:
  10th: 0.0000
  25th: 0.0000
  50th: 0.0000
  75th: 0.0000
  90th: 0.0000
  95th: 0.0000
  99th: 0.0000

Fire Damage Histogram:
    0.00 -   0.20: ████████████████████████████████████████ (398)
    0.20 -   0.40:  (0)
    0.40 -   0.60:  (0)
    0.60 -   0.80:  (0)
    0.80 -   1.00:  (0)
    1.00 -   1.20:  (0)
    1.20 -   1.40:  (0)
    1.40 -   1.60:  (0)
    1.60 -   1.80:  (0)
    1.80 -   2.00:  (2)

Fire Damage Categories:
  No damage (0.0):        398 ( 99.5%)
  Low damage (0-1):         0 (  0.0%)
  Medium damage (1-5):      2 (  0.5%)
  High damage (5+):         0 (  0.0%)
======================================================================
```

## 🔍 Accessing Data Programmatically

```python
import json

# Load results
with open('./monte_carlo_results/lowd_normal_TIMESTAMP/full_results.json') as f:
    data = json.load(f)

# Get per-agent fire exposures from each run
for run in data['individual_runs']:
    exposures = run['_phase2_agent_fire_exposures']
    print(f"Run {run['run_number']}: {len(exposures)} agents")
    print(f"  Mean damage: {sum(exposures)/len(exposures):.4f}")
    print(f"  Max damage: {max(exposures):.4f}")

    # Find agents with high exposure
    for i, damage in enumerate(exposures):
        if damage > 5.0:
            print(f"  Agent {i}: HIGH DAMAGE {damage:.2f}")
```

## 📈 Use Cases

### 1. Identify High-Risk Scenarios
Find which floor plans expose agents to most fire:
```bash
python monte_carlo.py --runs 100 --phase2 --parallel --config risky_layout.json
python analyze_fire_damage.py ./monte_carlo_results/risky_layout_TIMESTAMP
```

### 2. Compare Floor Plan Safety
```bash
# Test design A
python monte_carlo.py --runs 100 --phase2 --parallel --config design_a.json

# Test design B
python monte_carlo.py --runs 100 --phase2 --parallel --config design_b.json

# Compare fire damage distributions
python analyze_fire_damage.py ./monte_carlo_results/design_a_TIMESTAMP
python analyze_fire_damage.py ./monte_carlo_results/design_b_TIMESTAMP
```

### 3. Statistical Analysis
Export to CSV for further analysis:
```python
import json
import csv

with open('./monte_carlo_results/lowd_normal_TIMESTAMP/full_results.json') as f:
    data = json.load(f)

# Export all agent exposures
with open('agent_fire_damage.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['run', 'agent', 'fire_damage', 'evacuated'])

    for run in data['individual_runs']:
        run_num = run['run_number']
        exposures = run['_phase2_agent_fire_exposures']
        for agent_id, damage in enumerate(exposures):
            writer.writerow([run_num, agent_id, damage, 'yes' if agent_id < run['evacuated_agents'] else 'no'])
```

## ⚙️ Technical Details

### How Fire Damage Accumulates

```python
# Each timestep, for each agent:
fire_intensity = grid[agent.y, agent.x]
if fire_intensity > 0:
    agent.fire_damage += fire_intensity

# Fire intensity scale:
# 0.0 = No fire
# 1.0 = Ignition
# 2.0 = Growing fire
# 3.0 = Fully developed
# 4.0 = Flashover
```

### Example Scenarios

**No exposure:**
- Agent takes safe route
- Fire damage = 0.0

**Brief exposure:**
- Agent crosses edge of fire (intensity 1.0) for 2 steps
- Fire damage = 1.0 × 2 = 2.0

**Extended exposure:**
- Agent trapped near fire (intensity 2.0) for 10 steps
- Fire damage = 2.0 × 10 = 20.0

**Death:**
- Agent enters intense fire (intensity > 3.0)
- Status = 'dead'
- Final fire damage recorded

## 🎯 Summary

**Question:** Can it keep an eye on the fire damage of every agent?
**Answer:** ✅ **YES!** Already implemented and costs almost nothing.

**Memory cost:** 1.5 MB per 1000 runs (negligible)
**Included in:** Both full results AND minimal results (--no-full-results)
**Access:** Use `analyze_fire_damage.py` or parse JSON directly

You get complete per-agent fire damage tracking for free! 🎉
