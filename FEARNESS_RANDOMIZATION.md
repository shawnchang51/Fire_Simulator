# Random Fearness Assignment

## Feature Overview

The Monte Carlo queue system now **automatically assigns random fearness values** to each agent in every simulation run!

## How It Works

The system uses the **first two values** in `agent_fearness` to define a range:

```json
{
  "agent_fearness": [0.5, 1.5]
}
```

Each agent gets a **random fearness between 0.5 and 1.5** (uniform distribution).

## Configuration Examples

### Example 1: Random Range
```json
{
  "agent_fearness": [0.5, 2.0],
  "agent_num": 20
}
```
**Result**: Each of the 20 agents gets a random fearness between 0.5 and 2.0

### Example 2: Order Doesn't Matter
```json
{
  "agent_fearness": [2.0, 0.5],
  "agent_num": 20
}
```
**Result**: Same as above! Min/max are automatically determined.

### Example 3: Single Value (No Randomization)
```json
{
  "agent_fearness": [1.0],
  "agent_num": 20
}
```
**Result**: All 20 agents get fearness = 1.0 (no randomization)

### Example 4: No Value (Use Default)
```json
{
  "agent_num": 20
}
```
**Result**: All 20 agents get default fearness = 1.0

## How Fearness Affects Behavior

**Fearness** is a multiplier on how much agents avoid fire:

| Fearness | Behavior |
|----------|----------|
| 0.5 | **Brave** - Takes more risks, tolerates more fire |
| 1.0 | **Normal** - Default behavior |
| 1.5 | **Cautious** - Avoids fire more actively |
| 2.0 | **Very Cautious** - Strongly avoids any fire |

Higher fearness = Higher cost for fire cells in pathfinding.

## Example Scenarios

### Scenario 1: Mixed Population
```json
{
  "agent_fearness": [0.3, 2.0],
  "agent_num": 50
}
```
**Use case**: Simulate a realistic population with varying risk tolerance
- Some agents very brave (0.3-0.6)
- Some agents very cautious (1.5-2.0)
- Most agents somewhere in between

### Scenario 2: Brave Crowd
```json
{
  "agent_fearness": [0.2, 0.8],
  "agent_num": 30
}
```
**Use case**: Study how brave agents behave (e.g., trained firefighters, military)

### Scenario 3: Cautious Crowd
```json
{
  "agent_fearness": [1.5, 2.5],
  "agent_num": 30
}
```
**Use case**: Study how cautious agents behave (e.g., children, elderly)

## Monte Carlo Benefits

With randomization, each Monte Carlo run has:
- ✅ **Different agent positions** (from `replace_agents()`)
- ✅ **Different fire positions** (from `replace_fire()`)
- ✅ **Different agent fearness** (NEW!)

This creates **truly diverse scenarios** for statistical analysis!

## Viewing Results

The fearness values used in each run are saved in `full_results.json`:

```python
import json

with open('monte_carlo_results/scenario_20250123_143052_001/full_results.json') as f:
    data = json.load(f)

# Access fearness values for a specific run
run_0_fearness = data['configuration']['agent_fearness']
print(f"Fearness values in run 0: {run_0_fearness}")
```

## Statistical Analysis Example

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('monte_carlo_results/scenario_20250123_143052_001/full_results.json') as f:
    data = json.load(f)

# Extract fearness and success for each run
fearness_avg = []
success_rate = []

for run in data['individual_runs']:
    config = run.get('configuration', data['configuration'])
    avg_fear = np.mean(config['agent_fearness'])
    fearness_avg.append(avg_fear)

    success = run['evacuated_agents'] / config['agent_num']
    success_rate.append(success)

# Plot correlation
plt.scatter(fearness_avg, success_rate, alpha=0.5)
plt.xlabel('Average Fearness')
plt.ylabel('Success Rate')
plt.title('Impact of Fearness on Evacuation Success')
plt.show()

# Calculate correlation
correlation = np.corrcoef(fearness_avg, success_rate)[0,1]
print(f"Correlation: {correlation:.3f}")
```

## Tips

1. **Wide Range**: Use `[0.3, 2.0]` for maximum diversity
2. **Narrow Range**: Use `[0.9, 1.1]` for slight variations around default
3. **Single Value**: Use `[1.0]` to disable randomization
4. **Research**: Compare results with different fearness ranges to study impact

## Technical Details

- **Distribution**: Uniform random between min and max
- **When Applied**: Every time `replace_agents()` is called (once per simulation run)
- **Independence**: Each agent's fearness is independently sampled
- **Reproducibility**: Controlled by random seed (same seed = same fearness values)

---

**Happy simulating with diverse agent personalities! 🎉**
