# Monte Carlo Queue - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Prepare Your Configs

Create a folder and add your JSON configuration files:

```bash
mkdir my_configs
cp example_configuration.json my_configs/scenario1.json
cp example_configuration.json my_configs/scenario2.json
# Edit the files to create different scenarios
```

### Step 2: Run the Queue

**Windows:**
```bash
python monte_carlo_queue.py --queue-folder my_configs --runs 100 --parallel
```

**Or use the provided script:**
```bash
# Edit run_queue_example.bat first to set your parameters
run_queue_example.bat
```

**Linux/Mac:**
```bash
python monte_carlo_queue.py --queue-folder my_configs --runs 100 --parallel

# Or use the provided script:
chmod +x run_queue_example.sh
./run_queue_example.sh
```

### Step 3: Check Your Results

```bash
cd monte_carlo_results
ls -la
```

Each config will have a folder like:
```
scenario1_20250123_143052_001/
├── full_results.json    # Complete data - EVERY detail
├── statistics.json      # Just the numbers
├── config_used.json     # Config that was used
└── summary.txt          # Human-readable report
```

**Note**: The `_001`, `_002`, `_003` suffix ensures:
- ✅ No timestamp collisions (even if processed in same second)
- ✅ Alphabetical order = processing order
- ✅ Easy to identify which was processed when

## 📊 What You Get

### Every Simulation Run Includes:
- ✅ Complete agent trajectories (every position, every timestep)
- ✅ Fire spread history (complete evolution)
- ✅ Individual agent statistics (fire damage, temperature exposure, path taken)
- ✅ Evacuation times and outcomes
- ✅ Path frequency analysis
- ✅ Configuration snapshot

### Randomization Per Run:
- 🎲 **Random agent positions** - Different start positions each run
- 🔥 **Random fire positions** - Different fire locations each run
- 😨 **Random agent fearness** - Each agent gets random fear value between first two values in `agent_fearness`

### Aggregated Statistics:
- 📈 Success rate (% of agents evacuated)
- ⏱️ Average evacuation time
- 🔥 Average fire damage per agent
- 🌡️ Temperature exposure statistics
- 🚪 Most common evacuation paths
- ⚡ Performance metrics (time per run)

## 💡 Common Use Cases

### Research & Analysis
```bash
# Run 1000 simulations for statistical significance
python monte_carlo_queue.py \
  --queue-folder research_scenarios \
  --runs 1000 \
  --parallel \
  --seed 12345 \
  --output research_results
```

### Quick Testing
```bash
# Test with just 10 runs to verify everything works
python monte_carlo_queue.py \
  --configs test_config.json \
  --runs 10
```

### Batch Processing
```bash
# Process many scenarios overnight
python monte_carlo_queue.py \
  --queue-folder all_scenarios \
  --runs 500 \
  --parallel \
  --processes 8
```

### Reproducible Results
```bash
# Same seed = same results
python monte_carlo_queue.py \
  --queue-folder scenarios \
  --runs 200 \
  --seed 42 \
  --parallel
```

## 📁 Output File Sizes

Typical file sizes (depends on map size and agent count):

| Runs | full_results.json | statistics.json | summary.txt |
|------|-------------------|-----------------|-------------|
| 10   | ~5 MB            | ~50 KB          | ~5 KB       |
| 100  | ~50 MB           | ~500 KB         | ~10 KB      |
| 500  | ~250 MB          | ~2.5 MB         | ~15 KB      |
| 1000 | ~500 MB          | ~5 MB           | ~20 KB      |

**Tip**: Use `statistics.json` for analysis if you don't need full trajectories!

## 🔍 Example: Analyzing Results

```python
import json
import pandas as pd

# Load full results
with open('monte_carlo_results/scenario1_20250123_143052_001/full_results.json') as f:
    data = json.load(f)

# Convert to DataFrame for easy analysis
runs_df = pd.DataFrame(data['individual_runs'])

# Analyze success rates
print(f"Overall success rate: {runs_df['evacuated_agents'].sum() / runs_df['evacuated_agents'].count() * 100:.2f}%")

# Plot evacuation times
import matplotlib.pyplot as plt
runs_df['steps'].hist(bins=50)
plt.xlabel('Evacuation Time (steps)')
plt.ylabel('Frequency')
plt.title('Distribution of Evacuation Times')
plt.show()

# Compare fire damage
print(f"Average fire damage: {runs_df['average_fire_damage'].mean():.2f}")
print(f"Max fire damage: {runs_df['average_fire_damage'].max():.2f}")
```

## ⚙️ Performance Tips

1. **Use `--parallel`**: 5-10x faster on multi-core systems
2. **Start small**: Test with `--runs 10` first
3. **Limit processes**: Use `--processes 4` if system is slow
4. **Monitor memory**: Large runs can use lots of RAM
5. **Save incrementally**: Results saved after each config (safe to Ctrl+C between configs)

## 🆘 Common Issues

**"Not enough reachable positions"**
- Your fire/obstacles block too many positions
- Reduce number of agents or fire sources in config

**"No door path"**
- Agent can't reach any exit
- Fixed by the door path validation in `replace_agents()`

**Agents getting stuck**
- Should be fixed by all our recent bug fixes!
- Check individual run data to see where they got stuck

**Simulations too slow**
- Use `--parallel` mode
- Reduce map size or agent count
- Reduce `--runs` for testing

## 📞 Need Help?

See the full documentation:
- `MONTE_CARLO_QUEUE_README.md` - Detailed documentation
- `CLAUDE.md` - System architecture and code overview

---

**That's it! You're ready to run massive Monte Carlo simulations! 🎉**
