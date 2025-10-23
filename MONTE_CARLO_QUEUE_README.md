# Monte Carlo Queue System

Automatically run multiple Monte Carlo simulations sequentially from a queue of configuration files. Each run generates comprehensive output with complete traceability.

## Features

- ✅ **Queue Processing**: Run multiple JSON configs automatically, one after another
- ✅ **Comprehensive Logging**: Every simulation saved with full details
- ✅ **Parallel or Serial**: Choose execution mode
- ✅ **Complete Traceability**: Reconstruct any simulation from output files
- ✅ **Human-Readable Summaries**: TXT files for quick review
- ✅ **Organized Output**: Each config gets its own timestamped folder

## Quick Start

### 1. Prepare Your Configuration Files

Create a folder with JSON configuration files:

```
configs/
├── scenario_small_fire.json
├── scenario_large_fire.json
├── scenario_multiple_exits.json
└── scenario_narrow_corridors.json
```

### 2. Run the Queue

**Process all configs in a folder (RECOMMENDED):**
```bash
python monte_carlo_queue.py --queue-folder ./configs --runs 100 --parallel
```

**Process specific config files:**
```bash
python monte_carlo_queue.py --configs config1.json config2.json config3.json --runs 50
```

**Serial mode (slower but easier to debug):**
```bash
python monte_carlo_queue.py --queue-folder ./configs --runs 20
```

**Custom output directory:**
```bash
python monte_carlo_queue.py --queue-folder ./configs --runs 100 --output ./my_results
```

**Specify number of CPU cores:**
```bash
python monte_carlo_queue.py --queue-folder ./configs --runs 100 --parallel --processes 4
```

## Output Structure

After running, you'll get a folder structure like this:

```
monte_carlo_results/
├── scenario_small_fire_20250123_143052_001/
│   ├── full_results.json       # Complete data (all runs)
│   ├── statistics.json         # Aggregated statistics only
│   ├── config_used.json        # Configuration that was used
│   └── summary.txt             # Human-readable summary
│
├── scenario_large_fire_20250123_143052_002/
│   ├── full_results.json
│   ├── statistics.json
│   ├── config_used.json
│   └── summary.txt
│
├── scenario_narrow_corridor_20250123_143052_003/
│   ├── full_results.json
│   ├── statistics.json
│   ├── config_used.json
│   └── summary.txt
│
└── queue_summary_20250123_143052.json  # Overall queue results
```

**Note**: Each folder has a unique 3-digit counter (001, 002, 003...) to:
- Prevent timestamp collisions (even if configs finish within the same second)
- Maintain processing order (alphabetical sorting = chronological order)
- Easy identification of which config was processed when

## Output Files Explained

### `full_results.json` - Complete Simulation Data
Contains EVERYTHING:
- Metadata (timestamp, runtime, mode, etc.)
- Full configuration used
- **Every individual simulation run** with:
  - Agent trajectories
  - Fire spread history
  - Evacuation times
  - Path taken
  - Temperature exposure
  - Fire damage per agent
- Aggregated statistics

**Use this to**: Fully reconstruct and analyze any simulation

### `statistics.json` - Aggregated Statistics
Smaller file with just the summary statistics:
- Average evacuation time
- Success rate
- Average fire damage
- Temperature statistics
- Path frequency counts

**Use this to**: Quick analysis without loading full data

### `config_used.json` - Configuration
Exact configuration that was used for this run, including:
- Map dimensions
- Fire model parameters
- Agent settings
- Door/exit positions

**Use this to**: Reproduce the exact simulation

### `summary.txt` - Human-Readable Report
Easy-to-read text summary:
```
================================================================================
MONTE CARLO SIMULATION SUMMARY
================================================================================

Timestamp: 2025-01-23 14:30:52

CONFIGURATION
-------------
Map Size: 60 x 60
Agents per Simulation: 20
Number of Runs: 100

RESULTS
-------
Total Agents: 2000
Evacuated Agents: 1847
Success Rate: 92.35%

Average Steps: 156.23
Average Fire Damage: 2.45
Average Peak Temperature: 87.32°C
```

**Use this to**: Quick review without opening JSON files

### `queue_summary_{timestamp}.json` - Queue Results
Summary of the entire queue run:
- Which configs were processed
- Success/failure status
- Time taken for each
- Overall statistics

**Use this to**: Track which runs completed successfully

## Command-Line Options

```
Required (one of):
  --queue-folder PATH     Process all JSON files in this folder
  --configs FILE [FILE]   Process specific JSON files

Optional:
  --runs N               Number of Monte Carlo runs per config (default: 100)
  --seed N               Random seed for reproducibility (default: 42)
  --parallel             Run in parallel mode (faster)
  --processes N          Number of CPU cores to use (default: all)
  --output PATH          Output directory (default: ./monte_carlo_results)
```

## Examples

### Example 1: Quick Test Run
```bash
# Run 10 simulations per config in serial mode
python monte_carlo_queue.py --queue-folder ./configs --runs 10
```

### Example 2: Production Run
```bash
# Run 500 simulations per config using all CPU cores
python monte_carlo_queue.py --queue-folder ./configs --runs 500 --parallel
```

### Example 3: Specific Scenarios
```bash
# Run only specific scenarios
python monte_carlo_queue.py \
  --configs baseline.json stress_test.json \
  --runs 200 \
  --parallel \
  --processes 8
```

### Example 4: Reproducible Research
```bash
# Use specific seed for reproducibility
python monte_carlo_queue.py \
  --queue-folder ./research_configs \
  --runs 1000 \
  --seed 12345 \
  --parallel \
  --output ./research_results
```

## Analyzing Results

### Load Full Results in Python
```python
import json

with open('monte_carlo_results/scenario_small_fire_20250123_143052_001/full_results.json') as f:
    data = json.load(f)

# Access individual runs
for i, run in enumerate(data['individual_runs']):
    print(f"Run {i}: {run['evacuated_agents']} evacuated in {run['steps']} steps")

# Access aggregated statistics
stats = data['aggregated_statistics']
print(f"Overall success rate: {stats['evacuated_agents']} / {data['metadata']['num_runs']} agents")
```

### Compare Multiple Configurations
```python
import json
from pathlib import Path

results_dir = Path('monte_carlo_results')
summaries = []

for stats_file in results_dir.glob('*/statistics.json'):
    with open(stats_file) as f:
        summaries.append(json.load(f))

# Compare success rates
for i, summary in enumerate(summaries):
    print(f"Config {i}: {summary.get('success_rate', 'N/A')}% success rate")
```

## Tips

1. **Start Small**: Test with `--runs 10` first to verify configurations work
2. **Use Parallel Mode**: Much faster for large runs (`--parallel`)
3. **Monitor Progress**: The queue shows progress for each config
4. **Check Queue Summary**: Review `queue_summary_*.json` to ensure all ran successfully
5. **Organize Configs**: Group related scenarios in folders for easier management

## Troubleshooting

**"No JSON files found"**
- Ensure your folder contains `.json` files
- Check the folder path is correct

**"Config file not found"**
- Verify file paths when using `--configs`
- Use absolute paths if having issues

**Out of Memory**
- Reduce `--runs` number
- Process fewer configs at once
- Reduce number of agents in config files

**Simulations taking too long**
- Use `--parallel` mode
- Reduce `--runs` for testing
- Check if agents are getting stuck (review individual run data)

## Performance Expectations

Typical performance (depends on map size, agent count, fire spread):

- **Serial mode**: ~0.5-2 seconds per simulation
- **Parallel mode (8 cores)**: ~5-8x speedup
- **100 runs**: ~1-3 minutes (parallel), ~8-15 minutes (serial)
- **1000 runs**: ~10-30 minutes (parallel), ~80-150 minutes (serial)

## Notes

- Each config run is **independent** - if one fails, others continue
- Results are saved **immediately** after each config completes
- **Random seed** ensures reproducibility across runs
- Output files can be **large** (100+ MB for 1000 runs with detailed trajectories)
- Use `statistics.json` for analysis when you don't need full trajectory data
