# Training Data Generation V3 - Monte Carlo Integration

## Why V3?

**V3 properly leverages your optimized `monte_carlo.py --phase2`** infrastructure.

| Version | What It Does | Problem |
|---------|-------------|---------|
| V1 | Varies agents/fire, keeps doors fixed | ❌ Model can't learn door placement |
| V2 | Varies doors, manually calls simulations | ⚠️ Doesn't use monte_carlo's randomization |
| **V3** | **Varies doors, uses monte_carlo phase2** | ✅ **Correct approach** |

## Architecture

```
For each floor plan:
  1. CandidateGenerator creates 30 door/exit configs
  2. For each door config:
     ├─ Create temp SimulationConfig with that door config
     ├─ Run monte_carlo.py --phase2 --runs 10
     │  ├─ Run 1: Random agent positions, random fire
     │  ├─ Run 2: Random agent positions, random fire
     │  └─ ... (randomized by monte_carlo)
     └─ Get aggregated statistics (success_rate, avg_steps, etc.)
  3. Compare door configs using monte carlo statistics
```

## What Monte Carlo Provides

✅ **Random agent placement** per run
✅ **Random fire positions** per run
✅ **Phase 2 fast simulation** (10-20x speedup)
✅ **Parallel execution** with multiprocessing
✅ **Statistical aggregation** (median, percentiles)
✅ **Memory optimizations** (`--no-full-results`)

## Data Flow

```python
# For each door config:
door_config = [
    {"id": "e1", "position": "x5y0", "type": "exit"},
    {"id": "d1", "position": "x15y10", "type": "door"}
]

# Monte carlo runs 10 times with different:
# - Agent starting positions (randomized)
# - Fire starting positions (randomized)

# Returns aggregated statistics:
{
    'success_rate': 87.5,      # % of successful evacuations
    'average_steps': 145.3,    # Median evacuation time
    'average_fire_damage': 2.1,
    'evacuated_agents': 437,   # Total across runs
    'survived_agents': 450     # evacuated + stuck
}

# Score for pairwise comparison:
score = success_rate/100 - (average_steps/1000)
```

## Usage

### Quick Test (10-15 minutes)

```bash
python generate_training_data_v3.py \
    --num-floor-plans 10 \
    --door-configs-per-plan 10 \
    --monte-carlo-runs 5 \
    --workers 4 \
    --output-dir ./test_v3
```

### Full Production (EPYC)

```bash
python generate_training_data_v3.py \
    --num-floor-plans 1000 \
    --door-configs-per-plan 30 \
    --monte-carlo-runs 10 \
    --pairs-per-plan 200 \
    --workers 120 \
    --output-dir ./training_data_v3 \
    --seed 42
```

**Total simulations**: 1000 × 30 × 10 = 300,000 monte carlo runs
**Expected time**: 15-25 hours on EPYC 128-core
**Training pairs**: ~200,000

## Performance Estimates

| Configuration | MC Runs | Pairs | Time (EPYC) |
|--------------|---------|-------|-------------|
| Test (10 plans) | 500 | 500 | 10-15 min |
| Pilot (100 plans) | 30K | 20K | 2-3 hours |
| **Full (1000 plans)** | **300K** | **200K** | **15-25 hours** |

## Advantages Over V2

| Aspect | V2 | V3 |
|--------|----|----|
| Agent placement | Fixed per scenario | **Randomized per run** |
| Fire placement | Fixed per scenario | **Randomized per run** |
| Code reuse | Manual simulation calls | **Uses monte_carlo.py** |
| Robustness | 3 trials per scenario | **10 MC runs with randomization** |
| Statistics | Manual median | **Monte carlo aggregation** |

## Key Differences from Monte Carlo CLI

The script **wraps** `run_monte_carlo_parallel()` programmatically:

```python
# V3 creates temporary config for each door config
config = SimulationConfig.from_dict({
    'map_rows': 40,
    'map_cols': 40,
    'door_configs': door_config,  # THE KEY VARIATION
    'agent_num': 30,
    # ... other params
})

# Calls monte carlo directly
results, statistics = run_monte_carlo_parallel(
    config=config,
    num_runs=10,
    use_phase2=True,
    fire_spread_mode='always_real'
)

# Extracts statistics for comparison
survival_rate = statistics['success_rate'] / 100.0
```

## What Gets Randomized (by Monte Carlo)

Monte carlo's `replace_agents()` and `replace_fire()` functions randomize:

- **Agent positions**: Different starting positions each run
- **Fire positions**: Different fire locations each run
- **Agent count** (optional): Can vary per run

What stays **fixed** (for fair comparison):
- Floor plan structure
- **Door/exit positions** (the variable we're learning)
- Fire spread parameters

## Output Format

Same as V2:

```
training_data_v3/
├── train_pairs.jsonl
├── val_pairs.jsonl
├── test_pairs.jsonl
├── metadata.json
└── floor_plans/
    └── plan_*.npz
```

### Pair Entry Example

```json
{
  "floor_plan_id_a": 5,
  "floor_plan_id_b": 5,
  "config_a": {
    "door_config": [
      {"id": "e1", "position": "x5y0", "type": "exit"},
      {"id": "e2", "position": "x35y0", "type": "exit"}
    ]
  },
  "config_b": {
    "door_config": [
      {"id": "e1", "position": "x20y0", "type": "exit"}
    ]
  },
  "score_a": 0.875,
  "score_b": 0.623,
  "label": 1,
  "label_confidence": 0.89
}
```

## Validation

- ✅ Uses optimized monte_carlo phase2
- ✅ Proper randomization of agents/fire
- ✅ Statistical robustness (10 runs per config)
- ✅ Compares door configs on same floor plan
- ✅ No data leakage (splits by floor plan)

## Next Steps

1. **Test run** (10 plans) - verify correctness
2. **Pilot run** (100 plans) - check diversity/quality
3. **Full run** (1000 plans) on EPYC
4. Train ranking model on pairwise labels
5. Evaluate on held-out floor plans

## Recommended Command

```bash
# Full production run
python generate_training_data_v3.py \
    --num-floor-plans 1000 \
    --door-configs-per-plan 30 \
    --monte-carlo-runs 10 \
    --workers 120 \
    --output-dir ./training_data_v3
```

This generates **high-quality pairwise labels** by properly leveraging your optimized monte carlo infrastructure with Phase 2 fast simulation.
