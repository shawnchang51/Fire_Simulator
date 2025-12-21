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

## What Gets Randomized Per Run

Each Monte Carlo run randomizes:

✅ **Occupant density** (2%-10% of passable cells)
✅ **Agent positions** (random placement based on density)
✅ **Fire count** (2-5 fires for better differentiation)
✅ **Fire positions** (randomly placed, no overlap with agents)
✅ **Fire spread rate** (0.2-0.6: normal to aggressive)
✅ **Fire intensity growth** (0.3-1.0: slow to fast)
✅ **Fire discovery delay** (0-20 steps: early to late detection)

**Fixed for consistent baseline:**
🔒 **Fire damage threshold** (10.0 - same for all runs)

Plus:
✅ **Phase 2 fast simulation** (10-20x speedup)
✅ **Parallel execution** with multiprocessing
✅ **Statistical aggregation** (median across runs)

## Data Flow

```python
# For each door config:
door_config = [
    {"id": "e1", "position": "x5y0", "type": "exit"},
    {"id": "d1", "position": "x15y10", "type": "door"}
]

# Run 10 Monte Carlo trials, each randomizes:
for run in range(10):
    # Randomize occupant density → agent count
    occupant_density = random(0.02, 0.10)  # 2%-10% density
    num_agents = int(passable_cells × occupant_density)
    agent_positions = random_placement(num_agents)

    # Randomize fires (2-5 for better differentiation)
    num_fires = random(2, 5)
    fire_positions = random_placement(num_fires)

    # Randomize fire behavior
    fire_spread_rate = random(0.2, 0.6)       # Normal to aggressive
    fire_intensity_growth = random(0.3, 1.0)  # Slow to fast
    fire_discovery_delay = random(0, 20)      # Early to late
    fire_damage_threshold = 10.0              # FIXED baseline

    # Run Phase 2 simulation
    result = FastEvacuationSim(
        grid, agent_positions, door_config,
        fire_positions, fire_spread_rate, ...
    ).run()

# Aggregate across 10 runs (median for robustness):
{
    'survival_rate': 0.875,           # Median survival rate
    'avg_steps': 145.3,               # Median evacuation time
    'avg_fire_damage': 2.1,           # Median fire damage
    'occupant_density_range': [0.024, 0.095],
    'agent_count_range': [15, 47],
    'num_fires_range': [2, 5],        # Actual fire counts used
    'fire_spread_rate_range': [0.23, 0.58],
    'fire_delay_range': [2, 18]
}

# Score for pairwise comparison:
score = survival_rate - (avg_steps/1000)
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
| Agent count | Fixed per scenario | **Density-based (2-10% occupancy)** |
| Agent placement | Fixed per scenario | **Randomized per run** |
| Fire count | Fixed (1-3) | **More fires (2-5) for differentiation** |
| Fire placement | Fixed per scenario | **Randomized per run** |
| Fire spread rate | Fixed | **Randomized per run (0.2-0.6)** |
| Fire intensity | Fixed | **Randomized per run (0.3-1.0)** |
| Discovery delay | Fixed | **Randomized per run (0-20 steps)** |
| Damage threshold | Randomized | **FIXED (10.0) for consistent baseline** |
| Code approach | Manual simulation calls | **Phase 2 with full randomization** |
| Robustness | 3×3 trials | **10 MC runs, all randomized** |
| Statistics | Manual median | **Robust median aggregation** |

## Key Implementation Details

The script directly uses **Phase 2 FastEvacuationSim** with full parameter randomization:

```python
# For each Monte Carlo run:
for run in range(10):
    # Randomize occupant density → agent count
    occupant_density = random.uniform(0.02, 0.10)  # 2-10% density
    num_agents = int(len(passable_cells) * occupant_density)
    agent_positions = random_placement(num_agents)

    # Randomize fire count (2-5 for better differentiation)
    num_fires = random.randint(2, 5)
    fire_positions = random_placement(num_fires)

    # Randomize fire parameters
    fire_spread_rate = random.uniform(0.2, 0.6)
    fire_intensity_growth = random.uniform(0.3, 1.0)
    fire_discovery_delay = random.randint(0, 20)

    # FIXED threshold for consistent baseline
    fire_damage_threshold = 10.0

    # Run Phase 2 simulation
    sim = FastEvacuationSim(
        grid=grid,
        agent_starts=agent_positions,
        exits=exit_positions,  # From door_config (FIXED - what we're evaluating)
        fire_starts=fire_positions,
        fire_spread_rate=fire_spread_rate,
        fire_intensity_growth=fire_intensity_growth,
        fire_discovery_delay=fire_discovery_delay,
        fire_damage_threshold=fire_damage_threshold,
        fire_spread_mode='always_real'
    )

    result = sim.run(max_steps=500)

# Aggregate median statistics across 10 runs
median_survival_rate = np.median([run.survival_rate for run in results])
median_steps = np.median([run.steps for run in results])
```

## What Stays Fixed (for Fair Comparison)

Across all 10 runs for a given door configuration:

- ✅ **Floor plan structure** (walls, obstacles)
- ✅ **Door/exit positions** (what we're evaluating!)

This ensures door configs are compared fairly under varying conditions.

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
