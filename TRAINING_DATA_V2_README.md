# Training Data Generation V2 - CORRECTED

## Key Fix

**V1 Problem**: Varied agents/fire positions but kept doors fixed - model couldn't learn door placement quality

**V2 Solution**: Varies DOOR/EXIT positions on the same floor plan - model learns which door configurations are better

## What the Model Learns

For the **same floor plan**, which **door/exit configuration** leads to better evacuation outcomes.

```
Floor Plan A (fixed)
├── Door Config 1: exits at corners → survival 95%
├── Door Config 2: exits clustered → survival 72%
├── Door Config 3: exits distributed → survival 88%
└── ...30 different door configs

Model learns: Config 1 > Config 3 > Config 2
```

## Key Differences from V1

| Aspect | V1 (Wrong) | V2 (Correct) |
|--------|------------|--------------|
| **What varies** | Agents, fire | **Door/exit positions** |
| **What's fixed** | Door config | **Floor plan** |
| **Model learns** | Agent/fire robustness | **Door placement quality** |
| **Uses existing code** | No | **Yes** (CandidateGenerator, batch_evaluate) |

## Architecture

```
For each floor plan:
  1. CandidateGenerator creates 30 different door/exit configs
  2. For each door config:
     - Run on 3 agent/fire scenarios (for robustness)
     - Each scenario: 3 Monte Carlo trials
     - Aggregate to median survival rate
  3. Compare door configs pairwise (200 pairs per floor plan)

Total: 1000 plans × 30 configs × 3 scenarios × 3 trials = 270K simulations
       → ~1M pairwise labels
```

## Usage

### Quick Test (5-10 minutes)

```bash
python generate_training_data_v2.py \
    --num-floor-plans 10 \
    --door-configs-per-plan 10 \
    --scenarios-per-config 2 \
    --trials-per-scenario 2 \
    --pairs-per-plan 50 \
    --workers 4 \
    --output-dir ./test_data_v2
```

### Full EPYC Run (20-30 hours)

```bash
python generate_training_data_v2.py \
    --num-floor-plans 1000 \
    --door-configs-per-plan 30 \
    --scenarios-per-config 3 \
    --trials-per-scenario 3 \
    --pairs-per-plan 200 \
    --workers 120 \
    --output-dir ./training_data_v2
```

## Optimizations Used

1. **CandidateGenerator** (from `candidate_generator.py`)
   - Random + rule-based door placement
   - Minimum spacing constraints
   - Room boundary detection

2. **batch_evaluate()** (from `fast_simulation.py`)
   - Parallel simulation execution
   - Uses Phase 2 FastEvacuationSim
   - 10-20x faster than original simulation

3. **Median aggregation**
   - Robust to simulation variance
   - Reduces noise in labels

## Output Format

```
training_data_v2/
├── train_pairs.jsonl          # 70% of pairs (by floor plan)
├── val_pairs.jsonl            # 15% of pairs
├── test_pairs.jsonl           # 15% of pairs
├── metadata.json              # Config, stats, validation
└── floor_plans/               # NPZ compressed grids
    ├── plan_00000.npz
    └── ...
```

### Pair Format (JSONL)

```json
{
  "floor_plan_id_a": 0,
  "floor_plan_id_b": 0,
  "config_a": {
    "door_config": [
      {"id": "e1", "position": "x5y0", "type": "exit"},
      {"id": "d1", "position": "x15y10", "type": "door"}
    ]
  },
  "config_b": {
    "door_config": [...]
  },
  "score_a": 0.92,
  "score_b": 0.75,
  "label": 1,
  "label_confidence": 0.85,
  "pair_type": "within_plan_mixed"
}
```

## Performance Estimates (EPYC 128-core)

| Configuration | Simulations | Pairs | Time |
|--------------|-------------|-------|------|
| Test (10 plans) | 600 | 500 | 5-10 min |
| Pilot (100 plans) | 27K | 20K | 2-3 hours |
| **Full (1000 plans)** | **270K** | **200K** | **20-30 hours** |

## Model Training

The model should take two inputs:
1. Floor plan grid (walls, obstacles)
2. Door/exit configuration overlay

And output: relative score (which configuration is better)

```python
class DoorRankingModel(nn.Module):
    def forward(self, floor_plan, door_config_a, door_config_b):
        # Overlay door configs on floor plan
        plan_a = overlay(floor_plan, door_config_a)
        plan_b = overlay(floor_plan, door_config_b)

        # Shared encoder
        feat_a = self.encoder(plan_a)
        feat_b = self.encoder(plan_b)

        # Comparison
        return sigmoid(self.comparator(feat_a, feat_b))
```

## Validation

Data validator checks:
- ✓ Floor plan validity (connectivity, exits reachable)
- ✓ Label balance (45-55% split)
- ✓ No data leakage (splits by floor plan)
- ✓ Diversity coverage (all sizes, door counts)

## Next Steps

1. Run pilot (100 plans) to verify quality
2. Inspect samples manually
3. Full run on EPYC
4. Train baseline ranking model
5. Evaluate on held-out floor plans
