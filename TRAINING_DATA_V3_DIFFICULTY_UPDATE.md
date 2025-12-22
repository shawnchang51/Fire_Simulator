# Training Data V3 - Difficulty Improvements

## Problem
Initial test runs showed scenarios were too easy:
- 100% survival rate
- 0 fire damage
- ~4.5 average steps (very fast evacuation)
- All door configurations performing identically well

**Root cause:** Model can't learn if all configurations are equally good.

## Changes Made

### 1. Larger Floor Plans
**Before:** 20-80 cells (average ~50x50 = 2,500 cells)
**After:** 50-120 cells (average ~85x85 = 7,225 cells)

**Impact:** 
- Longer evacuation distances
- More complex navigation
- Door placement matters more

### 2. Higher Obstacle Density
**Before:** 2-8% obstacles
**After:** 8-25% obstacles (depending on generation method)

**Breakdown by generation method:**
- BSP: 8-20%
- Grid: 8-20%
- Template: 10-25%
- Cellular Automata: 45-55% initial fill (smoothed to ~30-40%)

**Impact:**
- More walls and furniture
- Narrower passages
- Harder pathfinding
- Creates bottlenecks

### 3. Higher Occupant Density
**Before:** 2-10% of passable cells
**After:** 5-15% of passable cells

**Impact:**
- More agents = more congestion
- Bottlenecks at doors/corridors
- Exit placement becomes critical

### 4. More Aggressive Fire
**Fire count:**
- Before: 2-5 fires
- After: 3-7 fires

**Fire spread rate:**
- Before: 0.2-0.6
- After: 0.3-0.8

**Fire intensity growth:**
- Before: 0.3-1.0
- After: 0.5-1.5

**Discovery delay:**
- Before: 0-20 steps
- After: 5-30 steps (minimum delay allows fire to establish)

**Impact:**
- Fire spreads faster and more aggressively
- More time for fire to establish before evacuation
- Agents face actual danger

### 5. Increased Door Spacing
**Before:** 3 cells minimum
**After:** 5 cells minimum

**Impact:**
- More distinct door configurations
- Reduces overlap between similar placements

## Expected Results

### Before (Too Easy):
```
Survival rate: 99-100%
Fire damage: 0-0.5
Avg steps: 4-10
Score range: 0.995-1.000 (no differentiation)
```

### After (Challenging):
```
Survival rate: 60-95% (wider range)
Fire damage: 0-5 (actual danger)
Avg steps: 20-150 (meaningful differences)
Score range: 0.600-0.950 (clear differentiation)
```

## Verification

Run a small test to verify scenarios are now challenging:

```bash
python generate_training_data_v3.py \
    --num-floor-plans 5 \
    --door-configs-per-plan 20 \
    --monte-carlo-runs 5 \
    --workers 6 \
    --output-dir ./test_difficulty \
    --seed 42
```

Check `test_difficulty/simulation_results.jsonl` for:
- Variable survival rates (not all 1.0)
- Non-zero fire damage
- Larger average step counts
- Score diversity

## Impact on Training

With challenging scenarios:
- **Better differentiation:** Scores now range 0.6-0.95 instead of 0.995-1.0
- **Meaningful labels:** Model can learn which door placements actually help
- **Robust model:** Trained on diverse difficulty levels
- **Practical value:** Model learns for realistic emergency scenarios

## Rollback

If scenarios become too hard (e.g., <30% survival across all configs), adjust:
- Reduce fire spread rate: 0.3-0.8 → 0.2-0.6
- Reduce obstacle density: 8-20% → 5-15%
- Reduce occupant density: 5-15% → 3-12%
