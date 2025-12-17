# Monte Carlo Phase 2 Integration

The `monte_carlo.py` script now supports **Phase 2 optimizations** for 10-20x faster simulations!

## What Changed

### New Command-Line Flag: `--phase2`

Use the `--phase2` flag to enable Phase 2 optimized simulations:

```bash
# Original (slow)
python monte_carlo.py --runs 100 --parallel

# With Phase 2 (10-20x faster!)
python monte_carlo.py --runs 100 --parallel --phase2
```

### Performance Comparison

| Configuration | Original | Phase 2 | Speedup |
|--------------|----------|---------|---------|
| 100 runs, 5 agents | ~200s | ~10-20s | **10-20x** |
| 1000 runs, 10 agents | ~40 min | ~2-4 min | **10-20x** |

### Automatic Fallback

If Phase 2 files aren't available, `monte_carlo.py` automatically falls back to the original simulation:

```bash
$ python monte_carlo.py --runs 100 --parallel --phase2

WARNING: Phase 2 optimization requested but fast_simulation.py not found!
Falling back to original simulation.
```

## Usage Examples

### Basic Phase 2 Usage
```bash
python monte_carlo.py --runs 100 --parallel --phase2
```

### Maximum Performance Mode
Combine `--phase2` with `--no-full-results` for lowest memory usage:
```bash
python monte_carlo.py --runs 1000 --parallel --phase2 --no-full-results
```

### Custom Configuration
```bash
python monte_carlo.py --config my_config.json --runs 500 --parallel --phase2 --processes 8
```

## Output Format

Phase 2 results are compatible with the original format:
- ✅ Statistics aggregation works the same
- ✅ Distribution analysis works the same
- ✅ Output files (summary.txt, statistics.json) unchanged
- ⚠️ Phase 2 doesn't track fire damage/temperature (set to 0.0)

## Requirements

Phase 2 optimization requires these files:
- `fast_simulation.py` - Lightweight simulation engine
- `optimized_d_star_lite.py` - Optimized pathfinding
- `fast_fire.py` - Vectorized fire model

If these files are missing, `monte_carlo.py` falls back to original simulation automatically.

## When to Use Phase 2

### ✅ Use Phase 2 When:
- Running 100+ simulations for statistical analysis
- Generating training data for AI/ML models
- Quick prototyping and iteration
- You need results fast and don't need detailed fire physics

### ❌ Use Original When:
- You need detailed fire damage/temperature tracking
- Visualizations are required
- Running <10 simulations (overhead not worth it)
- Debugging agent behavior in detail

## Technical Details

### What Phase 2 Optimizes

**D* Lite Pathfinding (3-5x faster):**
- Integer coordinates (no string parsing)
- NumPy-backed arrays
- Shared grid across agents
- Spatial filtering for environment updates

**Fire Model (5-10x faster):**
- Vectorized NumPy operations
- No oxygen/temperature/smoke tracking
- Deterministic spread option

**Simulation Engine (10-20x overall):**
- Minimal agent state
- No visualization overhead
- Early termination for bad designs
- Maintains D* Lite incremental replanning (preserves accuracy)

### Compatibility Notes

Phase 2 results are converted to original format:
```python
result = {
    'steps': phase2_result.steps,
    'evacuated_agents': phase2_result.evacuated,
    'survived_agents': phase2_result.evacuated,
    'average_fire_damage': 0.0,  # Not tracked in Phase 2
    'average_peak_temp': 0.0,    # Not tracked in Phase 2
    'average_avg_temp': 0.0,     # Not tracked in Phase 2
    '_phase2': True,  # Flag to indicate Phase 2 was used
}
```

## Troubleshooting

### Phase 2 not working?
```bash
# Check if files exist
ls fast_simulation.py optimized_d_star_lite.py fast_fire.py

# Try importing
python -c "from fast_simulation import FastEvacuationSim; print('OK')"
```

### ImportError for Phase 2 modules?
Ensure all three files are in the same directory as `monte_carlo.py`:
- `fast_simulation.py`
- `optimized_d_star_lite.py`
- `fast_fire.py`

### Results look different?
Phase 2 doesn't track fire damage/temperature, so these fields will be 0.0. Evacuation metrics (steps, evacuated agents, survival rate) should be similar but may differ slightly due to simplified fire model.

## Migration Guide

### From Original to Phase 2

**Before:**
```bash
python monte_carlo.py --runs 100 --parallel
```

**After (just add --phase2):**
```bash
python monte_carlo.py --runs 100 --parallel --phase2
```

That's it! Output format remains the same.

### Comparing Results

Run both modes and compare:
```bash
# Original
python monte_carlo.py --runs 100 --parallel --output ./results_original

# Phase 2
python monte_carlo.py --runs 100 --parallel --phase2 --output ./results_phase2

# Compare survival rates
cat results_original/*/summary.txt | grep "Success Rate"
cat results_phase2/*/summary.txt | grep "Success Rate"
```

## Performance Benchmarks

Expected performance on typical hardware (8-core CPU):

| Scenario | Original | Phase 2 | Speedup |
|----------|----------|---------|---------|
| 10 runs, 5 agents, 30x30 grid | 20s | 2s | 10x |
| 100 runs, 10 agents, 30x30 grid | 200s | 15s | 13x |
| 1000 runs, 5 agents, 50x50 grid | 2400s (40m) | 150s (2.5m) | 16x |

*Note: Actual speedup depends on CPU, grid size, agent count, and fire complexity*

## See Also

- `PHASE2_README.md` - Full Phase 2 implementation details
- `AI-Guided_Design_Optimization.md` - Overall optimization roadmap
- `PERFORMANCE_RESULTS.md` - Detailed performance analysis
