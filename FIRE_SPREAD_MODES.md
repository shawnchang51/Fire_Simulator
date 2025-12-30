# Fire Spread Modes

This document describes the three fire spread modes available in Phase 2 simulations (FastEvacuationSim).

## Overview

The fire spread mode controls how fire behaves during the simulation, balancing realism with computational performance.

## Available Modes

### 1. `always_real` (Default - Most Realistic)
**Behavior:** Continuous stochastic (random) fire spread throughout the entire simulation.

**Characteristics:**
- Fire spreads probabilistically based on neighbor influence
- New cells can ignite at any time during the simulation
- Fire intensity grows continuously until reaching maximum (4.0)
- Most realistic fire behavior
- Higher computational cost due to ongoing spread calculations

**Use cases:**
- Final validation of floor plans
- When realism is critical
- Research and publication-quality simulations

**Visual behavior:** Fire continuously spreads and grows throughout simulation

---

### 2. `real_then_simple` (Balanced)
**Behavior:** Stochastic spread initially, then switches to intensity growth only after fire stabilizes.

**Characteristics:**
- Phase 1: Stochastic spread (same as `always_real`)
- Detects stability: 3 consecutive steps with no new ignitions
- Phase 2: No new spread, only existing fire intensities grow
- Good balance between realism and performance
- Performance boost after stabilization (~20-40% faster)

**Use cases:**
- Medium-scale Monte Carlo simulations (100-500 runs)
- When you need realistic early spread but want performance gains
- Training RL agents with some realism

**Visual behavior:** Fire spreads actively at first, then existing fires intensify without new spread

---

### 3. `real_then_stop` (Maximum Performance)
**Behavior:** Stochastic spread initially, then fire becomes completely static after stabilization.

**Characteristics:**
- Phase 1: Stochastic spread (same as `always_real`)
- Detects stability: 3 consecutive steps with no new ignitions
- Phase 2: Fire completely frozen (no spread, no intensity growth)
- Maximum performance boost (~40-60% faster after stabilization)
- Fire hazards remain constant for predictable agent navigation

**Use cases:**
- Large-scale Monte Carlo simulations (1000+ runs)
- RL training where deterministic environments are preferred
- Performance-critical scenarios
- When you only care about final fire distribution, not dynamics

**Visual behavior:** Fire spreads initially, then becomes completely static (no visual changes)

---

## Usage

### Visual Validation (`run_phase2_visual.py`)

```bash
# Default: always_real (continuous stochastic spread)
python run_phase2_visual.py

# Balanced: stochastic then intensity-only
python run_phase2_visual.py --fire-spread-mode real_then_simple

# Maximum performance: stochastic then static
python run_phase2_visual.py --fire-spread-mode real_then_stop
```

### Monte Carlo Simulations (`monte_carlo.py`)

```bash
# Default: always_real
python monte_carlo.py --runs 100 --parallel --phase2

# Balanced mode for medium-scale runs
python monte_carlo.py --runs 500 --parallel --phase2 --fire-spread-mode real_then_simple

# Maximum performance for large-scale runs
python monte_carlo.py --runs 1000 --parallel --phase2 --fire-spread-mode real_then_stop
```

### In Code (FastEvacuationSim)

```python
from fast_simulation import FastEvacuationSim
from fast_fire import FireSpreadMode

# Option 1: Use string
sim = FastEvacuationSim(
    grid=fire_map,
    agent_starts=agent_starts,
    exits=exits,
    fire_spread_mode='real_then_simple'  # or 'always_real', 'real_then_stop'
)

# Option 2: Use enum directly
sim = FastEvacuationSim(
    grid=fire_map,
    agent_starts=agent_starts,
    exits=exits,
    fire_spread_mode=FireSpreadMode.REAL_THEN_SIMPLE
)
```

---

## Stability Detection

For `real_then_simple` and `real_then_stop` modes, stability is detected when:
- **3 consecutive fire update steps** produce zero new ignitions
- Adjustable via `stability_threshold` parameter in `FastFireModel`

When stability is reached, you'll see console output:
```
Fire spread stabilized after 3 steps without new ignitions
  → Switching to intensity growth only (mode: real_then_simple)
```

---

## Performance Comparison

Based on typical 50x50 grid with 50 agents over 500 steps:

| Mode | Relative Speed | Realism | Best For |
|------|---------------|---------|----------|
| `always_real` | 1.0x (baseline) | Highest | Validation, research |
| `real_then_simple` | 1.3-1.4x faster | High | Medium Monte Carlo |
| `real_then_stop` | 1.5-1.6x faster | Medium | Large Monte Carlo, RL training |

*Performance gains increase with larger grids and longer simulations*

---

## Technical Details

### Fire Spread Mechanism

**Stochastic Spread (all modes, Phase 1):**
- Uses convolution with spread kernel to calculate ignition probability
- Random number generation determines which cells ignite
- Neighbor influence: adjacent cells 0.15, diagonal 0.05
- Spread rate multiplier: 0.3 (default)

**Intensity Growth:**
- All fire cells grow by +0.5 per fire update step
- Maximum intensity: 4.0
- Growth occurs in all modes except `real_then_stop` after stabilization

**Static Fire (`real_then_stop` Phase 2):**
- No grid modifications after stability
- Zero computational overhead for fire updates
- Agents still accumulate damage from existing fire

---

## Recommendations

### For Different Scenarios:

1. **Validating a final floor plan design:**
   - Use `always_real` for maximum realism
   - Run 50-100 Monte Carlo simulations

2. **Training RL agents:**
   - Use `real_then_stop` for speed and determinism
   - Optionally use `real_then_simple` if fire dynamics matter

3. **Large-scale parameter sweeps:**
   - Use `real_then_stop` with 1000+ runs
   - Combine with `--no-full-results` for memory efficiency

4. **Debugging/Development:**
   - Use `always_real` with visual validation
   - Switch to `real_then_simple` for faster iteration

---

## Limitations

1. **Only available in Phase 2 simulations** (FastEvacuationSim)
   - Original simulation (EvacuationSimulation) uses continuous physics-based fire

2. **Stability detection is heuristic**
   - May transition too early if fire spread is naturally slow
   - May transition too late if fire spreads in bursts
   - Adjust `stability_threshold` if needed

3. **`real_then_stop` loses some realism**
   - Fire doesn't respond to changes after stabilization
   - May not be suitable for scenarios with late-stage fire dynamics

---

## Example Output

```bash
$ python run_phase2_visual.py --fire-spread-mode real_then_simple

============================================================
Phase 2 Visual Validation
============================================================
Config: example_configuration.json
Visualization: Pygame
Max steps: 500
Fire spread mode: real_then_simple
============================================================

Step 4: Fire spreading to 12 cells
Step 8: Fire spreading to 18 cells
Step 12: Fire spreading to 8 cells
Step 16: Fire spreading to 3 cells
Step 20: Fire spreading to 0 cells
Step 24: Fire spreading to 0 cells
Step 28: Fire spreading to 0 cells
Fire spread stabilized after 3 steps without new ignitions
  → Switching to intensity growth only (mode: real_then_simple)
```

---

## Migration from Old Code

If you were using `deterministic_fire=True/False`:

**Before:**
```python
sim = FastEvacuationSim(..., deterministic_fire=True)   # Old DeterministicFireModel
sim = FastEvacuationSim(..., deterministic_fire=False)  # Old stochastic
```

**After (recommended):**
```python
sim = FastEvacuationSim(..., fire_spread_mode='real_then_stop')   # Similar to deterministic
sim = FastEvacuationSim(..., fire_spread_mode='always_real')      # Pure stochastic
sim = FastEvacuationSim(..., fire_spread_mode='real_then_simple') # New balanced option
```

The `deterministic_fire` parameter is still supported for backward compatibility but may be deprecated in future versions.
