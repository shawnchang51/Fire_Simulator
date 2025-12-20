# Phase 2 Implementation - Moderate Optimizations

**Status:** ✅ Complete
**Target:** 10-50x speedup over Phase 1
**Expected Performance:** 36,000-90,000 simulations/hour

## Components Implemented

### 1. OptimizedDStarLite (`optimized_d_star_lite.py`)

**Performance:** 3-5x faster than original D* Lite

**Key optimizations:**
- Integer coordinates (no string parsing)
- NumPy-backed arrays for g/rhs values
- Preallocated data structures
- Efficient neighbor iteration with precomputed direction arrays
- Shared grid reference (no deep copies)

**Classes:**
- `OptimizedDStarLite`: Core pathfinder with NumPy optimizations
- `SharedGridDStarLite`: Multi-agent manager with spatial filtering

**Usage:**
```python
from optimized_d_star_lite import OptimizedDStarLite, SharedGridDStarLite
import numpy as np

# Single agent
grid = np.zeros((30, 30), dtype=np.float32)
pathfinder = OptimizedDStarLite(grid, start=(5, 5), goal=(25, 25))
pathfinder.compute_shortest_path()
next_move = pathfinder.get_next_move()

# Multi-agent with shared grid
manager = SharedGridDStarLite(grid)
agent1 = manager.add_agent(start=(5, 5), goal=(25, 25))
agent2 = manager.add_agent(start=(10, 10), goal=(25, 25))

# Batch environment update with spatial filtering
changed_cells = [(15, 15), (16, 15)]
manager.update_environment(changed_cells, spatial_filter=True)
```

### 2. FastFireModel (`fast_fire.py`)

**Performance:** 5-10x faster than AdvancedFireModel

**Simplifications for speed:**
- No oxygen/temperature/smoke tracking
- Fixed spread probabilities
- Vectorized NumPy operations
- Manual convolution (faster than scipy for small kernels)

**Classes:**
- `FastFireModel`: Stochastic fire spread with vectorization
- `DeterministicFireModel`: Threshold-based spread for reproducible RL training

**Usage:**
```python
from fast_fire import FastFireModel, DeterministicFireModel
import numpy as np

# Stochastic fire
grid = np.zeros((30, 30), dtype=np.float32)
grid[15, 15] = 2.0  # Initial fire
fire = FastFireModel(grid, spread_rate=0.3, intensity_growth=0.5)
fire.set_seed(42)

for _ in range(100):
    grid = fire.step()

fire_cells = fire.get_fire_cells()

# Deterministic fire (same input → same output)
fire_det = DeterministicFireModel(grid, spread_threshold=0.3)
grid_det = fire_det.step_n(10)
```

**Fire Configuration Parameters (JSON config):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fire_spread_rate` | 0.3 | Spread probability multiplier (0.3=normal, 0.6=aggressive) |
| `fire_intensity_growth` | 0.5 | Intensity growth per step (0.5=normal, 1.0=aggressive) |
| `fire_discovery_delay` | 0 | Steps of fire spread before agents start moving |
| `fire_damage_threshold` | 10.0 | Cumulative fire damage that counts as casualty (0=disabled) |

**Survival Rate Calculation:**

```python
# Agents accumulate fire_damage when in fire cells:
agent.fire_damage += fire_intensity  # intensity is 1-4

# Survival calculation:
if fire_damage_threshold > 0:
    fire_casualties = count(agents with fire_damage >= threshold)
    survived = total - dead - fire_casualties
else:
    survived = evacuated + stuck  # Legacy behavior
```

**Recommended Configurations for Variance:**

| Scenario | `fire_spread_rate` | `fire_intensity_growth` | `fire_discovery_delay` | `fire_damage_threshold` |
|----------|-------------------|------------------------|------------------------|------------------------|
| Baseline | 0.3 | 0.5 | 0 | 0 (disabled) |
| Moderate | 0.5 | 0.75 | 20 | 10.0 |
| Aggressive | 0.6 | 1.0 | 40 | 10.0 |
| Worst case | 0.7 | 1.5 | 60 | 5.0 |

### 3. FastEvacuationSim (`fast_simulation.py`)

**Performance:** 10-20x faster than full EvacuationSimulation

**Optimizations:**
- Minimal agent state with `__slots__`
- No visualization or I/O overhead
- Integer coordinates only
- Shared grid across agents
- Early termination for bad designs
- Maintains D* Lite incremental replanning (critical for accuracy)

**Classes:**
- `FastAgent`: Minimal agent dataclass
- `FastEvacuationSim`: Lightweight simulation engine
- `SimResult`: RL-relevant metrics

**Usage:**
```python
from fast_simulation import FastEvacuationSim, evaluate_floor_plan, batch_evaluate
import numpy as np

# Single simulation
grid = np.zeros((30, 30), dtype=np.float32)
agent_starts = [(5, 5), (10, 10), (15, 15)]
exits = [(28, 28)]
fire_starts = [(15, 15)]

sim = FastEvacuationSim(
    grid=grid,
    agent_starts=agent_starts,
    exits=exits,
    fire_starts=fire_starts,
    deterministic_fire=True,
    fire_update_interval=4,
    fire_discovery_delay=40,        # Steps before agents start moving
    fire_spread_rate=0.5,           # 0.3=normal, 0.6=aggressive
    fire_intensity_growth=0.75,     # 0.5=normal, 1.0=aggressive
    fire_damage_threshold=10.0      # Cumulative damage for casualty (0=disabled)
)

result = sim.run(max_steps=200, stuck_threshold=0.5, death_threshold=0.3)
print(f"Survival rate: {result.survival_rate:.2%}")
print(f"Reward: {result.reward}")

# Convenience function
result = evaluate_floor_plan(
    floor_plan=grid,
    agent_positions=agent_starts,
    exit_positions=exits,
    fire_positions=fire_starts,
    seed=42
)

# Parallel batch evaluation
scenarios = [
    {'floor_plan': grid, 'agent_positions': agent_starts,
     'exit_positions': exits, 'fire_positions': fire_starts, 'seed': i}
    for i in range(100)
]
results = batch_evaluate(scenarios, num_workers=8)
```

### 4. ScoringNetworkInterface (`pairwise_ranking_interface.py`)

**Purpose:** Integration layer for pairwise label generation and AI-guided search

**Key methods:**
- `evaluate_candidate()`: k Monte Carlo trials per door configuration
- `generate_candidate_labels()`: Generate pairwise comparison labels (A > B?)
- `batch_evaluate_topk()`: Validate scoring network's top-k predictions
- `log_evaluation()`: Track model vs simulator correlation

**Usage:**
```python
from pairwise_ranking_interface import ScoringNetworkInterface, generate_training_labels
import numpy as np

# Initialize interface
interface = ScoringNetworkInterface(
    grid_size=(30, 30),
    base_config='configs/ai_labeling_config.json',
    num_trials_per_eval=3  # k Monte Carlo trials
)

# Evaluate single candidate
floor_plan = np.zeros((30, 30), dtype=np.float32)
door_config = [
    {'id': 'd1', 'position': 'x15y10', 'type': 'door'},
    {'id': 'e1', 'position': 'x28y28', 'type': 'exit'}
]
result = interface.evaluate_candidate(floor_plan, door_config)
score = result['survival_rate'] - result['steps'] / 1000

# Generate pairwise labels
candidates = [...]  # List of door configs
labels = interface.generate_candidate_labels(
    floor_plan=floor_plan,
    candidate_pool=candidates,
    num_pairs=100,
    pair_selection='mixed'  # 'random', 'hard', or 'mixed'
)

# Labels format: (config_a, config_b, label, score_a, score_b)
# label = 1 if A > B, 0 if B > A, None if ambiguous
for config_a, config_b, label, score_a, score_b in labels:
    print(f"A: {score_a:.4f}, B: {score_b:.4f}, Winner: {'A' if label == 1 else 'B'}")

# Generate training dataset
floor_plans = [...]  # List of floor plan arrays
stats = generate_training_labels(
    simulator_interface=interface,
    floor_plans=floor_plans,
    candidates_per_plan=50,
    pairs_per_plan=100,
    output_path='pairwise_labels.jsonl'
)
print(f"Generated {stats['total_labels']} labels")
print(f"Label rate: {stats['label_rate']:.2%}")
```

## Testing

### Import Tests
```bash
python -c "from optimized_d_star_lite import OptimizedDStarLite; print('OK')"
python -c "from fast_fire import FastFireModel; print('OK')"
python -c "from fast_simulation import FastEvacuationSim; print('OK')"
python -c "from pairwise_ranking_interface import ScoringNetworkInterface; print('OK')"
```

### Simple Functionality Tests
```bash
python test_phase2_simple.py
```

### Full Performance Benchmark
```bash
python test_phase2_performance.py
```

Expected output:
- Single simulation: 40-100ms
- Monte Carlo (10 runs): ~1 second total
- Throughput: 36,000-90,000 sims/hour
- Pairwise label generation: ~2-5 seconds per pair (6 simulations @ 3 trials each)

## Integration with Existing Codebase

Phase 2 components are **standalone** and don't modify existing code:
- Original `simulation.py` remains unchanged
- Original D* Lite in `d_star_lite/` remains unchanged
- Original fire models remain unchanged

You can use Phase 2 for:
- **Label generation** for AI training (use `ScoringNetworkInterface`)
- **Fast evaluation** of door configurations (use `FastEvacuationSim`)
- **Prototyping** new features with faster iteration

Keep using original simulation for:
- **Visualization** and demos
- **Full physics** accuracy when needed
- **Backward compatibility** with existing configs

## Next Steps

### Week 3-4: Generate Training Dataset

Generate 5K-10K pairwise labels:

```python
from pairwise_ranking_interface import ScoringNetworkInterface, generate_training_labels
from candidate_generator import generate_door_candidates  # To be implemented
import numpy as np

# Create diverse floor plans
floor_plans = [
    create_office_layout(),
    create_hospital_layout(),
    create_school_layout(),
    # ... 10-20 different floor plans
]

# Generate training labels
interface = ScoringNetworkInterface(
    grid_size=(30, 30),
    num_trials_per_eval=3
)

stats = generate_training_labels(
    simulator_interface=interface,
    floor_plans=floor_plans,
    candidates_per_plan=50,
    pairs_per_plan=100,
    output_path='training_data/pairwise_labels.jsonl'
)

print(f"Generated {stats['total_labels']} labels")
print(f"Estimated training time for 10K pairs: {10000/100 * 5:.0f} hours")
# With Phase 2: ~8 hours instead of 80+ hours with Phase 1
```

### Week 5-6: Train Scoring Network

Train CNN to predict quality scores (ML team task):

```python
# Pseudocode - actual implementation in PyTorch/TensorFlow
model = ScoringCNN(input_channels=4)  # walls, doors, exits, connectivity
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = PairwiseRankingLoss()

for epoch in range(100):
    for config_a, config_b, label in dataloader:
        score_a = model(config_a)
        score_b = model(config_b)
        loss = loss_fn(score_a, score_b, label)
        loss.backward()
        optimizer.step()
```

Monitor:
- **Spearman correlation** between model scores and simulator scores (target: >0.7)
- **Top-k recall** (target: >60% for k=20)

### Week 7-8: AI-Guided Search (Phase 3)

Use trained model to accelerate search by 10-100x:

```python
from scoring_network_plugin import ScoringNetworkPlugin
from ai_search_pipeline import AIGuidedSearch

# Load trained model
scorer = ScoringNetworkPlugin(model_path='models/scoring_network.pt')
simulator = ScoringNetworkInterface()

# Search pipeline
search = AIGuidedSearch(
    scoring_network=scorer,
    simulator_interface=simulator,
    top_k=20  # Only validate top-20
)

# Generate large candidate pool
candidates = generate_door_candidates(floor_plan, num_candidates=1000)

# AI-guided search: score 1000, validate only 20
results = search.search(floor_plan, candidates)
best_design = results[0]

print(f"Best design: {best_design.door_config}")
print(f"Simulator score: {best_design.simulator_score}")
print(f"Speedup: {1000/20}x fewer simulator calls")
```

## Performance Targets

| Metric | Phase 1 | Phase 2 | Target |
|--------|---------|---------|--------|
| Time per sim | 0.4-0.6s | 0.04-0.1s | ✅ 5-10x |
| Memory per sim | 30MB | 10MB | ✅ 3x reduction |
| Sims per hour | 6,000-9,000 | 36,000-90,000 | ✅ 6-10x |
| Label gen (5K pairs) | ~40 hours | ~4 hours | ✅ 10x faster |

## Troubleshooting

### Simulation too slow
- Reduce grid size (30x30 instead of 50x50)
- Increase `fire_update_interval` to 8-10
- Use `deterministic_fire=True` for reproducibility
- Enable spatial filtering in `update_environment()`

### High memory usage
- Use `batch_evaluate()` with limited `num_workers`
- Clear results after processing
- Use smaller `num_trials_per_eval` (3 instead of 5)

### Poor label quality
- Increase `margin` in pairwise labeling (0.1 instead of 0.05)
- Use more Monte Carlo trials (5 instead of 3)
- Filter out ambiguous pairs (`label = None`)
- Ensure diverse floor plans in training set

### High survival rates / Low variance between runs
If you're seeing 99%+ survival rates with <5% variance:
- Increase `fire_spread_rate` to 0.5-0.7 (faster spread)
- Add `fire_discovery_delay` of 30-60 steps (delayed detection)
- Enable `fire_damage_threshold` at 10.0 (cumulative damage counts)
- Place fires strategically near exits/chokepoints
- Increase `fire_intensity_growth` to 0.75-1.0

Example config for meaningful variance:
```json
{
  "fire_spread_rate": 0.6,
  "fire_intensity_growth": 1.0,
  "fire_discovery_delay": 40,
  "fire_damage_threshold": 10.0
}
```

## Dependencies

Phase 2 only requires:
```bash
pip install numpy  # Already installed
```

Optional for Phase 3:
```bash
pip install torch scipy  # Neural network and correlation metrics
```

## References

- Main roadmap: `AI-Guided_Design_Optimization.md`
- Phase 1 implementation: `ai_labeling_wrapper.py`
- Original simulation: `simulation.py`
- Performance analysis: `PERFORMANCE_RESULTS.md`
