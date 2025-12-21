# Training Data Generation Plan for Pairwise Ranking Model

## Overview

This plan outlines a systematic approach to generate diverse pairwise training data that ensures model generality. The key insight: **generality comes from diversity in floor plan structure, not just door placement**.

---

## 1. Diversity Dimensions

To ensure the ranking model generalizes well, training data must cover these orthogonal dimensions:

### 1.1 Floor Plan Structure
| Dimension | Range | Rationale |
|-----------|-------|-----------|
| Map size | 20x20 to 80x80 | Small offices to large buildings |
| Room count | 1-12 rooms | Open plan to compartmentalized |
| Room shapes | Rectangular, L-shaped, irregular | Architectural variety |
| Corridor width | 1-3 cells | Bottleneck variation |
| Obstacle density | 5%-25% of area | Furniture, columns, etc. |

### 1.2 Exit Configuration
| Dimension | Range | Rationale |
|-----------|-------|-----------|
| Exit count | 1-4 exits | Redundancy variation |
| Exit placement | Corners, edges, distributed | Accessibility patterns |
| Exit width | 1-2 cells | Throughput capacity |

### 1.3 Agent Configuration
| Dimension | Range | Rationale |
|-----------|-------|-----------|
| Agent count | 10-200 agents | Crowding effects |
| Agent density | 0.5%-5% of passable cells | Sparse to dense |
| Start distribution | Clustered, uniform, room-based | Realistic scenarios |

### 1.4 Fire Configuration
| Dimension | Range | Rationale |
|-----------|-------|-----------|
| Fire count | 1-3 fires | Single vs multi-source |
| Fire position | Center, corner, near exit | Blocking patterns |
| Spread rate | 0.2-0.6 | Slow to aggressive |
| Discovery delay | 0-20 steps | Early vs late detection |

---

## 2. Floor Plan Generation Strategy

### 2.1 Procedural Generation Methods

```
Method 1: BSP (Binary Space Partitioning)
├── Recursively divide space into rooms
├── Connect rooms with corridors
├── Good for: Regular office layouts
└── Parameters: min_room_size, split_ratio

Method 2: Cellular Automata
├── Start with random noise
├── Apply smoothing rules
├── Good for: Organic/irregular layouts
└── Parameters: iterations, birth/death thresholds

Method 3: Template-Based
├── Define room templates (L-shape, T-junction, etc.)
├── Combine templates with corridors
├── Good for: Realistic architectural patterns
└── Parameters: template_library, connection_rules

Method 4: Grid-Based Rooms
├── Divide into grid sectors
├── Randomly merge adjacent sectors
├── Good for: Simple variation with control
└── Parameters: grid_divisions, merge_probability
```

### 2.2 Recommended Distribution

| Method | Proportion | Purpose |
|--------|------------|---------|
| BSP | 40% | Structured office/building layouts |
| Grid-Based | 30% | Controlled variation |
| Template-Based | 20% | Realistic patterns |
| Cellular Automata | 10% | Edge cases, stress testing |

---

## 3. Data Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    MASTER PROCESS (Coordinator)                  │
├─────────────────────────────────────────────────────────────────┤
│  1. Generate floor plan batch (1000 plans)                      │
│  2. Distribute to worker pool                                    │
│  3. Collect results and construct pairs                         │
│  4. Save to JSONL shards                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Worker 1-32   │ │   Worker 33-64  │ │   Worker 65-96  │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ For each plan:  │ │ For each plan:  │ │ For each plan:  │
│ - Gen N configs │ │ - Gen N configs │ │ - Gen N configs │
│ - Run M trials  │ │ - Run M trials  │ │ - Run M trials  │
│ - Return scores │ │ - Return scores │ │ - Return scores │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 3.1 Pipeline Stages

**Stage 1: Floor Plan Generation (CPU-light)**
- Generate 5,000-10,000 unique floor plans
- Store as compressed NumPy arrays
- ~1-2 hours single-threaded

**Stage 2: Configuration Sampling (CPU-light)**
- For each floor plan: generate 20-50 door configurations
- Sample 5-10 agent start distributions
- Sample 3-5 fire scenarios
- Combinatorial: ~100-250 scenarios per floor plan

**Stage 3: Simulation Evaluation (CPU-heavy)**
- Run Monte Carlo trials (3-5 trials per scenario)
- Target: 1M-5M total simulations
- EPYC utilization: 128+ parallel workers

**Stage 4: Pair Construction (CPU-light)**
- Within each floor plan: create pairwise comparisons
- Cross-floor plan: create transfer pairs (harder)
- Label based on survival_rate primary, steps secondary

---

## 4. Pair Construction Strategy

### 4.1 Within-Plan Pairs (80% of data)
Compare different configurations on the **same** floor plan.
- Eliminates floor plan structure as confounding variable
- Model learns: "given this structure, which config is better?"

```
Floor Plan A:
  Config 1 (score: 0.95) vs Config 2 (score: 0.72) → Label: 1
  Config 1 (score: 0.95) vs Config 3 (score: 0.93) → Label: 1 (close)
  Config 2 (score: 0.72) vs Config 4 (score: 0.88) → Label: 0
```

### 4.2 Cross-Plan Pairs (20% of data)
Compare configurations across **different** floor plans.
- Tests whether model learns general quality signals
- Harder but improves generalization

```
Floor Plan A, Config 1 (score: 0.95) vs Floor Plan B, Config 1 (score: 0.82)
→ Label: 1 (A's config is better overall)
```

### 4.3 Pair Selection Strategies

| Strategy | Description | Proportion |
|----------|-------------|------------|
| Random | Uniform random pairing | 50% |
| Hard Negative | Similar scores (±5%) | 30% |
| Easy | Large score difference (>20%) | 20% |

### 4.4 Label Confidence

```python
def compute_label(score_a, score_b, margin=0.05):
    diff = score_a - score_b
    if abs(diff) < margin:
        return None  # Ambiguous, discard or soft-label
    return 1 if diff > 0 else 0
```

For soft labels (optional):
```python
def soft_label(score_a, score_b, temperature=0.1):
    return 1 / (1 + exp(-(score_a - score_b) / temperature))
```

---

## 5. EPYC Server Optimization

### 5.1 Hardware Assumptions
- AMD EPYC: 64-128 cores (128-256 threads)
- Memory: 256GB-512GB RAM
- Storage: NVMe SSD for I/O

### 5.2 Parallelization Strategy

```python
# Hierarchical parallelization
NUM_FLOOR_PLAN_WORKERS = 16      # Parallel floor plan processing
SIMS_PER_WORKER = 8              # Simulations per worker process
TOTAL_PARALLEL_SIMS = 128        # 16 * 8 = 128 concurrent simulations

# Memory-efficient batching
BATCH_SIZE = 100                 # Floor plans per batch
SAVE_INTERVAL = 10               # Save results every N batches
```

### 5.3 Recommended Configuration

```bash
# Generate training data with full EPYC utilization
python generate_training_data.py \
    --num-floor-plans 5000 \
    --configs-per-plan 30 \
    --trials-per-config 5 \
    --pairs-per-plan 200 \
    --workers 120 \
    --output-dir ./training_data \
    --shard-size 100000
```

### 5.4 Expected Performance

| Metric | Estimate |
|--------|----------|
| Simulations/hour | 50,000-80,000 (Phase 2) |
| Total simulations | 2.25M (5000 × 30 × 5 × 3 trials) |
| Generation time | ~30-45 hours |
| Training pairs | ~1M pairs |
| Storage | ~5-10 GB (compressed JSONL) |

---

## 6. Data Format

### 6.1 Floor Plan Storage

```json
// floor_plans.h5 (HDF5 for efficient array storage)
{
  "plan_0000": {
    "grid": [[...2D array...]],
    "metadata": {
      "size": [40, 40],
      "room_count": 5,
      "generation_method": "bsp",
      "obstacle_density": 0.12
    }
  }
}
```

### 6.2 Pairwise Labels (JSONL shards)

```jsonl
{"plan_id": "plan_0000", "config_a": [{"id":"e1","position":"x5y0","type":"exit"},...], "config_b": [...], "label": 1, "score_a": 0.923, "score_b": 0.845, "agents": 50, "fire_pos": ["x20y20"]}
{"plan_id": "plan_0000", "config_a": [...], "config_b": [...], "label": 0, "score_a": 0.756, "score_b": 0.891, "agents": 50, "fire_pos": ["x20y20"]}
```

### 6.3 Metadata Tracking

```json
// generation_metadata.json
{
  "total_floor_plans": 5000,
  "total_pairs": 1000000,
  "generation_params": {
    "size_range": [20, 80],
    "agent_range": [10, 200],
    "trials_per_config": 5
  },
  "diversity_stats": {
    "size_distribution": {"20-30": 1000, "30-50": 2500, ...},
    "room_count_distribution": {...},
    "agent_count_distribution": {...}
  },
  "label_distribution": {
    "label_1": 485000,
    "label_0": 485000,
    "discarded_ambiguous": 30000
  }
}
```

---

## 7. Quality Control

### 7.1 Validation Checks

1. **Floor Plan Validity**
   - All agents can reach at least one exit (pathfinding check)
   - No isolated rooms without exits
   - Minimum passable area threshold

2. **Simulation Validity**
   - No crashes or timeouts
   - Reasonable evacuation times (< max_steps)
   - Non-zero survival rates

3. **Label Validity**
   - Balanced label distribution (45-55% each class)
   - Score variance across configs (not all same)
   - Cross-validation consistency

### 7.2 Holdout Sets

```
Training:   70% of floor plans (with all their pairs)
Validation: 15% of floor plans
Test:       15% of floor plans

Important: Split by floor plan, not by pairs!
           Prevents data leakage.
```

### 7.3 Diversity Verification

```python
def verify_diversity(dataset):
    # Check coverage of all dimensions
    assert len(set(d['plan_size'] for d in dataset)) >= 10
    assert len(set(d['agent_count'] for d in dataset)) >= 15
    assert len(set(d['exit_count'] for d in dataset)) >= 4

    # Check balance
    labels = [d['label'] for d in dataset]
    assert 0.45 < sum(labels) / len(labels) < 0.55
```

---

## 8. Quick Start Guide

### 8.1 Test Run (Local Machine)

```bash
# Quick test with minimal data (5-10 minutes)
python generate_training_data.py \
    --num-floor-plans 10 \
    --configs-per-plan 5 \
    --trials-per-config 3 \
    --pairs-per-plan 20 \
    --workers 4 \
    --output-dir ./test_data
```

### 8.2 Pilot Run (Verify Quality)

```bash
# Medium run to verify data quality (2-4 hours)
python generate_training_data.py \
    --num-floor-plans 100 \
    --configs-per-plan 20 \
    --trials-per-config 5 \
    --pairs-per-plan 100 \
    --workers 16 \
    --output-dir ./pilot_data
```

### 8.3 Full Production Run (EPYC Server)

```bash
# Full dataset generation (30-45 hours on 128-core EPYC)
python generate_training_data.py \
    --num-floor-plans 5000 \
    --configs-per-plan 30 \
    --trials-per-config 5 \
    --pairs-per-plan 200 \
    --workers 120 \
    --output-dir ./training_data \
    --seed 42

# Resume from checkpoint if interrupted
python generate_training_data.py \
    --resume ./training_data/checkpoint.json
```

### 8.4 Output Structure

```
training_data/
├── train_pairs.jsonl          # 70% of pairs (by floor plan)
├── val_pairs.jsonl            # 15% of pairs
├── test_pairs.jsonl           # 15% of pairs
├── metadata.json              # Config, stats, validation report
├── checkpoint.json            # For resume capability
└── floor_plans/               # Saved floor plan grids
    ├── plan_00000.npz
    ├── plan_00001.npz
    └── ...
```

---

## 9. Implemented Files

All components have been implemented:

| File | Description | Status |
|------|-------------|--------|
| `generate_training_data.py` | Main orchestrator with parallel execution | Done |
| `floor_plan_generator.py` | BSP, Grid, Template, Cellular generators | Done |
| `diversity_sampler.py` | Stratified sampling across dimensions | Done |
| `pair_constructor.py` | Pairwise label construction | Done |
| `data_validator.py` | Quality checks and split validation | Done |

### Component Usage

```python
# Floor plan generation
from floor_plan_generator import FloorPlanGenerator
generator = FloorPlanGenerator(seed=42)
plans = generator.generate_batch(100, size_range=(20, 80))

# Diversity sampling
from diversity_sampler import DiversitySampler, AgentPlacer, FirePlacer
sampler = DiversitySampler(seed=42)
scenarios = sampler.sample_scenarios_for_plan(plan_id=0, floor_plan_size=(40, 40))

# Pair construction
from pair_constructor import PairConstructor, SimulationResult
constructor = PairConstructor(margin=0.05, seed=42)
pairs = constructor.construct_pairs(results, num_pairs=1000, strategy='mixed')

# Validation
from data_validator import DataValidator, create_dataset_splits
validator = DataValidator()
train, val, test = create_dataset_splits(pairs, train_ratio=0.7, seed=42)
report = validator.validate_dataset(train, val, test)
```

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Simulation crashes | Wrap in try-catch, log failures, skip invalid configs |
| Memory exhaustion | Process in batches, stream results to disk |
| Unbalanced labels | Stratified sampling, reject pairs with ambiguous margins |
| Poor generalization | Explicit diversity constraints, held-out plan testing |
| Long generation time | Checkpointing, resume capability, progress monitoring |

---

## 11. Model Training Recommendations

### 11.1 Pairwise Ranking Model Architecture

```python
# Recommended: Siamese CNN for floor plan comparison
class PairwiseRankingModel(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(
            # Floor plan encoder (shared weights)
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, 256)
        )

        # Comparison head
        self.comparator = nn.Sequential(
            nn.Linear(512, 128),  # Concatenated features
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, plan_a, plan_b):
        feat_a = self.encoder(plan_a)
        feat_b = self.encoder(plan_b)
        combined = torch.cat([feat_a, feat_b], dim=1)
        return self.comparator(combined)
```

### 11.2 Training Loop

```python
# RankNet-style loss
def ranknet_loss(pred, label):
    return F.binary_cross_entropy(pred, label.float())

# Training
for epoch in range(epochs):
    for batch in dataloader:
        plan_a, plan_b, label = batch
        pred = model(plan_a, plan_b)
        loss = ranknet_loss(pred, label)
        loss.backward()
        optimizer.step()
```

### 11.3 Data Loading

```python
import json
import numpy as np

class PairwiseDataset(Dataset):
    def __init__(self, pairs_file, floor_plans_dir):
        self.pairs = []
        with open(pairs_file) as f:
            for line in f:
                self.pairs.append(json.loads(line))
        self.floor_plans_dir = floor_plans_dir

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load floor plans
        plan_a = np.load(f"{self.floor_plans_dir}/plan_{pair['floor_plan_id_a']:05d}.npz")['grid']
        plan_b = np.load(f"{self.floor_plans_dir}/plan_{pair['floor_plan_id_b']:05d}.npz")['grid']

        # Overlay exit configurations on plans
        plan_a = self.overlay_config(plan_a, pair['config_a'])
        plan_b = self.overlay_config(plan_b, pair['config_b'])

        return torch.tensor(plan_a), torch.tensor(plan_b), pair['label']
```

---

## Summary

**Target Dataset:**
- 5,000 unique floor plans
- 150,000 unique (floor_plan, config, agents, fire) scenarios
- 1,000,000+ pairwise comparisons
- ~30-45 hours generation time on EPYC

**Key Generality Guarantees:**
1. Diverse floor plan structures (4 generation methods)
2. Varied sizes, room counts, obstacle densities
3. Multiple agent counts and distributions
4. Various fire scenarios
5. Cross-plan pairs for transfer learning
6. Held-out floor plan test set (not just held-out pairs)

---

## Quick Reference

```bash
# Test locally
python generate_training_data.py --num-floor-plans 10 --workers 4 --output-dir ./test

# Full EPYC run
python generate_training_data.py --num-floor-plans 5000 --workers 120 --output-dir ./data

# Resume interrupted run
python generate_training_data.py --resume ./data/checkpoint.json
```
