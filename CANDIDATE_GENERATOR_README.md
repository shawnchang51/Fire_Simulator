# Candidate Generator for Door Configurations

**Part of Phase 1: AI-Guided Design Optimization**

The candidate generator creates diverse door configuration candidates for pairwise comparison labeling. It supports both random and rule-based placement strategies to explore the design space effectively.

## Quick Start

```python
from candidate_generator import generate_door_candidates
import numpy as np

# Load your floor plan (2D numpy array, -2=wall, 0=empty)
floor_plan = np.array([[...]])

# Generate 50 candidates
candidates = generate_door_candidates(
    floor_plan=floor_plan,
    num_candidates=50,
    num_doors_range=(2, 4),  # 2-4 internal doors per candidate
    num_exits_range=(1, 2),  # 1-2 exits per candidate
    min_door_spacing=5,      # Min Manhattan distance between doors
    random_ratio=0.5,        # 50% random, 50% rule-based
    seed=42                  # For reproducibility
)

# Each candidate is a list of door dicts:
# [
#   {"id": "d1", "position": "x15y20", "type": "door"},
#   {"id": "e1", "position": "x3y10", "type": "exit"},
#   ...
# ]
```

## Features

### 1. Random Placement
- Uniformly samples valid wall positions
- Ensures doors are on walls adjacent to passable cells
- Maintains minimum spacing constraints

### 2. Rule-Based Strategies

#### Boundary Focused
Prioritizes room boundaries for internal doors and perimeter for exits.
```python
generator.generate_rule_based_candidate(
    num_doors=3,
    num_exits=2,
    strategy='boundary_focused'
)
```

#### Distributed
Evenly distributes doors across floor plan grid sectors.
```python
generator.generate_rule_based_candidate(
    num_doors=3,
    num_exits=2,
    strategy='distributed'
)
```

#### Corner Exits
Places exits in building corners, doors on room boundaries.
```python
generator.generate_rule_based_candidate(
    num_doors=3,
    num_exits=2,
    strategy='corner_exits'
)
```

### 3. Automatic Floor Plan Analysis
- **Room detection**: Uses connected components to identify separate rooms
- **Boundary identification**: Finds walls between different rooms
- **Perimeter detection**: Identifies exit-suitable locations on building edge
- **Validation**: Checks connectivity and spacing constraints

## Advanced Usage

### Custom Generator Instance

For fine-grained control:

```python
from candidate_generator import CandidateGenerator

generator = CandidateGenerator(
    floor_plan=floor_plan,
    min_door_spacing=5,
    wall_value=-2,
    seed=42
)

# Inspect analysis
print(f"Valid wall positions: {len(generator.valid_wall_positions)}")
print(f"Rooms identified: {len(generator.rooms)}")
print(f"Room boundaries: {len(generator.room_boundaries)}")
print(f"Perimeter positions: {len(generator.perimeter_positions)}")

# Generate with specific strategy
candidate = generator.generate_rule_based_candidate(
    num_doors=3,
    num_exits=2,
    strategy='boundary_focused'
)
```

### Generate Candidate Pool

```python
# Generate diverse pool with controlled parameters
candidates = generator.generate_candidate_pool(
    num_candidates=100,
    num_doors_range=(2, 5),
    num_exits_range=(1, 3),
    random_ratio=0.5
)
```

## Integration with Pairwise Labeling

Generate candidate pairs for pairwise comparison:

```python
import random

# Generate candidate pool
candidates = generate_door_candidates(
    floor_plan=floor_plan,
    num_candidates=100,
    seed=42
)

# Sample pairs for labeling
pairs = []
for _ in range(50):
    config_a, config_b = random.sample(candidates, 2)
    pairs.append((config_a, config_b))

# Now evaluate each pair with simulator to generate labels
# See ai_labeling_wrapper.py for evaluation
```

## Output Format

Each candidate is a list of door configuration dictionaries:

```python
[
    {
        "id": "d1",           # Unique identifier
        "position": "x15y20", # Grid position (col=15, row=20)
        "type": "door"        # "door" or "exit"
    },
    {
        "id": "e1",
        "position": "x3y10",
        "type": "exit"
    }
]
```

This format is directly compatible with `SimulationConfig.from_json()`.

## Configuration Guidelines

### Candidate Pool Size
- **Small experiments**: 20-50 candidates
- **Training dataset**: 100-500 candidates per floor plan
- **Large-scale labeling**: 1000+ candidates

### Door/Exit Ranges
- **Minimum viable**: 2-3 doors, 1 exit
- **Typical buildings**: 3-5 doors, 1-2 exits
- **Complex layouts**: 5-10 doors, 2-3 exits

### Spacing Constraints
- **Dense placement**: `min_door_spacing=3`
- **Normal spacing**: `min_door_spacing=5`
- **Sparse placement**: `min_door_spacing=7-10`

### Random vs Rule-Based Ratio
- **Pure exploration**: `random_ratio=1.0`
- **Balanced**: `random_ratio=0.5`
- **Structured designs**: `random_ratio=0.3`

## Examples

See comprehensive examples in `examples/candidate_generator_demo.py`:

```bash
cd examples
python candidate_generator_demo.py
```

Demos include:
1. Basic candidate generation
2. Advanced strategy control
3. Pairwise labeling workflow
4. Saving/loading candidates

## Testing

Run tests to verify functionality:

```bash
# Basic test
python candidate_generator.py

# Comprehensive test with example floor plan
python test_candidate_generator.py
```

## Dependencies

Required:
- `numpy`: Array operations
- `scipy`: Connected components for room detection

Install:
```bash
pip install numpy scipy
```

## How It Works

1. **Floor Plan Analysis**
   - Identifies valid wall positions (walls adjacent to passable cells)
   - Detects rooms using connected component labeling
   - Finds room boundaries (walls between different rooms)
   - Identifies perimeter positions for exits

2. **Constraint Validation**
   - Checks minimum spacing between doors
   - Ensures doors connect passable areas
   - Validates exit accessibility

3. **Generation Strategies**
   - **Random**: Samples uniformly from valid positions
   - **Rule-based**: Uses heuristics (boundaries, distribution, corners)
   - **Hybrid**: Combines both for diversity

## Performance

- Generation speed: ~100-1000 candidates per second
- Memory efficient: O(grid_size) for floor plan analysis
- Cached analysis: Reuse generator instance for multiple candidates

## Troubleshooting

### "No valid wall positions found"
- Check floor plan format (-2 for walls, 0 for empty)
- Ensure walls have adjacent passable cells
- Verify floor plan is not all walls or all empty

### "Not enough perimeter positions"
- Some floor plans have limited exit locations
- Reduce `num_exits` or increase perimeter zone size
- Check that building has exterior walls

### "Candidates have no doors/exits"
- Increase `num_candidates` to get more valid samples
- Reduce `min_door_spacing` constraint
- Check floor plan connectivity

## Next Steps

After generating candidates:

1. **Evaluate with Simulator**: Use `ai_labeling_wrapper.py` to run Monte Carlo trials
2. **Generate Pairwise Labels**: Compare candidate pairs using simulation results
3. **Train Scoring Network**: Use labels to train pairwise ranking model
4. **AI-Guided Search**: Use trained model to accelerate design search

See `AI-Guided_Design_Optimization.md` for the complete pipeline.

## Citation

If you use this candidate generator in your research:

```
Fire Evacuation Simulator - AI-Guided Design Optimization
Candidate Generator for Door Configuration Search
Phase 1: Conservative Optimizations
```

## License

Part of the Fire Evacuation Simulator project.
