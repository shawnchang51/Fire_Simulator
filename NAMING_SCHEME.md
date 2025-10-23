# Output Naming Scheme

## Problem Solved

When processing multiple configurations in quick succession, they could finish within the same second, leading to **timestamp collisions**.

## Solution

Each output folder now has a **unique 3-digit counter** based on queue position:

```
{config_name}_{timestamp}_{queue_index}
```

### Example:

```
monte_carlo_results/
├── scenario_A_20250123_143052_001/   ← 1st in queue
├── scenario_B_20250123_143052_002/   ← 2nd in queue
├── scenario_C_20250123_143052_003/   ← 3rd in queue
└── queue_summary_20250123_143052.json
```

Even if all 3 configs finish within the same second, they get unique folder names!

## Benefits

✅ **No Collisions**: Counter ensures uniqueness even with identical timestamps
✅ **Sorted Order**: Alphabetical sorting = chronological processing order
✅ **Easy Tracking**: Know exactly which config was processed when
✅ **Safety Net**: Additional duplicate detection (`_dup1`, `_dup2`) as fallback

## Naming Components

| Component | Example | Description |
|-----------|---------|-------------|
| Config name | `scenario_small_fire` | From JSON filename (without .json) |
| Timestamp | `20250123_143052` | YYYYMMdd_HHMMSS when processing started |
| Queue index | `001` | Zero-padded position in queue (1st, 2nd, 3rd...) |
| Duplicate suffix | `_dup1` | Only if folder somehow already exists (rare) |

## Full Example

**Input:**
```
configs/
├── baseline.json          ← Will be _001
├── stress_test.json       ← Will be _002
└── max_capacity.json      ← Will be _003
```

**Output:**
```
monte_carlo_results/
├── baseline_20250123_143052_001/
├── stress_test_20250123_143052_002/
├── max_capacity_20250123_143053_003/  ← Note: Even if timestamp changed!
└── queue_summary_20250123_143052.json
```

The counter **always increments** regardless of timestamp, ensuring perfect tracking of processing order!
