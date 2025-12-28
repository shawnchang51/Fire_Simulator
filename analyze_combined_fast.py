#!/usr/bin/env python3
"""
Comprehensive analysis of combined_fast training data
Answers questions about dataset structure, pair labels, and diversity
"""

import json
import os
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

DATA_DIR = Path("./combined_fast")

def load_jsonl(filepath):
    """Load JSONL file and return list of records"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def extract_exits_from_config(config):
    """Extract exit positions from door_config"""
    exits = []
    if 'door_config' in config:
        for item in config['door_config']:
            if item.get('type') == 'exit':
                exits.append(item['position'])
    return tuple(sorted(exits))

def config_to_hash(config):
    """Create a hash from door configuration for uniqueness check"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:16]

def analyze_dataset():
    print("=" * 70)
    print("COMBINED_FAST TRAINING DATA ANALYSIS")
    print("=" * 70)
    
    # Load all splits
    print("\nLoading data files...")
    train_pairs = load_jsonl(DATA_DIR / "train_pairs.jsonl")
    val_pairs = load_jsonl(DATA_DIR / "val_pairs.jsonl")
    test_pairs = load_jsonl(DATA_DIR / "test_pairs.jsonl")
    
    print(f"  train_pairs: {len(train_pairs)} records")
    print(f"  val_pairs:   {len(val_pairs)} records")
    print(f"  test_pairs:  {len(test_pairs)} records")
    print(f"  TOTAL:       {len(train_pairs) + len(val_pairs) + len(test_pairs)} records")
    
    # =========================================================================
    # A. DATASET STRUCTURE
    # =========================================================================
    print("\n" + "=" * 70)
    print("A. DATASET STRUCTURE")
    print("=" * 70)
    
    # Collect all map IDs from each split
    def get_map_ids(pairs):
        """Get unique map IDs from pairs (considering both sides)"""
        map_ids = set()
        for p in pairs:
            map_ids.add(p['floor_plan_id_a'])
            map_ids.add(p['floor_plan_id_b'])
        return map_ids
    
    train_maps = get_map_ids(train_pairs)
    val_maps = get_map_ids(val_pairs)
    test_maps = get_map_ids(test_pairs)
    all_maps = train_maps | val_maps | test_maps
    
    print(f"\n1. Unique Maps:")
    print(f"   num_unique_maps = {len(all_maps)}")
    print(f"   train_unique_maps = {len(train_maps)}")
    print(f"   val_unique_maps = {len(val_maps)}")
    print(f"   test_unique_maps = {len(test_maps)}")
    
    # Count pairs per map (considering a pair uses map_id_a and map_id_b)
    all_pairs = train_pairs + val_pairs + test_pairs
    pairs_per_map = defaultdict(int)
    for p in all_pairs:
        # Count once per pair, using the primary map (floor_plan_id_a)
        # Since most pairs have floor_plan_id_a == floor_plan_id_b
        pairs_per_map[p['floor_plan_id_a']] += 1
    
    pairs_counts = list(pairs_per_map.values())
    
    print(f"\n2. Pairs per Map Distribution:")
    print(f"   min_pairs_per_map    = {min(pairs_counts)}")
    print(f"   median_pairs_per_map = {int(np.median(pairs_counts))}")
    print(f"   mean_pairs_per_map   = {np.mean(pairs_counts):.2f}")
    print(f"   max_pairs_per_map    = {max(pairs_counts)}")
    print(f"   std_pairs_per_map    = {np.std(pairs_counts):.2f}")
    
    # Check map disjointness
    train_val_overlap = train_maps & val_maps
    train_test_overlap = train_maps & test_maps
    val_test_overlap = val_maps & test_maps
    
    map_disjoint = len(train_val_overlap) == 0 and len(train_test_overlap) == 0
    
    print(f"\n3. Map Disjoint Split Analysis:")
    print(f"   train ∩ val  overlap = {len(train_val_overlap)} maps")
    print(f"   train ∩ test overlap = {len(train_test_overlap)} maps")
    print(f"   val ∩ test overlap   = {len(val_test_overlap)} maps")
    print(f"   map_disjoint_split   = {map_disjoint}")
    
    if train_val_overlap:
        print(f"   Sample overlapping maps (train∩val): {list(train_val_overlap)[:5]}")
    if train_test_overlap:
        print(f"   Sample overlapping maps (train∩test): {list(train_test_overlap)[:5]}")
    
    # =========================================================================
    # B. PAIR LABEL AND DIFFICULTY
    # =========================================================================
    print("\n" + "=" * 70)
    print("B. PAIR LABEL AND DIFFICULTY")
    print("=" * 70)
    
    # Analyze label type
    sample = train_pairs[0]
    label_value = sample.get('label')
    score_a = sample.get('score_a')
    score_b = sample.get('score_b')
    
    print(f"\n1. Label Type Analysis:")
    print(f"   Sample label value: {label_value} (type: {type(label_value).__name__})")
    print(f"   Has score_a/score_b: {score_a is not None and score_b is not None}")
    
    # Check unique label values
    train_labels = [p['label'] for p in train_pairs]
    unique_labels = set(train_labels)
    print(f"   Unique labels in train: {sorted(unique_labels)}")
    
    if unique_labels == {0, 1}:
        label_type = "binary"
    elif all(isinstance(l, (int, float)) and l not in {0, 1} for l in unique_labels):
        label_type = "cost_diff"
    else:
        label_type = "ranking"
    
    print(f"   label_type = {label_type}")
    
    # Positive/Negative ratio (for binary: label=1 means A is better)
    print(f"\n2. Positive/Negative Ratio (Train):")
    num_positive = sum(1 for p in train_pairs if p['label'] == 1)
    num_negative = sum(1 for p in train_pairs if p['label'] == 0)
    total = len(train_pairs)
    
    print(f"   num_positive_pairs = {num_positive}")
    print(f"   num_negative_pairs = {num_negative}")
    print(f"   positive_ratio     = {num_positive / total:.4f} ({num_positive / total * 100:.2f}%)")
    print(f"   negative_ratio     = {num_negative / total:.4f} ({num_negative / total * 100:.2f}%)")
    
    # Hard pairs (near decision boundary)
    # Using multiple thresholds
    print(f"\n3. Hard Pairs Analysis (near decision boundary):")
    print(f"   Definition: |score_a - score_b| < threshold")
    
    score_diffs = []
    for p in train_pairs:
        if 'score_a' in p and 'score_b' in p:
            diff = abs(p['score_a'] - p['score_b'])
            score_diffs.append(diff)
    
    if score_diffs:
        score_diffs = np.array(score_diffs)
        print(f"\n   Score difference statistics:")
        print(f"     min   = {np.min(score_diffs):.6f}")
        print(f"     25%   = {np.percentile(score_diffs, 25):.4f}")
        print(f"     50%   = {np.percentile(score_diffs, 50):.4f}")
        print(f"     75%   = {np.percentile(score_diffs, 75):.4f}")
        print(f"     max   = {np.max(score_diffs):.4f}")
        print(f"     mean  = {np.mean(score_diffs):.4f}")
        print(f"     std   = {np.std(score_diffs):.4f}")
        
        # Use label_confidence if available
        confidences = [p.get('label_confidence', 1.0) for p in train_pairs]
        low_conf = sum(1 for c in confidences if c < 0.3)
        
        print(f"\n   Hard pairs by score_diff threshold:")
        for threshold in [0.05, 0.10, 0.15, 0.20]:
            hard_count = np.sum(score_diffs < threshold)
            print(f"     |score_diff| < {threshold}: {hard_count} ({hard_count/len(score_diffs)*100:.2f}%)")
        
        # Using relative threshold (5% of score range)
        score_range = np.max(score_diffs)
        rel_threshold = score_range * 0.05
        hard_count_rel = np.sum(score_diffs < rel_threshold)
        print(f"\n   Relative threshold (5% of range = {rel_threshold:.4f}):")
        print(f"     num_hard_pairs   = {hard_count_rel}")
        print(f"     hard_pair_ratio  = {hard_count_rel/len(score_diffs):.4f}")
        
        # Using label_confidence
        print(f"\n   Hard pairs by label_confidence:")
        for conf_thresh in [0.1, 0.2, 0.3, 0.5]:
            low_conf_count = sum(1 for c in confidences if c < conf_thresh)
            print(f"     confidence < {conf_thresh}: {low_conf_count} ({low_conf_count/len(confidences)*100:.2f}%)")
    
    # =========================================================================
    # C. PAIR SOURCE DIVERSITY
    # =========================================================================
    print("\n" + "=" * 70)
    print("C. PAIR SOURCE DIVERSITY")
    print("=" * 70)
    
    # Analyze pair types
    print("\n1. Pair Type Distribution:")
    pair_types = Counter(p.get('pair_type', 'unknown') for p in train_pairs)
    for ptype, count in pair_types.most_common():
        print(f"     {ptype}: {count} ({count/len(train_pairs)*100:.2f}%)")
    
    # Analyze exit sharing
    print("\n2. Exit/Goal Sharing Analysis:")
    
    # Group pairs by their exit configuration
    exit_groups = defaultdict(list)
    for i, p in enumerate(train_pairs):
        exits_a = extract_exits_from_config(p['config_a'])
        exits_b = extract_exits_from_config(p['config_b'])
        map_id = p['floor_plan_id_a']
        # Create a key for this goal/exit combination
        exit_key = (map_id, exits_a, exits_b)
        exit_groups[exit_key].append(i)
    
    group_sizes = [len(g) for g in exit_groups.values()]
    print(f"   Number of unique (map, exit_config) combinations: {len(exit_groups)}")
    print(f"   avg_pairs_sharing_same_goal = {np.mean(group_sizes):.2f}")
    print(f"   max_pairs_sharing_same_goal = {max(group_sizes)}")
    print(f"   median_pairs_sharing_same_goal = {int(np.median(group_sizes))}")
    
    # Same-exit pairs analysis
    same_exit_pairs = [p for p in train_pairs if p.get('pair_type') == 'same_exit']
    print(f"\n   Same-exit pairs: {len(same_exit_pairs)} ({len(same_exit_pairs)/len(train_pairs)*100:.2f}%)")
    
    # Map hash uniqueness (using floor plan files)
    print("\n3. Map Structure Uniqueness:")
    
    floor_plans_dir = DATA_DIR / "floor_plans"
    if floor_plans_dir.exists():
        plan_files = list(floor_plans_dir.glob("*.npz"))
        print(f"   Number of floor plan files: {len(plan_files)}")
        
        # Sample some plans and compute hashes
        map_hashes = {}
        obstacle_densities = []
        
        for plan_file in plan_files[:100]:  # Sample first 100
            try:
                data = np.load(plan_file)
                grid = data['grid'] if 'grid' in data else data[list(data.keys())[0]]
                
                # Compute hash of the grid
                grid_hash = hashlib.md5(grid.tobytes()).hexdigest()
                map_hashes[plan_file.name] = grid_hash
                
                # Compute obstacle density
                if grid.ndim == 2:
                    density = np.sum(grid == 1) / grid.size  # Assuming 1 = obstacle
                    obstacle_densities.append(density)
            except Exception as e:
                pass
        
        unique_hashes = len(set(map_hashes.values()))
        print(f"   Sampled plans: {len(map_hashes)}")
        print(f"   num_unique_map_hashes = {unique_hashes}")
        print(f"   Duplicate maps found: {len(map_hashes) - unique_hashes}")
        
        if obstacle_densities:
            print(f"\n   Obstacle density distribution:")
            print(f"     min    = {min(obstacle_densities):.4f}")
            print(f"     median = {np.median(obstacle_densities):.4f}")
            print(f"     max    = {max(obstacle_densities):.4f}")
            print(f"     std    = {np.std(obstacle_densities):.4f}")
            
            # Bucket by density
            buckets = defaultdict(int)
            for d in obstacle_densities:
                bucket = int(d * 10) / 10  # 0.0, 0.1, 0.2, ...
                buckets[bucket] += 1
            print(f"\n   Density buckets:")
            for bucket in sorted(buckets.keys()):
                print(f"     [{bucket:.1f}-{bucket+0.1:.1f}): {buckets[bucket]}")
    else:
        print(f"   Floor plans directory not found: {floor_plans_dir}")
    
    # Config diversity
    print("\n4. Configuration Diversity (Train):")
    config_hashes = set()
    for p in train_pairs:
        config_hashes.add(config_to_hash(p['config_a']))
        config_hashes.add(config_to_hash(p['config_b']))
    
    print(f"   Unique door/exit configurations: {len(config_hashes)}")
    print(f"   Total configs used (2 per pair): {len(train_pairs) * 2}")
    print(f"   Config reuse ratio: {len(train_pairs) * 2 / len(config_hashes):.2f}x")
    
    # Scenario diversity
    print("\n5. Scenario Diversity (Train):")
    agent_counts = []
    fire_counts = []
    fire_rates = []
    
    for p in train_pairs:
        if 'scenario_a' in p:
            agent_counts.append(p['scenario_a'].get('agent_count', 0))
            fire_counts.append(p['scenario_a'].get('num_fires', 0))
            fire_rates.append(p['scenario_a'].get('fire_spread_rate', 0))
        if 'scenario_b' in p:
            agent_counts.append(p['scenario_b'].get('agent_count', 0))
            fire_counts.append(p['scenario_b'].get('num_fires', 0))
            fire_rates.append(p['scenario_b'].get('fire_spread_rate', 0))
    
    print(f"   Agent count range: [{min(agent_counts)}, {max(agent_counts)}], std={np.std(agent_counts):.2f}")
    print(f"   Fire count range:  [{min(fire_counts)}, {max(fire_counts)}], std={np.std(fire_counts):.2f}")
    print(f"   Fire rate range:   [{min(fire_rates):.3f}, {max(fire_rates):.3f}], std={np.std(fire_rates):.3f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY (Copy-Paste Ready)")
    print("=" * 70)
    
    # Choose a reasonable hard pair threshold (0.10 seems reasonable)
    hard_threshold = 0.10
    num_hard = int(np.sum(score_diffs < hard_threshold)) if len(score_diffs) > 0 else 0
    
    print(f"""
# A. Dataset Structure
num_unique_maps = {len(all_maps)}
min_pairs_per_map = {min(pairs_counts)}
median_pairs_per_map = {int(np.median(pairs_counts))}
max_pairs_per_map = {max(pairs_counts)}
map_disjoint_split = {map_disjoint}

# B. Pair Label and Difficulty  
label_type = "{label_type}"
num_positive_pairs = {num_positive}
num_negative_pairs = {num_negative}
positive_ratio = {num_positive / total:.4f}

# Hard pairs (threshold: |score_diff| < {hard_threshold})
num_hard_pairs = {num_hard}
hard_pair_ratio = {num_hard / len(train_pairs):.4f}

# C. Pair Source Diversity
avg_pairs_sharing_same_goal = {np.mean(group_sizes):.2f}
max_pairs_sharing_same_goal = {max(group_sizes)}
num_unique_map_hashes = {unique_hashes if 'unique_hashes' in dir() else 'N/A'} (sampled {len(map_hashes) if 'map_hashes' in dir() else 0} plans)
""")

if __name__ == "__main__":
    analyze_dataset()
