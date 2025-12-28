"""Quick script to verify all.jsonl structure"""
import json
from collections import defaultdict

floor_plans = set()
exit_configs = defaultdict(set)
door_configs = defaultdict(lambda: defaultdict(set))
scenario_hashes = set()
keys_seen = None
errors = []

with open('all.jsonl', 'r') as f:
    for line_num, line in enumerate(f, 1):
        try:
            data = json.loads(line)

            # Track first line's keys as expected schema
            if keys_seen is None:
                keys_seen = set(data.keys())
            else:
                # Check if all lines have same keys
                if set(data.keys()) != keys_seen:
                    errors.append(f"Line {line_num}: Mismatched keys")

            # Collect statistics
            fp_id = data['floor_plan_id']
            exit_id = data['exit_config_id']
            config_id = data['config_id']

            floor_plans.add(fp_id)
            exit_configs[fp_id].add(exit_id)
            door_configs[fp_id][exit_id].add(config_id)
            scenario_hashes.add(data['scenario_hash'])

        except json.JSONDecodeError as e:
            errors.append(f"Line {line_num}: JSON parse error - {e}")
        except KeyError as e:
            errors.append(f"Line {line_num}: Missing key - {e}")

print("=" * 60)
print("JSONL Verification Report")
print("=" * 60)

if errors:
    print("\n[ERROR] ERRORS FOUND:")
    for err in errors:
        print(f"  {err}")
else:
    print("\n[OK] No parsing errors")

print(f"\n[OK] Total lines: {line_num}")
print(f"[OK] Unique floor_plan_ids: {sorted(floor_plans)}")
print(f"[OK] Total floor plans: {len(floor_plans)}")
print(f"[OK] Unique scenario_hashes: {len(scenario_hashes)}")

print("\n[OK] Hierarchical structure:")
for fp_id in sorted(floor_plans):
    num_exits = len(exit_configs[fp_id])
    configs_per_exit = [len(door_configs[fp_id][eid]) for eid in sorted(exit_configs[fp_id])]
    print(f"  Floor plan {fp_id}: {num_exits} exit configs, {configs_per_exit} door configs per exit")

print(f"\n[OK] Expected keys: {sorted(keys_seen)}")

# Check required fields for run_pairing_phase.py
required_fields = {
    'floor_plan_id', 'exit_config_id', 'config_id', 'config', 'scenario',
    'survival_rate', 'avg_evacuation_time', 'steps', 'evacuated',
    'stuck', 'dead', 'avg_fire_damage', 'scenario_hash'
}

missing = required_fields - keys_seen
if missing:
    print(f"\n[ERROR] Missing required fields: {missing}")
else:
    print(f"\n[OK] All required fields present for run_pairing_phase.py")

print("\n" + "=" * 60)
if not errors and not missing:
    print("[OK] all.jsonl is USABLE")
else:
    print("[ERROR] all.jsonl has ISSUES")
print("=" * 60)
