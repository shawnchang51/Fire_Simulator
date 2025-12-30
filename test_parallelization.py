"""Quick test to verify worker parallelization"""
import time
import subprocess
import sys

print("="*60)
print("Testing parallelization - this will take ~2 minutes")
print("="*60)

configs = [
    ("1 worker", 1),
    ("24 workers", 24)
]

results = []

for name, workers in configs:
    print(f"\n{name}:")
    print(f"  Running with --workers {workers}...")
    
    start = time.time()
    
    cmd = [
        sys.executable, "generate_training_data_v3.py",
        "--num-floor-plans", "1",
        "--door-configs-per-plan", "50",
        "--monte-carlo-runs", "3",
        "--workers", str(workers),
        "--output-dir", f"./test_workers_{workers}",
        "--seed", "42"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    # Extract completion info from logs
    for line in result.stdout.split('\n'):
        if 'configs/sec' in line:
            print(f"  {line.strip()}")
    
    print(f"  Total time: {elapsed:.1f}s")
    results.append((name, workers, elapsed))

print("\n" + "="*60)
print("RESULTS:")
print("="*60)
for name, workers, elapsed in results:
    print(f"{name:12s}: {elapsed:5.1f}s")

if len(results) == 2:
    speedup = results[0][2] / results[1][2]
    print(f"\nSpeedup: {speedup:.1f}x")
    print(f"Expected with {configs[1][1]} workers: ~{configs[1][1] * 0.5:.0f}-{configs[1][1] * 0.8:.0f}x")
    
    if speedup < 2:
        print("\n⚠️  WARNING: Low speedup - workers may not be active!")
    else:
        print(f"\n✅ Workers are active! ({speedup:.1f}x speedup)")
