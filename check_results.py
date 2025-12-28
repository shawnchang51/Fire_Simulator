"""
Verify model predictions by inspecting actual vs predicted values
"""

import json
import numpy as np
import torch
from pathlib import Path

from ml.config import ModelConfig
from ml.model import create_model
from ml.dataset import FireSimulationDataset, get_floor_plan_ids_from_pairs
from ml.train import load_checkpoint


def main():
    # Load config and normalization stats
    config = ModelConfig(max_plans=100)

    with open('checkpoints/normalization_stats.json', 'r') as f:
        stats = json.load(f)

    scenario_stats = stats['scenario_stats']
    target_stats = stats['target_stats']

    print("Target normalization stats:")
    print(f"  Means: {target_stats['means']}")
    print(f"  Stds:  {target_stats['stds']}")
    print()

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config)
    load_checkpoint(model, 'checkpoints/best_model.pt', device)
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_fp_ids = get_floor_plan_ids_from_pairs('combined_fast/test_pairs.jsonl')
    test_dataset = FireSimulationDataset(
        simulation_results_file='combined_fast/simulation_results.jsonl',
        floor_plans_dir='combined_fast/floor_plans',
        floor_plan_ids=test_fp_ids,
        target_size=config.target_grid_size,
        scenario_stats=scenario_stats,
        target_stats=target_stats,
        max_plans=config.max_plans
    )

    print(f"Test dataset size: {len(test_dataset)}")
    print()

    # Get random samples
    np.random.seed(42)
    indices = np.random.choice(len(test_dataset), size=20, replace=False)

    output_names = ['survival_rate', 'avg_evacuation_time', 'steps', 'avg_fire_damage']

    print("=" * 100)
    print("SAMPLE PREDICTIONS (Denormalized)")
    print("=" * 100)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = test_dataset[idx]

            # Get normalized prediction
            grid = sample['grid'].unsqueeze(0).to(device)
            scenario = sample['scenario'].unsqueeze(0).to(device)
            targets_norm = sample['targets'].numpy()

            pred_norm = model(grid, scenario).cpu().numpy()[0]

            # Denormalize
            means = np.array(target_stats['means'])
            stds = np.array(target_stats['stds'])

            pred_denorm = pred_norm * stds + means
            target_denorm = targets_norm * stds + means

            all_predictions.append(pred_denorm)
            all_targets.append(target_denorm)

            print(f"\nSample {i+1} (idx={idx}):")
            print(f"{'Output':<20} {'Actual':>12} {'Predicted':>12} {'Error':>12} {'% Error':>12}")
            print("-" * 100)

            for j, name in enumerate(output_names):
                actual = target_denorm[j]
                predicted = pred_denorm[j]
                error = predicted - actual
                pct_error = (error / actual * 100) if actual != 0 else 0

                print(f"{name:<20} {actual:>12.4f} {predicted:>12.4f} {error:>12.4f} {pct_error:>11.2f}%")

    # Compute overall statistics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    print("\n" + "=" * 100)
    print("OVERALL STATISTICS (20 samples)")
    print("=" * 100)

    for j, name in enumerate(output_names):
        actual = all_targets[:, j]
        predicted = all_predictions[:, j]

        mae = np.mean(np.abs(predicted - actual))
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))

        # Correlation
        corr = np.corrcoef(actual, predicted)[0, 1]

        print(f"\n{name}:")
        print(f"  Actual range:    [{actual.min():.4f}, {actual.max():.4f}]")
        print(f"  Predicted range: [{predicted.min():.4f}, {predicted.max():.4f}]")
        print(f"  MAE:             {mae:.4f}")
        print(f"  RMSE:            {rmse:.4f}")
        print(f"  Correlation:     {corr:.4f}")

    # Check if predictions are reasonable
    print("\n" + "=" * 100)
    print("SANITY CHECKS")
    print("=" * 100)

    checks = []

    # Survival rate should be [0, 1]
    surv_pred = all_predictions[:, 0]
    surv_valid = (surv_pred >= 0).all() and (surv_pred <= 1.2).all()  # Allow slight overshoot
    checks.append(("Survival rate in [0, 1.2]", surv_valid))

    # Evacuation time should be positive
    evac_pred = all_predictions[:, 1]
    evac_valid = (evac_pred > 0).all() and (evac_pred < 100).all()
    checks.append(("Avg evacuation time in (0, 100)", evac_valid))

    # Steps should be positive
    steps_pred = all_predictions[:, 2]
    steps_valid = (steps_pred > 0).all() and (steps_pred < 200).all()
    checks.append(("Steps in (0, 200)", steps_valid))

    # Fire damage should be non-negative
    fire_pred = all_predictions[:, 3]
    fire_valid = (fire_pred >= 0).all() and (fire_pred < 10).all()
    checks.append(("Avg fire damage in [0, 10)", fire_valid))

    for check_name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {check_name}")

    print("\n" + "=" * 100)
    all_passed = all(passed for _, passed in checks)
    if all_passed:
        print("All sanity checks PASSED! Model predictions are reasonable.")
    else:
        print("Some sanity checks FAILED. Review model predictions.")
    print("=" * 100)


if __name__ == "__main__":
    main()
