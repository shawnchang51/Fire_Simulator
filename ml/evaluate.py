"""
Evaluation metrics for Fire Simulation Surrogate Model
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ModelConfig


def compute_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive regression metrics for multi-output predictions.

    Args:
        targets: (N, num_outputs) ground truth values
        predictions: (N, num_outputs) predicted values
        output_names: List of output names for keys

    Returns:
        Dict with per-output metrics and overall summary
    """
    if output_names is None:
        output_names = ['survival_rate', 'avg_evacuation_time', 'steps', 'avg_fire_damage']

    num_outputs = targets.shape[1]
    results = {}

    # Per-output metrics
    for i, name in enumerate(output_names[:num_outputs]):
        t = targets[:, i]
        p = predictions[:, i]

        mse = mean_squared_error(t, p)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(t, p)
        r2 = r2_score(t, p)

        # Correlation coefficients
        pearson_r, pearson_p = pearsonr(t, p)
        spearman_r, spearman_p = spearmanr(t, p)

        results[name] = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p)
        }

    # Overall metrics (mean across outputs)
    overall_mse = np.mean([results[name]['mse'] for name in output_names[:num_outputs]])
    overall_rmse = np.mean([results[name]['rmse'] for name in output_names[:num_outputs]])
    overall_mae = np.mean([results[name]['mae'] for name in output_names[:num_outputs]])
    overall_r2 = np.mean([results[name]['r2'] for name in output_names[:num_outputs]])
    overall_pearson = np.mean([results[name]['pearson_r'] for name in output_names[:num_outputs]])

    results['overall'] = {
        'overall_mean_loss': float(overall_mse),
        'mean_rmse': float(overall_rmse),
        'mean_mae': float(overall_mae),
        'mean_r2': float(overall_r2),
        'mean_pearson_r': float(overall_pearson)
    }

    return results


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: Optional[ModelConfig] = None,
    denormalize: bool = False,
    target_stats: Optional[Dict] = None
) -> Dict:
    """
    Evaluate model on test set.

    Args:
        model: The model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
        config: Model configuration (for output names)
        denormalize: Whether to denormalize predictions/targets for evaluation
        target_stats: Stats for denormalization (required if denormalize=True)

    Returns:
        Dict with comprehensive metrics
    """
    model.eval()
    model = model.to(device)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            grid = batch['grid'].to(device)
            scenario = batch['scenario'].to(device)
            targets = batch['targets']

            predictions = model(grid, scenario)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Denormalize if requested
    if denormalize and target_stats is not None:
        means = np.array(target_stats['means'])
        stds = np.array(target_stats['stds'])
        all_predictions = all_predictions * stds + means
        all_targets = all_targets * stds + means

    # Get output names
    output_names = config.output_names if config else None

    # Compute metrics
    metrics = compute_metrics(all_targets, all_predictions, output_names)

    # Add raw predictions/targets for further analysis
    metrics['_raw'] = {
        'predictions': all_predictions,
        'targets': all_targets
    }

    return metrics


def print_evaluation_report(metrics: Dict, title: str = "Evaluation Results"):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

    # Overall metrics
    overall = metrics.get('overall', {})
    print("\n OVERALL METRICS:")
    print(f"   overall_mean_loss (MSE): {overall.get('overall_mean_loss', 0):.6f}")
    print(f"   Mean RMSE:               {overall.get('mean_rmse', 0):.6f}")
    print(f"   Mean MAE:                {overall.get('mean_mae', 0):.6f}")
    print(f"   Mean R2:                 {overall.get('mean_r2', 0):.6f}")
    print(f"   Mean Pearson R:          {overall.get('mean_pearson_r', 0):.6f}")

    # Per-output metrics
    print("\n PER-OUTPUT METRICS:")
    print("-" * 70)
    print(f"{'Output':<20} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'Pearson':>10}")
    print("-" * 70)

    for key, values in metrics.items():
        if key in ['overall', '_raw']:
            continue
        print(f"{key:<20} "
              f"{values['mse']:>10.6f} "
              f"{values['rmse']:>10.6f} "
              f"{values['mae']:>10.6f} "
              f"{values['r2']:>10.6f} "
              f"{values['pearson_r']:>10.6f}")

    print("-" * 70)
    print()
