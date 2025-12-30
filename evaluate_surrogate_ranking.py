"""
Evaluate Surrogate Model for Pairwise Ranking

This script evaluates how well the surrogate model can predict which floor plan
configuration is better by:
1. Using the surrogate model to predict simulation outcomes
2. Computing composite scores from predictions using the same formula as pair_constructor.py
3. Creating pairwise comparisons and evaluating ranking metrics

Usage:
    python evaluate_surrogate_ranking.py --checkpoint path/to/model.pt --data_dir combined_fast
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import ndcg_score, roc_auc_score, roc_curve
from tqdm import tqdm

from ml.surrogate.model import FireSimulationSurrogate
from ml.surrogate.dataset import FireSimulationDataset, get_floor_plan_ids_from_pairs
from ml.surrogate.config import ModelConfig
from pair_constructor import PairwiseLabel


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def compute_score_from_predictions(
    survival_rate: float,
    avg_evacuation_time: float,
    steps: float,
    avg_fire_damage: float
) -> float:
    """
    Compute composite ranking score from predicted outputs.

    This uses the EXACT same formula as pair_constructor.py line 71.
    Score is normalized to approximately [0, 1] range.

    Args:
        survival_rate: Predicted survival rate [0, 1]
        avg_evacuation_time: Predicted average evacuation time
        steps: Predicted number of simulation steps
        avg_fire_damage: Predicted average fire damage [0, ~3]

    Returns:
        Composite score (higher is better)
    """
    # Formula from pair_constructor.py:71
    # Normalization constants based on observed ranges:
    # steps: [11, 70]
    # avg_fire_damage: [0.0167, 3.0111]
    score = (
        0.7 * survival_rate
        - 0.15 * ((steps - 11) / (70 - 11))
        - 0.15 * ((avg_fire_damage - 0.0167) / (3.0111 - 0.0167))
        + 0.118
    ) / 0.818

    return score


def load_model_and_config(checkpoint_path: str) -> Tuple[nn.Module, ModelConfig, Dict]:
    """
    Load trained surrogate model from checkpoint.

    Returns:
        Tuple of (model, config, checkpoint_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract config
    config_dict = checkpoint.get('config', {})
    config = ModelConfig(**config_dict) if config_dict else ModelConfig()

    # Create model
    model = FireSimulationSurrogate(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Model has {model.count_parameters():,} parameters")

    return model, config, checkpoint


def predict_all_scores(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_stats: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict scores for all test samples.

    Args:
        model: Trained surrogate model
        dataloader: Test data loader
        device: Device to run on
        target_stats: Denormalization statistics

    Returns:
        Tuple of (predicted_scores, ground_truth_scores, predictions_array)
        - predicted_scores: (N,) scores computed from model predictions
        - ground_truth_scores: (N,) scores computed from actual results
        - predictions_array: (N, 4) raw model predictions
    """
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            grid = batch['grid'].to(device)
            scenario = batch['scenario'].to(device)
            targets = batch['targets']

            predictions = model(grid, scenario)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)  # (N, 4)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 4)

    # Denormalize predictions and targets
    if target_stats is not None:
        means = np.array(target_stats['means'])
        stds = np.array(target_stats['stds'])
        all_predictions = all_predictions * stds + means
        all_targets = all_targets * stds + means

    # Compute scores from predictions
    predicted_scores = np.array([
        compute_score_from_predictions(
            pred[0],  # survival_rate
            pred[1],  # avg_evacuation_time
            pred[2],  # steps
            pred[3]   # avg_fire_damage
        )
        for pred in all_predictions
    ])

    # Compute scores from ground truth
    ground_truth_scores = np.array([
        compute_score_from_predictions(
            target[0],  # survival_rate
            target[1],  # avg_evacuation_time
            target[2],  # steps
            target[3]   # avg_fire_damage
        )
        for target in all_targets
    ])

    return predicted_scores, ground_truth_scores, all_predictions


def evaluate_pairwise_ranking(
    predicted_scores: np.ndarray,
    ground_truth_scores: np.ndarray,
    num_pairs: int = 10000,
    margin: float = 0.002,
    seed: int = 42
) -> Dict:
    """
    Evaluate pairwise ranking performance.

    Creates random pairs and checks if the model correctly predicts which is better.

    Args:
        predicted_scores: (N,) scores from model predictions
        ground_truth_scores: (N,) scores from actual results
        num_pairs: Number of pairs to sample
        margin: Minimum score difference to consider valid pair
        seed: Random seed

    Returns:
        Dict with ranking metrics
    """
    rng = np.random.default_rng(seed)
    n = len(predicted_scores)

    correct = 0
    total = 0
    score_diffs = []

    for _ in range(num_pairs):
        # Sample two different indices
        i, j = rng.choice(n, size=2, replace=False)

        gt_diff = ground_truth_scores[i] - ground_truth_scores[j]
        pred_diff = predicted_scores[i] - predicted_scores[j]

        # Skip ambiguous pairs (below margin)
        if abs(gt_diff) < margin:
            continue

        # Check if prediction agrees with ground truth
        if (gt_diff > 0 and pred_diff > 0) or (gt_diff < 0 and pred_diff < 0):
            correct += 1

        total += 1
        score_diffs.append(abs(gt_diff))

    accuracy = correct / total if total > 0 else 0

    return {
        'pairwise_accuracy': accuracy,
        'total_pairs_evaluated': total,
        'correct_pairs': correct,
        'avg_score_difference': np.mean(score_diffs) if score_diffs else 0
    }


def evaluate_auc_roc(
    predicted_scores: np.ndarray,
    ground_truth_scores: np.ndarray,
    num_pairs: int = 10000,
    margin: float = 0.002,
    seed: int = 42
) -> Dict:
    """
    Evaluate AUC-ROC for pairwise ranking.

    AUC-ROC measures the probability that the model assigns a higher score
    to a randomly chosen better configuration than to a worse one.

    Args:
        predicted_scores: (N,) scores from model predictions
        ground_truth_scores: (N,) scores from actual results
        num_pairs: Number of pairs to sample
        margin: Minimum score difference to consider valid pair
        seed: Random seed

    Returns:
        Dict with AUC-ROC metrics
    """
    rng = np.random.default_rng(seed)
    n = len(predicted_scores)

    labels = []
    scores = []

    for _ in range(num_pairs):
        # Sample two different indices
        i, j = rng.choice(n, size=2, replace=False)

        gt_diff = ground_truth_scores[i] - ground_truth_scores[j]
        pred_diff = predicted_scores[i] - predicted_scores[j]

        # Skip ambiguous pairs (below margin)
        if abs(gt_diff) < margin:
            continue

        # Label: 1 if A is better than B (gt_diff > 0), 0 otherwise
        label = 1 if gt_diff > 0 else 0
        labels.append(label)

        # Score: predicted difference (higher = more confident A is better)
        scores.append(pred_diff)

    if len(labels) == 0:
        return {
            'auc_roc': 0.0,
            'num_pairs_for_auc': 0
        }

    # Compute AUC-ROC
    auc = roc_auc_score(labels, scores)

    return {
        'auc_roc': float(auc),
        'num_pairs_for_auc': len(labels)
    }


def evaluate_ranking_correlations(
    predicted_scores: np.ndarray,
    ground_truth_scores: np.ndarray
) -> Dict:
    """
    Evaluate ranking correlations (Kendall Tau, Spearman).

    Args:
        predicted_scores: (N,) scores from model predictions
        ground_truth_scores: (N,) scores from actual results

    Returns:
        Dict with correlation metrics
    """
    # Kendall Tau (measures ranking agreement)
    kendall_tau, kendall_p = kendalltau(ground_truth_scores, predicted_scores)

    # Spearman correlation (monotonic relationship)
    spearman_r, spearman_p = spearmanr(ground_truth_scores, predicted_scores)

    # Pearson correlation (linear relationship)
    pearson_r = np.corrcoef(ground_truth_scores, predicted_scores)[0, 1]

    return {
        'kendall_tau': float(kendall_tau),
        'kendall_p_value': float(kendall_p),
        'spearman_r': float(spearman_r),
        'spearman_p_value': float(spearman_p),
        'pearson_r': float(pearson_r)
    }


def evaluate_per_plan_ranking(
    predicted_scores: np.ndarray,
    ground_truth_scores: np.ndarray,
    floor_plan_ids: List[int],
    top_k: int = 10
) -> Dict:
    """
    Evaluate ranking performance within each floor plan.

    For each floor plan, evaluate how well the model ranks configurations.

    Args:
        predicted_scores: (N,) scores from model predictions
        ground_truth_scores: (N,) scores from actual results
        floor_plan_ids: (N,) floor plan ID for each sample
        top_k: Evaluate top-K accuracy

    Returns:
        Dict with per-plan metrics
    """
    # Group by floor plan
    from collections import defaultdict
    by_plan = defaultdict(list)

    for i, plan_id in enumerate(floor_plan_ids):
        by_plan[plan_id].append({
            'pred_score': predicted_scores[i],
            'gt_score': ground_truth_scores[i],
            'idx': i
        })

    # Evaluate each plan
    plan_kendalls = []
    plan_spearmans = []
    plan_top_k_overlaps = []

    for plan_id, samples in by_plan.items():
        if len(samples) < 2:
            continue

        pred_scores = np.array([s['pred_score'] for s in samples])
        gt_scores = np.array([s['gt_score'] for s in samples])

        # Kendall Tau for this plan
        tau, _ = kendalltau(gt_scores, pred_scores)
        plan_kendalls.append(tau)

        # Spearman for this plan
        rho, _ = spearmanr(gt_scores, pred_scores)
        plan_spearmans.append(rho)

        # Top-K overlap (how many of the true top-K are in predicted top-K?)
        if len(samples) >= top_k:
            gt_top_k_indices = np.argsort(gt_scores)[-top_k:]
            pred_top_k_indices = np.argsort(pred_scores)[-top_k:]
            overlap = len(set(gt_top_k_indices) & set(pred_top_k_indices))
            plan_top_k_overlaps.append(overlap / top_k)

    return {
        'num_plans_evaluated': len(plan_kendalls),
        'mean_per_plan_kendall_tau': float(np.mean(plan_kendalls)) if plan_kendalls else 0,
        'std_per_plan_kendall_tau': float(np.std(plan_kendalls)) if plan_kendalls else 0,
        'mean_per_plan_spearman_r': float(np.mean(plan_spearmans)) if plan_spearmans else 0,
        'std_per_plan_spearman_r': float(np.std(plan_spearmans)) if plan_spearmans else 0,
        'mean_top_k_overlap': float(np.mean(plan_top_k_overlaps)) if plan_top_k_overlaps else 0,
        'top_k': top_k
    }


def evaluate_on_pairs_file(
    model: nn.Module,
    pairs_file: str,
    simulation_results_file: str,
    floor_plans_dir: str,
    device: torch.device,
    target_stats: Optional[Dict] = None,
    config: Optional[ModelConfig] = None
) -> Dict:
    """
    Evaluate model on explicit pairwise labels from a pairs file.

    Args:
        model: Trained surrogate model
        pairs_file: Path to pairs.jsonl file
        simulation_results_file: Path to simulation_results.jsonl
        floor_plans_dir: Path to floor_plans/ directory
        device: Device to run on
        target_stats: Denormalization statistics
        config: Model config

    Returns:
        Dict with evaluation metrics
    """
    # Load pairs
    pairs = []
    with open(pairs_file, 'r') as f:
        for line in f:
            pairs.append(PairwiseLabel.from_dict(json.loads(line)))

    logger.info(f"Loaded {len(pairs)} pairs from {pairs_file}")

    # Build a lookup of simulation results
    # We need to predict for each unique (floor_plan_id, config, scenario)
    # This is complex, so for simplicity, let's just evaluate pairwise accuracy
    # by predicting both A and B for each pair

    # Actually, this is tricky because we need to encode each config on-the-fly
    # For now, let's skip this and recommend using the simpler approach

    raise NotImplementedError(
        "Evaluating on explicit pairs requires encoding configs on-the-fly. "
        "Use the main evaluation approach instead."
    )


def print_ranking_report(metrics: Dict, title: str = "Surrogate Model Ranking Evaluation"):
    """Print formatted ranking evaluation report."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

    # Pairwise accuracy
    if 'pairwise_accuracy' in metrics:
        print("\n PAIRWISE RANKING ACCURACY:")
        print(f"   Accuracy:             {metrics['pairwise_accuracy']:.4f} ({metrics['pairwise_accuracy']*100:.2f}%)")
        print(f"   Total pairs:          {metrics['total_pairs_evaluated']}")
        print(f"   Correct predictions:  {metrics['correct_pairs']}")
        print(f"   Avg score difference: {metrics['avg_score_difference']:.4f}")

    # AUC-ROC
    if 'auc_roc' in metrics:
        print("\n AUC-ROC (Area Under ROC Curve):")
        print(f"   AUC-ROC:              {metrics['auc_roc']:.4f} ({metrics['auc_roc']*100:.2f}%)")
        print(f"   Pairs evaluated:      {metrics['num_pairs_for_auc']}")
        print(f"   Interpretation:       Probability model correctly ranks random pair")

    # Correlation metrics
    if 'kendall_tau' in metrics:
        print("\n RANKING CORRELATIONS:")
        print(f"   Kendall Tau:  {metrics['kendall_tau']:.4f} (p={metrics['kendall_p_value']:.2e})")
        print(f"   Spearman R:   {metrics['spearman_r']:.4f} (p={metrics['spearman_p_value']:.2e})")
        print(f"   Pearson R:    {metrics['pearson_r']:.4f}")

    # Per-plan metrics
    if 'num_plans_evaluated' in metrics:
        print("\n PER-PLAN RANKING METRICS:")
        print(f"   Plans evaluated:           {metrics['num_plans_evaluated']}")
        print(f"   Mean Kendall Tau:          {metrics['mean_per_plan_kendall_tau']:.4f} ± {metrics['std_per_plan_kendall_tau']:.4f}")
        print(f"   Mean Spearman R:           {metrics['mean_per_plan_spearman_r']:.4f} ± {metrics['std_per_plan_spearman_r']:.4f}")
        print(f"   Mean Top-{metrics['top_k']} Overlap:  {metrics['mean_top_k_overlap']:.4f} ({metrics['mean_top_k_overlap']*100:.2f}%)")

    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate surrogate model for pairwise ranking")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--data_dir', type=str, default='combined_fast',
                        help='Path to data directory')
    parser.add_argument('--floor_plans_dir', type=str, default=None,
                        help='Path to floor_plans directory (default: data_dir/floor_plans)')
    parser.add_argument('--num_pairs', type=int, default=10000,
                        help='Number of pairs to evaluate for pairwise accuracy')
    parser.add_argument('--margin', type=float, default=0.002,
                        help='Minimum score difference for valid pairs')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top-K for per-plan evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for metrics (default: save to checkpoint directory)')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    floor_plans_dir = args.floor_plans_dir or str(data_dir / 'floor_plans')

    # Set output path: if not specified, save to checkpoint directory
    if args.output is None:
        checkpoint_dir = Path(args.checkpoint).parent
        output_path = checkpoint_dir / 'surrogate_ranking_metrics.json'
    else:
        output_path = Path(args.output)

    # Load model
    model, config, checkpoint = load_model_and_config(args.checkpoint)
    config.data_dir = args.data_dir
    config.floor_plans_dir = floor_plans_dir

    # Get target stats from checkpoint
    target_stats = checkpoint.get('target_stats', None)

    # Load test data
    test_fp_ids = get_floor_plan_ids_from_pairs(str(data_dir / "test_pairs.jsonl"))
    logger.info(f"Test set has {len(test_fp_ids)} floor plans")

    test_dataset = FireSimulationDataset(
        simulation_results_file=str(data_dir / "simulation_results.jsonl"),
        floor_plans_dir=floor_plans_dir,
        floor_plan_ids=test_fp_ids,
        target_size=config.target_grid_size,
        scenario_stats=checkpoint.get('scenario_stats', None),
        target_stats=target_stats
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Single-threaded for simplicity
        pin_memory=False
    )

    logger.info(f"Test dataset has {len(test_dataset)} samples")

    # Predict scores
    logger.info("Predicting scores for all test samples...")
    predicted_scores, ground_truth_scores, predictions = predict_all_scores(
        model, test_loader, device, target_stats
    )

    # Extract floor plan IDs from test dataset
    floor_plan_ids = [record['floor_plan_id'] for record in test_dataset.records]

    # Evaluate pairwise ranking
    logger.info(f"Evaluating pairwise ranking on {args.num_pairs} random pairs...")
    pairwise_metrics = evaluate_pairwise_ranking(
        predicted_scores,
        ground_truth_scores,
        num_pairs=args.num_pairs,
        margin=args.margin
    )

    # Evaluate AUC-ROC
    logger.info(f"Computing AUC-ROC on {args.num_pairs} random pairs...")
    auc_metrics = evaluate_auc_roc(
        predicted_scores,
        ground_truth_scores,
        num_pairs=args.num_pairs,
        margin=args.margin
    )

    # Evaluate ranking correlations
    logger.info("Computing ranking correlations...")
    correlation_metrics = evaluate_ranking_correlations(
        predicted_scores,
        ground_truth_scores
    )

    # Evaluate per-plan ranking
    logger.info("Evaluating per-plan ranking...")
    per_plan_metrics = evaluate_per_plan_ranking(
        predicted_scores,
        ground_truth_scores,
        floor_plan_ids,
        top_k=args.top_k
    )

    # Combine all metrics
    all_metrics = {
        **pairwise_metrics,
        **auc_metrics,
        **correlation_metrics,
        **per_plan_metrics
    }

    # Add score statistics
    all_metrics['score_statistics'] = {
        'predicted_mean': float(np.mean(predicted_scores)),
        'predicted_std': float(np.std(predicted_scores)),
        'predicted_min': float(np.min(predicted_scores)),
        'predicted_max': float(np.max(predicted_scores)),
        'ground_truth_mean': float(np.mean(ground_truth_scores)),
        'ground_truth_std': float(np.std(ground_truth_scores)),
        'ground_truth_min': float(np.min(ground_truth_scores)),
        'ground_truth_max': float(np.max(ground_truth_scores)),
    }

    # Print report
    print_ranking_report(all_metrics)

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Save metrics
    with open(output_path, 'w') as f:
        json.dump(convert_to_native(all_metrics), f, indent=2)
    logger.info(f"Metrics saved to {output_path}")

    # Additional analysis: show score distribution comparison
    print("\n SCORE DISTRIBUTION:")
    print(f"   Predicted:     μ={all_metrics['score_statistics']['predicted_mean']:.4f}, "
          f"σ={all_metrics['score_statistics']['predicted_std']:.4f}, "
          f"range=[{all_metrics['score_statistics']['predicted_min']:.4f}, "
          f"{all_metrics['score_statistics']['predicted_max']:.4f}]")
    print(f"   Ground Truth:  μ={all_metrics['score_statistics']['ground_truth_mean']:.4f}, "
          f"σ={all_metrics['score_statistics']['ground_truth_std']:.4f}, "
          f"range=[{all_metrics['score_statistics']['ground_truth_min']:.4f}, "
          f"{all_metrics['score_statistics']['ground_truth_max']:.4f}]")
    print()


if __name__ == '__main__':
    main()
