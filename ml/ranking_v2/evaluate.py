"""
Evaluation Metrics for Pairwise Ranking Model V2

Key Metrics:
    - Pairwise: Accuracy, AUC (uses logit for comparison)
    - Per-plan ranking: Kendall Tau, Spearman, NDCG (uses RAW scores)
    - Auxiliary: MAE, RMSE for each auxiliary task

Important:
    - Kendall Tau and Spearman compare RANKINGS, not probabilities
    - Use raw scores s(x) for per-plan metrics, NOT sigmoid output
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import CrossAttentionRanker
from .dataset import SingleConfigDataset


@torch.no_grad()
def evaluate_pairwise(
    model: CrossAttentionRanker,
    test_loader: DataLoader,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Evaluate pairwise ranking metrics.

    Uses logit = s(A) - s(B) for comparison:
        - Accuracy: (logit > 0) == label
        - AUC: roc_auc_score(labels, logits)

    Args:
        model: Trained ranking model
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Dict with pairwise metrics
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    all_logits = []
    all_labels = []
    all_confidences = []

    for batch in tqdm(test_loader, desc="Evaluating pairwise"):
        grid_a = batch['grid_a'].to(device)
        scenario_a = batch['scenario_a'].to(device)
        grid_b = batch['grid_b'].to(device)
        scenario_b = batch['scenario_b'].to(device)
        label = batch['label']
        confidence = batch['confidence']

        # Forward pass
        outputs = model(grid_a, scenario_a, grid_b, scenario_b)
        logit = outputs['logit']

        all_logits.extend(logit.cpu().numpy())
        all_labels.extend(label.numpy())
        all_confidences.extend(confidence.numpy())

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)

    # Pairwise accuracy
    predictions = (all_logits > 0).astype(int)
    accuracy = (predictions == all_labels).mean()

    # AUC
    try:
        auc = roc_auc_score(all_labels, all_logits)
    except ValueError:
        auc = 0.5

    # Weighted accuracy (by confidence)
    weighted_correct = ((predictions == all_labels) * all_confidences).sum()
    weighted_accuracy = weighted_correct / all_confidences.sum()

    return {
        'pairwise_accuracy': float(accuracy),
        'pairwise_auc': float(auc),
        'weighted_accuracy': float(weighted_accuracy),
        'num_pairs': len(all_labels)
    }


@torch.no_grad()
def evaluate_per_plan_ranking(
    model: CrossAttentionRanker,
    eval_dataset: SingleConfigDataset,
    device: torch.device = None,
    batch_size: int = 256,
    num_workers: int = 0
) -> Dict[str, float]:
    """
    Evaluate per-plan ranking metrics using RAW scores.

    For each floor plan, score ALL configs and compare ranking
    to ground truth scores from simulator.

    Uses RAW scores s(x) - NOT probabilities or logits.

    Args:
        model: Trained ranking model
        eval_dataset: SingleConfigDataset with all configs
        device: Device to evaluate on
        batch_size: Batch size for scoring
        num_workers: Number of workers for data loading

    Returns:
        Dict with ranking metrics
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Use DataLoader for efficient batched loading
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Collect predictions and ground truth
    all_pred_scores = []
    all_true_scores = []
    all_plan_ids = []

    for batch in tqdm(eval_loader, desc="Scoring configs"):
        grids = batch['grid'].to(device)
        scenarios = batch['scenario'].to(device)

        # Forward pass
        scores = model.score_single(grids, scenarios)

        # Collect results
        all_pred_scores.extend(scores.cpu().numpy())
        all_true_scores.extend(batch['ground_truth_score'].numpy())
        all_plan_ids.extend(batch['floor_plan_id'].numpy())

    all_pred_scores = np.array(all_pred_scores)
    all_true_scores = np.array(all_true_scores)

    # Compute overall correlations (across all configs)
    overall_kendall_tau, _ = kendalltau(all_pred_scores, all_true_scores)
    overall_spearman_r, _ = spearmanr(all_pred_scores, all_true_scores)
    overall_pearson_r = np.corrcoef(all_pred_scores, all_true_scores)[0, 1]

    # Group by plan_id for per-plan metrics
    plan_preds = defaultdict(list)
    plan_trues = defaultdict(list)
    for i, plan_id in enumerate(all_plan_ids):
        plan_preds[plan_id].append(all_pred_scores[i])
        plan_trues[plan_id].append(all_true_scores[i])

    # Compute per-plan metrics
    tau_scores = []
    rho_scores = []
    ndcg_scores = []
    top1_correct = 0
    num_plans = 0

    for plan_id in plan_preds.keys():
        pred_scores = np.array(plan_preds[plan_id])
        true_scores = np.array(plan_trues[plan_id])

        if len(pred_scores) < 2:
            continue  # Need at least 2 configs

        num_plans += 1

        # Kendall Tau: correlation of rankings
        tau, _ = kendalltau(pred_scores, true_scores)
        if not np.isnan(tau):
            tau_scores.append(tau)

        # Spearman Rho: rank correlation
        rho, _ = spearmanr(pred_scores, true_scores)
        if not np.isnan(rho):
            rho_scores.append(rho)

        # NDCG@5
        ndcg = compute_ndcg(pred_scores, true_scores, k=5)
        ndcg_scores.append(ndcg)

        # Top-1 accuracy: is the predicted best also the true best?
        if np.argmax(pred_scores) == np.argmax(true_scores):
            top1_correct += 1

    return {
        # Overall correlations (what the notebook expects)
        'kendall_tau': float(overall_kendall_tau) if not np.isnan(overall_kendall_tau) else 0.0,
        'spearman_r': float(overall_spearman_r) if not np.isnan(overall_spearman_r) else 0.0,
        'pearson_r': float(overall_pearson_r) if not np.isnan(overall_pearson_r) else 0.0,

        # Per-plan statistics
        'mean_per_plan_kendall': float(np.mean(tau_scores)) if tau_scores else 0.0,
        'std_per_plan_kendall': float(np.std(tau_scores)) if tau_scores else 0.0,
        'per_plan_kendall': tau_scores,  # For histogram plotting
        'mean_per_plan_spearman': float(np.mean(rho_scores)) if rho_scores else 0.0,
        'std_per_plan_spearman': float(np.std(rho_scores)) if rho_scores else 0.0,
        'per_plan_spearman': rho_scores,  # For histogram plotting

        # Other metrics
        'ndcg_at_10': float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,  # Renamed from mean_ndcg@5
        'top1_accuracy': float(top1_correct / num_plans) if num_plans > 0 else 0.0,
        'num_plans': num_plans,
        'num_configs': len(all_pred_scores)
    }


@torch.no_grad()
def evaluate_auxiliary(
    model: CrossAttentionRanker,
    eval_dataset: SingleConfigDataset,
    device: torch.device = None,
    batch_size: int = 256,
    num_workers: int = 0
) -> Dict[str, float]:
    """
    Evaluate auxiliary task predictions.

    Args:
        model: Trained ranking model
        eval_dataset: SingleConfigDataset with ground truth metrics
        device: Device to evaluate on
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading

    Returns:
        Dict with MAE, RMSE, R² for each auxiliary task
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Use DataLoader for efficient batched loading
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    task_predictions = defaultdict(list)
    task_targets = defaultdict(list)

    # Batch inference
    for batch in tqdm(eval_loader, desc="Evaluating auxiliary"):
        grids = batch['grid'].to(device)

        # Get predictions from model
        predictions = model.predict_auxiliary(grids)

        # Collect predictions and targets
        for task in ['survival_rate', 'steps', 'avg_fire_damage']:
            if task in predictions:
                task_predictions[task].extend(predictions[task].cpu().numpy())
            if task in batch:
                task_targets[task].extend(batch[task].numpy())

    # Compute metrics
    metrics = {}
    for task in task_predictions.keys():
        if task not in task_targets or len(task_targets[task]) == 0:
            continue

        preds = np.array(task_predictions[task])
        targets = np.array(task_targets[task])

        # Ensure same length
        min_len = min(len(preds), len(targets))
        preds = preds[:min_len]
        targets = targets[:min_len]

        # Compute metrics
        mae = mean_absolute_error(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))

        # R² score
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics[task] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }

    return metrics


def compute_ndcg(
    pred_scores: np.ndarray,
    true_scores: np.ndarray,
    k: int = 5
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.

    Higher is better - measures quality of top-k ranking.

    Args:
        pred_scores: Predicted scores for all configs
        true_scores: Ground truth scores for all configs
        k: Number of top items to consider

    Returns:
        NDCG@k value in [0, 1]
    """
    n = len(pred_scores)
    k = min(k, n)

    # Sort by predicted scores (descending), get top-k indices
    pred_ranking = np.argsort(-pred_scores)[:k]

    # Ideal ranking: sort by true scores (descending)
    ideal_ranking = np.argsort(-true_scores)[:k]

    # DCG: sum of true_scores[rank_i] / log2(i + 2)
    dcg = sum(
        true_scores[pred_ranking[i]] / np.log2(i + 2)
        for i in range(k)
    )

    # Ideal DCG
    idcg = sum(
        true_scores[ideal_ranking[i]] / np.log2(i + 2)
        for i in range(k)
    )

    return dcg / idcg if idcg > 0 else 0.0


def print_evaluation_report(
    pairwise_metrics: Dict[str, float],
    ranking_metrics: Optional[Dict[str, float]] = None,
    auxiliary_metrics: Optional[Dict[str, float]] = None
):
    """
    Print formatted evaluation report.

    Args:
        pairwise_metrics: Metrics from evaluate_pairwise()
        ranking_metrics: Metrics from evaluate_per_plan_ranking()
        auxiliary_metrics: Metrics from evaluate_auxiliary()
    """
    print("\n" + "=" * 60)
    print("EVALUATION REPORT - RANKING MODEL V2")
    print("=" * 60)

    print("\nPAIRWISE METRICS:")
    print(f"  Accuracy:          {pairwise_metrics['pairwise_accuracy']:.4f}")
    print(f"  AUC:               {pairwise_metrics['pairwise_auc']:.4f}")
    print(f"  Weighted Accuracy: {pairwise_metrics['weighted_accuracy']:.4f}")
    print(f"  Total Pairs:       {pairwise_metrics['num_pairs']:,}")

    if ranking_metrics is not None:
        print("\nPER-PLAN RANKING METRICS (using raw scores):")
        print(f"  Overall Kendall Tau:      {ranking_metrics['kendall_tau']:.4f}")
        print(f"  Overall Spearman R:       {ranking_metrics['spearman_r']:.4f}")
        print(f"  Overall Pearson R:        {ranking_metrics['pearson_r']:.4f}")
        print(f"  Mean Per-Plan Kendall:    {ranking_metrics['mean_per_plan_kendall']:.4f} "
              f"(+/- {ranking_metrics['std_per_plan_kendall']:.4f})")
        print(f"  Mean Per-Plan Spearman:   {ranking_metrics['mean_per_plan_spearman']:.4f} "
              f"(+/- {ranking_metrics['std_per_plan_spearman']:.4f})")
        print(f"  NDCG@10:                  {ranking_metrics['ndcg_at_10']:.4f}")
        print(f"  Top-1 Accuracy:           {ranking_metrics['top1_accuracy']:.4f}")
        print(f"  Floor Plans:              {ranking_metrics['num_plans']:,}")
        print(f"  Total Configs:            {ranking_metrics['num_configs']:,}")

    if auxiliary_metrics is not None:
        print("\nAUXILIARY TASK METRICS:")
        for task, metrics in auxiliary_metrics.items():
            if isinstance(metrics, dict):
                print(f"  {task}:")
                print(f"    MAE:   {metrics['mae']:.4f}")
                print(f"    RMSE:  {metrics['rmse']:.4f}")
                print(f"    R²:    {metrics['r2']:.4f}")
            else:
                print(f"  {task}: {metrics:.4f}")

    print("\n" + "=" * 60)


def evaluate_model_full(
    model: CrossAttentionRanker,
    test_loader: DataLoader,
    eval_dataset: Optional[SingleConfigDataset] = None,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Full evaluation: pairwise + per-plan ranking + auxiliary metrics.

    Args:
        model: Trained ranking model
        test_loader: Test data loader for pairwise metrics
        eval_dataset: SingleConfigDataset for per-plan and auxiliary metrics
        device: Device to evaluate on

    Returns:
        Combined metrics dict
    """
    if device is None:
        device = next(model.parameters()).device

    # Pairwise metrics
    pairwise_metrics = evaluate_pairwise(model, test_loader, device)

    # Per-plan ranking metrics (if dataset provided)
    ranking_metrics = None
    auxiliary_metrics = None
    if eval_dataset is not None:
        ranking_metrics = evaluate_per_plan_ranking(model, eval_dataset, device)
        auxiliary_metrics = evaluate_auxiliary(model, eval_dataset, device)

    # Combine
    all_metrics = {**pairwise_metrics}
    if ranking_metrics is not None:
        all_metrics.update(ranking_metrics)
    if auxiliary_metrics is not None:
        all_metrics.update(auxiliary_metrics)

    # Print report
    print_evaluation_report(pairwise_metrics, ranking_metrics, auxiliary_metrics)

    return all_metrics
