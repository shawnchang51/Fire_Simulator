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
    batch_size: int = 256
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

    Returns:
        Dict with ranking metrics
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Batch score all configs at once, then group by plan
    all_grids = []
    all_scenarios = []
    all_plan_ids = []
    all_true_scores = []

    for i in range(len(eval_dataset)):
        item = eval_dataset[i]
        all_grids.append(item['grid'])
        all_scenarios.append(item['scenario'])
        all_plan_ids.append(item['floor_plan_id'])
        all_true_scores.append(item['ground_truth_score'])

    # Batch forward pass
    all_pred_scores = []
    num_samples = len(all_grids)

    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Scoring configs"):
        end_idx = min(start_idx + batch_size, num_samples)

        grids = torch.stack(all_grids[start_idx:end_idx]).to(device)
        scenarios = torch.stack(all_scenarios[start_idx:end_idx]).to(device)

        scores = model.score_single(grids, scenarios)
        all_pred_scores.extend(scores.cpu().numpy())

    all_pred_scores = np.array(all_pred_scores)

    # Group by plan_id
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
        'mean_kendall_tau': float(np.mean(tau_scores)) if tau_scores else 0.0,
        'std_kendall_tau': float(np.std(tau_scores)) if tau_scores else 0.0,
        'mean_spearman_rho': float(np.mean(rho_scores)) if rho_scores else 0.0,
        'std_spearman_rho': float(np.std(rho_scores)) if rho_scores else 0.0,
        'mean_ndcg@5': float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        'std_ndcg@5': float(np.std(ndcg_scores)) if ndcg_scores else 0.0,
        'top1_accuracy': float(top1_correct / num_plans) if num_plans > 0 else 0.0,
        'num_plans': num_plans,
        'num_configs': len(all_pred_scores)
    }


@torch.no_grad()
def evaluate_auxiliary(
    model: CrossAttentionRanker,
    eval_dataset: SingleConfigDataset,
    device: torch.device = None,
    batch_size: int = 256
) -> Dict[str, float]:
    """
    Evaluate auxiliary task predictions.

    Args:
        model: Trained ranking model
        eval_dataset: SingleConfigDataset with ground truth metrics
        device: Device to evaluate on
        batch_size: Batch size for inference

    Returns:
        Dict with MAE, RMSE for each auxiliary task
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    task_predictions = defaultdict(list)
    task_targets = defaultdict(list)

    # Batch inference
    for start_idx in tqdm(range(0, len(eval_dataset), batch_size), desc="Evaluating auxiliary"):
        end_idx = min(start_idx + batch_size, len(eval_dataset))

        grids = []
        for i in range(start_idx, end_idx):
            item = eval_dataset[i]
            grids.append(item['grid'])

            # Collect targets
            for task in ['survival_rate', 'steps', 'avg_fire_damage']:
                if task in item:
                    task_targets[task].append(item[task])

        grids = torch.stack(grids).to(device)
        predictions = model.predict_auxiliary(grids)

        for task, preds in predictions.items():
            task_predictions[task].extend(preds.cpu().numpy())

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

        metrics[f'{task}_mae'] = float(mean_absolute_error(targets, preds))
        metrics[f'{task}_rmse'] = float(np.sqrt(mean_squared_error(targets, preds)))

        # Correlation
        if np.std(preds) > 0 and np.std(targets) > 0:
            rho, _ = spearmanr(preds, targets)
            if not np.isnan(rho):
                metrics[f'{task}_spearman'] = float(rho)

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
        print(f"  Kendall Tau:       {ranking_metrics['mean_kendall_tau']:.4f} "
              f"(+/- {ranking_metrics['std_kendall_tau']:.4f})")
        print(f"  Spearman Rho:      {ranking_metrics['mean_spearman_rho']:.4f} "
              f"(+/- {ranking_metrics['std_spearman_rho']:.4f})")
        print(f"  NDCG@5:            {ranking_metrics['mean_ndcg@5']:.4f} "
              f"(+/- {ranking_metrics['std_ndcg@5']:.4f})")
        print(f"  Top-1 Accuracy:    {ranking_metrics['top1_accuracy']:.4f}")
        print(f"  Floor Plans:       {ranking_metrics['num_plans']:,}")
        print(f"  Total Configs:     {ranking_metrics['num_configs']:,}")

    if auxiliary_metrics is not None:
        print("\nAUXILIARY TASK METRICS:")
        for key, value in auxiliary_metrics.items():
            print(f"  {key}: {value:.4f}")

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
