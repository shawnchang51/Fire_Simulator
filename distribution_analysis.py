"""
Distribution Analysis for Monte Carlo Simulation Results
=========================================================

Computes histograms, percentiles, and statistical summaries for per-agent metrics
across multiple Monte Carlo simulation runs.

Usage:
    from distribution_analysis import compute_distributions

    # agent_records is a list of dicts from all simulation runs
    distributions = compute_distributions(agent_records)

    # Save to JSON
    import json
    with open('distributions.json', 'w') as f:
        json.dump(distributions, f, indent=2)
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict


def compute_histogram(values: List[float], num_bins: int = 20, bin_edges: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Compute histogram with automatic or custom bin edges.

    Args:
        values: List of numerical values
        num_bins: Number of bins (used if bin_edges is None)
        bin_edges: Custom bin edges (optional)

    Returns:
        Dictionary with histogram data:
        - bins: List of bin ranges as strings
        - counts: List of counts per bin
        - bin_edges: Numerical bin edges
        - density: Normalized density per bin
    """
    if not values:
        return {
            'bins': [],
            'counts': [],
            'bin_edges': [],
            'density': []
        }

    values_array = np.array(values)

    # Compute histogram
    if bin_edges is not None:
        counts, edges = np.histogram(values_array, bins=bin_edges)
    else:
        counts, edges = np.histogram(values_array, bins=num_bins)

    # Create bin labels
    bin_labels = []
    for i in range(len(edges) - 1):
        bin_labels.append(f"{edges[i]:.2f}-{edges[i+1]:.2f}")

    # Compute normalized density
    total_count = len(values)
    density = [count / total_count for count in counts]

    return {
        'bins': bin_labels,
        'counts': counts.tolist(),
        'bin_edges': edges.tolist(),
        'density': density
    }


def compute_percentiles(values: List[float], percentiles: List[float] = None) -> Dict[str, float]:
    """
    Compute percentiles for a list of values.

    Args:
        values: List of numerical values
        percentiles: List of percentiles to compute (default: [0, 25, 50, 75, 100])

    Returns:
        Dictionary mapping percentile to value
    """
    if not values:
        return {}

    if percentiles is None:
        percentiles = [0, 25, 50, 75, 100]

    values_array = np.array(values)
    result = {}

    for p in percentiles:
        result[f"p{int(p)}"] = float(np.percentile(values_array, p))

    return result


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute basic statistics for a list of values.

    Args:
        values: List of numerical values

    Returns:
        Dictionary with mean, std_dev, min, max, count
    """
    if not values:
        return {
            'count': 0,
            'mean': None,
            'std_dev': None,
            'min': None,
            'max': None
        }

    values_array = np.array(values)

    return {
        'count': len(values),
        'mean': float(np.mean(values_array)),
        'std_dev': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array))
    }


def compute_distributions(
    agent_records: List[Dict[str, Any]],
    metrics: List[str] = None,
    num_bins: int = 20,
    include_raw_values: bool = False,
    custom_bin_edges: Dict[str, List[float]] = None
) -> Dict[str, Any]:
    """
    Compute full distribution analysis for all agent records.

    Args:
        agent_records: List of agent record dictionaries from all simulation runs
        metrics: List of metric names to analyze (default: all numeric metrics)
        num_bins: Number of histogram bins
        include_raw_values: Whether to include raw values in output (can be large)
        custom_bin_edges: Dictionary mapping metric names to custom bin edges

    Returns:
        Dictionary with distribution data for each metric
    """
    if not agent_records:
        return {'error': 'No agent records provided'}

    # Default metrics to analyze
    if metrics is None:
        metrics = ['steps', 'fire_damage', 'peak_temp', 'average_temp']

    if custom_bin_edges is None:
        custom_bin_edges = {}

    # Collect values for each metric
    metric_values = defaultdict(list)
    status_counts = defaultdict(int)
    survival_counts = {'survived': 0, 'died': 0}

    for record in agent_records:
        # Count statuses
        status_counts[record.get('status', 'unknown')] += 1

        # Count survival
        if record.get('survived', False):
            survival_counts['survived'] += 1
        else:
            survival_counts['died'] += 1

        # Collect metric values
        for metric in metrics:
            if metric in record and record[metric] is not None:
                metric_values[metric].append(record[metric])

    # Compute distributions for each metric
    distributions = {}

    for metric in metrics:
        if metric not in metric_values or not metric_values[metric]:
            distributions[metric] = {
                'error': f'No data available for metric: {metric}'
            }
            continue

        values = metric_values[metric]
        bin_edges = custom_bin_edges.get(metric, None)

        distribution = {
            'statistics': compute_statistics(values),
            'percentiles': compute_percentiles(values),
            'histogram': compute_histogram(values, num_bins=num_bins, bin_edges=bin_edges)
        }

        if include_raw_values:
            distribution['raw_values'] = values

        distributions[metric] = distribution

    # Add summary statistics
    distributions['_summary'] = {
        'total_agents': len(agent_records),
        'status_counts': dict(status_counts),
        'survival_counts': survival_counts,
        'survival_rate': survival_counts['survived'] / len(agent_records) if agent_records else 0
    }

    return distributions


def compute_per_run_distributions(
    all_results: List[Dict[str, Any]],
    metrics: List[str] = None,
    num_bins: int = 20
) -> Dict[str, Any]:
    """
    Compute distributions across entire Monte Carlo runs (not individual agents).
    This gives you distributions of per-run averages.

    Args:
        all_results: List of simulation result dictionaries (one per run)
        metrics: List of metric names from result dicts
        num_bins: Number of histogram bins

    Returns:
        Dictionary with distribution data for per-run metrics
    """
    if not all_results:
        return {'error': 'No results provided'}

    # Default metrics
    if metrics is None:
        metrics = ['steps', 'average_fire_damage', 'average_peak_temp', 'average_avg_temp']

    # Collect values for each metric
    metric_values = defaultdict(list)

    for result in all_results:
        for metric in metrics:
            if metric in result and result[metric] is not None:
                metric_values[metric].append(result[metric])

    # Compute distributions
    distributions = {}

    for metric in metrics:
        if metric not in metric_values or not metric_values[metric]:
            distributions[metric] = {
                'error': f'No data available for metric: {metric}'
            }
            continue

        values = metric_values[metric]

        distributions[metric] = {
            'statistics': compute_statistics(values),
            'percentiles': compute_percentiles(values),
            'histogram': compute_histogram(values, num_bins=num_bins)
        }

    distributions['_summary'] = {
        'total_runs': len(all_results)
    }

    return distributions


def print_distribution_summary(distributions: Dict[str, Any]):
    """
    Print a human-readable summary of distributions.

    Args:
        distributions: Distribution dictionary from compute_distributions()
    """
    print("\n" + "="*80)
    print("DISTRIBUTION ANALYSIS SUMMARY")
    print("="*80)

    if '_summary' in distributions:
        summary = distributions['_summary']
        print(f"\nTotal Agents: {summary.get('total_agents', 'N/A')}")
        print(f"Survival Rate: {summary.get('survival_rate', 0) * 100:.2f}%")

        if 'status_counts' in summary:
            print("\nStatus Distribution:")
            for status, count in summary['status_counts'].items():
                print(f"  {status}: {count}")

    print("\nMetric Distributions:")
    print("-" * 80)

    for metric, data in distributions.items():
        if metric.startswith('_'):  # Skip summary fields
            continue

        if 'error' in data:
            print(f"\n{metric}: {data['error']}")
            continue

        stats = data.get('statistics', {})
        percentiles = data.get('percentiles', {})

        print(f"\n{metric.upper()}:")
        print(f"  Mean: {stats.get('mean', 'N/A'):.2f}")
        print(f"  Std Dev: {stats.get('std_dev', 'N/A'):.2f}")
        print(f"  Range: [{stats.get('min', 'N/A'):.2f}, {stats.get('max', 'N/A'):.2f}]")
        print(f"  Percentiles: 25th={percentiles.get('p25', 'N/A'):.2f}, "
              f"50th={percentiles.get('p50', 'N/A'):.2f}, "
              f"75th={percentiles.get('p75', 'N/A'):.2f}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Distribution Analysis Module")
    print("Import this module to use: from distribution_analysis import compute_distributions")
