"""
V5 Dataset Comparison Tool

Compare multiple V5 datasets side-by-side to evaluate:
- Generation parameter effects
- Data quality differences
- Performance metric distributions

Usage:
    python compare_v5_datasets.py \
        --datasets ./data_v5_baseline ./data_v5_aggressive \
        --labels "Baseline" "Aggressive Fire" \
        --output comparison_report.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

import numpy as np

from analyze_v5_data import V5DataAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonReport:
    """Report comparing multiple V5 datasets"""
    dataset_labels: List[str]
    dataset_paths: List[str]
    comparisons: Dict[str, Any] = field(default_factory=dict)

    def print_summary(self):
        """Print comparison summary"""
        print("\n" + "=" * 100)
        print("V5 DATASET COMPARISON REPORT")
        print("=" * 100)

        print(f"\nDatasets ({len(self.dataset_labels)}):")
        for i, (label, path) in enumerate(zip(self.dataset_labels, self.dataset_paths), 1):
            print(f"  [{i}] {label}: {path}")

        # Dataset sizes
        print(f"\n{'Dataset Sizes':-^100}")
        print(f"{'Dataset':<30} {'Plans':<10} {'Configs':<10} {'Pairs':<10} {'Valid':<10}")
        print("-" * 100)
        for label, sizes in zip(self.dataset_labels, self.comparisons.get('sizes', [])):
            valid = "✓" if sizes.get('is_valid', False) else "✗"
            print(f"{label:<30} {sizes.get('floor_plans', 0):<10,} "
                  f"{sizes.get('door_configs', 0):<10,} "
                  f"{sizes.get('pairs', 0):<10,} "
                  f"{valid:<10}")

        # Pair distributions
        print(f"\n{'Pair Type Distribution':-^100}")
        print(f"{'Dataset':<30} {'Same-Exit':<15} {'Cross-Exit':<15} {'Cross-Plan':<15}")
        print("-" * 100)
        for label, dist in zip(self.dataset_labels, self.comparisons.get('pair_distributions', [])):
            same = dist.get('same', 0)
            cross = dist.get('cross', 0)
            plan = dist.get('plan', 0)
            print(f"{label:<30} {same:<15.2%} {cross:<15.2%} {plan:<15.2%}")

        # Performance metrics
        print(f"\n{'Performance Metrics (Mean ± Std)':-^100}")
        print(f"{'Dataset':<30} {'Survival Rate':<20} {'Score':<20} {'Steps':<20}")
        print("-" * 100)
        for label, perf in zip(self.dataset_labels, self.comparisons.get('performance', [])):
            surv = perf.get('survival_rate', {})
            score = perf.get('score', {})
            steps = perf.get('steps', {})

            surv_str = f"{surv.get('mean', 0):.3f} ± {surv.get('std', 0):.3f}"
            score_str = f"{score.get('mean', 0):.3f} ± {score.get('std', 0):.3f}"
            steps_str = f"{steps.get('mean', 0):.0f} ± {steps.get('std', 0):.0f}"

            print(f"{label:<30} {surv_str:<20} {score_str:<20} {steps_str:<20}")

        # Recommendations
        print(f"\n{'Recommendations':-^100}")
        recs = self.comparisons.get('recommendations', [])
        if recs:
            for rec in recs:
                print(f"  • {rec}")
        else:
            print("  No specific recommendations")

        print("=" * 100 + "\n")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'dataset_labels': self.dataset_labels,
            'dataset_paths': self.dataset_paths,
            'comparisons': self.comparisons
        }


class V5DatasetComparator:
    """Compare multiple V5 datasets"""

    def __init__(self, dataset_paths: List[str], labels: List[str] = None):
        self.dataset_paths = [Path(p) for p in dataset_paths]

        if labels is None:
            self.labels = [f"Dataset {i+1}" for i in range(len(dataset_paths))]
        else:
            assert len(labels) == len(dataset_paths), "Labels must match dataset count"
            self.labels = labels

        self.analyzers: List[V5DataAnalyzer] = []
        self.reports = []

    def load_and_analyze(self):
        """Load and analyze all datasets"""
        logger.info(f"Loading and analyzing {len(self.dataset_paths)} datasets...")

        for i, (path, label) in enumerate(zip(self.dataset_paths, self.labels), 1):
            logger.info(f"  [{i}/{len(self.dataset_paths)}] {label}...")

            analyzer = V5DataAnalyzer(str(path))
            analyzer.load_data()
            report = analyzer.analyze()

            self.analyzers.append(analyzer)
            self.reports.append(report)

    def compare(self) -> ComparisonReport:
        """Generate comparison report"""
        logger.info("Generating comparison report...")

        comparison = ComparisonReport(
            dataset_labels=self.labels,
            dataset_paths=[str(p) for p in self.dataset_paths]
        )

        # Compare sizes
        comparison.comparisons['sizes'] = [
            {
                'floor_plans': r.total_floor_plans,
                'exit_configs': r.total_exit_configs,
                'door_configs': r.total_door_configs,
                'pairs': r.total_pairs,
                'is_valid': r.is_valid()
            }
            for r in self.reports
        ]

        # Compare pair distributions
        comparison.comparisons['pair_distributions'] = [
            r.pair_stats.get('type_distribution', {})
            for r in self.reports
        ]

        # Compare performance metrics
        comparison.comparisons['performance'] = [
            r.performance_stats
            for r in self.reports
        ]

        # Compare hierarchical structure
        comparison.comparisons['structure'] = [
            r.structure_stats
            for r in self.reports
        ]

        # Compare validation issues
        comparison.comparisons['validation'] = [
            {
                'errors': len(r.get_errors()),
                'warnings': len(r.get_warnings()),
                'is_valid': r.is_valid()
            }
            for r in self.reports
        ]

        # Generate recommendations
        comparison.comparisons['recommendations'] = self._generate_recommendations()

        return comparison

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on comparison"""
        recommendations = []

        # Check for invalid datasets
        invalid_datasets = [
            label for label, report in zip(self.labels, self.reports)
            if not report.is_valid()
        ]
        if invalid_datasets:
            recommendations.append(
                f"Fix validation errors in: {', '.join(invalid_datasets)}"
            )

        # Check for low variance
        for label, report in zip(self.labels, self.reports):
            score_std = report.performance_stats.get('score', {}).get('std', 0)
            if score_std < 0.01:
                recommendations.append(
                    f"{label}: Very low score variance ({score_std:.4f}) - "
                    "configurations may be too similar"
                )

        # Check for low survival rates
        for label, report in zip(self.labels, self.reports):
            surv_mean = report.performance_stats.get('survival_rate', {}).get('mean', 1.0)
            if surv_mean < 0.3:
                recommendations.append(
                    f"{label}: Low average survival rate ({surv_mean:.2%}) - "
                    "consider using less aggressive fire parameters"
                )

        # Find best dataset
        if len(self.reports) > 1:
            # Score by: validity, pair count, score variance
            scores = []
            for report in self.reports:
                score = 0
                if report.is_valid():
                    score += 100
                score += min(report.total_pairs / 100000, 10)  # Up to 10 points for pairs
                score += min(report.performance_stats.get('score', {}).get('std', 0) * 100, 10)
                scores.append(score)

            best_idx = np.argmax(scores)
            recommendations.append(
                f"Recommended dataset for training: {self.labels[best_idx]} "
                f"(score: {scores[best_idx]:.1f})"
            )

        return recommendations

    def plot_comparison(self, output_dir: str):
        """Generate comparison plots"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            logger.warning("matplotlib not installed. Skipping plots.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info(f"Generating comparison plots in {output_dir}...")

        # Plot 1: Survival rate comparison
        self._plot_survival_rate_comparison(output_path)

        # Plot 2: Pair distribution comparison
        self._plot_pair_distribution_comparison(output_path)

        # Plot 3: Score variance comparison
        self._plot_score_comparison(output_path)

        logger.info(f"Comparison plots saved to {output_dir}")

    def _plot_survival_rate_comparison(self, output_path: Path):
        """Plot survival rate distributions"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (label, analyzer) in enumerate(zip(self.labels, self.analyzers)):
            survival_rates = [r.get('survival_rate', 0) for r in analyzer.results]
            ax.hist(survival_rates, bins=30, alpha=0.5, label=label, edgecolor='black')

        ax.set_xlabel('Survival Rate')
        ax.set_ylabel('Frequency')
        ax.set_title('Survival Rate Distribution Comparison')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'survival_rate_comparison.png', dpi=150)
        plt.close()

    def _plot_pair_distribution_comparison(self, output_path: Path):
        """Plot pair type distribution comparison"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(self.labels))
        width = 0.25

        pair_types = ['same', 'cross', 'plan']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, ptype in enumerate(pair_types):
            values = [
                report.pair_stats.get('type_distribution', {}).get(ptype, 0)
                for report in self.reports
            ]
            ax.bar(x + i * width, values, width, label=ptype.capitalize(),
                   color=colors[i], alpha=0.8)

        ax.set_xlabel('Dataset')
        ax.set_ylabel('Proportion')
        ax.set_title('Pair Type Distribution Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.labels, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'pair_distribution_comparison.png', dpi=150)
        plt.close()

    def _plot_score_comparison(self, output_path: Path):
        """Plot score statistics comparison"""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Score mean
        means = [
            report.performance_stats.get('score', {}).get('mean', 0)
            for report in self.reports
        ]
        stds = [
            report.performance_stats.get('score', {}).get('std', 0)
            for report in self.reports
        ]

        x = np.arange(len(self.labels))
        ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='#2ca02c')
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Mean Score')
        ax1.set_title('Average Score Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.labels, rotation=15, ha='right')
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Score variance
        variances = [s ** 2 for s in stds]
        ax2.bar(x, variances, alpha=0.7, color='#d62728')
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Score Variance')
        ax2.set_title('Score Variance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.labels, rotation=15, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'score_comparison.png', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple V5 hierarchical training datasets'
    )

    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                        help='Paths to V5 dataset directories')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Labels for datasets (optional)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save comparison report to JSON file')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')
    parser.add_argument('--plot-dir', type=str, default='./comparison_plots',
                        help='Directory for plots')

    args = parser.parse_args()

    # Validate inputs
    if args.labels and len(args.labels) != len(args.datasets):
        logger.error("Number of labels must match number of datasets")
        return

    # Create comparator
    comparator = V5DatasetComparator(args.datasets, args.labels)

    # Load and analyze
    comparator.load_and_analyze()

    # Generate comparison
    comparison = comparator.compare()

    # Print summary
    comparison.print_summary()

    # Save report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(comparison.to_dict(), f, indent=2)
        logger.info(f"Comparison report saved to {args.output}")

    # Generate plots
    if args.plot:
        comparator.plot_comparison(args.plot_dir)


if __name__ == '__main__':
    main()
