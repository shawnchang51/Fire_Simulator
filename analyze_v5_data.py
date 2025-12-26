"""
Data Analyzer and Validator for V5 Hierarchical Training Data

Validates and analyzes the three-tier hierarchical structure:
- Floor plans → Exit configs → Door configs

Provides detailed statistics on:
- Hierarchical structure integrity
- Pair type distribution (same-exit, cross-exit, cross-plan)
- Score distributions and correlations
- Data quality checks

Usage:
    python analyze_v5_data.py --data-dir ./training_data_v5
    python analyze_v5_data.py --data-dir ./training_data_v5 --save-report report.json
    python analyze_v5_data.py --data-dir ./training_data_v5 --plot
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue found in the dataset"""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'structure', 'pairs', 'scores', 'distribution'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    """Complete analysis report for V5 dataset"""
    dataset_path: str
    version: str
    total_floor_plans: int
    total_exit_configs: int
    total_door_configs: int
    total_pairs: int

    # Hierarchical structure stats
    structure_stats: Dict[str, Any] = field(default_factory=dict)

    # Pair distribution stats
    pair_stats: Dict[str, Any] = field(default_factory=dict)

    # Score and performance stats
    performance_stats: Dict[str, Any] = field(default_factory=dict)

    # Validation issues
    issues: List[ValidationIssue] = field(default_factory=list)

    def add_issue(self, severity: str, category: str, message: str, **details):
        """Add a validation issue"""
        self.issues.append(ValidationIssue(severity, category, message, details))

    def get_errors(self) -> List[ValidationIssue]:
        """Get all error-level issues"""
        return [i for i in self.issues if i.severity == 'error']

    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues"""
        return [i for i in self.issues if i.severity == 'warning']

    def is_valid(self) -> bool:
        """Check if dataset is valid (no errors)"""
        return len(self.get_errors()) == 0

    def print_summary(self):
        """Print a human-readable summary"""
        print("\n" + "=" * 80)
        print("V5 HIERARCHICAL TRAINING DATA ANALYSIS REPORT")
        print("=" * 80)
        print(f"Dataset: {self.dataset_path}")
        print(f"Version: {self.version}")
        print(f"\nDataset Size:")
        print(f"  Floor Plans: {self.total_floor_plans:,}")
        print(f"  Exit Configs: {self.total_exit_configs:,}")
        print(f"  Door Configs: {self.total_door_configs:,}")
        print(f"  Training Pairs: {self.total_pairs:,}")

        # Structure stats
        print(f"\n{'Hierarchical Structure':-^80}")
        for key, value in self.structure_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        # Pair distribution
        print(f"\n{'Pair Distribution':-^80}")
        for key, value in self.pair_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.3f}")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        # Performance stats
        print(f"\n{'Performance Metrics':-^80}")
        for key, value in self.performance_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.4f}")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        # Validation issues
        errors = self.get_errors()
        warnings = self.get_warnings()

        print(f"\n{'Validation Results':-^80}")
        if len(errors) == 0 and len(warnings) == 0:
            print("  ✓ No issues found - dataset is valid!")
        else:
            if errors:
                print(f"  ✗ {len(errors)} ERROR(S):")
                for issue in errors[:5]:  # Show first 5
                    print(f"    - [{issue.category}] {issue.message}")
                if len(errors) > 5:
                    print(f"    ... and {len(errors) - 5} more errors")

            if warnings:
                print(f"  ⚠ {len(warnings)} WARNING(S):")
                for issue in warnings[:5]:  # Show first 5
                    print(f"    - [{issue.category}] {issue.message}")
                if len(warnings) > 5:
                    print(f"    ... and {len(warnings) - 5} more warnings")

        print("=" * 80 + "\n")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'dataset_path': self.dataset_path,
            'version': self.version,
            'total_floor_plans': self.total_floor_plans,
            'total_exit_configs': self.total_exit_configs,
            'total_door_configs': self.total_door_configs,
            'total_pairs': self.total_pairs,
            'structure_stats': self.structure_stats,
            'pair_stats': self.pair_stats,
            'performance_stats': self.performance_stats,
            'issues': [
                {
                    'severity': i.severity,
                    'category': i.category,
                    'message': i.message,
                    'details': i.details
                }
                for i in self.issues
            ]
        }


class V5DataAnalyzer:
    """
    Analyzer and validator for V5 hierarchical training data.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

        # Data storage
        self.metadata: Dict[str, Any] = {}
        self.results: List[Dict[str, Any]] = []
        self.pairs: Dict[str, List[Dict[str, Any]]] = {
            'train': [],
            'val': [],
            'test': []
        }

        # Analysis results
        self.report: Optional[AnalysisReport] = None

    def load_data(self):
        """Load all data files from the directory"""
        logger.info("Loading data...")

        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"  Loaded metadata (version: {self.metadata.get('version', 'unknown')})")
        else:
            logger.warning("  metadata.json not found")

        # Load simulation results
        results_path = self.data_dir / 'simulation_results.jsonl'
        if results_path.exists():
            with open(results_path, 'r') as f:
                for line in f:
                    if line.strip():
                        self.results.append(json.loads(line))
            logger.info(f"  Loaded {len(self.results)} simulation results")
        else:
            logger.warning("  simulation_results.jsonl not found")

        # Load pairs
        for split in ['train', 'val', 'test']:
            pairs_path = self.data_dir / f'{split}_pairs.jsonl'
            if pairs_path.exists():
                with open(pairs_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            self.pairs[split].append(json.loads(line))
                logger.info(f"  Loaded {len(self.pairs[split])} {split} pairs")
            else:
                logger.warning(f"  {split}_pairs.jsonl not found")

    def analyze(self) -> AnalysisReport:
        """Run complete analysis and validation"""
        logger.info("\nAnalyzing dataset...")

        # Initialize report
        version = self.metadata.get('version', 'unknown')
        self.report = AnalysisReport(
            dataset_path=str(self.data_dir),
            version=version,
            total_floor_plans=0,
            total_exit_configs=0,
            total_door_configs=len(self.results),
            total_pairs=sum(len(pairs) for pairs in self.pairs.values())
        )

        # Run analysis steps
        self._validate_version()
        self._analyze_hierarchical_structure()
        self._analyze_pair_distribution()
        self._analyze_performance_metrics()
        self._validate_pair_integrity()
        self._check_data_quality()

        logger.info("Analysis complete!")
        return self.report

    def _validate_version(self):
        """Validate this is V5 data"""
        version = self.metadata.get('version', 'unknown')
        if version != 'v5':
            self.report.add_issue(
                'warning', 'structure',
                f"Expected version 'v5', found '{version}'"
            )

    def _analyze_hierarchical_structure(self):
        """Analyze the hierarchical structure: Plans → Exits → Doors"""
        logger.info("  Analyzing hierarchical structure...")

        # Group results by hierarchy
        by_plan = defaultdict(lambda: defaultdict(list))

        for result in self.results:
            plan_id = result.get('floor_plan_id')
            exit_id = result.get('exit_config_id', 0)
            by_plan[plan_id][exit_id].append(result)

        num_plans = len(by_plan)
        self.report.total_floor_plans = num_plans

        # Analyze exit configs per plan
        exit_counts = [len(exits) for exits in by_plan.values()]
        num_exit_configs = sum(exit_counts)
        self.report.total_exit_configs = num_exit_configs

        # Analyze door configs per exit
        door_counts_per_exit = []
        for plan_exits in by_plan.values():
            for exit_doors in plan_exits.values():
                door_counts_per_exit.append(len(exit_doors))

        # Store stats
        self.report.structure_stats = {
            'avg_exit_configs_per_plan': np.mean(exit_counts) if exit_counts else 0,
            'min_exit_configs_per_plan': int(np.min(exit_counts)) if exit_counts else 0,
            'max_exit_configs_per_plan': int(np.max(exit_counts)) if exit_counts else 0,
            'avg_door_configs_per_exit': np.mean(door_counts_per_exit) if door_counts_per_exit else 0,
            'min_door_configs_per_exit': int(np.min(door_counts_per_exit)) if door_counts_per_exit else 0,
            'max_door_configs_per_exit': int(np.max(door_counts_per_exit)) if door_counts_per_exit else 0,
            'exit_config_distribution': dict(Counter(exit_counts)),
            'door_config_distribution': dict(Counter(door_counts_per_exit))
        }

        # Validate expected structure
        expected_exits = self.metadata.get('config', {}).get('exit_configs_per_plan', None)
        expected_doors = self.metadata.get('config', {}).get('door_configs_per_exit', None)

        if expected_exits and np.mean(exit_counts) < expected_exits * 0.9:
            self.report.add_issue(
                'warning', 'structure',
                f"Average exit configs ({np.mean(exit_counts):.1f}) below expected ({expected_exits})",
                expected=expected_exits,
                actual=np.mean(exit_counts)
            )

        if expected_doors and np.mean(door_counts_per_exit) < expected_doors * 0.9:
            self.report.add_issue(
                'warning', 'structure',
                f"Average door configs ({np.mean(door_counts_per_exit):.1f}) below expected ({expected_doors})",
                expected=expected_doors,
                actual=np.mean(door_counts_per_exit)
            )

    def _analyze_pair_distribution(self):
        """Analyze distribution of pair types"""
        logger.info("  Analyzing pair distribution...")

        all_pairs = []
        for pairs in self.pairs.values():
            all_pairs.extend(pairs)

        if not all_pairs:
            self.report.add_issue('error', 'pairs', "No pairs found in dataset")
            return

        # Count pair types
        pair_type_counts = Counter()
        label_counts = Counter()

        for pair in all_pairs:
            pair_type = pair.get('pair_type', 'unknown')
            label = pair.get('label', -1)

            # Normalize pair type (remove strategy suffix)
            base_type = pair_type.split('_')[0] if pair_type != 'unknown' else 'unknown'
            pair_type_counts[base_type] += 1
            label_counts[label] += 1

        total_pairs = len(all_pairs)

        # Calculate distributions
        type_distribution = {
            pt: count / total_pairs
            for pt, count in pair_type_counts.items()
        }

        label_distribution = {
            f'label_{label}': count / total_pairs
            for label, count in label_counts.items()
        }

        # Score differences by pair type
        score_diffs_by_type = defaultdict(list)
        for pair in all_pairs:
            pair_type = pair.get('pair_type', 'unknown').split('_')[0]
            score_a = pair.get('score_a', 0)
            score_b = pair.get('score_b', 0)
            score_diffs_by_type[pair_type].append(abs(score_a - score_b))

        avg_score_diff_by_type = {
            pt: np.mean(diffs) if diffs else 0
            for pt, diffs in score_diffs_by_type.items()
        }

        # Store stats
        self.report.pair_stats = {
            'type_counts': dict(pair_type_counts),
            'type_distribution': type_distribution,
            'label_distribution': label_distribution,
            'avg_score_diff_by_type': avg_score_diff_by_type,
            'total_score_diff_avg': np.mean([d for diffs in score_diffs_by_type.values() for d in diffs])
        }

        # Validate expected ratios
        expected_ratios = {
            'same': self.metadata.get('config', {}).get('same_exit_ratio', 0.7),
            'cross': self.metadata.get('config', {}).get('cross_exit_ratio', 0.2),
        }

        actual_same = type_distribution.get('same', 0)
        actual_cross_exit = type_distribution.get('cross', 0)  # Assuming 'cross' captures cross-exit

        if abs(actual_same - expected_ratios['same']) > 0.1:
            self.report.add_issue(
                'warning', 'distribution',
                f"Same-exit pair ratio ({actual_same:.2%}) deviates from expected ({expected_ratios['same']:.2%})",
                expected=expected_ratios['same'],
                actual=actual_same
            )

        # Check label balance
        label_1_ratio = label_distribution.get('label_1', 0)
        if abs(label_1_ratio - 0.5) > 0.1:
            self.report.add_issue(
                'warning', 'distribution',
                f"Labels are imbalanced: {label_1_ratio:.2%} positive",
                target=0.5,
                actual=label_1_ratio
            )

    def _analyze_performance_metrics(self):
        """Analyze survival rates, scores, and other performance metrics"""
        logger.info("  Analyzing performance metrics...")

        if not self.results:
            return

        # Extract metrics
        survival_rates = [r.get('survival_rate', 0) for r in self.results]
        scores = [r.get('score', 0) for r in self.results]
        steps = [r.get('steps', 0) for r in self.results]
        fire_damages = [r.get('avg_fire_damage', 0) for r in self.results]

        # Group by hierarchy level
        by_plan = defaultdict(list)
        by_exit = defaultdict(list)

        for result in self.results:
            plan_id = result.get('floor_plan_id')
            exit_id = result.get('exit_config_id', 0)

            by_plan[plan_id].append(result.get('survival_rate', 0))
            by_exit[(plan_id, exit_id)].append(result.get('survival_rate', 0))

        # Calculate variance at each level
        plan_variances = [np.var(rates) if rates else 0 for rates in by_plan.values()]
        exit_variances = [np.var(rates) if rates else 0 for rates in by_exit.values()]

        # Store stats
        self.report.performance_stats = {
            'survival_rate': {
                'mean': np.mean(survival_rates),
                'std': np.std(survival_rates),
                'min': np.min(survival_rates),
                'max': np.max(survival_rates),
                'median': np.median(survival_rates)
            },
            'score': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores)
            },
            'steps': {
                'mean': np.mean(steps),
                'std': np.std(steps),
                'min': int(np.min(steps)),
                'max': int(np.max(steps)),
                'median': np.median(steps)
            },
            'fire_damage': {
                'mean': np.mean(fire_damages),
                'std': np.std(fire_damages),
                'min': np.min(fire_damages),
                'max': np.max(fire_damages),
                'median': np.median(fire_damages)
            },
            'variance_analysis': {
                'avg_variance_across_plans': np.mean(plan_variances),
                'avg_variance_within_exits': np.mean(exit_variances)
            }
        }

        # Check for anomalies
        if np.mean(survival_rates) < 0.3:
            self.report.add_issue(
                'warning', 'scores',
                f"Low average survival rate: {np.mean(survival_rates):.2%}",
                avg_survival_rate=np.mean(survival_rates)
            )

        if np.std(scores) < 0.01:
            self.report.add_issue(
                'warning', 'scores',
                f"Very low score variance ({np.std(scores):.4f}) - configurations may be too similar",
                score_std=np.std(scores)
            )

    def _validate_pair_integrity(self):
        """Validate pair data integrity and consistency"""
        logger.info("  Validating pair integrity...")

        all_pairs = []
        for pairs in self.pairs.values():
            all_pairs.extend(pairs)

        if not all_pairs:
            return

        # Check for required fields
        required_fields = [
            'floor_plan_id_a', 'floor_plan_id_b',
            'config_a', 'config_b',
            'score_a', 'score_b',
            'label', 'pair_type'
        ]

        missing_fields_count = 0
        for i, pair in enumerate(all_pairs[:100]):  # Check first 100
            for field in required_fields:
                if field not in pair:
                    missing_fields_count += 1
                    if missing_fields_count <= 5:  # Report first 5
                        self.report.add_issue(
                            'error', 'pairs',
                            f"Pair {i} missing required field: {field}"
                        )

        if missing_fields_count > 5:
            self.report.add_issue(
                'error', 'pairs',
                f"Total {missing_fields_count} missing field issues found"
            )

        # Validate label consistency
        label_errors = 0
        for pair in all_pairs[:1000]:  # Check first 1000
            score_a = pair.get('score_a', 0)
            score_b = pair.get('score_b', 0)
            label = pair.get('label', -1)

            expected_label = 1 if score_a > score_b else 0
            if label != expected_label and abs(score_a - score_b) > 0.01:  # Allow small margin
                label_errors += 1
                if label_errors <= 3:
                    self.report.add_issue(
                        'error', 'pairs',
                        f"Label mismatch: score_a={score_a:.4f}, score_b={score_b:.4f}, label={label}",
                        score_a=score_a,
                        score_b=score_b,
                        label=label,
                        expected=expected_label
                    )

        if label_errors > 3:
            self.report.add_issue(
                'error', 'pairs',
                f"Total {label_errors} label consistency errors found",
                total_errors=label_errors
            )

    def _check_data_quality(self):
        """Check for data quality issues"""
        logger.info("  Checking data quality...")

        # Check for duplicate results
        result_keys = set()
        duplicates = 0

        for result in self.results:
            key = (
                result.get('floor_plan_id'),
                result.get('exit_config_id', 0),
                result.get('config_id')
            )
            if key in result_keys:
                duplicates += 1
            result_keys.add(key)

        if duplicates > 0:
            self.report.add_issue(
                'warning', 'structure',
                f"Found {duplicates} duplicate result entries",
                count=duplicates
            )

        # Check for missing exit_config_id
        missing_exit_id = sum(1 for r in self.results if 'exit_config_id' not in r)
        if missing_exit_id > 0:
            self.report.add_issue(
                'error', 'structure',
                f"{missing_exit_id} results missing 'exit_config_id' field",
                count=missing_exit_id
            )

        # Check split sizes
        train_size = len(self.pairs['train'])
        val_size = len(self.pairs['val'])
        test_size = len(self.pairs['test'])
        total = train_size + val_size + test_size

        if total > 0:
            train_ratio = train_size / total
            val_ratio = val_size / total
            test_ratio = test_size / total

            # Expected: 70/15/15
            if abs(train_ratio - 0.7) > 0.05:
                self.report.add_issue(
                    'warning', 'distribution',
                    f"Train split ratio ({train_ratio:.2%}) deviates from expected (70%)",
                    expected=0.7,
                    actual=train_ratio
                )

            if abs(val_ratio - 0.15) > 0.05:
                self.report.add_issue(
                    'warning', 'distribution',
                    f"Val split ratio ({val_ratio:.2%}) deviates from expected (15%)",
                    expected=0.15,
                    actual=val_ratio
                )

    def save_report(self, output_path: str):
        """Save analysis report to JSON file"""
        if self.report is None:
            logger.error("No report to save. Run analyze() first.")
            return

        with open(output_path, 'w') as f:
            json.dump(self.report.to_dict(), f, indent=2)

        logger.info(f"Report saved to {output_path}")

    def plot_distributions(self, output_dir: Optional[str] = None):
        """Generate visualization plots (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
        except ImportError:
            logger.warning("matplotlib not installed. Skipping plots.")
            return

        if output_dir is None:
            output_dir = self.data_dir / 'analysis_plots'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)
        logger.info(f"Generating plots in {output_dir}...")

        # Plot 1: Pair type distribution
        self._plot_pair_distribution(output_dir)

        # Plot 2: Score distributions
        self._plot_score_distributions(output_dir)

        # Plot 3: Hierarchical structure
        self._plot_hierarchical_structure(output_dir)

        # Plot 4: Performance metrics
        self._plot_performance_metrics(output_dir)

        logger.info(f"Plots saved to {output_dir}")

    def _plot_pair_distribution(self, output_dir: Path):
        """Plot pair type distribution"""
        import matplotlib.pyplot as plt

        type_counts = self.report.pair_stats.get('type_counts', {})

        if not type_counts:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        types = list(type_counts.keys())
        counts = list(type_counts.values())

        ax.bar(types, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xlabel('Pair Type')
        ax.set_ylabel('Count')
        ax.set_title('Pair Type Distribution')
        ax.grid(axis='y', alpha=0.3)

        # Add percentages on bars
        total = sum(counts)
        for i, (t, c) in enumerate(zip(types, counts)):
            ax.text(i, c, f'{c:,}\n({c/total:.1%})', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / 'pair_distribution.png', dpi=150)
        plt.close()

    def _plot_score_distributions(self, output_dir: Path):
        """Plot score distributions by pair type"""
        import matplotlib.pyplot as plt

        all_pairs = []
        for pairs in self.pairs.values():
            all_pairs.extend(pairs)

        # Group score differences by pair type
        score_diffs = defaultdict(list)
        for pair in all_pairs:
            pair_type = pair.get('pair_type', 'unknown').split('_')[0]
            diff = abs(pair.get('score_a', 0) - pair.get('score_b', 0))
            score_diffs[pair_type].append(diff)

        fig, ax = plt.subplots(figsize=(12, 6))

        data_to_plot = [score_diffs[t] for t in score_diffs.keys()]
        labels = list(score_diffs.keys())

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

        # Color the boxes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel('Pair Type')
        ax.set_ylabel('Score Difference |score_a - score_b|')
        ax.set_title('Score Difference Distribution by Pair Type')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'score_differences.png', dpi=150)
        plt.close()

    def _plot_hierarchical_structure(self, output_dir: Path):
        """Plot hierarchical structure statistics"""
        import matplotlib.pyplot as plt

        # Group results by plan and exit
        by_plan = defaultdict(lambda: defaultdict(list))

        for result in self.results:
            plan_id = result.get('floor_plan_id')
            exit_id = result.get('exit_config_id', 0)
            by_plan[plan_id][exit_id].append(result)

        # Count exits per plan and doors per exit
        exit_counts = [len(exits) for exits in by_plan.values()]
        door_counts = []
        for plan_exits in by_plan.values():
            for exit_doors in plan_exits.values():
                door_counts.append(len(exit_doors))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Exit configs per plan
        ax1.hist(exit_counts, bins=20, edgecolor='black', alpha=0.7, color='#1f77b4')
        ax1.set_xlabel('Number of Exit Configs')
        ax1.set_ylabel('Number of Floor Plans')
        ax1.set_title('Exit Configurations per Floor Plan')
        ax1.axvline(np.mean(exit_counts), color='red', linestyle='--',
                    label=f'Mean: {np.mean(exit_counts):.1f}')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Door configs per exit
        ax2.hist(door_counts, bins=20, edgecolor='black', alpha=0.7, color='#ff7f0e')
        ax2.set_xlabel('Number of Door Configs')
        ax2.set_ylabel('Number of Exit Configs')
        ax2.set_title('Door Configurations per Exit Config')
        ax2.axvline(np.mean(door_counts), color='red', linestyle='--',
                    label=f'Mean: {np.mean(door_counts):.1f}')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'hierarchical_structure.png', dpi=150)
        plt.close()

    def _plot_performance_metrics(self, output_dir: Path):
        """Plot performance metric distributions"""
        import matplotlib.pyplot as plt

        survival_rates = [r.get('survival_rate', 0) for r in self.results]
        scores = [r.get('score', 0) for r in self.results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Survival rate distribution
        ax1.hist(survival_rates, bins=30, edgecolor='black', alpha=0.7, color='#2ca02c')
        ax1.set_xlabel('Survival Rate')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Survival Rate Distribution')
        ax1.axvline(np.mean(survival_rates), color='red', linestyle='--',
                    label=f'Mean: {np.mean(survival_rates):.3f}')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Score distribution
        ax2.hist(scores, bins=30, edgecolor='black', alpha=0.7, color='#d62728')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Score Distribution')
        ax2.axvline(np.mean(scores), color='red', linestyle='--',
                    label=f'Mean: {np.mean(scores):.3f}')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'performance_metrics.png', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and validate V5 hierarchical training data'
    )

    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to V5 training data directory')
    parser.add_argument('--save-report', type=str, default=None,
                        help='Save analysis report to JSON file')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--plot-dir', type=str, default=None,
                        help='Directory for plots (default: <data-dir>/analysis_plots)')

    args = parser.parse_args()

    # Create analyzer
    analyzer = V5DataAnalyzer(args.data_dir)

    # Load data
    analyzer.load_data()

    # Run analysis
    report = analyzer.analyze()

    # Print summary
    report.print_summary()

    # Save report if requested
    if args.save_report:
        analyzer.save_report(args.save_report)

    # Generate plots if requested
    if args.plot:
        analyzer.plot_distributions(args.plot_dir)

    # Exit with appropriate code
    if not report.is_valid():
        logger.error("Dataset validation FAILED - errors found")
        exit(1)
    else:
        logger.info("Dataset validation PASSED")
        exit(0)


if __name__ == '__main__':
    main()
