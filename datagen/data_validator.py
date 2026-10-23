"""
Data Validator for Training Data Quality Control

Validates floor plans, simulation results, and pairwise labels to ensure:
- Floor plan validity (connectivity, exits reachable)
- Simulation result sanity (no crashes, reasonable metrics)
- Label quality (balanced, diverse, consistent)
- Dataset splits are non-leaking (by floor plan, not by pair)

Usage:
    validator = DataValidator()
    issues = validator.validate_floor_plan(grid, exits)
    report = validator.generate_quality_report(pairs)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'floor_plan', 'simulation', 'label', 'diversity'
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'severity': self.severity,
            'category': self.category,
            'message': self.message,
            'details': self.details
        }


@dataclass
class ValidationReport:
    """Complete validation report"""
    is_valid: bool
    error_count: int
    warning_count: int
    issues: List[ValidationIssue]
    statistics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'issues': [i.to_dict() for i in self.issues],
            'statistics': self.statistics
        }

    def print_summary(self):
        """Print human-readable summary"""
        status = "VALID" if self.is_valid else "INVALID"
        print(f"\n{'='*60}")
        print(f"Validation Report: {status}")
        print(f"{'='*60}")
        print(f"Errors: {self.error_count}, Warnings: {self.warning_count}")

        if self.issues:
            print(f"\nIssues:")
            for issue in self.issues[:20]:  # Show first 20
                icon = {'error': 'X', 'warning': '!', 'info': 'i'}[issue.severity]
                print(f"  [{icon}] {issue.category}: {issue.message}")

            if len(self.issues) > 20:
                print(f"  ... and {len(self.issues) - 20} more issues")

        if self.statistics:
            print(f"\nStatistics:")
            for key, value in self.statistics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")


class FloorPlanValidator:
    """Validates floor plan grids"""

    WALL = -2
    PASSABLE = 0

    def validate(
        self,
        grid: np.ndarray,
        exit_positions: List[Tuple[int, int]],
        min_passable_ratio: float = 0.15,
        require_single_component: bool = True
    ) -> List[ValidationIssue]:
        """
        Validate a floor plan grid.

        Args:
            grid: 2D numpy array (-2 = wall, 0 = passable)
            exit_positions: List of (col, row) exit positions
            min_passable_ratio: Minimum ratio of passable cells
            require_single_component: Require single connected component

        Returns:
            List of validation issues
        """
        issues = []
        rows, cols = grid.shape

        # Check dimensions
        if rows < 10 or cols < 10:
            issues.append(ValidationIssue(
                severity='error',
                category='floor_plan',
                message=f'Floor plan too small: {rows}x{cols}, minimum 10x10',
                details={'rows': rows, 'cols': cols}
            ))

        # Check passable ratio
        passable_count = np.sum(grid == self.PASSABLE)
        passable_ratio = passable_count / (rows * cols)

        if passable_ratio < min_passable_ratio:
            issues.append(ValidationIssue(
                severity='error',
                category='floor_plan',
                message=f'Insufficient passable area: {passable_ratio:.1%} < {min_passable_ratio:.1%}',
                details={'passable_ratio': passable_ratio, 'passable_count': int(passable_count)}
            ))

        # Check perimeter walls
        if not self._check_perimeter_walls(grid):
            issues.append(ValidationIssue(
                severity='warning',
                category='floor_plan',
                message='Perimeter is not fully walled (exits should be marked separately)',
                details={}
            ))

        # Check connectivity
        components = self._find_connected_components(grid)
        if len(components) == 0:
            issues.append(ValidationIssue(
                severity='error',
                category='floor_plan',
                message='No passable areas found',
                details={}
            ))
        elif len(components) > 1 and require_single_component:
            issues.append(ValidationIssue(
                severity='warning',
                category='floor_plan',
                message=f'Multiple disconnected areas: {len(components)} components',
                details={'component_sizes': [len(c) for c in components]}
            ))

        # Check exits
        if len(exit_positions) == 0:
            issues.append(ValidationIssue(
                severity='error',
                category='floor_plan',
                message='No exits defined',
                details={}
            ))
        else:
            # Check exit validity
            for exit_pos in exit_positions:
                col, row = exit_pos
                if not (0 <= row < rows and 0 <= col < cols):
                    issues.append(ValidationIssue(
                        severity='error',
                        category='floor_plan',
                        message=f'Exit position out of bounds: ({col}, {row})',
                        details={'exit_position': exit_pos}
                    ))
                elif not self._is_on_perimeter(row, col, rows, cols):
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='floor_plan',
                        message=f'Exit not on perimeter: ({col}, {row})',
                        details={'exit_position': exit_pos}
                    ))

            # Check exit reachability
            reachable_exits = self._count_reachable_exits(grid, exit_positions)
            if components and reachable_exits == 0:
                issues.append(ValidationIssue(
                    severity='error',
                    category='floor_plan',
                    message='No exits reachable from passable areas',
                    details={'exit_count': len(exit_positions)}
                ))

        return issues

    def _check_perimeter_walls(self, grid: np.ndarray) -> bool:
        """Check if perimeter is walled (with allowance for exits)"""
        # Allow some non-wall cells for exits
        non_wall_top = np.sum(grid[0, :] != self.WALL)
        non_wall_bottom = np.sum(grid[-1, :] != self.WALL)
        non_wall_left = np.sum(grid[:, 0] != self.WALL)
        non_wall_right = np.sum(grid[:, -1] != self.WALL)

        # Allow up to 10% of perimeter for exits
        max_exits = max(4, int(0.1 * (grid.shape[0] + grid.shape[1]) * 2))
        total_non_wall = non_wall_top + non_wall_bottom + non_wall_left + non_wall_right

        return total_non_wall <= max_exits

    def _find_connected_components(self, grid: np.ndarray) -> List[Set[Tuple[int, int]]]:
        """Find connected components of passable cells"""
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        components = []

        for y in range(rows):
            for x in range(cols):
                if grid[y, x] == self.PASSABLE and not visited[y, x]:
                    component = set()
                    queue = deque([(y, x)])
                    visited[y, x] = True

                    while queue:
                        cy, cx = queue.popleft()
                        component.add((cx, cy))

                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < rows and 0 <= nx < cols:
                                if grid[ny, nx] == self.PASSABLE and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    queue.append((ny, nx))

                    components.append(component)

        return components

    def _is_on_perimeter(self, row: int, col: int, rows: int, cols: int) -> bool:
        """Check if position is on the grid perimeter"""
        return row == 0 or row == rows - 1 or col == 0 or col == cols - 1

    def _count_reachable_exits(
        self,
        grid: np.ndarray,
        exit_positions: List[Tuple[int, int]]
    ) -> int:
        """Count exits reachable from any passable cell"""
        components = self._find_connected_components(grid)
        if not components:
            return 0

        # Get largest component
        largest = max(components, key=len)

        # Check which exits are adjacent to the component
        reachable = 0
        rows, cols = grid.shape

        for col, row in exit_positions:
            # Check if any adjacent cell is in the component
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = row + dy, col + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    if (nx, ny) in largest:
                        reachable += 1
                        break

        return reachable


class SimulationResultValidator:
    """Validates simulation results"""

    def validate(
        self,
        results: List[Dict[str, Any]],
        expected_agent_count: Optional[int] = None
    ) -> List[ValidationIssue]:
        """
        Validate simulation results.

        Args:
            results: List of result dictionaries
            expected_agent_count: Expected number of agents (if known)

        Returns:
            List of validation issues
        """
        issues = []

        if not results:
            issues.append(ValidationIssue(
                severity='error',
                category='simulation',
                message='No simulation results provided',
                details={}
            ))
            return issues

        # Check for crashes/errors
        error_count = sum(1 for r in results if r.get('error') or r.get('crashed'))
        if error_count > 0:
            issues.append(ValidationIssue(
                severity='warning',
                category='simulation',
                message=f'{error_count}/{len(results)} simulations had errors',
                details={'error_count': error_count}
            ))

        # Check metric ranges
        survival_rates = [r.get('survival_rate', 0) for r in results if not r.get('error')]
        steps_list = [r.get('steps', 0) for r in results if not r.get('error')]

        if survival_rates:
            avg_survival = np.mean(survival_rates)
            min_survival = np.min(survival_rates)
            max_survival = np.max(survival_rates)

            if avg_survival < 0.1:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='simulation',
                    message=f'Very low average survival rate: {avg_survival:.1%}',
                    details={'avg_survival_rate': avg_survival}
                ))

            if max_survival - min_survival < 0.05:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='simulation',
                    message='Low variance in survival rates - may indicate simulation issues',
                    details={'min': min_survival, 'max': max_survival}
                ))

        if steps_list:
            avg_steps = np.mean(steps_list)
            if avg_steps < 10:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='simulation',
                    message=f'Very short simulations: avg {avg_steps:.1f} steps',
                    details={'avg_steps': avg_steps}
                ))

        # Check agent counts
        if expected_agent_count:
            for i, r in enumerate(results):
                evacuated = r.get('evacuated', 0)
                stuck = r.get('stuck', 0)
                dead = r.get('dead', 0)
                total = evacuated + stuck + dead

                if total != expected_agent_count:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='simulation',
                        message=f'Result {i}: agent count mismatch ({total} vs {expected_agent_count})',
                        details={'result_index': i, 'total': total, 'expected': expected_agent_count}
                    ))

        return issues


class LabelValidator:
    """Validates pairwise labels"""

    def validate(
        self,
        pairs: List[Dict[str, Any]],
        min_pairs: int = 100,
        max_imbalance: float = 0.1
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """
        Validate pairwise labels.

        Args:
            pairs: List of pair dictionaries
            min_pairs: Minimum required pairs
            max_imbalance: Maximum allowed label imbalance (0.1 = 40-60% split)

        Returns:
            Tuple of (issues list, statistics dict)
        """
        issues = []
        stats = {}

        if len(pairs) < min_pairs:
            issues.append(ValidationIssue(
                severity='error',
                category='label',
                message=f'Insufficient pairs: {len(pairs)} < {min_pairs}',
                details={'pair_count': len(pairs), 'minimum': min_pairs}
            ))

        if not pairs:
            return issues, stats

        # Label distribution
        labels = [p.get('label', 0) for p in pairs]
        label_1_ratio = sum(labels) / len(labels)
        stats['label_1_ratio'] = label_1_ratio

        if abs(label_1_ratio - 0.5) > max_imbalance:
            issues.append(ValidationIssue(
                severity='warning',
                category='label',
                message=f'Label imbalance: {label_1_ratio:.1%} label=1',
                details={'label_1_ratio': label_1_ratio}
            ))

        # Score difference distribution
        score_diffs = []
        for p in pairs:
            score_a = p.get('score_a', 0)
            score_b = p.get('score_b', 0)
            score_diffs.append(abs(score_a - score_b))

        stats['avg_score_diff'] = np.mean(score_diffs)
        stats['min_score_diff'] = np.min(score_diffs)
        stats['max_score_diff'] = np.max(score_diffs)

        # Check for degenerate pairs (identical scores)
        zero_diff_count = sum(1 for d in score_diffs if d < 0.001)
        if zero_diff_count > len(pairs) * 0.01:
            issues.append(ValidationIssue(
                severity='warning',
                category='label',
                message=f'{zero_diff_count} pairs have nearly identical scores',
                details={'zero_diff_count': zero_diff_count}
            ))

        # Floor plan coverage
        plan_ids = set()
        for p in pairs:
            plan_ids.add(p.get('floor_plan_id_a'))
            plan_ids.add(p.get('floor_plan_id_b'))

        stats['unique_floor_plans'] = len(plan_ids)

        # Pair type distribution
        pair_types = defaultdict(int)
        for p in pairs:
            pair_types[p.get('pair_type', 'unknown')] += 1
        stats['pair_type_distribution'] = dict(pair_types)

        return issues, stats


class DatasetSplitValidator:
    """Validates train/val/test splits"""

    def validate_split(
        self,
        train_pairs: List[Dict[str, Any]],
        val_pairs: List[Dict[str, Any]],
        test_pairs: List[Dict[str, Any]]
    ) -> List[ValidationIssue]:
        """
        Validate that splits don't leak floor plan information.

        Args:
            train_pairs: Training pairs
            val_pairs: Validation pairs
            test_pairs: Test pairs

        Returns:
            List of validation issues
        """
        issues = []

        def get_plan_ids(pairs):
            ids = set()
            for p in pairs:
                ids.add(p.get('floor_plan_id_a'))
                ids.add(p.get('floor_plan_id_b'))
            return ids

        train_plans = get_plan_ids(train_pairs)
        val_plans = get_plan_ids(val_pairs)
        test_plans = get_plan_ids(test_pairs)

        # Check for overlap
        train_val_overlap = train_plans & val_plans
        train_test_overlap = train_plans & test_plans
        val_test_overlap = val_plans & test_plans

        if train_val_overlap:
            issues.append(ValidationIssue(
                severity='error',
                category='split',
                message=f'Train/val floor plan overlap: {len(train_val_overlap)} plans',
                details={'overlapping_plans': list(train_val_overlap)[:10]}
            ))

        if train_test_overlap:
            issues.append(ValidationIssue(
                severity='error',
                category='split',
                message=f'Train/test floor plan overlap: {len(train_test_overlap)} plans',
                details={'overlapping_plans': list(train_test_overlap)[:10]}
            ))

        if val_test_overlap:
            issues.append(ValidationIssue(
                severity='warning',
                category='split',
                message=f'Val/test floor plan overlap: {len(val_test_overlap)} plans',
                details={'overlapping_plans': list(val_test_overlap)[:10]}
            ))

        # Check split ratios
        total_pairs = len(train_pairs) + len(val_pairs) + len(test_pairs)
        if total_pairs > 0:
            train_ratio = len(train_pairs) / total_pairs
            val_ratio = len(val_pairs) / total_pairs
            test_ratio = len(test_pairs) / total_pairs

            if train_ratio < 0.5:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='split',
                    message=f'Training set too small: {train_ratio:.1%}',
                    details={'train_ratio': train_ratio}
                ))

            if val_ratio < 0.05 or test_ratio < 0.05:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='split',
                    message=f'Val or test set very small (val: {val_ratio:.1%}, test: {test_ratio:.1%})',
                    details={'val_ratio': val_ratio, 'test_ratio': test_ratio}
                ))

        return issues


class DataValidator:
    """Main validator combining all validation checks"""

    def __init__(self):
        self.floor_plan_validator = FloorPlanValidator()
        self.simulation_validator = SimulationResultValidator()
        self.label_validator = LabelValidator()
        self.split_validator = DatasetSplitValidator()

    def validate_floor_plan(
        self,
        grid: np.ndarray,
        exit_positions: List[Tuple[int, int]]
    ) -> ValidationReport:
        """Validate a single floor plan"""
        issues = self.floor_plan_validator.validate(grid, exit_positions)

        return ValidationReport(
            is_valid=not any(i.severity == 'error' for i in issues),
            error_count=sum(1 for i in issues if i.severity == 'error'),
            warning_count=sum(1 for i in issues if i.severity == 'warning'),
            issues=issues,
            statistics={
                'size': grid.shape,
                'passable_ratio': float(np.sum(grid == 0)) / grid.size,
                'exit_count': len(exit_positions)
            }
        )

    def validate_simulation_results(
        self,
        results: List[Dict[str, Any]]
    ) -> ValidationReport:
        """Validate simulation results"""
        issues = self.simulation_validator.validate(results)

        valid_results = [r for r in results if not r.get('error')]
        stats = {}
        if valid_results:
            stats['count'] = len(results)
            stats['error_count'] = len(results) - len(valid_results)
            stats['avg_survival_rate'] = np.mean([r.get('survival_rate', 0) for r in valid_results])
            stats['avg_steps'] = np.mean([r.get('steps', 0) for r in valid_results])

        return ValidationReport(
            is_valid=not any(i.severity == 'error' for i in issues),
            error_count=sum(1 for i in issues if i.severity == 'error'),
            warning_count=sum(1 for i in issues if i.severity == 'warning'),
            issues=issues,
            statistics=stats
        )

    def validate_labels(
        self,
        pairs: List[Dict[str, Any]]
    ) -> ValidationReport:
        """Validate pairwise labels"""
        issues, stats = self.label_validator.validate(pairs)

        return ValidationReport(
            is_valid=not any(i.severity == 'error' for i in issues),
            error_count=sum(1 for i in issues if i.severity == 'error'),
            warning_count=sum(1 for i in issues if i.severity == 'warning'),
            issues=issues,
            statistics=stats
        )

    def validate_dataset(
        self,
        train_pairs: List[Dict[str, Any]],
        val_pairs: List[Dict[str, Any]],
        test_pairs: List[Dict[str, Any]]
    ) -> ValidationReport:
        """Validate complete dataset with splits"""
        all_issues = []

        # Validate splits
        split_issues = self.split_validator.validate_split(train_pairs, val_pairs, test_pairs)
        all_issues.extend(split_issues)

        # Validate each split
        for name, pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
            label_issues, _ = self.label_validator.validate(pairs, min_pairs=10)
            for issue in label_issues:
                issue.message = f"[{name}] {issue.message}"
            all_issues.extend(label_issues)

        total_pairs = len(train_pairs) + len(val_pairs) + len(test_pairs)

        return ValidationReport(
            is_valid=not any(i.severity == 'error' for i in all_issues),
            error_count=sum(1 for i in all_issues if i.severity == 'error'),
            warning_count=sum(1 for i in all_issues if i.severity == 'warning'),
            issues=all_issues,
            statistics={
                'total_pairs': total_pairs,
                'train_pairs': len(train_pairs),
                'val_pairs': len(val_pairs),
                'test_pairs': len(test_pairs),
                'train_ratio': len(train_pairs) / total_pairs if total_pairs > 0 else 0,
                'val_ratio': len(val_pairs) / total_pairs if total_pairs > 0 else 0,
                'test_ratio': len(test_pairs) / total_pairs if total_pairs > 0 else 0
            }
        )


def create_dataset_splits(
    pairs: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split pairs by floor plan (not by individual pair) to prevent data leakage.

    Cross-plan pairs are filtered to ensure both floor_plan_id_a and floor_plan_id_b
    are in the same split, preventing test data from leaking into training.

    Args:
        pairs: All pairwise labels
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        seed: Random seed

    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs)
    """
    rng = np.random.default_rng(seed)

    # Collect all unique floor plan IDs from both sides of pairs
    all_plan_ids = set()
    for pair in pairs:
        all_plan_ids.add(pair.get('floor_plan_id_a'))
        all_plan_ids.add(pair.get('floor_plan_id_b'))

    # Shuffle plan IDs
    plan_ids = list(all_plan_ids)
    rng.shuffle(plan_ids)

    # Split by plans
    n_plans = len(plan_ids)
    n_train = int(n_plans * train_ratio)
    n_val = int(n_plans * val_ratio)

    train_plan_ids = set(plan_ids[:n_train])
    val_plan_ids = set(plan_ids[n_train:n_train + n_val])
    test_plan_ids = set(plan_ids[n_train + n_val:])

    # Assign pairs to splits - BOTH floor plan IDs must be in the same split
    # This prevents cross-plan pairs from leaking data across splits
    train_pairs = []
    val_pairs = []
    test_pairs = []
    filtered_count = 0

    for pair in pairs:
        plan_a = pair.get('floor_plan_id_a')
        plan_b = pair.get('floor_plan_id_b')

        # Check if both plans are in the same split
        if plan_a in train_plan_ids and plan_b in train_plan_ids:
            train_pairs.append(pair)
        elif plan_a in val_plan_ids and plan_b in val_plan_ids:
            val_pairs.append(pair)
        elif plan_a in test_plan_ids and plan_b in test_plan_ids:
            test_pairs.append(pair)
        else:
            # Cross-plan pair spans splits - filter it out to prevent leakage
            filtered_count += 1

    if filtered_count > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Filtered {filtered_count} cross-split pairs to prevent data leakage")

    return train_pairs, val_pairs, test_pairs


if __name__ == '__main__':
    # Test validators
    print("Testing DataValidator...")

    validator = DataValidator()

    # Test floor plan validation
    print("\n1. Floor Plan Validation:")
    grid = np.zeros((30, 30), dtype=np.float32)
    grid[:, :] = -2  # All walls
    grid[1:29, 1:29] = 0  # Inner passable
    grid[0, 15] = 0  # Exit

    report = validator.validate_floor_plan(grid, [(15, 0)])
    report.print_summary()

    # Test label validation
    print("\n2. Label Validation:")
    mock_pairs = []
    for i in range(200):
        mock_pairs.append({
            'floor_plan_id_a': i % 10,
            'floor_plan_id_b': i % 10,
            'score_a': np.random.uniform(0.5, 0.95),
            'score_b': np.random.uniform(0.5, 0.95),
            'label': np.random.randint(0, 2),
            'pair_type': 'within_plan'
        })

    report = validator.validate_labels(mock_pairs)
    report.print_summary()

    # Test split validation
    print("\n3. Dataset Split Validation:")
    train, val, test = create_dataset_splits(mock_pairs, seed=42)
    report = validator.validate_dataset(train, val, test)
    report.print_summary()
