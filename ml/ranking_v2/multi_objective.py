"""
Multi-Objective Ranking for Evacuation Configuration Optimization

Handles multiple conflicting objectives simultaneously:
1. Evacuation Performance (survival rate, evacuation time)
2. Modification Cost (construction, permits)
3. Building Code Compliance
4. Daily Usage Convenience
5. Accessibility Requirements

Approaches:
1. Scalarization: Weighted sum of objectives
2. Pareto Optimization: Find non-dominated solutions
3. Preference Learning: Learn user preferences from comparisons
"""

from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from .model import CrossAttentionRanker, FloorPlanEncoder, ScenarioEncoder
from .config import RankingV2Config


class ObjectiveType(Enum):
    """Types of objectives for optimization."""
    EVACUATION_TIME = "evacuation_time"  # Minimize
    SURVIVAL_RATE = "survival_rate"  # Maximize
    MODIFICATION_COST = "modification_cost"  # Minimize
    COMPLIANCE_SCORE = "compliance_score"  # Maximize
    DAILY_CONVENIENCE = "daily_convenience"  # Maximize
    ACCESSIBILITY = "accessibility"  # Maximize


@dataclass
class ObjectiveConfig:
    """Configuration for a single objective."""
    name: str
    type: ObjectiveType
    weight: float = 1.0
    minimize: bool = True  # True if lower is better
    bounds: Tuple[float, float] = (0.0, 1.0)
    importance: str = "medium"  # "low", "medium", "high", "critical"


@dataclass
class MultiObjectiveResult:
    """Result of multi-objective evaluation."""
    configuration_id: str
    objective_values: Dict[str, float]
    scalarized_score: float
    pareto_rank: int
    dominated_by: List[str]
    dominates: List[str]


@dataclass
class ParetoFront:
    """Collection of Pareto-optimal solutions."""
    solutions: List[MultiObjectiveResult]
    hypervolume: float
    spread: float
    reference_point: Tuple[float, ...]


class MultiObjectiveHead(nn.Module):
    """
    Multi-objective prediction head.

    Predicts multiple objectives from shared representation.
    """

    def __init__(
        self,
        input_dim: int,
        objectives: List[ObjectiveConfig],
        hidden_dim: int = 64,
    ):
        """
        Initialize multi-objective head.

        Args:
            input_dim: Input feature dimension
            objectives: List of objective configurations
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.objectives = objectives
        self.objective_names = [obj.name for obj in objectives]

        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Per-objective heads
        self.heads = nn.ModuleDict()
        for obj in objectives:
            self.heads[obj.name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),
            )

            # Add activation based on objective type
            if obj.type == ObjectiveType.SURVIVAL_RATE:
                self.heads[obj.name].add_module('sigmoid', nn.Sigmoid())
            elif obj.type == ObjectiveType.COMPLIANCE_SCORE:
                self.heads[obj.name].add_module('sigmoid', nn.Sigmoid())

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: Input features (B, D)

        Returns:
            Dict mapping objective name to predictions (B,)
        """
        shared = self.shared(features)

        outputs = {}
        for name in self.objective_names:
            outputs[name] = self.heads[name](shared).squeeze(-1)

        return outputs


class MultiObjectiveRanker(nn.Module):
    """
    Multi-objective ranking model.

    Predicts multiple objectives and supports various aggregation methods.
    """

    def __init__(
        self,
        config: RankingV2Config,
        objectives: List[ObjectiveConfig],
        aggregation: str = "weighted_sum",
    ):
        """
        Initialize multi-objective ranker.

        Args:
            config: Model configuration
            objectives: List of objective configurations
            aggregation: Aggregation method ("weighted_sum", "chebyshev", "hypervolume")
        """
        super().__init__()
        self.config = config
        self.objectives = objectives
        self.aggregation = aggregation

        # Encoders (shared)
        self.encoder = FloorPlanEncoder(config)
        self.scenario_encoder = ScenarioEncoder(config)

        # Multi-objective head
        feature_dim = config.latent_dim + config.scenario_output_dim
        self.mo_head = MultiObjectiveHead(
            input_dim=feature_dim,
            objectives=objectives,
            hidden_dim=config.scoring_hidden_dim,
        )

        # Learnable weights for weighted sum (optional)
        if aggregation == "weighted_sum":
            self.objective_weights = nn.Parameter(
                torch.tensor([obj.weight for obj in objectives])
            )

    def forward(
        self,
        grid: torch.Tensor,
        scenario: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            grid: Floor plan grid (B, 5, H, W)
            scenario: Scenario parameters (B, 4)

        Returns:
            Dict with objective predictions and aggregated score
        """
        # Encode
        latent = self.encoder(grid)
        scenario_feat = self.scenario_encoder(scenario)

        # Combine features
        features = torch.cat([latent, scenario_feat], dim=1)

        # Predict objectives
        objectives = self.mo_head(features)

        # Aggregate
        aggregated = self._aggregate(objectives)

        return {
            'objectives': objectives,
            'aggregated_score': aggregated,
            'latent': latent,
        }

    def _aggregate(self, objectives: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate multiple objectives into single score."""
        if self.aggregation == "weighted_sum":
            return self._weighted_sum(objectives)
        elif self.aggregation == "chebyshev":
            return self._chebyshev(objectives)
        elif self.aggregation == "hypervolume":
            return self._hypervolume_contribution(objectives)
        else:
            return self._weighted_sum(objectives)

    def _weighted_sum(self, objectives: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Weighted sum aggregation."""
        weights = F.softmax(self.objective_weights, dim=0)
        values = []

        for i, obj in enumerate(self.objectives):
            val = objectives[obj.name]
            # Negate if minimizing (so higher aggregate = better)
            if obj.minimize:
                val = -val
            values.append(weights[i] * val)

        return torch.stack(values, dim=0).sum(dim=0)

    def _chebyshev(self, objectives: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Chebyshev (minimax) aggregation."""
        weights = F.softmax(self.objective_weights, dim=0)
        values = []

        for i, obj in enumerate(self.objectives):
            val = objectives[obj.name]
            # Normalize to [0, 1] using bounds
            val_norm = (val - obj.bounds[0]) / (obj.bounds[1] - obj.bounds[0])
            if not obj.minimize:
                val_norm = 1 - val_norm  # Flip for maximization

            values.append(weights[i] * val_norm)

        # Return negative of max (so higher = better)
        return -torch.stack(values, dim=0).max(dim=0)[0]

    def _hypervolume_contribution(
        self,
        objectives: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Estimate hypervolume contribution.

        Note: Exact hypervolume is expensive, this is an approximation.
        """
        # Simple approximation: product of normalized objectives
        values = []
        for obj in self.objectives:
            val = objectives[obj.name]
            val_norm = (val - obj.bounds[0]) / (obj.bounds[1] - obj.bounds[0] + 1e-8)
            if obj.minimize:
                val_norm = 1 - val_norm
            values.append(val_norm.clamp(0, 1))

        return torch.stack(values, dim=0).prod(dim=0)

    def compare_pairwise(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compare two configurations across all objectives.

        Returns:
            Dict with per-objective comparisons and overall preference
        """
        out_a = self.forward(grid_a, scenario_a)
        out_b = self.forward(grid_b, scenario_b)

        comparisons = {}
        for obj in self.objectives:
            name = obj.name
            diff = out_a['objectives'][name] - out_b['objectives'][name]
            if obj.minimize:
                diff = -diff  # Lower is better, so negate
            comparisons[name] = diff

        # Overall preference
        overall = out_a['aggregated_score'] - out_b['aggregated_score']

        return {
            'per_objective': comparisons,
            'overall_preference': overall,
            'score_a': out_a['aggregated_score'],
            'score_b': out_b['aggregated_score'],
        }


class ParetoOptimizer:
    """
    Pareto optimization utilities.

    Finds and analyzes Pareto-optimal solutions.
    """

    def __init__(self, objectives: List[ObjectiveConfig]):
        """
        Initialize Pareto optimizer.

        Args:
            objectives: List of objective configurations
        """
        self.objectives = objectives
        self.objective_names = [obj.name for obj in objectives]

    def dominates(
        self,
        a: Dict[str, float],
        b: Dict[str, float],
    ) -> bool:
        """
        Check if solution a dominates solution b.

        a dominates b if:
        - a is at least as good as b in all objectives
        - a is strictly better than b in at least one objective
        """
        at_least_as_good = True
        strictly_better = False

        for obj in self.objectives:
            val_a = a[obj.name]
            val_b = b[obj.name]

            if obj.minimize:
                if val_a > val_b:
                    at_least_as_good = False
                if val_a < val_b:
                    strictly_better = True
            else:
                if val_a < val_b:
                    at_least_as_good = False
                if val_a > val_b:
                    strictly_better = True

        return at_least_as_good and strictly_better

    def find_pareto_front(
        self,
        solutions: List[Dict[str, float]],
        ids: Optional[List[str]] = None,
    ) -> List[int]:
        """
        Find Pareto-optimal solutions.

        Args:
            solutions: List of objective value dicts
            ids: Optional solution IDs

        Returns:
            Indices of Pareto-optimal solutions
        """
        n = len(solutions)
        is_dominated = [False] * n

        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                if self.dominates(solutions[j], solutions[i]):
                    is_dominated[i] = True
                    break

        return [i for i in range(n) if not is_dominated[i]]

    def compute_pareto_ranks(
        self,
        solutions: List[Dict[str, float]],
    ) -> List[int]:
        """
        Compute Pareto ranks for all solutions.

        Rank 0 = Pareto front
        Rank 1 = Pareto front after removing rank 0
        etc.
        """
        n = len(solutions)
        ranks = [-1] * n
        remaining = list(range(n))
        current_rank = 0

        while remaining:
            # Filter to remaining solutions
            remaining_solutions = [solutions[i] for i in remaining]

            # Find Pareto front of remaining
            front_indices = self.find_pareto_front(remaining_solutions)
            front_original = [remaining[i] for i in front_indices]

            # Assign rank
            for idx in front_original:
                ranks[idx] = current_rank

            # Remove from remaining
            remaining = [i for i in remaining if i not in front_original]
            current_rank += 1

        return ranks

    def compute_hypervolume(
        self,
        solutions: List[Dict[str, float]],
        reference_point: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute hypervolume indicator.

        Args:
            solutions: List of Pareto-optimal solutions
            reference_point: Reference point for hypervolume

        Returns:
            Hypervolume value
        """
        if not solutions:
            return 0.0

        # Default reference point: worst case for each objective
        if reference_point is None:
            reference_point = {}
            for obj in self.objectives:
                values = [s[obj.name] for s in solutions]
                if obj.minimize:
                    reference_point[obj.name] = max(values) * 1.1
                else:
                    reference_point[obj.name] = min(values) * 0.9

        # Normalize objectives
        normalized = []
        for sol in solutions:
            norm = []
            for obj in self.objectives:
                val = sol[obj.name]
                ref = reference_point[obj.name]
                if obj.minimize:
                    norm.append(ref - val)  # Higher = better
                else:
                    norm.append(val - ref)  # Higher = better
            normalized.append(norm)

        # 2D case: exact computation
        if len(self.objectives) == 2:
            return self._hypervolume_2d(normalized)

        # Higher dimensions: Monte Carlo approximation
        return self._hypervolume_monte_carlo(normalized, n_samples=10000)

    def _hypervolume_2d(self, points: List[List[float]]) -> float:
        """Exact 2D hypervolume computation."""
        if not points:
            return 0.0

        # Sort by first objective (descending)
        sorted_points = sorted(points, key=lambda p: p[0], reverse=True)

        hypervolume = 0.0
        prev_y = 0.0

        for x, y in sorted_points:
            if y > prev_y:
                hypervolume += x * (y - prev_y)
                prev_y = y

        return hypervolume

    def _hypervolume_monte_carlo(
        self,
        points: List[List[float]],
        n_samples: int = 10000,
    ) -> float:
        """Monte Carlo hypervolume approximation."""
        if not points:
            return 0.0

        points = np.array(points)
        n_dims = points.shape[1]

        # Bounding box
        mins = points.min(axis=0)
        maxs = points.max(axis=0)

        # Sample random points
        samples = np.random.uniform(
            mins, maxs,
            size=(n_samples, n_dims)
        )

        # Count dominated samples
        dominated = 0
        for sample in samples:
            for point in points:
                if np.all(point >= sample):
                    dominated += 1
                    break

        # Estimate hypervolume
        box_volume = np.prod(maxs - mins)
        return box_volume * dominated / n_samples


class MultiObjectiveLoss(nn.Module):
    """
    Loss function for multi-objective learning.

    Combines per-objective losses with optional Pareto-aware weighting.
    """

    def __init__(
        self,
        objectives: List[ObjectiveConfig],
        loss_type: str = "mse",
        pareto_aware: bool = False,
    ):
        """
        Initialize multi-objective loss.

        Args:
            objectives: List of objective configurations
            loss_type: Per-objective loss type ("mse", "huber", "mae")
            pareto_aware: Use Pareto-aware loss weighting
        """
        super().__init__()

        self.objectives = objectives
        self.pareto_aware = pareto_aware

        if loss_type == "mse":
            self.per_obj_loss = nn.MSELoss(reduction='none')
        elif loss_type == "huber":
            self.per_obj_loss = nn.SmoothL1Loss(reduction='none')
        else:
            self.per_obj_loss = nn.L1Loss(reduction='none')

        # Learnable loss weights (for uncertainty weighting)
        self.log_vars = nn.Parameter(torch.zeros(len(objectives)))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-objective loss.

        Args:
            predictions: Predicted objective values
            targets: Target objective values

        Returns:
            (total_loss, per_objective_losses)
        """
        losses = {}
        total_loss = 0.0

        for i, obj in enumerate(self.objectives):
            name = obj.name
            if name not in targets:
                continue

            pred = predictions[name]
            target = targets[name]

            # Per-objective loss
            loss = self.per_obj_loss(pred, target).mean()

            # Uncertainty weighting (Kendall & Gal)
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]

            # Apply objective importance
            if obj.importance == "critical":
                weighted_loss = weighted_loss * 2.0
            elif obj.importance == "high":
                weighted_loss = weighted_loss * 1.5
            elif obj.importance == "low":
                weighted_loss = weighted_loss * 0.5

            losses[name] = loss.item()
            total_loss = total_loss + weighted_loss

        return total_loss, losses


class PreferenceLearner(nn.Module):
    """
    Learns user preferences from pairwise comparisons.

    Infers objective weights from user feedback.
    """

    def __init__(
        self,
        objectives: List[ObjectiveConfig],
        hidden_dim: int = 64,
    ):
        """
        Initialize preference learner.

        Args:
            objectives: List of objective configurations
            hidden_dim: Hidden dimension
        """
        super().__init__()

        self.objectives = objectives
        n_objectives = len(objectives)

        # Preference model: maps objective differences to preference
        self.preference_net = nn.Sequential(
            nn.Linear(n_objectives, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Inferred weights
        self.weight_net = nn.Sequential(
            nn.Linear(n_objectives, n_objectives),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        objective_diff: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict preference from objective differences.

        Args:
            objective_diff: Difference in objectives (A - B) (B, n_obj)

        Returns:
            Preference logit (B,) - positive means prefer A
        """
        return self.preference_net(objective_diff).squeeze(-1)

    def get_weights(self) -> torch.Tensor:
        """Get inferred objective weights."""
        # Use gradient of preference w.r.t. objectives as weights
        dummy = torch.ones(1, len(self.objectives), requires_grad=True)
        pref = self.preference_net(dummy)
        pref.backward()
        weights = dummy.grad.abs()
        return F.softmax(weights, dim=-1).squeeze()

    def fit(
        self,
        comparisons: List[Dict],
        epochs: int = 100,
        lr: float = 0.01,
    ):
        """
        Fit preference model from comparisons.

        Args:
            comparisons: List of {obj_diff: tensor, preference: 0 or 1}
            epochs: Training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0

            for comp in comparisons:
                obj_diff = comp['obj_diff']
                preference = comp['preference']

                optimizer.zero_grad()
                pred = self.forward(obj_diff.unsqueeze(0))
                loss = F.binary_cross_entropy_with_logits(
                    pred,
                    torch.tensor([preference], dtype=torch.float32)
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()


def create_default_objectives() -> List[ObjectiveConfig]:
    """Create default set of objectives for evacuation optimization."""
    return [
        ObjectiveConfig(
            name="survival_rate",
            type=ObjectiveType.SURVIVAL_RATE,
            weight=1.0,
            minimize=False,
            bounds=(0.0, 1.0),
            importance="critical",
        ),
        ObjectiveConfig(
            name="evacuation_time",
            type=ObjectiveType.EVACUATION_TIME,
            weight=0.8,
            minimize=True,
            bounds=(0.0, 1000.0),
            importance="high",
        ),
        ObjectiveConfig(
            name="modification_cost",
            type=ObjectiveType.MODIFICATION_COST,
            weight=0.5,
            minimize=True,
            bounds=(0.0, 100000.0),
            importance="medium",
        ),
        ObjectiveConfig(
            name="compliance_score",
            type=ObjectiveType.COMPLIANCE_SCORE,
            weight=0.7,
            minimize=False,
            bounds=(0.0, 1.0),
            importance="high",
        ),
        ObjectiveConfig(
            name="accessibility",
            type=ObjectiveType.ACCESSIBILITY,
            weight=0.6,
            minimize=False,
            bounds=(0.0, 1.0),
            importance="medium",
        ),
    ]


def create_multi_objective_ranker(
    config: RankingV2Config,
    objectives: Optional[List[ObjectiveConfig]] = None,
    aggregation: str = "weighted_sum",
) -> MultiObjectiveRanker:
    """
    Factory function to create multi-objective ranker.

    Args:
        config: Model configuration
        objectives: Objective configurations (None = defaults)
        aggregation: Aggregation method

    Returns:
        MultiObjectiveRanker instance
    """
    if objectives is None:
        objectives = create_default_objectives()

    return MultiObjectiveRanker(
        config=config,
        objectives=objectives,
        aggregation=aggregation,
    )
