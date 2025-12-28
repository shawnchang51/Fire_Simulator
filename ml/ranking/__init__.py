"""
Pairwise Ranking Module for Floor Plan Evacuation Quality

This module provides a CNN-based pairwise ranking system for comparing
floor plan configurations based on evacuation quality metrics.

Architecture:
    Grid (4, 96, 128) → FloorPlanEncoder → Latent (K=8)
    Scenario (4,) → ScenarioEncoder → Features (16,)
    Combined (24,) → ScoringHead → Raw Score s(x)
    Pairwise: logit = s(A) - s(B)

Key Components:
    - FloorPlanEncoder: CNN backbone for grid encoding
    - ScenarioEncoder: MLP for scenario parameters
    - PointwiseScorer: Single config → raw score
    - SiameseRanker: Pairwise comparison wrapper
    - RankNetLoss: Logistic pairwise loss (sigmoid applied to logit)
    - MarginHingeLoss: Margin-based pairwise loss (raw scores)
"""

from .config import RankingConfig
from .model import FloorPlanEncoder, ScenarioEncoder, PointwiseScorer, SiameseRanker
from .losses import RankNetLoss, MarginHingeLoss
from .dataset import PairwiseDataset, create_pairwise_dataloaders
from .evaluate import evaluate_pairwise, evaluate_per_plan_ranking

__all__ = [
    'RankingConfig',
    'FloorPlanEncoder',
    'ScenarioEncoder',
    'PointwiseScorer',
    'SiameseRanker',
    'RankNetLoss',
    'MarginHingeLoss',
    'PairwiseDataset',
    'create_pairwise_dataloaders',
    'evaluate_pairwise',
    'evaluate_per_plan_ranking',
]
