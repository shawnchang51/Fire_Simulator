"""
Pairwise Ranking Model for Floor Plan Evacuation Quality

Architecture:
    - FloorPlanEncoder: CNN backbone for grid encoding (4ch → K-dim latent)
    - ScenarioEncoder: MLP for scenario parameters (4 → 16)
    - PointwiseScorer: Combined encoder → raw scalar score s(x)
    - SiameseRanker: Shared PointwiseScorer for pairwise comparison

Key Design Decisions:
    - Pointwise scorer outputs RAW scores, not probabilities
    - Pairwise logit = s(A) - s(B), sigmoid applied only in loss
    - Independent encoding: A and B encoded separately (no cross-attention)
    - Named `final_conv` layer for Grad-CAM robustness
"""

from typing import Tuple

import torch
import torch.nn as nn

from .config import RankingConfig


class FloorPlanEncoder(nn.Module):
    """
    CNN encoder for floor plan grids.

    Input: (B, 5, 96, 128) - 5 channels (wall, passable, doors, exits, valid_mask)
    Output: (B, latent_dim) - K-dimensional latent vector

    Architecture:
        Conv layers with BN, ReLU, MaxPool → AdaptiveAvgPool → Linear projection

    Attributes:
        final_conv: Named reference to last conv layer for Grad-CAM
        latent_dim: Dimension of output latent vector
    """

    def __init__(self, config: RankingConfig):
        super().__init__()

        channels = config.cnn_channels  # Default: [16, 32, 64]
        in_ch = config.num_grid_channels  # 4

        # Build conv layers (all except final)
        layers = []
        for out_ch in channels[:-1]:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)

        # Final conv layer - NAMED for Grad-CAM
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_ch, channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU(inplace=True)
        )

        # Adaptive pooling to fixed spatial size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Linear projection to latent space
        # After pool: (channels[-1], 4, 4) = channels[-1] * 16 features
        projection_input_dim = channels[-1] * 16
        self.project = nn.Sequential(
            nn.Linear(projection_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, config.latent_dim)
        )

        self.latent_dim = config.latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Grid tensor of shape (B, 4, H, W)

        Returns:
            Latent vector of shape (B, latent_dim)
        """
        x = self.conv_layers(x)
        x = self.final_conv(x)  # Named for Grad-CAM hooks
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.project(x)


class ScenarioEncoder(nn.Module):
    """
    MLP encoder for scenario parameters.

    Input: (B, 4) - agent_count, num_fires, fire_spread_rate, fire_discovery_delay
    Output: (B, output_dim) - scenario feature vector

    Architecture: Linear → ReLU → Linear → ReLU
    """

    def __init__(self, config: RankingConfig):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(config.scenario_input_dim, config.scenario_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.scenario_hidden_dim, config.scenario_output_dim),
            nn.ReLU(inplace=True)
        )
        self.output_dim = config.scenario_output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Scenario tensor of shape (B, 4)

        Returns:
            Scenario features of shape (B, output_dim)
        """
        return self.mlp(x)


class PointwiseScorer(nn.Module):
    """
    Pointwise scorer: grid + scenario → raw scalar score s(x).

    Combines FloorPlanEncoder and ScenarioEncoder outputs, then passes
    through a scoring head to produce a raw (unbounded) score.

    Used for:
        - Per-plan ranking metrics (Kendall Tau, Spearman, NDCG)
        - Hinge loss computation
        - Interpretability analysis (Grad-CAM, latent correlation)

    Note: Output is RAW score, not probability. Can be any real number.
    """

    def __init__(self, config: RankingConfig):
        super().__init__()

        self.encoder = FloorPlanEncoder(config)
        self.scenario_encoder = ScenarioEncoder(config)

        # Scoring head: combined features → raw score
        feature_dim = config.latent_dim + config.scenario_output_dim
        self.scoring_head = nn.Sequential(
            nn.Linear(feature_dim, config.scoring_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.scoring_hidden_dim, 1)  # Raw score, NO activation
        )

        self.config = config

    def forward(self, grid: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
        """
        Compute raw score for a configuration.

        Args:
            grid: Grid tensor of shape (B, 4, H, W)
            scenario: Scenario tensor of shape (B, 4)

        Returns:
            Raw scores of shape (B,) - NOT bounded to [0, 1]
        """
        latent = self.encoder(grid)
        scenario_feat = self.scenario_encoder(scenario)
        features = torch.cat([latent, scenario_feat], dim=1)
        return self.scoring_head(features).squeeze(-1)

    def get_latent(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation for interpretability analysis.

        Args:
            grid: Grid tensor of shape (B, 4, H, W)

        Returns:
            Latent vectors of shape (B, latent_dim)
        """
        return self.encoder(grid)

    def get_features(self, grid: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
        """
        Get combined features (latent + scenario).

        Args:
            grid: Grid tensor of shape (B, 4, H, W)
            scenario: Scenario tensor of shape (B, 4)

        Returns:
            Combined features of shape (B, latent_dim + scenario_output_dim)
        """
        latent = self.encoder(grid)
        scenario_feat = self.scenario_encoder(scenario)
        return torch.cat([latent, scenario_feat], dim=1)


class SiameseRanker(nn.Module):
    """
    Siamese wrapper for pairwise ranking.

    Uses a shared PointwiseScorer to encode and score both configurations
    independently, then computes the pairwise logit as s(A) - s(B).

    Key Design:
        - A and B are encoded INDEPENDENTLY (no cross-attention)
        - Returns raw scores and logit, NOT probabilities
        - Sigmoid is applied only in RankNet loss, not here

    Usage:
        model = SiameseRanker(config)
        score_a, score_b, logit = model(grid_a, scenario_a, grid_b, scenario_b)

        # For RankNet loss:
        prob = torch.sigmoid(logit)
        loss = F.binary_cross_entropy(prob, label)

        # For Hinge loss:
        loss = torch.relu(margin - (2*label - 1) * (score_a - score_b))
    """

    def __init__(self, config: RankingConfig):
        super().__init__()

        self.scorer = PointwiseScorer(config)
        self.config = config

    def forward(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute pairwise comparison.

        Args:
            grid_a: Grid tensor for config A, shape (B, 4, H, W)
            scenario_a: Scenario tensor for config A, shape (B, 4)
            grid_b: Grid tensor for config B, shape (B, 4, H, W)
            scenario_b: Scenario tensor for config B, shape (B, 4)

        Returns:
            score_a: Raw scores for A, shape (B,)
            score_b: Raw scores for B, shape (B,)
            logit: s(A) - s(B), shape (B,) - for loss computation
        """
        score_a = self.scorer(grid_a, scenario_a)
        score_b = self.scorer(grid_b, scenario_b)
        logit = score_a - score_b
        return score_a, score_b, logit

    def score_single(self, grid: torch.Tensor, scenario: torch.Tensor) -> torch.Tensor:
        """
        Score a single configuration.

        Args:
            grid: Grid tensor of shape (B, 4, H, W)
            scenario: Scenario tensor of shape (B, 4)

        Returns:
            Raw scores of shape (B,)
        """
        return self.scorer(grid, scenario)

    def get_latent(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation for interpretability.

        Args:
            grid: Grid tensor of shape (B, 4, H, W)

        Returns:
            Latent vectors of shape (B, latent_dim)
        """
        return self.scorer.get_latent(grid)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_ranking_model(config: RankingConfig = None) -> SiameseRanker:
    """
    Factory function to create a ranking model.

    Args:
        config: Model configuration. If None, uses default.

    Returns:
        SiameseRanker model instance
    """
    if config is None:
        config = RankingConfig()
    return SiameseRanker(config)
