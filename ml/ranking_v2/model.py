"""
Pairwise Ranking Model V2 with Cross-Attention and Auxiliary Tasks

Architecture:
    - FloorPlanEncoder: CNN backbone for grid encoding (5ch -> K-dim latent)
    - ScenarioEncoder: MLP for scenario parameters (4 -> 32)
    - CrossAttentionStack: Bidirectional attention between A and B (optional)
    - PointwiseScorer: Combined encoder -> raw scalar score s(x)
    - AuxiliaryHeads: Latent -> survival_rate, steps, etc. (optional)

Key Design Decisions:
    - Cross-attention operates on latent vectors for efficiency
    - Auxiliary heads branch BEFORE cross-attention to predict config-specific metrics
    - Pointwise scoring works with or without cross-attention context
    - Named `final_conv` layer for Grad-CAM robustness
"""

from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RankingV2Config
from .attention import CrossAttentionStack, create_attention_module


class ResidualBlock(nn.Module):
    """
    Residual block with optional downsampling.

    Uses standard pattern: Conv -> BN -> ReLU -> Conv -> BN + skip connection -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection with optional projection
        if in_channels != out_channels or downsample:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out, inplace=True)
        return out


class FloorPlanEncoder(nn.Module):
    """
    CNN encoder for floor plan grids with optional residual connections.

    Input: (B, 5, 96, 128) - 5 channels (wall, passable, doors, exits, valid_mask)
    Output: (B, latent_dim) - K-dimensional latent vector

    Architecture:
        - If use_residual: Stem conv + ResidualBlocks with downsampling -> AdaptiveAvgPool -> Linear
        - Otherwise: Conv layers with BN, ReLU, MaxPool -> AdaptiveAvgPool -> Linear projection

    Attributes:
        final_conv: Named reference to last conv layer for Grad-CAM
        latent_dim: Dimension of output latent vector
        use_residual: Whether residual connections are enabled
    """

    def __init__(self, config: RankingV2Config):
        super().__init__()

        channels = config.cnn_channels  # Default: [32, 64, 128, 256]
        in_ch = config.num_grid_channels  # 5
        use_residual = getattr(config, 'use_residual', True)

        if use_residual:
            # Residual CNN architecture
            # Initial conv to expand channels
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, channels[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True)
            )

            # Residual blocks with downsampling
            blocks = []
            for i in range(len(channels) - 1):
                blocks.append(ResidualBlock(channels[i], channels[i+1], downsample=True))
            self.conv_layers = nn.Sequential(*blocks)

            # Final conv for Grad-CAM (named layer)
            self.final_conv = nn.Sequential(
                nn.Conv2d(channels[-1], channels[-1], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels[-1]),
                nn.ReLU(inplace=True)
            )
        else:
            # Original architecture (for backward compatibility)
            self.stem = None
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

        # Linear projection to latent space with larger intermediate dimension
        # After pool: (channels[-1], 4, 4) = channels[-1] * 16 features
        projection_input_dim = channels[-1] * 16
        intermediate_dim = max(256, config.latent_dim * 2)  # At least 256 or 2x latent_dim
        self.project = nn.Sequential(
            nn.Linear(projection_input_dim, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, config.latent_dim)
        )

        self.latent_dim = config.latent_dim
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Grid tensor of shape (B, 5, H, W)

        Returns:
            Latent vector of shape (B, latent_dim)
        """
        if self.stem is not None:
            x = self.stem(x)
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

    Architecture: Linear -> ReLU -> Linear -> ReLU
    """

    def __init__(self, config: RankingV2Config):
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


class AuxiliaryHead(nn.Module):
    """
    Regression head for auxiliary task prediction.

    Predicts a single metric (e.g., survival_rate) from latent vector.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        task: str,
        dropout: float = 0.1,
    ):
        """
        Initialize auxiliary head.

        Args:
            input_dim: Input dimension (latent_dim)
            hidden_dim: Hidden layer dimension
            task: Task name for determining output activation
            dropout: Dropout rate
        """
        super().__init__()

        self.task = task
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Survival rate should be bounded [0, 1]
        # Others (steps, fire_damage) are unbounded
        self.use_sigmoid = (task == "survival_rate")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            latent: Latent vector of shape (B, D)

        Returns:
            Prediction of shape (B,)
        """
        out = self.net(latent).squeeze(-1)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out


class CrossAttentionRanker(nn.Module):
    """
    Pairwise ranking model V2 with cross-attention and auxiliary tasks.

    Architecture:
        1. FloorPlanEncoder: grid -> latent (shared)
        2. ScenarioEncoder: scenario -> scenario_feat (shared)
        3. AuxiliaryHeads: latent -> metrics (BEFORE cross-attention)
        4. CrossAttentionStack: (latent_a, latent_b) -> (attended_a, attended_b)
        5. ScoringHead: concat(attended, scenario) -> raw score

    Key Design:
        - Cross-attention operates on LATENT level (not feature maps)
        - Auxiliary predictions happen BEFORE cross-attention
        - Pointwise scoring works by skipping cross-attention
        - Returns dict with all outputs for flexible loss computation

    Usage:
        model = CrossAttentionRanker(config)

        # Training: pairwise comparison
        outputs = model(grid_a, scenario_a, grid_b, scenario_b)
        # outputs = {score_a, score_b, logit, latent_a, latent_b, aux_a, aux_b}

        # Evaluation: single config scoring
        score = model.score_single(grid, scenario)
    """

    def __init__(self, config: RankingV2Config):
        super().__init__()
        self.config = config

        # Shared encoders
        self.encoder = FloorPlanEncoder(config)
        self.scenario_encoder = ScenarioEncoder(config)

        # Cross-attention module (optional)
        if config.use_cross_attention:
            # Ensure attention_dim matches latent_dim for residual connections
            if config.attention_dim != config.latent_dim:
                self.latent_proj = nn.Linear(config.latent_dim, config.attention_dim)
                self.latent_unproj = nn.Linear(config.attention_dim, config.latent_dim)
            else:
                self.latent_proj = None
                self.latent_unproj = None

            self.cross_attention = CrossAttentionStack(
                dim=config.attention_dim,
                num_layers=config.num_attention_layers,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout,
                use_ffn=config.use_attention_ffn,
            )
        else:
            self.cross_attention = None
            self.latent_proj = None
            self.latent_unproj = None

        # Scoring head
        feature_dim = config.latent_dim + config.scenario_output_dim
        self.scoring_head = self._build_scoring_head(config, feature_dim)

        # Auxiliary task heads (branch from latent, BEFORE cross-attention)
        self.auxiliary_heads = nn.ModuleDict()
        if config.auxiliary_tasks:
            for task in config.auxiliary_tasks:
                self.auxiliary_heads[task] = AuxiliaryHead(
                    input_dim=config.latent_dim,
                    hidden_dim=config.aux_hidden_dim,
                    task=task,
                    dropout=config.dropout,
                )

    def _build_scoring_head(
        self,
        config: RankingV2Config,
        feature_dim: int
    ) -> nn.Module:
        """Build the scoring head MLP."""
        hidden_dim = config.scoring_hidden_dim
        num_layers = config.scoring_num_layers
        use_layer_norm = config.use_layer_norm
        dropout = config.dropout

        layers = []
        in_features = feature_dim

        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_features = hidden_dim

        # Final output layer (raw score, NO activation)
        layers.append(nn.Linear(hidden_dim, 1))

        return nn.Sequential(*layers)

    def forward(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full pairwise forward pass.

        Args:
            grid_a: Grid tensor for config A, shape (B, 5, H, W)
            scenario_a: Scenario tensor for config A, shape (B, 4)
            grid_b: Grid tensor for config B, shape (B, 5, H, W)
            scenario_b: Scenario tensor for config B, shape (B, 4)
            return_attention: Whether to store attention weights for visualization

        Returns:
            Dict with:
                - score_a, score_b: Raw scores (B,)
                - logit: score_a - score_b (B,)
                - latent_a, latent_b: Latent vectors (B, K)
                - aux_a, aux_b: Auxiliary predictions {task: (B,)}
        """
        # 1. Encode independently
        latent_a = self.encoder(grid_a)
        latent_b = self.encoder(grid_b)

        scenario_feat_a = self.scenario_encoder(scenario_a)
        scenario_feat_b = self.scenario_encoder(scenario_b)

        # 2. Auxiliary predictions (BEFORE cross-attention)
        # These predict config-specific metrics independent of the comparison
        aux_a, aux_b = {}, {}
        if self.auxiliary_heads:
            for task, head in self.auxiliary_heads.items():
                aux_a[task] = head(latent_a)
                aux_b[task] = head(latent_b)

        # 3. Apply cross-attention (if enabled)
        if self.cross_attention is not None:
            # Project to attention dim if needed
            if self.latent_proj is not None:
                latent_a_attn = self.latent_proj(latent_a)
                latent_b_attn = self.latent_proj(latent_b)
            else:
                latent_a_attn = latent_a
                latent_b_attn = latent_b

            attended_a, attended_b = self.cross_attention(
                latent_a_attn, latent_b_attn, return_attention
            )

            # Unproject if needed
            if self.latent_unproj is not None:
                attended_a = self.latent_unproj(attended_a)
                attended_b = self.latent_unproj(attended_b)
        else:
            attended_a, attended_b = latent_a, latent_b

        # 4. Combine with scenario and score
        features_a = torch.cat([attended_a, scenario_feat_a], dim=1)
        features_b = torch.cat([attended_b, scenario_feat_b], dim=1)

        score_a = self.scoring_head(features_a).squeeze(-1)
        score_b = self.scoring_head(features_b).squeeze(-1)
        logit = score_a - score_b

        return {
            'score_a': score_a,
            'score_b': score_b,
            'logit': logit,
            'latent_a': latent_a,
            'latent_b': latent_b,
            'aux_a': aux_a,
            'aux_b': aux_b,
        }

    def score_single(
        self,
        grid: torch.Tensor,
        scenario: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a single configuration (for evaluation).

        Note: This bypasses cross-attention since there's no comparison target.
        The score represents absolute quality, not relative ranking.

        Args:
            grid: Grid tensor of shape (B, 5, H, W)
            scenario: Scenario tensor of shape (B, 4)

        Returns:
            Raw scores of shape (B,)
        """
        latent = self.encoder(grid)
        scenario_feat = self.scenario_encoder(scenario)

        # Skip cross-attention for single config scoring
        features = torch.cat([latent, scenario_feat], dim=1)
        return self.scoring_head(features).squeeze(-1)

    def predict_auxiliary(
        self,
        grid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict auxiliary metrics for a single configuration.

        Args:
            grid: Grid tensor of shape (B, 5, H, W)

        Returns:
            Dict mapping task name to predictions (B,)
        """
        latent = self.encoder(grid)
        predictions = {}
        for task, head in self.auxiliary_heads.items():
            predictions[task] = head(latent)
        return predictions

    def get_latent(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation for interpretability.

        Args:
            grid: Grid tensor of shape (B, 5, H, W)

        Returns:
            Latent vectors of shape (B, latent_dim)
        """
        return self.encoder(grid)

    def get_attention_weights(self) -> Optional[Tuple[list, list]]:
        """
        Get attention weights from cross-attention module.

        Returns:
            Tuple of (weights_a, weights_b) lists, or None if no cross-attention
        """
        if self.cross_attention is not None:
            return self.cross_attention.get_attention_weights()
        return None

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_ranking_model(config: RankingV2Config = None) -> CrossAttentionRanker:
    """
    Factory function to create a ranking model.

    Args:
        config: Model configuration. If None, uses default.

    Returns:
        CrossAttentionRanker model instance
    """
    if config is None:
        config = RankingV2Config()
    return CrossAttentionRanker(config)
