"""
Loss Functions for Pairwise Ranking V2

Extends V1 with:
- MultiTaskRankingLoss: Combined ranking + auxiliary losses
- Auxiliary loss scheduling (warmup, decay)
- Per-task loss weighting

Key Design:
    - RankNet: sigmoid(logit) -> BCE with confidence weighting
    - Hinge: max(0, margin - sign * score_diff)
    - Auxiliary: SmoothL1 (Huber) for robustness
    - L1 regularization on latent vectors
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RankingV2Config


class RankNetLoss(nn.Module):
    """
    RankNet logistic pairwise loss.

    Applies sigmoid to the raw logit s(A) - s(B) and computes
    binary cross-entropy with the pairwise label.

    More robust to noisy labels due to smooth gradients.
    Supports optional confidence weighting and label smoothing.

    Reference:
        Burges et al. "Learning to Rank using Gradient Descent" (2005)

    Args:
        sigma: Scaling factor for logit (default: 1.0)
            Higher sigma makes the sigmoid sharper.
        label_smoothing: Smooth labels toward 0.5 to handle noise (default: 0.0)
            With smoothing=0.1, labels become 0.05 and 0.95 instead of 0 and 1.
    """

    def __init__(self, sigma: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.sigma = sigma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logit: torch.Tensor,
        label: torch.Tensor,
        confidence: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute RankNet loss.

        Args:
            logit: s(A) - s(B), raw score difference, shape (B,)
            label: 1 if A > B, else 0, shape (B,)
            confidence: Optional weight per pair, shape (B,)

        Returns:
            Scalar loss value
        """
        # Apply sigmoid to get P(A > B)
        prob = torch.sigmoid(self.sigma * logit)

        # Apply label smoothing for noisy pairs
        target = label.float()
        if self.label_smoothing > 0:
            # Smooth labels toward 0.5
            # label=1 -> 1 - smoothing/2 (e.g., 0.95)
            # label=0 -> smoothing/2 (e.g., 0.05)
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Binary cross-entropy
        loss = F.binary_cross_entropy(
            prob,
            target,
            reduction='none'
        )

        # Apply confidence weighting if provided
        if confidence is not None:
            loss = loss * confidence
            return loss.sum() / confidence.sum()
        else:
            return loss.mean()


class MarginHingeLoss(nn.Module):
    """
    Margin-based hinge loss for pairwise ranking.

    Uses raw scores directly without sigmoid. Enforces that
    the correct pair has a score difference of at least `margin`.

    Good for clean labels where explicit separation is desired.
    Does NOT support confidence weighting by design.

    Args:
        margin: Minimum score difference for correct pairs (default: 0.1)
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        score_a: torch.Tensor,
        score_b: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute margin hinge loss.

        Args:
            score_a: Raw scores for config A, shape (B,)
            score_b: Raw scores for config B, shape (B,)
            label: 1 if A > B, else 0, shape (B,)

        Returns:
            Scalar loss value
        """
        # Convert 0/1 label to -1/+1 sign
        sign = 2 * label.float() - 1

        # Score difference
        score_diff = score_a - score_b

        # Hinge: max(0, margin - sign * score_diff)
        # If label=1 (A>B), want score_diff > margin -> loss = max(0, margin - score_diff)
        # If label=0 (B>A), want score_diff < -margin -> loss = max(0, margin + score_diff)
        loss = torch.relu(self.margin - sign * score_diff)

        return loss.mean()


class MultiTaskRankingLoss(nn.Module):
    """
    Combined loss for ranking + auxiliary tasks.

    Components:
        1. Ranking loss (RankNet or Hinge)
        2. Auxiliary regression losses (SmoothL1/Huber)
        3. L1 regularization on latent vectors

    Loss = w_rank * L_rank + w_aux * sum(w_task * L_task) + l1_lambda * L_l1

    Supports auxiliary loss scheduling:
        - constant: Fixed weight throughout training
        - warmup: Gradually increase aux weight over epochs
        - decay: Gradually decrease aux weight over epochs

    Attributes:
        ranking_loss: Primary ranking loss (RankNet or Hinge)
        aux_losses: Dict of auxiliary task losses
        aux_weights: Per-task weight multipliers
        config: Configuration with loss parameters
    """

    def __init__(self, config: RankingV2Config):
        """
        Initialize multi-task loss.

        Args:
            config: Configuration with loss parameters
        """
        super().__init__()
        self.config = config

        # Primary ranking loss
        if config.loss_type == 'ranknet':
            self.ranking_loss = RankNetLoss(
                sigma=config.sigma,
                label_smoothing=config.label_smoothing
            )
        else:
            self.ranking_loss = MarginHingeLoss(margin=config.margin)

        # Auxiliary task losses
        self.aux_losses = nn.ModuleDict()
        self.aux_weights = {}

        for task in config.auxiliary_tasks:
            # Use SmoothL1 (Huber) for robustness to outliers
            self.aux_losses[task] = nn.SmoothL1Loss(reduction='mean')

            # Per-task weights
            if task == "survival_rate":
                self.aux_weights[task] = config.aux_survival_weight
            elif task == "steps":
                self.aux_weights[task] = config.aux_steps_weight
            elif task == "avg_fire_damage":
                self.aux_weights[task] = config.aux_fire_damage_weight
            else:
                self.aux_weights[task] = 1.0

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        confidence: torch.Tensor,
        aux_targets: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            outputs: Model output dict with keys:
                - score_a, score_b: Raw scores
                - logit: score_a - score_b
                - latent_a, latent_b: Latent vectors
                - aux_a, aux_b: Auxiliary predictions
            labels: Pairwise labels (0/1), shape (B,)
            confidence: Label confidence weights, shape (B,)
            aux_targets: Dict with 'a' and 'b' keys, each mapping task -> target
                Example: {'a': {'survival_rate': tensor}, 'b': {'survival_rate': tensor}}
            epoch: Current epoch (for loss scheduling)

        Returns:
            Dict with:
                - 'total': Total combined loss
                - 'ranking': Ranking loss component
                - 'aux_{task}': Per-task auxiliary losses
                - 'l1': L1 regularization loss
                - 'aux_weight': Current auxiliary weight multiplier
        """
        losses = {}

        # 1. Ranking loss
        if self.config.loss_type == 'ranknet':
            losses['ranking'] = self.ranking_loss(
                outputs['logit'],
                labels,
                confidence
            )
        else:
            losses['ranking'] = self.ranking_loss(
                outputs['score_a'],
                outputs['score_b'],
                labels
            )

        # 2. Auxiliary losses
        total_aux = torch.tensor(0.0, device=labels.device)
        if aux_targets is not None and self.aux_losses:
            for task, loss_fn in self.aux_losses.items():
                if task in outputs.get('aux_a', {}) and 'a' in aux_targets and task in aux_targets['a']:
                    # Loss for configuration A
                    loss_a = loss_fn(
                        outputs['aux_a'][task],
                        aux_targets['a'][task]
                    )
                    # Loss for configuration B
                    loss_b = loss_fn(
                        outputs['aux_b'][task],
                        aux_targets['b'][task]
                    )

                    task_loss = (loss_a + loss_b) / 2
                    losses[f'aux_{task}'] = task_loss
                    total_aux = total_aux + self.aux_weights[task] * task_loss

        # 3. L1 regularization on latent vectors
        if self.config.l1_lambda > 0:
            l1_loss = self.config.l1_lambda * (
                outputs['latent_a'].abs().mean() +
                outputs['latent_b'].abs().mean()
            ) / 2
            losses['l1'] = l1_loss
        else:
            l1_loss = torch.tensor(0.0, device=labels.device)
            losses['l1'] = l1_loss

        # 4. Compute aux weight with schedule
        aux_weight = self._get_aux_weight(epoch)
        losses['aux_weight'] = torch.tensor(aux_weight)

        # Total loss
        losses['total'] = (
            self.config.ranking_loss_weight * losses['ranking'] +
            aux_weight * total_aux +
            l1_loss
        )

        return losses

    def _get_aux_weight(self, epoch: int) -> float:
        """
        Get auxiliary loss weight based on schedule.

        Args:
            epoch: Current epoch

        Returns:
            Weight multiplier for auxiliary losses
        """
        schedule = self.config.aux_loss_schedule
        base_weight = self.config.aux_loss_weight

        if schedule == "constant":
            return base_weight

        elif schedule == "warmup":
            # Gradually increase aux weight from 0 to base_weight
            warmup_epochs = self.config.curriculum_warmup_epochs
            if epoch < warmup_epochs:
                return base_weight * (epoch / warmup_epochs)
            return base_weight

        elif schedule == "decay":
            # Decay aux weight over time (focus more on ranking later)
            decay_rate = 0.95
            return base_weight * (decay_rate ** epoch)

        return base_weight


class FocalRankNetLoss(nn.Module):
    """
    Focal RankNet loss for hard example mining.

    Applies focal weighting to down-weight easy examples and
    focus on hard-to-classify pairs.

    L = -alpha * (1 - p)^gamma * log(p)  for positive class
    L = -(1-alpha) * p^gamma * log(1-p)  for negative class

    Args:
        sigma: Scaling factor for logit
        alpha: Balancing factor (default: 0.5)
        gamma: Focusing parameter (default: 2.0)
    """

    def __init__(
        self,
        sigma: float = 1.0,
        alpha: float = 0.5,
        gamma: float = 2.0
    ):
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logit: torch.Tensor,
        label: torch.Tensor,
        confidence: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute focal RankNet loss.

        Args:
            logit: s(A) - s(B), shape (B,)
            label: 1 if A > B, else 0, shape (B,)
            confidence: Optional weight per pair, shape (B,)

        Returns:
            Scalar loss value
        """
        prob = torch.sigmoid(self.sigma * logit)
        target = label.float()

        # Compute focal weights
        pt = prob * target + (1 - prob) * (1 - target)  # p if label=1, 1-p if label=0
        focal_weight = (1 - pt) ** self.gamma

        # Alpha balancing
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        # Binary cross entropy
        bce = F.binary_cross_entropy(prob, target, reduction='none')

        # Focal loss
        loss = alpha_t * focal_weight * bce

        # Apply confidence weighting
        if confidence is not None:
            loss = loss * confidence
            return loss.sum() / confidence.sum()
        else:
            return loss.mean()


def get_loss_function(config: RankingV2Config) -> nn.Module:
    """
    Factory function to create loss function from config.

    For simple ranking-only training, returns RankNetLoss or MarginHingeLoss.
    For multi-task training, use MultiTaskRankingLoss directly.

    Args:
        config: Configuration with loss parameters

    Returns:
        Loss module
    """
    if config.loss_type == 'ranknet':
        label_smoothing = getattr(config, 'label_smoothing', 0.0)
        return RankNetLoss(sigma=config.sigma, label_smoothing=label_smoothing)
    else:
        return MarginHingeLoss(margin=config.margin)


def create_multi_task_loss(config: RankingV2Config) -> MultiTaskRankingLoss:
    """
    Factory function to create multi-task loss.

    Args:
        config: Configuration with loss parameters

    Returns:
        MultiTaskRankingLoss instance
    """
    return MultiTaskRankingLoss(config)
