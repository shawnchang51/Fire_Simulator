"""
Loss Functions for Pairwise Ranking

Key Distinction:
    - RankNet: Apply sigmoid to logit internally, use BCE loss
    - Hinge: Use raw scores directly, no sigmoid

Design Decisions:
    - Confidence weighting is applied ONLY to RankNet (handles noisy labels)
    - Hinge loss treats all pairs equally
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankNetLoss(nn.Module):
    """
    RankNet logistic pairwise loss.

    Applies sigmoid to the raw logit s(A) - s(B) and computes
    binary cross-entropy with the pairwise label.

    More robust to noisy labels due to smooth gradients.
    Supports optional confidence weighting.

    Reference:
        Burges et al. "Learning to Rank using Gradient Descent" (2005)

    Args:
        sigma: Scaling factor for logit (default: 1.0)
            Higher sigma makes the sigmoid sharper.
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

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

        # Binary cross-entropy
        loss = F.binary_cross_entropy(
            prob,
            label.float(),
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
        # If label=1 (A>B), want score_diff > margin → loss = max(0, margin - score_diff)
        # If label=0 (B>A), want score_diff < -margin → loss = max(0, margin + score_diff)
        loss = torch.relu(self.margin - sign * score_diff)

        return loss.mean()


class CombinedRankingLoss(nn.Module):
    """
    Combined ranking loss with L1 regularization on latent vectors.

    This is a convenience wrapper that combines the pairwise loss
    with latent sparsity regularization.

    Args:
        loss_type: 'ranknet' or 'hinge'
        margin: Margin for hinge loss
        sigma: Sigma for RankNet
        l1_lambda: L1 regularization strength
    """

    def __init__(
        self,
        loss_type: str = 'ranknet',
        margin: float = 0.1,
        sigma: float = 1.0,
        l1_lambda: float = 0.005
    ):
        super().__init__()

        self.loss_type = loss_type
        self.l1_lambda = l1_lambda

        if loss_type == 'ranknet':
            self.pairwise_loss = RankNetLoss(sigma=sigma)
        else:
            self.pairwise_loss = MarginHingeLoss(margin=margin)

    def forward(
        self,
        score_a: torch.Tensor,
        score_b: torch.Tensor,
        logit: torch.Tensor,
        label: torch.Tensor,
        latent_a: torch.Tensor = None,
        confidence: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            score_a: Raw scores for A, shape (B,)
            score_b: Raw scores for B, shape (B,)
            logit: s(A) - s(B), shape (B,)
            label: Pairwise label, shape (B,)
            latent_a: Latent vectors for A (for L1 reg), shape (B, K)
            confidence: Optional confidence weights, shape (B,)

        Returns:
            Scalar loss value
        """
        # Compute pairwise loss
        if self.loss_type == 'ranknet':
            pair_loss = self.pairwise_loss(logit, label, confidence)
        else:
            pair_loss = self.pairwise_loss(score_a, score_b, label)

        # Add L1 regularization on latent vectors
        total_loss = pair_loss
        if latent_a is not None and self.l1_lambda > 0:
            l1_loss = self.l1_lambda * latent_a.abs().mean()
            total_loss = total_loss + l1_loss

        return total_loss


def get_loss_function(config) -> nn.Module:
    """
    Factory function to create loss function from config.

    Args:
        config: RankingConfig with loss parameters

    Returns:
        Loss module (RankNetLoss or MarginHingeLoss)
    """
    if config.loss_type == 'ranknet':
        return RankNetLoss(sigma=config.sigma)
    else:
        return MarginHingeLoss(margin=config.margin)
