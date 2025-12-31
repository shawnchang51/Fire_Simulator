"""
Uncertainty Quantification for Pairwise Ranking Model V2

Provides methods to estimate prediction uncertainty:
1. MC Dropout: Monte Carlo sampling with dropout during inference
2. Deep Ensembles: Aggregation of multiple independently trained models
3. Evidential Deep Learning: Direct prediction of Dirichlet distribution parameters

Uncertainty estimates are crucial for:
- Identifying when to fall back to expensive simulations
- Active learning sample selection
- Risk-aware decision making in evacuation planning
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

from .model import CrossAttentionRanker
from .config import RankingV2Config


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimation results."""
    mean_logit: torch.Tensor  # Mean prediction (B,)
    std_logit: torch.Tensor   # Standard deviation (B,)
    mean_prob: torch.Tensor   # Mean probability P(A > B) (B,)
    entropy: torch.Tensor     # Predictive entropy (B,)
    epistemic: torch.Tensor   # Epistemic (model) uncertainty (B,)
    aleatoric: torch.Tensor   # Aleatoric (data) uncertainty (B,)
    confidence: torch.Tensor  # Confidence score = 1 - normalized_uncertainty (B,)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary."""
        return {
            'mean_logit': self.mean_logit,
            'std_logit': self.std_logit,
            'mean_prob': self.mean_prob,
            'entropy': self.entropy,
            'epistemic': self.epistemic,
            'aleatoric': self.aleatoric,
            'confidence': self.confidence,
        }


class MCDropoutWrapper(nn.Module):
    """
    Monte Carlo Dropout wrapper for uncertainty estimation.

    Enables dropout during inference and performs multiple forward passes
    to estimate predictive distribution.

    Usage:
        model = CrossAttentionRanker(config)
        mc_model = MCDropoutWrapper(model, n_samples=30)
        uncertainty = mc_model.predict_with_uncertainty(grid_a, scenario_a, grid_b, scenario_b)
    """

    def __init__(
        self,
        model: CrossAttentionRanker,
        n_samples: int = 30,
        dropout_rate: Optional[float] = None,
    ):
        """
        Initialize MC Dropout wrapper.

        Args:
            model: Base ranking model
            n_samples: Number of Monte Carlo samples
            dropout_rate: Override dropout rate (None = use model's dropout)
        """
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        # Store original dropout rates if we want to modify them
        self._original_dropout_rates = {}

    def _enable_dropout(self):
        """Enable dropout during inference."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                module.train()
                if self.dropout_rate is not None:
                    self._original_dropout_rates[name] = module.p
                    module.p = self.dropout_rate

    def _restore_dropout(self):
        """Restore original dropout settings."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                if name in self._original_dropout_rates:
                    module.p = self._original_dropout_rates[name]
        self._original_dropout_rates.clear()

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
    ) -> UncertaintyEstimate:
        """
        Predict with uncertainty estimation using MC Dropout.

        Args:
            grid_a, scenario_a: Config A inputs
            grid_b, scenario_b: Config B inputs

        Returns:
            UncertaintyEstimate with mean, std, entropy, etc.
        """
        self._enable_dropout()

        # Collect samples
        logits = []
        probs = []

        for _ in range(self.n_samples):
            outputs = self.model(grid_a, scenario_a, grid_b, scenario_b)
            logit = outputs['logit']
            prob = torch.sigmoid(logit)
            logits.append(logit)
            probs.append(prob)

        self._restore_dropout()

        # Stack samples: (n_samples, B)
        logits = torch.stack(logits, dim=0)
        probs = torch.stack(probs, dim=0)

        # Compute statistics
        mean_logit = logits.mean(dim=0)
        std_logit = logits.std(dim=0)
        mean_prob = probs.mean(dim=0)

        # Predictive entropy: -sum(p * log(p))
        # For binary classification: -p*log(p) - (1-p)*log(1-p)
        eps = 1e-8
        entropy = -(mean_prob * torch.log(mean_prob + eps) +
                   (1 - mean_prob) * torch.log(1 - mean_prob + eps))

        # Decompose uncertainty (approximation)
        # Epistemic: variance of mean predictions across samples
        epistemic = probs.var(dim=0)

        # Aleatoric: mean of per-sample variance (approximated by entropy of mean)
        aleatoric = entropy - epistemic.clamp(min=0)
        aleatoric = aleatoric.clamp(min=0)

        # Confidence: 1 - normalized uncertainty (higher = more confident)
        max_entropy = np.log(2)  # Maximum entropy for binary
        confidence = 1.0 - (entropy / max_entropy)

        return UncertaintyEstimate(
            mean_logit=mean_logit,
            std_logit=std_logit,
            mean_prob=mean_prob,
            entropy=entropy,
            epistemic=epistemic,
            aleatoric=aleatoric,
            confidence=confidence,
        )

    def forward(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Standard forward pass (training mode)."""
        return self.model(grid_a, scenario_a, grid_b, scenario_b)


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble for uncertainty estimation.

    Maintains multiple independently trained models and aggregates
    their predictions for uncertainty estimation.

    Usage:
        ensemble = DeepEnsemble(config, n_models=5)
        ensemble.train_member(0, train_loader, val_loader, device)
        # ... train other members ...
        uncertainty = ensemble.predict_with_uncertainty(grid_a, scenario_a, grid_b, scenario_b)
    """

    def __init__(
        self,
        config: RankingV2Config,
        n_models: int = 5,
    ):
        """
        Initialize Deep Ensemble.

        Args:
            config: Model configuration
            n_models: Number of ensemble members
        """
        super().__init__()
        self.config = config
        self.n_models = n_models

        # Create ensemble members
        self.models = nn.ModuleList([
            CrossAttentionRanker(config) for _ in range(n_models)
        ])

        # Track which models have been trained
        self._trained = [False] * n_models

    def load_member(self, idx: int, checkpoint_path: str, device: torch.device):
        """
        Load a pre-trained model as ensemble member.

        Args:
            idx: Member index
            checkpoint_path: Path to checkpoint file
            device: Device to load to
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.models[idx].load_state_dict(checkpoint['model_state_dict'])
        self._trained[idx] = True

    def save_member(self, idx: int, checkpoint_path: str):
        """Save an ensemble member."""
        torch.save({
            'model_state_dict': self.models[idx].state_dict(),
            'member_idx': idx,
        }, checkpoint_path)

    @property
    def num_trained(self) -> int:
        """Number of trained ensemble members."""
        return sum(self._trained)

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
    ) -> UncertaintyEstimate:
        """
        Predict with uncertainty using ensemble disagreement.

        Only uses trained members for prediction.
        """
        if self.num_trained == 0:
            raise RuntimeError("No trained ensemble members available")

        # Collect predictions from trained members
        logits = []
        probs = []

        for idx, model in enumerate(self.models):
            if not self._trained[idx]:
                continue
            model.eval()
            outputs = model(grid_a, scenario_a, grid_b, scenario_b)
            logit = outputs['logit']
            prob = torch.sigmoid(logit)
            logits.append(logit)
            probs.append(prob)

        # Stack: (n_trained, B)
        logits = torch.stack(logits, dim=0)
        probs = torch.stack(probs, dim=0)

        # Statistics
        mean_logit = logits.mean(dim=0)
        std_logit = logits.std(dim=0)
        mean_prob = probs.mean(dim=0)

        # Entropy
        eps = 1e-8
        entropy = -(mean_prob * torch.log(mean_prob + eps) +
                   (1 - mean_prob) * torch.log(1 - mean_prob + eps))

        # Ensemble disagreement as epistemic uncertainty
        epistemic = probs.var(dim=0)

        # Aleatoric estimated from average entropy per member
        member_entropies = []
        for p in probs:
            h = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
            member_entropies.append(h)
        aleatoric = torch.stack(member_entropies, dim=0).mean(dim=0)

        # Confidence
        max_entropy = np.log(2)
        confidence = 1.0 - (entropy / max_entropy)

        return UncertaintyEstimate(
            mean_logit=mean_logit,
            std_logit=std_logit,
            mean_prob=mean_prob,
            entropy=entropy,
            epistemic=epistemic,
            aleatoric=aleatoric,
            confidence=confidence,
        )

    def forward(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using ensemble mean."""
        uncertainty = self.predict_with_uncertainty(
            grid_a, scenario_a, grid_b, scenario_b
        )
        return {
            'logit': uncertainty.mean_logit,
            'score_a': torch.zeros_like(uncertainty.mean_logit),  # Not available
            'score_b': torch.zeros_like(uncertainty.mean_logit),
            'uncertainty': uncertainty.to_dict(),
        }


class EvidentialRankingHead(nn.Module):
    """
    Evidential Deep Learning head for ranking.

    Instead of predicting point estimates, predicts parameters of a
    Dirichlet distribution over class probabilities, enabling direct
    uncertainty quantification without sampling.

    Reference: "Evidential Deep Learning to Quantify Classification Uncertainty"
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize evidential head.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Output 2 evidence values (for binary classification: A > B, A < B)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
            nn.Softplus(),  # Evidence must be positive
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: Combined features (B, D)

        Returns:
            Dict with:
                - alpha: Dirichlet parameters (B, 2)
                - prob: Expected probability P(A > B) (B,)
                - uncertainty: Total uncertainty (B,)
                - epistemic: Epistemic uncertainty (B,)
        """
        # Predict evidence
        evidence = self.net(features)  # (B, 2)

        # Dirichlet parameters: alpha = evidence + 1
        alpha = evidence + 1.0

        # Strength (total evidence)
        S = alpha.sum(dim=-1, keepdim=True)  # (B, 1)

        # Expected probabilities
        probs = alpha / S  # (B, 2)
        prob_a_wins = probs[:, 0]  # P(A > B)

        # Total uncertainty: K / S (where K=2 for binary)
        total_uncertainty = 2.0 / S.squeeze(-1)

        # Epistemic uncertainty approximation
        # Higher when evidence is low
        epistemic = total_uncertainty

        return {
            'alpha': alpha,
            'prob': prob_a_wins,
            'uncertainty': total_uncertainty,
            'epistemic': epistemic,
        }


class EvidentialLoss(nn.Module):
    """
    Loss function for Evidential Deep Learning.

    Combines:
    1. Type II Maximum Likelihood (negative log-likelihood of Dirichlet)
    2. KL divergence regularizer to prevent evidence inflation
    """

    def __init__(
        self,
        lambda_kl: float = 0.1,
        annealing_step: int = 10,
    ):
        """
        Initialize evidential loss.

        Args:
            lambda_kl: KL divergence regularization weight
            annealing_step: Epochs for KL annealing (starts at 0, reaches lambda_kl)
        """
        super().__init__()
        self.lambda_kl = lambda_kl
        self.annealing_step = annealing_step

    def forward(
        self,
        alpha: torch.Tensor,
        labels: torch.Tensor,
        current_epoch: int = 0,
    ) -> torch.Tensor:
        """
        Compute evidential loss.

        Args:
            alpha: Dirichlet parameters (B, 2)
            labels: Ground truth labels (B,) - 1 if A > B, 0 otherwise
            current_epoch: Current training epoch for annealing

        Returns:
            Loss value
        """
        # Convert labels to one-hot
        y = F.one_hot(labels.long(), num_classes=2).float()  # (B, 2)

        S = alpha.sum(dim=-1, keepdim=True)  # (B, 1)

        # Type II Maximum Likelihood loss
        # L = sum_k y_k * (log(S) - log(alpha_k))
        loss_nll = (y * (torch.log(S) - torch.log(alpha))).sum(dim=-1).mean()

        # KL divergence regularizer
        # Encourages uniform distribution for wrong predictions
        alpha_tilde = y + (1 - y) * alpha
        S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)

        # KL(Dir(alpha_tilde) || Dir(1, 1))
        kl = torch.lgamma(S_tilde.squeeze(-1)) - \
             torch.lgamma(torch.tensor(2.0, device=alpha.device)) - \
             (torch.lgamma(alpha_tilde)).sum(dim=-1) + \
             ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) -
              torch.digamma(S_tilde))).sum(dim=-1)

        loss_kl = kl.mean()

        # Annealing coefficient
        anneal_coef = min(1.0, current_epoch / max(1, self.annealing_step))

        return loss_nll + self.lambda_kl * anneal_coef * loss_kl


class EvidentialRanker(nn.Module):
    """
    Full evidential ranking model with uncertainty quantification.

    Replaces the scoring head with an evidential head for direct
    uncertainty estimation without Monte Carlo sampling.
    """

    def __init__(self, config: RankingV2Config):
        """
        Initialize evidential ranker.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Use the same encoders as CrossAttentionRanker
        from .model import FloorPlanEncoder, ScenarioEncoder
        from .attention import CrossAttentionStack

        self.encoder = FloorPlanEncoder(config)
        self.scenario_encoder = ScenarioEncoder(config)

        # Optional cross-attention
        if config.use_cross_attention:
            self.cross_attention = CrossAttentionStack(
                dim=config.attention_dim,
                num_layers=config.num_attention_layers,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout,
                use_ffn=config.use_attention_ffn,
            )
        else:
            self.cross_attention = None

        # Evidential head instead of scoring head
        feature_dim = config.latent_dim + config.scenario_output_dim
        # For pairwise comparison, we concatenate difference features
        self.evidential_head = EvidentialRankingHead(
            input_dim=feature_dim * 2,  # Concatenate A and B features
            hidden_dim=config.scoring_hidden_dim,
        )

    def forward(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.

        Returns:
            Dict with alpha, prob, uncertainty, epistemic, logit
        """
        # Encode
        latent_a = self.encoder(grid_a)
        latent_b = self.encoder(grid_b)

        scenario_feat_a = self.scenario_encoder(scenario_a)
        scenario_feat_b = self.scenario_encoder(scenario_b)

        # Cross-attention
        if self.cross_attention is not None:
            attended_a, attended_b = self.cross_attention(latent_a, latent_b)
        else:
            attended_a, attended_b = latent_a, latent_b

        # Combine features
        features_a = torch.cat([attended_a, scenario_feat_a], dim=1)
        features_b = torch.cat([attended_b, scenario_feat_b], dim=1)

        # Concatenate for pairwise comparison
        features = torch.cat([features_a, features_b], dim=1)

        # Evidential prediction
        outputs = self.evidential_head(features)

        # Add logit for compatibility
        # logit > 0 means A > B
        outputs['logit'] = torch.log(outputs['prob'] + 1e-8) - \
                          torch.log(1 - outputs['prob'] + 1e-8)

        return outputs

    def predict_with_uncertainty(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
    ) -> UncertaintyEstimate:
        """
        Get uncertainty estimate (no sampling required).
        """
        with torch.no_grad():
            outputs = self.forward(grid_a, scenario_a, grid_b, scenario_b)

        prob = outputs['prob']
        uncertainty = outputs['uncertainty']
        epistemic = outputs['epistemic']

        # Compute entropy
        eps = 1e-8
        entropy = -(prob * torch.log(prob + eps) +
                   (1 - prob) * torch.log(1 - prob + eps))

        # Aleatoric = total - epistemic (approximation)
        aleatoric = (entropy - epistemic).clamp(min=0)

        # Confidence
        max_entropy = np.log(2)
        confidence = 1.0 - (entropy / max_entropy)

        return UncertaintyEstimate(
            mean_logit=outputs['logit'],
            std_logit=uncertainty,  # Use uncertainty as std proxy
            mean_prob=prob,
            entropy=entropy,
            epistemic=epistemic,
            aleatoric=aleatoric,
            confidence=confidence,
        )


class UncertaintyCalibrator:
    """
    Calibrates uncertainty estimates using held-out validation data.

    Maps raw uncertainty scores to calibrated probabilities that
    better reflect true error rates.
    """

    def __init__(self, n_bins: int = 10):
        """
        Initialize calibrator.

        Args:
            n_bins: Number of bins for calibration
        """
        self.n_bins = n_bins
        self.bin_boundaries = None
        self.bin_accuracies = None
        self._fitted = False

    def fit(
        self,
        uncertainties: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Fit calibrator on validation data.

        Args:
            uncertainties: Raw uncertainty scores
            predictions: Binary predictions (0 or 1)
            labels: Ground truth labels
        """
        # Sort by uncertainty
        sorted_idx = np.argsort(uncertainties)
        sorted_correct = (predictions == labels)[sorted_idx]
        sorted_uncertainty = uncertainties[sorted_idx]

        # Create bins
        bin_size = len(uncertainties) // self.n_bins
        self.bin_boundaries = []
        self.bin_accuracies = []

        for i in range(self.n_bins):
            start = i * bin_size
            end = start + bin_size if i < self.n_bins - 1 else len(uncertainties)

            bin_correct = sorted_correct[start:end]
            bin_uncertainty = sorted_uncertainty[start:end]

            self.bin_boundaries.append(bin_uncertainty[-1])
            self.bin_accuracies.append(bin_correct.mean())

        self._fitted = True

    def calibrate(self, uncertainties: np.ndarray) -> np.ndarray:
        """
        Map raw uncertainties to calibrated values.

        Args:
            uncertainties: Raw uncertainty scores

        Returns:
            Calibrated uncertainties (expected error rates)
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        calibrated = np.zeros_like(uncertainties)

        for i, u in enumerate(uncertainties):
            for j, boundary in enumerate(self.bin_boundaries):
                if u <= boundary:
                    calibrated[i] = 1.0 - self.bin_accuracies[j]
                    break
            else:
                calibrated[i] = 1.0 - self.bin_accuracies[-1]

        return calibrated


def create_uncertainty_model(
    config: RankingV2Config,
    method: str = "mc_dropout",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create uncertainty-aware model.

    Args:
        config: Model configuration
        method: Uncertainty method ("mc_dropout", "ensemble", "evidential")
        **kwargs: Method-specific arguments

    Returns:
        Uncertainty-aware model
    """
    if method == "mc_dropout":
        base_model = CrossAttentionRanker(config)
        return MCDropoutWrapper(
            base_model,
            n_samples=kwargs.get("n_samples", 30),
            dropout_rate=kwargs.get("dropout_rate", None),
        )
    elif method == "ensemble":
        return DeepEnsemble(
            config,
            n_models=kwargs.get("n_models", 5),
        )
    elif method == "evidential":
        return EvidentialRanker(config)
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")


def compute_uncertainty_metrics(
    uncertainty_estimates: List[UncertaintyEstimate],
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute uncertainty-related evaluation metrics.

    Args:
        uncertainty_estimates: List of UncertaintyEstimate objects
        predictions: Binary predictions
        labels: Ground truth labels

    Returns:
        Dict with:
            - ece: Expected Calibration Error
            - brier: Brier score
            - auroc_uncertainty: AUROC for error detection using uncertainty
            - sparsification_auc: AUC for sparsification curve
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss

    # Concatenate estimates
    confidences = torch.cat([e.confidence for e in uncertainty_estimates]).numpy()
    mean_probs = torch.cat([e.mean_prob for e in uncertainty_estimates]).numpy()

    # Binary correctness
    correct = (predictions == labels).astype(float)

    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)

    # Brier score
    brier = brier_score_loss(labels, mean_probs)

    # AUROC for error detection
    # High uncertainty should correlate with errors
    errors = 1 - correct
    uncertainties = 1 - confidences
    try:
        auroc_uncertainty = roc_auc_score(errors, uncertainties)
    except ValueError:
        auroc_uncertainty = 0.5

    # Sparsification AUC
    # Remove most uncertain predictions and measure accuracy improvement
    n_points = 20
    fractions = np.linspace(0, 0.9, n_points)
    accuracies = []

    sorted_idx = np.argsort(confidences)[::-1]  # Most confident first

    for frac in fractions:
        n_keep = max(1, int((1 - frac) * len(correct)))
        kept_idx = sorted_idx[:n_keep]
        acc = correct[kept_idx].mean()
        accuracies.append(acc)

    # AUC of sparsification curve (normalized)
    sparsification_auc = np.trapz(accuracies, fractions) / 0.9

    return {
        'ece': ece,
        'brier': brier,
        'auroc_uncertainty': auroc_uncertainty,
        'sparsification_auc': sparsification_auc,
    }
