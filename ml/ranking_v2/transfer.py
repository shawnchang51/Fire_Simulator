"""
Cross-Building Transfer Learning

Enables the ranking model to transfer knowledge between different
building types, reducing the need for extensive simulation on new buildings.

Approaches:
1. Domain Adaptation: Align feature distributions across building types
2. Meta-Learning (MAML): Learn initialization that adapts quickly
3. Few-Shot Learning: Adapt with minimal labeled data
4. Feature Extraction: Use pre-trained encoder with fine-tuned head
"""

from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy
from collections import OrderedDict

from .model import CrossAttentionRanker, FloorPlanEncoder, ScenarioEncoder
from .config import RankingV2Config


@dataclass
class BuildingDomain:
    """Represents a building type domain."""
    name: str
    characteristics: Dict[str, any]  # e.g., {"floors": 1, "type": "residential"}
    n_samples: int = 0
    train_loader: Optional[torch.utils.data.DataLoader] = None
    val_loader: Optional[torch.utils.data.DataLoader] = None


@dataclass
class TransferResult:
    """Result of transfer learning."""
    source_domain: str
    target_domain: str
    pre_transfer_accuracy: float
    post_transfer_accuracy: float
    n_target_samples_used: int
    adaptation_epochs: int
    transfer_method: str


class DomainEncoder(nn.Module):
    """
    Domain-aware encoder with domain-specific batch normalization.

    Uses separate BN statistics for each domain while sharing conv weights.
    """

    def __init__(
        self,
        base_encoder: FloorPlanEncoder,
        domains: List[str],
    ):
        """
        Initialize domain encoder.

        Args:
            base_encoder: Pre-trained encoder
            domains: List of domain names
        """
        super().__init__()

        self.base_encoder = base_encoder
        self.domains = domains
        self.current_domain = domains[0] if domains else "default"

        # Create domain-specific BN layers
        self.domain_bns = nn.ModuleDict()
        for domain in domains:
            domain_bns = {}
            for name, module in base_encoder.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    domain_bns[name] = nn.BatchNorm2d(
                        module.num_features,
                        eps=module.eps,
                        momentum=module.momentum,
                        affine=module.affine,
                    )
            self.domain_bns[domain] = nn.ModuleDict(domain_bns)

    def set_domain(self, domain: str):
        """Set current domain for forward pass."""
        if domain in self.domains:
            self.current_domain = domain
        else:
            raise ValueError(f"Unknown domain: {domain}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with domain-specific batch normalization.
        """
        # This is a simplified implementation
        # In practice, would need to hook into the encoder's forward
        return self.base_encoder(x)


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for adversarial domain adaptation.

    Tries to distinguish which domain a latent representation comes from.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_domains: int = 2,
    ):
        """
        Initialize discriminator.

        Args:
            input_dim: Latent vector dimension
            hidden_dim: Hidden layer dimension
            n_domains: Number of domains
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Latent vectors (B, D)

        Returns:
            Domain logits (B, n_domains)
        """
        return self.net(x)


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.

    Forward: identity
    Backward: negate and scale gradients
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DomainAdaptationRanker(nn.Module):
    """
    Domain-adversarial ranking model.

    Uses adversarial training to learn domain-invariant representations.
    """

    def __init__(
        self,
        config: RankingV2Config,
        n_domains: int = 2,
        lambda_domain: float = 0.1,
    ):
        """
        Initialize domain adaptation ranker.

        Args:
            config: Model configuration
            n_domains: Number of domains
            lambda_domain: Domain adversarial loss weight
        """
        super().__init__()

        self.base_model = CrossAttentionRanker(config)
        self.domain_discriminator = DomainDiscriminator(
            input_dim=config.latent_dim,
            n_domains=n_domains,
        )
        self.lambda_domain = lambda_domain
        self.n_domains = n_domains

    def forward(
        self,
        grid_a: torch.Tensor,
        scenario_a: torch.Tensor,
        grid_b: torch.Tensor,
        scenario_b: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        return_domain: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional domain adversarial training.

        Args:
            grid_a, scenario_a, grid_b, scenario_b: Inputs
            domain_labels: Domain labels for adversarial training
            return_domain: Whether to return domain predictions

        Returns:
            Model outputs with optional domain predictions
        """
        outputs = self.base_model(grid_a, scenario_a, grid_b, scenario_b)

        if return_domain or domain_labels is not None:
            # Get latent representations
            latent_a = outputs['latent_a']
            latent_b = outputs['latent_b']

            # Apply gradient reversal
            reversed_a = GradientReversalLayer.apply(latent_a, self.lambda_domain)
            reversed_b = GradientReversalLayer.apply(latent_b, self.lambda_domain)

            # Domain prediction
            domain_pred_a = self.domain_discriminator(reversed_a)
            domain_pred_b = self.domain_discriminator(reversed_b)

            outputs['domain_pred_a'] = domain_pred_a
            outputs['domain_pred_b'] = domain_pred_b

        return outputs

    def compute_domain_loss(
        self,
        domain_pred: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute domain classification loss."""
        return F.cross_entropy(domain_pred, domain_labels)


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning for few-shot adaptation.

    Learns an initialization that can quickly adapt to new building types.
    """

    def __init__(
        self,
        model: CrossAttentionRanker,
        inner_lr: float = 0.01,
        n_inner_steps: int = 5,
    ):
        """
        Initialize MAML.

        Args:
            model: Base ranking model
            inner_lr: Learning rate for inner loop
            n_inner_steps: Number of inner loop steps
        """
        super().__init__()

        self.model = model
        self.inner_lr = inner_lr
        self.n_inner_steps = n_inner_steps

    def adapt(
        self,
        support_data: Dict[str, torch.Tensor],
        clone: bool = True,
    ) -> nn.Module:
        """
        Adapt model to support set.

        Args:
            support_data: Support set data
            clone: Whether to clone model (True for training)

        Returns:
            Adapted model
        """
        if clone:
            adapted_model = deepcopy(self.model)
        else:
            adapted_model = self.model

        # Enable gradients
        for param in adapted_model.parameters():
            param.requires_grad = True

        # Inner loop optimization
        optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.inner_lr,
        )

        for _ in range(self.n_inner_steps):
            optimizer.zero_grad()

            outputs = adapted_model(
                support_data['grid_a'],
                support_data['scenario_a'],
                support_data['grid_b'],
                support_data['scenario_b'],
            )

            loss = F.binary_cross_entropy_with_logits(
                outputs['logit'],
                support_data['labels'].float(),
            )

            loss.backward()
            optimizer.step()

        return adapted_model

    def forward(
        self,
        support_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Meta-learning forward pass.

        Args:
            support_data: Support set for adaptation
            query_data: Query set for evaluation

        Returns:
            (query_loss, query_logits)
        """
        # Adapt to support set
        adapted_model = self.adapt(support_data, clone=True)

        # Evaluate on query set
        with torch.no_grad():
            outputs = adapted_model(
                query_data['grid_a'],
                query_data['scenario_a'],
                query_data['grid_b'],
                query_data['scenario_b'],
            )

        query_logits = outputs['logit']
        query_loss = F.binary_cross_entropy_with_logits(
            query_logits,
            query_data['labels'].float(),
        )

        return query_loss, query_logits


class FeatureExtractor:
    """
    Feature extraction for transfer learning.

    Freezes encoder and fine-tunes only the scoring head.
    """

    def __init__(
        self,
        model: CrossAttentionRanker,
        freeze_encoder: bool = True,
        freeze_attention: bool = False,
    ):
        """
        Initialize feature extractor.

        Args:
            model: Pre-trained model
            freeze_encoder: Whether to freeze encoder
            freeze_attention: Whether to freeze attention layers
        """
        self.model = model

        if freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False

        if freeze_attention and model.cross_attention is not None:
            for param in model.cross_attention.parameters():
                param.requires_grad = False

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def fine_tune(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
        lr: float = 1e-3,
        device: torch.device = torch.device('cpu'),
    ) -> Dict[str, List[float]]:
        """
        Fine-tune on target domain.

        Args:
            train_loader: Target domain training data
            val_loader: Target domain validation data
            epochs: Number of epochs
            lr: Learning rate
            device: Device

        Returns:
            Training history
        """
        self.model.to(device)
        optimizer = torch.optim.Adam(self.get_trainable_params(), lr=lr)

        history = {'train_loss': [], 'val_acc': []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()

                outputs = self.model(
                    batch['grid_a'].to(device),
                    batch['scenario_a'].to(device),
                    batch['grid_b'].to(device),
                    batch['scenario_b'].to(device),
                )

                loss = F.binary_cross_entropy_with_logits(
                    outputs['logit'],
                    batch['label'].to(device).float(),
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            history['train_loss'].append(total_loss / n_batches)

            # Validation
            if val_loader is not None:
                self.model.eval()
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        outputs = self.model(
                            batch['grid_a'].to(device),
                            batch['scenario_a'].to(device),
                            batch['grid_b'].to(device),
                            batch['scenario_b'].to(device),
                        )

                        pred = (outputs['logit'] > 0).long()
                        correct += (pred == batch['label'].to(device)).sum().item()
                        total += batch['label'].size(0)

                history['val_acc'].append(correct / total)

        return history


class DomainBank:
    """
    Stores and manages domain-specific knowledge.

    Enables quick adaptation to seen domains.
    """

    def __init__(self, base_model: CrossAttentionRanker):
        """
        Initialize domain bank.

        Args:
            base_model: Base ranking model
        """
        self.base_model = base_model
        self.domains: Dict[str, Dict] = {}

    def add_domain(
        self,
        domain_name: str,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Add a new domain to the bank.

        Args:
            domain_name: Name of the domain
            train_loader: Training data for the domain
            epochs: Fine-tuning epochs
            device: Device
        """
        # Clone model for this domain
        domain_model = deepcopy(self.base_model)

        # Fine-tune on domain data
        extractor = FeatureExtractor(domain_model, freeze_encoder=True)
        history = extractor.fine_tune(
            train_loader,
            epochs=epochs,
            device=device,
        )

        # Store domain information
        self.domains[domain_name] = {
            'model_state': domain_model.state_dict(),
            'training_history': history,
            'n_samples': len(train_loader.dataset),
        }

    def get_domain_model(self, domain_name: str) -> CrossAttentionRanker:
        """
        Get model adapted for a specific domain.

        Args:
            domain_name: Name of the domain

        Returns:
            Domain-adapted model
        """
        if domain_name not in self.domains:
            raise ValueError(f"Unknown domain: {domain_name}")

        model = deepcopy(self.base_model)
        model.load_state_dict(self.domains[domain_name]['model_state'])
        return model

    def find_similar_domain(
        self,
        new_domain_data: torch.utils.data.DataLoader,
        device: torch.device = torch.device('cpu'),
    ) -> str:
        """
        Find the most similar existing domain.

        Args:
            new_domain_data: Sample from new domain
            device: Device

        Returns:
            Name of most similar domain
        """
        self.base_model.to(device)
        self.base_model.eval()

        # Get latent statistics for new domain
        new_latents = []
        with torch.no_grad():
            for batch in new_domain_data:
                latent = self.base_model.encoder(batch['grid_a'].to(device))
                new_latents.append(latent)

        new_latents = torch.cat(new_latents, dim=0)
        new_mean = new_latents.mean(dim=0)
        new_std = new_latents.std(dim=0)

        # Compare with each stored domain
        best_domain = None
        best_distance = float('inf')

        for domain_name in self.domains:
            domain_model = self.get_domain_model(domain_name)
            domain_model.to(device)
            domain_model.eval()

            # This is a simplified similarity measure
            # In practice, would use distribution matching
            distance = 0  # Placeholder

            if distance < best_distance:
                best_distance = distance
                best_domain = domain_name

        return best_domain or list(self.domains.keys())[0]


class TransferLearningPipeline:
    """
    Complete pipeline for transfer learning.

    Combines multiple transfer learning strategies.
    """

    def __init__(
        self,
        source_model: CrossAttentionRanker,
        config: RankingV2Config,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize pipeline.

        Args:
            source_model: Pre-trained model on source domain
            config: Model configuration
            device: Device
        """
        self.source_model = source_model
        self.config = config
        self.device = device
        self.domain_bank = DomainBank(source_model)

    def transfer(
        self,
        target_train: torch.utils.data.DataLoader,
        target_val: torch.utils.data.DataLoader,
        method: str = "fine_tune",
        **kwargs,
    ) -> TransferResult:
        """
        Transfer to target domain.

        Args:
            target_train: Target domain training data
            target_val: Target domain validation data
            method: Transfer method ("fine_tune", "maml", "domain_adapt")
            **kwargs: Method-specific arguments

        Returns:
            TransferResult
        """
        # Evaluate before transfer
        pre_acc = self._evaluate(self.source_model, target_val)

        # Apply transfer method
        if method == "fine_tune":
            adapted_model = self._fine_tune(target_train, **kwargs)
        elif method == "maml":
            adapted_model = self._maml_adapt(target_train, **kwargs)
        elif method == "domain_adapt":
            adapted_model = self._domain_adapt(target_train, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Evaluate after transfer
        post_acc = self._evaluate(adapted_model, target_val)

        return TransferResult(
            source_domain="source",
            target_domain="target",
            pre_transfer_accuracy=pre_acc,
            post_transfer_accuracy=post_acc,
            n_target_samples_used=len(target_train.dataset),
            adaptation_epochs=kwargs.get('epochs', 10),
            transfer_method=method,
        )

    def _evaluate(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> float:
        """Evaluate model accuracy."""
        model.to(self.device)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                outputs = model(
                    batch['grid_a'].to(self.device),
                    batch['scenario_a'].to(self.device),
                    batch['grid_b'].to(self.device),
                    batch['scenario_b'].to(self.device),
                )

                pred = (outputs['logit'] > 0).long()
                correct += (pred == batch['label'].to(self.device)).sum().item()
                total += batch['label'].size(0)

        return correct / total if total > 0 else 0.0

    def _fine_tune(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        lr: float = 1e-4,
        freeze_encoder: bool = True,
    ) -> nn.Module:
        """Fine-tune on target domain."""
        model = deepcopy(self.source_model)
        extractor = FeatureExtractor(model, freeze_encoder=freeze_encoder)
        extractor.fine_tune(train_loader, epochs=epochs, lr=lr, device=self.device)
        return model

    def _maml_adapt(
        self,
        train_loader: torch.utils.data.DataLoader,
        n_steps: int = 5,
        inner_lr: float = 0.01,
    ) -> nn.Module:
        """Adapt using MAML."""
        maml = MAML(self.source_model, inner_lr=inner_lr, n_inner_steps=n_steps)

        # Get support set from loader
        support_batch = next(iter(train_loader))
        support_data = {
            'grid_a': support_batch['grid_a'].to(self.device),
            'scenario_a': support_batch['scenario_a'].to(self.device),
            'grid_b': support_batch['grid_b'].to(self.device),
            'scenario_b': support_batch['scenario_b'].to(self.device),
            'labels': support_batch['label'].to(self.device),
        }

        return maml.adapt(support_data, clone=True)

    def _domain_adapt(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        lambda_domain: float = 0.1,
    ) -> nn.Module:
        """Adapt using domain adversarial training."""
        model = DomainAdaptationRanker(
            self.config,
            n_domains=2,
            lambda_domain=lambda_domain,
        )
        model.base_model.load_state_dict(self.source_model.state_dict())
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            model.train()

            for batch in train_loader:
                optimizer.zero_grad()

                outputs = model(
                    batch['grid_a'].to(self.device),
                    batch['scenario_a'].to(self.device),
                    batch['grid_b'].to(self.device),
                    batch['scenario_b'].to(self.device),
                    return_domain=True,
                )

                # Ranking loss
                rank_loss = F.binary_cross_entropy_with_logits(
                    outputs['logit'],
                    batch['label'].to(self.device).float(),
                )

                # Domain loss (try to confuse discriminator)
                domain_labels = torch.ones(
                    batch['label'].size(0),
                    dtype=torch.long,
                    device=self.device,
                )  # Target domain = 1
                domain_loss = model.compute_domain_loss(
                    outputs['domain_pred_a'],
                    domain_labels,
                )

                loss = rank_loss + lambda_domain * domain_loss
                loss.backward()
                optimizer.step()

        return model.base_model
