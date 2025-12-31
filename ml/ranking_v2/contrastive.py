"""
Contrastive Learning Framework for Floor Plan Representations

Implements self-supervised pre-training methods to learn robust floor plan
representations before fine-tuning on ranking tasks.

Methods:
1. SimCLR: Simple Contrastive Learning of Visual Representations
2. MoCo: Momentum Contrast for self-supervised learning
3. RankingContrastive: Custom contrastive learning for ranking pairs

Key Ideas:
- Same floor plan with different door configurations = positive pair
- Different floor plans = negative pairs
- Learn representations that capture floor plan structure
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from .config import RankingV2Config
from .model import FloorPlanEncoder


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    Used in SimCLR for contrastive learning.
    """

    def __init__(self, temperature: float = 0.5):
        """
        Initialize NT-Xent loss.

        Args:
            temperature: Temperature scaling factor
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss.

        Args:
            z_i: First view embeddings (B, D)
            z_j: Second view embeddings (B, D)

        Returns:
            Loss value
        """
        batch_size = z_i.size(0)
        device = z_i.device

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2,
        )  # (2B, 2B)

        # Remove self-similarities
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # Positive pairs: (i, B+i) and (B+i, i)
        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size),
        ])  # (2B,)

        # All pairs as negatives (excluding self)
        negatives = similarity_matrix[~mask].view(2 * batch_size, -1)  # (2B, 2B-1)

        # Logits: [positive, negatives]
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        logits = logits / self.temperature

        # Labels: positive is always first
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)

        return F.cross_entropy(logits, labels)


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for contrastive learning.

    Used in MoCo and other methods.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            query: Query embeddings (B, D)
            positive_key: Positive key embeddings (B, D)
            negative_keys: Negative key embeddings (K, D)

        Returns:
            Loss value
        """
        # Normalize
        query = F.normalize(query, dim=1)
        positive_key = F.normalize(positive_key, dim=1)
        negative_keys = F.normalize(negative_keys, dim=1)

        # Positive logits
        l_pos = torch.einsum('bd,bd->b', query, positive_key).unsqueeze(1)  # (B, 1)

        # Negative logits
        l_neg = torch.einsum('bd,kd->bk', query, negative_keys)  # (B, K)

        # Logits
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature  # (B, 1+K)

        # Labels (positive is first)
        labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)

        return F.cross_entropy(logits, labels)


class FloorPlanAugmenter:
    """
    Augmentation module for floor plan grids.

    Creates different views of the same floor plan for contrastive learning.
    """

    def __init__(
        self,
        random_shift: bool = True,
        shift_range: int = 5,
        random_noise: bool = True,
        noise_std: float = 0.05,
        random_mask: bool = True,
        mask_ratio: float = 0.1,
        random_flip: bool = False,  # May break spatial semantics
    ):
        """
        Initialize augmenter.

        Args:
            random_shift: Apply random spatial shift
            shift_range: Maximum shift in pixels
            random_noise: Add Gaussian noise
            noise_std: Standard deviation of noise
            random_mask: Randomly mask patches
            mask_ratio: Fraction of grid to mask
            random_flip: Apply random horizontal flip
        """
        self.random_shift = random_shift
        self.shift_range = shift_range
        self.random_noise = random_noise
        self.noise_std = noise_std
        self.random_mask = random_mask
        self.mask_ratio = mask_ratio
        self.random_flip = random_flip

    def __call__(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations.

        Args:
            grid: Floor plan grid (C, H, W) or (B, C, H, W)

        Returns:
            Augmented grid with same shape
        """
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        augmented = grid.clone()

        # Random shift
        if self.random_shift:
            augmented = self._apply_shift(augmented)

        # Random noise (only on continuous channels)
        if self.random_noise:
            augmented = self._apply_noise(augmented)

        # Random masking
        if self.random_mask:
            augmented = self._apply_mask(augmented)

        # Random flip
        if self.random_flip and torch.rand(1).item() > 0.5:
            augmented = torch.flip(augmented, dims=[-1])

        if squeeze:
            augmented = augmented.squeeze(0)

        return augmented

    def _apply_shift(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply random spatial shift."""
        B, C, H, W = grid.shape
        shift_y = torch.randint(-self.shift_range, self.shift_range + 1, (1,)).item()
        shift_x = torch.randint(-self.shift_range, self.shift_range + 1, (1,)).item()

        if shift_y != 0 or shift_x != 0:
            grid = torch.roll(grid, shifts=(shift_y, shift_x), dims=(2, 3))

        return grid

    def _apply_noise(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise."""
        noise = torch.randn_like(grid) * self.noise_std
        # Only add noise to valid regions
        valid_mask = (grid[:, 4:5] > 0).float()
        grid = grid + noise * valid_mask
        return grid

    def _apply_mask(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply random patch masking."""
        B, C, H, W = grid.shape

        # Calculate number of patches to mask
        patch_size = 8
        n_patches_h = H // patch_size
        n_patches_w = W // patch_size
        n_patches = n_patches_h * n_patches_w
        n_mask = int(n_patches * self.mask_ratio)

        for b in range(B):
            # Select random patches to mask
            mask_indices = torch.randperm(n_patches)[:n_mask]

            for idx in mask_indices:
                ph = idx // n_patches_w
                pw = idx % n_patches_w
                y_start = ph * patch_size
                x_start = pw * patch_size
                # Mask with zeros (or could use mean)
                grid[b, :, y_start:y_start + patch_size, x_start:x_start + patch_size] = 0

        return grid


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.

    Maps encoder output to a lower-dimensional space where
    contrastive loss is computed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """
    SimCLR model for self-supervised floor plan representation learning.

    Architecture:
        Encoder (FloorPlanEncoder) -> Projection Head -> Contrastive Loss
    """

    def __init__(
        self,
        config: RankingV2Config,
        projection_dim: int = 128,
        temperature: float = 0.5,
    ):
        """
        Initialize SimCLR model.

        Args:
            config: Model configuration
            projection_dim: Output dimension of projection head
            temperature: Temperature for NT-Xent loss
        """
        super().__init__()

        self.encoder = FloorPlanEncoder(config)
        self.projection_head = ProjectionHead(
            input_dim=config.latent_dim,
            output_dim=projection_dim,
        )
        self.augmenter = FloorPlanAugmenter()
        self.loss_fn = NTXentLoss(temperature=temperature)

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Floor plan grid (B, 5, H, W)
            return_embeddings: If True, return (loss, z_i, z_j)

        Returns:
            Contrastive loss (and optionally embeddings)
        """
        # Create two augmented views
        x_i = self.augmenter(x)
        x_j = self.augmenter(x)

        # Encode
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        # Project
        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)

        # Compute loss
        loss = self.loss_fn(z_i, z_j)

        if return_embeddings:
            return loss, z_i, z_j
        return loss

    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get encoder representations (without projection head).

        Use this for downstream tasks after pre-training.
        """
        return self.encoder(x)


class MoCoModel(nn.Module):
    """
    Momentum Contrast (MoCo) for self-supervised learning.

    Uses a momentum-updated encoder and a queue of negative samples
    for more efficient contrastive learning.
    """

    def __init__(
        self,
        config: RankingV2Config,
        projection_dim: int = 128,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
    ):
        """
        Initialize MoCo model.

        Args:
            config: Model configuration
            projection_dim: Output dimension of projection head
            queue_size: Size of negative sample queue
            momentum: Momentum for key encoder update
            temperature: Temperature for InfoNCE loss
        """
        super().__init__()

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # Query encoder
        self.encoder_q = FloorPlanEncoder(config)
        self.projection_q = ProjectionHead(
            input_dim=config.latent_dim,
            output_dim=projection_dim,
        )

        # Key encoder (momentum-updated)
        self.encoder_k = FloorPlanEncoder(config)
        self.projection_k = ProjectionHead(
            input_dim=config.latent_dim,
            output_dim=projection_dim,
        )

        # Initialize key encoder with query encoder weights
        self._copy_weights(self.encoder_q, self.encoder_k)
        self._copy_weights(self.projection_q, self.projection_k)

        # Disable gradient for key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projection_k.parameters():
            param.requires_grad = False

        # Augmenter
        self.augmenter = FloorPlanAugmenter()

        # Queue for negative samples
        self.register_buffer('queue', torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # Loss function
        self.loss_fn = InfoNCELoss(temperature=temperature)

    def _copy_weights(self, source: nn.Module, target: nn.Module):
        """Copy weights from source to target."""
        for param_q, param_k in zip(source.parameters(), target.parameters()):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Update key encoder with momentum."""
        for param_q, param_k in zip(
            list(self.encoder_q.parameters()) + list(self.projection_q.parameters()),
            list(self.encoder_k.parameters()) + list(self.projection_k.parameters()),
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update the queue with new keys."""
        batch_size = keys.size(0)

        ptr = int(self.queue_ptr)
        remaining = self.queue_size - ptr

        if batch_size <= remaining:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T

        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Floor plan grid (B, 5, H, W)

        Returns:
            Contrastive loss
        """
        # Create two augmented views
        x_q = self.augmenter(x)
        x_k = self.augmenter(x)

        # Query encoding
        q = self.projection_q(self.encoder_q(x_q))
        q = F.normalize(q, dim=1)

        # Key encoding (no gradients)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.projection_k(self.encoder_k(x_k))
            k = F.normalize(k, dim=1)

        # Compute loss
        loss = self.loss_fn(q, k, self.queue.clone().detach().T)

        # Update queue
        self._dequeue_and_enqueue(k)

        return loss

    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder representations for downstream tasks."""
        return self.encoder_q(x)


class RankingContrastiveLoss(nn.Module):
    """
    Contrastive loss designed specifically for ranking tasks.

    Uses ranking relationships to define positive/negative pairs:
    - Positive: Same floor plan, different configurations
    - Hard negative: Different floor plans with similar performance
    """

    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        anchor_scores: Optional[torch.Tensor] = None,
        negative_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute ranking-aware contrastive loss.

        Args:
            anchor: Anchor embeddings (B, D)
            positive: Positive embeddings (B, D) - same floor plan
            negatives: Negative embeddings (B, N, D) - different floor plans
            anchor_scores: Optional scores for anchors (B,)
            negative_scores: Optional scores for negatives (B, N)

        Returns:
            Loss value
        """
        B, D = anchor.shape
        N = negatives.size(1) if negatives.dim() == 3 else 1

        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives.view(-1, D), dim=-1).view(B, N, D)

        # Positive similarity
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature  # (B,)

        # Negative similarities
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature  # (B, N)

        # Score-weighted negatives (harder negatives get higher weight)
        if anchor_scores is not None and negative_scores is not None:
            score_diff = torch.abs(anchor_scores.unsqueeze(1) - negative_scores)
            hard_weights = 1.0 / (score_diff + 0.1)
            hard_weights = F.softmax(hard_weights, dim=-1)
            neg_sim = neg_sim * hard_weights

        # InfoNCE-style loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+N)
        labels = torch.zeros(B, dtype=torch.long, device=anchor.device)

        return F.cross_entropy(logits, labels)


class ContrastivePretrainer:
    """
    Pre-training manager for contrastive learning.

    Handles training loop, checkpointing, and transfer to downstream tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize pre-trainer.

        Args:
            model: Contrastive model (SimCLR, MoCo, etc.)
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.history = {
            'loss': [],
            'learning_rate': [],
        }

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader providing floor plan grids

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            # Extract grids (handle different batch formats)
            if isinstance(batch, dict):
                grids = batch.get('grid_a', batch.get('grid', None))
            elif isinstance(batch, (list, tuple)):
                grids = batch[0]
            else:
                grids = batch

            grids = grids.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            loss = self.model(grids)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        self.history['loss'].append(avg_loss)
        self.history['learning_rate'].append(
            self.optimizer.param_groups[0]['lr']
        )

        return avg_loss

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full pre-training loop.

        Args:
            dataloader: DataLoader for pre-training
            epochs: Number of epochs
            verbose: Print progress

        Returns:
            Training history
        """
        for epoch in range(epochs):
            loss = self.train_epoch(dataloader)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

        return self.history

    def get_pretrained_encoder(self) -> nn.Module:
        """
        Extract pre-trained encoder for downstream tasks.

        Returns:
            Pre-trained encoder (FloorPlanEncoder)
        """
        if hasattr(self.model, 'encoder'):
            return self.model.encoder
        elif hasattr(self.model, 'encoder_q'):
            return self.model.encoder_q
        else:
            raise AttributeError("Model does not have a recognizable encoder")

    def save_checkpoint(self, path: str):
        """Save pre-training checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
        }, path)

    def load_checkpoint(self, path: str):
        """Load pre-training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']


def create_contrastive_model(
    config: RankingV2Config,
    method: str = 'simclr',
    **kwargs,
) -> nn.Module:
    """
    Factory function to create contrastive learning model.

    Args:
        config: Model configuration
        method: Contrastive method ('simclr', 'moco')
        **kwargs: Method-specific arguments

    Returns:
        Contrastive model
    """
    if method == 'simclr':
        return SimCLRModel(
            config,
            projection_dim=kwargs.get('projection_dim', 128),
            temperature=kwargs.get('temperature', 0.5),
        )
    elif method == 'moco':
        return MoCoModel(
            config,
            projection_dim=kwargs.get('projection_dim', 128),
            queue_size=kwargs.get('queue_size', 65536),
            momentum=kwargs.get('momentum', 0.999),
            temperature=kwargs.get('temperature', 0.07),
        )
    else:
        raise ValueError(f"Unknown contrastive method: {method}")


def transfer_pretrained_encoder(
    pretrained_encoder: nn.Module,
    target_model: nn.Module,
    freeze_encoder: bool = False,
) -> nn.Module:
    """
    Transfer pre-trained encoder weights to target model.

    Args:
        pretrained_encoder: Pre-trained FloorPlanEncoder
        target_model: Target model (CrossAttentionRanker)
        freeze_encoder: Whether to freeze encoder weights

    Returns:
        Target model with transferred weights
    """
    # Copy encoder weights
    target_model.encoder.load_state_dict(pretrained_encoder.state_dict())

    if freeze_encoder:
        for param in target_model.encoder.parameters():
            param.requires_grad = False

    return target_model
