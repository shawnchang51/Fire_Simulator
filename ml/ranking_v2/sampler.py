"""
Hard Negative Mining Sampler for Ranking V2

Implements multiple hard negative mining strategies:
- Online: Use model predictions to identify hard pairs each batch
- Offline: Pre-compute hard pairs using cached scores
- Curriculum: Gradually increase hard ratio over training

Key Design:
    - HardNegativeSampler: Core logic for identifying hard pairs
    - HardNegativeBatchSampler: PyTorch Sampler for DataLoader integration
    - Supports both ground-truth based and model-prediction based hardness
"""

from typing import Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from .config import RankingV2Config
from .dataset import PairwiseDatasetV2


class HardNegativeSampler:
    """
    Implements multiple hard negative mining strategies.

    Strategies:
    - none: Standard random sampling (baseline)
    - online: Use model predictions to identify hard pairs each batch
    - offline: Pre-compute hard pairs using cached scores
    - curriculum: Gradually increase hard ratio over training

    Attributes:
        dataset: Training dataset with precomputed hardness
        config: Configuration with mining parameters
        model: Optional model for online/offline mining
        hard_ratio: Fraction of batch that should be hard negatives
        threshold: Score difference threshold for "hard" classification
        cached_predictions: Model predictions cache (offline mining)
        hard_indices: Indices of hard pairs (offline mining)
    """

    def __init__(
        self,
        dataset: PairwiseDatasetV2,
        config: RankingV2Config,
        model: Optional[nn.Module] = None,
    ):
        """
        Initialize the sampler.

        Args:
            dataset: Training dataset with hardness info
            config: Configuration with mining parameters
            model: Optional model for online/offline mining
        """
        self.dataset = dataset
        self.config = config
        self.model = model

        self.hard_ratio = config.hard_negative_ratio
        self.threshold = config.margin_threshold
        self.strategy = config.mining_strategy

        # Offline mining cache
        self.cached_predictions: Optional[np.ndarray] = None
        self.hard_indices: Optional[np.ndarray] = None

        # Precompute ground-truth based hard indices
        self._init_gt_hard_indices()

    def _init_gt_hard_indices(self):
        """Initialize hard indices based on ground truth score differences."""
        self.gt_hard_indices = self.dataset.get_hard_indices(self.threshold)
        print(f"[Sampler] Found {len(self.gt_hard_indices)} hard pairs "
              f"(threshold={self.threshold}, {100*len(self.gt_hard_indices)/len(self.dataset):.1f}%)")

    def get_batch_indices(
        self,
        batch_size: int,
        epoch: int = 0
    ) -> np.ndarray:
        """
        Generate batch indices with specified hard negative ratio.

        The batch is composed of:
        - hard_ratio * batch_size hard negatives
        - (1 - hard_ratio) * batch_size random samples

        Args:
            batch_size: Total batch size
            epoch: Current epoch (for curriculum learning)

        Returns:
            Array of dataset indices for the batch
        """
        if self.strategy == "none":
            # Standard random sampling
            return np.random.choice(
                len(self.dataset),
                size=batch_size,
                replace=False
            )

        # Determine effective hard ratio
        if self.strategy == "curriculum":
            effective_ratio = self._curriculum_ratio(epoch)
        else:
            effective_ratio = self.hard_ratio

        n_hard = int(batch_size * effective_ratio)
        n_random = batch_size - n_hard

        # Select hard negatives source
        if self.strategy == "offline" and self.hard_indices is not None:
            # Use model-prediction based hard indices
            hard_pool = self.hard_indices
        else:
            # Use ground-truth based hard indices
            hard_pool = self.gt_hard_indices

        # Sample hard negatives
        if len(hard_pool) > 0:
            hard_samples = np.random.choice(
                hard_pool,
                size=min(n_hard, len(hard_pool)),
                replace=len(hard_pool) < n_hard  # Allow replacement if not enough
            )
        else:
            hard_samples = np.array([], dtype=np.int64)
            n_random = batch_size  # Fall back to all random

        # Sample random pairs (excluding hard samples to avoid duplicates)
        all_indices = np.arange(len(self.dataset))
        if len(hard_samples) > 0:
            available = np.setdiff1d(all_indices, hard_samples)
        else:
            available = all_indices

        random_samples = np.random.choice(
            available,
            size=min(n_random, len(available)),
            replace=False
        )

        # Combine and shuffle
        batch_indices = np.concatenate([hard_samples, random_samples])
        np.random.shuffle(batch_indices)

        return batch_indices

    def _curriculum_ratio(self, epoch: int) -> float:
        """
        Gradually increase hard ratio during warmup.

        Starts from 0 and linearly increases to hard_ratio over warmup epochs.

        Args:
            epoch: Current epoch

        Returns:
            Effective hard negative ratio for this epoch
        """
        warmup = self.config.curriculum_warmup_epochs
        if epoch < warmup:
            return self.hard_ratio * (epoch / warmup)
        return self.hard_ratio

    @torch.no_grad()
    def refresh_predictions(self, device: torch.device):
        """
        Update cached predictions for offline mining.

        Called periodically during training to update the hard pairs
        based on current model predictions.

        Args:
            device: Device to run inference on
        """
        if self.model is None:
            print("[Sampler] No model provided, skipping prediction refresh")
            return

        if self.strategy not in ("offline", "online"):
            return

        self.model.eval()
        predictions = []

        # Score all pairs with current model
        loader = DataLoader(
            self.dataset,
            batch_size=256,
            shuffle=False,
            num_workers=4
        )

        print("[Sampler] Refreshing hard negative cache...")
        for batch in loader:
            grid_a = batch['grid_a'].to(device)
            scenario_a = batch['scenario_a'].to(device)
            grid_b = batch['grid_b'].to(device)
            scenario_b = batch['scenario_b'].to(device)

            # Get model predictions
            outputs = self.model(grid_a, scenario_a, grid_b, scenario_b)
            logit = outputs['logit'] if isinstance(outputs, dict) else outputs[2]
            predictions.extend(logit.cpu().numpy())

        self.cached_predictions = np.array(predictions)

        # Identify hard pairs: model is uncertain (logit near 0)
        # Lower |logit| means model is less confident
        hardness = np.abs(self.cached_predictions)
        self.hard_indices = np.where(hardness < self.threshold)[0]

        # Also include pairs where model disagrees with label
        labels = np.array([self.dataset.pairs[i]['label'] for i in range(len(self.dataset))])
        model_preds = (self.cached_predictions > 0).astype(int)
        disagree_indices = np.where(model_preds != labels)[0]

        # Combine uncertain and disagreeing pairs
        self.hard_indices = np.unique(np.concatenate([self.hard_indices, disagree_indices]))

        print(f"[Sampler] Found {len(self.hard_indices)} hard pairs "
              f"({100*len(self.hard_indices)/len(self.dataset):.1f}% of dataset)")

    def get_stats(self) -> dict:
        """Get sampler statistics for logging."""
        return {
            'strategy': self.strategy,
            'hard_ratio': self.hard_ratio,
            'threshold': self.threshold,
            'n_gt_hard': len(self.gt_hard_indices),
            'n_model_hard': len(self.hard_indices) if self.hard_indices is not None else 0,
            'dataset_size': len(self.dataset),
        }


class HardNegativeBatchSampler(Sampler):
    """
    PyTorch Batch Sampler that integrates with HardNegativeSampler.

    Usage:
        sampler = HardNegativeSampler(dataset, config)
        batch_sampler = HardNegativeBatchSampler(sampler, batch_size=128)
        loader = DataLoader(dataset, batch_sampler=batch_sampler)

    Attributes:
        sampler: HardNegativeSampler instance
        batch_size: Number of samples per batch
        drop_last: Whether to drop the last incomplete batch
        epoch: Current epoch (set via set_epoch())
    """

    def __init__(
        self,
        sampler: HardNegativeSampler,
        batch_size: int,
        drop_last: bool = True,
    ):
        """
        Initialize the batch sampler.

        Args:
            sampler: HardNegativeSampler instance
            batch_size: Number of samples per batch
            drop_last: Whether to drop the last incomplete batch
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield batches of indices.

        Each batch is generated by the HardNegativeSampler with the
        appropriate hard/random ratio.
        """
        dataset_size = len(self.sampler.dataset)
        n_batches = dataset_size // self.batch_size

        if not self.drop_last and dataset_size % self.batch_size != 0:
            n_batches += 1

        # Track which indices have been used
        used_indices = set()
        remaining = set(range(dataset_size))

        for batch_idx in range(n_batches):
            # Determine batch size for this iteration
            current_batch_size = self.batch_size
            if batch_idx == n_batches - 1 and not self.drop_last:
                current_batch_size = dataset_size - batch_idx * self.batch_size

            # Get batch indices from sampler
            batch_indices = self.sampler.get_batch_indices(
                current_batch_size,
                self.epoch
            )

            yield batch_indices.tolist()

    def __len__(self) -> int:
        """Return number of batches."""
        dataset_size = len(self.sampler.dataset)
        if self.drop_last:
            return dataset_size // self.batch_size
        else:
            return (dataset_size + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int):
        """
        Set the current epoch for curriculum learning.

        Should be called at the start of each epoch.

        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch


class AdaptiveHardNegativeSampler(HardNegativeSampler):
    """
    Adaptive hard negative sampler that adjusts threshold based on model performance.

    If the model becomes too accurate on hard pairs, the threshold is decreased
    to find even harder pairs. Conversely, if accuracy is too low, threshold
    is increased.

    Attributes:
        target_accuracy: Target accuracy on hard pairs
        threshold_lr: Learning rate for threshold adjustment
        min_threshold: Minimum allowed threshold
        max_threshold: Maximum allowed threshold
    """

    def __init__(
        self,
        dataset: PairwiseDatasetV2,
        config: RankingV2Config,
        model: Optional[nn.Module] = None,
        target_accuracy: float = 0.6,
        threshold_lr: float = 0.1,
        min_threshold: float = 0.05,
        max_threshold: float = 0.5,
    ):
        """
        Initialize adaptive sampler.

        Args:
            dataset: Training dataset
            config: Configuration
            model: Model for prediction-based mining
            target_accuracy: Desired accuracy on hard pairs (0.5-0.7)
            threshold_lr: How quickly to adjust threshold
            min_threshold: Minimum threshold value
            max_threshold: Maximum threshold value
        """
        super().__init__(dataset, config, model)

        self.target_accuracy = target_accuracy
        self.threshold_lr = threshold_lr
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.hard_accuracy_history: List[float] = []

    def update_threshold(self, hard_accuracy: float):
        """
        Adjust threshold based on observed accuracy on hard pairs.

        Args:
            hard_accuracy: Accuracy on hard pairs in recent epoch
        """
        self.hard_accuracy_history.append(hard_accuracy)

        # Adjust threshold based on accuracy
        if hard_accuracy > self.target_accuracy + 0.1:
            # Model is too good on "hard" pairs, make threshold stricter
            self.threshold *= (1 - self.threshold_lr)
            self.threshold = max(self.threshold, self.min_threshold)
            print(f"[AdaptiveSampler] Decreased threshold to {self.threshold:.3f}")
        elif hard_accuracy < self.target_accuracy - 0.1:
            # Model struggles too much, relax threshold
            self.threshold *= (1 + self.threshold_lr)
            self.threshold = min(self.threshold, self.max_threshold)
            print(f"[AdaptiveSampler] Increased threshold to {self.threshold:.3f}")

        # Recompute hard indices with new threshold
        self._init_gt_hard_indices()

    def get_stats(self) -> dict:
        """Get extended stats including threshold history."""
        stats = super().get_stats()
        stats['hard_accuracy_history'] = self.hard_accuracy_history
        return stats


def create_sampler(
    dataset: PairwiseDatasetV2,
    config: RankingV2Config,
    model: Optional[nn.Module] = None,
    adaptive: bool = False
) -> HardNegativeSampler:
    """
    Factory function to create appropriate sampler based on config.

    Args:
        dataset: Training dataset
        config: Configuration
        model: Optional model for prediction-based mining
        adaptive: Whether to use adaptive threshold adjustment

    Returns:
        HardNegativeSampler or AdaptiveHardNegativeSampler
    """
    if config.mining_strategy == "none":
        # Return base sampler that just does random sampling
        return HardNegativeSampler(dataset, config, model)

    if adaptive:
        return AdaptiveHardNegativeSampler(dataset, config, model)
    else:
        return HardNegativeSampler(dataset, config, model)
