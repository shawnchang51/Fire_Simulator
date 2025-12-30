"""
Training Loop for Pairwise Ranking V2

Features:
- Integration with HardNegativeSampler for mining
- Multi-task loss computation
- Gradient accumulation support
- Warmup + Cosine annealing LR schedule
- Checkpoint saving and resume
"""

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RankingV2Config
from .model import CrossAttentionRanker, create_ranking_model
from .losses import MultiTaskRankingLoss, create_multi_task_loss
from .sampler import HardNegativeSampler, HardNegativeBatchSampler, create_sampler
from .dataset import PairwiseDatasetV2, create_train_loader_with_sampler


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01
) -> LambdaLR:
    """
    Create a schedule with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as fraction of initial

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: CrossAttentionRanker,
    train_loader: DataLoader,
    criterion: MultiTaskRankingLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    config: RankingV2Config,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Multi-task loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Dict with average losses and metrics
    """
    model.train()

    metrics = defaultdict(float)
    n_batches = 0
    accumulation_steps = config.gradient_accumulation_steps

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        grid_a = batch['grid_a'].to(device)
        scenario_a = batch['scenario_a'].to(device)
        grid_b = batch['grid_b'].to(device)
        scenario_b = batch['scenario_b'].to(device)
        labels = batch['label'].to(device)
        confidence = batch['confidence'].to(device)

        # Prepare auxiliary targets
        aux_targets = None
        if config.auxiliary_tasks:
            aux_targets = {'a': {}, 'b': {}}
            for task in config.auxiliary_tasks:
                key_a = f'{task}_a'
                key_b = f'{task}_b'
                if key_a in batch:
                    aux_targets['a'][task] = batch[key_a].to(device)
                    aux_targets['b'][task] = batch[key_b].to(device)

        # Forward pass
        outputs = model(grid_a, scenario_a, grid_b, scenario_b)

        # Compute multi-task loss
        losses = criterion(
            outputs,
            labels,
            confidence,
            aux_targets,
            epoch=epoch
        )

        # Scale loss for gradient accumulation
        loss = losses['total'] / accumulation_steps
        loss.backward()

        # Update weights
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping (optional, helps stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        # Track metrics
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                metrics[key] += value.item()
            else:
                metrics[key] += value

        # Compute accuracy for progress bar
        with torch.no_grad():
            predictions = (outputs['logit'] > 0).long()
            accuracy = (predictions == labels).float().mean().item()
            metrics['accuracy'] += accuracy

        n_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{metrics['total'] / n_batches:.4f}",
            'acc': f"{metrics['accuracy'] / n_batches:.4f}",
        })

    # Average metrics
    return {k: v / n_batches for k, v in metrics.items()}


@torch.no_grad()
def validate_epoch(
    model: CrossAttentionRanker,
    val_loader: DataLoader,
    criterion: MultiTaskRankingLoss,
    config: RankingV2Config,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Validate for one epoch.

    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Multi-task loss function
        config: Configuration
        device: Device to validate on
        epoch: Current epoch number

    Returns:
        Dict with average losses, accuracy, and AUC
    """
    model.eval()

    metrics = defaultdict(float)
    all_logits = []
    all_labels = []
    n_batches = 0

    for batch in tqdm(val_loader, desc="Validating"):
        # Move to device
        grid_a = batch['grid_a'].to(device)
        scenario_a = batch['scenario_a'].to(device)
        grid_b = batch['grid_b'].to(device)
        scenario_b = batch['scenario_b'].to(device)
        labels = batch['label'].to(device)
        confidence = batch['confidence'].to(device)

        # Prepare auxiliary targets
        aux_targets = None
        if config.auxiliary_tasks:
            aux_targets = {'a': {}, 'b': {}}
            for task in config.auxiliary_tasks:
                key_a = f'{task}_a'
                key_b = f'{task}_b'
                if key_a in batch:
                    aux_targets['a'][task] = batch[key_a].to(device)
                    aux_targets['b'][task] = batch[key_b].to(device)

        # Forward pass
        outputs = model(grid_a, scenario_a, grid_b, scenario_b)

        # Compute loss
        losses = criterion(
            outputs,
            labels,
            confidence,
            aux_targets,
            epoch=epoch
        )

        # Track metrics
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                metrics[key] += value.item()
            else:
                metrics[key] += value

        # Track for AUC
        all_logits.extend(outputs['logit'].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Compute accuracy
        predictions = (outputs['logit'] > 0).long()
        accuracy = (predictions == labels).float().mean().item()
        metrics['accuracy'] += accuracy

        n_batches += 1

    # Average metrics
    avg_metrics = {k: v / n_batches for k, v in metrics.items()}

    # Compute AUC
    try:
        auc = roc_auc_score(all_labels, all_logits)
    except ValueError:
        auc = 0.5

    avg_metrics['auc'] = auc

    return avg_metrics


def save_checkpoint(
    model: CrossAttentionRanker,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    epoch: int,
    val_auc: float,
    history: Dict[str, List[float]],
    config: RankingV2Config,
    checkpoint_path: str,
    scenario_stats: Dict,
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        val_auc: Current validation AUC
        history: Training history
        config: Configuration
        checkpoint_path: Path to save checkpoint
        scenario_stats: Normalization statistics
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_auc': val_auc,
        'history': history,
        'config': {
            field: getattr(config, field)
            for field in config.__dataclass_fields__
        },
        'scenario_stats': scenario_stats,
    }

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device = None
) -> Tuple[CrossAttentionRanker, Dict]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load to

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Recreate config
    config = RankingV2Config(**checkpoint['config'])

    # Create model
    model = create_ranking_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def train_ranking_model(
    config: RankingV2Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    scenario_stats: Optional[Dict] = None,
    train_dataset: Optional[PairwiseDatasetV2] = None,
) -> Tuple[CrossAttentionRanker, Dict[str, List[float]]]:
    """
    Full training loop for ranking model V2.

    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        scenario_stats: Normalization statistics
        train_dataset: Training dataset (for hard negative mining)

    Returns:
        Tuple of (trained_model, training_history)
    """
    print(f"\n{'='*60}")
    print("RANKING MODEL V2 - TRAINING")
    print(f"{'='*60}")

    # Create model
    model = create_ranking_model(config)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Create scheduler
    num_training_steps = config.epochs * len(train_loader)
    num_warmup_steps = config.warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    # Create loss function
    criterion = create_multi_task_loss(config)

    # Setup hard negative mining (if enabled)
    hard_sampler = None
    if config.mining_strategy != "none" and train_dataset is not None:
        hard_sampler = create_sampler(train_dataset, config, model)
        batch_sampler = HardNegativeBatchSampler(
            hard_sampler, config.batch_size, drop_last=True
        )
        # Replace train loader with mining-enabled loader
        train_loader = create_train_loader_with_sampler(
            train_dataset, batch_sampler, config
        )
        print(f"Hard negative mining: {config.mining_strategy}")
        print(f"  - Hard ratio: {config.hard_negative_ratio}")
        print(f"  - Threshold: {config.margin_threshold}")

    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_auc': [],
        'learning_rate': [],
    }

    # Add auxiliary task tracking
    for task in config.auxiliary_tasks:
        history[f'train_aux_{task}'] = []
        history[f'val_aux_{task}'] = []

    # Checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    best_val_auc = 0.0
    patience_counter = 0

    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Cross-attention: {config.use_cross_attention}")
    print(f"Auxiliary tasks: {config.auxiliary_tasks}")
    print()

    for epoch in range(config.epochs):
        # Update sampler epoch for curriculum learning
        if hard_sampler is not None:
            if hasattr(train_loader, 'batch_sampler'):
                train_loader.batch_sampler.set_epoch(epoch)

            # Refresh hard negatives periodically (offline mining)
            if config.mining_strategy == "offline":
                if epoch > 0 and epoch % config.mining_refresh_epochs == 0:
                    hard_sampler.refresh_predictions(device)

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            config, device, epoch
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, config, device, epoch
        )

        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0] if scheduler else config.learning_rate

        # Update history
        history['train_loss'].append(train_metrics['total'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['total'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        history['learning_rate'].append(current_lr)

        for task in config.auxiliary_tasks:
            key = f'aux_{task}'
            if key in train_metrics:
                history[f'train_{key}'].append(train_metrics[key])
            if key in val_metrics:
                history[f'val_{key}'].append(val_metrics[key])

        # Print progress
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print(f"  Train Loss: {train_metrics['total']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['total']:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"  LR: {current_lr:.6f}")

        if config.auxiliary_tasks:
            aux_str = ", ".join([
                f"{task}: {val_metrics.get(f'aux_{task}', 0):.4f}"
                for task in config.auxiliary_tasks
            ])
            print(f"  Aux losses: {aux_str}")

        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_metrics['auc'],
            history, config,
            str(checkpoint_dir / "latest_model.pt"),
            scenario_stats or {}
        )

        # Check for best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0

            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics['auc'],
                history, config,
                str(checkpoint_dir / "best_model.pt"),
                scenario_stats or {}
            )
            print(f"  -> New best AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{config.early_stopping_patience})")

        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    best_checkpoint_path = checkpoint_dir / "best_model.pt"
    if best_checkpoint_path.exists():
        model, _ = load_checkpoint(str(best_checkpoint_path), device)
        print(f"\nLoaded best model with AUC: {best_val_auc:.4f}")

    print(f"\nTraining complete!")
    print(f"Best validation AUC: {best_val_auc:.4f}")

    return model, history


def load_resume_checkpoint(
    checkpoint_dir: str,
    config: RankingV2Config,
    device: torch.device,
) -> Tuple[CrossAttentionRanker, torch.optim.Optimizer, Dict, int, float, Dict]:
    """
    Load checkpoint for resuming training.

    Args:
        checkpoint_dir: Directory containing checkpoints
        config: New configuration (may override some params)
        device: Device to load to

    Returns:
        Tuple of (model, optimizer, history, start_epoch, best_val_auc, scenario_stats)
    """
    checkpoint_path = Path(checkpoint_dir) / "latest_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model with original config
    orig_config = RankingV2Config(**checkpoint['config'])
    model = create_ranking_model(orig_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Create optimizer with potentially new learning rate
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Load optimizer state but update lr
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.learning_rate

    history = checkpoint.get('history', {})
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_auc = checkpoint.get('val_auc', 0.0)
    scenario_stats = checkpoint.get('scenario_stats', {})

    return model, optimizer, history, start_epoch, best_val_auc, scenario_stats
