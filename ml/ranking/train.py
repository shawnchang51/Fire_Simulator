"""
Training Loop for Pairwise Ranking Model

Key Features:
    - Clean branching by loss type (RankNet vs Hinge)
    - Confidence weighting for RankNet only
    - L1 regularization on latent vectors
    - Warmup + cosine annealing LR schedule
    - Early stopping on validation AUC
    - Checkpointing best model
    - TensorBoard logging (optional)
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RankingConfig
from .model import SiameseRanker
from .losses import RankNetLoss, MarginHingeLoss


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01
):
    """
    Create a schedule with warmup followed by cosine annealing.

    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial LR

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: SiameseRanker,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    config: RankingConfig,
    device: torch.device
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: The ranking model
        train_loader: Training data loader
        criterion: Loss function (RankNet or Hinge)
        optimizer: Optimizer
        scheduler: LR scheduler (step per batch)
        config: Training configuration
        device: Device to train on

    Returns:
        Dict with training metrics
    """
    model.train()

    total_loss = 0.0
    total_pair_loss = 0.0
    total_l1_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        # Move to device
        grid_a = batch['grid_a'].to(device)
        scenario_a = batch['scenario_a'].to(device)
        grid_b = batch['grid_b'].to(device)
        scenario_b = batch['scenario_b'].to(device)
        label = batch['label'].to(device)
        confidence = batch['confidence'].to(device)

        # Forward pass
        score_a, score_b, logit = model(grid_a, scenario_a, grid_b, scenario_b)

        # Compute pairwise loss (branch by loss type)
        if config.loss_type == 'ranknet':
            # RankNet uses logit, applies sigmoid internally
            # Confidence weighting ONLY for RankNet
            pair_loss = criterion(logit, label, confidence)
        else:
            # Hinge uses raw scores, NO confidence weighting
            pair_loss = criterion(score_a, score_b, label)

        # L1 regularization on latent vectors
        latent_a = model.get_latent(grid_a)
        l1_loss = config.l1_lambda * latent_a.abs().mean()

        # Total loss
        loss = pair_loss + l1_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Track metrics
        total_loss += loss.item()
        total_pair_loss += pair_loss.item()
        total_l1_loss += l1_loss.item()
        num_batches += 1

        # Accuracy (logit > 0 means A > B)
        pred = (logit > 0).long()
        correct += (pred == label).sum().item()
        total += label.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })

    return {
        'train_loss': total_loss / num_batches,
        'train_pair_loss': total_pair_loss / num_batches,
        'train_l1_loss': total_l1_loss / num_batches,
        'train_accuracy': correct / total,
        'learning_rate': optimizer.param_groups[0]['lr']
    }


@torch.no_grad()
def validate_epoch(
    model: SiameseRanker,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: RankingConfig,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate for one epoch.

    Args:
        model: The ranking model
        val_loader: Validation data loader
        criterion: Loss function
        config: Training configuration
        device: Device

    Returns:
        Dict with validation metrics
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0

    all_logits = []
    all_labels = []

    for batch in val_loader:
        grid_a = batch['grid_a'].to(device)
        scenario_a = batch['scenario_a'].to(device)
        grid_b = batch['grid_b'].to(device)
        scenario_b = batch['scenario_b'].to(device)
        label = batch['label'].to(device)
        confidence = batch['confidence'].to(device)

        # Forward pass
        score_a, score_b, logit = model(grid_a, scenario_a, grid_b, scenario_b)

        # Compute loss
        if config.loss_type == 'ranknet':
            loss = criterion(logit, label, confidence)
        else:
            loss = criterion(score_a, score_b, label)

        total_loss += loss.item()
        num_batches += 1

        # Accuracy
        pred = (logit > 0).long()
        correct += (pred == label).sum().item()
        total += label.size(0)

        # Collect for AUC
        all_logits.extend(logit.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    # Compute AUC
    try:
        from sklearn.metrics import roc_auc_score
        import numpy as np
        auc = roc_auc_score(np.array(all_labels), np.array(all_logits))
    except Exception:
        auc = 0.5

    return {
        'val_loss': total_loss / num_batches,
        'val_accuracy': correct / total,
        'val_auc': auc
    }


def train_ranking_model(
    config: RankingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device = None
) -> Tuple[SiameseRanker, Dict]:
    """
    Full training loop for pairwise ranking model.

    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on (default: auto-detect)

    Returns:
        Tuple of (trained model, training history dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on device: {device}")

    # Create model
    model = SiameseRanker(config)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Create scheduler with warmup
    num_training_steps = config.epochs * len(train_loader)
    num_warmup_steps = config.warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Create loss function
    if config.loss_type == 'ranknet':
        criterion = RankNetLoss(sigma=config.sigma)
    else:
        criterion = MarginHingeLoss(margin=config.margin)

    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_auc': [],
        'learning_rate': []
    }

    # Early stopping
    best_val_auc = 0.0
    patience_counter = 0

    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Loss type: {config.loss_type}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"L1 lambda: {config.l1_lambda}")
    print()

    for epoch in range(config.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, config, device
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, config, device
        )

        # Update history
        history['train_loss'].append(train_metrics['train_loss'])
        history['train_accuracy'].append(train_metrics['train_accuracy'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_accuracy'].append(val_metrics['val_accuracy'])
        history['val_auc'].append(val_metrics['val_auc'])
        history['learning_rate'].append(train_metrics['learning_rate'])

        # Print progress
        print(
            f"Epoch {epoch + 1:3d}/{config.epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Train Acc: {train_metrics['train_accuracy']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Acc: {val_metrics['val_accuracy']:.4f} | "
            f"Val AUC: {val_metrics['val_auc']:.4f} | "
            f"LR: {train_metrics['learning_rate']:.2e}"
        )

        # Check for improvement
        if val_metrics['val_auc'] > best_val_auc:
            best_val_auc = val_metrics['val_auc']
            patience_counter = 0

            # Save best model
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'config': config.__dict__,
                'history': history
            }, checkpoint_path)
            print(f"  -> Saved best model (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Load best model
    checkpoint_path = checkpoint_dir / "best_model.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")

    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


def load_checkpoint(
    checkpoint_path: str,
    config: RankingConfig = None,
    device: torch.device = None
) -> Tuple[SiameseRanker, Dict]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Optional config (uses saved config if None)
        device: Device to load to

    Returns:
        Tuple of (model, checkpoint dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Use saved config if not provided
    if config is None:
        config = RankingConfig(**checkpoint['config'])

    model = SiameseRanker(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    return model, checkpoint
