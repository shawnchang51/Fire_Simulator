"""
Training Loop for Pairwise Ranking Model

Key Features:
    - Clean branching by loss type (RankNet vs Hinge)
    - Confidence weighting for RankNet only
    - L1 regularization on latent vectors
    - Warmup + cosine annealing LR schedule
    - Early stopping on validation AUC
    - Checkpointing best model (best_model.pt) and latest model (latest_model.pt)
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
    Train for one epoch with optional gradient accumulation.

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

    # Gradient accumulation settings
    accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)

    pbar = tqdm(train_loader, desc="Training", leave=False)
    optimizer.zero_grad()  # Zero gradients at start

    for batch_idx, batch in enumerate(pbar):
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

        # Total loss (scaled for gradient accumulation)
        loss = (pair_loss + l1_loss) / accumulation_steps

        # Backward pass (accumulate gradients)
        loss.backward()

        # Step optimizer every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Track metrics (unscaled loss for reporting)
        unscaled_loss = pair_loss.item() + l1_loss.item()
        total_loss += unscaled_loss
        total_pair_loss += pair_loss.item()
        total_l1_loss += l1_loss.item()
        num_batches += 1

        # Accuracy (logit > 0 means A > B)
        pred = (logit > 0).long()
        correct += (pred == label).sum().item()
        total += label.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{unscaled_loss:.4f}',
            'acc': f'{correct/total:.4f}'
        })

    # Handle remaining gradients if batches not divisible by accumulation_steps
    if len(train_loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

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
    device: torch.device = None,
    resume_checkpoint: Optional[Dict] = None
) -> Tuple[SiameseRanker, Dict]:
    """
    Full training loop for pairwise ranking model.

    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on (default: auto-detect)
        resume_checkpoint: Optional dict with resume state (model, optimizer, history, etc.)

    Returns:
        Tuple of (trained model, training history dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on device: {device}")

    # Check if resuming
    if resume_checkpoint is not None:
        # Resuming training
        model = resume_checkpoint['model']
        optimizer = resume_checkpoint['optimizer']
        history = resume_checkpoint['history']
        start_epoch = resume_checkpoint['start_epoch']
        best_val_auc = resume_checkpoint['best_val_auc']
        patience_counter = 0  # Reset patience

        print(f"\nResuming training from epoch {start_epoch}")
        print(f"Previous best val AUC: {best_val_auc:.4f}")
        print(f"Early stopping patience reset")
    else:
        # Fresh training
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
        start_epoch = 0
        best_val_auc = 0.0
        patience_counter = 0

    # Validate epochs for resume
    if start_epoch >= config.epochs:
        raise ValueError(
            f"Cannot resume: requested epochs ({config.epochs}) <= "
            f"already trained epochs ({start_epoch})\n"
            f"Set --epochs to a value > {start_epoch}"
        )

    # Create scheduler with warmup (fresh for both cases)
    remaining_epochs = config.epochs - start_epoch
    num_training_steps = remaining_epochs * len(train_loader)
    num_warmup_steps = config.warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Create loss function
    if config.loss_type == 'ranknet':
        label_smoothing = getattr(config, 'label_smoothing', 0.0)
        criterion = RankNetLoss(sigma=config.sigma, label_smoothing=label_smoothing)
    else:
        criterion = MarginHingeLoss(margin=config.margin)

    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if resume_checkpoint is None:
        print(f"\nStarting training for {config.epochs} epochs...")
    else:
        print(f"\nContinuing training from epoch {start_epoch} to {config.epochs}...")
    print(f"Loss type: {config.loss_type}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"L1 lambda: {config.l1_lambda}")
    print()

    for epoch in range(start_epoch, config.epochs):
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

        # Save latest model (every epoch)
        latest_checkpoint_path = checkpoint_dir / "latest_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_metrics['val_auc'],
            'config': config.__dict__,
            'history': history
        }, latest_checkpoint_path)

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


def validate_resume_config(
    resume_config: Dict,
    new_config: RankingConfig,
    strict: bool = True
) -> Tuple[bool, list]:
    """
    Validate that new config is compatible with checkpoint config.

    Args:
        resume_config: Config dict from checkpoint
        new_config: New RankingConfig with potential overrides
        strict: If True, raise error on incompatibility

    Returns:
        Tuple of (is_compatible, list_of_warnings)

    Raises:
        ValueError: If incompatible and strict=True
    """
    # Define immutable architectural parameters
    IMMUTABLE_PARAMS = {
        'target_grid_size', 'num_grid_channels', 'cnn_channels',
        'latent_dim', 'scenario_input_dim', 'scenario_hidden_dim',
        'scenario_output_dim', 'scoring_hidden_dim', 'dropout',
        'use_residual', 'scoring_num_layers', 'use_layer_norm'  # New architecture params
    }

    # Define mutable parameters that can be changed
    MUTABLE_PARAMS = {
        'loss_type', 'margin', 'sigma', 'l1_lambda', 'weight_decay',
        'batch_size', 'gradient_accumulation_steps', 'learning_rate',
        'warmup_epochs', 'epochs', 'early_stopping_patience',
        'num_workers', 'augment_shift', 'seed', 'label_smoothing'
    }

    # Define parameters that require warnings
    WARN_IF_DIFFERENT = {
        'data_dir', 'floor_plans_dir'
    }

    errors = []
    warnings = []

    # Check immutable parameters
    for param in IMMUTABLE_PARAMS:
        resume_val = resume_config.get(param)
        new_val = getattr(new_config, param, None)

        if resume_val != new_val:
            errors.append(
                f"Architecture parameter '{param}' cannot be changed.\n"
                f"  Checkpoint: {resume_val}\n"
                f"  New config: {new_val}"
            )

    # Check parameters that need warnings
    for param in WARN_IF_DIFFERENT:
        resume_val = resume_config.get(param)
        new_val = getattr(new_config, param, None)

        if resume_val != new_val:
            warnings.append(
                f"Data path '{param}' changed (will use checkpoint's scenario_stats.json):\n"
                f"  Original: {resume_val}\n"
                f"  New: {new_val}"
            )

    # Check for changed mutable parameters (informational)
    changed_params = []
    for param in MUTABLE_PARAMS:
        resume_val = resume_config.get(param)
        new_val = getattr(new_config, param, None)

        if resume_val != new_val:
            changed_params.append(f"  {param}: {resume_val} → {new_val}")

    if changed_params:
        warnings.append(
            "Training hyperparameters changed:\n" + "\n".join(changed_params)
        )

    # Raise or return
    if errors:
        error_msg = "Cannot resume training due to incompatible config:\n" + "\n".join(errors)
        if strict:
            raise ValueError(error_msg)
        return False, warnings + [error_msg]

    return True, warnings


def load_resume_checkpoint(
    resume_dir: str,
    config: RankingConfig,
    device: torch.device
) -> Tuple[SiameseRanker, torch.optim.Optimizer, Dict, int, float, Dict]:
    """
    Load checkpoint for resume training.

    Args:
        resume_dir: Directory containing checkpoint files
        config: New training config (validated against checkpoint config)
        device: Device to load to

    Returns:
        Tuple of:
            - model: Loaded model with weights
            - optimizer: Optimizer with restored state
            - history: Training history dict
            - start_epoch: Epoch to resume from (checkpoint epoch + 1)
            - best_val_auc: Best validation AUC from checkpoint
            - scenario_stats: Loaded scenario normalization stats

    Raises:
        FileNotFoundError: If required files missing
        ValueError: If config incompatible
    """
    resume_path = Path(resume_dir)

    # Validate directory exists
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume directory does not exist: {resume_dir}")

    # Load checkpoint file (auto-detect)
    checkpoint_file = None
    for candidate in ['best_model.pt', 'latest_model.pt', 'checkpoint.pt', 'model.pt']:
        candidate_path = resume_path / candidate
        if candidate_path.exists():
            checkpoint_file = candidate_path
            break

    if checkpoint_file is None:
        # Try any .pt file
        pt_files = list(resume_path.glob('*.pt'))
        if pt_files:
            checkpoint_file = pt_files[0]
        else:
            raise FileNotFoundError(
                f"No checkpoint file found in: {resume_dir}\n"
                f"Expected: best_model.pt, checkpoint.pt, or any .pt file"
            )

    print(f"Loading checkpoint: {checkpoint_file}")

    try:
        checkpoint = torch.load(checkpoint_file, map_location=device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint from {checkpoint_file}\n"
            f"File may be corrupted. Error: {e}"
        )

    # Validate config compatibility
    resume_config = checkpoint['config']
    is_compatible, warnings_list = validate_resume_config(resume_config, config, strict=True)

    # Print warnings
    for warning in warnings_list:
        print(f"WARNING: {warning}")

    # Load scenario stats (REQUIRED)
    stats_file = resume_path / "scenario_stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(
            f"Missing scenario_stats.json in: {resume_dir}\n"
            f"This file is required for consistent normalization."
        )

    with open(stats_file, 'r') as f:
        stats_data = json.load(f)
        # Handle both formats: direct stats or nested under 'scenario_stats' key
        if 'scenario_stats' in stats_data:
            scenario_stats = stats_data['scenario_stats']
        elif 'means' in stats_data and 'stds' in stats_data:
            scenario_stats = stats_data
        else:
            raise ValueError(
                f"Invalid scenario_stats.json format. Expected:\n"
                f"  {{'means': [...], 'stds': [...]}}\n"
                f"Got: {list(stats_data.keys())}"
            )

    print(f"Loaded scenario stats from: {stats_file}")

    # Load or use checkpoint history
    history_file = resume_path / "training_history.json"
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        print(f"Loaded training history: {len(history['train_loss'])} epochs")
    else:
        history = checkpoint.get('history', {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'val_auc': [], 'learning_rate': []
        })
        print("Using history from checkpoint")

    # Create model with checkpoint config (architecture must match)
    checkpoint_config = RankingConfig(**resume_config)
    model = SiameseRanker(checkpoint_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Loaded model weights (from epoch {checkpoint['epoch'] + 1})")

    # Create optimizer with NEW config (allows LR changes)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Restore optimizer state (momentum buffers)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Update learning rate in optimizer state (in case user changed it)
    for param_group in optimizer.param_groups:
        param_group['lr'] = config.learning_rate

    print(f"Restored optimizer state (LR: {config.learning_rate})")

    # Extract resume info
    start_epoch = checkpoint['epoch'] + 1
    best_val_auc = checkpoint.get('val_auc', 0.0)

    return model, optimizer, history, start_epoch, best_val_auc, scenario_stats


def load_checkpoint(
    checkpoint_path: str,
    config: RankingConfig = None,
    device: torch.device = None
) -> Tuple[SiameseRanker, Dict]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file or directory containing best_model.pt
        config: Optional config (uses saved config if None)
        device: Device to load to

    Returns:
        Tuple of (model, checkpoint dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Handle directory path - auto-detect checkpoint file
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        # Look for common checkpoint file names
        candidates = ['best_model.pt', 'latest_model.pt', 'checkpoint.pt', 'model.pt']
        found = None
        for name in candidates:
            candidate_path = checkpoint_path / name
            if candidate_path.exists():
                found = candidate_path
                break
        
        if found is None:
            # Try to find any .pt file
            pt_files = list(checkpoint_path.glob('*.pt'))
            if pt_files:
                found = pt_files[0]
        
        if found is None:
            raise FileNotFoundError(
                f"No checkpoint file found in directory: {checkpoint_path}\n"
                f"Expected one of: {candidates} or any .pt file"
            )
        
        print(f"  Auto-detected checkpoint: {found}")
        checkpoint_path = found

    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    # Use saved config if not provided
    if config is None:
        config = RankingConfig(**checkpoint['config'])

    model = SiameseRanker(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    return model, checkpoint
