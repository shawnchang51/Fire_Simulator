"""
Training loop for Fire Simulation Surrogate Model
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ModelConfig
from .model import FireSimulationSurrogate


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        grid = batch['grid'].to(device)
        scenario = batch['scenario'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad()
        predictions = model(grid, scenario)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            grid = batch['grid'].to(device)
            scenario = batch['scenario'].to(device)
            targets = batch['targets'].to(device)

            predictions = model(grid, scenario)
            loss = criterion(predictions, targets)

            total_loss += loss.item()
            num_batches += 1

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute per-output MSE
    per_output_mse = np.mean((all_predictions - all_targets) ** 2, axis=0)

    return {
        'loss': total_loss / num_batches,
        'per_output_mse': per_output_mse.tolist(),
        'overall_mean_loss': np.mean(per_output_mse)
    }


def train_model(
    model: FireSimulationSurrogate,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ModelConfig,
    device: torch.device,
    checkpoint_path: Optional[str] = None
) -> Dict[str, List]:
    """
    Full training loop with early stopping and checkpointing.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Model configuration
        device: Device to train on
        checkpoint_path: Path to save best checkpoint (optional)

    Returns:
        Training history dict with losses and metrics
    """
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Loss function
    criterion = nn.MSELoss()

    # Checkpoint path
    if checkpoint_path is None:
        checkpoint_path = Path(config.checkpoint_dir) / "best_model.pt"
    else:
        checkpoint_path = Path(checkpoint_path)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_per_output_mse': [],
        'val_overall_mean_loss': [],
        'learning_rate': []
    }

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Model parameters: {model.count_parameters():,}")
    print()

    for epoch in range(config.epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_per_output_mse'].append(val_metrics['per_output_mse'])
        history['val_overall_mean_loss'].append(val_metrics['overall_mean_loss'])
        history['learning_rate'].append(current_lr)

        # Print progress
        output_names = config.output_names
        per_output_str = ", ".join([
            f"{name[:8]}={mse:.4f}"
            for name, mse in zip(output_names, val_metrics['per_output_mse'])
        ])

        print(f"Epoch {epoch + 1:3d}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Overall: {val_metrics['overall_mean_loss']:.4f} | "
              f"LR: {current_lr:.2e}")
        print(f"           Per-output MSE: {per_output_str}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
            print(f"           -> Saved checkpoint (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (patience: {config.early_stopping_patience})")
            break

        print()

    # Load best model
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch']} (val_loss: {checkpoint['val_loss']:.4f})")

    return history


def load_checkpoint(
    model: FireSimulationSurrogate,
    checkpoint_path: str,
    device: torch.device
) -> Dict:
    """Load a model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint
