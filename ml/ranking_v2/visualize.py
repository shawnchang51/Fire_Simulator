"""
Visualization Tools for Pairwise Ranking Model V2

Key Visualizations:
    - Grad-CAM: Attention heatmaps on floor plan grids
    - Cross-Attention: Attention weight visualization
    - Latent PCA: 2D projection of latent space colored by score
    - Training Curves: Loss, AUC, auxiliary losses over epochs
    - Auxiliary Predictions: Scatter plots of predictions vs ground truth
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model import CrossAttentionRanker


class GradCAM:
    """
    Grad-CAM visualization for understanding CNN attention.

    Uses NAMED layer reference (`final_conv`) for robustness.
    All interpretability is on the PointwiseScorer, not pairwise comparison.
    """

    def __init__(
        self,
        model: CrossAttentionRanker,
        target_layer_name: str = 'final_conv'
    ):
        self.model = model
        self.gradients = None
        self.activations = None

        # Get encoder
        encoder = model.encoder

        # Get named layer
        target_layer = getattr(encoder, target_layer_name)

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        grid: torch.Tensor,
        scenario: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for a single input."""
        self.model.eval()

        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        if scenario.dim() == 1:
            scenario = scenario.unsqueeze(0)

        grid = grid.clone().requires_grad_(True)

        # Forward pass
        score = self.model.score_single(grid, scenario)

        # Backward pass
        self.model.zero_grad()
        score.backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        if target_size is None:
            target_size = grid.shape[-2:]
        cam = F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None
):
    """Plot training history curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    # Determine number of subplots
    has_aux = any(k.startswith('train_aux_') for k in history.keys())
    n_rows = 3 if has_aux else 2

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))

    # Loss curves
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curves
    ax = axes[0, 1]
    ax.plot(history['train_accuracy'], label='Train Acc')
    ax.plot(history['val_accuracy'], label='Val Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC curve
    ax = axes[1, 0]
    ax.plot(history['val_auc'], label='Val AUC', color='green')
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random baseline')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('Validation AUC')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 1]
    ax.plot(history['learning_rate'], label='LR', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Auxiliary losses (if available)
    if has_aux:
        ax = axes[2, 0]
        for key in history.keys():
            if key.startswith('train_aux_'):
                task = key.replace('train_aux_', '')
                ax.plot(history[key], label=f'Train {task}')
        for key in history.keys():
            if key.startswith('val_aux_'):
                task = key.replace('val_aux_', '')
                ax.plot(history[key], label=f'Val {task}', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Auxiliary Loss')
        ax.set_title('Auxiliary Task Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Empty subplot
        axes[2, 1].axis('off')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_auxiliary_predictions(
    model: CrossAttentionRanker,
    dataset,
    n_samples: int = 500,
    output_path: Optional[str] = None,
    device: torch.device = None
):
    """
    Plot auxiliary predictions vs ground truth.

    Creates scatter plots for each auxiliary task showing
    predicted vs actual values.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    predictions = {'survival_rate': [], 'steps': [], 'avg_fire_damage': []}
    targets = {'survival_rate': [], 'steps': [], 'avg_fire_damage': []}

    n_samples = min(n_samples, len(dataset))

    with torch.no_grad():
        for i in range(n_samples):
            item = dataset[i]
            grid = item['grid'].unsqueeze(0).to(device)

            preds = model.predict_auxiliary(grid)

            for task in predictions.keys():
                if task in preds:
                    predictions[task].append(preds[task].cpu().item())
                if task in item:
                    targets[task].append(item[task])

    # Filter to tasks with data
    tasks = [t for t in predictions.keys() if len(predictions[t]) > 0 and len(targets[t]) > 0]

    if not tasks:
        print("No auxiliary tasks to visualize")
        return

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        preds = np.array(predictions[task])
        targs = np.array(targets[task])

        ax.scatter(targs, preds, alpha=0.5, s=10)
        ax.plot([targs.min(), targs.max()], [targs.min(), targs.max()],
                'r--', label='Ideal')
        ax.set_xlabel(f'Ground Truth {task}')
        ax.set_ylabel(f'Predicted {task}')
        ax.set_title(f'{task} Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved auxiliary predictions to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_gradcam_sample(
    model: CrossAttentionRanker,
    sample: Dict,
    output_path: Optional[str] = None,
    device: torch.device = None
):
    """Generate and display Grad-CAM visualization for a sample."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    if device is None:
        device = next(model.parameters()).device

    gradcam = GradCAM(model)
    grid = sample['grid'].to(device)
    scenario = sample['scenario'].to(device)

    cam = gradcam.generate(grid, scenario)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original floor plan (passable areas)
    ax = axes[0]
    floor_plan = grid[1].cpu().numpy()
    ax.imshow(floor_plan, cmap='gray')
    ax.set_title('Floor Plan (Passable Areas)')
    ax.axis('off')

    # Grad-CAM heatmap
    ax = axes[1]
    ax.imshow(cam, cmap='jet')
    ax.set_title('Grad-CAM Attention')
    ax.axis('off')

    # Overlay
    ax = axes[2]
    ax.imshow(floor_plan, cmap='gray')
    ax.imshow(cam, cmap='jet', alpha=0.5)
    ax.set_title('Grad-CAM Overlay')
    ax.axis('off')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved Grad-CAM visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_latent_pca(
    model: CrossAttentionRanker,
    dataset,
    n_samples: int = 1000,
    output_path: Optional[str] = None,
    device: torch.device = None
):
    """Create PCA visualization of latent space colored by score."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        print("matplotlib and sklearn required for visualization")
        return

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    latents = []
    scores = []

    n_samples = min(n_samples, len(dataset))

    with torch.no_grad():
        for i in range(n_samples):
            item = dataset[i]
            grid = item['grid'].unsqueeze(0).to(device)

            latent = model.get_latent(grid)
            latents.append(latent.cpu().numpy().squeeze())
            scores.append(item['ground_truth_score'])

    latents = np.stack(latents)
    scores = np.array(scores)

    # PCA to 2D
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=scores,
        cmap='RdYlGn',
        alpha=0.7,
        s=25,
        edgecolors='none'
    )
    plt.colorbar(scatter, ax=ax, label='Ground Truth Score')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('Latent Space PCA')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved PCA plot to {output_path}")
    else:
        plt.show()

    plt.close()


def generate_all_visualizations(
    model: CrossAttentionRanker,
    dataset,
    history: Dict,
    output_dir: str,
    n_samples: int = 500,
    device: torch.device = None
):
    """Generate all visualizations and save to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualizations in {output_dir}...")

    # Training history
    if history:
        plot_training_history(
            history,
            output_path=str(output_dir / "training_history.png")
        )

    # Latent PCA
    try:
        plot_latent_pca(
            model, dataset, n_samples,
            output_path=str(output_dir / "latent_pca.png"),
            device=device
        )
    except Exception as e:
        print(f"Failed to generate PCA plot: {e}")

    # Auxiliary predictions
    try:
        plot_auxiliary_predictions(
            model, dataset, n_samples,
            output_path=str(output_dir / "auxiliary_predictions.png"),
            device=device
        )
    except Exception as e:
        print(f"Failed to generate auxiliary plot: {e}")

    # Grad-CAM for first sample
    try:
        sample = dataset[0]
        visualize_gradcam_sample(
            model, sample,
            output_path=str(output_dir / "gradcam_sample_0.png"),
            device=device
        )
    except Exception as e:
        print(f"Failed to generate Grad-CAM: {e}")

    print("Visualization generation complete!")


def encode_grid_5ch(grid_2d: np.ndarray, target_size=(96, 128)) -> np.ndarray:
    """
    Create 5-channel tensor from 2D grid.

    Channels:
        0: Wall mask (grid == -2)
        1: Passable mask (grid == 0)
        2: Door positions (empty for visualization)
        3: Exit positions (empty for visualization)
        4: Valid mask (1.0 for real grid, 0.0 for padding)
    """
    H, W = grid_2d.shape
    tH, tW = target_size

    encoded = np.full((5, tH, tW), -1.0, dtype=np.float32)

    H_copy = min(H, tH)
    W_copy = min(W, tW)

    encoded[0, :H_copy, :W_copy] = (grid_2d[:H_copy, :W_copy] == -2).astype(np.float32)
    encoded[1, :H_copy, :W_copy] = (grid_2d[:H_copy, :W_copy] == 0).astype(np.float32)
    encoded[2, :H_copy, :W_copy] = 0.0
    encoded[3, :H_copy, :W_copy] = 0.0
    encoded[4, :H_copy, :W_copy] = 1.0
    encoded[4, H_copy:, :] = 0.0
    encoded[4, :, W_copy:] = 0.0

    return encoded


def generate_gradcam_from_floor_plans(
    model: CrossAttentionRanker,
    floor_plans_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (96, 128),
    n_samples: int = 5,
    scenario_input_dim: int = 4,
    device: torch.device = None
):
    """
    Generate GradCAM visualizations directly from floor plan NPZ files.

    This function doesn't require simulation_results.jsonl - it loads
    floor plans directly and generates GradCAM heatmaps.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    if device is None:
        device = next(model.parameters()).device

    floor_plans_path = Path(floor_plans_dir)
    npz_files = sorted(floor_plans_path.glob("*.npz"))[:n_samples]

    if not npz_files:
        print(f"No NPZ files found in {floor_plans_dir}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gradcam = GradCAM(model)
    scenario = torch.zeros(scenario_input_dim, device=device)

    for i, npz_path in enumerate(npz_files):
        try:
            data = np.load(npz_path)
            grid_2d = data['grid']

            encoded = encode_grid_5ch(grid_2d, target_size)
            grid_tensor = torch.from_numpy(encoded).to(device)

            cam = gradcam.generate(grid_tensor, scenario)

            # Crop to valid region
            valid_mask = grid_tensor[4].cpu().numpy()
            rows = np.any(valid_mask > 0, axis=1)
            cols = np.any(valid_mask > 0, axis=0)

            if not np.any(rows) or not np.any(cols):
                continue

            row_min, row_max = np.where(rows)[0][[0, -1]]
            col_min, col_max = np.where(cols)[0][[0, -1]]

            floor_plan_np = grid_tensor[1].cpu().numpy()[row_min:row_max+1, col_min:col_max+1]
            cam_cropped = cam[row_min:row_max+1, col_min:col_max+1]

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(floor_plan_np, cmap='gray')
            axes[0].set_title('Floor Plan (Passable Areas)')
            axes[0].axis('off')

            axes[1].imshow(cam_cropped, cmap='jet')
            axes[1].set_title('GradCAM Attention')
            axes[1].axis('off')

            axes[2].imshow(floor_plan_np, cmap='gray')
            axes[2].imshow(cam_cropped, cmap='jet', alpha=0.5)
            axes[2].set_title('GradCAM Overlay')
            axes[2].axis('off')

            plt.tight_layout()
            save_path = output_path / f"gradcam_{npz_path.stem}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved GradCAM to {save_path}")

        except Exception as e:
            print(f"Failed to generate GradCAM for {npz_path.name}: {e}")


def apply_rotation(grid: torch.Tensor, k: int, target_size: Tuple[int, int]) -> torch.Tensor:
    """
    Apply 90-degree rotation k times and reposition to top-left.

    Args:
        grid: Grid tensor of shape (5, H, W)
        k: Number of 90-degree counter-clockwise rotations (0-3)
        target_size: (tH, tW) target size for output

    Returns:
        Rotated grid tensor of shape (5, tH, tW)
    """
    if k == 0:
        return grid.clone()

    tH, tW = target_size

    # Apply rotation
    rotated = torch.rot90(grid, k=k, dims=(1, 2))

    # Create new grid with padding values
    result = torch.full((5, tH, tW), -1.0, dtype=grid.dtype, device=grid.device)
    result[4] = 0.0  # Valid mask padding is 0.0

    # Find the valid content region
    valid_mask = rotated[4]
    valid_rows = (valid_mask > 0.5).any(dim=1).nonzero(as_tuple=True)[0]
    valid_cols = (valid_mask > 0.5).any(dim=0).nonzero(as_tuple=True)[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        return result

    # Get bounding box of valid content
    min_row, max_row = valid_rows[0].item(), valid_rows[-1].item()
    min_col, max_col = valid_cols[0].item(), valid_cols[-1].item()
    content_h = max_row - min_row + 1
    content_w = max_col - min_col + 1

    # Crop if needed
    copy_h = min(content_h, tH)
    copy_w = min(content_w, tW)

    # Copy to top-left
    result[:, :copy_h, :copy_w] = rotated[
        :,
        min_row:min_row + copy_h,
        min_col:min_col + copy_w
    ]

    return result


def apply_shift(grid: torch.Tensor, shift_down: int, shift_right: int) -> torch.Tensor:
    """
    Apply shift augmentation to grid.

    Args:
        grid: Grid tensor of shape (5, H, W)
        shift_down: Pixels to shift down (positive = down)
        shift_right: Pixels to shift right (positive = right)

    Returns:
        Shifted grid tensor
    """
    _, tH, tW = grid.shape
    shifted = torch.full_like(grid, -1.0)
    shifted[4] = 0.0  # Valid mask

    # Compute source region
    src_h = tH - abs(shift_down)
    src_w = tW - abs(shift_right)

    if src_h <= 0 or src_w <= 0:
        return shifted

    # Handle negative shifts
    src_row_start = max(0, -shift_down)
    src_col_start = max(0, -shift_right)
    dst_row_start = max(0, shift_down)
    dst_col_start = max(0, shift_right)

    shifted[:, dst_row_start:dst_row_start+src_h, dst_col_start:dst_col_start+src_w] = \
        grid[:, src_row_start:src_row_start+src_h, src_col_start:src_col_start+src_w]

    return shifted


def visualize_augmentation_comparison(
    model: CrossAttentionRanker,
    sample: Dict,
    output_path: Optional[str] = None,
    device: torch.device = None,
    shift_amounts: List[Tuple[int, int]] = None
):
    """
    Visualize how model scores change under rotation and shift transformations.

    Creates a comparison showing:
    - Original floor plan with GradCAM and score
    - Rotated versions (90°, 180°, 270°) with GradCAM and scores
    - Shifted versions with GradCAM and scores

    Args:
        model: The ranking model
        sample: Dataset sample with 'grid' and 'scenario'
        output_path: Path to save the figure
        device: Computation device
        shift_amounts: List of (down, right) shift amounts to test
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    if device is None:
        device = next(model.parameters()).device

    if shift_amounts is None:
        shift_amounts = [(10, 0), (0, 10), (10, 10)]

    grid = sample['grid'].to(device)
    scenario = sample['scenario'].to(device)
    target_size = grid.shape[-2:]

    gradcam = GradCAM(model)
    model.eval()

    # Collect all transformations
    transformations = []

    # Original
    transformations.append(('Original', grid.clone()))

    # Rotations
    for k in [1, 2, 3]:
        angle = k * 90
        rotated = apply_rotation(grid, k, target_size)
        transformations.append((f'Rotate {angle}°', rotated))

    # Shifts
    for down, right in shift_amounts:
        shifted = apply_shift(grid, down, right)
        label = f'Shift ({down:+d}, {right:+d})'
        transformations.append((label, shifted))

    # Compute scores and GradCAMs
    results = []
    with torch.no_grad():
        for label, g in transformations:
            score = model.score_single(g.unsqueeze(0), scenario.unsqueeze(0))
            cam = gradcam.generate(g, scenario)
            results.append((label, g, score.item(), cam))

    # Create figure
    n_cols = len(results)
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))

    for col, (label, g, score, cam) in enumerate(results):
        floor_plan = g[1].cpu().numpy()

        # Row 0: Floor plan
        ax = axes[0, col]
        ax.imshow(floor_plan, cmap='gray')
        ax.set_title(f'{label}\nScore: {score:.4f}', fontsize=10)
        ax.axis('off')

        # Row 1: GradCAM
        ax = axes[1, col]
        ax.imshow(cam, cmap='jet')
        ax.set_title('GradCAM', fontsize=10)
        ax.axis('off')

        # Row 2: Overlay
        ax = axes[2, col]
        ax.imshow(floor_plan, cmap='gray')
        ax.imshow(cam, cmap='jet', alpha=0.5)
        ax.set_title('Overlay', fontsize=10)
        ax.axis('off')

    # Add row labels
    axes[0, 0].set_ylabel('Floor Plan', fontsize=12)
    axes[1, 0].set_ylabel('GradCAM', fontsize=12)
    axes[2, 0].set_ylabel('Overlay', fontsize=12)

    # Add summary
    original_score = results[0][2]
    score_diffs = [abs(r[2] - original_score) for r in results[1:]]
    avg_diff = np.mean(score_diffs)
    max_diff = np.max(score_diffs)

    fig.suptitle(
        f'Augmentation Comparison | Avg Score Diff: {avg_diff:.4f} | Max Diff: {max_diff:.4f}',
        fontsize=14, y=1.02
    )

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved augmentation comparison to {output_path}")
    else:
        plt.show()

    plt.close()

    return results


def visualize_multiple_augmentation_samples(
    model: CrossAttentionRanker,
    dataset,
    n_samples: int = 3,
    output_dir: str = "viz_v2",
    device: torch.device = None,
    shift_amounts: List[Tuple[int, int]] = None
):
    """
    Generate augmentation comparison visualizations for multiple samples.

    Args:
        model: The ranking model
        dataset: Dataset to sample from
        n_samples: Number of samples to visualize
        output_dir: Output directory for figures
        device: Computation device
        shift_amounts: List of (down, right) shift amounts
    """
    if device is None:
        device = next(model.parameters()).device

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_samples = min(n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    print(f"Generating augmentation comparisons for {n_samples} samples...")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        save_path = str(output_path / f"augmentation_comparison_{i}.png")
        visualize_augmentation_comparison(
            model=model,
            sample=sample,
            output_path=save_path,
            device=device,
            shift_amounts=shift_amounts
        )

    print(f"Saved {n_samples} augmentation comparisons to {output_dir}")
