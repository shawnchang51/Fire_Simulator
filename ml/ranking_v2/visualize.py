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
    scenario_dim = 5  # Default scenario dimension
    scenario = torch.zeros(scenario_dim, device=device)

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
