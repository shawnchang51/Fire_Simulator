"""
Visualization Tools for Fire Simulation Surrogate Model

Provides:
    - GradCAM: Multi-output attention heatmaps for understanding CNN attention
    - Cell-wise Counterfactual: Perturbation-based importance maps
    - Combined visualizations with floor plan overlay

Usage:
    python -m ml.surrogate.run_training --visualize --checkpoint checkpoints/best_model.pt
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import FireSimulationSurrogate

# Output names for labeling
OUTPUT_NAMES = ['survival_rate', 'avg_evacuation_time', 'steps', 'avg_fire_damage']


def get_floor_plan_bounds(grid: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    Find the bounding box of the actual floor plan content (excluding padding).

    The grid is padded with walls (channel 0 = 1, others = 0).
    We find the region where there's actual floor plan content.

    Args:
        grid: Grid tensor (4, H, W) or (1, 4, H, W)

    Returns:
        (y_min, y_max, x_min, x_max) bounding box indices
    """
    if grid.dim() == 4:
        grid = grid.squeeze(0)

    # Content exists where passable (ch1), doors (ch2), or exits (ch3) have values
    # Or where walls form actual structure (not uniform padding)
    content_mask = (grid[1] > 0) | (grid[2] > 0) | (grid[3] > 0)

    # Also check for interior walls (wall channel with neighbors that are passable)
    # Simpler: just use passable + doors + exits as the content indicator

    if not content_mask.any():
        # Fallback: return full grid
        return 0, grid.shape[1], 0, grid.shape[2]

    # Find bounding box
    rows = content_mask.any(dim=1)
    cols = content_mask.any(dim=0)

    y_indices = torch.where(rows)[0]
    x_indices = torch.where(cols)[0]

    y_min, y_max = y_indices[0].item(), y_indices[-1].item() + 1
    x_min, x_max = x_indices[0].item(), x_indices[-1].item() + 1

    # Add small padding for better visualization
    pad = 2
    y_min = max(0, y_min - pad)
    y_max = min(grid.shape[1], y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(grid.shape[2], x_max + pad)

    return y_min, y_max, x_min, x_max


def crop_to_bounds(
    arr: np.ndarray,
    bounds: Tuple[int, int, int, int]
) -> np.ndarray:
    """Crop a 2D array to the given bounds."""
    y_min, y_max, x_min, x_max = bounds
    return arr[y_min:y_max, x_min:x_max]


class SurrogateGradCAM:
    """
    Multi-output Grad-CAM for FireSimulationSurrogate.

    Generates separate attention heatmaps for each of the 4 outputs,
    showing which spatial regions the CNN focuses on for each prediction.

    Reference:
        Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks" (2017)
    """

    def __init__(
        self,
        model: FireSimulationSurrogate,
        target_layer_index: int = -3  # BatchNorm2d before last ReLU (avoids inplace ReLU issue)
    ):
        """
        Initialize Grad-CAM.

        Args:
            model: Trained FireSimulationSurrogate model
            target_layer_index: Index in conv_layers Sequential to hook
                               Default -3 targets BatchNorm2d (before ReLU with inplace=True)
                               Structure: [Conv2d, BatchNorm2d, ReLU, MaxPool2d] * 4
        """
        self.model = model
        self.gradients = None
        self.activations = None

        # Access the CNN's conv_layers Sequential
        # Structure: [Conv2d(-4), BatchNorm2d(-3), ReLU(-2), MaxPool2d(-1)] in last block
        # We hook BatchNorm2d to avoid conflict with inplace ReLU
        conv_layers = model.cnn.conv_layers
        target_layer = conv_layers[target_layer_index]

        # Register hooks (use register_full_backward_hook for PyTorch >= 1.9)
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook to save gradients."""
        self.gradients = grad_output[0].detach()

    def _disable_inplace_relu(self):
        """Temporarily disable inplace ReLU to avoid gradient conflicts."""
        self._original_inplace = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU) and module.inplace:
                self._original_inplace[name] = True
                module.inplace = False

    def _restore_inplace_relu(self):
        """Restore original inplace settings."""
        for name, module in self.model.named_modules():
            if name in self._original_inplace:
                module.inplace = True
        self._original_inplace = {}

    def generate(
        self,
        grid: torch.Tensor,
        scenario: torch.Tensor,
        output_index: int = 0,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific output.

        Args:
            grid: Grid tensor of shape (4, H, W) or (1, 4, H, W)
            scenario: Scenario tensor of shape (4,) or (1, 4)
            output_index: Which output to compute gradient for (0-3)
            target_size: Optional output size (H, W), defaults to input size

        Returns:
            Heatmap of shape (H, W) with values in [0, 1]
        """
        self.model.eval()

        # Temporarily disable inplace ReLU to avoid gradient hook conflicts
        self._disable_inplace_relu()

        try:
            # Add batch dimension if needed
            if grid.dim() == 3:
                grid = grid.unsqueeze(0)
            if scenario.dim() == 1:
                scenario = scenario.unsqueeze(0)

            # Ensure requires_grad for backward
            grid = grid.clone().requires_grad_(True)

            # Forward pass
            outputs = self.model(grid, scenario)

            # Backward pass from specific output
            self.model.zero_grad()
            target_output = outputs[0, output_index]
            target_output.backward()

            # Compute Grad-CAM
            # Global average pooling of gradients
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)

            # Weighted combination of activations
            cam = (weights * self.activations).sum(dim=1, keepdim=True)

            # ReLU: only positive contributions
            cam = F.relu(cam)

            # Resize to input size
            if target_size is None:
                target_size = grid.shape[-2:]
            cam = F.interpolate(
                cam,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )

            # Normalize to [0, 1]
            cam = cam.squeeze().cpu().numpy()
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

            return cam

        finally:
            # Restore original inplace settings
            self._restore_inplace_relu()

    def generate_all(
        self,
        grid: torch.Tensor,
        scenario: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmaps for all 4 outputs.

        Args:
            grid: Grid tensor of shape (4, H, W) or (1, 4, H, W)
            scenario: Scenario tensor of shape (4,) or (1, 4)
            target_size: Optional output size (H, W)

        Returns:
            Dict mapping output name to heatmap array
        """
        heatmaps = {}
        for i, name in enumerate(OUTPUT_NAMES):
            heatmaps[name] = self.generate(grid, scenario, output_index=i, target_size=target_size)
        return heatmaps


def compute_cell_counterfactual(
    model: FireSimulationSurrogate,
    grid: torch.Tensor,
    scenario: torch.Tensor,
    perturbation_mode: str = 'wall',
    output_index: int = 0,
    device: torch.device = None
) -> np.ndarray:
    """
    Compute cell-wise importance by perturbation (counterfactual analysis).

    For each grid cell, perturbs it and measures the change in model output.
    This reveals which cells are most important for the prediction.

    Args:
        model: Trained surrogate model
        grid: Original grid tensor (4, H, W)
        scenario: Scenario parameters (4,)
        perturbation_mode: 'wall' (make impassable) or 'passable' (make passable)
        output_index: Which output to analyze (0-3)
        device: Device to use

    Returns:
        importance_map: (H, W) array of importance scores (absolute change)
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Ensure correct dimensions
    if grid.dim() == 4:
        grid = grid.squeeze(0)
    if scenario.dim() == 2:
        scenario = scenario.squeeze(0)

    grid = grid.to(device)
    scenario = scenario.to(device)

    H, W = grid.shape[1], grid.shape[2]

    # Get baseline prediction
    with torch.no_grad():
        baseline_pred = model(grid.unsqueeze(0), scenario.unsqueeze(0))
        baseline_value = baseline_pred[0, output_index].item()

    importance_map = np.zeros((H, W), dtype=np.float32)

    # For each cell
    for i in range(H):
        for j in range(W):
            # Create perturbed grid
            perturbed = grid.clone()

            if perturbation_mode == 'wall':
                # Make cell impassable (wall)
                perturbed[0, i, j] = 1.0  # Set wall channel
                perturbed[1, i, j] = 0.0  # Clear passable channel
            else:  # 'passable'
                # Make cell passable
                perturbed[0, i, j] = 0.0  # Clear wall channel
                perturbed[1, i, j] = 1.0  # Set passable channel

            # Get perturbed prediction
            with torch.no_grad():
                perturbed_pred = model(perturbed.unsqueeze(0), scenario.unsqueeze(0))
                perturbed_value = perturbed_pred[0, output_index].item()

            # Importance = absolute change in output
            importance_map[i, j] = abs(baseline_value - perturbed_value)

    return importance_map


def compute_cell_counterfactual_batched(
    model: FireSimulationSurrogate,
    grid: torch.Tensor,
    scenario: torch.Tensor,
    perturbation_mode: str = 'wall',
    output_index: int = 0,
    batch_size: int = 256,
    device: torch.device = None,
    show_progress: bool = True
) -> np.ndarray:
    """
    Batched version of cell counterfactual for efficiency.

    Processes multiple cell perturbations in parallel batches.

    Args:
        model: Trained surrogate model
        grid: Original grid tensor (4, H, W)
        scenario: Scenario parameters (4,)
        perturbation_mode: 'wall' or 'passable'
        output_index: Which output to analyze (0-3)
        batch_size: Number of perturbations per batch
        device: Device to use
        show_progress: Whether to print progress

    Returns:
        importance_map: (H, W) array of importance scores
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    # Ensure correct dimensions
    if grid.dim() == 4:
        grid = grid.squeeze(0)
    if scenario.dim() == 2:
        scenario = scenario.squeeze(0)

    grid = grid.to(device)
    scenario = scenario.to(device)

    H, W = grid.shape[1], grid.shape[2]
    total_cells = H * W

    # Get baseline prediction
    with torch.no_grad():
        baseline_pred = model(grid.unsqueeze(0), scenario.unsqueeze(0))
        baseline_value = baseline_pred[0, output_index].item()

    # Collect all cell indices
    cell_indices = [(i, j) for i in range(H) for j in range(W)]

    importance_flat = np.zeros(total_cells, dtype=np.float32)

    # Process in batches
    num_batches = (total_cells + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_cells)
        batch_indices = cell_indices[start:end]
        current_batch_size = len(batch_indices)

        if show_progress and batch_idx % 10 == 0:
            print(f"  Processing batch {batch_idx + 1}/{num_batches} ({start}/{total_cells} cells)")

        # Create batch of perturbed grids
        batch_grids = grid.unsqueeze(0).repeat(current_batch_size, 1, 1, 1)
        batch_scenarios = scenario.unsqueeze(0).repeat(current_batch_size, 1)

        for k, (i, j) in enumerate(batch_indices):
            if perturbation_mode == 'wall':
                batch_grids[k, 0, i, j] = 1.0
                batch_grids[k, 1, i, j] = 0.0
            else:
                batch_grids[k, 0, i, j] = 0.0
                batch_grids[k, 1, i, j] = 1.0

        # Forward pass
        with torch.no_grad():
            batch_preds = model(batch_grids, batch_scenarios)
            batch_values = batch_preds[:, output_index].cpu().numpy()

        # Compute importance
        importance_flat[start:end] = np.abs(baseline_value - batch_values)

    # Reshape to 2D
    importance_map = importance_flat.reshape(H, W)

    return importance_map


def compute_all_counterfactuals(
    model: FireSimulationSurrogate,
    grid: torch.Tensor,
    scenario: torch.Tensor,
    batch_size: int = 256,
    device: torch.device = None,
    show_progress: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute counterfactual importance maps for all outputs and both perturbation modes.

    Args:
        model: Trained surrogate model
        grid: Original grid tensor (4, H, W)
        scenario: Scenario parameters (4,)
        batch_size: Batch size for processing
        device: Device to use
        show_progress: Whether to print progress

    Returns:
        Nested dict: {output_name: {'wall': map, 'passable': map}}
    """
    results = {}

    for i, name in enumerate(OUTPUT_NAMES):
        if show_progress:
            print(f"\nComputing counterfactual for output: {name}")

        results[name] = {}

        for mode in ['wall', 'passable']:
            if show_progress:
                print(f"  Mode: {mode}")
            results[name][mode] = compute_cell_counterfactual_batched(
                model, grid, scenario,
                perturbation_mode=mode,
                output_index=i,
                batch_size=batch_size,
                device=device,
                show_progress=show_progress
            )

    return results


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_gradcam_grid(
    grid: torch.Tensor,
    heatmaps: Dict[str, np.ndarray],
    predictions: Optional[Dict[str, float]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    crop_to_content: bool = True
):
    """
    Plot GradCAM heatmaps for all 4 outputs in a grid layout.

    Layout: 4 rows (one per output) x 3 columns (floor plan, heatmap, overlay)

    Args:
        grid: Original grid tensor (4, H, W)
        heatmaps: Dict mapping output name to heatmap array
        predictions: Optional dict of prediction values for titles
        output_path: Optional path to save figure
        figsize: Figure size
        crop_to_content: If True, crop to actual floor plan bounds
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    # Extract floor plan (passable channel)
    if grid.dim() == 4:
        grid = grid.squeeze(0)

    # Get crop bounds
    if crop_to_content:
        bounds = get_floor_plan_bounds(grid)
    else:
        bounds = (0, grid.shape[1], 0, grid.shape[2])

    floor_plan = crop_to_bounds(grid[1].cpu().numpy(), bounds)

    fig, axes = plt.subplots(4, 3, figsize=figsize)

    for row, name in enumerate(OUTPUT_NAMES):
        cam_full = heatmaps.get(name, np.zeros((grid.shape[1], grid.shape[2])))
        cam = crop_to_bounds(cam_full, bounds)

        # Floor plan
        ax = axes[row, 0]
        ax.imshow(floor_plan, cmap='gray')
        ax.set_title(f'{name}\n(Floor Plan)', fontsize=10)
        ax.axis('off')

        # Heatmap
        ax = axes[row, 1]
        im = ax.imshow(cam, cmap='jet', vmin=0, vmax=1)
        ax.set_title('GradCAM', fontsize=10)
        ax.axis('off')

        # Overlay
        ax = axes[row, 2]
        ax.imshow(floor_plan, cmap='gray')
        ax.imshow(cam, cmap='jet', alpha=0.5)
        title = 'Overlay'
        if predictions and name in predictions:
            title += f'\nPred: {predictions[name]:.4f}'
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # Add colorbar
    fig.colorbar(im, ax=axes, shrink=0.6, label='Attention')

    plt.suptitle('GradCAM Visualization (All Outputs)', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved GradCAM visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_counterfactual(
    grid: torch.Tensor,
    importance_map: np.ndarray,
    perturbation_mode: str,
    output_name: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot a single counterfactual importance map.

    Args:
        grid: Original grid tensor (4, H, W)
        importance_map: Importance values (H, W)
        perturbation_mode: 'wall' or 'passable'
        output_name: Name of the output being analyzed
        output_path: Optional path to save figure
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    if grid.dim() == 4:
        grid = grid.squeeze(0)
    floor_plan = grid[1].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Floor plan
    axes[0].imshow(floor_plan, cmap='gray')
    axes[0].set_title('Floor Plan', fontsize=12)
    axes[0].axis('off')

    # Importance heatmap
    im = axes[1].imshow(importance_map, cmap='hot')
    axes[1].set_title(f'Cell Importance\n({perturbation_mode} perturbation)', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    # Overlay
    axes[2].imshow(floor_plan, cmap='gray')
    axes[2].imshow(importance_map, cmap='hot', alpha=0.6)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')

    plt.suptitle(f'Cell-wise Counterfactual: {output_name}', fontsize=14)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved counterfactual visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_counterfactual_comparison(
    grid: torch.Tensor,
    wall_map: np.ndarray,
    passable_map: np.ndarray,
    output_name: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4),
    crop_to_content: bool = True
):
    """
    Plot wall vs passable counterfactual comparison side by side.

    Args:
        grid: Original grid tensor (4, H, W)
        wall_map: Importance map for wall perturbation
        passable_map: Importance map for passable perturbation
        output_name: Name of the output being analyzed
        output_path: Optional path to save figure
        figsize: Figure size
        crop_to_content: If True, crop to actual floor plan bounds
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    if grid.dim() == 4:
        grid = grid.squeeze(0)

    # Get crop bounds
    if crop_to_content:
        bounds = get_floor_plan_bounds(grid)
    else:
        bounds = (0, grid.shape[1], 0, grid.shape[2])

    floor_plan = crop_to_bounds(grid[1].cpu().numpy(), bounds)
    wall_map_crop = crop_to_bounds(wall_map, bounds)
    passable_map_crop = crop_to_bounds(passable_map, bounds)

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Floor plan
    axes[0].imshow(floor_plan, cmap='gray')
    axes[0].set_title('Floor Plan', fontsize=12)
    axes[0].axis('off')

    # Wall perturbation
    im1 = axes[1].imshow(wall_map_crop, cmap='hot')
    axes[1].set_title('Wall Perturbation\n(Make Impassable)', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Passable perturbation
    im2 = axes[2].imshow(passable_map_crop, cmap='hot')
    axes[2].set_title('Passable Perturbation\n(Make Passable)', fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    # Difference (wall - passable)
    diff_map = wall_map_crop - passable_map_crop
    vmax = np.abs(diff_map).max() if np.abs(diff_map).max() > 0 else 1
    im3 = axes[3].imshow(diff_map, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[3].set_title('Difference\n(Wall - Passable)', fontsize=11)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], shrink=0.8)

    plt.suptitle(f'Cell-wise Counterfactual: {output_name}', fontsize=14)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved counterfactual comparison to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_sample(
    model: FireSimulationSurrogate,
    grid: torch.Tensor,
    scenario: torch.Tensor,
    output_dir: Optional[str] = None,
    sample_name: str = "sample",
    compute_counterfactual: bool = True,
    counterfactual_batch_size: int = 256,
    device: torch.device = None,
    show_progress: bool = True
) -> Dict:
    """
    Generate complete visualization for a single sample.

    Generates:
    - GradCAM heatmaps for all 4 outputs
    - Cell-wise counterfactual maps (if requested)

    Args:
        model: Trained surrogate model
        grid: Grid tensor (4, H, W) or (1, 4, H, W)
        scenario: Scenario tensor (4,) or (1, 4)
        output_dir: Directory to save visualizations (None to display)
        sample_name: Name prefix for output files
        compute_counterfactual: Whether to compute counterfactual analysis
        counterfactual_batch_size: Batch size for counterfactual computation
        device: Device to use
        show_progress: Whether to print progress

    Returns:
        Dict with all generated heatmaps and predictions
    """
    if device is None:
        device = next(model.parameters()).device

    # Ensure correct dimensions
    if grid.dim() == 4:
        grid = grid.squeeze(0)
    if scenario.dim() == 2:
        scenario = scenario.squeeze(0)

    grid = grid.to(device)
    scenario = scenario.to(device)

    results = {}

    # Get predictions
    model.eval()
    with torch.no_grad():
        preds = model(grid.unsqueeze(0), scenario.unsqueeze(0))
        predictions = {name: preds[0, i].item() for i, name in enumerate(OUTPUT_NAMES)}
    results['predictions'] = predictions

    if show_progress:
        print(f"Predictions: {predictions}")

    # Generate GradCAM
    if show_progress:
        print("\nGenerating GradCAM heatmaps...")
    gradcam = SurrogateGradCAM(model)
    heatmaps = gradcam.generate_all(grid, scenario)
    results['gradcam'] = heatmaps

    # Plot GradCAM
    gradcam_path = Path(output_dir) / f"{sample_name}_gradcam.png" if output_dir else None
    plot_gradcam_grid(grid, heatmaps, predictions, output_path=gradcam_path)

    # Compute counterfactual if requested
    if compute_counterfactual:
        if show_progress:
            print("\nComputing cell-wise counterfactual...")

        counterfactual_results = {}

        for i, name in enumerate(OUTPUT_NAMES):
            if show_progress:
                print(f"\n  Output: {name}")

            counterfactual_results[name] = {}

            # Compute for both modes
            for mode in ['wall', 'passable']:
                if show_progress:
                    print(f"    Mode: {mode}")
                importance = compute_cell_counterfactual_batched(
                    model, grid, scenario,
                    perturbation_mode=mode,
                    output_index=i,
                    batch_size=counterfactual_batch_size,
                    device=device,
                    show_progress=False
                )
                counterfactual_results[name][mode] = importance

            # Plot comparison
            cf_path = Path(output_dir) / f"{sample_name}_counterfactual_{name}.png" if output_dir else None
            plot_counterfactual_comparison(
                grid,
                counterfactual_results[name]['wall'],
                counterfactual_results[name]['passable'],
                name,
                output_path=cf_path
            )

        results['counterfactual'] = counterfactual_results

    if show_progress:
        print("\nVisualization complete!")

    return results


def generate_visualizations(
    model: FireSimulationSurrogate,
    dataset,
    sample_indices: List[int],
    output_dir: str,
    compute_counterfactual: bool = True,
    counterfactual_batch_size: int = 256,
    device: torch.device = None
):
    """
    Generate visualizations for multiple samples.

    Args:
        model: Trained surrogate model
        dataset: Dataset containing samples
        sample_indices: List of sample indices to visualize
        output_dir: Directory to save visualizations
        compute_counterfactual: Whether to compute counterfactual analysis
        counterfactual_batch_size: Batch size for counterfactual computation
        device: Device to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualizations for {len(sample_indices)} samples...")
    print(f"Output directory: {output_dir}")

    for idx, sample_idx in enumerate(sample_indices):
        if sample_idx >= len(dataset):
            print(f"Warning: Sample index {sample_idx} out of range, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Sample {idx + 1}/{len(sample_indices)} (index: {sample_idx})")
        print('='*60)

        sample = dataset[sample_idx]
        grid = sample['grid']
        scenario = sample['scenario']

        sample_dir = output_dir / f"sample_{sample_idx:04d}"
        sample_dir.mkdir(exist_ok=True)

        visualize_sample(
            model=model,
            grid=grid,
            scenario=scenario,
            output_dir=str(sample_dir),
            sample_name=f"sample_{sample_idx:04d}",
            compute_counterfactual=compute_counterfactual,
            counterfactual_batch_size=counterfactual_batch_size,
            device=device,
            show_progress=True
        )

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print('='*60)
