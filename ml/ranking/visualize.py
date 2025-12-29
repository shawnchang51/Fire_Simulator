"""
Visualization Tools for Pairwise Ranking Model

Key Visualizations:
    - Grad-CAM: Attention heatmaps on floor plan grids
    - Latent PCA: 2D projection of latent space colored by score
    - Latent-Metric Correlation: Heatmap of latent dims vs metrics
    - Perturbation Sensitivity: Score vs latent dimension perturbation
    - Training Curves: Loss, AUC, LR over epochs

Important:
    - Interpretability analysis is done on PointwiseScorer (single config → score)
    - NOT on the pairwise subtraction (provides cleaner attribution)
    - Uses named `final_conv` layer for Grad-CAM robustness

PyTorch Version Note:
    - register_full_backward_hook (PyTorch >= 1.9) preferred
    - Handles gradient accumulation correctly
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model import SiameseRanker


class GradCAM:
    """
    Grad-CAM visualization for understanding CNN attention.

    Uses NAMED layer reference (`final_conv`) for robustness.
    All interpretability is on the PointwiseScorer, not pairwise comparison.

    Reference:
        Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks" (2017)
    """

    def __init__(
        self,
        model: SiameseRanker,
        target_layer_name: str = 'final_conv'
    ):
        """
        Initialize Grad-CAM.

        Args:
            model: SiameseRanker or PointwiseScorer
            target_layer_name: Name of the final conv layer in encoder
        """
        self.model = model
        self.gradients = None
        self.activations = None

        # Get encoder (handle both SiameseRanker and PointwiseScorer)
        if hasattr(model, 'scorer'):
            encoder = model.scorer.encoder
        else:
            encoder = model.encoder

        # Get named layer (robust to architecture changes)
        target_layer = getattr(encoder, target_layer_name)

        # Register hooks (use register_full_backward_hook for PyTorch >= 1.9)
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook to save gradients."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        grid: torch.Tensor,
        scenario: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single input.

        Args:
            grid: Grid tensor of shape (4, H, W) or (1, 4, H, W)
            scenario: Scenario tensor of shape (4,) or (1, 4)
            target_size: Optional output size (H, W)

        Returns:
            Heatmap of shape (H, W) with values in [0, 1]
        """
        self.model.eval()

        # Add batch dimension if needed
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        if scenario.dim() == 1:
            scenario = scenario.unsqueeze(0)

        # Ensure requires_grad for backward
        grid = grid.clone().requires_grad_(True)

        # Forward pass
        score = self.model.score_single(grid, scenario)

        # Backward pass
        self.model.zero_grad()
        score.backward()

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
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def collect_latent_vectors(
    model: SiameseRanker,
    dataset,
    n_samples: int = 1000,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect latent vectors and scores for analysis.

    Args:
        model: Trained ranking model
        dataset: Dataset with grid, scenario, ground_truth_score
        n_samples: Maximum samples to collect
        device: Device to use

    Returns:
        Tuple of (latent_vectors (N, K), gt_scores (N,), predicted_scores (N,))
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    latents = []
    gt_scores = []
    pred_scores = []

    n_samples = min(n_samples, len(dataset))

    with torch.no_grad():
        for i in range(n_samples):
            item = dataset[i]
            grid = item['grid'].unsqueeze(0).to(device)
            scenario = item['scenario'].unsqueeze(0).to(device)

            latent = model.get_latent(grid)
            score = model.score_config(grid, scenario)
            
            latents.append(latent.cpu().numpy().squeeze())
            gt_scores.append(item['ground_truth_score'])
            pred_scores.append(score.cpu().item())

    return np.stack(latents), np.array(gt_scores), np.array(pred_scores)


def plot_latent_pca(
    latents: np.ndarray,
    scores: np.ndarray,
    predicted_scores: Optional[np.ndarray] = None,
    output_path: Optional[str] = None
):
    """
    Create PCA visualization of latent space colored by score.

    Args:
        latents: Latent vectors of shape (N, K)
        scores: Ground truth scores of shape (N,)
        predicted_scores: Optional predicted scores for comparison
        output_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        print("matplotlib and sklearn required for visualization")
        return

    # PCA to 2D
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    # Create figure with 1 or 2 subplots
    n_plots = 2 if predicted_scores is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Plot 1: Colored by ground truth score
    ax = axes[0]
    scatter = ax.scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=scores,
        cmap='RdYlGn',  # Red=bad, Yellow=mid, Green=good
        alpha=0.7,
        s=25,
        edgecolors='none'
    )
    plt.colorbar(scatter, ax=ax, label='Ground Truth Score')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('Latent Space (Ground Truth)')
    ax.grid(True, alpha=0.3)

    # Plot 2: Colored by predicted score (if available)
    if predicted_scores is not None:
        ax = axes[1]
        scatter = ax.scatter(
            latents_2d[:, 0],
            latents_2d[:, 1],
            c=predicted_scores,
            cmap='RdYlGn',
            alpha=0.7,
            s=25,
            edgecolors='none'
        )
        plt.colorbar(scatter, ax=ax, label='Predicted Score')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax.set_title('Latent Space (Predicted)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved PCA plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_latent_metric_correlation(
    model: SiameseRanker,
    dataset,
    n_samples: int = 1000,
    output_path: Optional[str] = None,
    device: torch.device = None
):
    """
    Create correlation heatmap between latent dimensions and metrics.

    Shows how each latent dimension correlates with:
    - survival_rate
    - steps (normalized)
    - avg_fire_damage
    - ground_truth_score
    - predicted_score

    Args:
        model: Trained ranking model
        dataset: Dataset with metric information
        n_samples: Maximum samples to use
        output_path: Optional path to save figure
        device: Device to use
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import spearmanr
    except ImportError:
        print("matplotlib and scipy required for visualization")
        return

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Collect latents and metrics
    latents = []
    survival_rates = []
    steps_list = []
    fire_damages = []
    gt_scores = []
    pred_scores = []

    n_samples = min(n_samples, len(dataset))

    with torch.no_grad():
        for i in range(n_samples):
            item = dataset[i]
            grid = item['grid'].unsqueeze(0).to(device)
            scenario = item['scenario'].unsqueeze(0).to(device)

            latent = model.get_latent(grid)
            score = model.score_config(grid, scenario)
            
            latents.append(latent.cpu().numpy().squeeze())
            pred_scores.append(score.cpu().item())

            # Get actual metrics from dataset
            survival_rates.append(item.get('survival_rate', item['ground_truth_score']))
            steps_list.append(item.get('steps', 50))
            fire_damages.append(item.get('avg_fire_damage', 0.5))
            gt_scores.append(item['ground_truth_score'])

    latents = np.stack(latents)  # (N, K)
    survival_rates = np.array(survival_rates)
    steps = np.array(steps_list)
    fire_damages = np.array(fire_damages)
    gt_scores = np.array(gt_scores)
    pred_scores = np.array(pred_scores)

    # Compute correlations
    K = latents.shape[1]
    metrics = ['Survival\nRate', 'Steps', 'Fire\nDamage', 'GT\nScore', 'Predicted\nScore']
    metric_values = [survival_rates, steps, fire_damages, gt_scores, pred_scores]

    correlations = np.zeros((K, len(metrics)))

    for i in range(K):
        for j, values in enumerate(metric_values):
            # Check for constant values
            if np.std(latents[:, i]) < 1e-8 or np.std(values) < 1e-8:
                correlations[i, j] = 0.0
            else:
                rho, _ = spearmanr(latents[:, i], values)
                correlations[i, j] = rho if not np.isnan(rho) else 0.0

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, K * 0.6)))

    im = ax.imshow(correlations, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Labels
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f'Latent {i}' for i in range(K)], fontsize=10)
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Latent Dimension', fontsize=11)
    ax.set_title('Latent-Metric Correlation (Spearman ρ)', fontsize=12)

    # Add correlation values as text
    for i in range(K):
        for j in range(len(metrics)):
            color = 'white' if abs(correlations[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{correlations[i, j]:.2f}',
                    ha='center', va='center', color=color, fontsize=9,
                    fontweight='bold' if abs(correlations[i, j]) > 0.3 else 'normal')

    plt.colorbar(im, ax=ax, label='Spearman Correlation', shrink=0.8)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved correlation heatmap to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_perturbation_sensitivity(
    model: SiameseRanker,
    sample: Dict,
    dimension_idx: int,
    value_range: Tuple[float, float] = (-2.0, 2.0),
    num_points: int = 21,
    output_path: Optional[str] = None,
    device: torch.device = None
):
    """
    Plot score sensitivity to perturbation of a single latent dimension.

    Args:
        model: Trained ranking model
        sample: Sample dict with 'grid' and 'scenario'
        dimension_idx: Which latent dimension to perturb
        value_range: (min, max) range of perturbation values
        num_points: Number of points to evaluate
        output_path: Optional path to save figure
        device: Device to use
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    grid = sample['grid'].unsqueeze(0).to(device)
    scenario = sample['scenario'].unsqueeze(0).to(device)

    # Get base latent
    with torch.no_grad():
        base_latent = model.get_latent(grid)
        scenario_feat = model.scorer.scenario_encoder(scenario)

    # Perturb and score
    values = np.linspace(value_range[0], value_range[1], num_points)
    scores = []

    with torch.no_grad():
        for v in values:
            perturbed = base_latent.clone()
            perturbed[0, dimension_idx] = v

            # Manually compute score from perturbed latent
            features = torch.cat([perturbed, scenario_feat], dim=1)
            score = model.scorer.scoring_head(features).item()
            scores.append(score)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(values, scores, 'b-', linewidth=2)
    ax.axvline(x=base_latent[0, dimension_idx].item(), color='r',
               linestyle='--', label='Original value')
    ax.set_xlabel(f'Latent Dimension {dimension_idx} Value')
    ax.set_ylabel('Predicted Score')
    ax.set_title(f'Score Sensitivity to Latent[{dimension_idx}] Perturbation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved perturbation plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None
):
    """
    Plot training history curves.

    Args:
        history: Training history dict from train_ranking_model()
        output_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_gradcam_sample(
    model: SiameseRanker,
    sample: Dict,
    output_path: Optional[str] = None,
    device: torch.device = None
):
    """
    Generate and display Grad-CAM visualization for a sample.

    Args:
        model: Trained ranking model
        sample: Sample dict with 'grid' and 'scenario'
        output_path: Optional path to save figure
        device: Device to use
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    if device is None:
        device = next(model.parameters()).device

    # Generate Grad-CAM
    gradcam = GradCAM(model)
    grid = sample['grid'].to(device)
    scenario = sample['scenario'].to(device)

    cam = gradcam.generate(grid, scenario)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original floor plan (passable areas)
    ax = axes[0]
    floor_plan = grid[1].cpu().numpy()  # Passable channel
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


def visualize_multiple_gradcam(
    model: SiameseRanker,
    dataset,
    plan_indices: List[int],
    output_path: Optional[str] = None,
    device: torch.device = None,
    cols: int = 3,
    save_individual: bool = False,
    output_dir: Optional[str] = None
):
    """
    Generate Grad-CAM visualization for multiple plans.

    Args:
        model: Trained ranking model
        dataset: Dataset containing samples
        plan_indices: List of plan indices to visualize (e.g., [0, 5, 10, 15])
        output_path: Optional path to save combined figure (ignored if save_individual=True)
        device: Device to use
        cols: Number of columns in the grid layout (for combined figure)
        save_individual: If True, save each plan as a separate image file
        output_dir: Directory for individual images (required if save_individual=True)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    if device is None:
        device = next(model.parameters()).device

    # Validate indices
    valid_indices = [i for i in plan_indices if 0 <= i < len(dataset)]
    if len(valid_indices) != len(plan_indices):
        invalid = set(plan_indices) - set(valid_indices)
        print(f"Warning: Invalid indices ignored: {invalid}")
    
    if not valid_indices:
        print("No valid plan indices provided!")
        return

    gradcam = GradCAM(model)

    # === Save as individual images ===
    if save_individual:
        if output_dir is None:
            output_dir = "gradcam_outputs"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for plan_idx in valid_indices:
            sample = dataset[plan_idx]
            grid = sample['grid'].to(device)
            scenario = sample['scenario'].to(device)

            # Generate Grad-CAM
            cam = gradcam.generate(grid, scenario)

            # Get score
            with torch.no_grad():
                score = model.score_single(
                    grid.unsqueeze(0) if grid.dim() == 3 else grid,
                    scenario.unsqueeze(0) if scenario.dim() == 1 else scenario
                ).item()

            # Floor plan
            floor_plan = grid[1].cpu().numpy()

            # Create individual figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Floor plan
            axes[0].imshow(floor_plan, cmap='gray')
            axes[0].set_title(f'Floor Plan', fontsize=12)
            axes[0].axis('off')

            # Grad-CAM heatmap
            axes[1].imshow(cam, cmap='jet')
            axes[1].set_title('Grad-CAM Attention', fontsize=12)
            axes[1].axis('off')

            # Overlay
            axes[2].imshow(floor_plan, cmap='gray')
            axes[2].imshow(cam, cmap='jet', alpha=0.5)
            axes[2].set_title('Overlay', fontsize=12)
            axes[2].axis('off')

            plt.suptitle(f'Plan {plan_idx} | Score: {score:.4f}', fontsize=14)
            plt.tight_layout()

            # Save individual file
            individual_path = output_dir / f"gradcam_plan_{plan_idx:05d}.png"
            plt.savefig(individual_path, dpi=150, bbox_inches='tight')
            plt.close()

        print(f"Saved {len(valid_indices)} individual Grad-CAM images to {output_dir}/")
        return

    # === Save as combined grid image ===
    n_plans = len(valid_indices)
    rows = (n_plans + cols - 1) // cols  # Ceiling division

    # Each plan gets 3 columns: floor plan, heatmap, overlay
    fig, axes = plt.subplots(rows, cols * 3, figsize=(5 * cols * 3, 5 * rows))
    
    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1)

    gradcam = GradCAM(model)

    for idx, plan_idx in enumerate(valid_indices):
        row = idx // cols
        col_base = (idx % cols) * 3

        sample = dataset[plan_idx]
        grid = sample['grid'].to(device)
        scenario = sample['scenario'].to(device)

        # Generate Grad-CAM
        cam = gradcam.generate(grid, scenario)

        # Get score
        with torch.no_grad():
            score = model.score_single(
                grid.unsqueeze(0) if grid.dim() == 3 else grid,
                scenario.unsqueeze(0) if scenario.dim() == 1 else scenario
            ).item()

        # Floor plan
        floor_plan = grid[1].cpu().numpy()  # Passable channel
        
        ax = axes[row, col_base]
        ax.imshow(floor_plan, cmap='gray')
        ax.set_title(f'Plan {plan_idx}\nScore: {score:.3f}', fontsize=10)
        ax.axis('off')

        # Grad-CAM heatmap
        ax = axes[row, col_base + 1]
        ax.imshow(cam, cmap='jet')
        ax.set_title('Grad-CAM', fontsize=10)
        ax.axis('off')

        # Overlay
        ax = axes[row, col_base + 2]
        ax.imshow(floor_plan, cmap='gray')
        ax.imshow(cam, cmap='jet', alpha=0.5)
        ax.set_title('Overlay', fontsize=10)
        ax.axis('off')

    # Hide unused axes
    total_slots = rows * cols
    for idx in range(n_plans, total_slots):
        row = idx // cols
        col_base = (idx % cols) * 3
        for offset in range(3):
            axes[row, col_base + offset].axis('off')

    plt.suptitle(f'Grad-CAM Visualization for {n_plans} Plans', fontsize=14)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-plan Grad-CAM to {output_path}")
    else:
        plt.show()

    plt.close()


def generate_all_visualizations(
    model: SiameseRanker,
    dataset,
    history: Dict,
    output_dir: str,
    n_samples: int = 500,
    device: torch.device = None,
    gradcam_indices: Optional[List[int]] = None
):
    """
    Generate all visualizations and save to output directory.

    Args:
        model: Trained ranking model
        dataset: Evaluation dataset
        history: Training history
        output_dir: Directory to save visualizations
        n_samples: Number of samples for latent analysis
        device: Device to use
        gradcam_indices: Optional list of plan indices for Grad-CAM.
                        If None, uses first sample only.
                        Example: [0, 10, 20, 50] for 4 specific plans
    """
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
        latents, gt_scores, pred_scores = collect_latent_vectors(model, dataset, n_samples, device)
        plot_latent_pca(
            latents, gt_scores, pred_scores,
            output_path=str(output_dir / "latent_pca.png")
        )
    except Exception as e:
        print(f"Failed to generate PCA plot: {e}")

    # Latent-metric correlation
    try:
        plot_latent_metric_correlation(
            model, dataset, n_samples,
            output_path=str(output_dir / "latent_correlation.png"),
            device=device
        )
    except Exception as e:
        print(f"Failed to generate correlation heatmap: {e}")

    # Grad-CAM visualization
    try:
        if gradcam_indices is not None and len(gradcam_indices) > 1:
            # Multiple plans
            visualize_multiple_gradcam(
                model, dataset, gradcam_indices,
                output_path=str(output_dir / "gradcam_multiple.png"),
                device=device
            )
        else:
            # Single plan (default: first sample, or first in gradcam_indices)
            sample_idx = gradcam_indices[0] if gradcam_indices else 0
            sample = dataset[sample_idx]
            visualize_gradcam_sample(
                model, sample,
                output_path=str(output_dir / f"gradcam_sample_{sample_idx}.png"),
                device=device
            )
    except Exception as e:
        print(f"Failed to generate Grad-CAM: {e}")

    print("Visualization generation complete!")
