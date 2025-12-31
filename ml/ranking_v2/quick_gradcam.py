"""Quick GradCAM visualization without needing simulation_results.jsonl"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import CrossAttentionRanker
from .train import load_checkpoint
from .visualize import GradCAM


def main():
    parser = argparse.ArgumentParser(description="Quick GradCAM visualization")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--floor-plan', type=str, default=None, help="Path to floor plan NPZ (optional, will find one)")
    parser.add_argument('--floor-plans-dir', type=str, default='combined_721/floor_plans', help="Directory with floor plans")
    parser.add_argument('--output', type=str, default='gradcam_output.png', help="Output path")
    parser.add_argument('--device', type=str, default=None, help="Device")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    config = checkpoint['config']
    target_size = config.get('target_grid_size', (96, 128))

    # Find a floor plan
    if args.floor_plan:
        npz_path = Path(args.floor_plan)
    else:
        floor_plans_dir = Path(args.floor_plans_dir)
        npz_files = list(floor_plans_dir.glob("*.npz"))
        if not npz_files:
            print(f"No NPZ files found in {floor_plans_dir}")
            return
        npz_path = npz_files[0]

    print(f"Loading floor plan from {npz_path}...")

    # Load and process floor plan
    data = np.load(npz_path)
    grid = data['grid']  # Shape: (C, H, W) or (H, W)

    if grid.ndim == 2:
        # Single channel, expand
        grid = np.expand_dims(grid, 0)

    # Resize if needed
    from torch.nn.functional import interpolate
    grid_tensor = torch.from_numpy(grid).float().unsqueeze(0)  # (1, C, H, W)
    if grid_tensor.shape[-2:] != target_size:
        grid_tensor = interpolate(grid_tensor, size=target_size, mode='bilinear', align_corners=False)

    grid_tensor = grid_tensor.to(device)

    # Create dummy scenario (zeros, normalized)
    scenario_dim = config.get('scenario_input_dim', 5)
    scenario = torch.zeros(1, scenario_dim, device=device)

    # Generate GradCAM
    print("Generating GradCAM...")
    gradcam = GradCAM(model)
    cam = gradcam.generate(grid_tensor.squeeze(0), scenario.squeeze(0))

    # Plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Floor plan (first channel or passable areas channel)
    floor_plan_np = grid_tensor.squeeze(0)[min(1, grid_tensor.shape[1]-1)].cpu().numpy()

    ax = axes[0]
    ax.imshow(floor_plan_np, cmap='gray')
    ax.set_title('Floor Plan')
    ax.axis('off')

    ax = axes[1]
    ax.imshow(cam, cmap='jet')
    ax.set_title('GradCAM Attention')
    ax.axis('off')

    ax = axes[2]
    ax.imshow(floor_plan_np, cmap='gray')
    ax.imshow(cam, cmap='jet', alpha=0.5)
    ax.set_title('GradCAM Overlay')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved GradCAM to {args.output}")
    plt.close()


if __name__ == '__main__':
    main()
