"""Quick GradCAM visualization without needing simulation_results.jsonl"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import CrossAttentionRanker
from .train import load_checkpoint
from .visualize import GradCAM


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

    # Create 5-channel encoding with -1.0 padding
    encoded = np.full((5, tH, tW), -1.0, dtype=np.float32)

    # Handle cases where grid is larger than target size - clip
    H_copy = min(H, tH)
    W_copy = min(W, tW)

    # Channel 0: Wall mask
    encoded[0, :H_copy, :W_copy] = (grid_2d[:H_copy, :W_copy] == -2).astype(np.float32)

    # Channel 1: Passable mask
    encoded[1, :H_copy, :W_copy] = (grid_2d[:H_copy, :W_copy] == 0).astype(np.float32)

    # Channels 2 & 3: Door/Exit positions (empty for this viz)
    encoded[2, :H_copy, :W_copy] = 0.0
    encoded[3, :H_copy, :W_copy] = 0.0

    # Channel 4: Valid mask
    encoded[4, :H_copy, :W_copy] = 1.0
    encoded[4, H_copy:, :] = 0.0
    encoded[4, :, W_copy:] = 0.0

    return encoded


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
    target_size = tuple(config.get('target_grid_size', (96, 128)))

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
    grid_2d = data['grid']  # Shape: (H, W)

    # Encode to 5 channels
    encoded = encode_grid_5ch(grid_2d, target_size)
    grid_tensor = torch.from_numpy(encoded).to(device)  # (5, H, W)

    # Create dummy scenario (zeros, normalized)
    scenario_dim = config.get('scenario_input_dim', 5)
    scenario = torch.zeros(scenario_dim, device=device)

    # Generate GradCAM
    print("Generating GradCAM...")
    gradcam = GradCAM(model)
    cam = gradcam.generate(grid_tensor, scenario)

    # Plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Floor plan - use passable mask (channel 1)
    floor_plan_np = grid_tensor[1].cpu().numpy()

    ax = axes[0]
    ax.imshow(floor_plan_np, cmap='gray')
    ax.set_title('Floor Plan (Passable Areas)')
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
