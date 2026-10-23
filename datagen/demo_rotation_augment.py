"""
Demo script to visualize rotation augmentation effects on floor plans.
Shows all 4 rotation variants (0°, 90°, 180°, 270°) and the valid mask.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Find a floor plan to demo
def find_floor_plan():
    """Find an available floor plan NPZ file."""
    possible_dirs = [
        "combined_fast/floor_plans",
        "training_data_v5/floor_plans",
        "server_results/floor_plans",
    ]

    for dir_path in possible_dirs:
        p = Path(dir_path)
        if p.exists():
            npz_files = list(p.glob("plan_*.npz"))
            if npz_files:
                return npz_files[0], dir_path

    return None, None


def encode_grid(grid: np.ndarray, target_size=(96, 128)) -> torch.Tensor:
    """
    Create 5-channel tensor from grid (simplified version).

    Channels:
        0: Wall mask (grid == -2)
        1: Passable mask (grid == 0)
        2: Door positions (empty for demo)
        3: Exit positions (empty for demo)
        4: Valid mask (1.0 for real grid, 0.0 for padding)
    """
    H, W = grid.shape
    tH, tW = target_size

    # Create 5-channel encoding with -1.0 padding
    encoded = np.full((5, tH, tW), -1.0, dtype=np.float32)

    # Handle cases where grid is larger than target size
    H_copy = min(H, tH)
    W_copy = min(W, tW)

    # Channel 0: Wall mask
    encoded[0, :H_copy, :W_copy] = (grid[:H_copy, :W_copy] == -2).astype(np.float32)

    # Channel 1: Passable mask
    encoded[1, :H_copy, :W_copy] = (grid[:H_copy, :W_copy] == 0).astype(np.float32)

    # Channels 2 & 3: Doors and exits (leave as 0 for demo)
    encoded[2, :H_copy, :W_copy] = 0.0
    encoded[3, :H_copy, :W_copy] = 0.0

    # Channel 4: Valid mask
    encoded[4, :H_copy, :W_copy] = 1.0
    encoded[4, H_copy:, :] = 0.0
    encoded[4, :, W_copy:] = 0.0

    return torch.from_numpy(encoded)


def rotate_grid(grid: torch.Tensor, k: int, target_size=(96, 128)) -> torch.Tensor:
    """
    Rotate grid by k*90 degrees and re-pad to target size.

    Always repositions content to top-left after rotation.

    Args:
        grid: Grid tensor of shape (5, H, W)
        k: Number of 90-degree rotations (0, 1, 2, 3)
        target_size: (tH, tW) target dimensions

    Returns:
        Rotated and re-padded grid
    """
    if k == 0:
        return grid.clone()

    tH, tW = target_size

    # Apply rotation
    rotated = torch.rot90(grid, k=k, dims=(1, 2))

    # Always reposition content to top-left (needed for all rotations including 180°)
    result = torch.full((5, tH, tW), -1.0, dtype=grid.dtype)
    result[4] = 0.0  # Valid mask padding is 0.0

    # Find the valid content region in rotated grid
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

    # Crop content if it doesn't fit in target size
    copy_h = min(content_h, tH)
    copy_w = min(content_w, tW)

    # Copy valid content to top-left of result
    result[:, :copy_h, :copy_w] = rotated[
        :,
        min_row:min_row + copy_h,
        min_col:min_col + copy_w
    ]

    return result


def visualize_rotations(grid_tensor: torch.Tensor, plan_name: str):
    """
    Visualize all rotation variants showing walls, passable areas, and valid mask.
    """
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    rotation_labels = ["0° (original)", "90° CCW", "180°", "270° CCW"]
    channel_labels = ["Walls (ch0)", "Passable (ch1)", "Doors+Exits (ch2+3)", "Valid Mask (ch4)"]

    for rot_idx in range(4):
        # Get rotated grid
        rotated = rotate_grid(grid_tensor, k=rot_idx)

        for ch_idx in range(4):
            ax = axes[rot_idx, ch_idx]

            if ch_idx == 2:
                # Combine doors and exits for visualization
                data = rotated[2].numpy() + rotated[3].numpy()
                cmap = 'Greens'
            elif ch_idx == 3:
                # Valid mask (channel 4)
                data = rotated[4].numpy()
                cmap = 'Blues'
            else:
                data = rotated[ch_idx].numpy()
                cmap = 'gray_r' if ch_idx == 0 else 'YlGn'

            im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1)

            if rot_idx == 0:
                ax.set_title(channel_labels[ch_idx], fontsize=12, fontweight='bold')
            if ch_idx == 0:
                ax.set_ylabel(rotation_labels[rot_idx], fontsize=12, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])

            # Show dimensions
            h, w = data.shape
            ax.text(0.02, 0.98, f"{h}x{w}", transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', color='red')

    plt.suptitle(f"Rotation Augmentation Demo: {plan_name}\n"
                 f"Values: -1.0 = padding (dark), 0.0 = false/empty, 1.0 = true/valid",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("rotation_demo.png", dpi=150, bbox_inches='tight')
    print("Saved: rotation_demo.png")
    plt.show()


def shift_grid(grid: torch.Tensor, shift_down: int, shift_right: int) -> torch.Tensor:
    """
    Shift grid content by specified amounts.

    Args:
        grid: Grid tensor of shape (5, H, W)
        shift_down: Pixels to shift down
        shift_right: Pixels to shift right

    Returns:
        Shifted grid tensor
    """
    if shift_down == 0 and shift_right == 0:
        return grid.clone()

    _, tH, tW = grid.shape

    # Create new grid with padding values
    shifted = torch.full_like(grid, -1.0)
    shifted[4] = 0.0  # Valid mask padding is 0.0

    # Copy content to new position
    src_h = tH - shift_down
    src_w = tW - shift_right
    shifted[:, shift_down:, shift_right:] = grid[:, :src_h, :src_w]

    return shifted


def visualize_composite(grid_tensor: torch.Tensor, plan_name: str):
    """
    Visualize composite floor plan view for each rotation.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    rotation_labels = ["0° (original)", "90° CCW", "180°", "270° CCW"]

    for rot_idx in range(4):
        rotated = rotate_grid(grid_tensor, k=rot_idx)

        # Top row: Composite view (walls + passable)
        ax_top = axes[0, rot_idx]

        # Create RGB composite: walls=black, passable=white, padding=gray
        walls = rotated[0].numpy()
        passable = rotated[1].numpy()
        valid = rotated[4].numpy()

        # RGB image
        composite = np.zeros((*walls.shape, 3))
        composite[valid > 0.5] = [0.9, 0.9, 0.9]  # Valid area background (light gray)
        composite[passable > 0.5] = [1.0, 1.0, 1.0]  # Passable (white)
        composite[walls > 0.5] = [0.2, 0.2, 0.2]  # Walls (dark gray)
        composite[valid < 0.5] = [0.5, 0.5, 0.7]  # Padding (blue-gray)

        ax_top.imshow(composite)
        ax_top.set_title(rotation_labels[rot_idx], fontsize=12, fontweight='bold')
        ax_top.set_xticks([])
        ax_top.set_yticks([])

        # Bottom row: Valid mask only
        ax_bottom = axes[1, rot_idx]
        im = ax_bottom.imshow(valid, cmap='Blues', vmin=0, vmax=1)
        ax_bottom.set_title("Valid Mask", fontsize=10)
        ax_bottom.set_xticks([])
        ax_bottom.set_yticks([])

        # Show content dimensions
        valid_h = (valid > 0.5).any(axis=1).sum()
        valid_w = (valid > 0.5).any(axis=0).sum()
        ax_bottom.text(0.5, -0.1, f"Content: {valid_h}x{valid_w}",
                      transform=ax_bottom.transAxes, ha='center', fontsize=9)

    plt.suptitle(f"Floor Plan Rotation Demo: {plan_name}\n"
                 f"Top: Composite (white=passable, dark=walls, blue-gray=padding) | "
                 f"Bottom: Valid mask",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("rotation_composite.png", dpi=150, bbox_inches='tight')
    print("Saved: rotation_composite.png")
    plt.show()


def visualize_combined_augmentation(grid_tensor: torch.Tensor, plan_name: str):
    """
    Visualize combined rotation + shift augmentation (what training actually sees).
    """
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    rotation_labels = ["0°", "90°", "180°", "270°"]
    shift_amounts = [(0, 0), (15, 20), (30, 40), (45, 60)]  # (down, right)

    for rot_idx in range(4):
        for shift_idx in range(4):
            ax = axes[rot_idx, shift_idx]

            # Apply rotation first
            rotated = rotate_grid(grid_tensor, k=rot_idx)

            # Then apply shift
            shift_down, shift_right = shift_amounts[shift_idx]
            shifted = shift_grid(rotated, shift_down, shift_right)

            # Create composite view
            walls = shifted[0].numpy()
            passable = shifted[1].numpy()
            valid = shifted[4].numpy()

            composite = np.zeros((*walls.shape, 3))
            composite[valid > 0.5] = [0.9, 0.9, 0.9]
            composite[passable > 0.5] = [1.0, 1.0, 1.0]
            composite[walls > 0.5] = [0.2, 0.2, 0.2]
            composite[valid < 0.5] = [0.5, 0.5, 0.7]

            ax.imshow(composite)
            ax.set_xticks([])
            ax.set_yticks([])

            # Labels
            if rot_idx == 0:
                ax.set_title(f"Shift: +{shift_down}↓ +{shift_right}→", fontsize=10)
            if shift_idx == 0:
                ax.set_ylabel(f"Rotate: {rotation_labels[rot_idx]}", fontsize=10, fontweight='bold')

    plt.suptitle(f"Combined Rotation + Shift Augmentation: {plan_name}\n"
                 f"Rows = rotation angle | Columns = shift amount (down, right)\n"
                 f"Training randomly picks ONE rotation + ONE shift per sample",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("rotation_shift_combined.png", dpi=150, bbox_inches='tight')
    print("Saved: rotation_shift_combined.png")
    plt.show()


def main():
    print("=" * 60)
    print("Rotation Augmentation Demo")
    print("=" * 60)

    # Find a floor plan
    plan_file, plans_dir = find_floor_plan()

    if plan_file is None:
        print("No floor plan files found. Creating synthetic demo...")
        # Create a synthetic non-square floor plan for demo
        grid = np.full((50, 80), 0, dtype=np.int8)  # Passable
        grid[0, :] = -2  # Top wall
        grid[-1, :] = -2  # Bottom wall
        grid[:, 0] = -2  # Left wall
        grid[:, -1] = -2  # Right wall
        # Add some internal walls
        grid[10:40, 30] = -2
        grid[20, 30:60] = -2
        plan_name = "Synthetic 50x80"
    else:
        print(f"Loading: {plan_file}")
        data = np.load(plan_file, allow_pickle=True)
        grid = data['grid']
        plan_name = plan_file.name

    print(f"Original grid shape: {grid.shape}")
    print(f"Target size: (96, 128)")
    print()

    # Encode grid
    grid_tensor = encode_grid(grid, target_size=(96, 128))
    print(f"Encoded tensor shape: {grid_tensor.shape}")
    print(f"Channels: [0]=walls, [1]=passable, [2]=doors, [3]=exits, [4]=valid_mask")
    print()

    # Show rotation effects on dimensions
    print("Rotation dimension changes:")
    for k in range(4):
        rotated = rotate_grid(grid_tensor, k=k)
        valid = rotated[4].numpy()
        valid_h = (valid > 0.5).any(axis=1).sum()
        valid_w = (valid > 0.5).any(axis=0).sum()
        print(f"  {k*90:3d}°: tensor={tuple(rotated.shape)}, content={valid_h}x{valid_w}")
    print()

    # Visualize
    visualize_composite(grid_tensor, plan_name)
    visualize_combined_augmentation(grid_tensor, plan_name)
    visualize_rotations(grid_tensor, plan_name)

    print("\nDone!")


if __name__ == "__main__":
    main()
