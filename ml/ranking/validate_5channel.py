"""
Validation script to verify 5-channel grid encoding changes.

This script validates that:
1. Grid tensors have the correct shape (5, 96, 128)
2. Channel 4 (valid mask) only contains {0.0, 1.0}
3. Padding areas in channels 0-3 contain -1.0
4. Model can be instantiated and accepts 5-channel input
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.ranking.config import RankingConfig
from ml.ranking.dataset import PairwiseDataset, SingleConfigDataset
from ml.ranking.model import SiameseRanker


def validate_grid_encoding():
    """Validate grid encoding produces correct 5-channel tensors."""
    print("=" * 60)
    print("Validating Grid Encoding")
    print("=" * 60)

    # Load config
    config = RankingConfig()
    print(f"[OK] Loaded config: num_grid_channels = {config.num_grid_channels}")
    assert config.num_grid_channels == 5, f"Expected 5 channels, got {config.num_grid_channels}"

    # Create dummy floor plan data
    floor_plans = {
        0: {
            'grid': np.array([[-2, -2, -2, 0, 0],
                             [-2, 0, 0, 0, 0],
                             [-2, 0, 0, 0, -2],
                             [0, 0, 0, -2, -2]], dtype=np.int32),
            'floor_plan_id': 0
        }
    }

    # Create small PairwiseDataset instance for testing
    dataset = PairwiseDataset.__new__(PairwiseDataset)
    dataset.floor_plans = floor_plans
    dataset.target_size = (96, 128)

    # Test encoding
    config_dict = {
        'door_config': [
            {'position': 'x2y1', 'type': 'door'},
            {'position': 'x4y3', 'type': 'exit'}
        ]
    }

    encoded = dataset._encode_grid(0, config_dict)

    # Validate shape
    print(f"\n1. Shape validation:")
    print(f"   Expected: (5, 96, 128)")
    print(f"   Actual:   {tuple(encoded.shape)}")
    assert encoded.shape == (5, 96, 128), f"Wrong shape: {encoded.shape}"
    print("   [OK]Shape is correct")

    # Validate Channel 4 (valid mask)
    print(f"\n2. Valid mask channel (Channel 4) validation:")
    unique_vals = torch.unique(encoded[4]).tolist()
    print(f"   Unique values: {unique_vals}")
    assert set(unique_vals).issubset({0.0, 1.0}), f"Channel 4 should only have {{0.0, 1.0}}, got {unique_vals}"
    print("   [OK]Valid mask contains only 0.0 and 1.0")

    # Check valid mask structure (top-left 4x5 should be 1.0, rest should be 0.0)
    valid_region = encoded[4, :4, :5]
    padding_region = encoded[4, 4:, :]  # All rows after the grid
    print(f"   Valid region (top-left 4x5) all 1.0: {torch.all(valid_region == 1.0).item()}")
    print(f"   Padding region all 0.0: {torch.all(padding_region == 0.0).item()}")
    assert torch.all(valid_region == 1.0), "Valid region should be all 1.0"
    assert torch.all(padding_region == 0.0), "Padding region should be all 0.0"
    print("   [OK]Valid mask structure is correct")

    # Validate padding in channels 0-3
    print(f"\n3. Padding value validation (channels 0-3):")
    for ch in range(4):
        padding_vals = torch.unique(encoded[ch, 4:, :]).tolist()  # Check padding rows
        print(f"   Channel {ch} padding values: {padding_vals}")
        assert -1.0 in padding_vals, f"Channel {ch} padding should contain -1.0"
    print("   [OK]Padding areas contain -1.0")

    # Validate wall and passable channels
    print(f"\n4. Wall and passable channel validation:")
    # Check a known wall position (0, 0) should be 1.0 in channel 0
    assert encoded[0, 0, 0] == 1.0, "Position (0,0) should be a wall"
    # Check a known passable position (1, 2) should be 1.0 in channel 1
    assert encoded[1, 1, 2] == 1.0, "Position (1,2) should be passable"
    print("   [OK] Wall and passable channels encoded correctly")

    # Validate door/exit placement
    print(f"\n5. Door/exit placement validation:")
    # Door at x2y1 should be 1.0 in channel 2
    assert encoded[2, 1, 2] == 1.0, "Door at (1,2) should be marked in channel 2"
    # Exit at x4y3 should be 1.0 in channel 3
    assert encoded[3, 3, 4] == 1.0, "Exit at (3,4) should be marked in channel 3"
    print("   [OK] Doors and exits placed correctly")

    print("\n" + "=" * 60)
    print("[OK] All grid encoding validations passed!")
    print("=" * 60)


def validate_model():
    """Validate model can be instantiated and accepts 5-channel input."""
    print("\n" + "=" * 60)
    print("Validating Model Architecture")
    print("=" * 60)

    # Load config
    config = RankingConfig()

    # Instantiate model
    print(f"\n1. Instantiating SiameseRanker model...")
    model = SiameseRanker(config)
    print(f"   [OK] Model instantiated successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Test forward pass
    print(f"\n2. Testing forward pass with 5-channel input...")
    batch_size = 4
    grid_a = torch.randn(batch_size, 5, 96, 128)
    grid_b = torch.randn(batch_size, 5, 96, 128)
    scenario_a = torch.randn(batch_size, 4)
    scenario_b = torch.randn(batch_size, 4)

    try:
        with torch.no_grad():
            output = model(grid_a, scenario_a, grid_b, scenario_b)
        print(f"   [OK] Forward pass successful")

        # Handle tuple output (score, latent_a, latent_b)
        if isinstance(output, tuple):
            score, latent_a, latent_b = output
            print(f"   Score shape: {score.shape}")
            print(f"   Latent A shape: {latent_a.shape}")
            print(f"   Latent B shape: {latent_b.shape}")
            assert score.shape == (batch_size,), f"Expected score shape ({batch_size},), got {score.shape}"
        else:
            print(f"   Output shape: {output.shape}")
            assert output.shape == (batch_size,), f"Expected output shape ({batch_size},), got {output.shape}"
        print(f"   [OK] Output shape is correct")
    except Exception as e:
        print(f"   [FAIL] Forward pass failed: {e}")
        raise

    print("\n" + "=" * 60)
    print("[OK] All model validations passed!")
    print("=" * 60)


def main():
    """Run all validations."""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  5-Channel Grid Encoding Validation".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print()

    try:
        validate_grid_encoding()
        validate_model()

        print("\n" + "*" * 60)
        print("*" + " " * 58 + "*")
        print("*" + "  ALL VALIDATIONS PASSED! [OK]".center(58) + "*")
        print("*" + " " * 58 + "*")
        print("*" * 60)
        print()
        print("The 5-channel encoding changes are working correctly.")
        print("You can now proceed with training the model.")
        print()

    except AssertionError as e:
        print(f"\n[FAIL] Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
