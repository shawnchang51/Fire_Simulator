# 5-Channel Grid Encoding Changes Summary

## Overview
Successfully modified the ranking model to use 5-channel grid encoding with special padding values and a valid mask channel.

## Changes Made

### 1. Configuration Files

#### [ml/ranking/config.py](ml/ranking/config.py:46)
- Changed `num_grid_channels` from `4` to `5`
- Updated comment to include `valid_mask`

#### [ml/ranking/train_config.yaml](ml/ranking/train_config.yaml:18)
- Changed `num_grid_channels` from `4` to `5`
- Updated comment to include `valid_mask`

#### [ml/ranking/big_model.yaml](ml/ranking/big_model.yaml:18)
- Changed `num_grid_channels` from `4` to `5`
- Updated comment to include `valid_mask`

### 2. Dataset Encoding Logic

#### [ml/ranking/dataset.py](ml/ranking/dataset.py)

**PairwiseDataset._encode_grid()** (lines 178-227)
- Changed from 4-channel to 5-channel encoding
- Initialization: Changed from `np.zeros((4, ...))` to `np.full((5, ...), -1.0)`
- Padding value: Changed from `1.0` (walls) to `-1.0` (special value)
- Added Channel 4: Valid mask (1.0 for real grid, 0.0 for padding)
- Removed old padding-as-walls logic

**SingleConfigDataset._encode_grid()** (lines 373-407)
- Applied identical changes as PairwiseDataset
- Added H_copy/W_copy clipping logic for consistency

### 3. Documentation Updates

#### [ml/ranking/dataset.py](ml/ranking/dataset.py)
- Updated PairwiseDataset._encode_grid() docstring (lines 179-194)
  - Changed "4-channel" to "5-channel"
  - Added Channel 4 description
  - Updated return shape from `(4, H, W)` to `(5, H, W)`

- Updated SingleConfigDataset._encode_grid() docstring (line 374)
  - Changed "4-channel" to "5-channel"

#### [ml/ranking/model.py](ml/ranking/model.py:29)
- Updated FloorPlanEncoder docstring
  - Changed input shape from `(B, 4, 96, 128)` to `(B, 5, 96, 128)`
  - Updated channel description to include `valid_mask`

### 4. Model Architecture
- **No code changes required** - automatically adapts via `config.num_grid_channels`
- First Conv2d layer now accepts 5 input channels instead of 4
- Total parameters: **157,849** (increased from ~157,705 due to extra input channel)

## New Channel Encoding

| Channel | Description | Real Grid Area | Padding Area |
|---------|-------------|---------------|--------------|
| 0 | Wall mask | 0.0 or 1.0 (grid == -2) | -1.0 |
| 1 | Passable mask | 0.0 or 1.0 (grid == 0) | -1.0 |
| 2 | Door positions | 0.0 or 1.0 (has door) | -1.0 |
| 3 | Exit positions | 0.0 or 1.0 (has exit) | -1.0 |
| 4 | Valid mask | **1.0** | **0.0** |

## Validation Results

All validations passed successfully:

### Grid Encoding Tests ✓
- [x] Tensor shape is `(5, 96, 128)`
- [x] Channel 4 (valid mask) only contains `{0.0, 1.0}`
- [x] Padding areas in channels 0-3 contain `-1.0`
- [x] Valid mask: real grid area = 1.0, padding = 0.0
- [x] Wall and passable channels encoded correctly
- [x] Doors and exits placed correctly

### Model Architecture Tests ✓
- [x] Model instantiates without errors
- [x] Forward pass succeeds with 5-channel input
- [x] Output shapes are correct
  - Score: `(batch_size,)`
  - Latent A: `(batch_size,)`
  - Latent B: `(batch_size,)`

## Important Notes

### ⚠️ Checkpoint Compatibility
**Existing checkpoints are NOT compatible** with the new 5-channel architecture. The first Conv2d layer's input changed from 4 to 5 channels, making weight shapes incompatible.

**Required Action**: Train from scratch with the new architecture.

### Edge Cases Handled
1. **Grid exactly matches target size (96×128)**: No padding, valid mask all 1.0
2. **Grid smaller than target**: Padding on right/bottom with -1.0, valid mask 0.0 in padding areas
3. **Grid larger than target**: Clips to (96×128), no padding, valid mask all 1.0

## Next Steps

1. **Backup old checkpoints** (if any exist)
2. **Start new training** with updated configuration:
   ```bash
   python -m ml.ranking.run_training --mode train --config ml/ranking/train_config.yaml
   ```
3. **Monitor training** for any issues
4. **Verify gradients** on Channel 4 to ensure the model uses the valid mask

## Testing

Run the validation script to verify the changes:
```bash
python ml/ranking/validate_5channel.py
```

All tests should pass with "[OK]" status.

## Files Modified

1. `ml/ranking/config.py`
2. `ml/ranking/train_config.yaml`
3. `ml/ranking/big_model.yaml`
4. `ml/ranking/dataset.py`
5. `ml/ranking/model.py` (docstrings only)

## Files Created

1. `ml/ranking/validate_5channel.py` - Validation script
2. `ml/ranking/CHANGES_5CHANNEL.md` - This summary document
