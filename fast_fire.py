"""
Vectorized Fire Model for RL Training
=====================================

Simplified cellular automata fire spread using NumPy operations.
Removes complex physics (oxygen, temperature, smoke) for speed.

Performance: 5-10x faster than AdvancedFireModel
"""

import numpy as np
from typing import Tuple, Optional

class FastFireModel:
    """
    Vectorized fire spread using NumPy convolution.

    Simplifications vs AdvancedFireModel:
    - No oxygen/temperature/smoke tracking
    - No fuel depletion
    - Fixed spread probabilities
    - No wind effects (can be added if needed)
    """

    # Spread kernel - probability of igniting neighbors
    SPREAD_KERNEL = np.array([
        [0.05, 0.15, 0.05],
        [0.15, 0.00, 0.15],
        [0.05, 0.15, 0.05]
    ], dtype=np.float32)

    def __init__(self, grid: np.ndarray,
                 spread_rate: float = 0.3,
                 intensity_growth: float = 0.5,
                 max_intensity: float = 4.0):
        """
        Initialize fire model.

        Args:
            grid: 2D array with initial fire state (-2=wall, 0=empty, >0=fire)
            spread_rate: Probability multiplier for fire spread
            intensity_growth: How fast fire intensity grows per step
            max_intensity: Maximum fire intensity
        """
        self.grid = grid.astype(np.float32)
        self.rows, self.cols = grid.shape
        self.spread_rate = spread_rate
        self.intensity_growth = intensity_growth
        self.max_intensity = max_intensity

        # Precompute obstacle mask
        self.walls = (grid == -2)

        # Random state for reproducibility
        self.rng = np.random.default_rng()

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def step(self) -> np.ndarray:
        """
        Advance fire by one step.

        Returns:
            Updated grid
        """
        # Get current fire cells (intensity > 0, < max)
        active_fire = (self.grid > 0) & (self.grid < self.max_intensity)

        # Grow existing fire intensity
        self.grid = np.where(
            active_fire,
            np.minimum(self.grid + self.intensity_growth, self.max_intensity),
            self.grid
        )

        # Calculate spread probability using convolution
        fire_mask = (self.grid > 0).astype(np.float32)
        spread_prob = self._convolve(fire_mask, self.SPREAD_KERNEL)
        spread_prob *= self.spread_rate

        # Determine which cells ignite
        random_vals = self.rng.random((self.rows, self.cols), dtype=np.float32)
        ignite = (random_vals < spread_prob) & (self.grid == 0) & ~self.walls

        # Ignite new cells
        self.grid = np.where(ignite, 1.0, self.grid)

        return self.grid

    def step_n(self, n: int) -> np.ndarray:
        """Advance fire by n steps."""
        for _ in range(n):
            self.step()
        return self.grid

    def get_fire_cells(self) -> np.ndarray:
        """Get coordinates of cells on fire."""
        fire_y, fire_x = np.where(self.grid > 0)
        return np.column_stack((fire_x, fire_y))

    def get_intensity(self, x: int, y: int) -> float:
        """Get fire intensity at position."""
        if 0 <= x < self.cols and 0 <= y < self.rows:
            return max(0, self.grid[y, x])
        return 0.0

    def _convolve(self, arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Manual convolution (faster than scipy for small kernels)."""
        pad = kernel.shape[0] // 2
        padded = np.pad(arr, pad, mode='constant', constant_values=0)

        result = np.zeros_like(arr)
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                result += kernel[i, j] * padded[i:i+self.rows, j:j+self.cols]

        return result


class DeterministicFireModel(FastFireModel):
    """
    Deterministic fire spread for consistent RL training.

    Instead of random ignition, uses threshold-based spread.
    Ensures same input always produces same output.
    """

    def __init__(self, grid: np.ndarray,
                 spread_threshold: float = 0.3,
                 intensity_growth: float = 0.5):
        super().__init__(grid, spread_threshold, intensity_growth)
        self.spread_threshold = spread_threshold

    def step(self) -> np.ndarray:
        """Deterministic fire step."""
        active_fire = (self.grid > 0) & (self.grid < self.max_intensity)

        # Grow existing fire
        self.grid = np.where(
            active_fire,
            np.minimum(self.grid + self.intensity_growth, self.max_intensity),
            self.grid
        )

        # Calculate spread probability
        fire_mask = (self.grid > 0).astype(np.float32)
        spread_prob = self._convolve(fire_mask, self.SPREAD_KERNEL)

        # Deterministic ignition based on threshold
        ignite = (spread_prob >= self.spread_threshold) & (self.grid == 0) & ~self.walls
        self.grid = np.where(ignite, 1.0, self.grid)

        return self.grid
