"""
Vectorized Fire Model for RL Training
=====================================

Simplified cellular automata fire spread using NumPy operations.
Removes complex physics (oxygen, temperature, smoke) for speed.

Performance: 5-10x faster than AdvancedFireModel

Fire Spread Modes:
- 'always_real': Stochastic spread continues throughout simulation (most realistic)
- 'real_then_simple': Stochastic spread until stable, then deterministic intensity growth only
- 'real_then_stop': Stochastic spread until stable, then fire becomes completely static
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum

class FireSpreadMode(Enum):
    """Fire spread behavior modes."""
    ALWAYS_REAL = 'always_real'           # Continuous stochastic spread (most realistic)
    REAL_THEN_SIMPLE = 'real_then_simple' # Stochastic spread, then intensity growth only
    REAL_THEN_STOP = 'real_then_stop'     # Stochastic spread, then completely static

class FastFireModel:
    """
    Vectorized fire spread using NumPy convolution.

    Simplifications vs AdvancedFireModel:
    - No oxygen/temperature/smoke tracking
    - No fuel depletion
    - Fixed spread probabilities
    - No wind effects (can be added if needed)

    Supports three fire spread modes:
    - ALWAYS_REAL: Continuous stochastic spread (default, most realistic)
    - REAL_THEN_SIMPLE: Stochastic until stable, then intensity growth only
    - REAL_THEN_STOP: Stochastic until stable, then completely static
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
                 max_intensity: float = 4.0,
                 spread_mode: FireSpreadMode = FireSpreadMode.ALWAYS_REAL,
                 stability_threshold: int = 3):
        """
        Initialize fire model.

        Args:
            grid: 2D array with initial fire state (-2=wall, 0=empty, >0=fire)
            spread_rate: Probability multiplier for fire spread
            intensity_growth: How fast fire intensity grows per step
            max_intensity: Maximum fire intensity
            spread_mode: Fire spread behavior mode
            stability_threshold: Number of consecutive steps with no new ignitions to consider stable
        """
        self.grid = grid.astype(np.float32)
        self.rows, self.cols = grid.shape
        self.spread_rate = spread_rate
        self.intensity_growth = intensity_growth
        self.max_intensity = max_intensity
        self.spread_mode = spread_mode
        self.stability_threshold = stability_threshold

        # Precompute obstacle mask
        self.walls = (grid == -2)

        # Random state for reproducibility
        self.rng = np.random.default_rng()

        # Tracking for spread mode transitions
        self.steps_without_spread = 0
        self.is_stable = False

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

        # Mode: REAL_THEN_STOP - completely static after stability
        if self.spread_mode == FireSpreadMode.REAL_THEN_STOP and self.is_stable:
            return self.grid  # No changes at all

        # Grow existing fire intensity (unless in REAL_THEN_STOP and stable)
        self.grid = np.where(
            active_fire,
            np.minimum(self.grid + self.intensity_growth, self.max_intensity),
            self.grid
        )

        # Handle spread based on mode
        ignite_count = 0

        # Mode: ALWAYS_REAL or not yet stable - perform stochastic spread
        if self.spread_mode == FireSpreadMode.ALWAYS_REAL or not self.is_stable:
            # Calculate spread probability using convolution
            fire_mask = (self.grid > 0).astype(np.float32)
            spread_prob = self._convolve(fire_mask, self.SPREAD_KERNEL)
            spread_prob *= self.spread_rate

            # Determine which cells ignite
            random_vals = self.rng.random((self.rows, self.cols), dtype=np.float32)
            ignite = (random_vals < spread_prob) & (self.grid == 0) & ~self.walls

            # Count new ignitions
            ignite_count = np.sum(ignite)

            # Ignite new cells
            self.grid = np.where(ignite, 1.0, self.grid)

        # Mode: REAL_THEN_SIMPLE - no spread after stability, only intensity growth (handled above)
        # Mode: REAL_THEN_STOP - completely static (handled at top)

        # Track stability for mode transitions
        if self.spread_mode != FireSpreadMode.ALWAYS_REAL:
            if ignite_count == 0:
                self.steps_without_spread += 1
                if self.steps_without_spread >= self.stability_threshold and not self.is_stable:
                    self.is_stable = True
                    print(f"Fire spread stabilized after {self.stability_threshold} steps without new ignitions")
                    if self.spread_mode == FireSpreadMode.REAL_THEN_SIMPLE:
                        print(f"  → Switching to intensity growth only (mode: {self.spread_mode.value})")
                    elif self.spread_mode == FireSpreadMode.REAL_THEN_STOP:
                        print(f"  → Fire now completely static (mode: {self.spread_mode.value})")
            else:
                self.steps_without_spread = 0

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
