"""
Coordinate Utilities for D* Lite Pathfinding
============================================

Provides efficient coordinate conversion functions. The optimized versions
use integer indices for internal operations, with string conversion only
for external interfaces (visualization, JSON output, etc.).

Performance Notes:
- Integer index operations: ~10x faster than string parsing
- LRU cache on string functions: handles backwards compatibility efficiently
- Use integer indices internally, convert at boundaries only
"""

from functools import lru_cache
from typing import Tuple, List, Union


# =============================================================================
# FAST INTEGER COORDINATE FUNCTIONS (USE THESE INTERNALLY)
# =============================================================================

def coords_to_index(x: int, y: int, cols: int) -> int:
    """Convert (x, y) coordinates to flat integer index. FAST - use internally."""
    return y * cols + x


def index_to_coords(index: int, cols: int) -> Tuple[int, int]:
    """Convert flat integer index to (x, y) coordinates. FAST - use internally."""
    return (index % cols, index // cols)


def index_to_coords_array(index: int, cols: int) -> List[int]:
    """Convert flat integer index to [x, y] list (for backwards compatibility)."""
    return [index % cols, index // cols]


# =============================================================================
# STRING-BASED FUNCTIONS (FOR BACKWARDS COMPATIBILITY)
# =============================================================================

@lru_cache(maxsize=10000)
def stateNameToCoords(name: str) -> List[int]:
    """Convert state name to coordinates with caching for performance.

    Cache up to 10k conversions to avoid repeated string parsing.
    For a 60x60 grid, this covers all 3600 cells plus extras.
    
    Args:
        name: State name in format 'x{col}y{row}'
        
    Returns:
        [col, row] as integers
    """
    # Optimized parsing - avoid multiple splits
    y_pos = name.find('y')
    x_val = int(name[1:y_pos])
    y_val = int(name[y_pos+1:])
    return [x_val, y_val]


def coordsToStateName(col: int, row: int) -> str:
    """Convert column and row coordinates to state name format 'x{col}y{row}'"""
    return f"x{col}y{row}"


@lru_cache(maxsize=10000)
def stateNameToIndex(name: str, cols: int) -> int:
    """Convert state name directly to integer index. Cached for performance."""
    coords = stateNameToCoords(name)
    return coords_to_index(coords[0], coords[1], cols)


def indexToStateName(index: int, cols: int) -> str:
    """Convert integer index to state name."""
    x, y = index_to_coords(index, cols)
    return f"x{x}y{y}"


# =============================================================================
# HEURISTIC FUNCTIONS
# =============================================================================

def heuristic_int(from_idx: int, to_idx: int, cols: int) -> float:
    """
    Calculate heuristic (Chebyshev distance) between two integer indices.
    
    This is the octile/Chebyshev distance which is admissible for 8-connected grids.
    Much faster than string-based version.
    """
    from_x, from_y = index_to_coords(from_idx, cols)
    to_x, to_y = index_to_coords(to_idx, cols)
    return max(abs(from_x - to_x), abs(from_y - to_y))


def heuristic_str(from_state: str, to_state: str) -> float:
    """
    Calculate heuristic (Chebyshev distance) between two state name strings.
    
    Backwards compatible version - use heuristic_int internally when possible.
    """
    from_coords = stateNameToCoords(from_state)
    to_coords = stateNameToCoords(to_state)
    return max(abs(from_coords[0] - to_coords[0]), abs(from_coords[1] - to_coords[1]))
