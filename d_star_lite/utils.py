from functools import lru_cache

@lru_cache(maxsize=10000)
def stateNameToCoords(name):
    """Convert state name to coordinates with caching for performance.

    Cache up to 10k conversions to avoid repeated string parsing.
    For a 60x60 grid, this covers all 3600 cells plus extras.
    """
    return [int(name.split('x')[1].split('y')[0]), int(name.split('x')[1].split('y')[1])]

def coordsToStateName(col, row):
    """Convert column and row coordinates to state name format 'x{col}y{row}'"""
    return f"x{col}y{row}"
