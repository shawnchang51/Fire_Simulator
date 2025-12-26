"""
resplan_utils.py — Non–deep-learning helpers for the ResPlan-style floorplan datasets.

Dependencies (install as needed):
    pip install shapely geopandas matplotlib networkx numpy opencv-python

Contents:
    - Color maps and constants
    - Geometry utilities (get_geometries, centroid, perturb, noise)
    - Mask conversion (geometry_to_mask)
    - Grid rasterization (multilinestring_to_grid) - supercover line algorithm
    - Augmentations (rotate/flip/scale)
    - Buffer helpers (shrink→expand, expand→shrink)
    - Plan plotting (plot_plan)
    - Plan→graph (plan_to_graph) + graph overlay plotting (plot_plan_and_graph)
    - Dataset helpers (normalize_keys, get_plan_width, calculate_grid_size_for_plan)
"""

from __future__ import annotations
import math
from typing import Iterable, List, Dict, Any, Tuple, Optional, Union

import numpy as np
import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString, Point, GeometryCollection, base, box
)
from shapely.ops import unary_union
from shapely import affinity

# -----------------------------
# Colors & constants
# -----------------------------

CATEGORY_COLORS: Dict[str, str] = {
    "living": "#d9d9d9",    # light gray
    "bedroom": "#66c2a5",    # greenish
    "bathroom": "#fc8d62",   # orange
    "kitchen": "#8da0cb",    # blue
    "door": "#e78ac3",       # pink
    "window": "#a6d854",     # lime
    "wall": "#ffd92f",       # yellow
    "front_door": "#a63603", # dark reddish-brown
    "balcony": "#b3b3b3"     # dark gray
}

DEFAULT_CANVAS_SIZE = (256, 256)  # (H, W)

# -----------------------------
# Dataset helpers
# -----------------------------

def normalize_keys(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common key typos / variations in-place (balacony→balcony)."""
    if "balacony" in plan and "balcony" not in plan:
        plan["balcony"] = plan.pop("balacony")
    return plan

def get_plan_width(plan: Dict[str, Any]) -> float:
    """Returns the max(width, height) of the inner polygon bounds."""
    inner = plan.get("inner")
    if inner is None or inner.is_empty:
        return 0.0
    x1, y1, x2, y2 = inner.bounds
    return max(x2 - x1, y2 - y1)


def calculate_grid_size_for_plan(plan: dict, cell_size_m: float = 0.3) -> Optional[Tuple[int, int]]:
    """Compute recommended grid size (H, W) for a single plan dictionary.

    Returns a tuple (H, W) of integers, or None if calculation cannot be performed.
    """
    # Expect plan to be normalized (keys like 'inner' and 'area')
    inner = plan.get('inner')
    if inner is None or inner.is_empty:
        return None

    inner_area_shapely = inner.area
    floor_plan_area = plan.get('area')

    if not floor_plan_area or inner_area_shapely <= 0:
        return None

    # Linear scale from shapely pixels -> meters
    scale_factor_area = floor_plan_area / inner_area_shapely
    scale_factor_linear = np.sqrt(scale_factor_area)

    x1, y1, x2, y2 = inner.bounds
    width_pixels = x2 - x1
    height_pixels = y2 - y1

    # Compute grid dimensions to achieve desired cell resolution
    grid_width = int(np.ceil(width_pixels * scale_factor_linear / cell_size_m))
    grid_height = int(np.ceil(height_pixels * scale_factor_linear / cell_size_m))

    # sanity lower bound
    grid_width = max(8, grid_width)
    grid_height = max(8, grid_height)

    return (grid_height, grid_width)

def shrink_short_side(polygon: Polygon, scale: float) -> Polygon:
    """
    Shrink the side perpendicular to the longest side of a polygon by a scale factor.
    
    Assumes the polygon is roughly rectangular or quadrilateral.
    The function identifies the longest axis and scales the polygon
    along the perpendicular axis toward the centroid.
    
    Args:
        polygon: A Shapely Polygon (assumed to be rectangular/quadrilateral)
        scale: Scale factor for the perpendicular side (0 < scale < 1 shrinks, > 1 expands)
    
    Returns:
        A new Polygon with the side perpendicular to the longest side scaled.
    """
    if polygon.is_empty or not polygon.is_valid:
        return polygon
    
    # Get the minimum rotated rectangle to find principal axes
    min_rect = polygon.minimum_rotated_rectangle
    rect_coords = list(min_rect.exterior.coords)[:-1]  # Remove closing point
    
    if len(rect_coords) < 4:
        return polygon
    
    # Calculate side lengths of the minimum rotated rectangle
    side1_len = Point(rect_coords[0]).distance(Point(rect_coords[1]))
    side2_len = Point(rect_coords[1]).distance(Point(rect_coords[2]))
    
    # angle of the longest side
    long_angle = math.degrees(math.atan2(
        rect_coords[1][1] - rect_coords[0][1],
        rect_coords[1][0] - rect_coords[0][0]
    )) if side1_len >= side2_len else math.degrees(math.atan2(
        rect_coords[2][1] - rect_coords[1][1],
        rect_coords[2][0] - rect_coords[1][0]
    ))

    # thickness direction = perpendicular to longest side
    angle = long_angle + 90.0
    
    # Get centroid as the origin for transformation
    cx, cy = polygon.centroid.x, polygon.centroid.y
    
    # Rotate polygon so the perpendicular side aligns with x-axis
    rotated = affinity.rotate(polygon, -angle, origin=(cx, cy))
    
    # Scale along x-axis (the perpendicular side direction after rotation)
    scaled = affinity.scale(rotated, xfact=scale, yfact=1.0, origin=(cx, cy))
    
    # Rotate back to original orientation
    result = affinity.rotate(scaled, angle, origin=(cx, cy))
    
    return result

def get_structural_plan(plan: Dict[str, Any], scale: float = 1.0) -> Dict[str, Any]:
    """
    Generate a modified plan containing only structural elements:
    walls, doors, windows, and front_door.

    Args:
        plan: The original plan dictionary containing geometry data.

    Returns:
        A new dictionary with only 'wall', 'door', 'window', and 'front_door' keys
        (if they exist in the original plan), plus 'inner' and 'wall_width' if present.
    """
    structural_keys_multipolygon = ["wall", "door", "window"]
    structural_keys_polygon = ["front_door"]
    metadata_keys = ["inner", "wall_width"]  # commonly needed metadata

    modified_plan = {}

    # Copy structural elements
    for key in structural_keys_multipolygon:
        if key in plan and plan[key] is not None:
            mod_mp = []
            for poly in plan[key].geoms:
                modified_poly = shrink_short_side(poly, scale)
                mod_mp.append(modified_poly)
            modified_plan[key] = MultiPolygon(mod_mp)
                
    for key in structural_keys_polygon:
        if key in plan and plan[key] is not None:
            modified_plan[key] = shrink_short_side(plan[key], scale)

    # Copy metadata that may be needed for plotting/processing
    for key in metadata_keys:
        if key in plan and plan[key] is not None:
            modified_plan[key] = plan[key]

    return modified_plan

def _supercover_line(
    x0: float, y0: float, x1: float, y1: float,
    grid_width: int, grid_height: int
) -> List[Tuple[int, int]]:
    """
    Supercover line algorithm - visits ALL cells the line passes through.

    This algorithm ensures direction-independent results: drawing a line from
    A to B produces the exact same set of cells as drawing from B to A.

    For diagonal lines that pass through grid corners, this is achieved by
    always including both cells adjacent to the corner in a consistent manner.

    Args:
        x0, y0: Start point in grid coordinates (float)
        x1, y1: End point in grid coordinates (float)
        grid_width, grid_height: Grid dimensions for bounds checking

    Returns:
        Set of (col, row) tuples for all cells visited (as a list)
    """
    # Normalize direction: always process from lower to higher coordinates
    # This ensures consistent results regardless of input direction
    if (x0, y0) > (x1, y1):
        x0, y0, x1, y1 = x1, y1, x0, y0

    cells_set = set()

    # Always use floor for consistent cell mapping
    cx = int(math.floor(x0))
    cy = int(math.floor(y0))
    cx_end = int(math.floor(x1))
    cy_end = int(math.floor(y1))

    # Handle degenerate case: start and end in same cell
    if cx == cx_end and cy == cy_end:
        if 0 <= cx < grid_width and 0 <= cy < grid_height:
            cells_set.add((cx, cy))
        return list(cells_set)

    # Direction vector (after normalization, dx >= 0 always)
    dx = x1 - x0
    dy = y1 - y0

    # Step direction (+1 or -1)
    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1

    # t_max: parametric t where ray crosses next cell boundary
    # t_delta: parametric t to cross one full cell

    if dx != 0:
        t_delta_x = abs(1.0 / dx)
        if step_x > 0:
            t_max_x = (math.floor(x0) + 1 - x0) / dx
        else:
            if x0 == math.floor(x0):
                t_max_x = t_delta_x
            else:
                t_max_x = (math.floor(x0) - x0) / dx
    else:
        t_max_x = float('inf')
        t_delta_x = float('inf')

    if dy != 0:
        t_delta_y = abs(1.0 / dy)
        if step_y > 0:
            t_max_y = (math.floor(y0) + 1 - y0) / dy
        else:
            if y0 == math.floor(y0):
                t_max_y = t_delta_y
            else:
                t_max_y = (math.floor(y0) - y0) / dy
    else:
        t_max_y = float('inf')
        t_delta_y = float('inf')

    # Tolerance for detecting corner crossings (floating point comparison)
    eps = 1e-9

    # Traverse the line
    max_iterations = abs(cx_end - cx) + abs(cy_end - cy) + 10
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        # Add current cell if in bounds
        if 0 <= cx < grid_width and 0 <= cy < grid_height:
            cells_set.add((cx, cy))

        # Check if we've reached the destination
        if cx == cx_end and cy == cy_end:
            break

        # Check for corner crossing (line passes exactly through grid intersection)
        if abs(t_max_x - t_max_y) < eps:
            # Corner crossing: add BOTH adjacent cells for complete coverage
            # Move in x first
            cx += step_x
            if 0 <= cx < grid_width and 0 <= cy < grid_height:
                cells_set.add((cx, cy))
            # Then move in y
            cy += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            cx += step_x
            t_max_x += t_delta_x
        else:
            cy += step_y
            t_max_y += t_delta_y

    return list(cells_set)


def _remove_spurs(
    grid: np.ndarray,
    structural_value: int,
    empty_value: int,
    max_spur_length: int = 2
) -> np.ndarray:
    """
    Remove small spur (protrusion) artifacts from the rasterized grid.

    Spurs are short dead-end branches that stick out from walls at junctions,
    typically caused by the supercover algorithm marking extra cells when
    line segments meet at angles. This function traces paths from junctions
    and removes short dead-end branches.

    The algorithm:
    1. Finds all junction cells (degree >= 3)
    2. For each junction, traces each outgoing branch
    3. If a branch leads to a dead end within max_spur_length cells, removes it
    4. Only removes branches that won't reduce the junction below degree 2

    The algorithm preserves:
    - The 4-connectivity blocking property (no diagonal gaps introduced)
    - Main wall structures (long branches or branches connecting to other junctions)
    - Legitimate wall endpoints (branches longer than max_spur_length)

    Args:
        grid: 2D numpy array with structural and empty values
        structural_value: Value marking structural cells (e.g., -2)
        empty_value: Value marking passable cells (e.g., 0)
        max_spur_length: Maximum length of spurs to remove. Default is 2.

    Returns:
        Modified grid with spurs removed (in-place modification)
    """
    h, w = grid.shape

    # 4-connected neighbor offsets: up, down, left, right
    neighbors_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_structural_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        """Get coordinates of structural neighbors."""
        neighbors = []
        for dr, dc in neighbors_4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if grid[nr, nc] == structural_value:
                    neighbors.append((nr, nc))
        return neighbors

    # Find all junction cells (degree >= 3)
    junctions = set()
    for r in range(h):
        for c in range(w):
            if grid[r, c] == structural_value:
                if len(get_structural_neighbors(r, c)) >= 3:
                    junctions.add((r, c))

    if not junctions:
        return grid

    # For each junction, check each branch for spurs
    cells_to_remove = set()

    for jr, jc in junctions:
        neighbors = get_structural_neighbors(jr, jc)

        # Count how many branches are valid (not spurs)
        # A junction must retain at least 2 valid branches
        spur_branches = []

        for start_nr, start_nc in neighbors:
            # Trace this branch to see if it's a spur
            path = [(start_nr, start_nc)]
            current = (start_nr, start_nc)
            prev = (jr, jc)
            is_spur = False

            for step in range(max_spur_length):
                current_neighbors = get_structural_neighbors(current[0], current[1])
                # Remove the previous cell from consideration
                next_cells = [n for n in current_neighbors if n != prev]

                if len(next_cells) == 0:
                    # Dead end within max_spur_length - this is a spur
                    is_spur = True
                    break
                elif len(next_cells) == 1:
                    # Continue along the path
                    next_cell = next_cells[0]
                    # If we hit another junction, this is not a spur
                    if next_cell in junctions:
                        break
                    prev = current
                    current = next_cell
                    path.append(current)
                else:
                    # Multiple paths - hit another junction or fork
                    break

            if is_spur:
                spur_branches.append(path)

        # Only remove spurs if the junction will retain at least 2 branches
        valid_branch_count = len(neighbors) - len(spur_branches)
        if valid_branch_count >= 2:
            # Safe to remove all spurs
            for path in spur_branches:
                cells_to_remove.update(path)
        elif valid_branch_count == 1 and len(spur_branches) > 1:
            # Keep one spur to maintain junction, remove the rest
            # Sort by length and remove all but the longest spur
            spur_branches.sort(key=len)
            for path in spur_branches[:-1]:
                cells_to_remove.update(path)

    # Remove the identified spur cells
    for r, c in cells_to_remove:
        grid[r, c] = empty_value

    return grid


def _polygon_centerline(poly: Polygon, densify_distance: float = 1.0) -> List[LineString]:
    """
    Extract the centerline (medial axis) of a polygon using Voronoi diagram.

    Args:
        poly: The polygon to extract centerline from.
        densify_distance: Distance for densifying the polygon boundary before Voronoi.

    Returns:
        List of LineStrings representing the centerline.
    """
    from scipy.spatial import Voronoi

    if poly.is_empty or not poly.is_valid:
        return []

    # Densify the boundary to get more Voronoi vertices
    boundary = poly.exterior
    densified = boundary.segmentize(densify_distance)
    coords = np.array(densified.coords)[:-1]  # Remove duplicate closing point

    if len(coords) < 3:
        return []

    try:
        vor = Voronoi(coords)
    except Exception:
        return []

    # Extract Voronoi edges that lie inside the polygon
    centerlines = []
    for ridge_vertices in vor.ridge_vertices:
        if -1 in ridge_vertices:
            continue  # Skip infinite ridges
        v0, v1 = ridge_vertices
        p0 = Point(vor.vertices[v0])
        p1 = Point(vor.vertices[v1])

        # Only keep edges where both endpoints are inside the polygon
        if poly.contains(p0) and poly.contains(p1):
            line = LineString([vor.vertices[v0], vor.vertices[v1]])
            if not line.is_empty and line.length > 0:
                centerlines.append(line)

    return centerlines


def structural_plan_to_multilinestring(plan: Dict[str, Any],
                                        densify_distance: float = 1.0) -> MultiLineString:
    """
    Convert structural elements (walls, doors, windows, front_door) into a single MultiLineString.

    Extracts the centerline (medial axis) of polygon geometries using Voronoi diagram.
    LineString geometries are kept as-is.

    Args:
        plan: The plan dictionary containing geometry data.
        densify_distance: Distance for densifying polygon boundaries before Voronoi.
                          Smaller values give more accurate centerlines but slower.

    Returns:
        A MultiLineString containing centerlines of all structural elements.
    """
    structural_keys = ["wall", "door", "window", "front_door"]
    all_lines = []

    for key in structural_keys:
        geom = plan.get(key)
        if geom is None:
            continue

        geoms = get_geometries(geom)
        for g in geoms:
            if isinstance(g, (LineString, MultiLineString)):
                # Already a line geometry - keep as-is
                if isinstance(g, LineString):
                    all_lines.append(g)
                else:
                    all_lines.extend(g.geoms)
            elif isinstance(g, Polygon):
                # Extract centerline of the polygon
                centerlines = _polygon_centerline(g, densify_distance)
                all_lines.extend(centerlines)
            elif isinstance(g, MultiPolygon):
                for poly in g.geoms:
                    centerlines = _polygon_centerline(poly, densify_distance)
                    all_lines.extend(centerlines)

    if not all_lines:
        return MultiLineString()

    return MultiLineString(all_lines)


def multilinestring_to_grid(
    multiline: MultiLineString,
    plan: Dict[str, Any],
    cell_size_m: float = 0.3,
    structural_value: int = -2,
    empty_value: int = 0,
    add_border: bool = True,
    mark_outside: bool = True,
    remove_spurs: bool = True,
    max_spur_length: int = 2,
    min_segment_length: float = 0.1,
    filter_branch_endpoints: bool = True,
    max_branch_length: float = 0.5
) -> np.ndarray:
    """
    Rasterize a MultiLineString onto a grid using Amanatides & Woo grid traversal.

    Converts vector-based structural elements into a discrete grid where structural
    elements occupy exactly one cell in width and cannot be bypassed by diagonal
    movement. The algorithm visits every grid cell that a line passes through,
    ensuring a strictly 4-connected blocking structure with no corner cutting.

    Coordinate mapping uses floor-based rounding: grid_cell = floor((world - min) * scale)
    This ensures deterministic, stable rasterization across platforms.

    The output grid is flipped vertically to match intuitive visualization (y=0 at bottom).

    Args:
        multiline: MultiLineString geometry (from structural_plan_to_multilinestring)
        plan: Plan dictionary containing 'inner' polygon and 'area' for scaling
        cell_size_m: Cell size in meters (default 0.3m)
        structural_value: Value for structural cells (default -2)
        empty_value: Value for passable cells (default 0)
        add_border: If True, adds a 1-cell border of structural_value around the grid
                    to represent outside walls (default True)
        mark_outside: If True, marks cells outside the 'inner' polygon with
                      structural_value (default True)
        remove_spurs: If True, removes small protrusions/spikes at line junctions
                      caused by supercover algorithm over-coverage (default True)
        max_spur_length: Maximum length of spurs to remove (default 2). Only used
                         when remove_spurs=True.
        min_segment_length: Minimum segment length in world coordinates to rasterize.
                            Segments shorter than this are skipped to avoid
                            artifacts from tiny Voronoi centerline fragments.
                            Default is 0.1 world units.
        filter_branch_endpoints: If True, filters out short branch segments that have
                                 one endpoint with degree 1 (free endpoint). This removes
                                 Y-shaped branches at junctions between structural elements.
                                 Default is True.
        max_branch_length: Maximum length of branch segments to filter (in world coordinates).
                          Only used when filter_branch_endpoints=True. Segments with one
                          free endpoint and length < this value will be skipped.
                          Default is 0.5 world units.

    Returns:
        2D numpy array of shape (H, W) or (H+2, W+2) if add_border=True.
        Grid is flipped vertically for intuitive visualization.
        Grid uses image convention: grid[row, col] = grid[y, x]

    Raises:
        ValueError: If grid size cannot be calculated from plan
    """
    # Calculate grid dimensions
    grid_size = calculate_grid_size_for_plan(plan, cell_size_m)
    if grid_size is None:
        raise ValueError("Cannot calculate grid size for plan. "
                         "Ensure plan has valid 'inner' polygon and 'area'.")
    grid_height, grid_width = grid_size

    # Get bounds for coordinate transformation
    inner = plan.get('inner')
    if inner is None or inner.is_empty:
        raise ValueError("Plan must have a valid 'inner' polygon")

    x_min, y_min, x_max, y_max = inner.bounds

    # Scale factors: world units -> grid units
    width_world = x_max - x_min
    height_world = y_max - y_min

    if width_world <= 0 or height_world <= 0:
        raise ValueError("Plan 'inner' polygon has zero or negative dimensions")

    scale_x = grid_width / width_world
    scale_y = grid_height / height_world

    # Initialize grid with empty value
    grid = np.full((grid_height, grid_width), empty_value, dtype=np.int8)

    # Build endpoint degree map for filtering Y-shaped branches
    endpoint_degree = {}
    if filter_branch_endpoints and not multiline.is_empty:
        for linestring in multiline.geoms:
            coords = list(linestring.coords)
            for i in range(len(coords) - 1):
                # Round coordinates to avoid floating point precision issues
                p0 = (round(coords[i][0], 6), round(coords[i][1], 6))
                p1 = (round(coords[i + 1][0], 6), round(coords[i + 1][1], 6))

                # Count connections at each endpoint
                endpoint_degree[p0] = endpoint_degree.get(p0, 0) + 1
                endpoint_degree[p1] = endpoint_degree.get(p1, 0) + 1

    # Handle empty MultiLineString
    if not multiline.is_empty:
        # Process each LineString in the MultiLineString
        for linestring in multiline.geoms:
            coords = list(linestring.coords)

            # Process each segment of the LineString
            for i in range(len(coords) - 1):
                # Get endpoints in world coordinates
                wx0, wy0 = coords[i][0], coords[i][1]
                wx1, wy1 = coords[i + 1][0], coords[i + 1][1]

                # Calculate segment length
                segment_length_world = math.sqrt((wx1 - wx0)**2 + (wy1 - wy0)**2)

                # Skip very short segments that create spur artifacts
                # These often come from Voronoi centerline extraction
                if segment_length_world < min_segment_length:
                    continue

                # Filter out Y-shaped branches at junctions
                if filter_branch_endpoints:
                    p0 = (round(wx0, 6), round(wy0, 6))
                    p1 = (round(wx1, 6), round(wy1, 6))

                    # Check if this is a short branch with a free endpoint (degree 1)
                    degree0 = endpoint_degree.get(p0, 0)
                    degree1 = endpoint_degree.get(p1, 0)

                    # Skip if segment has one free endpoint and is short
                    if segment_length_world < max_branch_length:
                        if degree0 == 1 or degree1 == 1:
                            continue

                # Transform to grid coordinates (float)
                # floor() is applied inside _supercover_line
                gx0 = (wx0 - x_min) * scale_x
                gy0 = (wy0 - y_min) * scale_y
                gx1 = (wx1 - x_min) * scale_x
                gy1 = (wy1 - y_min) * scale_y

                # Get all cells using Amanatides & Woo traversal
                cells = _supercover_line(gx0, gy0, gx1, gy1, grid_width, grid_height)

                # Mark cells as structural
                # Note: grid[row, col] = grid[y, x], cells are (col, row) = (x, y)
                for col, row in cells:
                    grid[row, col] = structural_value

    # Remove spur artifacts from line junctions before marking outside cells
    # This must happen before mark_outside to avoid removing boundary cells
    if remove_spurs:
        _remove_spurs(grid, structural_value, empty_value, max_spur_length)

    # Mark cells outside the inner polygon as structural
    if mark_outside:
        for row in range(grid_height):
            for col in range(grid_width):
                # Convert grid cell center to world coordinates
                wx = x_min + (col + 0.5) / scale_x
                wy = y_min + (row + 0.5) / scale_y
                point = Point(wx, wy)
                if not inner.contains(point):
                    grid[row, col] = structural_value

    # Flip vertically to match intuitive orientation (y=0 at bottom)
    grid = np.flipud(grid)

    # Add 1-cell border of structural value to represent outside walls
    if add_border:
        bordered_grid = np.full(
            (grid_height + 2, grid_width + 2),
            structural_value,
            dtype=np.int8
        )
        bordered_grid[1:-1, 1:-1] = grid
        grid = bordered_grid

    return grid


# -----------------------------
# Geometry utilities
# -----------------------------

def get_geometries(geom_data: Any) -> List[Any]:
    """Safely extract individual geometries from single/multi/collections."""
    if geom_data is None:
        return []
    if isinstance(geom_data, (Polygon, LineString, Point)):
        return [] if geom_data.is_empty else [geom_data]
    if isinstance(geom_data, (MultiPolygon, MultiLineString, GeometryCollection)):
        return [g for g in geom_data.geoms if g is not None and not g.is_empty]
    return []

def centroid(poly: Union[Polygon, MultiPolygon]) -> Point:
    """Centroid for Polygon/MultiPolygon (largest part if multi)."""
    if isinstance(poly, Polygon):
        return poly.centroid
    if isinstance(poly, MultiPolygon) and len(poly.geoms) > 0:
        largest = max(poly.geoms, key=lambda p: p.area)
        return largest.centroid
    return Point(-1e6, -1e6)

def perturb_polygon(polygon: Polygon, x_range: Tuple[float, float]=(-2, 2),
                    y_range: Tuple[float, float]=(-2, 2)) -> Polygon:
    """Apply random per-vertex perturbation to a polygon."""
    coords = np.asarray(polygon.exterior.coords, dtype=float)
    dx = np.random.uniform(x_range[0], x_range[1], size=len(coords))
    dy = np.random.uniform(y_range[0], y_range[1], size=len(coords))
    perturbed = np.column_stack([coords[:,0] + dx, coords[:,1] + dy])
    return Polygon(perturbed)

def noise(point: Point, noise_scale: float = 10.0) -> Point:
    """Jitter a point by uniform noise within ±noise_scale."""
    x, y = point.x, point.y
    return Point(x + np.random.uniform(-noise_scale, noise_scale),
                 y + np.random.uniform(-noise_scale, noise_scale))

# -----------------------------
# Augmentations
# -----------------------------

def augment_geom(geom: base.BaseGeometry,
                 degree: float = 0.0,
                 flip_vertical: bool = False,
                 scale: float = 1.0,
                 size: int = 256) -> base.BaseGeometry:
    """Rotate around image center, optional vertical flip (via negative y-scale), and scale."""
    if geom is None:
        return Point(-1e6, -1e6)
    g = affinity.rotate(geom, degree, origin=(size/2, size/2))
    flip = -1.0 if flip_vertical else 1.0
    return affinity.scale(g, xfact=scale, yfact=scale * flip, origin=(size/2, size/2))

# -----------------------------
# Buffer helpers
# -----------------------------

def buffer_shrink_expand(geom: base.BaseGeometry, w: float,
                         join_style: int = 2, cap_style: int = 2) -> base.BaseGeometry:
    """Shrink then expand by w (useful for cleaning)."""
    return geom.buffer(-w, join_style=join_style, cap_style=cap_style)                   .buffer(+w, join_style=join_style, cap_style=cap_style)

def buffer_expand_shrink(geom: base.BaseGeometry, w: float,
                         join_style: int = 2, cap_style: int = 2) -> base.BaseGeometry:
    """Expand then shrink by w (useful for filling tiny gaps)."""
    return geom.buffer(+w, join_style=join_style, cap_style=cap_style)                   .buffer(-w, join_style=join_style, cap_style=cap_style)

# -----------------------------
# Geometry → mask
# -----------------------------

def _poly_to_mask(poly: Polygon, shape: Tuple[int, int], line_thickness: int = 0) -> np.ndarray:
    h, w = shape
    img = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly.exterior.coords, dtype=np.int32)
    if line_thickness > 0:
        cv2.polylines(img, [pts], isClosed=True, color=255, thickness=line_thickness)
    else:
        cv2.fillPoly(img, [pts], color=255)
    for interior in poly.interiors:
        pts_in = np.array(interior.coords, dtype=np.int32)
        if line_thickness > 0:
            cv2.polylines(img, [pts_in], isClosed=True, color=0, thickness=line_thickness)
        else:
            cv2.fillPoly(img, [pts_in], color=0)
    return img

def geometry_to_mask(geom: Any,
                     shape: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
                     point_radius: int = 5,
                     line_thickness: int = 0) -> np.ndarray:
    """Rasterize Polygon/MultiPolygon/LineString/Point/iterables to a binary mask [0,255]."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)

    # Single geometry
    if isinstance(geom, Polygon):
        return _poly_to_mask(geom, shape, line_thickness)
    if isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            out = np.maximum(out, _poly_to_mask(p, shape, line_thickness))
        return out
    if isinstance(geom, LineString):
        pts = np.array(geom.coords, dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=False, color=255, thickness=max(1, line_thickness or 1))
        return out
    if isinstance(geom, MultiLineString):
        for ls in geom.geoms:
            pts = np.array(ls.coords, dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=False, color=255, thickness=max(1, line_thickness or 1))
        return out
    if isinstance(geom, Point):
        cx, cy = int(round(geom.x)), int(round(geom.y))
        cv2.circle(out, (cx, cy), point_radius, 255, -1)
        return out
    if isinstance(geom, Iterable):
        for g in geom:
            out = np.maximum(out, geometry_to_mask(g, shape, point_radius, line_thickness))
        return out
    # Unrecognized → empty
    return out

# -----------------------------
# Plotting
# -----------------------------

def plot_plan(plan: Dict[str, Any],
              categories: Optional[List[str]] = None,
              colors: Dict[str, str] = CATEGORY_COLORS,
              ax: Optional[plt.Axes] = None,
              legend: bool = True,
              title: Optional[str] = None,
              tight: bool = True) -> plt.Axes:
    """Plot a single plan with colored layers."""
    plan = normalize_keys(plan)
    if categories is None:
        categories = ["living","bedroom","bathroom","kitchen","door","window","wall","front_door","balcony"]

    geoms, color_list, present = [], [], []
    for key in categories:
        geom = plan.get(key)
        if geom is None:
            continue
        parts = get_geometries(geom)
        if not parts:
            continue
        geoms.extend(parts)
        color_list.extend([colors.get(key, "#000000")] * len(parts))
        present.append(key)

    if not geoms:
        raise ValueError("No geometries to plot.")

    gseries = gpd.GeoSeries(geoms)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    gseries.plot(ax=ax, color=color_list, edgecolor="black", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    if title:
        ax.set_title(title)

    if legend:
        from matplotlib.patches import Patch
        uniq_present = list(dict.fromkeys(present))  # preserve order
        handles = [Patch(facecolor=colors.get(k, "#000000"), edgecolor="black", label=k.replace("_"," ")) for k in uniq_present]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1,1), frameon=False)

    if tight:
        plt.tight_layout()
    return ax

# -----------------------------
# Plan → Graph
# -----------------------------

def plan_to_graph(plan: Dict[str, Any],
                  buffer_factor: float = 0.75) -> nx.Graph:
    """Create a simple room graph: nodes are room parts; edges denote adjacency or connections via door/window."""
    plan = normalize_keys(plan)
    G = nx.Graph()
    ww = float(plan.get("wall_width", 0.1) or 0.1)
    buf = max(ww * buffer_factor, 0.01)

    nodes_by_type: Dict[str, List[str]] = {k: [] for k in ["living","kitchen","bedroom","bathroom","balcony","front_door"]}

    # rooms
    for room_type in ["living","kitchen","bedroom","bathroom","balcony"]:
        parts = get_geometries(plan.get(room_type))
        # for living, keep separate parts; user can union beforehand if desired
        for i, geom in enumerate(parts):
            if isinstance(geom, Polygon) and not geom.is_empty:
                nid = f"{room_type}_{i}"
                G.add_node(nid, geometry=geom, type=room_type, area=geom.area)
                nodes_by_type[room_type].append(nid)

    # front door (may be line/polygon)
    for i, geom in enumerate(get_geometries(plan.get("front_door"))):
        nid = f"front_door_{i}"
        G.add_node(nid, geometry=geom, type="front_door", area=getattr(geom, "area", 0.0))
        nodes_by_type["front_door"].append(nid)

    doors  = get_geometries(plan.get("door"))
    wins   = get_geometries(plan.get("window"))
    conns  = [(d, "via_door") for d in doors] + [(w, "via_window") for w in wins]

    # front_door → living
    for fd in nodes_by_type["front_door"]:
        fd_geom = G.nodes[fd]["geometry"]
        for gen in nodes_by_type["living"]:
            gen_geom = G.nodes[gen]["geometry"]
            if fd_geom.intersects(gen_geom.buffer(buf)):
                G.add_edge(fd, gen, type="direct")

    # adjacency: kitchen/bedroom ↔ living
    for room_type in ["kitchen","bedroom"]:
        for rn in nodes_by_type[room_type]:
            rgeom = G.nodes[rn]["geometry"].buffer(buf)
            for gen in nodes_by_type["living"]:
                gen_geom = G.nodes[gen]["geometry"]
                if rgeom.buffer(buf).intersects(gen_geom.buffer(buf)):
                    G.add_edge(rn, gen, type="adjacency")

    # bathroom & balcony connections via door/window to living/bedroom
    for room_type in ["bathroom","balcony"]:
        for rn in nodes_by_type[room_type]:
            rgeom = G.nodes[rn]["geometry"].buffer(buf)
            for cgeom, ctype in conns:
                if not cgeom.intersects(rgeom):
                    continue
                for target_type in ["living","bedroom"]:
                    for tn in nodes_by_type[target_type]:
                        tgeom = G.nodes[tn]["geometry"].buffer(buf)
                        if cgeom.intersects(tgeom):
                            if not G.has_edge(rn, tn):
                                G.add_edge(rn, tn, type=ctype)
    return G

# -----------------------------
# Graph overlay on plan
# -----------------------------

def plot_plan_and_graph(plan: Dict[str, Any],
                        ax: Optional[plt.Axes] = None,
                        node_scale: Tuple[float,float]=(150, 1000),
                        title: Optional[str] = None) -> plt.Axes:
    """Plot plan and overlay the room graph (node size scaled by room area)."""
    G = plan["graph"] if "graph" in plan else plan_to_graph(plan)
    ax = plot_plan(plan, legend=True, ax=ax, title=title)

    # node positions = centroids
    pos = {}
    for n, data in G.nodes(data=True):
        geom = data.get("geometry")
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        pos[n] = (c.x, c.y)

    # style maps
    node_style = {
        "living":    dict(color="white",     shape="o", size=400, edgecolor="black"),
        "bedroom":    dict(color="cyan",      shape="s", size=300, edgecolor="black"),
        "bathroom":   dict(color="magenta",   shape="D", size=260, edgecolor="black"),
        "kitchen":    dict(color="yellow",    shape="^", size=300, edgecolor="black"),
        "balcony":    dict(color="lightgray", shape="X", size=260, edgecolor="black"),
        "front_door": dict(color="red",       shape="*", size=420, edgecolor="black"),
    }

    # draw nodes per type for shapes
    nodes_plotted = set()
    min_size, max_size = node_scale
    # area-based scaling
    areas = [G.nodes[n].get("area", 0.0) for n in G.nodes]
    a_min = min(areas) if areas else 0.0
    a_max = max(areas) if areas else 1.0
    def scale_size(a):
        if a_max <= a_min:
            return (min_size + max_size) / 2
        t = (a - a_min) / (a_max - a_min)
        return min_size + t * (max_size - min_size)

    for t, style in node_style.items():
        nlist = [n for n, d in G.nodes(data=True) if d.get("type")==t and n in pos]
        if not nlist:
            continue
        sizes = [scale_size(G.nodes[n].get("area", 0.0)) for n in nlist]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nlist, node_size=sizes,
            node_shape=style["shape"], node_color=style["color"],
            edgecolors=style["edgecolor"], linewidths=1.0, ax=ax, alpha=0.9
        )
        nodes_plotted.update(nlist)

    # edges by type
    edge_style = {
        "direct":     dict(color="darkred",   width=2.0,  style="-"),
        "adjacency":  dict(color="darkgreen", width=1.5,  style="--"),
        "via_door":   dict(color="darkblue",  width=1.2,  style="-"),
        "via_window": dict(color="orange",    width=1.0,  style=":"),
    }
    for etype, style in edge_style.items():
        elist = [(u,v) for u,v,d in G.edges(data=True) if d.get("type")==etype and u in pos and v in pos]
        if not elist:
            continue
        nx.draw_networkx_edges(G, pos, edgelist=elist,
                               width=style["width"], edge_color=style["color"],
                               style=style["style"], ax=ax, alpha=0.8)

    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax
