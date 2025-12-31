"""
Graph Neural Network Encoder for Floor Plans

Models floor plans as graphs where:
- Nodes: Rooms, corridors, doors, exits (spatial regions)
- Edges: Connectivity between regions (passages, doors)

Advantages over CNN:
- Captures topological structure (bottlenecks, connectivity)
- Invariant to spatial transformations
- Explicitly models evacuation paths

Architecture Options:
1. GATv2: Graph Attention Network with dynamic attention
2. GraphSAGE: Sampling and aggregating neighborhood features
3. GIN: Graph Isomorphism Network for expressive power
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class FloorPlanGraph:
    """
    Graph representation of a floor plan.

    Attributes:
        node_features: Node feature matrix (N, D_node)
        edge_index: Edge connectivity in COO format (2, E)
        edge_attr: Edge attributes (E, D_edge)
        node_types: Node type labels (N,) - 0=room, 1=corridor, 2=door, 3=exit
        batch: Batch assignment for each node (N,) - used for batching
    """
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor] = None
    node_types: Optional[torch.Tensor] = None
    batch: Optional[torch.Tensor] = None
    num_nodes: int = 0


class GridToGraphConverter:
    """
    Converts grid-based floor plan representation to graph.

    Extracts spatial regions and their connectivity from the grid encoding.
    """

    def __init__(
        self,
        min_region_size: int = 16,
        connectivity_radius: int = 3,
    ):
        """
        Initialize converter.

        Args:
            min_region_size: Minimum pixels for a region to be a node
            connectivity_radius: Distance threshold for edge creation
        """
        self.min_region_size = min_region_size
        self.connectivity_radius = connectivity_radius

    def convert(self, grid: torch.Tensor) -> FloorPlanGraph:
        """
        Convert grid to graph representation.

        Args:
            grid: Floor plan grid (5, H, W) - [wall, passable, doors, exits, valid]

        Returns:
            FloorPlanGraph representation
        """
        # Ensure CPU numpy for processing
        if grid.is_cuda:
            grid = grid.cpu()
        grid_np = grid.numpy()

        # Extract channels
        walls = grid_np[0]
        passable = grid_np[1]
        doors = grid_np[2]
        exits = grid_np[3]
        valid = grid_np[4]

        # Find connected components in passable area
        from scipy import ndimage

        # Label connected passable regions
        passable_binary = (passable > 0.5).astype(np.int32)
        labeled, num_regions = ndimage.label(passable_binary)

        nodes = []
        node_types = []
        node_positions = []

        # Create nodes for each region
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled == region_id)
            region_size = region_mask.sum()

            if region_size < self.min_region_size:
                continue

            # Compute region properties
            y_coords, x_coords = np.where(region_mask)
            centroid_y = y_coords.mean()
            centroid_x = x_coords.mean()

            # Check if region contains doors or exits
            has_door = (doors[region_mask] > 0.5).any()
            has_exit = (exits[region_mask] > 0.5).any()

            # Node features
            features = [
                region_size / (grid_np.shape[1] * grid_np.shape[2]),  # Normalized area
                centroid_x / grid_np.shape[2],  # Normalized x position
                centroid_y / grid_np.shape[1],  # Normalized y position
                self._compute_aspect_ratio(y_coords, x_coords),  # Aspect ratio
                self._count_wall_neighbors(region_mask, walls) / region_size,  # Wall exposure
                float(has_door),
                float(has_exit),
                self._compute_compactness(region_mask),  # Compactness measure
            ]

            nodes.append(features)
            node_positions.append((centroid_y, centroid_x))

            # Determine node type
            if has_exit:
                node_types.append(3)  # Exit
            elif has_door:
                node_types.append(2)  # Door
            elif self._is_corridor(region_mask):
                node_types.append(1)  # Corridor
            else:
                node_types.append(0)  # Room

        # Add explicit door and exit nodes
        door_positions = list(zip(*np.where(doors > 0.5)))
        exit_positions = list(zip(*np.where(exits > 0.5)))

        for pos in door_positions:
            features = self._create_point_features(pos, grid_np, 'door')
            nodes.append(features)
            node_types.append(2)
            node_positions.append(pos)

        for pos in exit_positions:
            features = self._create_point_features(pos, grid_np, 'exit')
            nodes.append(features)
            node_types.append(3)
            node_positions.append(pos)

        if len(nodes) == 0:
            # Fallback: create single node from entire passable area
            nodes = [[0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5]]
            node_types = [0]
            node_positions = [(grid_np.shape[1] // 2, grid_np.shape[2] // 2)]

        # Create edges based on spatial proximity and connectivity
        edges = []
        edge_attrs = []

        for i in range(len(node_positions)):
            for j in range(i + 1, len(node_positions)):
                pos_i = node_positions[i]
                pos_j = node_positions[j]

                distance = np.sqrt(
                    (pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2
                )

                # Check if connected (no wall between)
                if self._is_connected(pos_i, pos_j, walls, passable):
                    edges.append([i, j])
                    edges.append([j, i])  # Bidirectional

                    # Edge features
                    attr = [
                        distance / max(grid_np.shape[1], grid_np.shape[2]),  # Normalized distance
                        self._path_quality(pos_i, pos_j, passable, walls),  # Path quality
                    ]
                    edge_attrs.append(attr)
                    edge_attrs.append(attr)  # Same for reverse edge

        # Convert to tensors
        node_features = torch.tensor(nodes, dtype=torch.float32)
        node_types_tensor = torch.tensor(node_types, dtype=torch.long)

        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            # Self-loop for single node
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        return FloorPlanGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_types=node_types_tensor,
            num_nodes=len(nodes),
        )

    def _compute_aspect_ratio(self, y_coords: np.ndarray, x_coords: np.ndarray) -> float:
        """Compute aspect ratio of region."""
        if len(y_coords) == 0:
            return 1.0
        height = y_coords.max() - y_coords.min() + 1
        width = x_coords.max() - x_coords.min() + 1
        return min(height, width) / max(height, width)

    def _count_wall_neighbors(self, region_mask: np.ndarray, walls: np.ndarray) -> int:
        """Count wall pixels adjacent to region."""
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(region_mask)
        boundary = dilated & ~region_mask
        return int((walls[boundary] > 0.5).sum())

    def _compute_compactness(self, region_mask: np.ndarray) -> float:
        """Compute compactness (circularity) of region."""
        area = region_mask.sum()
        if area == 0:
            return 0.0

        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(region_mask)
        perimeter = (dilated & ~region_mask).sum()

        if perimeter == 0:
            return 1.0

        # Circularity formula: 4 * pi * area / perimeter^2
        return min(1.0, 4 * np.pi * area / (perimeter ** 2))

    def _is_corridor(self, region_mask: np.ndarray) -> bool:
        """Heuristic to detect if region is a corridor."""
        y_coords, x_coords = np.where(region_mask)
        if len(y_coords) < 4:
            return False

        aspect = self._compute_aspect_ratio(y_coords, x_coords)
        compactness = self._compute_compactness(region_mask)

        # Corridors are elongated (low aspect ratio) and less compact
        return aspect < 0.3 or compactness < 0.3

    def _create_point_features(
        self,
        pos: Tuple[int, int],
        grid_np: np.ndarray,
        point_type: str
    ) -> List[float]:
        """Create features for a point node (door/exit)."""
        y, x = pos
        h, w = grid_np.shape[1], grid_np.shape[2]

        return [
            1.0 / (h * w),  # Minimal area (point)
            x / w,  # Normalized x
            y / h,  # Normalized y
            1.0,  # Aspect ratio (point)
            0.0,  # Wall exposure
            1.0 if point_type == 'door' else 0.0,
            1.0 if point_type == 'exit' else 0.0,
            1.0,  # Compactness (point)
        ]

    def _is_connected(
        self,
        pos_i: Tuple[float, float],
        pos_j: Tuple[float, float],
        walls: np.ndarray,
        passable: np.ndarray,
    ) -> bool:
        """Check if two positions are connected (no wall blocking)."""
        # Simple line-of-sight check
        y1, x1 = int(pos_i[0]), int(pos_i[1])
        y2, x2 = int(pos_j[0]), int(pos_j[1])

        # Bresenham-like sampling along line
        n_samples = max(abs(y2 - y1), abs(x2 - x1)) + 1
        if n_samples <= 1:
            return True

        ys = np.linspace(y1, y2, n_samples).astype(int)
        xs = np.linspace(x1, x2, n_samples).astype(int)

        # Clip to valid range
        ys = np.clip(ys, 0, walls.shape[0] - 1)
        xs = np.clip(xs, 0, walls.shape[1] - 1)

        # Check for walls along path
        wall_count = (walls[ys, xs] > 0.5).sum()
        return wall_count < n_samples * 0.2  # Allow some tolerance

    def _path_quality(
        self,
        pos_i: Tuple[float, float],
        pos_j: Tuple[float, float],
        passable: np.ndarray,
        walls: np.ndarray,
    ) -> float:
        """Estimate path quality between two positions."""
        y1, x1 = int(pos_i[0]), int(pos_i[1])
        y2, x2 = int(pos_j[0]), int(pos_j[1])

        n_samples = max(abs(y2 - y1), abs(x2 - x1)) + 1
        if n_samples <= 1:
            return 1.0

        ys = np.linspace(y1, y2, n_samples).astype(int)
        xs = np.linspace(x1, x2, n_samples).astype(int)

        ys = np.clip(ys, 0, passable.shape[0] - 1)
        xs = np.clip(xs, 0, passable.shape[1] - 1)

        passable_count = (passable[ys, xs] > 0.5).sum()
        return passable_count / n_samples


class GATv2Layer(nn.Module):
    """
    GATv2 (Graph Attention Network v2) layer.

    Improved attention mechanism that is more expressive than GAT v1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat

        # Linear transformations
        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, out_features))

        # Edge feature projection (optional)
        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, heads * out_features, bias=False)
        else:
            self.edge_proj = None

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (N, in_features)
            edge_index: Edge connectivity (2, E)
            edge_attr: Edge features (E, edge_dim)

        Returns:
            Updated node features (N, heads * out_features) or (N, out_features)
        """
        N = x.size(0)
        H = self.heads
        C = self.out_features

        # Linear transformation
        x_proj = self.W(x).view(N, H, C)  # (N, H, C)

        # Get source and target nodes
        src, dst = edge_index[0], edge_index[1]

        # GATv2 attention: a^T * LeakyReLU(W*h_i + W*h_j)
        x_src = x_proj[src]  # (E, H, C)
        x_dst = x_proj[dst]  # (E, H, C)

        # Add edge features if available
        if self.edge_proj is not None and edge_attr is not None:
            edge_feat = self.edge_proj(edge_attr).view(-1, H, C)
            alpha = (self.att * self.leaky_relu(x_src + x_dst + edge_feat)).sum(dim=-1)
        else:
            alpha = (self.att * self.leaky_relu(x_src + x_dst)).sum(dim=-1)  # (E, H)

        # Softmax over neighbors
        alpha = self._softmax(alpha, dst, N)
        alpha = self.dropout(alpha)

        # Aggregate
        out = torch.zeros(N, H, C, device=x.device)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, H, C), alpha.unsqueeze(-1) * x_src)

        if self.concat:
            return out.view(N, H * C)
        else:
            return out.mean(dim=1)

    def _softmax(
        self,
        alpha: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Compute softmax over neighbors."""
        alpha_max = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_max.scatter_reduce_(0, index.view(-1, 1).expand(-1, alpha.size(1)),
                                   alpha, reduce='amax', include_self=False)
        alpha = alpha - alpha_max[index]
        alpha_exp = alpha.exp()

        alpha_sum = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_sum.scatter_add_(0, index.view(-1, 1).expand(-1, alpha.size(1)), alpha_exp)
        alpha_sum = alpha_sum[index]

        return alpha_exp / (alpha_sum + 1e-8)


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE layer with mean aggregation.

    Samples and aggregates features from local neighborhoods.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize

        self.linear = nn.Linear(in_features * 2, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (N, in_features)
            edge_index: Edge connectivity (2, E)

        Returns:
            Updated node features (N, out_features)
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Mean aggregation of neighbors
        neighbor_sum = torch.zeros(N, self.in_features, device=x.device)
        neighbor_count = torch.zeros(N, 1, device=x.device)

        neighbor_sum.scatter_add_(0, dst.view(-1, 1).expand(-1, self.in_features), x[src])
        neighbor_count.scatter_add_(0, dst.view(-1, 1), torch.ones(src.size(0), 1, device=x.device))

        neighbor_mean = neighbor_sum / (neighbor_count + 1e-8)

        # Concatenate self and neighbor features
        out = torch.cat([x, neighbor_mean], dim=-1)
        out = self.linear(out)
        out = F.relu(out)
        out = self.dropout(out)

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out


class GINLayer(nn.Module):
    """
    Graph Isomorphism Network layer.

    Most expressive GNN layer for distinguishing graph structures.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Optional[int] = None,
        eps: float = 0.0,
        train_eps: bool = True,
    ):
        super().__init__()

        hidden_features = hidden_features or out_features

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer('eps', torch.tensor([eps]))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (N, in_features)
            edge_index: Edge connectivity (2, E)

        Returns:
            Updated node features (N, out_features)
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Sum aggregation
        neighbor_sum = torch.zeros(N, x.size(1), device=x.device)
        neighbor_sum.scatter_add_(0, dst.view(-1, 1).expand(-1, x.size(1)), x[src])

        # GIN update: MLP((1 + eps) * x + neighbor_sum)
        out = (1 + self.eps) * x + neighbor_sum
        return self.mlp(out)


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for floor plans.

    Converts grid to graph, applies GNN layers, and produces
    a fixed-size latent vector via graph-level pooling.
    """

    def __init__(
        self,
        node_features: int = 8,
        edge_features: int = 2,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 3,
        gnn_type: str = 'gatv2',
        heads: int = 4,
        dropout: float = 0.1,
        pooling: str = 'mean_max',
    ):
        """
        Initialize GNN encoder.

        Args:
            node_features: Input node feature dimension
            edge_features: Edge feature dimension
            hidden_dim: Hidden layer dimension
            latent_dim: Output latent dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN layer ('gatv2', 'sage', 'gin')
            heads: Number of attention heads (for GAT)
            dropout: Dropout rate
            pooling: Pooling method ('mean', 'max', 'mean_max', 'attention')
        """
        super().__init__()

        self.gnn_type = gnn_type
        self.pooling = pooling
        self.latent_dim = latent_dim

        # Grid to graph converter
        self.converter = GridToGraphConverter()

        # Node feature projection
        self.node_proj = nn.Linear(node_features, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else (hidden_dim * heads if gnn_type == 'gatv2' else hidden_dim)
            out_dim = hidden_dim

            if gnn_type == 'gatv2':
                layer = GATv2Layer(
                    in_dim, out_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=(i < num_layers - 1),
                    edge_dim=edge_features if i == 0 else None,
                )
            elif gnn_type == 'sage':
                layer = GraphSAGELayer(in_dim, out_dim, dropout=dropout)
            elif gnn_type == 'gin':
                layer = GINLayer(in_dim, out_dim)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.gnn_layers.append(layer)

        # Pooling layer
        if pooling == 'attention':
            pool_in_dim = hidden_dim * heads if gnn_type == 'gatv2' else hidden_dim
            self.pool_attention = nn.Sequential(
                nn.Linear(pool_in_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        # Final projection
        pool_out_dim = hidden_dim * 2 if pooling == 'mean_max' else hidden_dim
        if gnn_type == 'gatv2' and pooling != 'mean_max':
            pool_out_dim = hidden_dim * heads

        self.output_proj = nn.Sequential(
            nn.Linear(pool_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            grid: Floor plan grid (B, 5, H, W)

        Returns:
            Latent vectors (B, latent_dim)
        """
        batch_size = grid.size(0)
        device = grid.device

        latents = []

        for b in range(batch_size):
            # Convert grid to graph
            graph = self.converter.convert(grid[b])

            # Move to device
            x = graph.node_features.to(device)
            edge_index = graph.edge_index.to(device)
            edge_attr = graph.edge_attr.to(device) if graph.edge_attr is not None else None

            # Project node features
            x = self.node_proj(x)

            # Apply GNN layers
            for i, layer in enumerate(self.gnn_layers):
                if i == 0:
                    x = layer(x, edge_index, edge_attr)
                else:
                    x = layer(x, edge_index)

            # Graph-level pooling
            if self.pooling == 'mean':
                pooled = x.mean(dim=0)
            elif self.pooling == 'max':
                pooled = x.max(dim=0)[0]
            elif self.pooling == 'mean_max':
                pooled = torch.cat([x.mean(dim=0), x.max(dim=0)[0]], dim=-1)
            elif self.pooling == 'attention':
                weights = F.softmax(self.pool_attention(x), dim=0)
                pooled = (weights * x).sum(dim=0)
            else:
                pooled = x.mean(dim=0)

            latents.append(pooled)

        # Stack batch
        latents = torch.stack(latents, dim=0)

        # Final projection
        return self.output_proj(latents)


class HybridEncoder(nn.Module):
    """
    Hybrid encoder combining CNN and GNN representations.

    Uses CNN for local texture features and GNN for global topology.
    """

    def __init__(
        self,
        cnn_encoder: nn.Module,
        gnn_config: Optional[Dict] = None,
        fusion: str = 'concat',
        latent_dim: int = 64,
    ):
        """
        Initialize hybrid encoder.

        Args:
            cnn_encoder: Pre-existing CNN encoder (FloorPlanEncoder)
            gnn_config: Configuration for GNN encoder
            fusion: Fusion method ('concat', 'add', 'attention')
            latent_dim: Final latent dimension
        """
        super().__init__()

        self.cnn_encoder = cnn_encoder
        self.fusion = fusion
        self.latent_dim = latent_dim

        # GNN encoder
        gnn_config = gnn_config or {}
        self.gnn_encoder = GNNEncoder(
            latent_dim=cnn_encoder.latent_dim,
            **gnn_config,
        )

        # Fusion layer
        if fusion == 'concat':
            self.fuse = nn.Linear(cnn_encoder.latent_dim * 2, latent_dim)
        elif fusion == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(cnn_encoder.latent_dim * 2, cnn_encoder.latent_dim),
                nn.Tanh(),
                nn.Linear(cnn_encoder.latent_dim, 2),
                nn.Softmax(dim=-1),
            )
            self.fuse = nn.Linear(cnn_encoder.latent_dim, latent_dim)
        else:
            self.fuse = nn.Linear(cnn_encoder.latent_dim, latent_dim)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            grid: Floor plan grid (B, 5, H, W)

        Returns:
            Latent vectors (B, latent_dim)
        """
        # CNN features
        cnn_latent = self.cnn_encoder(grid)

        # GNN features
        gnn_latent = self.gnn_encoder(grid)

        # Fusion
        if self.fusion == 'concat':
            combined = torch.cat([cnn_latent, gnn_latent], dim=-1)
            return self.fuse(combined)
        elif self.fusion == 'add':
            return self.fuse(cnn_latent + gnn_latent)
        elif self.fusion == 'attention':
            combined = torch.cat([cnn_latent, gnn_latent], dim=-1)
            weights = self.attention(combined)
            weighted = weights[:, 0:1] * cnn_latent + weights[:, 1:2] * gnn_latent
            return self.fuse(weighted)
        else:
            return self.fuse(cnn_latent + gnn_latent)


def create_gnn_encoder(
    config_or_latent_dim: Union['RankingV2Config', int] = 64,
    gnn_type: str = 'gatv2',
    **kwargs,
) -> GNNEncoder:
    """
    Factory function to create GNN encoder.

    Args:
        config_or_latent_dim: Config object or latent dimension
        gnn_type: Type of GNN ('gatv2', 'sage', 'gin')
        **kwargs: Additional arguments for GNNEncoder

    Returns:
        GNNEncoder instance
    """
    if hasattr(config_or_latent_dim, 'latent_dim'):
        latent_dim = config_or_latent_dim.latent_dim
    else:
        latent_dim = config_or_latent_dim

    return GNNEncoder(
        latent_dim=latent_dim,
        gnn_type=gnn_type,
        **kwargs,
    )