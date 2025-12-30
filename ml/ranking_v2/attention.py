"""
Cross-Attention Modules for Ranking V2

Implements bidirectional cross-attention between two configurations:
- A attends to B: What aspects of B are relevant for scoring A?
- B attends to A: What aspects of A are relevant for scoring B?

Key Design:
    - Operates on latent vectors (not feature maps) for efficiency
    - Bidirectional attention enables learning relative differences
    - Residual connections preserve original information
    - LayerNorm for stable training
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """
    Bidirectional cross-attention layer for comparing two configurations.

    Given latent vectors for configs A and B:
    - A attends to B: output_a = A + Attention(Q=A, K=B, V=B)
    - B attends to A: output_b = B + Attention(Q=B, K=A, V=A)

    This allows each configuration to "see" what's relevant about the other,
    enabling the model to learn relative comparisons.

    Attributes:
        dim: Input/output dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-attention layer.

        Args:
            dim: Input/output dimension
            num_heads: Number of attention heads
            dropout: Dropout rate for attention weights
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections for A attending to B
        self.q_proj_a = nn.Linear(dim, dim)
        self.k_proj_b = nn.Linear(dim, dim)
        self.v_proj_b = nn.Linear(dim, dim)
        self.out_proj_a = nn.Linear(dim, dim)

        # Projections for B attending to A
        self.q_proj_b = nn.Linear(dim, dim)
        self.k_proj_a = nn.Linear(dim, dim)
        self.v_proj_a = nn.Linear(dim, dim)
        self.out_proj_b = nn.Linear(dim, dim)

        # Normalization and dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_a = nn.LayerNorm(dim)
        self.layer_norm_b = nn.LayerNorm(dim)

        # Store attention weights for visualization
        self.attention_weights_a: Optional[torch.Tensor] = None
        self.attention_weights_b: Optional[torch.Tensor] = None

    def forward(
        self,
        latent_a: torch.Tensor,
        latent_b: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-attention between A and B.

        Args:
            latent_a: Latent vector for config A, shape (B, D)
            latent_b: Latent vector for config B, shape (B, D)
            return_attention: Whether to store attention weights

        Returns:
            Tuple of (attended_a, attended_b), each shape (B, D)
        """
        B = latent_a.size(0)

        # === A attends to B ===
        # Query from A, Key and Value from B
        q_a = self.q_proj_a(latent_a).view(B, 1, self.num_heads, self.head_dim)
        k_b = self.k_proj_b(latent_b).view(B, 1, self.num_heads, self.head_dim)
        v_b = self.v_proj_b(latent_b).view(B, 1, self.num_heads, self.head_dim)

        # Scaled dot-product attention: (B, H, 1, 1)
        # Since we have single vectors (not sequences), the attention is simple
        attn_a = torch.einsum('bqhd,bkhd->bhqk', q_a, k_b) * self.scale
        attn_a = F.softmax(attn_a, dim=-1)
        attn_a = self.dropout(attn_a)

        if return_attention:
            self.attention_weights_a = attn_a.detach()

        # Apply attention to values
        out_a = torch.einsum('bhqk,bkhd->bqhd', attn_a, v_b)
        out_a = out_a.reshape(B, self.dim)
        out_a = self.out_proj_a(out_a)

        # === B attends to A ===
        # Query from B, Key and Value from A
        q_b = self.q_proj_b(latent_b).view(B, 1, self.num_heads, self.head_dim)
        k_a = self.k_proj_a(latent_a).view(B, 1, self.num_heads, self.head_dim)
        v_a = self.v_proj_a(latent_a).view(B, 1, self.num_heads, self.head_dim)

        attn_b = torch.einsum('bqhd,bkhd->bhqk', q_b, k_a) * self.scale
        attn_b = F.softmax(attn_b, dim=-1)
        attn_b = self.dropout(attn_b)

        if return_attention:
            self.attention_weights_b = attn_b.detach()

        out_b = torch.einsum('bhqk,bkhd->bqhd', attn_b, v_a)
        out_b = out_b.reshape(B, self.dim)
        out_b = self.out_proj_b(out_b)

        # Residual connection + LayerNorm
        attended_a = self.layer_norm_a(latent_a + out_a)
        attended_b = self.layer_norm_b(latent_b + out_b)

        return attended_a, attended_b


class FeedForward(nn.Module):
    """
    Feed-forward network for transformer-style processing.

    Two-layer MLP with GELU activation and dropout.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize feed-forward network.

        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (default: 4 * dim)
            dropout: Dropout rate
        """
        super().__init__()

        hidden_dim = hidden_dim or dim * 4

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.norm(x + self.net(x))


class CrossAttentionBlock(nn.Module):
    """
    Full cross-attention block: Attention + FFN.

    Combines cross-attention with feed-forward network for
    richer feature transformation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_ffn: bool = True,
        ffn_hidden_dim: Optional[int] = None,
    ):
        """
        Initialize cross-attention block.

        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_ffn: Whether to include feed-forward network
            ffn_hidden_dim: FFN hidden dimension
        """
        super().__init__()

        self.attention = CrossAttentionLayer(dim, num_heads, dropout)

        if use_ffn:
            self.ffn_a = FeedForward(dim, ffn_hidden_dim, dropout)
            self.ffn_b = FeedForward(dim, ffn_hidden_dim, dropout)
        else:
            self.ffn_a = None
            self.ffn_b = None

    def forward(
        self,
        latent_a: torch.Tensor,
        latent_b: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention and optional FFN.

        Args:
            latent_a: Latent vector for A, shape (B, D)
            latent_b: Latent vector for B, shape (B, D)
            return_attention: Whether to store attention weights

        Returns:
            Tuple of processed (latent_a, latent_b)
        """
        # Cross-attention
        attended_a, attended_b = self.attention(latent_a, latent_b, return_attention)

        # Feed-forward (if enabled)
        if self.ffn_a is not None:
            attended_a = self.ffn_a(attended_a)
            attended_b = self.ffn_b(attended_b)

        return attended_a, attended_b


class CrossAttentionStack(nn.Module):
    """
    Stack of cross-attention blocks.

    Multiple layers of cross-attention allow for deeper interaction
    between the two configurations.

    Attributes:
        layers: List of CrossAttentionBlock modules
        num_layers: Number of stacked layers
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_ffn: bool = True,
        ffn_hidden_dim: Optional[int] = None,
    ):
        """
        Initialize cross-attention stack.

        Args:
            dim: Feature dimension
            num_layers: Number of stacked attention blocks
            num_heads: Number of attention heads per block
            dropout: Dropout rate
            use_ffn: Whether to include FFN in each block
            ffn_hidden_dim: FFN hidden dimension
        """
        super().__init__()

        self.layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                use_ffn=use_ffn,
                ffn_hidden_dim=ffn_hidden_dim,
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(
        self,
        latent_a: torch.Tensor,
        latent_b: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all attention blocks.

        Args:
            latent_a: Latent vector for A, shape (B, D)
            latent_b: Latent vector for B, shape (B, D)
            return_attention: Whether to store attention weights

        Returns:
            Tuple of processed (latent_a, latent_b)
        """
        for layer in self.layers:
            latent_a, latent_b = layer(latent_a, latent_b, return_attention)

        return latent_a, latent_b

    def get_attention_weights(self) -> Tuple[list, list]:
        """
        Get stored attention weights from all layers.

        Returns:
            Tuple of (weights_a_list, weights_b_list) for each layer
        """
        weights_a = []
        weights_b = []

        for layer in self.layers:
            if hasattr(layer.attention, 'attention_weights_a'):
                weights_a.append(layer.attention.attention_weights_a)
            if hasattr(layer.attention, 'attention_weights_b'):
                weights_b.append(layer.attention.attention_weights_b)

        return weights_a, weights_b


class DifferenceAttention(nn.Module):
    """
    Alternative attention mechanism that explicitly models differences.

    Instead of standard cross-attention, this module:
    1. Computes diff = A - B
    2. Attends to the difference
    3. Uses difference-weighted features

    This is more directly aligned with the ranking objective.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize difference attention.

        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Project difference to attention queries
        self.diff_proj = nn.Linear(dim, dim)

        # Project original features to keys and values
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

    def forward(
        self,
        latent_a: torch.Tensor,
        latent_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with difference-based attention.

        Args:
            latent_a: Latent vector for A, shape (B, D)
            latent_b: Latent vector for B, shape (B, D)

        Returns:
            Tuple of processed (latent_a, latent_b)
        """
        B = latent_a.size(0)

        # Compute difference
        diff = latent_a - latent_b

        # Query from difference, Key/Value from concatenated features
        q = self.diff_proj(diff).view(B, 1, self.num_heads, self.head_dim)

        # Stack A and B features
        combined = torch.stack([latent_a, latent_b], dim=1)  # (B, 2, D)
        k = self.k_proj(combined).view(B, 2, self.num_heads, self.head_dim)
        v = self.v_proj(combined).view(B, 2, self.num_heads, self.head_dim)

        # Attention over A and B features weighted by difference
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bqhd,bkhd->bhqk', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted combination
        out = torch.einsum('bhqk,bkhd->bqhd', attn, v)
        out = out.reshape(B, self.dim)
        out = self.out_proj(out)

        # Add to both A and B (with opposite signs to emphasize difference)
        attended_a = self.norm_a(latent_a + out)
        attended_b = self.norm_b(latent_b - out)

        return attended_a, attended_b


def create_attention_module(config) -> Optional[nn.Module]:
    """
    Factory function to create attention module from config.

    Args:
        config: RankingV2Config with attention parameters

    Returns:
        CrossAttentionStack or None if attention is disabled
    """
    if not config.use_cross_attention:
        return None

    return CrossAttentionStack(
        dim=config.attention_dim,
        num_layers=config.num_attention_layers,
        num_heads=config.attention_heads,
        dropout=config.attention_dropout,
        use_ffn=config.use_attention_ffn,
    )
