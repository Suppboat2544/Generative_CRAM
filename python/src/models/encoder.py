"""
Deep CRAM Encoder
=================
Graph-based encoder using Graph Attention Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

try:
    import dgl
    from dgl.nn import GATConv
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False


class DeepCRAMEncoder(nn.Module):
    """
    Graph-based encoder using Graph Attention Networks.

    Takes molecular graphs with 163D atom features and produces
    latent representations.

    Returns dict with 'mu', 'logvar', 'graph_repr' for compatibility
    with PropertyConditionedCRAMVAE.
    """

    def __init__(
        self,
        in_dim: int = 163,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        # Legacy parameter names for backward compatibility
        num_heads: int = None,
        num_layers: int = None
    ):
        super().__init__()

        if not DGL_AVAILABLE:
            raise ImportError("DGL is required for DeepCRAMEncoder")

        # Handle legacy parameter names
        n_heads = num_heads if num_heads is not None else n_heads
        n_layers = num_layers if num_layers is not None else n_layers

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(n_layers):
            self.gat_layers.append(
                GATConv(
                    in_feats=hidden_dim,
                    out_feats=hidden_dim // n_heads,
                    num_heads=n_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    residual=True,
                    activation=nn.GELU()
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Global attention pooling
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Latent projections
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights to prevent NaN during training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, g) -> Dict[str, torch.Tensor]:
        """
        Encode molecular graph to latent distribution.

        Args:
            g: DGL graph with 'feat' node features (N, in_dim)

        Returns:
            dict with 'mu', 'logvar', 'graph_repr'
        """
        # Get node features
        h = g.ndata['feat']  # (N, in_dim)

        # Input projection
        h = self.input_proj(h)  # (N, hidden_dim)

        # GAT message passing
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            h_new = gat(g, h)  # (N, n_heads, hidden_dim // n_heads)
            h_new = h_new.flatten(1)  # (N, hidden_dim)
            h = ln(h + h_new)  # Residual + LayerNorm

        # Global attention pooling
        with g.local_scope():
            gate = self.pool_gate(h)  # (N, 1)
            g.ndata['h'] = h * gate
            graph_repr = dgl.mean_nodes(g, 'h')  # (batch, hidden_dim)

        # Latent projections
        mu = self.fc_mu(graph_repr)
        logvar = self.fc_logvar(graph_repr)

        # Clamp logvar to prevent numerical issues
        logvar = torch.clamp(logvar, min=-10, max=2)

        return {
            'mu': mu,
            'logvar': logvar,
            'graph_repr': graph_repr
        }
