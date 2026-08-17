"""Notebook production DeepCRAMEncoder (GAT + ring-chain + FG attention).

Used by DeepCRAMVAE / sweep variants. The GRU CVAE uses src.models.encoder.DeepCRAMEncoder.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
except ImportError as e:
    raise ImportError("DGL is required for DeepCRAMEncoder") from e

from .ring_chain import RingChainEncoder
from .functional_group_attention import FunctionalGroupAttention
from .contrastive import ContrastiveCRAMLearning

"""
=============================================================================
PRODUCTION DEEP CRAM ENCODER: COMPLETE INTEGRATION
=============================================================================

This encoder represents the culmination of all deep improvements:

Architecture:
┌─────────────────────────────────────────────────────────────┐
│  Input: 163D atom features + molecular features             │
├─────────────────────────────────────────────────────────────┤
│  1. Initial Projection (163D → 256D)                        │
├─────────────────────────────────────────────────────────────┤
│  2. GAT Layers (4 layers × 4 heads)                         │
│     └── With residual connections                           │
├─────────────────────────────────────────────────────────────┤
│  3. Ring-Chain Encoder (parallel pathway)                   │
│     └── Specialized for alicyclic/aliphatic patterns        │
├─────────────────────────────────────────────────────────────┤
│  4. Functional Group Attention                              │
│     └── Within-group and between-group attention            │
├─────────────────────────────────────────────────────────────┤
│  5. Feature Fusion (GAT + Ring-Chain + FG)                  │
├─────────────────────────────────────────────────────────────┤
│  6. Graph Pooling → Molecular representation                │
├─────────────────────────────────────────────────────────────┤
│  7. Latent Space (μ, σ for VAE)                            │
├─────────────────────────────────────────────────────────────┤
│  8. Contrastive Projection Head (optional)                  │
├─────────────────────────────────────────────────────────────┤
│  9. Property Prediction Heads                               │
│     └── is_cram, cram_type, stability                       │
└─────────────────────────────────────────────────────────────┘
"""

class DeepCRAMEncoder(nn.Module):
    """
    Production-ready Deep CRAM Encoder.
    
    This is the complete encoder that captures:
    - Atomic-level chemistry (via enhanced features)
    - Ring/chain architecture (via specialized encoder)
    - Functional group relationships (via group attention)
    - CRAM-specific patterns (via multi-task heads)
    """
    
    def __init__(self, 
                 in_dim: int = 163,  # From DeepCRAMFeatureExtractor (with enhanced oxidation)
                 hidden_dim: int = 256,
                 latent_dim: int = 64,
                 num_gat_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_contrastive: bool = True,
                 num_layers: int = None,
                 **kwargs):
        super().__init__()
        if num_layers is not None:
            num_gat_layers = num_layers
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.use_contrastive = use_contrastive
        
        # === 1. Initial Projection ===
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # === 2. GAT Layers ===
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        
        for i in range(num_gat_layers):
            layer_in = hidden_dim
            layer_out = hidden_dim // num_heads
            
            self.gat_layers.append(
                dgl.nn.GATConv(
                    in_feats=layer_in,
                    out_feats=layer_out,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    residual=True,
                    activation=None
                )
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))
        
        # === 3. Ring-Chain Encoder (Parallel Pathway) ===
        self.ring_chain_encoder = RingChainEncoder(hidden_dim)
        
        # === 4. Functional Group Attention ===
        self.fg_attention = FunctionalGroupAttention(
            atom_dim=hidden_dim,
            group_dim=hidden_dim // 2,
            num_heads=num_heads,
            num_group_types=13
        )
        
        # === 5. Feature Fusion ===
        # Combine GAT, Ring-Chain, and FG pathways
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # === 6. Graph Pooling ===
        # Attention-weighted pooling for molecular representation
        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # === 7. Latent Space ===
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # === 8. Contrastive Projection Head ===
        if use_contrastive:
            self.contrastive_module = ContrastiveCRAMLearning(
                latent_dim=latent_dim,
                temperature=0.07,
                margin=1.0
            )
        
        # === 9. Property Prediction Heads ===
        self.property_heads = nn.ModuleDict({
            'is_cram': nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.GELU(),
                nn.Linear(latent_dim // 2, 1),
                nn.Sigmoid()
            ),
            'cram_type': nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.GELU(),
                nn.Linear(latent_dim // 2, 3)  # alicyclic, aliphatic, hybrid
            ),
            'num_cooh': nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.GELU(),
                nn.Linear(latent_dim // 2, 1),
                nn.ReLU()  # COOH count is non-negative
            ),
            'stability': nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.GELU(),
                nn.Linear(latent_dim // 2, 1),
                nn.Sigmoid()
            )
        })
        
        pass
    
    def forward(self, g: dgl.DGLGraph, 
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the deep CRAM encoder.
        
        Args:
            g: DGL graph with 'feat' and 'group' node data
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary containing:
            - 'mu': Latent mean
            - 'logvar': Latent log-variance
            - 'z': Sampled latent vector
            - 'predictions': Property predictions
            - 'node_features': Final node representations
            - 'attention_weights': (optional) Attention from GAT/FG
        """
        device = g.device
        h = g.ndata['feat']  # (num_nodes, 143)
        group_membership = g.ndata.get('group', None)  # (num_nodes, 13)
        
        # === 1. Initial Projection ===
        h = self.input_proj(h)  # (num_nodes, hidden_dim)
        
        # === 2. GAT Layers ===
        attention_weights = []
        h_gat = h
        for i, (gat_layer, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            try:
                h_new, attn = gat_layer(g, h_gat, get_attention=True)
            except TypeError:
                h_new = gat_layer(g, h_gat)
                attn = None
            h_new = h_new.view(h_new.shape[0], -1)  # Flatten heads
            h_gat = norm(h_gat + F.gelu(h_new))  # Residual + activation
            
            if return_attention and attn is not None:
                attention_weights.append(attn)
        
        # === 3. Ring-Chain Encoder (Parallel) ===
        def _mask(name):
            t = g.ndata.get(name, None)
            if t is None:
                return None
            return t.float() if t.dtype == torch.bool else t

        masks = {
            "alicyclic": _mask("mask_alicyclic_ring"),
            "alicyclic_ring": _mask("mask_alicyclic_ring"),
            "chain": _mask("mask_chain"),
            "junction": _mask("mask_junction"),
            "carboxyl": _mask("mask_carboxyl"),
        }
        h_ring_chain = self.ring_chain_encoder(h, masks)
        if isinstance(h_ring_chain, tuple):
            h_ring_chain = h_ring_chain[0]
        
        # === 4. Functional Group Attention ===
        if group_membership is not None:
            h_fg = self.fg_attention(
                h_gat.unsqueeze(0),  # Add batch dim
                group_membership.unsqueeze(0)
            ).squeeze(0)
        else:
            h_fg = h_gat
        
        # === 5. Feature Fusion ===
        h_combined = torch.cat([h_gat, h_ring_chain, h_fg], dim=-1)
        h_fused = self.fusion(h_combined)
        
        # === 6. Graph Pooling (batched-safe attention pool) ===
        with g.local_scope():
            g.ndata["h"] = h_fused
            g.ndata["a"] = self.pool_attention(h_fused)
            g.ndata["w"] = dgl.softmax_nodes(g, "a")
            g.ndata["hw"] = g.ndata["h"] * g.ndata["w"]
            graph_rep = dgl.sum_nodes(g, "hw")
        
        # === 7. Latent Space ===
        mu = self.fc_mu(graph_rep)
        logvar = self.fc_logvar(graph_rep)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # === 8. Property Predictions ===
        predictions = {}
        for name, head in self.property_heads.items():
            predictions[name] = head(z)
        
        # === Output ===
        output = {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'predictions': predictions,
            'node_features': h_fused,
            'graph_representation': graph_rep
        }
        
        if return_attention:
            output['attention_weights'] = attention_weights
        
        return output
    
    def encode(self, g: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method to get just mu and logvar."""
        output = self.forward(g)
        return output['mu'], output['logvar']
    
    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_contrastive_loss(self, z: torch.Tensor, 
                                  labels: torch.Tensor,
                                  cram_types: Optional[torch.Tensor] = None
                                  ) -> Dict[str, torch.Tensor]:
        """Compute contrastive losses for a batch."""
        if not self.use_contrastive:
            return {}
        return self.contrastive_module(z, labels, cram_types)

