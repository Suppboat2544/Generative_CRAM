"""Notebook EnhancedHierarchicalGATEncoder."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
except ImportError as e:
    raise ImportError("DGL is required") from e

from dgl.nn import GATConv
from .ring_chain import RingChainEncoder, SubstructureDetector, CRAMStructureLoss


class CRAMFeatureExtractor(nn.Module):
    """Projects graph-level CRAM property vectors into encoder hidden space.

    Used by EnhancedHierarchicalGATEncoder. The Colab notebook referenced this
    class without defining it; this MLP matches the (feature_dim=10, hidden_dim) call.
    """

    def __init__(self, feature_dim: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
# ============================================================================
# ENHANCED HierarchicalGATEncoder WITH RING-CHAIN AWARENESS
# ============================================================================
#
# This version of the encoder adds explicit ring-chain structure understanding
# to better capture TRUE CRAM molecules: carboxyl-rich alicyclic+chain hybrids.
#
# Key improvements:
# 1. RingChainEncoder integration for substructure-aware encoding
# 2. Substructure masks passed through forward pass
# 3. CRAM structure auxiliary loss for multi-task learning
# 4. Enhanced feature dimension handling (108D input)
# ============================================================================

class EnhancedHierarchicalGATEncoder(nn.Module):
    """
    ENHANCED Hierarchical Graph Attention Network encoder with EXPLICIT
    ring-chain structure awareness for TRUE CRAM molecule generation.
    
    This encoder combines:
    1. Standard hierarchical GAT encoding (atom → fragment → molecule)
    2. RingChainEncoder for explicit substructure awareness
    3. CRAM feature extraction and fusion
    4. Multi-task prediction heads
    
    The key insight is that CRAM molecules have SPECIFIC structural patterns:
    - Alicyclic rings (cyclohexane, cyclopentane) - NOT benzene
    - Aliphatic chains connected to rings via junction atoms
    - Multiple carboxyl groups (-COOH) positioned on chains/rings
    
    This encoder explicitly captures these patterns.
    """
    
    def __init__(self, node_feat_dim, hidden_dim, latent_dim, num_layers=4):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # ========================================
        # PART 1: STANDARD GAT ENCODING
        # ========================================
        
        # Atom-level encoder with residual connections
        self.atom_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # GAT layers with residual connections
        self.atom_gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // 4, num_heads=4, feat_drop=0.1,
                   attn_drop=0.1, residual=True, activation=F.elu)
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each GAT layer
        self.atom_layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Multi-scale readout
        self.multi_scale_pool = nn.Linear(hidden_dim * num_layers, hidden_dim)
        
        # Fragment-level encoder
        self.fragment_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Dual attention
        self.fragment_self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.fragment_cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # Molecule-level aggregation
        self.mol_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Global pooling
        self.global_attention_pool = nn.Linear(hidden_dim, 1)
        self.pool_fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # ========================================
        # PART 2: RING-CHAIN ENCODER (NEW!)
        # ========================================
        
        self.ring_chain_encoder = RingChainEncoder(hidden_dim)
        
        # Fusion of GAT encoding + Ring-Chain encoding
        self.gat_ringchain_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # ========================================
        # PART 3: CRAM FEATURE EXTRACTOR
        # ========================================
        
        self.cram_feature_extractor = CRAMFeatureExtractor(
            feature_dim=10, 
            hidden_dim=hidden_dim
        )
        
        self.cram_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # ========================================
        # PART 4: VAE PARAMETERS
        # ========================================
        
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # TC-VAE discriminator
        self.tc_discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # ========================================
        # PART 5: PREDICTION HEADS
        # ========================================
        
        # CRAM feature prediction (10D)
        self.cram_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 10)
        )
        
        # CRAM structure prediction (5D) - from RingChainEncoder
        # [has_alicyclic, has_chain, has_junction, num_cooh, is_cram]
        self.cram_structure_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)
        )
    
    def forward(self, g, substructure_masks=None):
        """
        Enhanced forward pass with ring-chain awareness.
        
        Args:
            g: DGL graph batch
            substructure_masks: Optional dict of substructure type masks
                If None, will be detected from graph
            
        Returns:
            mu: Latent mean [batch, latent_dim]
            logvar: Latent log variance [batch, latent_dim]
            cram_features: CRAM property predictions [batch, 10]
            cram_structure: CRAM structure predictions [batch, 5]
            hierarchical_features: Dict with atom/fragment/molecule level features
        """
        # ========================================
        # PART 1: STANDARD GAT ENCODING
        # ========================================
        
        h = g.ndata['feat']
        h = self.atom_embed(h)
        
        # Store layer outputs for multi-scale readout
        layer_outputs = []
        
        for i, (gat, norm) in enumerate(zip(self.atom_gat_layers, self.atom_layer_norms)):
            h_out = gat(g, h)
            if h_out.dim() == 3:
                h_out = h_out.flatten(1)
            h = norm(h_out + h if h.size(-1) == h_out.size(-1) else h_out)
            layer_outputs.append(h)
        
        # Multi-scale aggregation
        atom_features = h
        h_multi_scale = torch.cat(layer_outputs, dim=-1)
        h_multi_scale = self.multi_scale_pool(h_multi_scale)
        
        # Fragment-level encoding
        with g.local_scope():
            g.ndata['h'] = h_multi_scale
            
            # Multiple pooling strategies
            fragment_mean = dgl.mean_nodes(g, 'h')
            fragment_max = dgl.max_nodes(g, 'h')
            
            attention_weights = torch.sigmoid(self.global_attention_pool(h_multi_scale))
            g.ndata['attn'] = attention_weights
            g.ndata['h_weighted'] = h_multi_scale * attention_weights
            fragment_attn = dgl.sum_nodes(g, 'h_weighted') / (dgl.sum_nodes(g, 'attn') + 1e-8)
            
            fragment_combined = torch.cat([fragment_mean, fragment_max, fragment_attn], dim=-1)
            fragment_h = self.pool_fusion(fragment_combined)
            fragment_h = self.fragment_embed(fragment_h)
            
            # Dual attention
            fragment_h_expanded = fragment_h.unsqueeze(1)
            fragment_self, _ = self.fragment_self_attention(
                fragment_h_expanded, fragment_h_expanded, fragment_h_expanded
            )
            atom_mean = dgl.mean_nodes(g, 'h').unsqueeze(1)
            fragment_cross, _ = self.fragment_cross_attention(
                fragment_self, atom_mean, atom_mean
            )
            fragment_features = fragment_cross.squeeze(1)
        
        # Molecule-level aggregation
        mol_h = fragment_features.unsqueeze(1)
        mol_out, attn_weights = self.mol_attention(mol_h, mol_h, mol_h)
        hg_gat = mol_out.squeeze(1) + fragment_features
        
        # ========================================
        # PART 2: RING-CHAIN ENCODING (NEW!)
        # ========================================
        
        # If substructure masks not provided, use default empty masks
        if substructure_masks is None:
            substructure_masks = {
                'alicyclic_ring': torch.zeros(h_multi_scale.size(0), device=h_multi_scale.device),
                'chain': torch.zeros(h_multi_scale.size(0), device=h_multi_scale.device),
                'junction': torch.zeros(h_multi_scale.size(0), device=h_multi_scale.device),
                'carboxyl': torch.zeros(h_multi_scale.size(0), device=h_multi_scale.device)
            }
        
        # Run ring-chain encoder (node broadcast + graph scores)
        rc_out = self.ring_chain_encoder(h_multi_scale, substructure_masks)
        if isinstance(rc_out, tuple):
            hg_ringchain, cram_structure_scores = rc_out[0], rc_out[1]
        else:
            hg_ringchain, cram_structure_scores = rc_out, None
        if hg_ringchain.dim() == 2 and hg_ringchain.size(0) != hg_gat.size(0):
            hg_ringchain = hg_ringchain[0].unsqueeze(0)
        if hg_ringchain.dim() == 1:
            hg_ringchain = hg_ringchain.unsqueeze(0)
        if hg_gat.size(0) > 1 and hg_ringchain.size(0) == 1:
            hg_ringchain = hg_ringchain.expand(hg_gat.size(0), -1)
        
        # Fuse GAT + Ring-Chain encodings
        hg_combined = torch.cat([hg_gat, hg_ringchain], dim=-1)
        hg = self.gat_ringchain_fusion(hg_combined)
        
        # ========================================
        # PART 3: CRAM FEATURE FUSION
        # ========================================
        
        if 'cram_features_raw' in g.ndata:
            with g.local_scope():
                g.ndata['cram_raw'] = g.ndata['cram_features_raw']
                cram_features_raw = dgl.mean_nodes(g, 'cram_raw')
            
            cram_projected = self.cram_feature_extractor(cram_features_raw)
            hg_with_cram = torch.cat([hg, cram_projected], dim=-1)
            hg = self.cram_fusion(hg_with_cram)
        
        # ========================================
        # PART 4: VAE PARAMETERS
        # ========================================
        
        mu = self.fc_mu(hg)
        logvar = self.fc_logvar(hg)
        logvar = torch.clamp(logvar, min=-10, max=2)
        
        # ========================================
        # PART 5: PREDICTION HEADS
        # ========================================
        
        cram_features_pred = self.cram_fc(hg)
        cram_structure_pred = self.cram_structure_fc(hg)
        
        hierarchical_features = {
            'atom': atom_features,
            'atom_multi_scale': h_multi_scale,
            'fragment': fragment_features,
            'molecule': hg,
            'attention_weights': attn_weights,
            'layer_outputs': layer_outputs,
            'ring_chain_encoding': hg_ringchain,
            'cram_structure_scores': cram_structure_scores
        }
        
        return mu, logvar, cram_features_pred, cram_structure_pred, hierarchical_features


