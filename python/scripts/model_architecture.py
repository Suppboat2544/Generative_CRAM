"""
CRAM (Constrained Recurrent Autoencoder for Molecules) Generator
Complete Model Architecture and Components for Visualization

This file contains the complete architecture of the CRAM molecular generator
based on Transformer-VAE with Graph Neural Networks.

Architecture Overview:
======================
Input SMILES → Graph Encoder (GAT) → Latent Space (512D) → 
Transformer Decoder → SELFIES Output → Valid CRAM Molecules

Key Components:
1. GraphEncoder: GAT-based encoder with hierarchical pooling
2. TransformerDecoder: Autoregressive decoder with CRAM conditioning
3. TransformerVAE: Complete VAE with template conditioning
4. Feature Extraction: 73D/88D node features, 6D edge features
5. CRAM Scoring: Multi-criteria reward functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from typing import Tuple, Optional, List
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

class ModelConfig:
    """Model architecture configuration"""
    
    # Model dimensions
    NODE_FEAT_DIM = 73  # 73D for synthetic, 88D for unified
    EDGE_FEAT_DIM = 6
    HIDDEN_DIM = 1024
    LATENT_DIM = 512
    VOCAB_SIZE = 100  # SELFIES vocabulary size
    
    # Transformer settings
    NUM_LAYERS = 8
    NUM_HEADS = 8
    DROPOUT = 0.3
    
    # Sequence settings
    MAX_SEQ_LEN = 150
    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2
    
    # Training settings
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    BETA_START = 0.001  # KL annealing start
    BETA_END = 0.05     # KL annealing end
    
    # CRAM-specific settings
    MIN_CARBOXYL_GROUPS = 1
    MIN_ALIPHATIC_PERCENT = 70.0
    TARGET_MW_RANGE = (150, 450)
    TARGET_LOGP_RANGE = (2, 6)


# ============================================================================
# 1. GRAPH ENCODER
# ============================================================================

class GraphEncoder(nn.Module):
    """
    Enhanced Graph Neural Network encoder with hierarchical pooling.
    
    Architecture:
    - Input projection: node_feat_dim → hidden_dim
    - Edge projection: edge_feat_dim → hidden_dim
    - GAT layers (3): Multi-head graph attention with residual connections
    - Hierarchical pooling: Mean + Max + Attention-weighted
    - Output: mu, logvar for latent distribution + CRAM features
    
    Input:
        - Node features: [num_nodes, node_feat_dim] (73D or 88D)
        - Edge features: [num_edges, edge_feat_dim] (6D)
        
    Output:
        - mu: [batch, latent_dim] - mean of latent distribution
        - logvar: [batch, latent_dim] - log variance of latent distribution
        - cram_features: [batch, 2] - carboxyl count + aliphatic ratio
    """
    
    def __init__(
        self,
        node_feat_dim: int = 73,
        hidden_dim: int = 1024,
        latent_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Input projections
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_proj = nn.Linear(6, hidden_dim)  # 6D edge features
        
        # GAT layers with edge features
        self.gat_layers = nn.ModuleList([
            GATConv(
                hidden_dim, 
                hidden_dim // num_heads, 
                num_heads,
                feat_drop=dropout,
                attn_drop=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each GAT layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Hierarchical pooling
        # Attention-weighted pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Output to latent space
        # 3x hidden due to multi-scale pooling (mean + max + attention)
        self.fc_mu = nn.Linear(hidden_dim * 3, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 3, latent_dim)
        
    def forward(self, g):
        """
        Forward pass through graph encoder.
        
        Args:
            g: DGL graph with:
                - g.ndata['feat']: [num_nodes, node_feat_dim]
                - g.edata['feat']: [num_edges, edge_feat_dim]
        
        Returns:
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
            cram_features: [batch, 2]
        """
        # Project node features
        h = self.input_proj(g.ndata['feat'])  # [num_nodes, hidden_dim]
        
        # Process edge features (optional)
        if 'feat' in g.edata:
            edge_feat = self.edge_proj(g.edata['feat'])
        else:
            edge_feat = None
        
        # GAT layers with residual connections
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            h_new = gat(g, h).flatten(1)  # [num_nodes, hidden_dim]
            h = ln(h + h_new)  # Residual connection + LayerNorm
            h = F.relu(h)
        
        # Hierarchical pooling
        with g.local_scope():
            g.ndata['h'] = h
            
            # 1. Mean pooling: average over all nodes
            mean_pool = dgl.mean_nodes(g, 'h')  # [batch, hidden_dim]
            
            # 2. Max pooling: max over all nodes
            max_pool = dgl.max_nodes(g, 'h')  # [batch, hidden_dim]
            
            # 3. Attention-weighted pooling
            attn_scores = self.attention_pool(h)  # [num_nodes, 1]
            g.ndata['attn'] = attn_scores
            attn_weights = dgl.softmax_nodes(g, 'attn')  # Normalize per graph
            g.ndata['h_attn'] = h * attn_weights
            attn_pool = dgl.sum_nodes(g, 'h_attn')  # [batch, hidden_dim]
            
            # Combine all pooling strategies
            graph_embedding = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)
            # [batch, hidden_dim * 3]
            
            # Extract CRAM-specific features for conditioning
            # These are from node feature indices -7 and -6
            node_feats = g.ndata['feat']
            carboxyl_features = node_feats[:, -7]  # Is carboxyl atom
            aliphatic_features = node_feats[:, -6]  # Is aliphatic carbon
            
            g.ndata['cram_c'] = carboxyl_features.unsqueeze(-1)
            g.ndata['cram_a'] = aliphatic_features.unsqueeze(-1)
            
            # Aggregate CRAM features per graph
            carboxyl_count = dgl.sum_nodes(g, 'cram_c')  # [batch, 1]
            aliphatic_ratio = dgl.mean_nodes(g, 'cram_a')  # [batch, 1]
            
            cram_features = torch.cat([carboxyl_count, aliphatic_ratio], dim=-1)
            # [batch, 2]
        
        # Project to latent distribution parameters
        mu = self.fc_mu(graph_embedding)  # [batch, latent_dim]
        logvar = self.fc_logvar(graph_embedding)  # [batch, latent_dim]
        
        return mu, logvar, cram_features


# ============================================================================
# 2. TRANSFORMER DECODER
# ============================================================================

class TransformerDecoder(nn.Module):
    """
    Enhanced Transformer decoder with CRAM conditioning.
    
    Architecture:
    - CRAM conditioning: carboxyl_count + aliphatic_ratio → 32D embedding
    - Latent projection: (latent_dim + 32) → hidden_dim
    - Token embedding: vocab_size → hidden_dim
    - Positional encoding: Learnable [max_seq_len, hidden_dim]
    - Transformer decoder: 8 layers, 8 heads, pre-norm
    - Output projection: hidden_dim → vocab_size
    
    Input:
        - tokens: [batch, seq_len] - target tokens
        - latent_z: [batch, latent_dim] - latent representation
        - cram_features: [batch, 2] - carboxyl count + aliphatic ratio
        
    Output:
        - logits: [batch, seq_len, vocab_size] - predicted token probabilities
    """
    
    def __init__(
        self,
        vocab_size: int = 100,
        hidden_dim: int = 1024,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        max_seq_len: int = 150,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # CRAM conditioning embeddings
        self.cram_proj = nn.Linear(2, 32)  # 2D CRAM features → 32D embedding
        
        # Latent to hidden projection (with CRAM conditioning)
        self.latent_proj = nn.Linear(latent_dim + 32, hidden_dim)
        
        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size, hidden_dim, padding_idx=pad_idx
        )
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers
        )
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
    def forward(
        self,
        tokens: torch.Tensor,
        latent_z: torch.Tensor,
        cram_features: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through transformer decoder.
        
        Args:
            tokens: [batch, seq_len]
            latent_z: [batch, latent_dim]
            cram_features: [batch, 2]
            tgt_mask: [seq_len, seq_len] - causal mask
            tgt_key_padding_mask: [batch, seq_len] - padding mask
            
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = tokens.shape
        
        # CRAM conditioning
        cram_emb = self.cram_proj(cram_features)  # [batch, 32]
        
        # Concatenate latent with CRAM conditioning
        z_conditioned = torch.cat([latent_z, cram_emb], dim=-1)
        # [batch, latent_dim + 32]
        
        # Embed tokens and add positional encoding
        token_emb = self.token_embedding(tokens)  # [batch, seq_len, hidden_dim]
        token_emb = token_emb + self.pos_encoding[:, :seq_len, :]
        
        # Prepare memory (conditioned latent representation)
        memory = self.latent_proj(z_conditioned).unsqueeze(1)
        # [batch, 1, hidden_dim]
        
        # Transformer decode
        output = self.transformer_decoder(
            tgt=token_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [batch, seq_len, hidden_dim]
        
        # Normalize and project to vocabulary
        output = self.output_norm(output)
        logits = self.output_proj(output)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def generate_causal_mask(self, seq_len: int, device: torch.device):
        """
        Generate causal mask for autoregressive generation.
        
        Mask shape: [seq_len, seq_len]
        mask[i, j] = True means position i cannot attend to position j
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(device)


# ============================================================================
# 3. TRANSFORMER-VAE (COMPLETE MODEL)
# ============================================================================

class TransformerVAE(nn.Module):
    """
    Complete Transformer-VAE for CRAM molecular generation.
    
    Architecture:
    - Encoder: GraphEncoder (GAT + hierarchical pooling)
    - Decoder: TransformerDecoder (Transformer + CRAM conditioning)
    - Latent space: 512D with reparameterization trick
    - Template conditioning: Optional template encoder for guided generation
    
    Modes:
    1. Standard VAE: Encode molecule → Sample latent → Decode
    2. Template-based: Condition generation on template molecule
    3. Multi-mode: Similarity, diversity, or balanced generation
    
    Input:
        - graph: DGL graph batch
        - tokens: [batch, seq_len] - SELFIES tokens
        - template_graph: Optional template graph
        
    Output:
        - logits: [batch, seq_len, vocab_size]
        - mu, logvar: Latent distribution parameters
        - cram_features: CRAM conditioning features
    """
    
    def __init__(
        self,
        node_feat_dim: int = 73,
        vocab_size: int = 100,
        hidden_dim: int = 1024,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Main encoder and decoder
        self.encoder = GraphEncoder(
            node_feat_dim, hidden_dim, latent_dim, num_layers
        )
        self.decoder = TransformerDecoder(
            vocab_size, hidden_dim, latent_dim, num_layers, num_heads
        )
        
        # Template conditioning (optional)
        self.template_encoder = GraphEncoder(
            node_feat_dim, hidden_dim, latent_dim, num_layers
        )
        self.template_proj = nn.Linear(latent_dim, latent_dim)
        self.use_template_conditioning = True  # Toggle on/off
        
    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, 1)
        
        Args:
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
            
        Returns:
            z: [batch, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(
        self,
        graph: dgl.DGLGraph,
        tokens: torch.Tensor,
        template_graph: Optional[dgl.DGLGraph] = None
    ) -> Tuple:
        """
        Forward pass through complete VAE.
        
        Args:
            graph: Target molecule graph
            tokens: [batch, seq_len] - target SELFIES tokens
            template_graph: Optional template molecule graph
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
            cram_features: [batch, 2]
            template_z: [batch, latent_dim] if template provided
        """
        # Encode target molecule
        mu, logvar, cram_features = self.encoder(graph)
        z = self.reparameterize(mu, logvar)
        
        # Template conditioning (optional)
        template_z = None
        if template_graph is not None and self.use_template_conditioning:
            template_mu, _, _ = self.template_encoder(template_graph)
            template_z = self.template_proj(template_mu)
            # Blend latent with template
            z = z + 0.5 * template_z  # Weighted combination
        
        # Decode to SELFIES
        logits = self.decoder(tokens, z, cram_features)
        
        return logits, mu, logvar, cram_features, template_z
    
    def generate(
        self,
        num_samples: int = 1,
        max_len: int = 128,
        temperature: float = 1.0,
        template_graph: Optional[dgl.DGLGraph] = None,
        device: torch.device = torch.device('cuda')
    ) -> Tuple[List[str], np.ndarray]:
        """
        Generate CRAM molecules autoregressively.
        
        Args:
            num_samples: Number of molecules to generate
            max_len: Maximum sequence length
            temperature: Sampling temperature (higher = more random)
            template_graph: Optional template for guided generation
            device: torch device
            
        Returns:
            generated_smiles: List of generated SELFIES strings
            latent_vectors: [num_samples, latent_dim] latent representations
        """
        self.eval()
        generated_smiles = []
        
        with torch.no_grad():
            # Sample from prior N(0, I)
            z = torch.randn(num_samples, self.latent_dim, device=device)
            
            # Template conditioning
            if template_graph is not None and self.use_template_conditioning:
                template_mu, _, _ = self.template_encoder(template_graph)
                template_z = self.template_proj(template_mu)
                
                if template_z.size(0) == 1 and num_samples > 1:
                    template_z = template_z.repeat(num_samples, 1)
                
                z = z + 0.5 * template_z
            
            # Default CRAM features (2 COOH, 60% aliphatic)
            cram_features = torch.tensor(
                [[2.0, 0.6]], device=device, dtype=torch.float32
            ).repeat(num_samples, 1)
            
            # Initialize with START token
            START_IDX = 1
            END_IDX = 2
            sequences = torch.full(
                (num_samples, 1), START_IDX, dtype=torch.long, device=device
            )
            
            # Autoregressive generation
            for step in range(max_len - 1):
                # Create causal mask
                tgt_mask = self.decoder.generate_causal_mask(
                    sequences.size(1), device
                )
                
                # Get next token logits
                logits = self.decoder(sequences, z, cram_features, tgt_mask)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                # Append to sequences
                sequences = torch.cat([sequences, next_tokens], dim=1)
                
                # Check if all sequences have END token
                if (next_tokens == END_IDX).all():
                    break
            
            latent_vectors = z.cpu().numpy()
        
        # Convert sequences to SELFIES strings
        # (requires idx_to_token mapping)
        for seq in sequences:
            tokens_list = seq.cpu().numpy().tolist()
            # Remove START, END, PAD tokens
            PAD_IDX = 0
            tokens_list = [
                t for t in tokens_list 
                if t not in [START_IDX, END_IDX, PAD_IDX]
            ]
            # Convert to string (requires vocabulary)
            smiles_str = ''.join([str(t) for t in tokens_list])
            generated_smiles.append(smiles_str)
        
        return generated_smiles, latent_vectors


# ============================================================================
# 4. LOSS FUNCTIONS
# ============================================================================

def vae_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.01,
    pad_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss with cyclical KL annealing.
    
    Loss = Reconstruction Loss + β * KL Divergence
    
    Args:
        logits: [batch, seq_len, vocab_size]
        targets: [batch, seq_len]
        mu: [batch, latent_dim]
        logvar: [batch, latent_dim]
        beta: KL annealing weight (0.001 → 0.05)
        pad_idx: Padding token index
        
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence
    """
    # Reconstruction loss (cross-entropy)
    recon_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_idx,
        reduction='mean'
    )
    
    # KL divergence: KL(q(z|x) || p(z))
    # p(z) = N(0, I), q(z|x) = N(μ, σ²)
    # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    ) / mu.size(0)  # Average over batch
    
    # Total loss with annealing
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def cyclical_beta_schedule(
    epoch: int,
    total_epochs: int,
    n_cycles: int = 6,
    ratio: float = 0.5,
    beta_start: float = 0.001,
    beta_end: float = 0.05
) -> float:
    """
    Cyclical KL annealing schedule.
    
    Benefits:
    - Prevents posterior collapse
    - Improves latent space quality
    - Better reconstruction quality
    
    Args:
        epoch: Current epoch
        total_epochs: Total training epochs
        n_cycles: Number of annealing cycles
        ratio: Proportion of cycle for annealing
        beta_start: Starting β value
        beta_end: Maximum β value
        
    Returns:
        beta: Current β value for this epoch
    """
    cycle_length = total_epochs / n_cycles
    current_cycle_pos = (epoch % cycle_length) / cycle_length
    
    if current_cycle_pos < ratio:
        # Annealing phase: linearly increase β
        beta = beta_start + (beta_end - beta_start) * (current_cycle_pos / ratio)
    else:
        # Hold phase: keep β at maximum
        beta = beta_end
    
    return beta


# ============================================================================
# 5. MODEL ARCHITECTURE SUMMARY
# ============================================================================

def print_model_summary(model: TransformerVAE):
    """
    Print comprehensive model architecture summary.
    """
    print("=" * 80)
    print("CRAM MOLECULAR GENERATOR - MODEL ARCHITECTURE")
    print("=" * 80)
    print()
    
    print("🏗️  Architecture: Transformer-VAE with Graph Neural Networks")
    print()
    
    # Encoder
    print("1. GRAPH ENCODER (GAT-based)")
    print("   " + "-" * 70)
    print(f"   Input:")
    print(f"      • Node features: {model.encoder.node_feat_dim}D")
    print(f"      • Edge features: 6D (bond type, aromatic, conjugated, ring)")
    print(f"   Architecture:")
    print(f"      • Input projection: {model.encoder.node_feat_dim}D → {model.encoder.hidden_dim}D")
    print(f"      • Edge projection: 6D → {model.encoder.hidden_dim}D")
    print(f"      • GAT layers: 3 layers, 8 heads each")
    print(f"      • Hierarchical pooling: Mean + Max + Attention")
    print(f"   Output:")
    print(f"      • Latent distribution: μ, σ² ∈ ℝ^{model.encoder.latent_dim}")
    print(f"      • CRAM features: [carboxyl_count, aliphatic_ratio] ∈ ℝ^2")
    print()
    
    # Decoder
    print("2. TRANSFORMER DECODER (Autoregressive)")
    print("   " + "-" * 70)
    print(f"   Input:")
    print(f"      • Latent vector: {model.latent_dim}D")
    print(f"      • CRAM conditioning: 2D → 32D embedding")
    print(f"      • Token embeddings: {model.decoder.vocab_size} vocab → {model.decoder.hidden_dim}D")
    print(f"   Architecture:")
    print(f"      • Latent projection: {model.latent_dim + 32}D → {model.decoder.hidden_dim}D")
    print(f"      • Positional encoding: Learnable [{model.decoder.pos_encoding.size(1)}, {model.decoder.hidden_dim}D]")
    print(f"      • Transformer layers: 8 layers, 8 heads, 4096D feedforward")
    print(f"      • Pre-norm architecture for stability")
    print(f"   Output:")
    print(f"      • Token logits: {model.decoder.vocab_size} classes (SELFIES vocabulary)")
    print()
    
    # Latent space
    print("3. LATENT SPACE")
    print("   " + "-" * 70)
    print(f"   Dimension: {model.latent_dim}D")
    print(f"   Distribution: q(z|x) = N(μ, σ²)")
    print(f"   Prior: p(z) = N(0, I)")
    print(f"   Reparameterization: z = μ + σ * ε, ε ~ N(0, 1)")
    print(f"   KL annealing: β ∈ [0.001, 0.05] (cyclical)")
    print()
    
    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("4. MODEL STATISTICS")
    print("   " + "-" * 70)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / (1024**2):.1f} MB (FP32)")
    print()
    
    # Training info
    print("5. TRAINING CONFIGURATION")
    print("   " + "-" * 70)
    print(f"   Loss: Reconstruction + β * KL Divergence")
    print(f"   Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)")
    print(f"   Scheduler: ReduceLROnPlateau (patience=5)")
    print(f"   Batch size: 16")
    print(f"   Max sequence length: 150")
    print(f"   KL annealing: Cyclical (6 cycles, ratio=0.5)")
    print()
    
    # CRAM-specific
    print("6. CRAM-SPECIFIC FEATURES")
    print("   " + "-" * 70)
    print(f"   Node features (73D/88D):")
    print(f"      • Base features: 45D (atomic, structural, electronic)")
    print(f"      • Synthetic C3/C4: 13D (stereochemistry, chain properties)")
    print(f"      • Marine DOM: 15D (oxidation, degradation, photo-oxidation)")
    print(f"   CRAM conditioning:")
    print(f"      • Carboxyl count (target: ≥1)")
    print(f"      • Aliphatic ratio (target: ≥70%)")
    print(f"   Generation modes:")
    print(f"      • Similarity: Stay close to templates (Tanimoto >0.7)")
    print(f"      • Diversity: Explore chemical space (Tanimoto <0.4)")
    print(f"      • Balanced: Moderate novelty (Tanimoto ~0.5-0.6)")
    print()
    
    print("=" * 80)
    print("✅ Model architecture summary complete")
    print("=" * 80)


# ============================================================================
# 6. EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = ModelConfig()
    
    # Initialize model
    model = TransformerVAE(
        node_feat_dim=config.NODE_FEAT_DIM,
        vocab_size=config.VOCAB_SIZE,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS
    )
    
    # Print summary
    print_model_summary(model)
    
    # Example forward pass
    print("\n📊 Example forward pass:")
    print("-" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy graph
    num_nodes = 20
    num_edges = 25
    g = dgl.graph((
        torch.randint(0, num_nodes, (num_edges,)),
        torch.randint(0, num_nodes, (num_edges,))
    ))
    g.ndata['feat'] = torch.randn(num_nodes, config.NODE_FEAT_DIM)
    g.edata['feat'] = torch.randn(num_edges, config.EDGE_FEAT_DIM)
    g = g.to(device)
    
    # Dummy tokens
    batch_size = 4
    seq_len = 50
    tokens = torch.randint(0, config.VOCAB_SIZE, (batch_size, seq_len), device=device)
    
    # Forward pass
    logits, mu, logvar, cram_features, _ = model(g, tokens)
    
    print(f"Input:")
    print(f"   Graph: {num_nodes} nodes, {num_edges} edges")
    print(f"   Tokens: {tokens.shape}")
    print()
    print(f"Output:")
    print(f"   Logits: {logits.shape}")
    print(f"   Latent (μ): {mu.shape}")
    print(f"   Latent (log σ²): {logvar.shape}")
    print(f"   CRAM features: {cram_features.shape}")
    print()
    
    # Calculate loss
    beta = 0.01
    total_loss, recon_loss, kl_loss = vae_loss(logits, tokens, mu, logvar, beta)
    print(f"Loss:")
    print(f"   Total: {total_loss.item():.4f}")
    print(f"   Reconstruction: {recon_loss.item():.4f}")
    print(f"   KL (β={beta}): {kl_loss.item():.4f}")
    print()
    
    print("=" * 80)
    print("✅ Model verification complete - Architecture ready for visualization!")
    print("=" * 80)
