"""Notebook FunctionalGroupAttention and CRAMGroupPropertyPredictor."""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class FunctionalGroupAttention(nn.Module):
    """
    Attention module that operates at functional group level.
    
    Two-level attention:
    1. Within-group: Aggregate atom features to group representation
    2. Between-group: Cross-attention between different groups
    
    This captures CRAM chemistry:
    - How COOH groups interact with rings
    - How chain length affects group relationships
    - Overall molecular topology from group perspective
    """
    
    def __init__(self, atom_dim: int, group_dim: int = 64, 
                 num_heads: int = 4, num_group_types: int = 13):
        super().__init__()
        self.atom_dim = atom_dim
        self.group_dim = group_dim
        self.num_heads = num_heads
        self.num_group_types = num_group_types
        
        # === Within-Group Attention ===
        # Attention to aggregate atoms within each functional group
        self.within_group_attention = nn.MultiheadAttention(
            embed_dim=atom_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        
        # Project atom features to group space
        self.atom_to_group = nn.Sequential(
            nn.Linear(atom_dim, group_dim),
            nn.LayerNorm(group_dim),
            nn.GELU()
        )
        
        # === Group Type Embeddings ===
        # Learnable embeddings for each functional group type
        self.group_type_embeddings = nn.Embedding(num_group_types, group_dim)
        
        # === Between-Group Attention ===
        # Cross-attention between functional groups
        self.between_group_attention = nn.MultiheadAttention(
            embed_dim=group_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # === CRAM-specific group interaction layer ===
        # Special attention for COOH ↔ Ring interactions
        self.cram_interaction = nn.Sequential(
            nn.Linear(group_dim * 2, group_dim),
            nn.LayerNorm(group_dim),
            nn.GELU(),
            nn.Linear(group_dim, group_dim)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(group_dim * 3, atom_dim),  # 3 = within + between + interaction
            nn.LayerNorm(atom_dim),
            nn.GELU()
        )
    
    def forward(self, atom_features: torch.Tensor, 
                group_membership: torch.Tensor,
                group_masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Apply functional group attention.
        
        Args:
            atom_features: (batch_size, num_atoms, atom_dim)
            group_membership: (batch_size, num_atoms, num_group_types)
            group_masks: Optional masks for specific group types
        
        Returns:
            Enhanced atom features with group context
        """
        batch_size, num_atoms, _ = atom_features.shape
        device = atom_features.device
        
        # === 1. Within-Group Aggregation ===
        # Create group representations by attention-pooling atoms
        group_representations = []
        
        for group_idx in range(self.num_group_types):
            # Get atoms belonging to this group type
            group_mask = group_membership[:, :, group_idx] > 0  # (batch, atoms)
            
            # For each batch, pool atoms in this group
            batch_group_reps = []
            for b in range(batch_size):
                mask = group_mask[b]
                if mask.sum() > 0:
                    group_atoms = atom_features[b, mask]  # (n_group_atoms, atom_dim)
                    
                    # Self-attention within group
                    group_atoms = group_atoms.unsqueeze(0)  # (1, n, dim)
                    attended, _ = self.within_group_attention(
                        group_atoms, group_atoms, group_atoms
                    )
                    
                    # Pool to single representation
                    group_rep = attended.mean(dim=1).squeeze(0)  # (atom_dim,)
                else:
                    group_rep = torch.zeros(self.atom_dim, device=device)
                
                batch_group_reps.append(group_rep)
            
            group_representations.append(torch.stack(batch_group_reps))  # (batch, atom_dim)
        
        # Stack all group representations
        group_reps = torch.stack(group_representations, dim=1)  # (batch, num_groups, atom_dim)
        
        # Project to group space and add type embeddings
        group_reps = self.atom_to_group(group_reps)  # (batch, num_groups, group_dim)
        type_embeds = self.group_type_embeddings(
            torch.arange(self.num_group_types, device=device)
        ).unsqueeze(0).expand(batch_size, -1, -1)
        group_reps = group_reps + type_embeds
        
        # === 2. Between-Group Attention ===
        # Groups attend to each other
        between_attended, between_weights = self.between_group_attention(
            group_reps, group_reps, group_reps
        )
        
        # === 3. CRAM-Specific Interactions ===
        # Special handling for COOH (group 0) and rings (groups 2,3,4)
        carboxyl_rep = group_reps[:, 0, :]  # (batch, group_dim)
        ring_reps = group_reps[:, 2:5, :].mean(dim=1)  # Average of ring types
        
        cram_interaction = self.cram_interaction(
            torch.cat([carboxyl_rep, ring_reps], dim=-1)
        )  # (batch, group_dim)
        
        # === 4. Broadcast back to atoms ===
        # Enrich atom features with group-level information
        enriched_atoms = atom_features.clone()
        
        for group_idx in range(self.num_group_types):
            group_mask = group_membership[:, :, group_idx] > 0
            
            # Get this group's attended representation
            group_attended = between_attended[:, group_idx, :]  # (batch, group_dim)
            
            # Broadcast to atoms in this group
            for b in range(batch_size):
                mask = group_mask[b]
                if mask.sum() > 0:
                    # Combine within, between, and CRAM interaction
                    combined = torch.cat([
                        group_reps[b, group_idx],  # Within-group
                        group_attended[b],  # Between-group
                        cram_interaction[b]  # CRAM interaction
                    ])
                    
                    # Project back to atom dimension
                    atom_enrichment = self.output_proj(combined)
                    
                    # Add to original features
                    enriched_atoms[b, mask] = enriched_atoms[b, mask] + atom_enrichment
        
        return enriched_atoms


class CRAMGroupPropertyPredictor(nn.Module):
    """
    Predicts CRAM properties from functional group features.
    
    This is chemically meaningful because CRAM properties emerge from:
    1. Number and arrangement of COOH groups
    2. Ring system complexity
    3. Chain lengths between groups
    4. Overall molecular architecture
    """
    
    def __init__(self, group_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.fg_detector = FunctionalGroupDetector()
        
        # Count-based features
        self.count_encoder = nn.Sequential(
            nn.Linear(13, hidden_dim),  # 13 group types
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Property predictors
        self.is_cram_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.cram_type_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)  # alicyclic, aliphatic, hybrid
        )
        
        self.stability_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, mol: Chem.Mol) -> Dict[str, torch.Tensor]:
        """Predict CRAM properties from functional groups."""
        groups = self.fg_detector.detect_groups(mol)
        
        # Count each group type
        counts = torch.zeros(13)
        group_names = list(FunctionalGroupDetector.FG_PATTERNS.keys())
        for idx, name in enumerate(group_names):
            if name in groups:
                counts[idx] = len(groups[name])
        
        # Encode counts
        encoded = self.count_encoder(counts)
        
        # Predict properties
        predictions = {
            'is_cram': self.is_cram_predictor(encoded),
            'cram_type': self.cram_type_predictor(encoded),
            'stability': self.stability_predictor(encoded)
        }
        
        return predictions



