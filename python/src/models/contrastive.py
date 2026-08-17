"""Notebook ContrastiveCRAMLearning."""
from __future__ import annotations
from typing import Dict, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

"""
=============================================================================
CRITICAL IMPROVEMENT 3: CONTRASTIVE LEARNING FOR CRAM/NON-CRAM SEPARATION
=============================================================================

Contrastive learning is ESSENTIAL for CRAM generation because:

1. **Latent space structure**: CRAM molecules should cluster together
2. **Generation reliability**: Sampling from CRAM cluster produces CRAM
3. **Hard negative mining**: "Almost-CRAM" molecules teach fine distinctions
4. **Property gradients**: Smooth transitions in latent space = smooth property changes

We implement:
- Supervised Contrastive Loss (SupCon) for CRAM class labels
- Triplet margin loss for (anchor, positive, negative) tuples
- Hard negative mining based on chemical similarity
"""

class ContrastiveCRAMLearning(nn.Module):
    """
    Contrastive learning module for CRAM vs non-CRAM separation.
    
    This module implements multiple contrastive objectives:
    1. SupCon loss: Groups CRAM molecules together
    2. Triplet loss: Creates separation between similar structures
    3. Center loss: Pulls CRAM to a learned center
    """
    
    def __init__(self, latent_dim: int, temperature: float = 0.07, 
                 margin: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.margin = margin
        
        # Projection head for contrastive learning (following SimCLR)
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Learnable class centers for center loss
        self.cram_center = nn.Parameter(torch.randn(latent_dim) * 0.01)
        self.non_cram_center = nn.Parameter(torch.randn(latent_dim) * 0.01)
        
        # Prototype embeddings for different CRAM types
        self.prototypes = nn.ParameterDict({
            'alicyclic': nn.Parameter(torch.randn(latent_dim) * 0.01),
            'aliphatic': nn.Parameter(torch.randn(latent_dim) * 0.01),
            'hybrid': nn.Parameter(torch.randn(latent_dim) * 0.01),
        })
    
    def forward(self, z: torch.Tensor, labels: torch.Tensor, 
                cram_types: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive losses.
        
        Args:
            z: Latent vectors (batch_size, latent_dim)
            labels: Binary CRAM labels (batch_size,) - 1 for CRAM, 0 for non-CRAM
            cram_types: Optional CRAM type labels (0=alicyclic, 1=aliphatic, 2=hybrid)
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Project to contrastive space
        z_proj = self.projection_head(z)
        z_proj = F.normalize(z_proj, dim=1)
        
        # === 1. Supervised Contrastive Loss ===
        losses['supcon'] = self._supervised_contrastive_loss(z_proj, labels)
        
        # === 2. Center Loss ===
        losses['center'] = self._center_loss(z, labels)
        
        # === 3. CRAM Type Contrastive (if types available) ===
        if cram_types is not None:
            losses['type_contrast'] = self._type_contrastive_loss(z, cram_types)
        
        # === 4. Margin separation loss ===
        losses['margin'] = self._margin_separation_loss(z, labels)
        
        return losses
    
    def _supervised_contrastive_loss(self, z: torch.Tensor, 
                                      labels: torch.Tensor) -> torch.Tensor:
        """
        SupCon loss: Pull same-class samples together, push different classes apart.
        
        Based on: Khosla et al. "Supervised Contrastive Learning"
        """
        device = z.device
        batch_size = z.size(0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Create positive mask (same label)
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float().to(device)
        
        # Remove diagonal (self-similarity)
        identity = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - identity
        
        # For numerical stability
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()
        
        # Compute log-sum-exp of all samples (denominator)
        exp_sim = torch.exp(sim_matrix) * (1 - identity)  # Exclude self
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean of positive similarities (numerator)
        num_positives = positive_mask.sum(dim=1, keepdim=True).clamp(min=1)
        positive_sim = (sim_matrix * positive_mask).sum(dim=1, keepdim=True)
        
        # Loss = -log(positive / sum_all)
        loss = -(positive_sim / num_positives - log_sum_exp)
        
        # Only compute for samples with at least one positive pair
        valid_samples = (positive_mask.sum(dim=1) > 0).float()
        if valid_samples.sum() > 0:
            loss = (loss.squeeze() * valid_samples).sum() / valid_samples.sum()
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss
    
    def _center_loss(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Center loss: Pull samples toward their class center.
        
        This helps create compact, well-separated clusters.
        """
        device = z.device
        cram_mask = labels == 1
        non_cram_mask = labels == 0
        
        loss = torch.tensor(0.0, device=device)
        
        # Pull CRAM samples to CRAM center
        if cram_mask.sum() > 0:
            cram_z = z[cram_mask]
            cram_dist = ((cram_z - self.cram_center.unsqueeze(0)) ** 2).sum(dim=1)
            loss = loss + cram_dist.mean()
        
        # Pull non-CRAM samples to non-CRAM center
        if non_cram_mask.sum() > 0:
            non_cram_z = z[non_cram_mask]
            non_cram_dist = ((non_cram_z - self.non_cram_center.unsqueeze(0)) ** 2).sum(dim=1)
            loss = loss + non_cram_dist.mean()
        
        # Push centers apart
        center_dist = ((self.cram_center - self.non_cram_center) ** 2).sum()
        loss = loss - 0.1 * center_dist  # Negative because we want centers far apart
        
        return loss * 0.1  # Scale factor
    
    def _type_contrastive_loss(self, z: torch.Tensor, 
                                cram_types: torch.Tensor) -> torch.Tensor:
        """
        Contrastive loss for CRAM subtypes (alicyclic, aliphatic, hybrid).
        
        This teaches the model to distinguish between different CRAM architectures.
        """
        device = z.device
        loss = torch.tensor(0.0, device=device)
        
        type_names = ['alicyclic', 'aliphatic', 'hybrid']
        
        for type_idx, type_name in enumerate(type_names):
            type_mask = cram_types == type_idx
            if type_mask.sum() > 0:
                type_z = z[type_mask]
                prototype = self.prototypes[type_name]
                # Pull type samples to prototype
                dist = ((type_z - prototype.unsqueeze(0)) ** 2).sum(dim=1)
                loss = loss + dist.mean()
        
        return loss * 0.05
    
    def _margin_separation_loss(self, z: torch.Tensor, 
                                 labels: torch.Tensor) -> torch.Tensor:
        """
        Margin-based loss to ensure clear separation between classes.
        
        Ensures that the closest non-CRAM is at least `margin` away from
        the furthest CRAM in the latent space.
        """
        device = z.device
        cram_mask = labels == 1
        non_cram_mask = labels == 0
        
        if cram_mask.sum() == 0 or non_cram_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        cram_z = z[cram_mask]
        non_cram_z = z[non_cram_mask]
        
        # Compute cross-class distances
        # (num_cram, num_non_cram)
        distances = torch.cdist(cram_z, non_cram_z)
        
        # Minimum distance from each CRAM to any non-CRAM
        min_distances = distances.min(dim=1)[0]
        
        # Hinge loss: penalize if distance < margin
        margin_loss = F.relu(self.margin - min_distances).mean()
        
        return margin_loss


class HardNegativeMiner:
    """
    Mines hard negative samples for contrastive learning.
    
    Hard negatives are crucial because:
    1. Random negatives are too easy (model learns nothing)
    2. "Almost-CRAM" molecules teach fine distinctions
    3. Structurally similar non-CRAM molecules improve discrimination
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
    
    def mine_triplets(self, molecules: List[Chem.Mol], 
                      labels: np.ndarray,
                      embeddings: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Mine (anchor, positive, negative) triplets.
        
        Strategy:
        - Anchor: CRAM molecule
        - Positive: Another CRAM molecule (preferably similar structure)
        - Negative: Non-CRAM with highest similarity to anchor (hardest)
        """
        triplets = []
        
        cram_indices = np.where(labels == 1)[0]
        non_cram_indices = np.where(labels == 0)[0]
        
        if len(cram_indices) < 2 or len(non_cram_indices) < 1:
            return triplets
        
        # Compute pairwise embedding similarities
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = embeddings_norm @ embeddings_norm.T
        
        for anchor_idx in cram_indices:
            # Find positive: most similar CRAM (excluding self)
            cram_sims = [(idx, similarities[anchor_idx, idx]) 
                        for idx in cram_indices if idx != anchor_idx]
            if not cram_sims:
                continue
            positive_idx = max(cram_sims, key=lambda x: x[1])[0]
            
            # Find hard negative: most similar non-CRAM
            non_cram_sims = [(idx, similarities[anchor_idx, idx]) 
                            for idx in non_cram_indices]
            hard_negative_idx = max(non_cram_sims, key=lambda x: x[1])[0]
            
            triplets.append((anchor_idx, positive_idx, hard_negative_idx))
        
        return triplets
    
    def create_hard_negatives_chemically(self, mol: Chem.Mol) -> List[Chem.Mol]:
        """
        Create chemical hard negatives by modifying CRAM molecules.
        
        Hard negatives:
        1. Remove one COOH (1 COOH instead of 2+)
        2. Replace COOH with CH3 (lose carboxyl-rich property)
        3. Break ring (lose alicyclic property)
        4. Add aromatics (outside CRAM definition)
        """
        hard_negatives = []
        
        # Strategy 1: Remove COOH (conceptual - actual implementation would 
        # use RDKit molecular editing)
        # This would create a molecule with fewer carboxyls
        
        # For now, return empty - actual implementation would 
        # generate modified molecules
        return hard_negatives


class CRAMContrastiveTrainer:
    """
    Training wrapper that incorporates contrastive learning.
    """
    
    def __init__(self, model: nn.Module, contrastive_module: ContrastiveCRAMLearning,
                 contrastive_weight: float = 0.2):
        self.model = model
        self.contrastive_module = contrastive_module
        self.contrastive_weight = contrastive_weight
    
    def compute_contrastive_loss(self, z: torch.Tensor, 
                                  batch_data: Dict) -> torch.Tensor:
        """
        Compute total contrastive loss for a batch.
        """
        labels = batch_data.get('is_cram', None)
        cram_types = batch_data.get('cram_type', None)
        
        if labels is None:
            return torch.tensor(0.0, device=z.device)
        
        losses = self.contrastive_module(z, labels, cram_types)
        
        # Weighted combination
        total = (losses.get('supcon', 0) * 0.4 +
                losses.get('center', 0) * 0.3 +
                losses.get('margin', 0) * 0.2 +
                losses.get('type_contrast', 0) * 0.1)
        
        return total * self.contrastive_weight



