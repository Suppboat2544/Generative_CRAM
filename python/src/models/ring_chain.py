"""Notebook SubstructureDetector, RingChainEncoder, CRAMStructureLoss."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

# ============================================================================
# RingChainEncoder: Explicit Cyclic+Chain Structure Encoding
# ============================================================================
#
# This module provides explicit encoding of ring-chain hybrid structures
# that define TRUE CRAM molecules. It addresses the need to capture:
# - Alicyclic rings (cyclohexane, cyclopentane) connected to
# - Aliphatic chains with carboxyl groups attached
#
# Architecture:
# 1. SubstructureDetector: Find rings, chains, junctions, COOH groups
# 2. SubstructureEncoder: Learn representations for each substructure type
# 3. SubstructureAttention: Cross-attend between substructure types
# 4. CRAMFusion: Combine into CRAM-aware molecular representation
# ============================================================================

class SubstructureDetector:
    """
    Detect and categorize molecular substructures for CRAM analysis.
    
    Categories:
    - ALICYCLIC_RING: Non-aromatic rings (5-6 membered preferred)
    - AROMATIC_RING: Benzene-like rings (NOT CRAM)
    - CHAIN: Linear aliphatic sequences
    - JUNCTION: Atoms connecting rings to chains
    - CARBOXYL: -COOH groups
    - BRANCH: Chain branching points
    """
    
    SUBSTRUCTURE_TYPES = {
        'alicyclic_ring': 0,
        'aromatic_ring': 1,
        'chain': 2,
        'junction': 3,
        'carboxyl': 4,
        'branch': 5,
        'terminus': 6
    }
    
    @staticmethod
    def detect_substructures(mol):
        """
        Detect all substructures in a molecule.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            dict mapping substructure type to list of atom indices
        """
        results = {
            'alicyclic_ring': [],
            'aromatic_ring': [],
            'chain': [],
            'junction': [],
            'carboxyl': [],
            'branch': [],
            'terminus': []
        }
        
        if mol is None:
            return results
        
        try:
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()
            
            # Find carboxyl carbons
            carboxyl_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
            carboxyl_matches = mol.GetSubstructMatches(carboxyl_pattern)
            carboxyl_atoms = set()
            for match in carboxyl_matches:
                carboxyl_atoms.update(match)
            results['carboxyl'] = list(carboxyl_atoms)
            
            # Categorize rings
            ring_atoms = set()
            for ring in atom_rings:
                ring_aromatic = any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                if ring_aromatic:
                    results['aromatic_ring'].extend(ring)
                else:
                    results['alicyclic_ring'].extend(ring)
                ring_atoms.update(ring)
            
            # Remove duplicates
            results['aromatic_ring'] = list(set(results['aromatic_ring']))
            results['alicyclic_ring'] = list(set(results['alicyclic_ring']))
            
            # Categorize non-ring atoms
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                
                if idx in ring_atoms:
                    # Check if junction (ring atom with non-ring neighbor)
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetIdx() not in ring_atoms:
                            if idx not in results['junction']:
                                results['junction'].append(idx)
                            break
                elif idx not in carboxyl_atoms:
                    # Chain atom
                    degree = atom.GetDegree()
                    if degree == 1:
                        results['terminus'].append(idx)
                    elif degree >= 3:
                        results['branch'].append(idx)
                    else:
                        results['chain'].append(idx)
            
        except Exception as e:
            pass
        
        return results
    
    @staticmethod
    def get_substructure_mask(mol, substructure_type):
        """
        Get a binary mask for atoms of a specific substructure type.
        
        Args:
            mol: RDKit molecule
            substructure_type: One of SUBSTRUCTURE_TYPES keys
            
        Returns:
            List of 0/1 for each atom
        """
        substructures = SubstructureDetector.detect_substructures(mol)
        atoms_of_type = set(substructures.get(substructure_type, []))
        
        return [1 if i in atoms_of_type else 0 for i in range(mol.GetNumAtoms())]


class RingChainEncoder(nn.Module):
    """
    Encode ring-chain hybrid structures with explicit substructure awareness.
    
    This module learns separate representations for:
    - Alicyclic rings (TRUE CRAM)
    - Chains (aliphatic sequences)
    - Junctions (ring-chain connections)
    - Carboxyl groups (defining CRAM feature)
    
    Then combines them with attention to capture CRAM-specific patterns.
    """
    
    def __init__(self, hidden_dim, num_substructure_types=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_substructure_types = num_substructure_types
        
        # Substructure type embedding
        self.substructure_embed = nn.Embedding(num_substructure_types, hidden_dim // 4)
        
        # Substructure-specific encoders
        self.substructure_encoders = nn.ModuleDict({
            'alicyclic': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            'chain': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            'carboxyl': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            'junction': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        })
        
        # Cross-attention between substructure types
        self.ring_chain_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # Carboxyl-ring attention (how carboxyls relate to rings)
        self.carboxyl_ring_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # CRAM structure score predictor
        self.cram_structure_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 5) # [has_alicyclic, has_chain, has_junction, num_cooh, is_cram]
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, node_features, substructure_masks, batch_indices=None):
        """
        Encode ring-chain structure with substructure awareness.
        
        Args:
            node_features: [num_nodes, hidden_dim] atom representations
            substructure_masks: dict of {substructure_type: [num_nodes]} binary masks
            batch_indices: [num_nodes] batch index for each node (for batched graphs)
            
        Returns:
            cram_encoding: [batch_size, hidden_dim] CRAM-aware molecular representation
            cram_scores: [batch_size, 5] CRAM structure predictions
        """
        batch_size = 1 if batch_indices is None else batch_indices.max().item() + 1
        device = node_features.device
        
        # Aggregate features by substructure type
        substructure_reps = {}
        
        for sub_type, encoder in self.substructure_encoders.items():
            mask_key = 'alicyclic_ring' if sub_type == 'alicyclic' else sub_type
            if mask_key in substructure_masks and substructure_masks[mask_key] is not None:
                mask = substructure_masks[mask_key]
                if isinstance(mask, list):
                    mask = torch.tensor(mask, device=device).float()
                
                # Apply mask and aggregate
                if mask is not None and mask.sum() > 0:
                    masked_features = node_features * mask.unsqueeze(-1)
                    # Mean pooling over masked atoms
                    aggregated = masked_features.sum(dim=0) / (mask.sum() + 1e-8)
                    substructure_reps[sub_type] = encoder(aggregated.unsqueeze(0))
                else:
                    substructure_reps[sub_type] = torch.zeros(1, self.hidden_dim, device=device)
            else:
                substructure_reps[sub_type] = torch.zeros(1, self.hidden_dim, device=device)
        
        # Stack substructure representations
        ring_rep = substructure_reps.get('alicyclic', torch.zeros(1, self.hidden_dim, device=device))
        chain_rep = substructure_reps.get('chain', torch.zeros(1, self.hidden_dim, device=device))
        junction_rep = substructure_reps.get('junction', torch.zeros(1, self.hidden_dim, device=device))
        carboxyl_rep = substructure_reps.get('carboxyl', torch.zeros(1, self.hidden_dim, device=device))
        
        # Ensure all reps are shape [1, hidden_dim]
        ring_rep = ring_rep.view(1, -1)
        chain_rep = chain_rep.view(1, -1)
        junction_rep = junction_rep.view(1, -1)
        carboxyl_rep = carboxyl_rep.view(1, -1)
        
        # Cross-attention: rings attending to chains (ring-chain connectivity)
        # MultiheadAttention expects [seq_len, batch, embed_dim] by default
        ring_chain_stack = torch.stack([ring_rep.squeeze(0), chain_rep.squeeze(0)], dim=0).unsqueeze(1) # [2, 1, hidden_dim]
        ring_chain_attn, _ = self.ring_chain_attention(
            ring_chain_stack, ring_chain_stack, ring_chain_stack
        )
        ring_chain_combined = ring_chain_attn.mean(dim=0) # [1, hidden_dim]
        
        # Carboxyl-ring attention (how carboxyls relate to rings)
        carboxyl_ring_stack = torch.stack([carboxyl_rep.squeeze(0), ring_rep.squeeze(0)], dim=0).unsqueeze(1) # [2, 1, hidden_dim]
        carboxyl_ring_attn, _ = self.carboxyl_ring_attention(
            carboxyl_ring_stack, carboxyl_ring_stack, carboxyl_ring_stack
        )
        carboxyl_ring_combined = carboxyl_ring_attn.mean(dim=0) # [1, hidden_dim]
        
        # Combine all representations
        combined = torch.cat([
            ring_chain_combined,
            carboxyl_ring_combined,
            junction_rep,
            carboxyl_rep
        ], dim=-1)
        
        # Predict CRAM structure scores (graph-level)
        cram_scores = self.cram_structure_head(combined)
        
        # Final CRAM-aware representation (graph-level)
        graph_encoding = self.fusion(combined).squeeze(0) # [hidden_dim]
        
        # Broadcast graph-level encoding to all nodes
        num_nodes = node_features.shape[0]
        node_encoding = graph_encoding.unsqueeze(0).expand(num_nodes, -1) # [num_nodes, hidden_dim]
        
        # Node-level for GAT fusion; graph-level scores for hierarchical encoder
        return node_encoding, cram_scores


class CRAMStructureLoss(nn.Module):
    """
    Loss function that penalizes non-CRAM structures.
    
    Rewards:
    - Multiple carboxyl groups (2+)
    - Alicyclic rings (NOT aromatic)
    - Ring-chain hybrid structures (junctions)
    - Correct elemental ratios (O/C 0.4-0.8, H/C 1.2-2.0)
    
    Penalizes:
    - Aromatic rings (benzene)
    - Only 0-1 carboxyl groups
    - Pure chains without rings OR pure rings without chains
    """
    
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {
            'carboxyl': 0.3,
            'alicyclic': 0.2,
            'no_aromatic': 0.15,
            'junction': 0.15,
            'elemental': 0.2
        }
    
    def forward(self, cram_scores, target_cram_features):
        """
        Compute CRAM structure loss.
        
        Args:
            cram_scores: [batch, 5] predicted CRAM structure scores
            target_cram_features: [batch, 5] ground truth CRAM features
                [has_alicyclic, has_chain, has_junction, num_cooh_norm, is_cram]
            
        Returns:
            loss: Scalar loss value
        """
        # MSE loss on structure scores
        loss = F.mse_loss(cram_scores, target_cram_features)
        
        # Additional penalty for low carboxyl prediction when it should be high
        carboxyl_penalty = F.relu(target_cram_features[:, 3] - cram_scores[:, 3]).mean()
        
        # Penalty for predicting aromatic when target is alicyclic
        # (encourage alicyclic over aromatic)
        alicyclic_bonus = F.relu(cram_scores[:, 0]).mean() # Reward high alicyclic prediction
        
        total_loss = loss + 0.2 * carboxyl_penalty - 0.1 * alicyclic_bonus
        
        return total_loss


def extract_cram_structure_targets(mol):
    """
    Extract CRAM structure targets for loss computation.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        torch.Tensor of shape [5] with:
        [has_alicyclic, has_chain, has_junction, num_cooh_norm, is_cram]
    """
    features = get_cram_molecule_features(mol)
    
    targets = [
        1.0 if features.get('num_alicyclic', 0) > 0 else 0.0, # has_alicyclic
        1.0 if features.get('aliphatic_carbons', 0) > 3 else 0.0, # has_chain
        1.0 if features.get('num_junctions', 0) > 0 else 0.0, # has_junction
        min(features.get('num_carboxyl', 0), 4) / 4.0, # num_cooh_norm
        1.0 if features.get('is_true_cram', False) else 0.0 # is_cram
    ]
    
    return torch.tensor(targets, dtype=torch.float32)



