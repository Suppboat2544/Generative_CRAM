"""Notebook FunctionalGroupDetector (13D membership)."""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch
from rdkit import Chem

"""
=============================================================================
CRITICAL IMPROVEMENT 5: FUNCTIONAL GROUP ATTENTION MODULE
=============================================================================

Standard GNNs miss the chemistry: they treat molecules as atom graphs,
but chemistry happens at FUNCTIONAL GROUP level.

For CRAM molecules:
- COOH groups are the defining feature (must be 2+)
- Alicyclic rings are stability-providing units
- CH2 chains provide flexibility between groups
- Branching points (quaternary carbons) are structural landmarks

This module implements:
1. Functional group detection and embedding
2. Within-group attention (atoms → group representation)
3. Between-group attention (group → molecule understanding)
4. Property prediction from group-level features
"""

class FunctionalGroupDetector:
    """
    Detects and characterizes functional groups relevant to CRAM.
    
    Groups detected:
    1. Carboxyl (COOH/COO-) - defining CRAM feature
    2. Alicyclic rings (5-7 membered saturated)
    3. Methylene chains (CH2 sequences)
    4. Branching points (quaternary C)
    5. Hydroxyl (OH)
    6. Carbonyl (C=O not in COOH)
    7. Ether linkages (C-O-C)
    """
    
    # SMARTS patterns for CRAM-relevant functional groups
    FG_PATTERNS = {
        'carboxyl': '[CX3](=O)[OX2H1,OX1-]',
        'carboxylate': '[CX3](=O)[O-]',
        'ring_5': '[C]1[C][C][C][C]1',
        'ring_6': '[C]1[C][C][C][C][C]1',
        'ring_7': '[C]1[C][C][C][C][C][C]1',
        'methylene_chain': '[CH2][CH2][CH2]',
        'quaternary_c': '[CX4]([C,c])([C,c])([C,c])[C,c]',
        'tertiary_c': '[CHX4]([C,c])([C,c])[C,c]',
        'hydroxyl': '[OX2H1][C,c]',
        'carbonyl': '[CX3H0](=[O])[#6]',
        'ether': '[C,c][OX2][C,c]',
        'methyl_terminal': '[CH3][C,c]',
        'gem_dimethyl': '[CH3][CX4]([CH3])([C,c])[C,c]',
    }
    
    def __init__(self):
        self.compiled_patterns = {}
        for name, smarts in self.FG_PATTERNS.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    self.compiled_patterns[name] = pattern
            except:
                pass
    
    def detect_groups(self, mol: Chem.Mol) -> Dict[str, List[Tuple[int, ...]]]:
        """
        Detect all functional groups in a molecule.
        
        Returns:
            Dictionary mapping group type to list of atom index tuples
        """
        groups = {}
        for name, pattern in self.compiled_patterns.items():
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                groups[name] = list(matches)
        return groups
    
    def get_group_membership(self, mol: Chem.Mol) -> np.ndarray:
        """
        Get functional group membership for each atom.
        
        Returns:
            (num_atoms, num_group_types) binary matrix
        """
        num_atoms = mol.GetNumAtoms()
        num_groups = len(self.FG_PATTERNS)
        membership = np.zeros((num_atoms, num_groups), dtype=np.float32)
        
        groups = self.detect_groups(mol)
        
        for group_idx, (name, _) in enumerate(self.FG_PATTERNS.items()):
            if name in groups:
                for match in groups[name]:
                    for atom_idx in match:
                        if atom_idx < num_atoms:
                            membership[atom_idx, group_idx] = 1.0
        
        return membership
    
    def get_group_centroids(self, mol: Chem.Mol, 
                            atom_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute centroid features for each functional group instance.
        
        This aggregates atom features within each group to create
        group-level representations.
        """
        groups = self.detect_groups(mol)
        centroids = {}
        
        for name, matches in groups.items():
            group_features = []
            for match in matches:
                # Average features of atoms in this group instance
                atom_feats = [atom_features[idx] for idx in match if idx < len(atom_features)]
                if atom_feats:
                    group_feat = torch.stack(atom_feats).mean(dim=0)
                    group_features.append(group_feat)
            
            if group_features:
                centroids[name] = torch.stack(group_features)
        
        return centroids



