"""Notebook TerpenoidAncestryDetector (10D)."""
from __future__ import annotations
from typing import Tuple
import numpy as np
from rdkit import Chem

"""
=============================================================================
CRITICAL IMPROVEMENT 2: TERPENOID ANCESTRY DETECTION (10D)
=============================================================================

This module detects signatures of terpenoid origin - crucial for understanding
marine CRAM because:

1. Marine phytoplankton produce abundant terpenoids
2. Terpenoid degradation is a major source of oceanic CRAM
3. Terpenoid-derived CRAM has characteristic structural patterns
4. Isoprene rule patterns help distinguish biogenic from abiogenic structures

Key terpenoid signatures:
- Isoprene backbone patterns (C5 repeats)
- Gem-dimethyl groups (characteristic of terpenes)
- Specific ring sizes (5, 6, 7 common in terpenes)
- Methyl branching at characteristic positions
"""

class TerpenoidAncestryDetector:
    """
    Detects terpenoid ancestry in molecules - key for marine CRAM origins.
    
    Marine DOM (especially CRAM) often derives from:
    1. Carotenoid degradation (C40 → smaller fragments)
    2. Steroid degradation (C27-29 → ring fragments)
    3. Monoterpene oxidation (C10 → carboxylated products)
    4. Diterpene fragmentation (C20 → smaller units)
    """
    
    # SMARTS patterns for terpenoid signatures
    TERPENOID_PATTERNS = {
        # Isoprene-derived patterns
        'gem_dimethyl': '[CH3][C]([CH3])([!H])[!H]',  # Gem-dimethyl (very common in terpenes)
        'isopropyl_terminal': '[CH3][CH]([CH3])[C,c]',  # Isopropyl end group
        'isoprene_backbone': '[CH2][C]([CH3])=[CH][CH2]',  # Isoprene unit
        'methyl_branch': '[CH3][CH]([C,c])[C,c]',  # Methyl branching
        
        # Ring patterns common in terpenoids
        'cyclopentane': 'C1CCCC1',  # 5-membered ring
        'cyclohexane': 'C1CCCCC1',  # 6-membered ring  
        'cycloheptane': 'C1CCCCCC1',  # 7-membered ring
        'decalin': 'C1CCC2CCCCC2C1',  # Fused 6-6 system (steroid-like)
        'hydrindane': 'C1CCC2CCCC2C1',  # Fused 5-6 system
        
        # Oxidized terpenoid patterns (CRAM-like)
        'oxidized_isoprene': '[C](=O)[C]([CH3])([OH,O-])',  # Oxidized terpene
        'carboxylated_cyclopentane': '[C](=O)[OH,O-]~[C]1CCCC1',  # COOH on 5-ring
        'carboxylated_cyclohexane': '[C](=O)[OH,O-]~[C]1CCCCC1',  # COOH on 6-ring
    }
    
    def __init__(self):
        self.compiled_patterns = {}
        for name, smarts in self.TERPENOID_PATTERNS.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    self.compiled_patterns[name] = pattern
            except:
                pass
    
    def get_terpenoid_features_10D(self, mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract terpenoid ancestry features.
        
        Returns:
            mol_features: (10,) molecular-level terpenoid features
            atom_features: (num_atoms, 10) per-atom terpenoid features
        """
        num_atoms = mol.GetNumAtoms()
        mol_features = np.zeros(10, dtype=np.float32)
        atom_features = np.zeros((num_atoms, 10), dtype=np.float32)
        
        # === Molecular-level features (0-4) ===
        
        # [0] Isoprene rule compliance: C count divisible by 5 (or close)
        c_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        mol_features[0] = 1.0 - min(c_count % 5, 5 - (c_count % 5)) / 2.5
        
        # [1] Gem-dimethyl count (normalized)
        if 'gem_dimethyl' in self.compiled_patterns:
            matches = mol.GetSubstructMatches(self.compiled_patterns['gem_dimethyl'])
            mol_features[1] = min(len(matches) / 3.0, 1.0)
        
        # [2] Terpenoid ring patterns
        ring_score = 0
        for pattern_name in ['cyclopentane', 'cyclohexane', 'decalin', 'hydrindane']:
            if pattern_name in self.compiled_patterns:
                if mol.HasSubstructMatch(self.compiled_patterns[pattern_name]):
                    ring_score += 0.25
        mol_features[2] = min(ring_score, 1.0)
        
        # [3] Methyl branching index (characteristic of terpenes)
        if 'methyl_branch' in self.compiled_patterns:
            matches = mol.GetSubstructMatches(self.compiled_patterns['methyl_branch'])
            mol_features[3] = min(len(matches) / 4.0, 1.0)
        
        # [4] Oxidized terpenoid patterns (CRAM-like)
        oxidized_score = 0
        for pattern_name in ['oxidized_isoprene', 'carboxylated_cyclopentane', 'carboxylated_cyclohexane']:
            if pattern_name in self.compiled_patterns:
                if mol.HasSubstructMatch(self.compiled_patterns[pattern_name]):
                    oxidized_score += 0.33
        mol_features[4] = min(oxidized_score, 1.0)
        
        # === Per-atom features (5-9) ===
        
        # Mark atoms involved in terpenoid patterns
        for pattern_name, pattern in self.compiled_patterns.items():
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                for atom_idx in match:
                    if atom_idx < num_atoms:
                        if 'gem_dimethyl' in pattern_name or 'methyl' in pattern_name:
                            atom_features[atom_idx, 5] = 1.0  # Terpenoid branching
                        elif 'cyclo' in pattern_name or 'decalin' in pattern_name:
                            atom_features[atom_idx, 6] = 1.0  # Terpenoid ring
                        elif 'isoprene' in pattern_name:
                            atom_features[atom_idx, 7] = 1.0  # Isoprene backbone
                        elif 'carboxylated' in pattern_name or 'oxidized' in pattern_name:
                            atom_features[atom_idx, 8] = 1.0  # Oxidized terpene
        
        # [9] Combined terpenoid ancestry score per atom
        atom_features[:, 9] = np.clip(
            atom_features[:, 5:9].sum(axis=1) / 4.0, 0, 1
        )
        
        return mol_features, atom_features
    
    def estimate_terpenoid_class(self, mol: Chem.Mol) -> str:
        """
        Estimate the likely terpenoid class based on carbon count and structure.
        """
        c_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        
        if c_count <= 12:
            base_class = "monoterpenoid"
        elif c_count <= 17:
            base_class = "sesquiterpenoid"  
        elif c_count <= 22:
            base_class = "diterpenoid"
        elif c_count <= 27:
            base_class = "sesterterpenoid"
        elif c_count <= 32:
            base_class = "triterpenoid"
        else:
            base_class = "polyterpenoid"
        
        # Check for ring patterns to refine
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        
        if num_rings >= 4:
            return f"{base_class}_polycyclic"
        elif num_rings >= 2:
            return f"{base_class}_bicyclic"
        elif num_rings == 1:
            return f"{base_class}_monocyclic"
        else:
            return f"{base_class}_acyclic"


def get_marine_photodegradation_features_6D(mol: Chem.Mol) -> np.ndarray:
    """
    Features related to marine photodegradation stability - key for CRAM persistence.
    
    CRAM persists in oceans because it RESISTS photodegradation.
    This function estimates photostability based on structural features.
    
    Features (6D):
    [0] Aromatic content (photolabile)
    [1] Conjugated system extent (photolabile)
    [2] Saturated carbon fraction (photostable)
    [3] Carboxyl protection level
    [4] Ring protection (buried in ring systems)
    [5] Overall photostability score
    """
    features = np.zeros(6, dtype=np.float32)
    num_atoms = mol.GetNumAtoms()
    
    if num_atoms == 0:
        return features
    
    # [0] Aromatic content (photolabile - high absorption leads to degradation)
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    features[0] = aromatic_atoms / num_atoms
    
    # [1] Conjugated system (more conjugation = more photolabile)
    conjugated_bonds = sum(1 for bond in mol.GetBonds() 
                          if bond.GetIsConjugated())
    features[1] = min(conjugated_bonds / max(mol.GetNumBonds(), 1), 1.0)
    
    # [2] Saturated carbon fraction (photostable)
    sp3_carbons = sum(1 for atom in mol.GetAtoms() 
                     if atom.GetAtomicNum() == 6 and 
                     atom.GetHybridization() == Chem.HybridizationType.SP3)
    total_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    features[2] = sp3_carbons / max(total_carbons, 1)
    
    # [3] Carboxyl protection - COOH groups that are protected by neighboring groups
    cooh_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
    if cooh_pattern:
        cooh_matches = mol.GetSubstructMatches(cooh_pattern)
        if cooh_matches:
            protected_count = 0
            for match in cooh_matches:
                c_idx = match[0]
                c_atom = mol.GetAtomWithIdx(c_idx)
                # Check if COOH is attached to ring (protected)
                for neighbor in c_atom.GetNeighbors():
                    if neighbor.IsInRing():
                        protected_count += 1
                        break
            features[3] = protected_count / len(cooh_matches)
    
    # [4] Ring protection - atoms buried in fused ring systems
    ring_info = mol.GetRingInfo()
    atoms_in_multiple_rings = sum(1 for i in range(num_atoms) 
                                  if ring_info.NumAtomRings(i) > 1)
    features[4] = atoms_in_multiple_rings / num_atoms
    
    # [5] Overall photostability score
    # High saturated fraction + low aromatic + protected COOH = stable
    features[5] = (features[2] * 0.4 +  # Saturated (positive)
                   (1 - features[0]) * 0.3 +  # Not aromatic (positive)
                   features[3] * 0.2 +  # Protected COOH (positive)
                   features[4] * 0.1)  # Ring protection (positive)
    
    return features



