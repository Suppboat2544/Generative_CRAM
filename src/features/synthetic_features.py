"""
Synthetic Accessibility Features
================================
Synthetic accessibility features (28D per atom).
"""

import numpy as np
from rdkit import Chem


class SyntheticAccessibilityFeatures:
    """
    Synthetic accessibility features (28D per atom).
    
    Captures patterns related to synthetic complexity.
    """
    
    PATTERNS = {
        'bridgehead': '[x3,x4](@*)(@*)@*',
        'spiro': '[R2](@*)(@*)(@*)@*',
        'fused_ring': '[R2]',
        'macrocycle': '[r12,r13,r14,r15,r16]',
        'chiral_center': '[C@,C@@]',
        'double_bond': 'C=C',
        'triple_bond': 'C#C',
        'halogen': '[F,Cl,Br,I]',
        'nitro': '[N+](=O)[O-]',
        'azide': 'N=[N+]=[N-]',
        'epoxide': 'C1OC1',
        'aziridine': 'C1NC1',
        'peroxide': 'OO',
        'gem_dihalide': '[CX4]([F,Cl,Br,I])([F,Cl,Br,I])',
    }
    
    def __init__(self):
        self.feature_dim = 28
        self.compiled = {}
        
        for name, smarts in self.PATTERNS.items():
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self.compiled[name] = pat
            except:
                pass
    
    def get_features(self, mol: Chem.Mol) -> np.ndarray:
        """
        Extract 28D synthetic features for each atom.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Features array of shape (num_atoms, 28)
        """
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, self.feature_dim), dtype=np.float32)
        
        # Pattern-based features (0-13)
        for i, (name, pattern) in enumerate(self.compiled.items()):
            if i >= 14:
                break
            try:
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    for idx in match:
                        if idx < num_atoms:
                            features[idx, i] = 1.0
            except:
                pass
        
        # Atom-level synthetic features (14-27)
        for i, atom in enumerate(mol.GetAtoms()):
            features[i, 14] = min(atom.GetDegree() / 4.0, 1.0)
            features[i, 15] = 1.0 if atom.GetHybridization() == Chem.HybridizationType.SP3 else 0.0
            features[i, 16] = 1.0 if atom.GetHybridization() == Chem.HybridizationType.SP2 else 0.0
            features[i, 17] = 1.0 if atom.GetHybridization() == Chem.HybridizationType.SP else 0.0
            
            hetero_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() not in [1, 6])
            features[i, 18] = min(hetero_neighbors / 3.0, 1.0)
            
            features[i, 19] = 1.0 if atom.IsInRing() else 0.0
            features[i, 20] = 1.0 if atom.IsInRingSize(3) else 0.0
            features[i, 21] = 1.0 if atom.IsInRingSize(4) else 0.0
            features[i, 22] = 1.0 if atom.IsInRingSize(5) else 0.0
            features[i, 23] = 1.0 if atom.IsInRingSize(6) else 0.0
            features[i, 24] = 1.0 if atom.IsInRingSize(7) else 0.0
            features[i, 25] = 1.0 if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED else 0.0
            features[i, 26] = min(atom.GetNumRadicalElectrons() / 2.0, 1.0)
            features[i, 27] = min(abs(atom.GetFormalCharge()) / 2.0, 1.0)
        
        return features
