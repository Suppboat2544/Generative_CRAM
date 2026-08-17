"""
Marine DOM Features
===================
Marine-specific features (15D per atom).
"""

import numpy as np
from rdkit import Chem


class MarineDOMFeatures:
    """
    Marine DOM-specific features (15D per atom).
    
    Captures patterns common in marine dissolved organic matter.
    """
    
    PATTERNS = {
        'carboxyl': '[CX3](=O)[OX2H1,OX1-]',
        'carboxylate': '[CX3](=O)[O-]',
        'hydroxyl': '[OX2H]',
        'ether': '[OX2]([C])[C]',
        'carbonyl': '[CX3]=O',
        'methoxy': '[CH3][OX2]',
        'alicyclic_5': '[R1;!a]1~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]1',
        'alicyclic_6': '[R1;!a]1~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]1',
        'alicyclic_7': '[R1;!a]1~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]1',
        'fused_ring': '[R2]',
        'spiro': '[R3]',
    }
    
    def __init__(self):
        self.feature_dim = 15
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
        Extract 15D marine features for each atom.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Features array of shape (num_atoms, 15)
        """
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, self.feature_dim), dtype=np.float32)
        
        # Pattern-based features (0-10)
        for i, (name, pattern) in enumerate(self.compiled.items()):
            if i >= 11:
                break
            try:
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    for idx in match:
                        if idx < num_atoms:
                            features[idx, i] = 1.0
            except:
                pass
        
        # Atom-level marine features (11-14)
        for i, atom in enumerate(mol.GetAtoms()):
            # O/C ratio indicator
            if atom.GetSymbol() == 'O':
                features[i, 11] = 1.0
            
            # Aliphatic carbon
            if atom.GetSymbol() == 'C' and not atom.GetIsAromatic():
                features[i, 12] = 1.0
            
            # Ring membership in non-aromatic ring
            if atom.IsInRing() and not atom.GetIsAromatic():
                features[i, 13] = 1.0
            
            # Connected to carboxyl
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'C':
                    for nn in neighbor.GetNeighbors():
                        if nn.GetSymbol() == 'O':
                            features[i, 14] = 1.0
                            break
        
        return features
