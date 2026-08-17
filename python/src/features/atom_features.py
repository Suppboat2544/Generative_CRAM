"""
Atom Feature Extractor
======================
Base atom-level features (45D).
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


class AtomFeatureExtractor:
    """
    Extract base atom features (45D per atom).
    
    Features:
    - Atom type one-hot (14D)
    - Hybridization one-hot (6D)
    - Formal charge one-hot (5D)
    - Ring size one-hot (7D)
    - Basic properties (10D)
    - Gasteiger charge (1D)
    - Connectivity (2D)
    """
    
    ATOM_TYPES = ['C', 'H', 'O', 'N', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'Se', 'Unknown']
    HYBRIDIZATIONS = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ]
    FORMAL_CHARGES = [-2, -1, 0, 1, 2]
    
    ELECTRONEGATIVITY = {
        'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
        'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
        'Si': 1.90, 'B': 2.04, 'Se': 2.55
    }
    
    def __init__(self):
        self.feature_dim = 45
    
    def get_features(self, mol: Chem.Mol) -> np.ndarray:
        """
        Extract 45D features for each atom.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Features array of shape (num_atoms, 45)
        """
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, self.feature_dim), dtype=np.float32)
        
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except:
            pass
        
        ring_info = mol.GetRingInfo()
        
        for i, atom in enumerate(mol.GetAtoms()):
            idx = 0
            
            # Atom type (14D)
            symbol = atom.GetSymbol()
            atom_type_idx = self.ATOM_TYPES.index(symbol) if symbol in self.ATOM_TYPES else len(self.ATOM_TYPES) - 1
            features[i, atom_type_idx] = 1.0
            idx += len(self.ATOM_TYPES)
            
            # Hybridization (6D)
            hyb = atom.GetHybridization()
            hyb_idx = self.HYBRIDIZATIONS.index(hyb) if hyb in self.HYBRIDIZATIONS else len(self.HYBRIDIZATIONS) - 1
            features[i, idx + hyb_idx] = 1.0
            idx += len(self.HYBRIDIZATIONS)
            
            # Formal charge (5D)
            charge = atom.GetFormalCharge()
            charge_idx = self.FORMAL_CHARGES.index(charge) if charge in self.FORMAL_CHARGES else 2
            features[i, idx + charge_idx] = 1.0
            idx += len(self.FORMAL_CHARGES)
            
            # Ring size (7D) - sizes 3 to 9
            ring_sizes = [len(r) for r in ring_info.AtomRings() if i in r]
            for size in ring_sizes:
                if 3 <= size <= 9:
                    features[i, idx + size - 3] = 1.0
            idx += 7
            
            # Basic properties (10D)
            features[i, idx] = float(atom.GetIsAromatic())
            features[i, idx + 1] = float(atom.IsInRing())
            features[i, idx + 2] = min(atom.GetDegree() / 4.0, 1.0)
            features[i, idx + 3] = min(atom.GetTotalNumHs() / 4.0, 1.0)
            features[i, idx + 4] = atom.GetNumRadicalElectrons()
            features[i, idx + 5] = self.ELECTRONEGATIVITY.get(symbol, 2.5) / 4.0
            features[i, idx + 6] = float(atom.GetIsotope()) / 100.0
            features[i, idx + 7] = atom.GetMass() / 200.0
            features[i, idx + 8] = float(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
            features[i, idx + 9] = atom.GetImplicitValence() / 4.0
            idx += 10
            
            # Gasteiger charge (1D)
            try:
                gasteiger = float(atom.GetProp('_GasteigerCharge'))
                if np.isnan(gasteiger) or np.isinf(gasteiger):
                    gasteiger = 0.0
            except:
                gasteiger = 0.0
            features[i, idx] = gasteiger
            idx += 1
            
            # Connectivity (2D)
            features[i, idx] = min(len(list(atom.GetNeighbors())) / 4.0, 1.0)
            features[i, idx + 1] = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() not in [1, 6]) / 4.0
        
        return features
