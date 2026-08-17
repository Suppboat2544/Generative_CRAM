"""
Deep CRAM Feature Extractor
===========================
Combined 163D feature extraction.
"""

import numpy as np
import torch
from rdkit import Chem

from .atom_features import AtomFeatureExtractor
from .marine_features import MarineDOMFeatures
from .synthetic_features import SyntheticAccessibilityFeatures


class DeepCRAMFeatureExtractor:
    """
    Complete 163D feature extraction for CRAM molecules.
    
    Feature Breakdown:
    - Base features: 45D (positions 0-44)
    - Ring-Chain features: 20D (positions 45-64)
    - Stereochemistry: 12D (positions 65-76)
    - Marine: 15D (positions 77-91)
    - Synthetic: 28D (positions 92-119)
    - Terpenoid: 10D (positions 120-129)
    - Functional groups: 13D (positions 130-142)
    - Oxidation: 20D (positions 143-162)
    Total: 163D per atom
    """
    
    def __init__(self, feature_dim: int = 163):
        self.feature_dim = feature_dim
        
        self.atom_extractor = AtomFeatureExtractor()
        self.marine_extractor = MarineDOMFeatures()
        self.synthetic_extractor = SyntheticAccessibilityFeatures()
    
    def extract_features(self, mol) -> torch.Tensor:
        """
        Extract 163D features for all atoms.
        
        Args:
            mol: RDKit molecule or SMILES string
            
        Returns:
            Tensor of shape (num_atoms, 163)
        """
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if mol is None:
            return None
        
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None
        
        # Extract component features
        atom_feats = self.atom_extractor.get_features(mol)  # 45D
        marine_feats = self.marine_extractor.get_features(mol)  # 15D
        synthetic_feats = self.synthetic_extractor.get_features(mol)  # 28D
        
        # Additional features to reach 163D
        ring_chain = self._extract_ring_chain(mol)  # 20D
        stereo = self._extract_stereo(mol)  # 12D
        terpenoid = self._extract_terpenoid(mol)  # 10D
        functional = self._extract_functional(mol)  # 13D
        oxidation = self._extract_oxidation(mol)  # 20D
        
        # Concatenate all features
        features = np.concatenate([
            atom_feats,      # 45D
            ring_chain,      # 20D
            stereo,          # 12D
            marine_feats,    # 15D
            synthetic_feats, # 28D
            terpenoid,       # 10D
            functional,      # 13D
            oxidation        # 20D
        ], axis=1)
        
        # Ensure 163D
        if features.shape[1] < self.feature_dim:
            padding = np.zeros((features.shape[0], self.feature_dim - features.shape[1]))
            features = np.concatenate([features, padding], axis=1)
        elif features.shape[1] > self.feature_dim:
            features = features[:, :self.feature_dim]
        
        return torch.tensor(features, dtype=torch.float32)

    def extract(self, mol):
        """
        Compatibility wrapper used by dataset builders.

        Returns a dict with atom-level 163D features, or None if invalid.
        """
        atom_features = self.extract_features(mol)
        if atom_features is None:
            return None
        if torch.is_tensor(atom_features):
            atom_np = atom_features.detach().cpu().numpy()
        else:
            atom_np = np.asarray(atom_features)
        n_atoms = atom_np.shape[0]
        return {
            "atom_features": atom_np,
            "group_membership": np.zeros((n_atoms, 1), dtype=np.float32),
            "mol_features": np.zeros(17, dtype=np.float32),
        }

    def extract_all_features(self, mol):
        """Alias used by the CLI feature command."""
        return self.extract(mol)
    
    def _extract_ring_chain(self, mol) -> np.ndarray:
        """Extract ring-chain features (20D)."""
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 20), dtype=np.float32)
        
        ring_info = mol.GetRingInfo()
        
        for i, atom in enumerate(mol.GetAtoms()):
            # Ring sizes 3-8 (6D)
            for size in range(3, 9):
                if atom.IsInRingSize(size):
                    features[i, size - 3] = 1.0
            
            # Number of rings containing this atom (normalized)
            rings_count = sum(1 for r in ring_info.AtomRings() if i in r)
            features[i, 6] = min(rings_count / 3.0, 1.0)
            
            # Is bridgehead
            features[i, 7] = 1.0 if rings_count > 1 else 0.0
            
            # Chain length features
            if not atom.IsInRing():
                chain_length = self._get_chain_length(mol, i)
                features[i, 8] = min(chain_length / 10.0, 1.0)
            
            # Additional ring features
            features[i, 9] = 1.0 if atom.IsInRing() and not atom.GetIsAromatic() else 0.0
            features[i, 10] = 1.0 if atom.GetIsAromatic() else 0.0
        
        return features
    
    def _extract_stereo(self, mol) -> np.ndarray:
        """Extract stereochemistry features (12D)."""
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 12), dtype=np.float32)
        
        for i, atom in enumerate(mol.GetAtoms()):
            chiral = atom.GetChiralTag()
            features[i, 0] = 1.0 if chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CW else 0.0
            features[i, 1] = 1.0 if chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CCW else 0.0
            features[i, 2] = 1.0 if chiral != Chem.ChiralType.CHI_UNSPECIFIED else 0.0
            
            # Degree of chirality (number of different neighbors)
            neighbor_types = set(n.GetAtomicNum() for n in atom.GetNeighbors())
            features[i, 3] = len(neighbor_types) / 4.0
        
        return features
    
    def _extract_terpenoid(self, mol) -> np.ndarray:
        """Extract terpenoid features (10D)."""
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 10), dtype=np.float32)
        
        # Terpenoid patterns
        patterns = {
            'isoprene': 'CC(=C)C',
            'cyclohexane': 'C1CCCCC1',
            'cyclopentane': 'C1CCCC1',
            'methyl_branch': 'CC(C)',
        }
        
        for idx, (name, smarts) in enumerate(patterns.items()):
            if idx >= 10:
                break
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    matches = mol.GetSubstructMatches(pat)
                    for match in matches:
                        for atom_idx in match:
                            if atom_idx < num_atoms:
                                features[atom_idx, idx] = 1.0
            except:
                pass
        
        return features
    
    def _extract_functional(self, mol) -> np.ndarray:
        """Extract functional group features (13D)."""
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 13), dtype=np.float32)
        
        patterns = {
            'cooh': '[CX3](=O)[OX2H1]',
            'coo_minus': '[CX3](=O)[O-]',
            'oh': '[OX2H]',
            'nh2': '[NX3H2]',
            'nh': '[NX3H1]',
            'carbonyl': '[CX3]=O',
            'ester': '[CX3](=O)[OX2][C]',
            'amide': '[CX3](=O)[NX3]',
            'sulfone': '[SX4](=O)(=O)',
            'phosphate': '[PX4](=O)',
        }
        
        for idx, (name, smarts) in enumerate(patterns.items()):
            if idx >= 13:
                break
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    matches = mol.GetSubstructMatches(pat)
                    for match in matches:
                        for atom_idx in match:
                            if atom_idx < num_atoms:
                                features[atom_idx, idx] = 1.0
            except:
                pass
        
        return features
    
    def _extract_oxidation(self, mol) -> np.ndarray:
        """Extract oxidation state features (20D)."""
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 20), dtype=np.float32)
        
        for i, atom in enumerate(mol.GetAtoms()):
            # Oxidation proxies
            if atom.GetSymbol() == 'C':
                o_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == 'O')
                features[i, 0] = min(o_neighbors / 3.0, 1.0)
                
                # Double bonded to O
                for bond in atom.GetBonds():
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        other = bond.GetOtherAtom(atom)
                        if other.GetSymbol() == 'O':
                            features[i, 1] = 1.0
            
            if atom.GetSymbol() == 'O':
                features[i, 2] = 1.0
                
            if atom.GetSymbol() == 'N':
                features[i, 3] = 1.0
                
            if atom.GetSymbol() == 'S':
                features[i, 4] = 1.0
        
        return features
    
    def _get_chain_length(self, mol, atom_idx: int, visited: set = None) -> int:
        """Get length of chain from atom."""
        if visited is None:
            visited = set()
        
        if atom_idx in visited:
            return 0
        
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.IsInRing():
            return 0
        
        visited.add(atom_idx)
        
        max_length = 0
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx not in visited and not neighbor.IsInRing():
                length = self._get_chain_length(mol, n_idx, visited)
                max_length = max(max_length, length)
        
        return max_length + 1
    
    def get_adjacency(self, mol) -> torch.Tensor:
        """Get edge index for graph."""
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if mol is None:
            return None
        
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append([i, j])
            edges.append([j, i])
        
        if len(edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()


# Singleton instance
deep_cram_extractor = DeepCRAMFeatureExtractor()
