"""Notebook DeepCRAMFeatureExtractor — 163D per atom, same layout as the Colab notebook."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

try:
    import dgl
except ImportError:
    dgl = None

from .enhanced_nodes import (
    get_base_node_features_45D,
    get_synthetic_features_28D,
    get_marine_features_15D,
    get_cram_ring_chain_features_20D,
)
from .oxidation import EnhancedOxidationFeatures
from .terpenoid import TerpenoidAncestryDetector
from .functional_groups import FunctionalGroupDetector
from .stereochemistry import get_stereochemistry_features_12D, get_conformational_stability_score
from ..models.ring_chain import SubstructureDetector

"""
=============================================================================
COMPLETE ENHANCED FEATURE INTEGRATION (163D + MOLECULAR FEATURES)
=============================================================================

This module combines ALL deep improvements into one cohesive feature pipeline.

Feature architecture (per atom):
├── Base chemical features (45D)
│   └── Element, hybridization, bonds, charge, aromatic, ring info
├── Synthetic accessibility (28D)
│   └── Complexity, accessibility contributions
├── Marine DOM features (15D)
│   └── H/C, O/C ratios, DBE, marine-specific patterns
├── CRAM ring-chain features (20D)
│   └── Alicyclic rings, junctions, carboxyl proximity, chain character
├── Stereochemistry features (12D)
│   └── Chirality, ring conformation, strain, geometry
├── Terpenoid ancestry (10D)
│   └── Isoprene patterns, terpene signatures, oxidation level
├── Functional group membership (13D)
│   └── Which functional groups contain this atom
├── Enhanced oxidation features (20D)
│   └── Oxidation state, COOH proximity (α/β/γ/δ), hydroxyl/carbonyl types
└── TOTAL: 163D per atom + molecular-level features

Molecular-level features (used for property conditioning):
├── Terpenoid molecular features (5D)
├── Photostability features (6D)
├── Conformational stability (1D)
└── CRAM classification features (5D)
"""

class DeepCRAMFeatureExtractor:
    """
    Complete feature extraction for deep CRAM understanding.
    
    This class unifies all feature extraction modules into a single
    pipeline that produces publication-quality molecular representations.
    """
    
    # Feature dimensions
    FEATURE_DIMS = {
        'base': 45,
        'synthetic': 28,
        'marine': 15,
        'ring_chain': 20,
        'stereochemistry': 12,
        'terpenoid': 10,
        'functional_group': 13,
        'enhanced_oxidation': 20,  # Added enhanced oxidation features
    }
    TOTAL_ATOM_DIM = sum(FEATURE_DIMS.values())  # 163D
    
    def __init__(self, verbose: bool = False):
        self.terpenoid_detector = TerpenoidAncestryDetector()
        self.fg_detector = FunctionalGroupDetector()
        try:
            self.oxidation_extractor = EnhancedOxidationFeatures()
        except Exception:
            self.oxidation_extractor = None
        if verbose:
            print(f"DeepCRAMFeatureExtractor {self.TOTAL_ATOM_DIM}D")
    
    def extract_all_features(self, mol: Chem.Mol) -> Dict[str, np.ndarray]:
        """
        Extract complete feature set for a molecule.
        
        Returns:
            Dictionary containing:
            - 'atom_features': (num_atoms, 163) per-atom features
            - 'mol_features': (17,) molecular-level features
            - 'group_membership': (num_atoms, 13) functional group masks
            - 'substructure_masks': Dict of boolean masks
        """
        num_atoms = mol.GetNumAtoms()
        
        # Initialize feature arrays
        features = {
            'atom_features': np.zeros((num_atoms, self.TOTAL_ATOM_DIM), dtype=np.float32),
            'mol_features': np.zeros(17, dtype=np.float32),
            'group_membership': np.zeros((num_atoms, 13), dtype=np.float32),
            'substructure_masks': {}
        }
        
        # Track current position in feature vector
        idx = 0
        
        # === 1. Base chemical features (45D) ===
        try:
            base_features = np.asarray(get_base_node_features_45D(mol), dtype=np.float32)
            features['atom_features'][:, idx:idx+45] = base_features
        except Exception as e:
            print(f"Warning: Base features failed: {e}")
        idx += 45
        
        # === 2. Synthetic accessibility (28D) ===
        try:
            sa_features = np.array(
                [get_synthetic_features_28D(mol, i) for i in range(num_atoms)],
                dtype=np.float32,
            )
            features['atom_features'][:, idx:idx+28] = sa_features
        except Exception as e:
            print(f"Warning: Synthetic features failed: {e}")
        idx += 28
        
        # === 3. Marine DOM features (15D) ===
        try:
            marine_features = np.array(
                [get_marine_features_15D(mol, i) for i in range(num_atoms)],
                dtype=np.float32,
            )
            features['atom_features'][:, idx:idx+15] = marine_features
        except Exception as e:
            print(f"Warning: Marine features failed: {e}")
        idx += 15
        
        # === 4. CRAM ring-chain features (20D) ===
        try:
            ring_chain = np.array(
                [get_cram_ring_chain_features_20D(mol, i) for i in range(num_atoms)],
                dtype=np.float32,
            )
            features['atom_features'][:, idx:idx+20] = ring_chain
        except Exception as e:
            print(f"Warning: Ring-chain features failed: {e}")
        idx += 20
        
        # === 5. Stereochemistry features (12D) ===
        try:
            features['atom_features'][:, idx:idx+12] = get_stereochemistry_features_12D(mol)
        except Exception as e:
            print(f"Warning: Stereochemistry features failed: {e}")
        idx += 12
        
        # === 6. Terpenoid ancestry (10D per atom) ===
        try:
            mol_terp, atom_terp = self.terpenoid_detector.get_terpenoid_features_10D(mol)
            features['atom_features'][:, idx:idx+10] = atom_terp
            features['mol_features'][0:5] = mol_terp[:5]  # Molecular terpenoid features
        except Exception as e:
            print(f"Warning: Terpenoid features failed: {e}")
        idx += 10
        
        # === 7. Functional group membership (13D) ===
        try:
            fg_membership = self.fg_detector.get_group_membership(mol)
            features['atom_features'][:, idx:idx+13] = fg_membership
            features['group_membership'] = fg_membership
        except Exception as e:
            print(f"Warning: Functional group features failed: {e}")
        idx += 13
        
        # === 8. Enhanced oxidation features (20D) ===
        try:
            if self.oxidation_extractor is not None:
                oxidation_features = self.oxidation_extractor.get_molecule_features(mol)
                features['atom_features'][:, idx:idx+20] = oxidation_features
            else:
                # Fill with zeros if oxidation extractor not available
                features['atom_features'][:, idx:idx+20] = 0.0
        except Exception as e:
            print(f"Warning: Enhanced oxidation features failed: {e}")
            # Still fill with zeros to maintain dimension
            features['atom_features'][:, idx:idx+20] = 0.0
        idx += 20
        
        # === Molecular-level features ===
        try:
            # Basic molecular descriptors
            features['mol_features'][0] = Descriptors.MolWt(mol) / 500.0  # Normalized MW
            features['mol_features'][1] = Descriptors.MolLogP(mol) / 10.0  # Normalized LogP
            features['mol_features'][2] = Descriptors.TPSA(mol) / 200.0  # Normalized TPSA
            features['mol_features'][3] = mol.GetNumHeavyAtoms() / 50.0  # Normalized atom count
            features['mol_features'][4] = Descriptors.NumRotatableBonds(mol) / 20.0
            
            # Additional molecular features (fill with basic descriptors)
            features['mol_features'][5] = Descriptors.NumHDonors(mol) / 10.0
            features['mol_features'][6] = Descriptors.NumHAcceptors(mol) / 10.0
            features['mol_features'][7] = Descriptors.NumAromaticRings(mol) / 5.0
            features['mol_features'][8] = Descriptors.NumAliphaticRings(mol) / 5.0
            
            # Calculate fraction of carbons manually (FractionCarbons doesn't exist in RDKit)
            num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
            features['mol_features'][9] = num_carbons / max(mol.GetNumAtoms(), 1)
            
            # Calculate fraction of oxygens manually
            num_oxygens = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
            features['mol_features'][10] = num_oxygens / max(mol.GetNumAtoms(), 1)
            features['mol_features'][11] = float(get_conformational_stability_score(mol))
        except Exception as e:
            print(f"Warning: Molecular features failed: {e}")
        
        # === Substructure masks ===
        # Always initialize with empty masks to avoid None issues
        for name in ['alicyclic_ring', 'chain', 'junction', 'carboxyl']:
            features['substructure_masks'][name] = np.zeros(num_atoms, dtype=bool)
        
        try:
            substructures = SubstructureDetector.detect_substructures(mol)
            for name in ['alicyclic_ring', 'chain', 'junction', 'carboxyl']:
                if name in substructures and substructures[name]:
                    indices = substructures[name]
                    mask = np.zeros(num_atoms, dtype=bool)
                    mask[list(indices)] = True
                    features['substructure_masks'][name] = mask
        except Exception as e:
            print(f"Warning: Substructure masks failed: {e}")
            # Masks already initialized to empty above
        
        return features
    
    def extract_for_training(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Extract features in format ready for training.
        
        Returns:
            atom_features: (num_atoms, 143) tensor
            mol_features: (17,) tensor
            metadata: Dictionary with masks and labels
        """
        features = self.extract_all_features(mol)
        
        atom_features = torch.tensor(features['atom_features'], dtype=torch.float32)
        mol_features = torch.tensor(features['mol_features'], dtype=torch.float32)
        
        metadata = {
            'group_membership': torch.tensor(features['group_membership'], dtype=torch.float32),
            'substructure_masks': {
                k: torch.tensor(v, dtype=torch.bool) 
                for k, v in features['substructure_masks'].items()
            }
        }
        
        return atom_features, mol_features, metadata



    def extract(self, mol):
        """Dataset/CLI wrapper: dict with atom_features, mol_features, group_membership."""
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        feats = self.extract_all_features(mol)
        return feats

    def extract_features(self, mol):
        out = self.extract(mol)
        if out is None:
            return None
        return torch.as_tensor(out["atom_features"], dtype=torch.float32)



def smiles_to_graph_deep_cram(smiles: str, 
                               feature_extractor: DeepCRAMFeatureExtractor = None
                               ) -> Optional[dgl.DGLGraph]:
    """
    Convert SMILES to DGL graph with complete deep CRAM features.
    
    This is the production-ready graph conversion function that:
    1. Extracts all 143D atom features
    2. Adds molecular-level features as graph-level data
    3. Includes functional group membership for attention
    4. Adds substructure masks for targeted learning
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    if feature_extractor is None:
        feature_extractor = DeepCRAMFeatureExtractor()
    
    # Extract complete features
    features = feature_extractor.extract_all_features(mol)
    
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None

    # Build graph edges (skip self-bonds; add torch.int64 self-loops)
    edges_src = []
    edges_dst = []
    edge_types = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i == j:
            continue

        edges_src.extend([i, j])
        edges_dst.extend([j, i])

        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            edge_types.extend([0, 0])
        elif bond_type == Chem.BondType.DOUBLE:
            edge_types.extend([1, 1])
        elif bond_type == Chem.BondType.TRIPLE:
            edge_types.extend([2, 2])
        else:
            edge_types.extend([3, 3])

    for i in range(num_atoms):
        edges_src.append(i)
        edges_dst.append(i)
        edge_types.append(0)

    g = dgl.graph(
        (torch.tensor(edges_src, dtype=torch.int64),
         torch.tensor(edges_dst, dtype=torch.int64)),
        num_nodes=num_atoms,
    )

    g.ndata['feat'] = torch.tensor(features['atom_features'], dtype=torch.float32)
    g.ndata['group'] = torch.tensor(features['group_membership'], dtype=torch.float32)

    for mask_name, mask in features['substructure_masks'].items():
        g.ndata[f'mask_{mask_name}'] = torch.tensor(mask, dtype=torch.bool)

    g.edata['type'] = torch.tensor(edge_types, dtype=torch.long)
    g.mol_features = torch.tensor(features['mol_features'], dtype=torch.float32)

    return g


def smiles_to_graph(smiles, mode='unified', feature_extractor=None):
    """Canonical SMILES → DGL graph (163D Deep CRAM features)."""
    return smiles_to_graph_deep_cram(smiles, feature_extractor)


deep_cram_extractor = None