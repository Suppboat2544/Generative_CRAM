"""
Atom Feature Extractor
======================
Notebook 45D base atom features (get_base_node_features_45D).
"""

import numpy as np
from rdkit import Chem


class AtomFeatureExtractor:
    """Extract notebook 45D base features per atom."""

    def __init__(self):
        self.feature_dim = 45

    def get_features(self, mol: Chem.Mol) -> np.ndarray:
        from .enhanced_nodes import get_base_node_features_45D
        return np.asarray(get_base_node_features_45D(mol), dtype=np.float32)
