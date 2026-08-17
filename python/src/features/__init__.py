"""
CRAM Features Package
=====================
163D per-atom features matching the Colab notebook layout:

base 45 + synthetic 28 + marine 15 + ring-chain 20 + stereo 12
+ terpenoid 10 + functional group 13 + enhanced oxidation 20 = 163
"""

from .atom_features import AtomFeatureExtractor
from .marine_features import MarineDOMFeatures
from .synthetic_features import SyntheticAccessibilityFeatures
from .oxidation import EnhancedOxidationFeatures
from .terpenoid import TerpenoidAncestryDetector
from .functional_groups import FunctionalGroupDetector
from .extractor import DeepCRAMFeatureExtractor, smiles_to_graph_deep_cram, smiles_to_graph
from .stereochemistry import (
    get_stereochemistry_features_12D,
    get_conformational_stability_score,
    get_3d_conformer_features_8D,
)
from .template import TemplateExtractor
from .constraints import ConstraintSystem
from .enhanced_nodes import (
    get_base_node_features_45D,
    get_synthetic_features_28D,
    get_marine_features_15D,
    get_cram_ring_chain_features_20D,
    get_enhanced_node_features,
    get_enhanced_node_features_with_cram,
)

__all__ = [
    "AtomFeatureExtractor",
    "MarineDOMFeatures",
    "SyntheticAccessibilityFeatures",
    "EnhancedOxidationFeatures",
    "TerpenoidAncestryDetector",
    "FunctionalGroupDetector",
    "DeepCRAMFeatureExtractor",
    "smiles_to_graph_deep_cram",
    "smiles_to_graph",
    "get_stereochemistry_features_12D",
    "get_conformational_stability_score",
    "get_3d_conformer_features_8D",
    "TemplateExtractor",
    "ConstraintSystem",
    "get_base_node_features_45D",
    "get_synthetic_features_28D",
    "get_marine_features_15D",
    "get_cram_ring_chain_features_20D",
    "get_enhanced_node_features",
    "get_enhanced_node_features_with_cram",
]
