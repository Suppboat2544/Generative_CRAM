"""
CRAM Generator Source Package
==============================
Complete module for CRAM molecule generation and validation.

Structure:
- src.models: CVAE model components (encoder, decoder, cvae, losses)
- src.generators: Molecule generation strategies
- src.features: Feature extraction modules (163D)
"""

# Import from modular subpackages
from .models import (
    PropertyConditionedCRAMVAE,
    PropertyConditionEncoder,
    DeepCRAMEncoder,
    SELFIESDecoder,
    vae_loss,
    reconstruction_loss,
    kl_divergence,
)

from .generators import (
    BloomFilter,
    HybridDeduplicator,
    SELFIESMutationGenerator,
    FastPropertyConditionedGenerator,
    smiles_to_graph,
    get_device,
)

from .features import (
    DeepCRAMFeatureExtractor,
    AtomFeatureExtractor,
    MarineDOMFeatures,
    SyntheticAccessibilityFeatures,
)

# Import from single-file modules
from .validator import CRAMValidator, is_true_cram, is_true_cram_smiles
from .tokenizer import SELFIESTokenizer, BeamSearchDecoder
from .dataset import (
    CRAMDataset,
    SAFilteredDataset,
    collate_cram_batch,
    create_dataloaders,
)
from .training import LabelSmoothingCrossEntropy, CRAMTrainer, train_cram_model
from .utils import (
    SyntheticAccessibilityCalculator,
    SMILESAugmenter,
    ScaffoldOperator,
    LatentSpaceDiversifier,
)

# Subpackage access
from . import models
from . import generators
from . import features

__all__ = [
    # Models
    "PropertyConditionedCRAMVAE",
    "PropertyConditionEncoder",
    "DeepCRAMEncoder",
    "SELFIESDecoder",
    "vae_loss",
    "reconstruction_loss",
    "kl_divergence",
    # Generators
    "BloomFilter",
    "HybridDeduplicator",
    "SELFIESMutationGenerator",
    "FastPropertyConditionedGenerator",
    "smiles_to_graph",
    "get_device",
    # Features
    "DeepCRAMFeatureExtractor",
    "AtomFeatureExtractor",
    "MarineDOMFeatures",
    "SyntheticAccessibilityFeatures",
    # Validator
    "CRAMValidator",
    "is_true_cram",
    "is_true_cram_smiles",
    # Tokenizer
    "SELFIESTokenizer",
    "BeamSearchDecoder",
    # Dataset
    "CRAMDataset",
    "SAFilteredDataset",
    "collate_cram_batch",
    "create_dataloaders",
    # Training
    "LabelSmoothingCrossEntropy",
    "CRAMTrainer",
    "train_cram_model",
    # Utilities
    "SyntheticAccessibilityCalculator",
    "SMILESAugmenter",
    "ScaffoldOperator",
    "LatentSpaceDiversifier",
    # Subpackages
    "models",
    "generators",
    "features",
]
