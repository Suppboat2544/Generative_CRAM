"""
CRAM Models Package
===================
Notebook models plus the GRU production CVAE.
"""

from .encoder import DeepCRAMEncoder as CVAEEncoder
from .deep_cram_encoder import DeepCRAMEncoder
from .decoder import SELFIESDecoder
from .condition import PropertyConditionEncoder
from .cvae import PropertyConditionedCRAMVAE, create_cvae_model
from .deep_cram_vae import DeepCRAMDecoderVAE, training_style_vocab
from .transformer_vae import (
    DeepCRAMVAE,
    PropertyConditionedTransformerVAE,
    MultimodalTransformerVAE,
    TransformerVAE,
    cram_model_registry,
)
from .hierarchical_encoder import EnhancedHierarchicalGATEncoder
from .ring_chain import RingChainEncoder, SubstructureDetector, CRAMStructureLoss
from .functional_group_attention import FunctionalGroupAttention, CRAMGroupPropertyPredictor
from .contrastive import ContrastiveCRAMLearning, HardNegativeMiner, CRAMContrastiveTrainer
from .hierarchical_encoder import CRAMFeatureExtractor
from .losses import vae_loss, reconstruction_loss, kl_divergence

__all__ = [
    "DeepCRAMEncoder",
    "CVAEEncoder",
    "SELFIESDecoder",
    "PropertyConditionEncoder",
    "PropertyConditionedCRAMVAE",
    "create_cvae_model",
    "DeepCRAMVAE",
    "DeepCRAMDecoderVAE",
    "PropertyConditionedTransformerVAE",
    "MultimodalTransformerVAE",
    "TransformerVAE",
    "cram_model_registry",
    "EnhancedHierarchicalGATEncoder",
    "RingChainEncoder",
    "SubstructureDetector",
    "CRAMStructureLoss",
    "FunctionalGroupAttention",
    "CRAMGroupPropertyPredictor",
    "ContrastiveCRAMLearning",
    "HardNegativeMiner",
    "CRAMContrastiveTrainer",
    "CRAMFeatureExtractor",
    "training_style_vocab",
    "vae_loss",
    "reconstruction_loss",
    "kl_divergence",
]
