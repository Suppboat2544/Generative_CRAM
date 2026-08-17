"""
CRAM Generator Configuration Settings
=====================================
Central configuration for model parameters, paths, and generation settings.
"""

import os
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# RANDOM SEED
# ══════════════════════════════════════════════════════════════════════════════
RANDOM_SEED = 42

# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
ENCODER_CONFIG = {
    "in_dim": 163,          # Feature dimension per atom
    "hidden_dim": 256,      # Hidden dimension in encoder
    "latent_dim": 64,       # Latent space dimension
    "num_gat_layers": 4,    # Number of GAT layers
    "num_heads": 4,         # Attention heads
    "dropout": 0.1,         # Dropout rate
    "use_contrastive": True # Use contrastive learning
}

DECODER_CONFIG = {
    "hidden_dim": 256,      # Decoder hidden dimension
    "num_heads": 8,         # Transformer attention heads
    "ff_dim": 1024,         # Feed-forward dimension
    "dropout": 0.1,         # Dropout rate
    "num_layers": 4         # Transformer decoder layers
}

# ══════════════════════════════════════════════════════════════════════════════
# CRAM VALIDATION PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
CRAM_VALIDATION = {
    "min_cooh": 2,                  # Minimum carboxyl groups
    "min_aliphatic_pct": 0.6,       # Minimum aliphatic carbon percentage
    "require_alicyclic": True,      # Must have alicyclic ring
    "mw_range": (100, 750),         # Molecular weight range (Da)
}

# SMARTS patterns for CRAM validation
CRAM_SMARTS_PATTERNS = {
    # Multiple COOH detection patterns for strict 2+ validation
    "cooh_1": "[CX3](=O)[OX2H1]",           # Standard COOH
    "cooh_2": "[CX3](=O)[OX1-]",             # Carboxylate anion
    "cooh_3": "[C](=O)[OH]",                 # Simple notation
    "cooh_4": "[CX3](=[OX1])[OX2H1]",        # Explicit carbonyl
    "cooh_5": "[CX3](=[OX1])O",              # Relaxed
    
    # Alicyclic ring patterns
    "alicyclic_5": "[R1;!a]1~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]1",
    "alicyclic_6": "[R1;!a]1~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]1",
    "alicyclic_7": "[R1;!a]1~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]~[R1;!a]1",
    
    # Functional groups
    "hydroxyl": "[OX2H]",
    "ether": "[OX2]([C])[C]",
    "carbonyl": "[CX3]=O",
    "methoxy": "[CH3][OX2]",
}

# ══════════════════════════════════════════════════════════════════════════════
# GENERATION PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
GENERATION_CONFIG = {
    "target_molecules": 1_000_000,  # Target number of molecules
    "batch_size": 1024,             # Generation batch size
    "max_attempts": 500_000_000,    # Maximum generation attempts
    "save_interval": 50_000,        # Save checkpoint every N molecules
    "temperature": 0.8,             # Sampling temperature
    "top_k": 50,                    # Top-k sampling
    "top_p": 0.95,                  # Nucleus sampling threshold
}

# ══════════════════════════════════════════════════════════════════════════════
# SELFIES VOCABULARY
# ══════════════════════════════════════════════════════════════════════════════
SPECIAL_TOKENS = {
    "PAD": 0,
    "SOS": 1,
    "EOS": 2,
    "UNK": 3,
}

MAX_SELFIES_LEN = 150

# ══════════════════════════════════════════════════════════════════════════════
# CVAE MODEL CONFIGURATION (Property-Conditioned)
# ══════════════════════════════════════════════════════════════════════════════
CVAE_CONFIG = {
    "latent_dim": 64,           # Latent space dimension
    "condition_dim": 32,        # Condition vector dimension
    "hidden_dim": 256,          # Hidden layer dimension
    "in_dim": 163,              # Input feature dimension
    "num_heads": 4,             # GAT attention heads
    "num_encoder_layers": 3,    # GAT layers in encoder
    "num_decoder_layers": 2,    # GRU layers in decoder
    "max_len": 100,             # Maximum sequence length
    "dropout": 0.1,             # Dropout rate
    
    # Token IDs (CRITICAL - must match trained model vocab)
    "sos_idx": 3,               # Start of sequence token
    "eos_idx": 1,               # End of sequence token
    "pad_idx": 2,               # Padding token
}

# CVAE Generation settings
CVAE_GENERATION_CONFIG = {
    "temperature": 0.8,         # Sampling temperature
    "batch_size": 256,          # Generation batch size
    "max_attempts_multiplier": 10,  # max_attempts = target * multiplier
    
    # Allowed atoms for CRAM (C, H, O, N, S, F, Cl, Br, I)
    "allowed_atoms": {6, 1, 8, 7, 16, 9, 17, 35, 53},
}

# ══════════════════════════════════════════════════════════════════════════════
# BLOOM FILTER (for duplicate detection)
# ══════════════════════════════════════════════════════════════════════════════
BLOOM_FILTER_CONFIG = {
    "expected_items": 10_000_000,
    "false_positive_rate": 0.001,
}

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
VISUALIZATION_CONFIG = {
    "dpi": 700,
    "figure_format": "png",
    "color_palette": "viridis",
    "show_grid": False,
    "show_title": False,
}

# ══════════════════════════════════════════════════════════════════════════════
# DEVICE CONFIGURATION (M4/MPS Compatible)
# ══════════════════════════════════════════════════════════════════════════════
import torch

def get_device():
    """Get optimal device - supports M4 Mac with MPS."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
