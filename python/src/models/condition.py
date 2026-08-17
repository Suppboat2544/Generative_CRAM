"""
Property Condition Encoder
==========================
Encode molecular properties into condition vectors for CVAE.
"""

import torch
import torch.nn as nn


# Ring class encoding (6 classes)
RING_CLASS_TO_IDX = {
    'Fused Aro-Alicyclic': 0,
    'Mixed Aro+Alicyclic': 1,
    'Benzene+Aliphatic': 2,
    'Pure Alicyclic': 3,
    'Pure Aromatic': 4,
    'Acyclic': 5,
    'Unknown': 5
}
IDX_TO_RING_CLASS = {v: k for k, v in RING_CLASS_TO_IDX.items()}
NUM_RING_CLASSES = 6  # Actual count (excluding duplicate 'Unknown')


class PropertyConditionEncoder(nn.Module):
    """
    Encodes target properties into a condition vector for CVAE.

    Input properties:
    - n_cooh (int): Number of COOH groups (0-6)
    - ring_class (int): One-hot encoded ring class (0-5)
    - aliphatic_pct (float): Aliphatic carbon percentage (0-1)
    - mw (float): Molecular weight (normalized to 0-1)
    - has_benzene (bool): Whether benzene ring is present
    - dbe_c (float): DBE/C ratio (0-1 normalized)
    - dbe_o (float): DBE/O ratio (0-1 normalized)
    - dbe_h (float): DBE/H ratio (0-1 normalized)
    - avg_ring_size (float): Average ring size (normalized 3-9 -> 0-1)
    - ring_size_class (int): 0=small/strained(3-4), 1=CRAM-preferred(5-7), 2=large(8+)

    Output: condition vector of dimension `condition_dim`
    """

    def __init__(self, condition_dim: int = 32, hidden_dim: int = 64):
        super().__init__()

        # Property embeddings
        self.n_cooh_emb = nn.Embedding(7, 8)  # COOH count 0-6 -> 8 dims
        self.ring_class_emb = nn.Embedding(NUM_RING_CLASSES, 16)  # Ring class -> 16 dims
        self.ring_size_class_emb = nn.Embedding(3, 4)  # 0=strained, 1=CRAM-preferred(5-7), 2=large

        # Continuous properties: aliphatic_pct, mw, has_benzene, dbe_c, dbe_o, dbe_h, avg_ring_size
        # Total input: 8 (n_cooh) + 16 (ring_class) + 4 (ring_size_class) + 7 (continuous) = 35
        input_dim = 8 + 16 + 4 + 7

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, condition_dim)
        )

        self.condition_dim = condition_dim

    def forward(self, n_cooh, ring_class, aliphatic_pct, mw, has_benzene,
                dbe_c=None, dbe_o=None, dbe_h=None, avg_ring_size=None, ring_size_class=None):
        """
        Encode properties into condition vector.

        Args:
            n_cooh: (batch,) int tensor, number of COOH groups
            ring_class: (batch,) int tensor, ring class index
            aliphatic_pct: (batch,) float tensor, aliphatic percentage (0-1)
            mw: (batch,) float tensor, normalized molecular weight (0-1)
            has_benzene: (batch,) float tensor, benzene presence
            dbe_c: (batch,) float tensor, DBE/C ratio (normalized 0-1)
            dbe_o: (batch,) float tensor, DBE/O ratio (normalized 0-1)
            dbe_h: (batch,) float tensor, DBE/H ratio (normalized 0-1)
            avg_ring_size: (batch,) float tensor, avg ring size normalized (3-9 -> 0-1)
            ring_size_class: (batch,) int tensor, 0=strained, 1=CRAM-preferred(5-7), 2=large

        Returns:
            condition: (batch, condition_dim) condition vector
        """
        batch_size = n_cooh.shape[0]
        device = n_cooh.device

        # Clamp values to valid ranges
        n_cooh = n_cooh.clamp(0, 6)
        ring_class = ring_class.clamp(0, NUM_RING_CLASSES - 1)

        # Default DBE ratios if not provided (use CRAM-typical midpoints)
        if dbe_c is None:
            dbe_c = torch.full((batch_size,), 0.5, device=device)
        if dbe_o is None:
            dbe_o = torch.full((batch_size,), 0.5, device=device)
        if dbe_h is None:
            dbe_h = torch.full((batch_size,), 0.25, device=device)

        # Default ring size if not provided (CRAM-typical: 5-6 membered)
        if avg_ring_size is None:
            avg_ring_size = torch.full((batch_size,), 0.5, device=device)
        if ring_size_class is None:
            ring_size_class = torch.ones(batch_size, dtype=torch.long, device=device)

        # Clamp ring size class
        ring_size_class = ring_size_class.clamp(0, 2)

        # Embeddings
        cooh_emb = self.n_cooh_emb(n_cooh)  # (batch, 8)
        ring_emb = self.ring_class_emb(ring_class)  # (batch, 16)
        ring_size_emb = self.ring_size_class_emb(ring_size_class)  # (batch, 4)

        # Continuous features (including DBE ratios and avg_ring_size)
        continuous = torch.stack([
            aliphatic_pct.clamp(0, 1),
            mw.clamp(0, 1),  # Should be pre-normalized to 0-1
            has_benzene.float(),
            dbe_c.clamp(0, 1),  # DBE/C normalized
            dbe_o.clamp(0, 1),  # DBE/O normalized
            dbe_h.clamp(0, 1),  # DBE/H normalized
            avg_ring_size.clamp(0, 1),  # Avg ring size normalized
        ], dim=-1)  # (batch, 7)

        # Concatenate all
        x = torch.cat([cooh_emb, ring_emb, ring_size_emb, continuous], dim=-1)  # (batch, 35)

        # MLP to condition vector
        condition = self.mlp(x)  # (batch, condition_dim)

        return condition

    @staticmethod
    def normalize_mw(mw, min_mw=100, max_mw=800):
        """Normalize molecular weight to 0-1 range."""
        return (mw - min_mw) / (max_mw - min_mw)

    @staticmethod
    def denormalize_mw(mw_norm, min_mw=100, max_mw=800):
        """Denormalize molecular weight from 0-1 range."""
        return mw_norm * (max_mw - min_mw) + min_mw

    @staticmethod
    def normalize_dbe_c(dbe_c, min_val=0.0, max_val=1.0):
        """Normalize DBE/C to 0-1 range. CRAM typical: 0.3-0.7"""
        return (dbe_c - min_val) / (max_val - min_val)

    @staticmethod
    def normalize_dbe_o(dbe_o, min_val=0.0, max_val=3.0):
        """Normalize DBE/O to 0-1 range. CRAM typical: 0.5-2.5"""
        return (dbe_o - min_val) / (max_val - min_val)

    @staticmethod
    def normalize_dbe_h(dbe_h, min_val=0.0, max_val=0.5):
        """Normalize DBE/H to 0-1 range. CRAM typical: 0.1-0.4"""
        return (dbe_h - min_val) / (max_val - min_val)

