"""
Enhanced CRAM Generator for Bi-/Tri-/Quad-Alicyclic Molecules
==============================================================

This module extends PropertyConditionEncoder to support explicit control over
the NUMBER of alicyclic rings, enabling targeted generation of:
- Bicyclic alicyclic (2 fused alicyclic rings)
- Tricyclic alicyclic (3 fused alicyclic rings)
- Polycyclic alicyclic (4+ fused alicyclic rings)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import random
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED PROPERTY ENCODER WITH N_ALICYCLIC_RINGS CONDITIONING
# ═══════════════════════════════════════════════════════════════════════════════

# Alicyclic ring count classes for CRAM
N_ALICYCLIC_CLASSES = {
    'monocyclic': 1, # 1 alicyclic ring
    'bicyclic': 2, # 2 fused alicyclic rings (like decalin)
    'tricyclic': 3, # 3 fused rings (like adamantane skeleton)
    'tetracyclic': 4, # 4 fused rings
    'polycyclic': 5 # 5+ fused rings (steroid-like)
}

class EnhancedPropertyConditionEncoder(nn.Module):
    """
    Enhanced Property Condition Encoder with explicit N_ALICYCLIC_RINGS control.
    
    NEW FEATURE: n_alicyclic_rings embedding allows targeting:
    - Bicyclic: decalin-like fused bicyclic systems
    - Tricyclic: adamantane-like or perhydroanthracene systems
    - Tetracyclic: steroid-like polycyclic systems
    
    Input properties (11 total):
    - n_cooh (int): Number of COOH groups (2-6)
    - ring_class (int): Ring class index (0-5)
    - n_alicyclic_rings (int): Number of alicyclic rings (0-5+) ← NEW
    - aliphatic_pct (float): Aliphatic carbon percentage (0-1)
    - mw (float): Molecular weight (normalized to 0-1)
    - has_benzene (bool): Whether benzene ring is present
    - dbe_c (float): DBE/C ratio (0-1 normalized)
    - dbe_o (float): DBE/O ratio (0-1 normalized)
    - dbe_h (float): DBE/H ratio (0-1 normalized)
    - avg_ring_size (float): Average ring size (normalized 0-1)
    - ring_size_class (int): 0=strained(3-4), 1=CRAM-preferred(5-7), 2=large(8+)
    
    Output: condition vector of dimension `condition_dim`
    """
    
    def __init__(self, condition_dim=32, hidden_dim=64, num_ring_classes=6):
        super().__init__()
        
        # Property embeddings
        self.n_cooh_emb = nn.Embedding(7, 8) # COOH count 0-6 → 8 dims
        self.ring_class_emb = nn.Embedding(num_ring_classes, 16) # Ring class → 16 dims
        self.ring_size_class_emb = nn.Embedding(3, 4) # Ring size class → 4 dims
        
        # NEW: Alicyclic ring count embedding (0-5+ rings → 8 dims)
        # This is critical for generating bi-/tri-/quad-alicyclic structures
        self.n_alicyclic_emb = nn.Embedding(6, 8) # 0-5+ alicyclic rings → 8 dims
        
        # Continuous properties: aliphatic_pct, mw, has_benzene, dbe_c, dbe_o, dbe_h, avg_ring_size
        # Total input: 8 (n_cooh) + 16 (ring_class) + 4 (ring_size_class) + 8 (n_alicyclic) + 7 (continuous) = 43
        input_dim = 8 + 16 + 4 + 8 + 7 # 43 total
        
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
        self.num_ring_classes = num_ring_classes
    
    def forward(self, n_cooh, ring_class, aliphatic_pct, mw, has_benzene,
                dbe_c=None, dbe_o=None, dbe_h=None, avg_ring_size=None, 
                ring_size_class=None, n_alicyclic_rings=None):
        """
        Encode properties into condition vector.
        
        Args:
            n_cooh: (batch,) int tensor, number of COOH groups
            ring_class: (batch,) int tensor, ring class index
            aliphatic_pct: (batch,) float tensor, aliphatic percentage
            mw: (batch,) float tensor, normalized molecular weight
            has_benzene: (batch,) float tensor, benzene presence
            dbe_c: (batch,) float tensor, DBE/C ratio (normalized)
            dbe_o: (batch,) float tensor, DBE/O ratio (normalized)
            dbe_h: (batch,) float tensor, DBE/H ratio (normalized)
            avg_ring_size: (batch,) float tensor, avg ring size normalized
            ring_size_class: (batch,) int tensor, ring size classification
            n_alicyclic_rings: (batch,) int tensor, number of alicyclic rings (0-5+)
        
        Returns:
            condition: (batch, condition_dim) condition vector
        """
        batch_size = n_cooh.shape[0]
        device = n_cooh.device
        
        # Clamp values to valid ranges
        n_cooh = n_cooh.clamp(0, 6)
        ring_class = ring_class.clamp(0, self.num_ring_classes - 1)
        
        # Default DBE ratios if not provided
        if dbe_c is None:
            dbe_c = torch.full((batch_size,), 0.5, device=device)
        if dbe_o is None:
            dbe_o = torch.full((batch_size,), 0.5, device=device)
        if dbe_h is None:
            dbe_h = torch.full((batch_size,), 0.25, device=device)
        
        # Default ring size if not provided
        if avg_ring_size is None:
            avg_ring_size = torch.full((batch_size,), 0.5, device=device)
        if ring_size_class is None:
            ring_size_class = torch.ones(batch_size, dtype=torch.long, device=device)
        
        # Default n_alicyclic_rings if not provided (2 = bicyclic as default)
        if n_alicyclic_rings is None:
            n_alicyclic_rings = torch.full((batch_size,), 2, dtype=torch.long, device=device)
        
        # Clamp values
        ring_size_class = ring_size_class.clamp(0, 2)
        n_alicyclic_rings = n_alicyclic_rings.clamp(0, 5) # Max 5 for embedding
        
        # Embeddings
        cooh_emb = self.n_cooh_emb(n_cooh) # (batch, 8)
        ring_emb = self.ring_class_emb(ring_class) # (batch, 16)
        ring_size_emb = self.ring_size_class_emb(ring_size_class) # (batch, 4)
        alicyclic_emb = self.n_alicyclic_emb(n_alicyclic_rings) # (batch, 8) ← NEW
        
        # Continuous features
        continuous = torch.stack([
            aliphatic_pct.clamp(0, 1),
            mw.clamp(0, 1),
            has_benzene.float(),
            dbe_c.clamp(0, 1),
            dbe_o.clamp(0, 1),
            dbe_h.clamp(0, 1),
            avg_ring_size.clamp(0, 1),
        ], dim=-1) # (batch, 7)
        
        # Concatenate all (8 + 16 + 4 + 8 + 7 = 43)
        x = torch.cat([cooh_emb, ring_emb, ring_size_emb, alicyclic_emb, continuous], dim=-1)
        
        # MLP to condition vector
        condition = self.mlp(x)
        
        return condition
    
    @staticmethod
    def normalize_mw(mw, min_mw=100, max_mw=800):
        """Normalize molecular weight to 0-1 range."""
        return (mw - min_mw) / (max_mw - min_mw)
    
    @staticmethod
    def normalize_dbe_c(dbe_c, min_val=0.0, max_val=1.0):
        """Normalize DBE/C to 0-1 range."""
        return (dbe_c - min_val) / (max_val - min_val)
    
    @staticmethod
    def normalize_dbe_o(dbe_o, min_val=0.0, max_val=3.0):
        """Normalize DBE/O to 0-1 range."""
        return (dbe_o - min_val) / (max_val - min_val)
    
    @staticmethod
    def normalize_dbe_h(dbe_h, min_val=0.0, max_val=0.5):
        """Normalize DBE/H to 0-1 range."""
        return (dbe_h - min_val) / (max_val - min_val)


# ═══════════════════════════════════════════════════════════════════════════════
# POLYCYCLIC CRAM GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class PolycyclicCRAMGenerator:
    """
    Specialized generator for bi-/tri-/quad-alicyclic CRAM molecules.
    
    Extends FastPropertyConditionedGenerator with explicit control over
    the number of fused alicyclic rings.
    
    Example usage:
        generator = PolycyclicCRAMGenerator(model, vocab, device='cuda')
        
        # Generate 100 bicyclic CRAM molecules (decalin-like)
        bicyclic = generator.generate_bicyclic(n=100, n_cooh=3)
        
        # Generate 100 tricyclic CRAM molecules (adamantane-like)
        tricyclic = generator.generate_tricyclic(n=100, n_cooh=4)
        
        # Generate 100 tetracyclic CRAM molecules (steroid-like)
        tetracyclic = generator.generate_tetracyclic(n=100, n_cooh=5)
        
        # Or specify exact number of alicyclic rings
        polycyclic = generator.generate_polycyclic(
            n=100, 
            n_alicyclic_rings=4, # Tetracyclic
            n_cooh=4,
            has_benzene=False # Pure alicyclic
        )
    """
    
    RING_CLASSES = {
        'fused': 0, 'fused_aro_alicyclic': 0,
        'mixed': 1, 'mixed_aro_alicyclic': 1,
        'benzene_aliphatic': 2,
        'pure_alicyclic': 3, # ← Most relevant for pure polycyclic
        'pure_aromatic': 4,
        'acyclic': 5
    }
    
    def __init__(self, model, selfies_vocab: dict, idx_to_token: dict = None, 
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.selfies_vocab = selfies_vocab
        self.idx_to_token = idx_to_token or {v: k for k, v in selfies_vocab.items()}
        self.device = device
        self.latent_dim = model.encoder.latent_dim
        
        # Special tokens
        self.sos_token = selfies_vocab.get('<SOS>', selfies_vocab.get('[nop]', 0))
        self.eos_token = selfies_vocab.get('<EOS>', selfies_vocab.get('[nop]', 0))
        self.pad_token = selfies_vocab.get('<PAD>', selfies_vocab.get('[nop]', 0))
    
    @torch.no_grad()
    def generate_polycyclic(self, 
                            n: int = 100,
                            n_alicyclic_rings: int = 2, # ← KEY PARAMETER
                            n_cooh: int = 3,
                            ring_class: str = 'pure_alicyclic',
                            aliphatic_pct: float = 0.75, # Higher for pure alicyclic
                            mw: float = 400.0,
                            has_benzene: bool = False, # False for pure alicyclic
                            dbe_c: float = 0.4,
                            dbe_o: float = 1.0,
                            dbe_h: float = 0.2,
                            avg_ring_size: float = 5.5, # 5-6 membered rings
                            ring_size_class: int = 1, # CRAM-preferred
                            temperature: float = 1.0,
                            max_length: int = 100) -> List[str]:
        """
        Generate n CRAM molecules with specified number of alicyclic rings.
        
        Args:
            n: Number of molecules to generate
            n_alicyclic_rings: TARGET number of fused alicyclic rings (1-5+)
                - 1 = monocyclic (single 5-6 membered ring)
                - 2 = bicyclic (decalin-like fused rings)
                - 3 = tricyclic (perhydrophenanthrene-like)
                - 4 = tetracyclic (steroid skeleton-like)
                - 5 = highly polycyclic
            n_cooh: Target COOH groups (2-6 for CRAM)
            ring_class: 'pure_alicyclic' (default), 'fused', 'mixed'
            aliphatic_pct: Aliphatic carbon fraction (higher for pure alicyclic)
            mw: Target molecular weight
            has_benzene: Include benzene? (False for pure alicyclic)
            temperature: Sampling temperature
            
        Returns:
            List of SMILES strings
        """
        # Prepare condition tensors
        ring_idx = self.RING_CLASSES.get(ring_class.lower().replace(' ', '_'), 3)
        mw_norm = EnhancedPropertyConditionEncoder.normalize_mw(mw)
        dbe_c_norm = EnhancedPropertyConditionEncoder.normalize_dbe_c(dbe_c)
        dbe_o_norm = EnhancedPropertyConditionEncoder.normalize_dbe_o(dbe_o)
        dbe_h_norm = EnhancedPropertyConditionEncoder.normalize_dbe_h(dbe_h)
        avg_ring_size_norm = (avg_ring_size - 3.0) / 6.0
        
        # Create batch tensors
        n_cooh_t = torch.full((n,), n_cooh, dtype=torch.long, device=self.device)
        ring_t = torch.full((n,), ring_idx, dtype=torch.long, device=self.device)
        aliphatic_t = torch.full((n,), aliphatic_pct, dtype=torch.float32, device=self.device)
        mw_t = torch.full((n,), mw_norm, dtype=torch.float32, device=self.device)
        benzene_t = torch.full((n,), float(has_benzene), dtype=torch.float32, device=self.device)
        dbe_c_t = torch.full((n,), dbe_c_norm, dtype=torch.float32, device=self.device)
        dbe_o_t = torch.full((n,), dbe_o_norm, dtype=torch.float32, device=self.device)
        dbe_h_t = torch.full((n,), dbe_h_norm, dtype=torch.float32, device=self.device)
        avg_ring_size_t = torch.full((n,), avg_ring_size_norm, dtype=torch.float32, device=self.device)
        ring_size_class_t = torch.full((n,), ring_size_class, dtype=torch.long, device=self.device)
        
        # KEY: n_alicyclic_rings tensor
        n_alicyclic_t = torch.full((n,), min(n_alicyclic_rings, 5), dtype=torch.long, device=self.device)
        
        # Encode conditions (requires EnhancedPropertyConditionEncoder)
        conditions = self.model.condition_encoder(
            n_cooh_t, ring_t, aliphatic_t, mw_t, benzene_t,
            dbe_c_t, dbe_o_t, dbe_h_t,
            avg_ring_size_t, ring_size_class_t,
            n_alicyclic_t # ← Pass n_alicyclic_rings
        )
        
        # Sample from latent space
        z = torch.randn(n, self.latent_dim, device=self.device) * temperature
        
        # Generate sequences
        generated = self.model.generate_conditioned(
            z=z,
            conditions=conditions,
            max_length=max_length,
            temperature=temperature,
            start_token=self.sos_token
        )
        
        return self._batch_decode(generated)
    
    def generate_bicyclic(self, n: int = 100, **kwargs) -> List[str]:
        """Generate bicyclic CRAM molecules (2 fused alicyclic rings)."""
        kwargs.setdefault('ring_class', 'pure_alicyclic')
        kwargs.setdefault('has_benzene', False)
        kwargs.setdefault('aliphatic_pct', 0.8)
        return self.generate_polycyclic(n=n, n_alicyclic_rings=2, **kwargs)
    
    def generate_tricyclic(self, n: int = 100, **kwargs) -> List[str]:
        """Generate tricyclic CRAM molecules (3 fused alicyclic rings)."""
        kwargs.setdefault('ring_class', 'pure_alicyclic')
        kwargs.setdefault('has_benzene', False)
        kwargs.setdefault('aliphatic_pct', 0.8)
        kwargs.setdefault('mw', 450.0) # Slightly larger for tricyclic
        return self.generate_polycyclic(n=n, n_alicyclic_rings=3, **kwargs)
    
    def generate_tetracyclic(self, n: int = 100, **kwargs) -> List[str]:
        """Generate tetracyclic CRAM molecules (4 fused alicyclic rings)."""
        kwargs.setdefault('ring_class', 'pure_alicyclic')
        kwargs.setdefault('has_benzene', False)
        kwargs.setdefault('aliphatic_pct', 0.85)
        kwargs.setdefault('mw', 500.0) # Larger for tetracyclic
        kwargs.setdefault('n_cooh', 4) # More COOH for larger scaffold
        return self.generate_polycyclic(n=n, n_alicyclic_rings=4, **kwargs)
    
    def generate_mixed_polycyclic(self, n: int = 100, 
                                   n_alicyclic_rings: int = 2,
                                   **kwargs) -> List[str]:
        """Generate mixed aromatic + alicyclic fused systems."""
        kwargs.setdefault('ring_class', 'fused') # Fused aro-alicyclic
        kwargs.setdefault('has_benzene', True)
        kwargs.setdefault('aliphatic_pct', 0.6)
        return self.generate_polycyclic(n=n, n_alicyclic_rings=n_alicyclic_rings, **kwargs)
    
    def generate_million_polycyclic(self, n: int = 1_000_000,
                                     batch_size: int = 1000,
                                     polycyclic_distribution: dict = None,
                                     save_path: str = None):
        """
        Generate MILLION polycyclic CRAM molecules with varied ring counts.
        
        Args:
            n: Total molecules to generate
            batch_size: Batch size for generation
            polycyclic_distribution: Dict mapping n_alicyclic_rings → probability
                                    Default favors bicyclic and tricyclic
            save_path: If provided, saves results to CSV
            
        Returns:
            DataFrame with SMILES and properties
        """
        import pandas as pd
        
        if polycyclic_distribution is None:
            # Default distribution favoring CRAM-relevant polycyclic systems
            polycyclic_distribution = {
                1: 0.10, # 10% monocyclic
                2: 0.35, # 35% bicyclic (most common CRAM scaffold)
                3: 0.30, # 30% tricyclic
                4: 0.20, # 20% tetracyclic
                5: 0.05 # 5% highly polycyclic
            }
        
        def polycyclic_sampler():
            # Sample n_alicyclic_rings based on distribution
            n_alicyclic = random.choices(
                list(polycyclic_distribution.keys()),
                weights=list(polycyclic_distribution.values())
            )[0]
            
            # Adjust MW based on ring count
            base_mw = 280 + n_alicyclic * 50
            
            return {
                'n_alicyclic_rings': n_alicyclic,
                'n_cooh': random.choice([2, 3, 3, 4, 4] if n_alicyclic < 4 else [3, 4, 4, 5]),
                'ring_class': 'pure_alicyclic' if random.random() > 0.3 else 'fused',
                'aliphatic_pct': 0.6 + n_alicyclic * 0.05, # Higher for more rings
                'mw': random.gauss(base_mw, 50),
                'has_benzene': random.random() > 0.7,
                'dbe_c': random.uniform(0.3, 0.5),
                'dbe_o': random.uniform(0.5, 1.5),
                'dbe_h': random.uniform(0.15, 0.30)
            }
        
        all_results = []
        n_batches = (n + batch_size - 1) // batch_size
        
        print(f"Generating {n:,} polycyclic CRAM molecules...")
        print(f"Distribution: {polycyclic_distribution}")
        
        for i in tqdm(range(n_batches), desc="Batch generation"):
            current_batch = min(batch_size, n - len(all_results))
            props_list = [polycyclic_sampler() for _ in range(current_batch)]
            
            # Group by n_alicyclic_rings for efficient batch generation
            from collections import defaultdict
            grouped = defaultdict(list)
            for idx, props in enumerate(props_list):
                grouped[props['n_alicyclic_rings']].append((idx, props))
            
            batch_smiles = [None] * current_batch
            
            for n_alicyclic, items in grouped.items():
                indices = [idx for idx, _ in items]
                props_subset = [p for _, p in items]
                
                # Generate for this n_alicyclic group
                smiles = self.generate_polycyclic(
                    n=len(indices),
                    n_alicyclic_rings=n_alicyclic,
                    n_cooh=props_subset[0]['n_cooh'],
                    ring_class=props_subset[0]['ring_class'],
                    aliphatic_pct=props_subset[0]['aliphatic_pct'],
                    mw=props_subset[0]['mw'],
                    has_benzene=props_subset[0]['has_benzene'],
                    dbe_c=props_subset[0]['dbe_c'],
                    dbe_o=props_subset[0]['dbe_o'],
                    dbe_h=props_subset[0]['dbe_h']
                )
                
                # Assign back to correct positions
                for j, (idx, props) in enumerate(items):
                    if j < len(smiles) and smiles[j]:
                        all_results.append({'SMILES': smiles[j], **props})
            
            if len(all_results) >= n:
                break
        
        df = pd.DataFrame(all_results[:n])
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Saved {len(df)} molecules to {save_path}")
        
        # Print statistics
        if len(df) > 0:
            print(f"\n Generated {len(df):,} valid polycyclic CRAM molecules")
            print("Ring count distribution:")
            for n_rings, count in df['n_alicyclic_rings'].value_counts().sort_index().items():
                ring_type = {1: 'Monocyclic', 2: 'Bicyclic', 3: 'Tricyclic', 
                            4: 'Tetracyclic', 5: 'Polycyclic'}.get(n_rings, f'{n_rings}-cyclic')
                print(f" {ring_type}: {count:,} ({100*count/len(df):.1f}%)")
        
        return df
    
    def _batch_decode(self, token_indices):
        """Fast batch decode tokens to SMILES."""
        import selfies as sf
        from rdkit import Chem
        
        results = []
        for seq in token_indices:
            tokens = []
            for idx in seq:
                idx_val = idx.item()
                if idx_val == self.eos_token:
                    break
                if idx_val not in [self.pad_token, self.sos_token]:
                    token = self.idx_to_token.get(idx_val, '')
                    if token and not token.startswith('<'):
                        tokens.append(token)
            
            selfies_str = ''.join(tokens)
            
            try:
                smiles = sf.decoder(selfies_str)
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        results.append(Chem.MolToSmiles(mol))
                        continue
            except:
                pass
            results.append(None)
        
        return [s for s in results if s is not None]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: EXTRACT n_alicyclic_rings FROM MOLECULE
# ═══════════════════════════════════════════════════════════════════════════════

def count_alicyclic_rings(mol):
    """
    Count the number of alicyclic (non-aromatic) rings in a molecule.
    
    Args:
        mol: RDKit mol object
        
    Returns:
        int: Number of alicyclic rings
    """
    from rdkit import Chem
    
    if mol is None:
        return 0
    
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    
    n_alicyclic = 0
    for ring_atoms in atom_rings:
        # Check if ring is aromatic
        is_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_atoms)
        if not is_aromatic:
            n_alicyclic += 1
    
    return n_alicyclic


def enhanced_extract_properties(mol, is_true_cram_func, ring_class_to_idx):
    """
    Enhanced property extraction including n_alicyclic_rings.
    
    Use this in PropertyConditionedCRAMDataset._extract_properties()
    """
    from rdkit.Chem import Descriptors
    import numpy as np
    
    try:
        result = is_true_cram_func(mol)
        
        n_cooh = result.get('n_cooh', 0)
        ring_class = result.get('ring_class', 'Unknown')
        ring_idx = ring_class_to_idx.get(ring_class, 5)
        aliphatic_pct = result.get('aliphatic_pct', 0.5)
        mw = Descriptors.MolWt(mol)
        has_benzene = float(result.get('has_benzene', False))
        
        # Ring size analysis
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
        if ring_sizes:
            avg_ring_size = np.mean(ring_sizes)
            n_cram_preferred = sum(1 for s in ring_sizes if 5 <= s <= 7)
            n_strained = sum(1 for s in ring_sizes if 3 <= s <= 4)
            if n_cram_preferred > 0 and n_cram_preferred >= len(ring_sizes) * 0.5:
                ring_size_class = 1
            elif n_strained > len(ring_sizes) * 0.5:
                ring_size_class = 0
            else:
                ring_size_class = 2 if any(s >= 8 for s in ring_sizes) else 1
        else:
            avg_ring_size = 5.5
            ring_size_class = 1
        
        avg_ring_size_norm = np.clip((avg_ring_size - 3.0) / 6.0, 0.0, 1.0)
        mw_norm = (mw - 100) / (800 - 100)
        
        # KEY: Extract n_alicyclic_rings
        n_alicyclic_rings = result.get('n_alicyclic_rings', count_alicyclic_rings(mol))
        
        return {
            'n_cooh': n_cooh,
            'ring_class': ring_idx,
            'aliphatic_pct': aliphatic_pct,
            'mw': mw_norm,
            'has_benzene': has_benzene,
            'avg_ring_size': avg_ring_size_norm,
            'ring_size_class': ring_size_class,
            'n_alicyclic_rings': min(n_alicyclic_rings, 5) # Cap at 5 for embedding
        }
    except:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
"""
INTEGRATION STEPS:

1. Replace PropertyConditionEncoder with EnhancedPropertyConditionEncoder:
   
   # In model initialization:
   self.condition_encoder = EnhancedPropertyConditionEncoder(condition_dim=condition_dim)

2. Update _extract_properties in PropertyConditionedCRAMDataset:
   
   def _extract_properties(self, mol):
       return enhanced_extract_properties(mol, is_true_cram, RING_CLASS_TO_IDX)

3. Update collate_conditioned_batch to include n_alicyclic_rings:
   
   # Add to tensor creation:
   'n_alicyclic_rings': torch.tensor([p['n_alicyclic_rings'] for p in props], dtype=torch.long)

4. Update training loop to pass n_alicyclic_rings:
   
   output = model(g, target_seq,
                  target_n_cooh=batch['n_cooh'],
                  target_ring_class=batch['ring_class'],
                  ...,
                  target_n_alicyclic_rings=batch['n_alicyclic_rings'])

5. Use PolycyclicCRAMGenerator for targeted generation:
   
   generator = PolycyclicCRAMGenerator(model, vocab, device='cuda')
   
   # Generate specific polycyclic types:
   bicyclic = generator.generate_bicyclic(n=1000, n_cooh=3)
   tricyclic = generator.generate_tricyclic(n=1000, n_cooh=4)
   tetracyclic = generator.generate_tetracyclic(n=1000, n_cooh=5)
   
   # Or customize:
   custom = generator.generate_polycyclic(
       n=1000,
       n_alicyclic_rings=3, # Tricyclic
       n_cooh=4,
       has_benzene=False, # Pure alicyclic
       aliphatic_pct=0.85
   )

EXAMPLE STRUCTURES BY RING COUNT:

- Bicyclic (2 rings): Decalin derivatives with COOH groups
  Typical formula: C10-15H16-24(COOH)2-4
  
- Tricyclic (3 rings): Perhydrophenanthrene, adamantane-carboxylic acids
  Typical formula: C12-18H18-30(COOH)2-4
  
- Tetracyclic (4 rings): Perhydrochrysene, steroid-like scaffolds
  Typical formula: C17-23H24-38(COOH)3-5
"""

if __name__ == '__main__':
    print("Enhanced Polycyclic CRAM Generator loaded!")
    print("\nAvailable classes:")
    print(" - EnhancedPropertyConditionEncoder")
    print(" - PolycyclicCRAMGenerator")
    print("\nKey methods:")
    print(" generator.generate_bicyclic(n=100, n_cooh=3)")
    print(" generator.generate_tricyclic(n=100, n_cooh=4)")
    print(" generator.generate_tetracyclic(n=100, n_cooh=5)")
    print(" generator.generate_polycyclic(n=100, n_alicyclic_rings=4)")
