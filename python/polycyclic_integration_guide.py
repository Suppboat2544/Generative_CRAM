# =============================================================================
# INTEGRATION: ADD POLYCYCLIC GENERATION TO YOUR NOTEBOOK
# =============================================================================
# Copy these modifications into your CRAM_Generator_Production2.ipynb notebook
# to enable bi-/tri-/quad-alicyclic molecule generation.
# =============================================================================

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║ STEP 1: REPLACE PropertyConditionEncoder.__init__ WITH THIS VERSION          ║
# ╔═══════════════════════════════════════════════════════════════════════════════╝

"""
Find this code in your notebook:

    def __init__(self, condition_dim=32, hidden_dim=64):
        super().__init__()

        # Property embeddings
        self.n_cooh_emb = nn.Embedding(7, 8)  # COOH count 0-6 -> 8 dims
        self.ring_class_emb = nn.Embedding(NUM_RING_CLASSES, 16)  # Ring class -> 16 dims
        self.ring_size_class_emb = nn.Embedding(3, 4)  # 0=strained, 1=CRAM-preferred(5-7), 2=large

        # Total input: 8 (n_cooh) + 16 (ring_class) + 4 (ring_size_class) + 7 (continuous) = 35
        input_dim = 8 + 16 + 4 + 7

REPLACE WITH:
"""

def __init__(self, condition_dim=32, hidden_dim=64):
    super().__init__()
    
    # Property embeddings
    self.n_cooh_emb = nn.Embedding(7, 8)  # COOH count 0-6 → 8 dims
    self.ring_class_emb = nn.Embedding(NUM_RING_CLASSES, 16)  # Ring class → 16 dims
    self.ring_size_class_emb = nn.Embedding(3, 4)  # 0=strained, 1=CRAM-preferred(5-7), 2=large
    
    # ★ NEW: Alicyclic ring count embedding for polycyclic CRAM
    self.n_alicyclic_emb = nn.Embedding(6, 8)  # 0-5+ alicyclic rings → 8 dims
    
    # Total input: 8 (n_cooh) + 16 (ring_class) + 4 (ring_size_class) + 8 (n_alicyclic) + 7 (continuous) = 43
    input_dim = 8 + 16 + 4 + 8 + 7  # Updated from 35 to 43
    
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


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║ STEP 2: REPLACE PropertyConditionEncoder.forward WITH THIS VERSION           ║
# ╔═══════════════════════════════════════════════════════════════════════════════╝

def forward(self, n_cooh, ring_class, aliphatic_pct, mw, has_benzene,
            dbe_c=None, dbe_o=None, dbe_h=None, avg_ring_size=None, 
            ring_size_class=None, n_alicyclic_rings=None):  # ★ NEW parameter
    """
    Encode properties into condition vector.
    
    NEW: n_alicyclic_rings controls the number of fused alicyclic rings (1-5+)
    """
    batch_size = n_cooh.shape[0]
    device = n_cooh.device
    
    # Clamp values to valid ranges
    n_cooh = n_cooh.clamp(0, 6)
    ring_class = ring_class.clamp(0, NUM_RING_CLASSES - 1)
    
    # Default DBE ratios if not provided
    if dbe_c is None:
        dbe_c = torch.full((batch_size,), 0.5, device=device)
    if dbe_o is None:
        dbe_o = torch.full((batch_size,), 0.5, device=device)
    if dbe_h is None:
        dbe_h = torch.full((batch_size,), 0.25, device=device)
    if avg_ring_size is None:
        avg_ring_size = torch.full((batch_size,), 0.5, device=device)
    if ring_size_class is None:
        ring_size_class = torch.ones(batch_size, dtype=torch.long, device=device)
    
    # ★ NEW: Default n_alicyclic_rings (2 = bicyclic as baseline CRAM)
    if n_alicyclic_rings is None:
        n_alicyclic_rings = torch.full((batch_size,), 2, dtype=torch.long, device=device)
    
    # Clamp values
    ring_size_class = ring_size_class.clamp(0, 2)
    n_alicyclic_rings = n_alicyclic_rings.clamp(0, 5)  # ★ NEW
    
    # Embeddings
    cooh_emb = self.n_cooh_emb(n_cooh)                       # (batch, 8)
    ring_emb = self.ring_class_emb(ring_class)               # (batch, 16)
    ring_size_emb = self.ring_size_class_emb(ring_size_class)  # (batch, 4)
    alicyclic_emb = self.n_alicyclic_emb(n_alicyclic_rings)    # (batch, 8) ★ NEW
    
    # Continuous features
    continuous = torch.stack([
        aliphatic_pct.clamp(0, 1),
        mw.clamp(0, 1),
        has_benzene.float(),
        dbe_c.clamp(0, 1),
        dbe_o.clamp(0, 1),
        dbe_h.clamp(0, 1),
        avg_ring_size.clamp(0, 1),
    ], dim=-1)  # (batch, 7)
    
    # ★ UPDATED: Concatenate all (8 + 16 + 4 + 8 + 7 = 43)
    x = torch.cat([cooh_emb, ring_emb, ring_size_emb, alicyclic_emb, continuous], dim=-1)
    
    # MLP to condition vector
    condition = self.mlp(x)
    
    return condition


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║ STEP 3: UPDATE _extract_properties IN PropertyConditionedCRAMDataset         ║
# ╔═══════════════════════════════════════════════════════════════════════════════╝

def _extract_properties(self, mol):
    """Extract target properties from molecule - ENHANCED with n_alicyclic_rings."""
    try:
        result = is_true_cram(mol)
        
        n_cooh = result.get('n_cooh', 0)
        ring_class = result.get('ring_class', 'Unknown')
        ring_idx = RING_CLASS_TO_IDX.get(ring_class, 5)
        aliphatic_pct = result.get('aliphatic_pct', 0.5)
        mw = Descriptors.MolWt(mol)
        mw_norm = PropertyConditionEncoder.normalize_mw(mw)
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
        
        # ★ NEW: Extract n_alicyclic_rings from is_true_cram result
        n_alicyclic_rings = result.get('n_alicyclic_rings', 0)
        n_alicyclic_rings = min(n_alicyclic_rings, 5)  # Cap at 5 for embedding
        
        return {
            'n_cooh': n_cooh,
            'ring_class': ring_idx,
            'aliphatic_pct': aliphatic_pct,
            'mw': mw_norm,
            'has_benzene': has_benzene,
            'avg_ring_size': avg_ring_size_norm,
            'ring_size_class': ring_size_class,
            'n_alicyclic_rings': n_alicyclic_rings  # ★ NEW
        }
    except:
        return None


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║ STEP 4: UPDATE __getitem__ IN PropertyConditionedCRAMDataset                 ║
# ╔═══════════════════════════════════════════════════════════════════════════════╝

# In __getitem__, add n_alicyclic_rings to the returned dict:
"""
return {
    'graph': graph,
    'tokens': tokens,
    'smiles': smiles,
    'n_cooh': torch.tensor(props['n_cooh'], dtype=torch.long),
    'ring_class': torch.tensor(props['ring_class'], dtype=torch.long),
    'aliphatic_pct': torch.tensor(props['aliphatic_pct'], dtype=torch.float),
    'mw': torch.tensor(props['mw'], dtype=torch.float),
    'has_benzene': torch.tensor(props['has_benzene'], dtype=torch.float),
    'avg_ring_size': torch.tensor(props.get('avg_ring_size', 0.4), dtype=torch.float),
    'ring_size_class': torch.tensor(props.get('ring_size_class', 1), dtype=torch.long),
    'n_alicyclic_rings': torch.tensor(props.get('n_alicyclic_rings', 2), dtype=torch.long),  # ★ NEW
    'mol_features': graph.mol_features
}
"""


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║ STEP 5: UPDATE collate_conditioned_batch                                      ║
# ╔═══════════════════════════════════════════════════════════════════════════════╝

def collate_conditioned_batch(batch):
    """Collate function for property-conditioned dataset - ENHANCED."""
    graphs = [item['graph'] for item in batch]
    batched_graph = dgl.batch(graphs)
    
    tokens = torch.stack([item['tokens'] for item in batch])
    n_cooh = torch.stack([item['n_cooh'] for item in batch])
    ring_class = torch.stack([item['ring_class'] for item in batch])
    aliphatic_pct = torch.stack([item['aliphatic_pct'] for item in batch])
    mw = torch.stack([item['mw'] for item in batch])
    has_benzene = torch.stack([item['has_benzene'] for item in batch])
    avg_ring_size = torch.stack([item['avg_ring_size'] for item in batch])
    ring_size_class = torch.stack([item['ring_size_class'] for item in batch])
    n_alicyclic_rings = torch.stack([item['n_alicyclic_rings'] for item in batch])  # ★ NEW
    
    return {
        'graph': batched_graph,
        'tokens': tokens,
        'n_cooh': n_cooh,
        'ring_class': ring_class,
        'aliphatic_pct': aliphatic_pct,
        'mw': mw,
        'has_benzene': has_benzene,
        'avg_ring_size': avg_ring_size,
        'ring_size_class': ring_size_class,
        'n_alicyclic_rings': n_alicyclic_rings  # ★ NEW
    }


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║ STEP 6: UPDATE generate_fast IN FastPropertyConditionedGenerator             ║
# ╔═══════════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def generate_fast(self, n: int = 100,
                 n_cooh: int = 3,
                 ring_class: str = 'fused',
                 aliphatic_pct: float = 0.6,
                 mw: float = 400.0,
                 has_benzene: bool = True,
                 dbe_c: float = 0.5,
                 dbe_o: float = 1.0,
                 dbe_h: float = 0.25,
                 avg_ring_size: float = 5.5,
                 ring_size_class: int = 1,
                 n_alicyclic_rings: int = 2,  # ★ NEW: Target alicyclic ring count
                 temperature: float = 1.0,
                 max_length: int = 100) -> List[str]:
    """
    Generate n CRAM molecules FAST with specified properties.
    
    ★ NEW: n_alicyclic_rings parameter for polycyclic control:
        - 1 = monocyclic (single alicyclic ring)
        - 2 = bicyclic (decalin-like)
        - 3 = tricyclic (perhydrophenanthrene-like)
        - 4 = tetracyclic (steroid skeleton-like)
        - 5 = highly polycyclic
    """
    # Prepare condition tensors
    ring_idx = self.RING_CLASSES.get(ring_class.lower().replace(' ', '_'), 0)
    mw_norm = PropertyConditionEncoder.normalize_mw(mw)
    dbe_c_norm = PropertyConditionEncoder.normalize_dbe_c(dbe_c)
    dbe_o_norm = PropertyConditionEncoder.normalize_dbe_o(dbe_o)
    dbe_h_norm = PropertyConditionEncoder.normalize_dbe_h(dbe_h)
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
    n_alicyclic_t = torch.full((n,), min(n_alicyclic_rings, 5), dtype=torch.long, device=self.device)  # ★ NEW
    
    # Encode conditions with n_alicyclic_rings
    conditions = self.model.condition_encoder(
        n_cooh_t, ring_t, aliphatic_t, mw_t, benzene_t,
        dbe_c_t, dbe_o_t, dbe_h_t,
        avg_ring_size_t, ring_size_class_t,
        n_alicyclic_t  # ★ NEW
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


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║ STEP 7: ADD CONVENIENCE METHODS FOR POLYCYCLIC GENERATION                    ║
# ╔═══════════════════════════════════════════════════════════════════════════════╝

# Add these methods to FastPropertyConditionedGenerator class:

def generate_bicyclic(self, n: int = 100, n_cooh: int = 3, **kwargs) -> List[str]:
    """Generate bicyclic CRAM molecules (2 fused alicyclic rings) - decalin-like."""
    kwargs.setdefault('ring_class', 'pure_alicyclic')
    kwargs.setdefault('has_benzene', False)
    kwargs.setdefault('aliphatic_pct', 0.8)
    return self.generate_fast(n=n, n_cooh=n_cooh, n_alicyclic_rings=2, **kwargs)

def generate_tricyclic(self, n: int = 100, n_cooh: int = 4, **kwargs) -> List[str]:
    """Generate tricyclic CRAM molecules (3 fused alicyclic rings) - perhydrophenanthrene-like."""
    kwargs.setdefault('ring_class', 'pure_alicyclic')
    kwargs.setdefault('has_benzene', False)
    kwargs.setdefault('aliphatic_pct', 0.8)
    kwargs.setdefault('mw', 450.0)
    return self.generate_fast(n=n, n_cooh=n_cooh, n_alicyclic_rings=3, **kwargs)

def generate_tetracyclic(self, n: int = 100, n_cooh: int = 5, **kwargs) -> List[str]:
    """Generate tetracyclic CRAM molecules (4 fused alicyclic rings) - steroid-like."""
    kwargs.setdefault('ring_class', 'pure_alicyclic')
    kwargs.setdefault('has_benzene', False)
    kwargs.setdefault('aliphatic_pct', 0.85)
    kwargs.setdefault('mw', 500.0)
    return self.generate_fast(n=n, n_cooh=n_cooh, n_alicyclic_rings=4, **kwargs)

def generate_polycyclic(self, n: int = 100, n_alicyclic_rings: int = 2, **kwargs) -> List[str]:
    """Generate CRAM molecules with specified number of alicyclic rings."""
    kwargs.setdefault('ring_class', 'pure_alicyclic')
    kwargs.setdefault('has_benzene', False)
    kwargs.setdefault('aliphatic_pct', 0.75)
    return self.generate_fast(n=n, n_alicyclic_rings=n_alicyclic_rings, **kwargs)


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║ USAGE EXAMPLES                                                                ║
# ╔═══════════════════════════════════════════════════════════════════════════════╝

"""
After applying the above changes and re-training the model:

# Initialize generator
generator = FastPropertyConditionedGenerator(model, vocab, device='cuda')

# Generate 1000 BICYCLIC CRAM molecules (decalin-like, 2 fused rings)
bicyclic_smiles = generator.generate_bicyclic(
    n=1000,
    n_cooh=3,           # 3 carboxylic acid groups
    aliphatic_pct=0.8,  # 80% aliphatic carbon
    mw=380.0            # Target molecular weight
)

# Generate 1000 TRICYCLIC CRAM molecules (3 fused alicyclic rings)
tricyclic_smiles = generator.generate_tricyclic(
    n=1000,
    n_cooh=4,           # 4 carboxylic acid groups
    aliphatic_pct=0.8,
    mw=450.0
)

# Generate 1000 TETRACYCLIC CRAM molecules (steroid-like, 4 fused rings)
tetracyclic_smiles = generator.generate_tetracyclic(
    n=1000,
    n_cooh=5,           # 5 carboxylic acid groups
    mw=520.0
)

# Or use generate_fast with explicit n_alicyclic_rings:
custom_polycyclic = generator.generate_fast(
    n=500,
    n_alicyclic_rings=3,  # Tricyclic
    n_cooh=4,
    ring_class='pure_alicyclic',
    has_benzene=False,
    aliphatic_pct=0.85,
    mw=460.0
)

# Generate million with polycyclic distribution:
def polycyclic_sampler():
    import random
    n_alicyclic = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.35, 0.30, 0.20, 0.05])[0]
    return {
        'n_cooh': random.choice([2, 3, 3, 4, 4, 5] if n_alicyclic > 2 else [2, 3, 3, 4]),
        'ring_class': 'pure_alicyclic',
        'aliphatic_pct': 0.65 + n_alicyclic * 0.05,
        'mw': 280 + n_alicyclic * 50,
        'has_benzene': False,
        'dbe_c': random.uniform(0.3, 0.5),
        'dbe_o': random.uniform(0.5, 1.5),
        'dbe_h': random.uniform(0.15, 0.30),
        'n_alicyclic_rings': n_alicyclic
    }

df = generator.generate_million(n=1_000_000, property_sampler=polycyclic_sampler)
"""

print("✅ Polycyclic CRAM Generation Integration Guide loaded!")
print("\nKEY CHANGES REQUIRED:")
print("1. Add n_alicyclic_emb to PropertyConditionEncoder.__init__")
print("2. Update PropertyConditionEncoder.forward with n_alicyclic_rings param")
print("3. Update _extract_properties to extract n_alicyclic_rings")
print("4. Update __getitem__ and collate_conditioned_batch")
print("5. Update generate_fast with n_alicyclic_rings param")
print("6. Add convenience methods: generate_bicyclic, generate_tricyclic, etc.")
print("\n⚠️  IMPORTANT: Re-train the model after making these changes!")
