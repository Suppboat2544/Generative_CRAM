"""
Fast Property-Conditioned CVAE Generator
=========================================
Generate molecules with specific property conditions.
Matches the notebook CRAM_Generator_Production2.ipynb implementation.
"""

import random
from typing import List, Optional, Dict, Callable
from collections import Counter

import torch
import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from .base import get_device
from ..models.condition import PropertyConditionEncoder


class FastPropertyConditionedGenerator:
    """
    FAST property-conditioned CRAM generation using trained CVAE.

    Key difference from old generator:
    - NO validation loop - CVAE generates valid CRAM directly
    - Batch generation - generates all molecules in parallel
    - ~100x faster than mutation-based generation

    Token IDs (aligned with SELFIESTokenizer):
    - PAD: 0
    - SOS: 1
    - EOS: 2
    - UNK: 3

    Example:
        generator = FastPropertyConditionedGenerator(model, vocab, device='cuda')

        # Generate 1000 molecules in ~1 second (vs minutes with old method)
        smiles = generator.generate_fast(
            n=1000,
            n_cooh=3,
            ring_class='fused',
            dbe_c=0.5
        )
    """

    # Ring class mapping
    RING_CLASSES = {
        'fused': 0, 'fused_aro_alicyclic': 0,
        'mixed': 1, 'mixed_aro_alicyclic': 1,
        'benzene_aliphatic': 2,
        'pure_alicyclic': 3,
        'pure_aromatic': 4,
        'acyclic': 5
    }

    # Allowed atoms for CRAM molecules
    ALLOWED_ATOMS = {6, 1, 8, 7, 16, 9, 17, 35, 53}  # C, H, O, N, S, F, Cl, Br, I

    def __init__(
        self,
        model,
        selfies_vocab: dict,
        idx_to_token: dict = None,
        device=None,
        # Legacy parameters for backward compatibility
        tokenizer=None,
        sos_idx: int = None,
        eos_idx: int = None,
        pad_idx: int = None
    ):
        """
        Initialize generator.

        Args:
            model: PropertyConditionedCRAMVAE model
            selfies_vocab: Vocabulary dict (token -> index)
            idx_to_token: Reverse vocabulary (index -> token)
            device: Torch device
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.selfies_vocab = selfies_vocab
        self.idx_to_token = idx_to_token or {v: k for k, v in selfies_vocab.items()}
        self.latent_dim = model.latent_dim

        # Special tokens - from vocab or defaults
        self.sos_token = sos_idx if sos_idx is not None else selfies_vocab.get('<SOS>', 1)
        self.eos_token = eos_idx if eos_idx is not None else selfies_vocab.get('<EOS>', 2)
        self.pad_token = pad_idx if pad_idx is not None else selfies_vocab.get('<PAD>', 0)

        # Support legacy tokenizer parameter
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = None

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
                     temperature: float = 1.0,
                     max_length: int = 100) -> List[str]:
        """
        Generate n CRAM molecules FAST with specified properties.

        NO validation loop - CVAE generates valid CRAM directly!

        Args:
            n: Number of molecules to generate
            n_cooh: Target COOH groups (2-6)
            ring_class: 'fused', 'mixed', 'benzene_aliphatic', 'pure_alicyclic'
            aliphatic_pct: Aliphatic carbon fraction (0.4-0.9)
            mw: Target molecular weight (100-800)
            has_benzene: Whether to include benzene rings
            dbe_c: Target DBE/C ratio (0.3-0.7)
            dbe_o: Target DBE/O ratio (0.5-2.5)
            dbe_h: Target DBE/H ratio (0.1-0.4)
            avg_ring_size: Target average ring size (5.0-7.0 for CRAM)
            ring_size_class: 0=strained(3-4), 1=CRAM-preferred(5-7), 2=large(8+)
            temperature: Sampling temperature (1.0 = normal, >1 = more diverse)
            max_length: Maximum SELFIES sequence length

        Returns:
            List of SMILES strings
        """
        # Prepare condition tensors
        ring_idx = self.RING_CLASSES.get(ring_class.lower().replace(' ', '_'), 0)
        mw_norm = PropertyConditionEncoder.normalize_mw(mw)
        dbe_c_norm = PropertyConditionEncoder.normalize_dbe_c(dbe_c)
        dbe_o_norm = PropertyConditionEncoder.normalize_dbe_o(dbe_o)
        dbe_h_norm = PropertyConditionEncoder.normalize_dbe_h(dbe_h)

        # Create batch tensors
        n_cooh_t = torch.full((n,), n_cooh, dtype=torch.long, device=self.device)
        ring_t = torch.full((n,), ring_idx, dtype=torch.long, device=self.device)
        aliphatic_t = torch.full((n,), aliphatic_pct, dtype=torch.float32, device=self.device)
        mw_t = torch.full((n,), mw_norm, dtype=torch.float32, device=self.device)
        benzene_t = torch.full((n,), float(has_benzene), dtype=torch.float32, device=self.device)
        dbe_c_t = torch.full((n,), dbe_c_norm, dtype=torch.float32, device=self.device)
        dbe_o_t = torch.full((n,), dbe_o_norm, dtype=torch.float32, device=self.device)
        dbe_h_t = torch.full((n,), dbe_h_norm, dtype=torch.float32, device=self.device)

        # Ring size tensors
        avg_ring_size_norm = (avg_ring_size - 3.0) / 6.0
        avg_ring_size_t = torch.full((n,), avg_ring_size_norm, dtype=torch.float32, device=self.device)
        ring_size_class_t = torch.full((n,), ring_size_class, dtype=torch.long, device=self.device)

        # Encode conditions
        conditions = self.model.condition_encoder(
            n_cooh_t, ring_t, aliphatic_t, mw_t, benzene_t,
            dbe_c_t, dbe_o_t, dbe_h_t,
            avg_ring_size_t, ring_size_class_t
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

        # Decode to SMILES
        smiles_list = self._batch_decode(generated)

        return smiles_list

    def _batch_decode(self, token_indices: torch.Tensor) -> List[str]:
        """Fast batch decode tokens to SMILES."""
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

    def generate_batch_properties(self,
                                  properties_list: List[Dict],
                                  temperature: float = 1.0) -> List[str]:
        """
        Generate molecules with DIFFERENT properties in a single batch.

        Args:
            properties_list: List of property dicts, each with keys:
                             n_cooh, ring_class, aliphatic_pct, mw, dbe_c, dbe_o, dbe_h
            temperature: Sampling temperature

        Returns:
            List of SMILES, one per property dict
        """
        n = len(properties_list)

        n_cooh_list = []
        ring_list = []
        aliphatic_list = []
        mw_list = []
        benzene_list = []
        dbe_c_list = []
        dbe_o_list = []
        dbe_h_list = []

        for props in properties_list:
            n_cooh_list.append(props.get('n_cooh', 3))
            ring_idx = self.RING_CLASSES.get(props.get('ring_class', 'fused').lower(), 0)
            ring_list.append(ring_idx)
            aliphatic_list.append(props.get('aliphatic_pct', 0.6))
            mw_list.append(PropertyConditionEncoder.normalize_mw(props.get('mw', 400)))
            benzene_list.append(float(props.get('has_benzene', True)))
            dbe_c_list.append(PropertyConditionEncoder.normalize_dbe_c(props.get('dbe_c', 0.5)))
            dbe_o_list.append(PropertyConditionEncoder.normalize_dbe_o(props.get('dbe_o', 1.0)))
            dbe_h_list.append(PropertyConditionEncoder.normalize_dbe_h(props.get('dbe_h', 0.25)))

        n_cooh_t = torch.tensor(n_cooh_list, dtype=torch.long, device=self.device)
        ring_t = torch.tensor(ring_list, dtype=torch.long, device=self.device)
        aliphatic_t = torch.tensor(aliphatic_list, dtype=torch.float32, device=self.device)
        mw_t = torch.tensor(mw_list, dtype=torch.float32, device=self.device)
        benzene_t = torch.tensor(benzene_list, dtype=torch.float32, device=self.device)
        dbe_c_t = torch.tensor(dbe_c_list, dtype=torch.float32, device=self.device)
        dbe_o_t = torch.tensor(dbe_o_list, dtype=torch.float32, device=self.device)
        dbe_h_t = torch.tensor(dbe_h_list, dtype=torch.float32, device=self.device)

        conditions = self.model.condition_encoder(
            n_cooh_t, ring_t, aliphatic_t, mw_t, benzene_t,
            dbe_c_t, dbe_o_t, dbe_h_t
        )

        z = torch.randn(n, self.latent_dim, device=self.device) * temperature
        generated = self.model.generate_conditioned(z=z, conditions=conditions)

        return self._batch_decode(generated)

    def generate_unique(
        self,
        target_count: int,
        condition_sampler: Optional[Callable] = None,
        temperature: float = 1.0,
        batch_size: int = 256,
        max_attempts: Optional[int] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Generate unique molecules until target count is reached.

        Args:
            target_count: Number of unique molecules to generate
            condition_sampler: Callable that returns property dict
            temperature: Sampling temperature
            batch_size: Batch size for generation
            max_attempts: Maximum generation attempts
            show_progress: Show tqdm progress bar

        Returns:
            DataFrame with SMILES and properties
        """
        if max_attempts is None:
            max_attempts = target_count * 10

        unique_smiles = set()
        results = []
        attempts = 0

        if condition_sampler is None:
            def condition_sampler():
                return {
                    'n_cooh': random.choice([2, 3, 3, 4, 4, 5]),
                    'ring_class': random.choice(['fused', 'mixed', 'benzene_aliphatic', 'pure_alicyclic']),
                    'aliphatic_pct': random.uniform(0.5, 0.8),
                    'mw': random.uniform(200, 500),
                    'has_benzene': random.choice([True, False]),
                    'dbe_c': random.uniform(0.3, 0.7),
                    'dbe_o': random.uniform(0.5, 2.0),
                    'dbe_h': random.uniform(0.1, 0.4)
                }

        pbar = tqdm(total=target_count, desc="Generating") if show_progress else None

        while len(unique_smiles) < target_count and attempts < max_attempts:
            props = condition_sampler()
            batch_smiles = self.generate_fast(
                n=batch_size,
                temperature=temperature,
                **props
            )

            for smiles in batch_smiles:
                if smiles not in unique_smiles:
                    unique_smiles.add(smiles)
                    result = {'smiles': smiles}
                    result.update(props)
                    results.append(result)

                    if pbar:
                        pbar.update(1)

                    if len(unique_smiles) >= target_count:
                        break

            attempts += batch_size

        if pbar:
            pbar.close()

        return pd.DataFrame(results)

    def generate_million(self, n: int = 1_000_000,
                        batch_size: int = 1000,
                        property_sampler: Callable = None,
                        save_path: str = None) -> pd.DataFrame:
        """
        Generate MILLION molecules with mixed properties.

        Args:
            n: Total molecules to generate
            batch_size: Batch size for generation
            property_sampler: Function that returns random property dict
            save_path: If provided, saves results to CSV

        Returns:
            DataFrame with SMILES and properties
        """
        if property_sampler is None:
            def property_sampler():
                return {
                    'n_cooh': random.choice([2, 3, 3, 4, 4, 5]),
                    'ring_class': random.choice(['fused', 'fused', 'mixed', 'benzene_aliphatic', 'pure_alicyclic']),
                    'aliphatic_pct': random.uniform(0.5, 0.8),
                    'mw': random.uniform(200, 600),
                    'has_benzene': random.choice([True, True, False]),
                    'dbe_c': random.uniform(0.3, 0.7),
                    'dbe_o': random.uniform(0.5, 2.0),
                    'dbe_h': random.uniform(0.1, 0.4)
                }

        all_smiles = []
        unique_smiles = set()
        num_batches = (n + batch_size - 1) // batch_size

        for _ in tqdm(range(num_batches), desc="Generating batches"):
            props = property_sampler()
            batch_smiles = self.generate_fast(n=batch_size, **props)

            for smi in batch_smiles:
                if smi not in unique_smiles:
                    unique_smiles.add(smi)
                    all_smiles.append(smi)

                    if len(all_smiles) >= n:
                        break

            if len(all_smiles) >= n:
                break

        df = pd.DataFrame({'smiles': all_smiles[:n]})

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"✓ Saved {len(df):,} molecules to {save_path}")

        print(f"✓ Generated {len(df):,} valid CRAM molecules")
        return df

