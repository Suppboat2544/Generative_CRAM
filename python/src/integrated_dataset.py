"""Notebook IntegratedCRAMDataset."""
from __future__ import annotations
from typing import List, Optional, Dict
import torch
from torch.utils.data import Dataset
from rdkit import Chem

# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATED CRAM DATASET
# ══════════════════════════════════════════════════════════════════════════════
# 
# Full dataset class that integrates ALL deep improvements:
# - 163D node features (base + synthetic + marine + ring-chain + stereo + terpenoid + FG + oxidation)
# - 17D molecular features (photostability + terpenoid + conf + CRAM class)
# - SMILES augmentation for robustness
# - Feature caching for speed
# ══════════════════════════════════════════════════════════════════════════════

class IntegratedCRAMDataset(torch.utils.data.Dataset):
    """
    Complete dataset for DeepCRAMVAE training with all 163D features.
    
    Features per atom (163D total):
    - Base features: 45D (element, hybridization, etc.)
    - Synthetic features: 28D (extended properties)
    - Marine features: 15D (marine DOM-specific)
    - Ring-chain features: 20D (alicyclic detection)
    - Stereochemistry: 12D (chirality, conformation)
    - Terpenoid ancestry: 10D (isoprene patterns)
    - Functional groups: 13D (group indicators)
    - Enhanced oxidation: 20D (oxidation state, COOH proximity)
    
    Molecular features (17D):
    - Photostability: 6D
    - Terpenoid ancestry (mol): 5D
    - Conformational stability: 1D
    - CRAM class indicators: 5D
    """
    
    def __init__(self, smiles_list, vocab, max_len=128, 
                 use_augmentation=True, cache_features=True, feature_extractor=None):
        """
        Args:
            smiles_list: List of SMILES strings
            vocab: SELFIES vocabulary
            max_len: Maximum sequence length
            use_augmentation: Whether to use SMILES randomization
            cache_features: Whether to cache computed features
            feature_extractor: Pre-initialized DeepCRAMFeatureExtractor (if None, creates new one)
        """
        self.smiles_list = smiles_list
        self.vocab = vocab
        self.max_len = max_len
        self.use_augmentation = use_augmentation
        self.cache_features = cache_features
        
        # Initialize feature extractors
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = DeepCRAMFeatureExtractor() if 'DeepCRAMFeatureExtractor' in dir() else None
        self.augmenter = CRAMAugmenter() if 'CRAMAugmenter' in dir() and use_augmentation else None
        
        # Feature cache
        self._cache = {}
        
        # Filter valid SMILES (skip isolated hydrogens that break RDKit matrices)
        self.valid_indices = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                continue
            if any(a.GetAtomicNum() == 1 and a.GetDegree() == 0 for a in mol.GetAtoms()):
                continue
            self.valid_indices.append(i)
        
        print(f"IntegratedCRAMDataset initialized:")
        print(f" Total SMILES: {len(smiles_list)}")
        print(f" Valid SMILES: {len(self.valid_indices)}")
        print(f" Augmentation: {'Enabled' if use_augmentation else 'Disabled'}")
        print(f" Feature caching: {'Enabled' if cache_features else 'Disabled'}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def _get_node_features(self, mol):
        """Extract 163D node features for each atom."""
        if self.feature_extractor is not None:
            try:
                # Use extract_all_features which returns a dict with 'atom_features'
                features = self.feature_extractor.extract_all_features(mol)
                return features['atom_features']
            except Exception as e:
                print(f"Warning: Feature extraction failed: {e}")
                # Fall through to fallback
        
        # Fallback to basic features if extractor not available
        features = []
        for atom in mol.GetAtoms():
            feat = [
                atom.GetAtomicNum() / 100.0,
                atom.GetTotalDegree() / 4.0,
                atom.GetFormalCharge() / 2.0,
                atom.GetNumRadicalElectrons() / 2.0,
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
                atom.GetTotalNumHs() / 4.0,
            ]
            # Pad to expected dimension (163D)
            feat.extend([0.0] * (163 - len(feat)))
            features.append(feat)
        return np.array(features, dtype=np.float32)
    
    def _get_molecular_features(self, mol, smiles):
        """Extract 17D molecular-level features."""
        if self.feature_extractor is not None:
            try:
                # Use extract_all_features which returns a dict with 'mol_features'
                features = self.feature_extractor.extract_all_features(mol)
                return features['mol_features']
            except Exception as e:
                print(f"Warning: Molecular feature extraction failed: {e}")
                # Fall through to fallback
        
        # Basic molecular features fallback
        return np.zeros(17, dtype=np.float32)
    
    def _smiles_to_graph(self, smiles, node_features, mol_features):
        """Convert SMILES to DGL graph with all features."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None
        if node_features is None or len(node_features) != num_atoms:
            return None

        # Build undirected edges; skip self-bonds (aid1 == aid2 crashes RDKit/DGL)
        src_list, dst_list = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i == j:
                continue
            src_list.extend([i, j])
            dst_list.extend([j, i])

        # Add self-loops with torch.int64 (avoid dgl.add_self_loop numpy.int64 crash)
        for i in range(num_atoms):
            src_list.append(i)
            dst_list.append(i)

        g = dgl.graph(
            (torch.tensor(src_list, dtype=torch.int64),
             torch.tensor(dst_list, dtype=torch.int64)),
            num_nodes=num_atoms,
        )

        g.ndata['feat'] = torch.as_tensor(node_features, dtype=torch.float32)
        g.mol_features = torch.as_tensor(mol_features, dtype=torch.float32)
        return g
    
    def _tokenize_smiles(self, smiles):
        """Convert SMILES to SELFIES tokens."""
        try:
            selfies = sf.encoder(smiles)
            if selfies is None:
                return None
            
            tokens = list(sf.split_selfies(selfies))
            unk = self.vocab.get('<UNK>', self.vocab.get('<PAD>', 0))
            token_ids = [self.vocab.get(t, unk) for t in tokens]

            if '<SOS>' in self.vocab:
                token_ids = [self.vocab['<SOS>']] + token_ids
            elif '[start]' in self.vocab:
                token_ids = [self.vocab['[start]']] + token_ids
            if '<EOS>' in self.vocab:
                token_ids = token_ids + [self.vocab['<EOS>']]
            elif '[end]' in self.vocab:
                token_ids = token_ids + [self.vocab['[end]']]

            if len(token_ids) > self.max_len:
                token_ids = token_ids[:self.max_len]
            else:
                pad_id = self.vocab.get('<PAD>', self.vocab.get('[nop]', 0))
                token_ids = token_ids + [pad_id] * (self.max_len - len(token_ids))
            
            return torch.tensor(token_ids, dtype=torch.long)
        except:
            return None
    
    def __getitem__(self, idx):
        """Get a single sample with all features."""
        try:
            real_idx = self.valid_indices[idx]
            smiles = self.smiles_list[real_idx]

            cache_key = smiles
            if self.cache_features and cache_key in self._cache:
                return self._cache[cache_key]

            if self.use_augmentation and self.augmenter is not None:
                try:
                    aug = self.augmenter.randomize_smiles(Chem.MolFromSmiles(smiles), n_augments=1)
                    if isinstance(aug, (list, tuple)) and aug:
                        smiles = aug[0]
                    elif isinstance(aug, str):
                        smiles = aug
                except Exception:
                    pass

            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                return self._get_dummy_sample()

            node_features = self._get_node_features(mol)
            mol_features = self._get_molecular_features(mol, smiles)

            graph = self._smiles_to_graph(smiles, node_features, mol_features)
            if graph is None:
                return self._get_dummy_sample()

            tokens = self._tokenize_smiles(smiles)
            if tokens is None:
                return self._get_dummy_sample()

            is_cram = self._is_cram_molecule(mol)

            sample = {
                'graph': graph,
                'tokens': tokens,
                'smiles': smiles,
                'is_cram': torch.tensor(is_cram, dtype=torch.float32),
                'mol_features': torch.tensor(mol_features, dtype=torch.float32)
            }

            if self.cache_features:
                self._cache[cache_key] = sample

            return sample
        except Exception:
            return self._get_dummy_sample()
    
    def _is_cram_molecule(self, mol):
        """Check if molecule is TRUE CRAM (2+ COOH, alicyclic)."""
        # Count carboxyl groups
        cooh_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
        matches = mol.GetSubstructMatches(cooh_pattern) if cooh_pattern else []
        num_cooh = len(matches)
        
        # Check alicyclic content
        ring_info = mol.GetRingInfo()
        has_alicyclic = False
        for ring in ring_info.AtomRings():
            is_aromatic = any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
            if not is_aromatic and 5 <= len(ring) <= 7:
                has_alicyclic = True
                break
        
        return 1.0 if (num_cooh >= 2 or has_alicyclic) else 0.0
    
    def _get_dummy_sample(self):
        """Return a dummy sample for invalid molecules."""
        g = dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=1)
        g.ndata['feat'] = torch.zeros((1, 163), dtype=torch.float32) # Updated to 163D
        g.mol_features = torch.zeros(17, dtype=torch.float32)
        
        return {
            'graph': g,
            'tokens': torch.zeros(self.max_len, dtype=torch.long),
            'smiles': '',
            'is_cram': torch.tensor(0.0, dtype=torch.float32),
            'mol_features': torch.zeros(17, dtype=torch.float32)
        }


def collate_integrated_batch(batch):
    """Collate function for IntegratedCRAMDataset."""
    graphs = [item['graph'] for item in batch]
    tokens = torch.stack([item['tokens'] for item in batch])
    is_cram = torch.stack([item['is_cram'] for item in batch])
    mol_features = torch.stack([item['mol_features'] for item in batch])
    smiles = [item['smiles'] for item in batch]
    
    # Batch graphs
    batched_graph = dgl.batch(graphs)
    
    # Transfer molecular features to batched graph
    batched_graph.mol_features = mol_features
    
    return {
        'graph': batched_graph,
        'tokens': tokens,
        'is_cram': is_cram,
        'mol_features': mol_features,
        'smiles': smiles
    }
