"""
CRAM Dataset Classes
====================

PyTorch Dataset implementations for CRAM molecule training.
"""

from typing import List, Dict, Optional, Callable
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import dgl
from rdkit import Chem

from src.features import DeepCRAMFeatureExtractor
from src.utils import SMILESAugmenter, canonicalize_smiles


def smiles_to_graph(
    smiles: str, 
    feature_extractor: Optional[DeepCRAMFeatureExtractor] = None
) -> Optional[dgl.DGLGraph]:
    """
    Convert SMILES to DGL graph with 163D features.
    
    Args:
        smiles: SMILES string
        feature_extractor: Feature extractor instance
        
    Returns:
        DGL graph with node features, or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    if feature_extractor is None:
        feature_extractor = DeepCRAMFeatureExtractor()
    
    features = feature_extractor.extract(smiles)
    if features is None:
        return None
    
    # Build edges from bonds
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.extend([i, j])
        dst.extend([j, i])
    
    # Create graph
    if src:
        g = dgl.graph((src, dst), num_nodes=mol.GetNumAtoms())
    else:
        g = dgl.graph(([], []), num_nodes=mol.GetNumAtoms())
    
    # Add self-loops (critical for GNN stability)
    g = dgl.add_self_loop(g)
    
    # Add features
    g.ndata['feat'] = torch.tensor(features['atom_features'], dtype=torch.float32)
    g.ndata['group'] = torch.tensor(features['group_membership'], dtype=torch.float32)
    g.mol_features = torch.tensor(features['mol_features'], dtype=torch.float32)
    
    return g


class CRAMDataset(Dataset):
    """
    Basic CRAM Dataset with 163D features.
    
    Converts SMILES to molecular graphs with full feature extraction.
    """
    
    def __init__(
        self,
        smiles_list: List[str],
        tokenizer,
        feature_extractor: Optional[DeepCRAMFeatureExtractor] = None,
        use_augmentation: bool = False,
        augment_prob: float = 0.8,
        cache: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            smiles_list: List of SMILES strings
            tokenizer: SELFIESTokenizer instance
            feature_extractor: Feature extractor
            use_augmentation: Enable SMILES augmentation
            augment_prob: Probability of augmentation per sample
            cache: Cache computed graphs
        """
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor or DeepCRAMFeatureExtractor()
        self.use_augmentation = use_augmentation
        self.augment_prob = augment_prob
        self.augmenter = SMILESAugmenter() if use_augmentation else None
        
        # Filter valid molecules
        self.valid_indices = []
        for i, smi in enumerate(smiles_list):
            if Chem.MolFromSmiles(smi) is not None:
                self.valid_indices.append(i)
        
        # Cache
        self.cache = {} if cache else None
        
        print(f"CRAMDataset: {len(self.valid_indices)}/{len(smiles_list)} valid")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def _is_cram(self, mol: Chem.Mol) -> float:
        """Check if molecule is TRUE CRAM."""
        from src.validator import CRAMValidator
        validator = CRAMValidator()
        result = validator.validate(mol)
        return 1.0 if result.get('is_cram', False) else 0.0
    
    def __getitem__(self, idx: int) -> Dict:
        original_smiles = self.smiles_list[self.valid_indices[idx]]
        
        # Apply augmentation
        if self.use_augmentation and random.random() < self.augment_prob:
            smiles = self.augmenter.randomize_smiles(original_smiles)
        else:
            smiles = original_smiles
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            smiles = original_smiles
            mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            # Return dummy data
            return self._dummy_item()
        
        # Get canonical for caching
        canonical = Chem.MolToSmiles(mol, canonical=True)
        
        # Check cache
        if self.cache is not None and canonical in self.cache:
            graph, mol_features, is_cram = self.cache[canonical]
            graph = graph.clone()
        else:
            graph = smiles_to_graph(canonical, self.feature_extractor)
            if graph is None:
                return self._dummy_item()
            
            is_cram = self._is_cram(mol)
            mol_features = graph.mol_features
            
            if self.cache is not None:
                self.cache[canonical] = (graph.clone(), mol_features.clone(), is_cram)
        
        # Tokenize (possibly augmented SMILES)
        tokens = torch.tensor(self.tokenizer.encode(smiles), dtype=torch.long)
        
        return {
            'graph': graph,
            'tokens': tokens,
            'smiles': smiles,
            'is_cram': torch.tensor(is_cram, dtype=torch.float32),
            'mol_features': mol_features
        }
    
    def _dummy_item(self) -> Dict:
        """Return dummy item for invalid molecules."""
        g = dgl.graph(([0], [0]), num_nodes=1)
        g.ndata['feat'] = torch.zeros((1, 163), dtype=torch.float32)
        g.ndata['group'] = torch.zeros((1, 13), dtype=torch.float32)
        g.mol_features = torch.zeros(17, dtype=torch.float32)
        
        return {
            'graph': g,
            'tokens': torch.zeros(self.tokenizer.max_len, dtype=torch.long),
            'smiles': '',
            'is_cram': torch.tensor(0.0),
            'mol_features': torch.zeros(17)
        }


class SAFilteredDataset(Dataset):
    """
    Dataset filtered by Synthetic Accessibility score.
    
    Only includes molecules with SA score below threshold.
    """
    
    def __init__(
        self,
        smiles_list: List[str],
        tokenizer,
        feature_extractor: Optional[DeepCRAMFeatureExtractor] = None,
        sa_threshold: float = 4.5,
        cache: bool = True
    ):
        from src.utils import SyntheticAccessibilityCalculator
        
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor or DeepCRAMFeatureExtractor()
        self.sa_calculator = SyntheticAccessibilityCalculator()
        
        # Filter by SA score
        self.valid_smiles = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol and self.sa_calculator.calculate(mol) <= sa_threshold:
                self.valid_smiles.append(smi)
        
        self.cache = {} if cache else None
        print(f"SAFilteredDataset: {len(self.valid_smiles)}/{len(smiles_list)} passed SA filter")
    
    def __len__(self) -> int:
        return len(self.valid_smiles)
    
    def __getitem__(self, idx: int) -> Dict:
        smiles = self.valid_smiles[idx]
        canonical = canonicalize_smiles(smiles)
        
        if self.cache is not None and canonical in self.cache:
            return self.cache[canonical].copy()
        
        graph = smiles_to_graph(smiles, self.feature_extractor)
        if graph is None:
            return self._dummy_item()
        
        tokens = torch.tensor(self.tokenizer.encode(smiles), dtype=torch.long)
        
        item = {
            'graph': graph,
            'tokens': tokens,
            'smiles': smiles,
            'mol_features': graph.mol_features
        }
        
        if self.cache is not None:
            self.cache[canonical] = item.copy()
        
        return item
    
    def _dummy_item(self) -> Dict:
        g = dgl.graph(([0], [0]), num_nodes=1)
        g.ndata['feat'] = torch.zeros((1, 163), dtype=torch.float32)
        g.ndata['group'] = torch.zeros((1, 13), dtype=torch.float32)
        g.mol_features = torch.zeros(17, dtype=torch.float32)
        return {
            'graph': g,
            'tokens': torch.zeros(self.tokenizer.max_len, dtype=torch.long),
            'smiles': '',
            'mol_features': torch.zeros(17)
        }


def collate_cram_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for batching CRAM data.
    
    Args:
        batch: List of sample dicts from dataset
        
    Returns:
        Batched dict with stacked tensors and batched graph
    """
    graphs = dgl.batch([item['graph'] for item in batch])
    graphs.mol_features = torch.stack([item['mol_features'] for item in batch])
    
    result = {
        'graph': graphs,
        'tokens': torch.stack([item['tokens'] for item in batch]),
        'mol_features': graphs.mol_features,
        'smiles': [item['smiles'] for item in batch]
    }
    
    if 'is_cram' in batch[0]:
        result['is_cram'] = torch.stack([item['is_cram'] for item in batch])
    
    return result


def create_dataloaders(
    train_smiles: List[str],
    val_smiles: List[str],
    tokenizer,
    batch_size: int = 32,
    use_augmentation: bool = True,
    num_workers: int = 0
) -> tuple:
    """
    Create train and validation dataloaders.
    
    Args:
        train_smiles: Training SMILES
        val_smiles: Validation SMILES
        tokenizer: SELFIES tokenizer
        batch_size: Batch size
        use_augmentation: Enable augmentation for training
        num_workers: DataLoader workers
        
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = CRAMDataset(
        train_smiles, 
        tokenizer,
        use_augmentation=use_augmentation
    )
    
    val_dataset = CRAMDataset(
        val_smiles,
        tokenizer,
        use_augmentation=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_cram_batch,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_cram_batch,
        num_workers=num_workers
    )
    
    return train_loader, val_loader
