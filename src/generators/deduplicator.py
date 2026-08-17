"""
Hybrid Deduplicator
===================
Bloom filter + exact set for high-value molecules.
"""

from typing import Optional, Set, Tuple
from rdkit import Chem

from .bloom_filter import BloomFilter


class HybridDeduplicator:
    """
    Hybrid deduplication: Bloom filter + exact set for high-value molecules.
    
    Strategy:
    1. Bloom filter for fast approximate dedup (all molecules)
    2. Exact set for verified unique CRAM molecules (smaller subset)
    3. Canonical SMILES for exact matching
    """
    
    def __init__(
        self, 
        expected_total: int = 1_000_000, 
        expected_unique_cram: int = 100_000
    ):
        """
        Initialize hybrid deduplicator.
        
        Args:
            expected_total: Expected total candidates
            expected_unique_cram: Expected unique CRAM molecules
        """
        self.bloom = BloomFilter(expected_total, false_positive_rate=0.001)
        self.exact_cram_set: Set[str] = set()
        
        self.stats = {
            'total_checked': 0,
            'bloom_rejects': 0,
            'exact_rejects': 0,
            'unique_added': 0
        }
    
    def canonicalize(self, smiles: str) -> Optional[str]:
        """Get canonical SMILES for exact matching."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        except:
            pass
        return None
    
    def add_unique(self, smiles: str, is_cram: bool = False) -> bool:
        """Add a unique molecule. Returns True if successfully added."""
        canonical = self.canonicalize(smiles)
        if not canonical:
            return False
        
        if canonical in self.bloom:
            self.stats['bloom_rejects'] += 1
            return False
        
        self.bloom.add(canonical)
        self.stats['unique_added'] += 1
        
        if is_cram:
            self.exact_cram_set.add(canonical)
        
        return True
    
    def check_and_add(self, smiles: str, is_cram: bool = False) -> Tuple[bool, Optional[str]]:
        """Combined check and add. Returns (is_unique, canonical_smiles)."""
        canonical = self.canonicalize(smiles)
        if not canonical:
            return False, None
        
        self.stats['total_checked'] += 1
        
        if canonical in self.bloom:
            self.stats['bloom_rejects'] += 1
            return False, canonical
        
        self.bloom.add(canonical)
        self.stats['unique_added'] += 1
        
        if is_cram:
            self.exact_cram_set.add(canonical)
        
        return True, canonical
    
    def get_unique_cram_count(self) -> int:
        """Get count of verified unique CRAM molecules."""
        return len(self.exact_cram_set)
    
    def clear(self):
        """Clear all data."""
        self.bloom.clear()
        self.exact_cram_set.clear()
        self.stats = {
            'total_checked': 0,
            'bloom_rejects': 0,
            'exact_rejects': 0,
            'unique_added': 0
        }
