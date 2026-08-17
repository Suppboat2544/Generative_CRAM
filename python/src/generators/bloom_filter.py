"""
Bloom Filter
============
Memory-efficient duplicate detection.
"""

import hashlib
import numpy as np
from typing import Optional

try:
    import mmh3
except ImportError:
    mmh3 = None


class BloomFilter:
    """
    Memory-efficient Bloom filter for duplicate detection.
    
    Uses multiple hash functions to check set membership with
    configurable false positive rate.
    """
    
    def __init__(
        self, 
        expected_items: int = 10_000_000, 
        false_positive_rate: float = 0.001
    ):
        """
        Initialize Bloom filter.
        
        Args:
            expected_items: Expected number of items
            false_positive_rate: Target false positive rate
        """
        self.size = self._optimal_size(expected_items, false_positive_rate)
        self.num_hashes = self._optimal_hashes(self.size, expected_items)
        self.bit_array = np.zeros(self.size, dtype=np.bool_)
        self.count = 0
    
    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit array size."""
        return int(-n * np.log(p) / (np.log(2) ** 2))
    
    @staticmethod
    def _optimal_hashes(m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        return int(m / n * np.log(2))
    
    def _hash(self, item: str, seed: int) -> int:
        """Generate hash value for item."""
        if mmh3:
            return mmh3.hash(item, seed) % self.size
        else:
            h = hashlib.md5(f"{item}_{seed}".encode()).hexdigest()
            return int(h, 16) % self.size
    
    def add(self, item: str):
        """Add item to filter."""
        for seed in range(self.num_hashes):
            idx = self._hash(item, seed)
            self.bit_array[idx] = True
        self.count += 1
    
    def __contains__(self, item: str) -> bool:
        """Check if item might be in filter."""
        return all(
            self.bit_array[self._hash(item, seed)] 
            for seed in range(self.num_hashes)
        )
    
    def __len__(self) -> int:
        return self.count
    
    def clear(self):
        """Clear the filter."""
        self.bit_array = np.zeros(self.size, dtype=np.bool_)
        self.count = 0
