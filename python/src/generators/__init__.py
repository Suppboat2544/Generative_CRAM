"""
CRAM Generators Package
=======================
Modular generator components for CRAM molecule generation.
"""

from .bloom_filter import BloomFilter
from .deduplicator import HybridDeduplicator
from .mutation_gen import SELFIESMutationGenerator
from .cvae_gen import FastPropertyConditionedGenerator
from .base import get_device

try:
    from .high_throughput import HighThroughputCRAMGenerator
    from .fast_latent import FastLatentCRAMGenerator
except Exception:  # notebook generators need optional runtime deps
    HighThroughputCRAMGenerator = None
    FastLatentCRAMGenerator = None

__all__ = [
    "BloomFilter",
    "HybridDeduplicator",
    "SELFIESMutationGenerator",
    "FastPropertyConditionedGenerator",
    "HighThroughputCRAMGenerator",
    "FastLatentCRAMGenerator",
    "get_device",
]
