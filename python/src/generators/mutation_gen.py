"""
SELFIES Mutation Generator
==========================
Mutation-based molecule generation using SELFIES.
"""

import random
from typing import List, Optional, Set
from collections import Counter

import selfies as sf
from rdkit import Chem

from .deduplicator import HybridDeduplicator


class SELFIESMutationGenerator:
    """
    Generate molecules by mutating SELFIES strings.
    
    Strategies:
    - Point mutation: Replace single token
    - Insertion: Add new token
    - Deletion: Remove token
    - Crossover: Combine two molecules
    """
    
    # Common CRAM-relevant SELFIES tokens
    CRAM_TOKENS = [
        '[C]', '[=C]', '[C@@H1]', '[C@H1]',
        '[O]', '[=O]', '[OH1]', '[O-1]',
        '[N]', '[NH1]', '[NH2]',
        '[Ring1]', '[Ring2]', '[Branch1]', '[Branch2]',
        '[#C]', '[S]', '[F]', '[Cl]', '[Br]',
    ]
    
    def __init__(
        self,
        seed_smiles: List[str],
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.2
    ):
        """
        Initialize mutation generator.
        
        Args:
            seed_smiles: List of seed SMILES to mutate from
            mutation_rate: Probability of mutation per token
            crossover_rate: Probability of crossover vs mutation
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.deduplicator = HybridDeduplicator()
        
        # Convert seeds to SELFIES
        self.seed_selfies = []
        self.vocab_tokens = set(self.CRAM_TOKENS)
        
        for smiles in seed_smiles:
            try:
                selfies = sf.encoder(smiles)
                if selfies:
                    self.seed_selfies.append(selfies)
                    tokens = list(sf.split_selfies(selfies))
                    self.vocab_tokens.update(tokens)
            except:
                pass
        
        self.vocab_list = list(self.vocab_tokens)
    
    def mutate(self, selfies: str) -> str:
        """Apply random mutation to SELFIES string."""
        tokens = list(sf.split_selfies(selfies))
        if not tokens:
            return selfies
        
        mutation_type = random.choice(['replace', 'insert', 'delete'])
        
        if mutation_type == 'replace' and tokens:
            idx = random.randint(0, len(tokens) - 1)
            tokens[idx] = random.choice(self.vocab_list)
            
        elif mutation_type == 'insert':
            idx = random.randint(0, len(tokens))
            tokens.insert(idx, random.choice(self.vocab_list))
            
        elif mutation_type == 'delete' and len(tokens) > 5:
            idx = random.randint(0, len(tokens) - 1)
            tokens.pop(idx)
        
        return ''.join(tokens)
    
    def crossover(self, selfies1: str, selfies2: str) -> str:
        """Crossover two SELFIES strings."""
        tokens1 = list(sf.split_selfies(selfies1))
        tokens2 = list(sf.split_selfies(selfies2))
        
        if not tokens1 or not tokens2:
            return selfies1
        
        cut1 = random.randint(1, max(1, len(tokens1) - 1))
        cut2 = random.randint(1, max(1, len(tokens2) - 1))
        
        child = tokens1[:cut1] + tokens2[cut2:]
        return ''.join(child)
    
    def generate_one(self) -> Optional[str]:
        """Generate one molecule."""
        parent = random.choice(self.seed_selfies)
        
        if random.random() < self.crossover_rate and len(self.seed_selfies) > 1:
            parent2 = random.choice(self.seed_selfies)
            child_selfies = self.crossover(parent, parent2)
        else:
            child_selfies = self.mutate(parent)
        
        try:
            smiles = sf.decoder(child_selfies)
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    canonical = Chem.MolToSmiles(mol, canonical=True)
                    is_unique, _ = self.deduplicator.check_and_add(canonical)
                    if is_unique:
                        return canonical
        except:
            pass
        
        return None
    
    def generate(self, target_count: int, max_attempts: int = None) -> List[str]:
        """
        Generate molecules until target count.
        
        Args:
            target_count: Number of unique molecules to generate
            max_attempts: Maximum generation attempts
            
        Returns:
            List of unique SMILES
        """
        if max_attempts is None:
            max_attempts = target_count * 100
        
        results = []
        attempts = 0
        
        while len(results) < target_count and attempts < max_attempts:
            smiles = self.generate_one()
            if smiles:
                results.append(smiles)
            attempts += 1
        
        return results
