"""
CRAM Generator Utilities
========================

Utility classes for molecule generation:
- SyntheticAccessibilityCalculator: SA score calculation
- BloomFilter: Efficient duplicate detection
- LatentSpaceDiversifier: Latent space exploration
- SMILESAugmenter: SMILES data augmentation
- ScaffoldOperator: Scaffold extraction/analysis
"""

import numpy as np
import random
import hashlib
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter

# Optional mmh3 for faster hashing
try:
    import mmh3
    HAS_MMH3 = True
except ImportError:
    HAS_MMH3 = False

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


# =============================================================================
# SYNTHETIC ACCESSIBILITY SCORE
# =============================================================================

class SyntheticAccessibilityCalculator:
    """
    Calculate Synthetic Accessibility (SA) Score.
    
    Based on Ertl & Schuffenhauer (2009) approach.
    Lower scores = easier to synthesize (1-10 scale).
    Threshold: SA < 4.5 is generally "accessible".
    """
    
    FRAGMENT_PENALTIES = {
        'bridgehead': 1.5,
        'spiro': 1.2,
        'macrocycle': 2.0,
        'strained_3ring': 1.8,
        'strained_4ring': 1.5,
        'multiple_stereo': 0.5,
        'rare_element': 2.0,
        'complex_ring_system': 1.0,
    }
    
    FRAGMENT_PATTERNS = {
        'bridgehead': '[x3,x4](@*)(@*)@*',
        'spiro': '[R2](@*)(@*)(@*)@*',
        'strained_3ring': '[R1]1~[R1]~[R1]1',
        'strained_4ring': '[R1]1~[R1]~[R1]~[R1]1',
        'epoxide': 'C1OC1',
        'aziridine': 'C1NC1',
    }
    
    def __init__(self):
        self.compiled_patterns = {}
        for name, smarts in self.FRAGMENT_PATTERNS.items():
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self.compiled_patterns[name] = pat
            except:
                pass
    
    def calculate(self, mol_or_smiles) -> float:
        """
        Calculate SA score (1-10, lower = easier).
        
        Args:
            mol_or_smiles: RDKit Mol or SMILES string
            
        Returns:
            SA score (1.0-10.0)
        """
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles
            
        if mol is None:
            return 10.0
        
        try:
            score = 1.0
            
            # Ring complexity
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            
            if num_rings > 0:
                for ring in ring_info.AtomRings():
                    size = len(ring)
                    if size == 3:
                        score += 0.8
                    elif size == 4:
                        score += 0.5
                    elif size > 7:
                        score += 0.3 * (size - 7)
                
                if num_rings > 2:
                    score += 0.3 * (num_rings - 2)
            
            # Stereochemistry complexity
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            if len(chiral_centers) > 2:
                score += 0.3 * (len(chiral_centers) - 2)
            
            # Heteroatom ratio
            hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [1, 6])
            carbons = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
            if carbons > 0:
                ratio = hetero / carbons
                if ratio > 0.5:
                    score += 0.5 * (ratio - 0.5)
            
            # Problematic substructures
            for name, pattern in self.compiled_patterns.items():
                matches = mol.GetSubstructMatches(pattern)
                if matches:
                    penalty = self.FRAGMENT_PENALTIES.get(name, 0.5)
                    score += penalty * len(matches)
            
            # Size penalty
            if mol.GetNumAtoms() > 50:
                score += 0.02 * (mol.GetNumAtoms() - 50)
            
            # Rare elements
            rare = {a.GetAtomicNum() for a in mol.GetAtoms() 
                   if a.GetAtomicNum() not in [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]}
            if rare:
                score += 1.0 * len(rare)
            
            return min(max(score, 1.0), 10.0)
            
        except:
            return 10.0
    
    def is_accessible(self, mol_or_smiles, threshold: float = 4.5) -> bool:
        """Check if molecule is synthetically accessible."""
        return self.calculate(mol_or_smiles) <= threshold


# =============================================================================
# BLOOM FILTER FOR DUPLICATE DETECTION
# =============================================================================

class BloomFilter:
    """
    Space-efficient probabilistic duplicate detector.
    
    Uses multiple hash functions to check membership with
    configurable false positive rate. No false negatives.
    """
    
    def __init__(
        self, 
        expected_items: int = 1_000_000,
        false_positive_rate: float = 0.001
    ):
        """
        Initialize Bloom filter.
        
        Args:
            expected_items: Expected number of unique items
            false_positive_rate: Acceptable false positive rate
        """
        # Calculate optimal size and hash count
        self.size = self._optimal_size(expected_items, false_positive_rate)
        self.num_hashes = self._optimal_hashes(self.size, expected_items)
        
        # Bit array (using numpy for efficiency)
        self.bit_array = np.zeros(self.size, dtype=np.bool_)
        self.count = 0
        
        print(f"BloomFilter: {self.size:,} bits, {self.num_hashes} hashes")
    
    def _optimal_size(self, n: int, p: float) -> int:
        """Calculate optimal bit array size."""
        m = -n * np.log(p) / (np.log(2) ** 2)
        return int(m) + 1
    
    def _optimal_hashes(self, m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        k = (m / n) * np.log(2)
        return max(1, int(k) + 1)
    
    def _get_hash_indices(self, item: str) -> List[int]:
        """Get hash indices for an item."""
        indices = []
        
        if HAS_MMH3:
            # Use MurmurHash3 (faster)
            h1 = mmh3.hash(item, 0) % self.size
            h2 = mmh3.hash(item, 1) % self.size
        else:
            # Fallback to hashlib
            h1 = int(hashlib.md5(item.encode()).hexdigest(), 16) % self.size
            h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16) % self.size
        
        for i in range(self.num_hashes):
            idx = (h1 + i * h2) % self.size
            indices.append(idx)
        
        return indices
    
    def add(self, item: str) -> bool:
        """
        Add item to filter.
        
        Args:
            item: String to add
            
        Returns:
            True if item was new, False if probably existed
        """
        indices = self._get_hash_indices(item)
        is_new = not all(self.bit_array[i] for i in indices)
        
        for i in indices:
            self.bit_array[i] = True
        
        if is_new:
            self.count += 1
        
        return is_new
    
    def contains(self, item: str) -> bool:
        """
        Check if item might be in filter.
        
        Args:
            item: String to check
            
        Returns:
            True if probably in filter, False if definitely not
        """
        indices = self._get_hash_indices(item)
        return all(self.bit_array[i] for i in indices)
    
    def __contains__(self, item: str) -> bool:
        return self.contains(item)
    
    def __len__(self) -> int:
        return self.count
    
    def clear(self):
        """Clear the filter."""
        self.bit_array.fill(False)
        self.count = 0


# =============================================================================
# SMILES AUGMENTATION
# =============================================================================

class SMILESAugmenter:
    """
    Generate diverse SMILES representations for data augmentation.
    
    The same molecule can be represented by many valid SMILES strings.
    This augmenter generates random but valid representations.
    """
    
    def __init__(self, max_augmentations: int = 10):
        self.max_augmentations = max_augmentations
    
    def randomize_smiles(self, smiles: str, max_attempts: int = 10) -> str:
        """
        Generate a random SMILES representation.
        
        Args:
            smiles: Input SMILES
            max_attempts: Maximum randomization attempts
            
        Returns:
            Random valid SMILES (or original if fails)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        
        for _ in range(max_attempts):
            try:
                atom_order = list(range(mol.GetNumAtoms()))
                random.shuffle(atom_order)
                random_mol = Chem.RenumberAtoms(mol, atom_order)
                random_smiles = Chem.MolToSmiles(random_mol, canonical=False, doRandom=True)
                
                if Chem.MolFromSmiles(random_smiles) is not None:
                    return random_smiles
            except:
                continue
        
        return smiles
    
    def augment(self, smiles: str, n: int = 10) -> List[str]:
        """
        Generate multiple SMILES representations.
        
        Args:
            smiles: Input SMILES
            n: Number of augmentations
            
        Returns:
            List of unique valid SMILES
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]
        
        augmented = {Chem.MolToSmiles(mol, canonical=True)}
        attempts = 0
        max_attempts = n * 5
        
        while len(augmented) < n and attempts < max_attempts:
            try:
                order = list(range(mol.GetNumAtoms()))
                np.random.shuffle(order)
                new_smiles = Chem.MolToSmiles(mol, rootedAtAtom=order[0], canonical=False)
                if new_smiles:
                    augmented.add(new_smiles)
            except:
                pass
            attempts += 1
        
        return list(augmented)
    
    def augment_dataset(
        self, 
        smiles_list: List[str], 
        factor: int = 5
    ) -> List[str]:
        """
        Augment entire dataset.
        
        Args:
            smiles_list: List of SMILES
            factor: Augmentation factor
            
        Returns:
            Augmented SMILES list
        """
        augmented = []
        for smi in smiles_list:
            variants = self.augment(smi, factor)
            augmented.extend(variants)
        
        unique = list(set(augmented))
        return unique


# =============================================================================
# LATENT SPACE DIVERSIFICATION
# =============================================================================

class LatentSpaceDiversifier:
    """
    Strategies for exploring latent space to generate diverse molecules.
    """
    
    def __init__(self, latent_dim: int = 64):
        self.latent_dim = latent_dim
    
    def interpolate(
        self, 
        z1: np.ndarray, 
        z2: np.ndarray, 
        n_steps: int = 10
    ) -> List[np.ndarray]:
        """Linear interpolation between two latent vectors."""
        alphas = np.linspace(0, 1, n_steps)
        return [(1 - a) * z1 + a * z2 for a in alphas]
    
    def spherical_interpolate(
        self, 
        z1: np.ndarray, 
        z2: np.ndarray, 
        n_steps: int = 10
    ) -> List[np.ndarray]:
        """Spherical interpolation (slerp) between vectors."""
        z1_norm = z1 / (np.linalg.norm(z1) + 1e-8)
        z2_norm = z2 / (np.linalg.norm(z2) + 1e-8)
        
        dot = np.clip(np.dot(z1_norm.flatten(), z2_norm.flatten()), -1.0, 1.0)
        theta = np.arccos(dot)
        
        result = []
        for t in np.linspace(0, 1, n_steps):
            if np.abs(theta) < 1e-6:
                z = (1 - t) * z1 + t * z2
            else:
                z = (np.sin((1-t)*theta)/np.sin(theta)) * z1 + \
                    (np.sin(t*theta)/np.sin(theta)) * z2
            result.append(z)
        
        return result
    
    def perturb_gaussian(
        self, 
        z: np.ndarray, 
        noise_scale: float = 0.3, 
        n_samples: int = 10
    ) -> List[np.ndarray]:
        """Generate samples by adding Gaussian noise."""
        return [z + np.random.randn(*z.shape) * noise_scale for _ in range(n_samples)]
    
    def perturb_uniform_sphere(
        self, 
        z: np.ndarray, 
        radius: float = 0.5, 
        n_samples: int = 10
    ) -> List[np.ndarray]:
        """Generate samples uniformly in a ball around z."""
        samples = []
        for _ in range(n_samples):
            direction = np.random.randn(*z.shape)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            r = radius * np.random.random() ** (1.0 / self.latent_dim)
            samples.append(z + r * direction)
        return samples


# =============================================================================
# SCAFFOLD OPERATIONS
# =============================================================================

class ScaffoldOperator:
    """
    Extract and analyze molecular scaffolds.
    """
    
    CRAM_SIDECHAINS = [
        'C', 'CC', 'CCC', 'C(C)C', 'C(C)(C)C',
        'C(=O)O', 'CC(=O)O', 'CCC(=O)O',
        'CO', 'CCO', 'CCCO',
        'C=O', 'CC=O',
        'C(=O)C', 'C(=O)CC',
        'OC', 'OCC',
    ]
    
    def get_murcko_scaffold(self, mol_or_smiles) -> Optional[str]:
        """Extract Murcko scaffold."""
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            if isinstance(mol_or_smiles, str):
                mol = Chem.MolFromSmiles(mol_or_smiles)
            else:
                mol = mol_or_smiles
            
            if mol is None:
                return None
            
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except:
            return None
    
    def get_generic_scaffold(self, mol_or_smiles) -> Optional[str]:
        """Get generic scaffold (all atoms → C, bonds → single)."""
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            if isinstance(mol_or_smiles, str):
                mol = Chem.MolFromSmiles(mol_or_smiles)
            else:
                mol = mol_or_smiles
            
            if mol is None:
                return None
            
            scaffold = MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol)
            )
            return Chem.MolToSmiles(scaffold)
        except:
            return None
    
    def analyze_scaffolds(self, smiles_list: List[str]) -> Dict[str, List[str]]:
        """Analyze scaffold distribution in dataset."""
        scaffolds = {}
        
        for smi in smiles_list:
            scaffold = self.get_murcko_scaffold(smi)
            if scaffold:
                if scaffold not in scaffolds:
                    scaffolds[scaffold] = []
                scaffolds[scaffold].append(smi)
        
        return scaffolds
    
    def get_scaffold_diversity(self, smiles_list: List[str]) -> float:
        """Calculate scaffold diversity (unique scaffolds / total)."""
        scaffolds = self.analyze_scaffolds(smiles_list)
        return len(scaffolds) / len(smiles_list) if smiles_list else 0.0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_device():
    """
    Get the best available device.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    import torch
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Get canonical SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)
