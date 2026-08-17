"""Notebook MolecularAugmenter, CRAMSpecificAugmenter, AugmentedCRAMDataset."""
from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import random
import numpy as np
from rdkit import Chem

"""
=============================================================================
CRITICAL IMPROVEMENT 4: DATA AUGMENTATION FOR MOLECULAR ROBUSTNESS
=============================================================================

Data augmentation is ESSENTIAL for molecular generative models because:

1. SMILES is not canonical by default - many valid strings represent same molecule
2. Learning should be invariant to SMILES representation
3. Augmentation effectively increases dataset size
4. Contrastive learning with augmented views improves representations

Augmentation strategies:
- SMILES randomization (different atom orderings)
- Substructure masking (partial molecule views)
- Property noise injection
- Canonicalization variants
"""

from typing import List, Optional, Tuple
import random

class MolecularAugmenter:
    """
    Augmentation strategies for molecular data.
    
    Designed specifically for CRAM molecules with special handling for:
    - Preserving carboxyl groups (key CRAM feature)
    - Maintaining ring integrity
    - Generating valid chemical structures
    """
    
    def __init__(self, 
                 n_random_smiles: int = 5,
                 mask_prob: float = 0.15,
                 property_noise_std: float = 0.05):
        self.n_random_smiles = n_random_smiles
        self.mask_prob = mask_prob
        self.property_noise_std = property_noise_std
    
    def randomize_smiles(self, mol: Chem.Mol, n_augments: int = None) -> List[str]:
        """
        Generate random SMILES representations.
        
        Same molecule, different string representations.
        This teaches the encoder that different orderings = same molecule.
        """
        if n_augments is None:
            n_augments = self.n_random_smiles
        
        augmented = []
        atoms = list(range(mol.GetNumAtoms()))
        
        for _ in range(n_augments):
            try:
                # Randomize atom order
                random.shuffle(atoms)
                # Get SMILES with random root atom
                root_atom = random.choice(atoms)
                smiles = Chem.MolToSmiles(mol, rootedAtAtom=root_atom, 
                                          canonical=False, doRandom=True)
                augmented.append(smiles)
            except:
                # Fallback to canonical
                augmented.append(Chem.MolToSmiles(mol))
        
        return augmented
    
    def get_augmented_smiles_pair(self, mol: Chem.Mol) -> Tuple[str, str]:
        """
        Get two different SMILES for the same molecule.
        
        Useful for contrastive learning: (view1, view2) should map to same latent.
        """
        smiles_list = self.randomize_smiles(mol, n_augments=2)
        if len(smiles_list) >= 2:
            return smiles_list[0], smiles_list[1]
        else:
            canonical = Chem.MolToSmiles(mol)
            return canonical, canonical
    
    def mask_substructure(self, mol: Chem.Mol, 
                          protect_cooh: bool = True) -> Tuple[Chem.Mol, List[int]]:
        """
        Mask random atoms in the molecule (for denoising/masked modeling).
        
        Args:
            mol: Input molecule
            protect_cooh: If True, never mask COOH groups (preserve CRAM identity)
        
        Returns:
            masked_mol: Molecule with some atoms "masked" (replaced with *)
            masked_indices: List of masked atom indices
        """
        # Get atoms that can be masked
        maskable_atoms = []
        
        # Identify COOH atoms to protect
        protected_atoms = set()
        if protect_cooh:
            cooh_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
            if cooh_pattern:
                matches = mol.GetSubstructMatches(cooh_pattern)
                for match in matches:
                    protected_atoms.update(match)
        
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            # Don't mask protected atoms, ring junction atoms, or atoms with <2 neighbors
            if (idx not in protected_atoms and 
                atom.GetDegree() >= 2 and
                mol.GetRingInfo().NumAtomRings(idx) <= 1):
                maskable_atoms.append(idx)
        
        if not maskable_atoms:
            return mol, []
        
        # Randomly select atoms to mask
        n_to_mask = max(1, int(len(maskable_atoms) * self.mask_prob))
        masked_indices = random.sample(maskable_atoms, min(n_to_mask, len(maskable_atoms)))
        
        # Create masked molecule (conceptually - actual implementation would
        # modify the molecule structure or use a mask token in features)
        # For now, return original mol with mask indices
        return mol, masked_indices
    
    def perturb_properties(self, properties: Dict[str, float]) -> Dict[str, float]:
        """
        Add small noise to continuous properties for augmentation.
        
        This helps the model learn smooth property mappings rather than
        overfitting to exact property values.
        """
        perturbed = {}
        for key, value in properties.items():
            if isinstance(value, (int, float)):
                noise = np.random.normal(0, self.property_noise_std * abs(value))
                perturbed[key] = value + noise
            else:
                perturbed[key] = value
        return perturbed


class CRAMSpecificAugmenter(MolecularAugmenter):
    """
    CRAM-specific augmentation strategies.
    
    Special considerations:
    1. Never break COOH groups (essential for CRAM)
    2. Preserve ring system topology
    3. Maintain aliphatic/alicyclic character
    4. Generate chemically valid augmentations
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # SMARTS for structures that should never be modified
        self.protected_patterns = [
            '[CX3](=O)[OX2H1,OX1-]',  # Carboxyl
            '[C]1[C][C][C][C][C]1',   # Cyclohexane ring
            '[C]1[C][C][C][C]1',      # Cyclopentane ring
        ]
        self.compiled_protected = []
        for smarts in self.protected_patterns:
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    self.compiled_protected.append(pattern)
            except:
                pass
    
    def get_protected_atoms(self, mol: Chem.Mol) -> set:
        """Get all atoms that should not be modified during augmentation."""
        protected = set()
        for pattern in self.compiled_protected:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                protected.update(match)
        return protected
    
    def fragment_and_recombine(self, mol: Chem.Mol) -> Optional[str]:
        """
        Fragment molecule at safe points and recombine.
        
        This generates related but different molecules for hard negative mining.
        Only fragments at positions that won't destroy CRAM characteristics.
        """
        protected = self.get_protected_atoms(mol)
        
        # Find safe fragmentation points (single bonds not in rings, 
        # not connected to protected atoms)
        safe_bonds = []
        for bond in mol.GetBonds():
            if (bond.GetBondType() == Chem.BondType.SINGLE and
                not bond.IsInRing()):
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                if begin_idx not in protected and end_idx not in protected:
                    safe_bonds.append(bond.GetIdx())
        
        if not safe_bonds:
            return None
        
        # This is a placeholder - actual implementation would:
        # 1. Fragment at random safe bond
        # 2. Recombine fragments in different arrangement
        # 3. Validate resulting molecule
        return None
    
    def enumerate_tautomers(self, mol: Chem.Mol, max_tautomers: int = 3) -> List[str]:
        """
        Generate tautomers as augmented views.
        
        For CRAM with COOH groups, this includes:
        - Carboxylic acid ↔ carboxylate interconversion
        - Keto-enol tautomerism if present
        """
        tautomers = [Chem.MolToSmiles(mol)]  # Original
        
        # For COOH, we can show both protonated and deprotonated forms
        # This is chemically relevant in marine environments (pH variations)
        
        # Convert COOH to COO-
        smiles = Chem.MolToSmiles(mol)
        if 'C(=O)O' in smiles or 'C(O)=O' in smiles:
            # Replace COOH with COO- (simplified)
            deprotonated = smiles.replace('C(=O)O', 'C(=O)[O-]')
            try:
                mol_deprot = Chem.MolFromSmiles(deprotonated)
                if mol_deprot:
                    tautomers.append(Chem.MolToSmiles(mol_deprot))
            except:
                pass
        
        return tautomers[:max_tautomers]


class AugmentedCRAMDataset:
    """
    Dataset wrapper that provides augmented molecular data.
    
    Each call to __getitem__ can return a different augmented view
    of the same molecule, improving training robustness.
    """
    
    def __init__(self, molecules: List[Chem.Mol], 
                 labels: np.ndarray,
                 properties: Optional[List[Dict]] = None,
                 augment_prob: float = 0.5):
        self.molecules = molecules
        self.labels = labels
        self.properties = properties
        self.augment_prob = augment_prob
        self.augmenter = CRAMSpecificAugmenter()
    
    def __len__(self):
        return len(self.molecules)
    
    def get_augmented_sample(self, idx: int) -> Dict:
        """Get a potentially augmented sample."""
        mol = self.molecules[idx]
        label = self.labels[idx]
        props = self.properties[idx] if self.properties else {}
        
        # Decide whether to augment
        if random.random() < self.augment_prob:
            # Get random SMILES
            smiles_list = self.augmenter.randomize_smiles(mol, n_augments=1)
            smiles = smiles_list[0] if smiles_list else Chem.MolToSmiles(mol)
            
            # Maybe perturb properties
            if random.random() < 0.3:
                props = self.augmenter.perturb_properties(props)
        else:
            smiles = Chem.MolToSmiles(mol)
        
        return {
            'smiles': smiles,
            'mol': mol,
            'is_cram': label,
            'properties': props,
            'augmented': random.random() < self.augment_prob
        }
    
    def get_contrastive_pair(self, idx: int) -> Tuple[Dict, Dict]:
        """Get two augmented views for contrastive learning."""
        mol = self.molecules[idx]
        smiles1, smiles2 = self.augmenter.get_augmented_smiles_pair(mol)
        
        base_data = {
            'mol': mol,
            'is_cram': self.labels[idx],
            'properties': self.properties[idx] if self.properties else {}
        }
        
        view1 = {**base_data, 'smiles': smiles1}
        view2 = {**base_data, 'smiles': smiles2}
        
        return view1, view2
