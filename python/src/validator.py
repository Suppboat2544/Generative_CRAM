"""
CRAM Validation Module
======================
Strict validation for Carboxyl-Rich Alicyclic Molecules (CRAM).

Validation criteria:
- Minimum 2 carboxyl groups (COOH/COO-)
- At least one alicyclic ring (non-aromatic)
- >60% aliphatic carbon content
- Molecular weight 100-750 Da
"""

from typing import Dict, Optional, Tuple, List
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np

from config.settings import CRAM_SMARTS_PATTERNS, CRAM_VALIDATION


class CRAMValidator:
    """Validates molecules for CRAM characteristics."""
    
    def __init__(
        self,
        min_cooh: int = None,
        min_aliphatic_pct: float = None,
        require_alicyclic: bool = None,
        mw_range: Tuple[float, float] = None
    ):
        """
        Initialize CRAM validator.
        
        Args:
            min_cooh: Minimum carboxyl groups (default: 2)
            min_aliphatic_pct: Minimum aliphatic carbon % (default: 0.6)
            require_alicyclic: Require alicyclic ring (default: True)
            mw_range: Molecular weight range (default: 100-750)
        """
        self.min_cooh = min_cooh or CRAM_VALIDATION["min_cooh"]
        self.min_aliphatic_pct = min_aliphatic_pct or CRAM_VALIDATION["min_aliphatic_pct"]
        self.require_alicyclic = require_alicyclic if require_alicyclic is not None else CRAM_VALIDATION["require_alicyclic"]
        self.mw_range = mw_range or CRAM_VALIDATION["mw_range"]
        
        # Compile SMARTS patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile SMARTS patterns for efficient matching."""
        self.patterns = {}
        
        # COOH patterns (multiple for strict validation)
        self.cooh_patterns = []
        for key, smarts in CRAM_SMARTS_PATTERNS.items():
            if key.startswith("cooh"):
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    self.cooh_patterns.append(pat)
        
        # Alicyclic patterns
        self.alicyclic_patterns = []
        for size in [5, 6, 7]:
            key = f"alicyclic_{size}"
            if key in CRAM_SMARTS_PATTERNS:
                pat = Chem.MolFromSmarts(CRAM_SMARTS_PATTERNS[key])
                if pat:
                    self.alicyclic_patterns.append(pat)
    
    def count_cooh(self, mol: Chem.Mol) -> int:
        """
        Count carboxyl groups using multiple SMARTS patterns.
        
        Uses strict matching with multiple patterns to ensure
        accurate COOH count for CRAM validation.
        """
        if mol is None:
            return 0
        
        # Use set of matched carbons to avoid double counting
        cooh_carbons = set()
        
        for pattern in self.cooh_patterns:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                # First atom in COOH pattern is the carboxyl carbon
                cooh_carbons.add(match[0])
        
        return len(cooh_carbons)
    
    def has_alicyclic_ring(self, mol: Chem.Mol) -> bool:
        """Check if molecule has at least one alicyclic (non-aromatic) ring."""
        if mol is None:
            return False
        
        # Method 1: Use ring info
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if not any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                return True
        
        # Method 2: SMARTS pattern matching
        for pattern in self.alicyclic_patterns:
            if mol.HasSubstructMatch(pattern):
                return True
        
        return False
    
    def get_aliphatic_percentage(self, mol: Chem.Mol) -> float:
        """Calculate percentage of aliphatic (non-aromatic) carbons."""
        if mol is None:
            return 0.0
        
        c_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        if c_count == 0:
            return 0.0
        
        aromatic_c = sum(1 for a in mol.GetAtoms() 
                        if a.GetAtomicNum() == 6 and a.GetIsAromatic())
        
        return (c_count - aromatic_c) / c_count
    
    def get_molecular_weight(self, mol: Chem.Mol) -> float:
        """Get molecular weight."""
        if mol is None:
            return 0.0
        return Descriptors.MolWt(mol)
    
    def validate(self, mol: Chem.Mol) -> Dict:
        """
        Validate molecule for CRAM characteristics.
        
        Returns:
            Dictionary with validation results:
            - is_cram: bool - Overall CRAM validity
            - n_cooh: int - Number of carboxyl groups
            - has_alicyclic: bool - Has alicyclic ring
            - aliphatic_pct: float - Aliphatic carbon percentage
            - mw: float - Molecular weight
            - mw_valid: bool - Within MW range
            - failures: List[str] - Reasons for failure
        """
        result = {
            "is_cram": False,
            "n_cooh": 0,
            "has_alicyclic": False,
            "aliphatic_pct": 0.0,
            "mw": 0.0,
            "mw_valid": False,
            "failures": []
        }
        
        if mol is None:
            result["failures"].append("Invalid molecule")
            return result
        
        # Count COOH groups
        result["n_cooh"] = self.count_cooh(mol)
        if result["n_cooh"] < self.min_cooh:
            result["failures"].append(f"COOH count ({result['n_cooh']}) < {self.min_cooh}")
        
        # Check alicyclic ring
        result["has_alicyclic"] = self.has_alicyclic_ring(mol)
        if self.require_alicyclic and not result["has_alicyclic"]:
            result["failures"].append("No alicyclic ring")
        
        # Calculate aliphatic percentage
        result["aliphatic_pct"] = self.get_aliphatic_percentage(mol)
        if result["aliphatic_pct"] < self.min_aliphatic_pct:
            result["failures"].append(
                f"Aliphatic % ({result['aliphatic_pct']:.1%}) < {self.min_aliphatic_pct:.0%}"
            )
        
        # Check molecular weight
        result["mw"] = self.get_molecular_weight(mol)
        result["mw_valid"] = self.mw_range[0] <= result["mw"] <= self.mw_range[1]
        if not result["mw_valid"]:
            result["failures"].append(
                f"MW ({result['mw']:.1f}) outside range {self.mw_range}"
            )
        
        # Overall CRAM status
        result["is_cram"] = len(result["failures"]) == 0
        
        return result
    
    def validate_smiles(self, smiles: str) -> Dict:
        """Validate SMILES string for CRAM characteristics."""
        mol = Chem.MolFromSmiles(smiles)
        return self.validate(mol)
    
    def is_cram(self, mol_or_smiles) -> bool:
        """Quick check if molecule/SMILES is a valid CRAM."""
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
        else:
            mol = mol_or_smiles
        
        return self.validate(mol)["is_cram"]


# Global validator instance
default_validator = CRAMValidator()


def is_true_cram(mol, min_cooh=2, min_aliphatic_pct=0.6):
    """
    Legacy function for backwards compatibility.
    Check if molecule is a TRUE CRAM molecule.
    """
    validator = CRAMValidator(
        min_cooh=min_cooh,
        min_aliphatic_pct=min_aliphatic_pct
    )
    return validator.validate(mol)


def is_true_cram_smiles(smiles, min_cooh=2, min_aliphatic_pct=0.6):
    """
    Legacy function for backwards compatibility.
    Check if SMILES represents a TRUE CRAM molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    return is_true_cram(mol, min_cooh, min_aliphatic_pct)


class CRAMBatchValidator:
    """Efficient batch validation of molecules."""
    
    def __init__(self, validator: CRAMValidator = None):
        self.validator = validator or default_validator
    
    def validate_batch(self, molecules: List, return_valid_only: bool = False) -> List[Dict]:
        """
        Validate a batch of molecules.
        
        Args:
            molecules: List of SMILES strings or RDKit Mol objects
            return_valid_only: If True, only return valid CRAM molecules
            
        Returns:
            List of validation results
        """
        results = []
        
        for mol in molecules:
            if isinstance(mol, str):
                result = self.validator.validate_smiles(mol)
                result["smiles"] = mol
            else:
                result = self.validator.validate(mol)
                if mol is not None:
                    result["smiles"] = Chem.MolToSmiles(mol)
            
            if return_valid_only:
                if result["is_cram"]:
                    results.append(result)
            else:
                results.append(result)
        
        return results
    
    def filter_cram(self, molecules: List) -> List[str]:
        """Return only valid CRAM molecules."""
        valid = []
        for mol in molecules:
            if isinstance(mol, str):
                if self.validator.is_cram(mol):
                    valid.append(mol)
            else:
                if self.validator.is_cram(mol):
                    valid.append(Chem.MolToSmiles(mol))
        return valid


if __name__ == "__main__":
    # Test validation
    test_smiles = [
        "CC1CCC(CC1)(C(=O)O)C(=O)O",  # Cyclohexane with 2 COOH
        "c1ccccc1",                     # Benzene (no CRAM)
        "CC(=O)O",                      # Acetic acid (no alicyclic)
    ]
    
    validator = CRAMValidator()
    
    print("=" * 60)
    print("CRAM Validation Test")
    print("=" * 60)
    
    for smi in test_smiles:
        result = validator.validate_smiles(smi)
        print(f"\nSMILES: {smi}")
        print(f"  Is CRAM: {result['is_cram']}")
        print(f"  COOH count: {result['n_cooh']}")
        print(f"  Has alicyclic: {result['has_alicyclic']}")
        print(f"  Aliphatic %: {result['aliphatic_pct']:.1%}")
        print(f"  MW: {result['mw']:.1f}")
        if result["failures"]:
            print(f"  Failures: {', '.join(result['failures'])}")
