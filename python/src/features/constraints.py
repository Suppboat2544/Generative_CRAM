"""Notebook ConstraintSystem."""
from __future__ import annotations
from typing import Optional, Dict, Any, List
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

class ConstraintSystem:
    """
    Property-based constraint system for CRAM molecule generation.

    Supports two modes:
    - SYNTHETIC: Polymer chemistry (trans, C6-C8, low oxidation, MW 150-450)
    - MARINE_DOM: Geochemistry (degraded, oxidized, C3-C6, MW 120-280)
    """

    # Define default constraints for each mode
    MODES = {
        'synthetic': {
            'mw_range': (150, 450),
            'carboxyl_range': (1, 3),
            'aliphatic_min': 0.85,
            'ring_count_max': 2,
            'xlogp_range': (-2, 5),
            'hbd_max': 5,
            'hba_max': 8,
            'chain_length_range': (3, 10), # C6-C8 optimal
            'oxidation_state': 'low', # Low oxidation
            'stereochem_pref': 'trans', # Trans preferred
            'description': 'Synthetic polymer chemistry mode'
        },
        'marine_dom': {
            'mw_range': (120, 280),
            'carboxyl_range': (2, 4),
            'aliphatic_min': 0.70,
            'ring_count_max': 1,
            'xlogp_range': (-4, 2),
            'hbd_max': 8,
            'hba_max': 12,
            'chain_length_range': (3, 10), # C3-C6 biodegraded
            'oxidation_state': 'high', # High oxidation
            'stereochem_pref': None, # No preference
            'description': 'Marine dissolved organic matter mode'
        },
        'balanced': {
            'mw_range': (150, 7000),
            'carboxyl_range': (1, 4),
            'aliphatic_min': 0.75,
            'ring_count_max': 3,
            'xlogp_range': (-3, 4),
            'hbd_max': 6,
            'hba_max': 10,
            'chain_length_range': (4, 10),
            'oxidation_state': 'medium',
            'stereochem_pref': None,
            'description': 'Balanced mode for general CRAM generation'
        }
    }

    def __init__(self, mode: str = 'synthetic', custom_constraints: Optional[Dict] = None):
        """
        Initialize constraint system.

        Args:
            mode: 'synthetic', 'marine_dom', or 'balanced'
            custom_constraints: Optional custom constraints to override defaults
        """
        if mode not in self.MODES:
            raise ValueError(f"Mode must be one of {list(self.MODES.keys())}")

        self.mode = mode
        self.constraints = self.MODES[mode].copy()

        # Override with custom constraints
        if custom_constraints:
            self.constraints.update(custom_constraints)

        print(f"\\n{'='*80}")
        print(f" CONSTRAINT SYSTEM INITIALIZED: {mode.upper()} MODE")
        print('='*80)
        print(f"Description: {self.constraints['description']}")
        print(f"\\nActive Constraints:")
        for key, value in self.constraints.items():
            if key != 'description':
                print(f" • {key}: {value}")
        print('='*80)

    def validate_molecule(self, smiles: str) -> Dict:
        """
        Validate a molecule against all constraints.

        Returns:
            Dictionary with validation results and scores
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'valid': False, 'score': 0.0, 'violations': ['Invalid SMILES']}

        violations = []
        scores = {}

        # 1. Molecular weight
        mw = Descriptors.MolWt(mol)
        mw_min, mw_max = self.constraints['mw_range']
        if mw_min <= mw <= mw_max:
            scores['mw'] = 1.0
        else:
            scores['mw'] = max(0, 1 - abs(mw - (mw_min + mw_max)/2) / ((mw_max - mw_min)/2))
            if scores['mw'] < 0.5:
                violations.append(f"MW {mw:.1f} outside range [{mw_min}, {mw_max}]")

        # 2. Carboxyl count
        carboxyl_count = self._count_carboxyl_groups(mol)
        carb_min, carb_max = self.constraints['carboxyl_range']
        if carb_min <= carboxyl_count <= carb_max:
            scores['carboxyl'] = 1.0
        else:
            scores['carboxyl'] = max(0, 1 - abs(carboxyl_count - (carb_min + carb_max)/2) / 2)
            if scores['carboxyl'] < 0.5:
                violations.append(f"Carboxyl count {carboxyl_count} outside range [{carb_min}, {carb_max}]")

        # 3. Aliphatic ratio
        aliphatic_ratio = self._calculate_aliphatic_ratio(mol)
        if aliphatic_ratio >= self.constraints['aliphatic_min']:
            scores['aliphatic'] = 1.0
        else:
            scores['aliphatic'] = aliphatic_ratio / self.constraints['aliphatic_min']
            if scores['aliphatic'] < 0.7:
                violations.append(f"Aliphatic ratio {aliphatic_ratio:.2f} < {self.constraints['aliphatic_min']}")

        # 4. Ring count
        ring_count = len(Chem.GetSSSR(mol))
        if ring_count <= self.constraints['ring_count_max']:
            scores['rings'] = 1.0
        else:
            scores['rings'] = max(0, 1 - (ring_count - self.constraints['ring_count_max']) / 3)
            if scores['rings'] < 0.5:
                violations.append(f"Ring count {ring_count} > {self.constraints['ring_count_max']}")

        # 5. XLogP
        try:
            xlogp = Descriptors.MolLogP(mol)
            xlogp_min, xlogp_max = self.constraints['xlogp_range']
            if xlogp_min <= xlogp <= xlogp_max:
                scores['xlogp'] = 1.0
            else:
                scores['xlogp'] = max(0, 1 - abs(xlogp - (xlogp_min + xlogp_max)/2) / ((xlogp_max - xlogp_min)/2))
        except:
            scores['xlogp'] = 0.5

        # 6. H-bond donors
        hbd = Descriptors.NumHDonors(mol)
        if hbd <= self.constraints['hbd_max']:
            scores['hbd'] = 1.0
        else:
            scores['hbd'] = max(0, 1 - (hbd - self.constraints['hbd_max']) / 3)

        # 7. H-bond acceptors
        hba = Descriptors.NumHAcceptors(mol)
        if hba <= self.constraints['hba_max']:
            scores['hba'] = 1.0
        else:
            scores['hba'] = max(0, 1 - (hba - self.constraints['hba_max']) / 3)

        # Calculate overall score (weighted average)
        weights = {
            'mw': 0.20,
            'carboxyl': 0.25,
            'aliphatic': 0.20,
            'rings': 0.15,
            'xlogp': 0.10,
            'hbd': 0.05,
            'hba': 0.05
        }

        overall_score = sum(scores[k] * weights[k] for k in scores)

        return {
            'valid': len(violations) == 0 or overall_score >= 0.65,
            'score': overall_score,
            'component_scores': scores,
            'violations': violations,
            'properties': {
                'mw': mw,
                'carboxyl_count': carboxyl_count,
                'aliphatic_ratio': aliphatic_ratio,
                'ring_count': ring_count,
                'xlogp': scores.get('xlogp', 0),
                'hbd': hbd,
                'hba': hba
            }
        }

    def _count_carboxyl_groups(self, mol) -> int:
        """Count carboxyl groups in molecule."""
        # SMARTS pattern for carboxyl group: C(=O)O or C(=O)[OH]
        carboxyl_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
        matches = mol.GetSubstructMatches(carboxyl_pattern)
        return len(matches)

    def _calculate_aliphatic_ratio(self, mol) -> float:
        """Calculate ratio of non-aromatic atoms."""
        total_atoms = mol.GetNumAtoms()
        if total_atoms == 0:
            return 0.0

        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        aliphatic_atoms = total_atoms - aromatic_atoms

        return aliphatic_atoms / total_atoms

    def batch_validate(self, smiles_list: List[str], min_score: float = 0.65) -> pd.DataFrame:
        """
        Validate a batch of molecules.

        Args:
            smiles_list: List of SMILES strings
            min_score: Minimum score threshold

        Returns:
            DataFrame with validation results
        """
        results = []

        for smiles in tqdm(smiles_list, desc="Validating molecules"):
            validation = self.validate_molecule(smiles)
            validation['smiles'] = smiles
            results.append(validation)

        # Create DataFrame
        df_results = pd.DataFrame(results)

        # Filter by minimum score
        df_pass = df_results[df_results['score'] >= min_score]

        print(f" Total molecules: {len(smiles_list)}")
        print(f" Passed (score >= {min_score}): {len(df_pass)} ({len(df_pass)/len(smiles_list)*100:.1f}%)")
        print(f" Average score: {df_results['score'].mean():.3f}")

        return df_results

