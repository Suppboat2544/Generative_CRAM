"""Notebook SyntheticAccessibilityFilter."""
from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
from rdkit import Chem

# ============================================================================
# PRIORITY 2: Synthetic Accessibility Scoring
# ============================================================================

class SyntheticAccessibilityFilter:
    """
    Filter generated molecules by synthetic feasibility.
    Integrates multiple complexity metrics for CRAM-specific assessment.
    """

    def __init__(self):
        # SA Score thresholds (1=easy to synthesize, 10=very difficult)
        self.sa_thresholds = {
            'excellent': 3.0,
            'good': 4.5,
            'moderate': 6.0,
            'difficult': 8.0
        }

    def calculate_bertz_complexity(self, mol):
        """
        Calculate Bertz complexity index.
        Measures molecular complexity based on symmetry and connectivity.
        """
        try:
            from rdkit.Chem import GraphDescriptors
            return GraphDescriptors.BertzCT(mol)
        except:
            return 0

    def calculate_sa_score(self, mol):
        """
        Calculate Synthetic Accessibility Score (Ertl & Schuffenhauer).
        Range: 1 (easy) to 10 (difficult)
        """
        try:
            from rdkit.Chem import Descriptors

            # Import SA scorer if available
            try:
                import sascorer
                return sascorer.calculateScore(mol)
            except ImportError:
                # Fallback: simple heuristic based on complexity
                complexity = self.calculate_bertz_complexity(mol)
                num_rings = len(Chem.GetSSSR(mol))
                num_stereo = sum(1 for atom in mol.GetAtoms()
                                if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED)

                # Simple estimate
                sa_estimate = min(10, 1 + complexity/500 + num_rings*0.5 + num_stereo*0.3)
                return sa_estimate
        except:
            return 5.0 # Default moderate difficulty

    def count_chiral_centers(self, mol):
        """
        Count number of chiral centers (stereogenic atoms).
        More chiral centers → harder to synthesize with correct stereochemistry.
        """
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral_centers = 0
        for atom in mol.GetAtoms():
            if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                chiral_centers += 1
        return chiral_centers

    def count_rotatable_bonds(self, mol):
        """
        Count rotatable bonds (conformational flexibility).
        More rotatable bonds → more conformers → harder analysis.
        """
        return Descriptors.NumRotatableBonds(mol)

    def has_reactive_groups(self, mol):
        """
        Check for reactive/unstable functional groups that complicate synthesis.
        """
        reactive_patterns = {
            'peroxide': '[OX2][OX2]',
            'diazo': '[#6]=[N+]=[N-]',
            'azide': '[NX1]=[NX2]=[NX1-]',
            'nitro': '[N+](=O)[O-]',
            'acid_chloride': '[CX3](=[OX1])[ClX1]',
            'anhydride': '[CX3](=[OX1])[OX2][CX3](=[OX1])',
            'isocyanate': '[NX2]=[CX2]=[OX1]'
        }

        reactive = []
        for name, smarts in reactive_patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                reactive.append(name)

        return reactive

    def assess_ring_strain(self, mol):
        """
        Assess ring strain (small rings = high strain = difficult).
        """
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(ring) for ring in ring_info.AtomRings()]

        strain_score = 0
        for size in ring_sizes:
            if size == 3:
                strain_score += 3 # High strain (cyclopropane)
            elif size == 4:
                strain_score += 2 # Moderate strain (cyclobutane)
            elif size in [5, 6, 7]:
                strain_score += 0 # Low strain (favorable)
            else:
                strain_score += 1 # Unusual sizes

        return strain_score

    def evaluate_synthesizability(self, mol):
        """
        Comprehensive synthesizability evaluation.

        Returns:
            dict with scores and classification
        """
        metrics = {}

        # 1. SA Score (primary metric)
        metrics['sa_score'] = self.calculate_sa_score(mol)

        # 2. Complexity
        metrics['bertz_complexity'] = self.calculate_bertz_complexity(mol)

        # 3. Stereochemistry burden
        metrics['chiral_centers'] = self.count_chiral_centers(mol)

        # 4. Conformational complexity
        metrics['rotatable_bonds'] = self.count_rotatable_bonds(mol)

        # 5. Ring strain
        metrics['ring_strain'] = self.assess_ring_strain(mol)

        # 6. Reactive groups
        metrics['reactive_groups'] = self.has_reactive_groups(mol)
        metrics['has_reactive'] = len(metrics['reactive_groups']) > 0

        # 7. Molecular weight (very large = harder)
        mw = Descriptors.MolWt(mol)
        metrics['mw'] = mw
        metrics['mw_penalty'] = max(0, (mw - 500) / 200) # Penalize MW > 500

        # Classification
        if metrics['sa_score'] <= self.sa_thresholds['excellent']:
            metrics['classification'] = 'excellent'
        elif metrics['sa_score'] <= self.sa_thresholds['good']:
            metrics['classification'] = 'good'
        elif metrics['sa_score'] <= self.sa_thresholds['moderate']:
            metrics['classification'] = 'moderate'
        elif metrics['sa_score'] <= self.sa_thresholds['difficult']:
            metrics['classification'] = 'difficult'
        else:
            metrics['classification'] = 'very_difficult'

        # Overall score (0-1, higher = easier to synthesize)
        overall = 1.0
        overall -= (metrics['sa_score'] / 10) * 0.4 # 40% weight
        overall -= min(1.0, metrics['chiral_centers'] / 5) * 0.2 # 20% weight
        overall -= min(1.0, metrics['ring_strain'] / 5) * 0.15 # 15% weight
        overall -= min(1.0, metrics['rotatable_bonds'] / 10) * 0.1 # 10% weight
        overall -= metrics['mw_penalty'] * 0.1 # 10% weight
        overall -= (0.05 if metrics['has_reactive'] else 0) # 5% penalty

        metrics['overall_score'] = max(0, overall)

        return metrics

    def filter_by_synthesizability(self, molecules, min_score=0.5, max_sa_score=6.0):
        """
        Filter molecules by synthesizability criteria.

        Args:
            molecules: List of SMILES or RDKit mol objects
            min_score: Minimum overall synthesizability score (0-1)
            max_sa_score: Maximum SA score to accept (1-10, lower is easier)

        Returns:
            List of (mol, metrics) tuples for molecules passing filter
        """
        results = []

        for mol in molecules:
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)

            if mol is None:
                continue

            metrics = self.evaluate_synthesizability(mol)

            # Apply filters
            if (metrics['overall_score'] >= min_score and
                metrics['sa_score'] <= max_sa_score):
                results.append((mol, metrics))

        return results

    def generate_synthesizability_report(self, molecules):
        """
        Generate detailed synthesizability report for a set of molecules.
        """
        print("="*80)
        print(" SYNTHETIC ACCESSIBILITY ANALYSIS")
        print("="*80)

        all_metrics = []
        for mol in molecules:
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
            if mol is not None:
                metrics = self.evaluate_synthesizability(mol)
                all_metrics.append(metrics)

        if not all_metrics:
            print("No valid molecules to analyze!")
            return

        print(f"\nTotal molecules: {len(all_metrics)}")

        # Overall statistics
        print(f"\n OVERALL STATISTICS:")
        print("-"*80)
        sa_scores = [m['sa_score'] for m in all_metrics]
        overall_scores = [m['overall_score'] for m in all_metrics]

        print(f" SA Score (Ertl): {np.mean(sa_scores):.2f} ± {np.std(sa_scores):.2f}")
        print(f" Range: [{np.min(sa_scores):.2f}, {np.max(sa_scores):.2f}]")
        print(f" Overall Score: {np.mean(overall_scores):.3f} ± {np.std(overall_scores):.3f}")

        # Classification distribution
        print(f"\n CLASSIFICATION DISTRIBUTION:")
        print("-"*80)
        classifications = Counter([m['classification'] for m in all_metrics])
        for cls in ['excellent', 'good', 'moderate', 'difficult', 'very_difficult']:
            count = classifications.get(cls, 0)
            pct = count / len(all_metrics) * 100
            print(f" {cls.capitalize():15s}: {count:4d} ({pct:5.1f}%)")

        # Stereochemistry
        print(f"\n STEREOCHEMISTRY BURDEN:")
        print("-"*80)
        chiral_counts = [m['chiral_centers'] for m in all_metrics]
        print(f" Mean chiral centers: {np.mean(chiral_counts):.2f} ± {np.std(chiral_counts):.2f}")
        print(f" Molecules with chirality: {sum(1 for c in chiral_counts if c > 0)} ({sum(1 for c in chiral_counts if c > 0)/len(all_metrics)*100:.1f}%)")

        # Ring strain
        print(f"\n RING STRAIN:")
        print("-"*80)
        ring_strains = [m['ring_strain'] for m in all_metrics]
        print(f" Mean ring strain: {np.mean(ring_strains):.2f} ± {np.std(ring_strains):.2f}")
        high_strain = sum(1 for s in ring_strains if s > 3)
        print(f" High strain molecules: {high_strain} ({high_strain/len(all_metrics)*100:.1f}%)")

        # Reactive groups
        print(f"\n REACTIVE/UNSTABLE GROUPS:")
        print("-"*80)
        reactive_mols = sum(1 for m in all_metrics if m['has_reactive'])
        print(f" Molecules with reactive groups: {reactive_mols} ({reactive_mols/len(all_metrics)*100:.1f}%)")

        if reactive_mols > 0:
            all_reactive = []
            for m in all_metrics:
                all_reactive.extend(m['reactive_groups'])
            reactive_counts = Counter(all_reactive)
            print(f" Most common reactive groups:")
            for group, count in reactive_counts.most_common(5):
                print(f" • {group}: {count}")

        # Recommendations
        print(f"\n RECOMMENDATIONS:")
        print("-"*80)
        avg_sa = np.mean(sa_scores)
        avg_overall = np.mean(overall_scores)

        if avg_sa > 6.0:
            print(" High SA scores - molecules may be difficult to synthesize")
            print(" -> Simplify structures, reduce chiral centers, avoid strained rings")
        elif avg_sa < 3.5:
            print(" Excellent SA scores - molecules are synthetically accessible")
        else:
            print(" Moderate SA scores - reasonable synthetic accessibility")

        if avg_overall < 0.4:
            print(" Low overall scores - multiple synthesis challenges")
            print(" -> Review chiral centers, ring strain, and molecular weight")
        elif avg_overall > 0.7:
            print(" High overall scores - good synthetic prospects")

        print("="*80)

        return all_metrics

