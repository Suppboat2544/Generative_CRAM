"""Notebook EnhancedDiversityAnalyzer."""
from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
from rdkit import Chem

# ============================================================================
# PRIORITY 1: Enhanced Diversity Metrics
# ============================================================================

class EnhancedDiversityAnalyzer:
    """
    Comprehensive diversity analysis beyond simple uniqueness.
    """

    def __init__(self, molecules):
        """
        Args:
            molecules: List of RDKit mol objects or SMILES strings
        """
        self.molecules = []
        for mol in molecules:
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
            if mol is not None:
                self.molecules.append(mol)

    def calculate_tanimoto_diversity(self, fp_type='morgan', radius=2, n_bits=2048):
        """
        Calculate pairwise Tanimoto diversity using molecular fingerprints.

        Returns average pairwise distance (higher = more diverse).
        """
        fps = []
        for mol in self.molecules:
            if fp_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
            elif fp_type == 'rdkit':
                fp = Chem.RDKFingerprint(mol)
            elif fp_type == 'maccs':
                fp = AllChem.GetMACCSKeysFingerprint(mol)
            fps.append(fp)

        # Calculate pairwise distances
        n = len(fps)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                distances.append(1 - similarity) # Convert to distance

        if not distances:
            return 0.0

        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'diversity_score': np.mean(distances) # 0 = identical, 1 = completely different
        }

    def calculate_scaffold_diversity(self):
        """
        Calculate diversity of Murcko scaffolds.

        Uses Shannon entropy: H = -Σ(p_i * log2(p_i))
        Higher entropy = more diverse scaffolds.
        """
        scaffolds = []
        for mol in self.molecules:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                scaffolds.append(scaffold_smiles)
            except:
                scaffolds.append(None)

        # Count scaffold frequencies
        scaffold_counts = Counter([s for s in scaffolds if s is not None])
        total = len([s for s in scaffolds if s is not None])

        if total == 0:
            return {'entropy': 0, 'unique_scaffolds': 0, 'scaffold_ratio': 0}

        # Calculate Shannon entropy
        entropy = 0
        for count in scaffold_counts.values():
            p = count / total
            entropy -= p * np.log2(p)

        return {
            'entropy': entropy,
            'unique_scaffolds': len(scaffold_counts),
            'scaffold_ratio': len(scaffold_counts) / total,
            'max_entropy': np.log2(total), # Maximum possible entropy
            'normalized_entropy': entropy / np.log2(total) if total > 1 else 0
        }

    def calculate_functional_group_diversity(self):
        """
        Analyze diversity of functional group patterns.
        """
        # Define CRAM-relevant functional groups (SMARTS patterns)
        functional_groups = {
            'carboxylic_acid': '[CX3](=O)[OX2H1]',
            'hydroxyl': '[OX2H]',
            'ketone': '[CX3](=O)[#6]',
            'ester': '[CX3](=O)[OX2H0]',
            'ether': '[OD2]([#6])[#6]',
            'aldehyde': '[CX3H1](=O)[#6]',
            'primary_alcohol': '[CH2X4][OX2H]',
            'secondary_alcohol': '[CH1X4][OX2H]',
            'tertiary_alcohol': '[CX4][OX2H]',
            'cyclic_ether': '[#6]1~[#6]~[OX2]~[#6]~[#6]~1'
        }

        # Count functional groups in each molecule
        fg_profiles = []
        for mol in self.molecules:
            profile = {}
            for fg_name, smarts in functional_groups.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    matches = mol.GetSubstructMatches(pattern)
                    profile[fg_name] = len(matches)
            fg_profiles.append(profile)

        # Convert to matrix for analysis
        fg_matrix = pd.DataFrame(fg_profiles).fillna(0)

        # Calculate diversity metrics
        diversity_metrics = {
            'mean_fg_per_mol': fg_matrix.sum(axis=1).mean(),
            'std_fg_per_mol': fg_matrix.sum(axis=1).std(),
            'fg_richness': (fg_matrix > 0).sum(axis=1).mean(), # Avg # of FG types
            'fg_evenness': self._calculate_evenness(fg_matrix)
        }

        # Most/least common functional groups
        fg_totals = fg_matrix.sum(axis=0).sort_values(ascending=False)
        diversity_metrics['most_common_fg'] = fg_totals.head(3).to_dict()
        diversity_metrics['least_common_fg'] = fg_totals.tail(3).to_dict()

        return diversity_metrics

    def _calculate_evenness(self, fg_matrix):
        """
        Calculate Pielou's evenness index for functional group distribution.
        J = H / ln(S) where H = Shannon entropy, S = species richness
        """
        # Sum across molecules for each FG
        fg_totals = fg_matrix.sum(axis=0)
        total = fg_totals.sum()

        if total == 0:
            return 0

        # Shannon entropy
        H = 0
        for count in fg_totals:
            if count > 0:
                p = count / total
                H -= p * np.log(p)

        # Richness (number of FG types present)
        S = (fg_totals > 0).sum()

        if S <= 1:
            return 1.0

        # Pielou's evenness
        J = H / np.log(S)
        return J

    def calculate_ring_diversity(self):
        """
        Analyze diversity of ring systems.
        """
        ring_info_list = []

        for mol in self.molecules:
            ring_info = mol.GetRingInfo()

            # Ring counts by size
            ring_sizes = [len(ring) for ring in ring_info.AtomRings()]

            # Ring types
            num_aromatic = sum(1 for ring in ring_info.AtomRings()
                             if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring))
            num_alicyclic = len(ring_sizes) - num_aromatic

            # Bridged/fused/spiro
            num_fused = 0
            rings = list(ring_info.AtomRings())
            for i in range(len(rings)):
                for j in range(i + 1, len(rings)):
                    if len(set(rings[i]) & set(rings[j])) >= 2:
                        num_fused += 1

            ring_info_list.append({
                'total_rings': len(ring_sizes),
                'aromatic_rings': num_aromatic,
                'alicyclic_rings': num_alicyclic,
                'fused_rings': num_fused,
                'avg_ring_size': np.mean(ring_sizes) if ring_sizes else 0,
                'ring_sizes': tuple(sorted(ring_sizes))
            })

        df_rings = pd.DataFrame(ring_info_list)

        # Unique ring patterns
        unique_patterns = df_rings['ring_sizes'].nunique()

        return {
            'mean_total_rings': df_rings['total_rings'].mean(),
            'mean_alicyclic': df_rings['alicyclic_rings'].mean(),
            'mean_aromatic': df_rings['aromatic_rings'].mean(),
            'unique_ring_patterns': unique_patterns,
            'pattern_diversity': unique_patterns / len(self.molecules),
            'alicyclic_ratio': df_rings['alicyclic_rings'].sum() / df_rings['total_rings'].sum()
                              if df_rings['total_rings'].sum() > 0 else 0
        }

    def generate_diversity_report(self):
        """
        Generate comprehensive diversity report.
        """
        print("="*80)
        print(" COMPREHENSIVE DIVERSITY ANALYSIS")
        print("="*80)
        print(f"\nTotal molecules: {len(self.molecules)}")

        # 1. Tanimoto diversity
        print(f"\n1 MOLECULAR FINGERPRINT DIVERSITY (Tanimoto):")
        print("-"*80)
        tanimoto = self.calculate_tanimoto_diversity(fp_type='morgan')
        print(f" Mean pairwise distance: {tanimoto['mean_distance']:.3f}")
        print(f" Std deviation: {tanimoto['std_distance']:.3f}")
        print(f" Range: [{tanimoto['min_distance']:.3f}, {tanimoto['max_distance']:.3f}]")
        print(f" Diversity Score: {tanimoto['diversity_score']:.3f} (0=identical, 1=diverse)")

        if tanimoto['diversity_score'] < 0.3:
            print(" LOW diversity - molecules are too similar")
        elif tanimoto['diversity_score'] > 0.7:
            print(" HIGH diversity - excellent chemical variety")
        else:
            print(" MODERATE diversity")

        # 2. Scaffold diversity
        print(f"\n2 SCAFFOLD DIVERSITY:")
        print("-"*80)
        scaffold = self.calculate_scaffold_diversity()
        print(f" Unique scaffolds: {scaffold['unique_scaffolds']}")
        print(f" Scaffold ratio: {scaffold['scaffold_ratio']:.3f}")
        print(f" Shannon entropy: {scaffold['entropy']:.3f} bits")
        print(f" Normalized entropy: {scaffold['normalized_entropy']:.3f}")

        if scaffold['normalized_entropy'] > 0.8:
            print(" Excellent scaffold diversity")
        elif scaffold['normalized_entropy'] < 0.5:
            print(" Low scaffold diversity - too many duplicates")

        # 3. Functional group diversity
        print(f"\n3 FUNCTIONAL GROUP DIVERSITY:")
        print("-"*80)
        fg = self.calculate_functional_group_diversity()
        print(f" Mean FGs per molecule: {fg['mean_fg_per_mol']:.2f} ± {fg['std_fg_per_mol']:.2f}")
        print(f" Average FG richness: {fg['fg_richness']:.2f} types")
        print(f" Distribution evenness: {fg['fg_evenness']:.3f}")
        print(f"\n Most common functional groups:")
        for fg_name, count in fg['most_common_fg'].items():
            print(f" • {fg_name}: {int(count)} occurrences")

        # 4. Ring diversity
        print(f"\n4 RING SYSTEM DIVERSITY:")
        print("-"*80)
        rings = self.calculate_ring_diversity()
        print(f" Mean rings per molecule: {rings['mean_total_rings']:.2f}")
        print(f" Mean alicyclic rings: {rings['mean_alicyclic']:.2f}")
        print(f" Alicyclic ratio: {rings['alicyclic_ratio']:.1%}")
        print(f" Unique ring patterns: {rings['unique_ring_patterns']}")
        print(f" Pattern diversity: {rings['pattern_diversity']:.3f}")

        print("\n" + "="*80)

        # Summary score
        overall_score = (
            tanimoto['diversity_score'] * 0.3 +
            scaffold['normalized_entropy'] * 0.3 +
            fg['fg_evenness'] * 0.2 +
            rings['pattern_diversity'] * 0.2
        )

        print(f"\n OVERALL DIVERSITY SCORE: {overall_score:.3f} / 1.000")

        if overall_score > 0.7:
            print(" Excellent overall diversity")
        elif overall_score > 0.5:
            print(" Good - Moderate diversity, some improvement possible")
        else:
            print(" Poor - Low diversity, generation needs tuning")

        print("="*80)

        return {
            'tanimoto': tanimoto,
            'scaffold': scaffold,
            'functional_groups': fg,
            'rings': rings,
            'overall_score': overall_score
        }


# Example usage:
# diversity_analyzer = EnhancedDiversityAnalyzer(generated_molecules)
# report = diversity_analyzer.generate_diversity_report()

