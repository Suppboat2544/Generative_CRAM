"""Notebook TemplateExtractor."""
from __future__ import annotations
from collections import defaultdict
from typing import Optional, List, Dict
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

class TemplateExtractor:
    """
    Extract and analyze molecular templates/scaffolds from CRAM molecules.

    Methods:
    - extract_murcko_scaffold: Get Bemis-Murcko scaffold
    - extract_generic_scaffold: Get generic (atom-type independent) scaffold
    - find_common_substructures: Maximum common substructure analysis
    - cluster_by_similarity: Group similar scaffolds
    - extract_scaffolds: Quick method to extract top N scaffolds (for generation)
    """

    def __init__(self, df: pd.DataFrame, smiles_col: str = 'CanonicalSMILES'):
        """
        Initialize template extractor.

        Args:
            df: DataFrame containing CRAM molecules
            smiles_col: Name of SMILES column
        """
        self.df = df.copy()
        self.smiles_col = smiles_col
        self.templates = {}
        self.scaffold_to_molecules = defaultdict(list)

        # For generation compatibility
        self.scaffolds = []
        self.top_scaffolds = []

    def extract_murcko_scaffold(self, smiles: str) -> Optional[str]:
        """Extract Murcko scaffold (core structure without side chains)."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Get Bemis-Murcko scaffold
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except:
            return None

    def extract_generic_scaffold(self, smiles: str) -> Optional[str]:
        """Extract generic scaffold (atom types replaced with wildcards)."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Get generic scaffold (all atoms as *)
            scaffold = MurckoScaffold.MakeScaffoldGeneric(mol)
            return Chem.MolToSmiles(scaffold)
        except:
            return None

    def extract_scaffolds(self, top_n: int = 1000, scaffold_type: str = 'murcko'):
        """
        Quick method to extract top N scaffolds for generation.

        Args:
            top_n: Number of top scaffolds to extract
            scaffold_type: 'murcko' or 'generic'
        """
        print(f"\n Extracting top {top_n} {scaffold_type} scaffolds...")

        # Extract all scaffolds
        scaffolds = []
        extract_func = self.extract_murcko_scaffold if scaffold_type == 'murcko' else self.extract_generic_scaffold

        for smiles in self.df[self.smiles_col]:
            scaffold = extract_func(smiles)
            if scaffold:
                scaffolds.append(scaffold)

        # Count and get top N
        scaffold_counts = Counter(scaffolds)
        self.scaffolds = list(scaffold_counts.keys())
        self.top_scaffolds = [s for s, _ in scaffold_counts.most_common(top_n)]

        print(f" Top {top_n} scaffolds saved for generation")

        return self.top_scaffolds

    def analyze_all_scaffolds(self, scaffold_type: str = 'murcko') -> Dict:
        """
        Extract scaffolds from all molecules in dataset.

        Args:
            scaffold_type: 'murcko' or 'generic'

        Returns:
            Dictionary with scaffold statistics
        """
        print(f"\n{'='*80}")
        print(f" EXTRACTING {scaffold_type.upper()} SCAFFOLDS")
        print('='*80)

        scaffolds = []
        extract_func = self.extract_murcko_scaffold if scaffold_type == 'murcko' else self.extract_generic_scaffold

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting scaffolds"):
            smiles = row[self.smiles_col]
            scaffold = extract_func(smiles)

            if scaffold:
                scaffolds.append(scaffold)
                self.scaffold_to_molecules[scaffold].append(smiles)

        # Add to dataframe
        self.df[f'{scaffold_type}_scaffold'] = [extract_func(s) for s in self.df[self.smiles_col]]

        # Calculate statistics
        scaffold_counts = Counter(scaffolds)
        unique_scaffolds = len(scaffold_counts)
        total_molecules = len(scaffolds)

        # Update for generation compatibility
        self.scaffolds = list(scaffold_counts.keys())
        self.top_scaffolds = [s for s, _ in scaffold_counts.most_common(5)]

        results = {
            'type': scaffold_type,
            'total_molecules': total_molecules,
            'unique_scaffolds': unique_scaffolds,
            'coverage': total_molecules / len(self.df) * 100,
            'avg_molecules_per_scaffold': total_molecules / unique_scaffolds if unique_scaffolds > 0 else 0,
            'scaffold_counts': scaffold_counts,
            'top_scaffolds': scaffold_counts.most_common(20)
        }

        # Print summary
        print(f" Total molecules processed: {total_molecules:,}")
        print(f" Unique scaffolds found: {unique_scaffolds:,}")
        print(f" Coverage: {results['coverage']:.1f}%")
        print(f" Avg molecules per scaffold: {results['avg_molecules_per_scaffold']:.1f}")

        print(f"\n TOP 10 MOST COMMON SCAFFOLDS:")
        for i, (scaffold, count) in enumerate(results['top_scaffolds'][:10], 1):
            print(f" {i:2d}. {scaffold[:60]:60s} ({count:4d} molecules, {count/total_molecules*100:5.1f}%)")

        self.templates[scaffold_type] = results
        print('='*80)

        return results

    def find_common_substructures(self, smiles_list: List[str], min_atoms: int = 5) -> List[Dict]:
        """
        Find maximum common substructures (MCS) among a group of molecules.

        Args:
            smiles_list: List of SMILES strings
            min_atoms: Minimum number of atoms in MCS

        Returns:
            List of common substructures with statistics
        """
        if len(smiles_list) < 2:
            return []

        mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]

        if len(mols) < 2:
            return []

        results = []

        # Find MCS for all pairs
        for i in range(min(len(mols), 10)): # Limit to first 10 for performance
            for j in range(i+1, min(len(mols), 10)):
                try:
                    mcs = rdFMCS.FindMCS([mols[i], mols[j]],
                                         timeout=5,
                                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                                         bondCompare=rdFMCS.BondCompare.CompareOrder)

                    if mcs.numAtoms >= min_atoms:
                        results.append({
                            'smarts': mcs.smartsString,
                            'num_atoms': mcs.numAtoms,
                            'num_bonds': mcs.numBonds,
                            'mol_i': i,
                            'mol_j': j
                        })
                except:
                    continue

        return results

    def cluster_scaffolds(self, scaffold_type: str = 'murcko', n_clusters: int = 100) -> Dict:
        """
        Cluster scaffolds by structural similarity.

        Args:
            scaffold_type: Type of scaffold to cluster
            n_clusters: Number of clusters

        Returns:
            Clustering results
        """
        if scaffold_type not in self.templates:
            print(f" Please run analyze_all_scaffolds('{scaffold_type}') first")
            return {}

        print(f"\n CLUSTERING {scaffold_type.upper()} SCAFFOLDS")

        # Get unique scaffolds
        scaffolds = list(self.templates[scaffold_type]['scaffold_counts'].keys())

        if len(scaffolds) < n_clusters:
            print(f" Number of scaffolds ({len(scaffolds)}) < n_clusters ({n_clusters})")
            n_clusters = max(2, len(scaffolds) // 2)
            print(f" Adjusting to {n_clusters} clusters")

        # Calculate fingerprints
        fps = []
        valid_scaffolds = []

        for scaffold in scaffolds:
            mol = Chem.MolFromSmiles(scaffold)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(fp)
                valid_scaffolds.append(scaffold)

        if len(fps) < 2:
            print(" Not enough valid scaffolds for clustering")
            return {}

        # Convert to numpy array
        fp_array = np.array([list(fp) for fp in fps])

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=100)
        cluster_labels = kmeans.fit_predict(fp_array)

        # Organize results
        clusters = defaultdict(list)
        for scaffold, label in zip(valid_scaffolds, cluster_labels):
            count = self.templates[scaffold_type]['scaffold_counts'][scaffold]
            clusters[label].append({'scaffold': scaffold, 'count': count})

        # Sort clusters by size
        cluster_sizes = {k: sum(s['count'] for s in v) for k, v in clusters.items()}

        print(f" Number of clusters: {len(clusters)}")
        print(f"\n CLUSTER SUMMARY:")
        for cluster_id in sorted(cluster_sizes.keys(), key=lambda x: cluster_sizes[x], reverse=True):
            size = cluster_sizes[cluster_id]
            n_scaffolds = len(clusters[cluster_id])
            print(f" Cluster {cluster_id}: {n_scaffolds} scaffolds, {size} molecules total")

        return {
            'clusters': dict(clusters),
            'cluster_sizes': cluster_sizes,
            'labels': cluster_labels,
            'scaffolds': valid_scaffolds
        }

    def visualize_top_scaffolds(self, scaffold_type: str = 'murcko', n_show: int = 16):
        """Visualize top N most common scaffolds."""
        if scaffold_type not in self.templates:
            print(f" Please run analyze_all_scaffolds('{scaffold_type}') first")
            return

        top_scaffolds = self.templates[scaffold_type]['top_scaffolds'][:n_show]

        mols = []
        legends = []

        for i, (scaffold, count) in enumerate(top_scaffolds, 1):
            mol = Chem.MolFromSmiles(scaffold)
            if mol:
                mols.append(mol)
                legends.append(f"#{i}\n{count} mols\n({count/self.templates[scaffold_type]['total_molecules']*100:.1f}%)")

        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 300),
                                       legends=legends, returnPNG=False)

            plt.figure(figsize=(16, 12))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Top {len(mols)} Most Common {scaffold_type.capitalize()} Scaffolds',
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(f'02_top_{scaffold_type}_scaffolds.png', dpi=300, bbox_inches='tight')
            plt.show()

        else:
            print("No valid molecules to visualize")

