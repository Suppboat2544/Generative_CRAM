"""Notebook EnhancedOxidationFeatures (20D per atom)."""
from __future__ import annotations
import numpy as np
from rdkit import Chem

# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED OXIDATION FEATURES (20D per atom)
# ══════════════════════════════════════════════════════════════════════════════
#
# Detailed oxidation chemistry for CRAM:
# - Per-carbon oxidation state (continuous value)
# - α/β/γ carbon classification relative to -COOH
# - Hydroxyl group positioning (1°/2°/3°)
# - Carbonyl type detection
# ══════════════════════════════════════════════════════════════════════════════

class EnhancedOxidationFeatures:
    """
    Extract detailed oxidation state features for CRAM molecules.
    
    Features extracted (20D per atom):
    - Oxidation state: 3D (normalized state, is_reduced, is_oxidized)
    - COOH proximity: 5D (is_alpha, is_beta, is_gamma, is_delta, distance_to_cooh)
    - Hydroxyl type: 4D (is_primary_oh, is_secondary_oh, is_tertiary_oh, has_oh)
    - Carbonyl type: 5D (is_ketone_c, is_aldehyde_c, is_carboxyl_c, is_ester_c, is_amide_c)
    - Electron effects: 3D (inductive_effect, resonance_effect, hyperconjugation)
    """
    
    # SMARTS patterns for functional group detection
    PATTERNS = {
        # Carboxyl and derivatives
        'carboxyl': '[CX3](=O)[OX2H1]',
        'carboxylate': '[CX3](=O)[O-]',
        'ester': '[CX3](=O)[OX2][C]',
        'amide': '[CX3](=O)[NX3]',
        
        # Carbonyls
        'ketone': '[CX3](=O)([C])[C]',
        'aldehyde': '[CX3H1](=O)[C,H]',
        
        # Hydroxyls
        'primary_oh': '[CH2][OX2H]',
        'secondary_oh': '[CHX4][OX2H]',
        'tertiary_oh': '[CX4]([C])([C])([C])[OX2H]',
        
        # Ether
        'ether': '[OX2]([C])[C]',
    }
    
    def __init__(self):
        """Initialize SMARTS patterns."""
        self.compiled_patterns = {}
        for name, smarts in self.PATTERNS.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is not None:
                    self.compiled_patterns[name] = pattern
            except:
                pass
        
        print(f"EnhancedOxidationFeatures initialized with {len(self.compiled_patterns)} patterns")
    
    def calculate_carbon_oxidation_state(self, mol, atom_idx):
        """
        Calculate oxidation state for a carbon atom.
        
        Oxidation state = Σ(electronegativity difference with neighbors)
        For carbon: 
        - Each C-H bond: -1
        - Each C-C bond: 0
        - Each C-O bond: +1
        - Each C-N bond: +1 (simplified)
        - Each C=O bond: +2
        
        Returns: Normalized oxidation state (-4 to +4 → 0 to 1)
        """
        atom = mol.GetAtomWithIdx(atom_idx)
        
        if atom.GetAtomicNum() != 6: # Not carbon
            return 0.0
        
        oxidation_state = 0
        
        for neighbor in atom.GetNeighbors():
            neighbor_num = neighbor.GetAtomicNum()
            bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
            
            if bond is None:
                continue
            
            bond_order = bond.GetBondTypeAsDouble()
            
            if neighbor_num == 1: # Hydrogen
                oxidation_state -= bond_order
            elif neighbor_num == 6: # Carbon
                oxidation_state += 0 # No change
            elif neighbor_num == 7: # Nitrogen
                oxidation_state += bond_order
            elif neighbor_num == 8: # Oxygen
                oxidation_state += bond_order * 1.5 # O is more electronegative
            elif neighbor_num == 16: # Sulfur
                oxidation_state += bond_order * 0.5
            elif neighbor_num == 17: # Chlorine
                oxidation_state += bond_order
        
        # Add implicit hydrogens
        oxidation_state -= atom.GetTotalNumHs()
        
        # Normalize from typical range [-4, +4] to [0, 1]
        normalized = (oxidation_state + 4) / 8.0
        return max(0.0, min(1.0, normalized))
    
    def get_cooh_proximity(self, mol, atom_idx):
        """
        Classify carbon based on proximity to carboxyl group.
        
        Returns: (is_alpha, is_beta, is_gamma, is_delta, min_distance)
        - α-carbon: Directly attached to -COOH
        - β-carbon: 2 bonds from -COOH
        - γ-carbon: 3 bonds from -COOH
        - δ-carbon: 4 bonds from -COOH
        """
        # Find carboxyl carbons
        carboxyl_carbons = set()
        
        for name in ['carboxyl', 'carboxylate']:
            if name in self.compiled_patterns:
                matches = mol.GetSubstructMatches(self.compiled_patterns[name])
                for match in matches:
                    carboxyl_carbons.add(match[0]) # First atom is the carbonyl carbon
        
        if not carboxyl_carbons:
            return [0.0, 0.0, 0.0, 0.0, 1.0] # No COOH found
        
        # Calculate shortest path to any carboxyl carbon
        atom = mol.GetAtomWithIdx(atom_idx)
        
        if atom_idx in carboxyl_carbons:
            return [0.0, 0.0, 0.0, 0.0, 0.0] # Is the carboxyl carbon itself
        
        min_distance = float('inf')
        
        for cooh_idx in carboxyl_carbons:
            try:
                path = Chem.GetShortestPath(mol, atom_idx, cooh_idx)
                if path:
                    distance = len(path) - 1
                    min_distance = min(min_distance, distance)
            except:
                continue
        
        if min_distance == float('inf'):
            min_distance = 10 # No path found
        
        # Classify based on distance
        is_alpha = 1.0 if min_distance == 1 else 0.0
        is_beta = 1.0 if min_distance == 2 else 0.0
        is_gamma = 1.0 if min_distance == 3 else 0.0
        is_delta = 1.0 if min_distance == 4 else 0.0
        
        # Normalize distance (0-10 bonds → 0-1)
        norm_distance = min(min_distance / 10.0, 1.0)
        
        return [is_alpha, is_beta, is_gamma, is_delta, norm_distance]
    
    def get_hydroxyl_type(self, mol, atom_idx):
        """
        Determine if atom has attached hydroxyl and its type.
        
        Returns: (is_primary_oh, is_secondary_oh, is_tertiary_oh, has_oh)
        """
        atom = mol.GetAtomWithIdx(atom_idx)
        
        if atom.GetAtomicNum() != 6: # Not carbon
            return [0.0, 0.0, 0.0, 0.0]
        
        # Check for attached OH
        has_oh = False
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 8: # Oxygen
                # Check if it's hydroxyl (has H)
                for o_neighbor in neighbor.GetNeighbors():
                    if o_neighbor.GetAtomicNum() == 1:
                        has_oh = True
                        break
                if neighbor.GetTotalNumHs() > 0:
                    has_oh = True
        
        if not has_oh:
            return [0.0, 0.0, 0.0, 0.0]
        
        # Count carbon neighbors to determine primary/secondary/tertiary
        carbon_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 6)
        
        is_primary = 1.0 if carbon_neighbors <= 1 else 0.0
        is_secondary = 1.0 if carbon_neighbors == 2 else 0.0
        is_tertiary = 1.0 if carbon_neighbors >= 3 else 0.0
        
        return [is_primary, is_secondary, is_tertiary, 1.0]
    
    def get_carbonyl_type(self, mol, atom_idx):
        """
        Determine carbonyl type for carbon atoms.
        
        Returns: (is_ketone, is_aldehyde, is_carboxyl, is_ester, is_amide)
        """
        atom = mol.GetAtomWithIdx(atom_idx)
        
        if atom.GetAtomicNum() != 6:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        result = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Check each pattern
        for i, name in enumerate(['ketone', 'aldehyde', 'carboxyl', 'ester', 'amide']):
            if name in self.compiled_patterns:
                matches = mol.GetSubstructMatches(self.compiled_patterns[name])
                for match in matches:
                    if atom_idx in match:
                        result[i] = 1.0
                        break
        
        return result
    
    def get_electron_effects(self, mol, atom_idx):
        """
        Estimate electronic effects on the atom.
        
        Returns: (inductive_effect, resonance_effect, hyperconjugation)
        """
        atom = mol.GetAtomWithIdx(atom_idx)
        
        if atom.GetAtomicNum() != 6:
            return [0.0, 0.0, 0.0]
        
        inductive = 0.0
        resonance = 0.0
        hyperconj = 0.0
        
        for neighbor in atom.GetNeighbors():
            n_num = neighbor.GetAtomicNum()
            
            # Inductive effect (electron-withdrawing groups)
            if n_num in [7, 8, 9, 17]: # N, O, F, Cl
                inductive += 0.3
            elif n_num == 6 and neighbor.GetIsAromatic():
                resonance += 0.2
            
            # Hyperconjugation (C-H bonds on adjacent carbons)
            if n_num == 6:
                hyperconj += neighbor.GetTotalNumHs() * 0.1
        
        return [min(inductive, 1.0), min(resonance, 1.0), min(hyperconj, 1.0)]
    
    def get_atom_features(self, mol, atom_idx):
        """
        Get all 20D oxidation features for an atom.
        
        Returns: 20D feature vector
        """
        features = []
        
        # Oxidation state (3D)
        ox_state = self.calculate_carbon_oxidation_state(mol, atom_idx)
        is_reduced = 1.0 if ox_state < 0.4 else 0.0
        is_oxidized = 1.0 if ox_state > 0.6 else 0.0
        features.extend([ox_state, is_reduced, is_oxidized])
        
        # COOH proximity (5D)
        features.extend(self.get_cooh_proximity(mol, atom_idx))
        
        # Hydroxyl type (4D)
        features.extend(self.get_hydroxyl_type(mol, atom_idx))
        
        # Carbonyl type (5D)
        features.extend(self.get_carbonyl_type(mol, atom_idx))
        
        # Electron effects (3D)
        features.extend(self.get_electron_effects(mol, atom_idx))
        
        return features
    
    def get_molecule_features(self, mol):
        """
        Get oxidation features for all atoms in molecule.
        
        Returns: (num_atoms, 20) array
        """
        num_atoms = mol.GetNumAtoms()
        features = np.zeros((num_atoms, 20), dtype=np.float32)
        
        for i in range(num_atoms):
            features[i] = self.get_atom_features(mol, i)
        
        return features
    
    def get_molecule_summary(self, mol):
        """
        Get summary statistics for molecule-level features.
        
        Returns: 10D summary vector
        """
        atom_features = self.get_molecule_features(mol)
        
        # Average oxidation state
        avg_oxidation = np.mean(atom_features[:, 0])
        
        # Count α/β/γ carbons
        num_alpha = np.sum(atom_features[:, 3])
        num_beta = np.sum(atom_features[:, 4])
        num_gamma = np.sum(atom_features[:, 5])
        
        # Count hydroxyl types
        num_primary_oh = np.sum(atom_features[:, 8])
        num_secondary_oh = np.sum(atom_features[:, 9])
        num_tertiary_oh = np.sum(atom_features[:, 10])
        
        # Count carbonyl types
        num_carboxyl = np.sum(atom_features[:, 14])
        num_ketone = np.sum(atom_features[:, 12])
        num_aldehyde = np.sum(atom_features[:, 13])
        
        return np.array([
            avg_oxidation,
            num_alpha / max(mol.GetNumAtoms(), 1),
            num_beta / max(mol.GetNumAtoms(), 1),
            num_gamma / max(mol.GetNumAtoms(), 1),
            num_primary_oh,
            num_secondary_oh,
            num_tertiary_oh,
            num_carboxyl,
            num_ketone,
            num_aldehyde
        ], dtype=np.float32)



