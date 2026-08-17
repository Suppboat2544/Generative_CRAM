"""Notebook 163D building blocks: 45D base + 28D SA + 15D marine + 20D ring-chain."""
from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdPartialCharges

# Enhanced Feature Extraction (from CRAM_Generator_Clean_V2.ipynb)
from rdkit.Chem import Descriptors, rdPartialCharges

# Pauling electronegativity values
ELECTRONEGATIVITY = {
    1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
}

def get_base_node_features_45D(mol):
    """Extract 45D base features for all atoms."""
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except:
        pass

    node_features = []

    for atom in mol.GetAtoms():
        features = []

        # 1. Atomic number (1)
        features.append(atom.GetAtomicNum())

        # 2. Degree (7: 0-6)
        degree = min(atom.GetDegree(), 6)
        features.extend([1 if degree == i else 0 for i in range(7)])

        # 3. Formal charge (5: -2 to +2)
        charge = max(-2, min(2, atom.GetFormalCharge()))
        features.extend([1 if charge == i else 0 for i in range(-2, 3)])

        # 4. Hybridization (6: SP, SP2, SP3, SP3D, SP3D2, OTHER)
        hyb_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]
        hyb = atom.GetHybridization()
        features.extend([1 if hyb == h else 0 for h in hyb_types])
        features.append(1 if hyb not in hyb_types else 0)

        # 5. Aromaticity (1)
        features.append(1 if atom.GetIsAromatic() else 0)

        # 6. In ring (1)
        features.append(1 if atom.IsInRing() else 0)

        # 7. Ring size (6: 3-8)
        if atom.IsInRing():
            ring_info = mol.GetRingInfo()
            atom_rings = [len(r) for r in ring_info.AtomRings() if atom.GetIdx() in r]
            ring_size = min(atom_rings) if atom_rings else 0
            features.extend([1 if ring_size == i else 0 for i in range(3, 9)])
        else:
            features.extend([0] * 6)

        # 8. Chirality (2)
        try:
            chiral_tag = atom.GetChiralTag()
            features.append(1 if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0)
            features.append(1 if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0)
        except:
            features.extend([0, 0])

        # 9. Implicit hydrogens (5: 0-4)
        implicit_h = min(atom.GetTotalNumHs(), 4)
        features.extend([1 if implicit_h == i else 0 for i in range(5)])

        # 10. Total valence (1)
        valence = min(atom.GetTotalValence(), 6)
        features.append(valence / 6.0)

        # 11. Electronegativity (1)
        en = ELECTRONEGATIVITY.get(atom.GetAtomicNum(), 2.5)
        features.append(en / 4.0)

        # 12. Partial charge (1)
        try:
            partial_charge = float(atom.GetProp('_GasteigerCharge'))
            if not np.isnan(partial_charge) and not np.isinf(partial_charge):
                features.append(np.clip(partial_charge, -1, 1))
            else:
                features.append(0.0)
        except:
            features.append(0.0)

        # 13. Valence electrons (1)
        features.append(atom.GetTotalValence() / 8.0)

        # 14-15. CRAM-specific (2)
        features.append(0.0) # Carboxyl indicator (filled later)
        features.append(1 if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() else 0)

        # 16-20. Atom type (5: C, N, O, S, Other)
        atom_symbol = atom.GetSymbol()
        features.append(1 if atom_symbol == 'C' else 0)
        features.append(1 if atom_symbol == 'N' else 0)
        features.append(1 if atom_symbol == 'O' else 0)
        features.append(1 if atom_symbol == 'S' else 0)
        features.append(1 if atom_symbol not in ['C', 'N', 'O', 'S'] else 0)

        node_features.append(np.array(features, dtype=np.float32))

    return node_features

def get_synthetic_features_28D(mol, atom_idx):
    """Extract 28D synthetic CRAM features (C3/C4 analysis)."""
    features = [0.0] * 28

    # Detect cyclohexane
    cyclohexane = Chem.MolFromSmarts('C1CCCCC1')
    if mol.HasSubstructMatch(cyclohexane):
        features[0] = 1.0

    atom = mol.GetAtomWithIdx(atom_idx)

    # Basic properties
    features[3] = float(atom.GetDegree()) / 4.0
    features[4] = 1 if atom.IsInRing() else 0

    # Stereochemistry
    try:
        chiral = atom.GetChiralTag()
        features[5] = 1 if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0
        features[6] = 1 if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0
    except:
        pass

    features[7] = 1 if atom.IsInRingSize(6) else 0

    # Chain properties
    features[8] = float(len(mol.GetAtoms())) / 50.0
    features[9] = 1 if not atom.GetIsAromatic() else 0

    # Double bonds
    num_double = sum(1 for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)
    features[10] = float(num_double) / 3.0

    # Fill remaining with atom properties
    features[13] = 1 if atom.GetSymbol() == 'C' else 0
    features[14] = 1 if atom.GetSymbol() == 'O' else 0
    features[15] = 1 if atom.GetSymbol() == 'N' else 0
    features[16] = float(atom.GetTotalNumHs()) / 4.0
    features[17] = 1 if any(b.GetBondType() == Chem.rdchem.BondType.DOUBLE for b in atom.GetBonds()) else 0

    return np.array(features, dtype=np.float32)

def get_marine_features_15D(mol, atom_idx):
    """Extract 15D marine DOM features (oxidation, degradation)."""
    features = []
    atom = mol.GetAtomWithIdx(atom_idx)

    # Oxidation state (3)
    features.append(1 if atom.GetSymbol() == 'O' else 0)
    num_o_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == 'O')
    features.append(float(num_o_neighbors) / 3.0)
    features.append(1 if any(b.GetBondType() == Chem.rdchem.BondType.DOUBLE and
                            b.GetOtherAtom(atom).GetSymbol() == 'O'
                            for b in atom.GetBonds()) else 0)

    # Degradation markers (3)
    features.append(1 if atom.GetSymbol() in ['C', 'O', 'N'] else 0)
    features.append(float(atom.GetDegree()) / 4.0)
    features.append(1 if atom.IsInRing() and atom.GetIsAromatic() else 0)

    # Photo-oxidation (3)
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if any(n.GetSymbol() == 'O' for n in atom.GetNeighbors()) else 0)
    features.append(float(num_o_neighbors) / 2.0)

    # Bioavailability (3)
    features.append(1 if atom.GetSymbol() == 'C' and not atom.GetIsAromatic() else 0)
    features.append(float(atom.GetTotalNumHs()) / 3.0)
    features.append(1 if atom.GetDegree() <= 2 else 0)

    # Aromatic condensation (3)
    if atom.IsInRing():
        ring_info = mol.GetRingInfo()
        num_rings = len([r for r in ring_info.AtomRings() if atom_idx in r])
        features.append(float(num_rings) / 3.0)
    else:
        features.append(0.0)
    features.append(1 if atom.GetIsAromatic() and atom.IsInRing() else 0)
    features.append(1 if atom.GetSymbol() == 'C' and atom.GetIsAromatic() else 0)

    return np.array(features, dtype=np.float32)

def get_enhanced_node_features(mol, mode='synthetic'):
    """
    Extract enhanced node features based on selected mode.

    Returns:
    - synthetic: 73D (45 base + 28 synthetic)
    - marine: 88D (45 base + 28 synthetic + 15 marine)
    - unified: 88D (45 base + 28 synthetic + 15 marine)
    """
    base_features = get_base_node_features_45D(mol)
    enhanced_features = []

    for atom_idx in range(mol.GetNumAtoms()):
        base = base_features[atom_idx]
        synthetic = get_synthetic_features_28D(mol, atom_idx)

        if mode == 'synthetic':
            combined = np.concatenate([base, synthetic])
        else: # marine or unified
            marine = get_marine_features_15D(mol, atom_idx)
            combined = np.concatenate([base, synthetic, marine])

        enhanced_features.append(combined)

    return enhanced_features

def get_edge_features(mol):
    """Extract edge (bond) features - 6D per bond."""
    edge_features = {}

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = []

        # Bond type (4)
        bond_type = bond.GetBondType()
        features.append(1 if bond_type == Chem.rdchem.BondType.SINGLE else 0)
        features.append(1 if bond_type == Chem.rdchem.BondType.DOUBLE else 0)
        features.append(1 if bond_type == Chem.rdchem.BondType.TRIPLE else 0)
        features.append(1 if bond_type == Chem.rdchem.BondType.AROMATIC else 0)

        # Is conjugated (1)
        features.append(1 if bond.GetIsConjugated() else 0)

        # Is in ring (1)
        features.append(1 if bond.IsInRing() else 0)

        feat_array = np.array(features, dtype=np.float32)
        edge_features[(i, j)] = feat_array
        edge_features[(j, i)] = feat_array

    return edge_features


# ============================================================================
# CRAM-SPECIFIC RING-CHAIN FEATURES (20D)
# ============================================================================
# 
# These features explicitly capture what makes TRUE CRAM molecules:
# - Carboxyl-Rich: Multiple -COOH groups positioned on chains/rings
# - Aliphatic/Alicyclic: Non-aromatic carbons in chains and saturated rings
# - Cyclic+Chain Hybrid: Junction atoms connecting rings to side chains
#
# The model will learn to generate molecules with these structural patterns.
# ============================================================================

def get_cram_ring_chain_features_20D(mol, atom_idx):
    """
    Extract 20D CRAM-specific ring-chain features for a single atom.
    
    These features capture the UNIQUE chemistry of CRAM molecules:
    - Alicyclic ring membership (cyclohexane, cyclopentane, cyclobutane)
    - Chain connectivity patterns (branches, length)
    - Junction atoms (ring-chain connections)
    - Carboxyl proximity (distance to -COOH groups)
    - Hybridization patterns (sp3 for aliphatic character)
    
    Args:
        mol: RDKit molecule object
        atom_idx: Index of the atom
        
    Returns:
        np.array of 20 features
    """
    features = [0.0] * 20
    atom = mol.GetAtomWithIdx(atom_idx)
    
    try:
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        # ========================================
        # PART 1: ALICYCLIC RING FEATURES (5D)
        # ========================================
        
        # 1. Is this atom in an alicyclic (non-aromatic) ring?
        in_alicyclic = False
        ring_sizes = []
        for ring in atom_rings:
            if atom_idx in ring:
                # Check if ring is non-aromatic (alicyclic)
                ring_aromatic = any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                if not ring_aromatic:
                    in_alicyclic = True
                    ring_sizes.append(len(ring))
        
        features[0] = 1.0 if in_alicyclic else 0.0
        
        # 2-4. Ring size encoding (5-membered, 6-membered, other)
        if ring_sizes:
            features[1] = 1.0 if 5 in ring_sizes else 0.0 # Cyclopentane
            features[2] = 1.0 if 6 in ring_sizes else 0.0 # Cyclohexane
            features[3] = 1.0 if any(s not in [5, 6] for s in ring_sizes) else 0.0 # Other (3, 4, 7+)
        
        # 5. Number of alicyclic rings atom is part of (normalized)
        num_alicyclic_rings = sum(1 for ring in atom_rings 
                                   if atom_idx in ring and 
                                   not any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
        features[4] = min(num_alicyclic_rings, 3) / 3.0
        
        # ========================================
        # PART 2: JUNCTION ATOM FEATURES (5D)
        # ========================================
        # Junction atoms connect rings to chains - critical for CRAM structure!
        
        # 5. Is this a junction atom? (in ring but has non-ring neighbor)
        is_junction = False
        num_ring_neighbors = 0
        num_chain_neighbors = 0
        
        if atom.IsInRing():
            for neighbor in atom.GetNeighbors():
                if neighbor.IsInRing():
                    num_ring_neighbors += 1
                else:
                    num_chain_neighbors += 1
                    is_junction = True
        
        features[5] = 1.0 if is_junction else 0.0
        
        # 6. Ring connectivity ratio (how many neighbors in ring vs chain)
        total_neighbors = atom.GetDegree()
        if total_neighbors > 0:
            features[6] = num_chain_neighbors / total_neighbors
        
        # 7. Is this atom a chain terminus (degree 1, not in ring)?
        features[7] = 1.0 if (atom.GetDegree() == 1 and not atom.IsInRing()) else 0.0
        
        # 8. Chain branching point (degree >= 3, not in ring)
        features[8] = 1.0 if (atom.GetDegree() >= 3 and not atom.IsInRing()) else 0.0
        
        # 9. Pure chain atom (not in ring, not branching, not terminus)
        features[9] = 1.0 if (not atom.IsInRing() and 
                              1 < atom.GetDegree() < 3) else 0.0
        
        # ========================================
        # PART 3: CARBOXYL PROXIMITY FEATURES (5D)
        # ========================================
        # Carboxyl groups define CRAM - track their relationship to rings/chains
        
        # Find all carboxyl carbons in the molecule
        carboxyl_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
        carboxyl_matches = mol.GetSubstructMatches(carboxyl_pattern)
        carboxyl_carbons = [match[0] for match in carboxyl_matches]
        
        # 10. Is this the carboxyl carbon itself?
        features[10] = 1.0 if atom_idx in carboxyl_carbons else 0.0
        
        # 11. Is this atom directly bonded to a carboxyl carbon?
        bonded_to_carboxyl = False
        for carboxyl_c in carboxyl_carbons:
            if mol.GetBondBetweenAtoms(atom_idx, carboxyl_c) is not None:
                bonded_to_carboxyl = True
                break
        features[11] = 1.0 if bonded_to_carboxyl else 0.0
        
        # 12. Distance to nearest carboxyl (topological, normalized)
        if carboxyl_carbons:
            try:
                min_dist = float('inf')
                for carboxyl_c in carboxyl_carbons:
                    if atom_idx == carboxyl_c:
                        min_dist = 0
                        continue
                    path = Chem.GetShortestPath(mol, atom_idx, carboxyl_c)
                    if path:
                        min_dist = min(min_dist, len(path) - 1)
                features[12] = 1.0 / (1.0 + min_dist) if min_dist < float('inf') else 0.0
            except Exception:
                features[12] = 0.0
        
        # 13. Carboxyl attached to ring vs chain
        # If atom is in ring and close to carboxyl, the COOH is on ring
        if in_alicyclic and carboxyl_carbons:
            for carboxyl_c in carboxyl_carbons:
                if atom_idx == carboxyl_c:
                    features[13] = 1.0
                    break
                try:
                    path = Chem.GetShortestPath(mol, atom_idx, carboxyl_c)
                except Exception:
                    continue
                if path and len(path) <= 2: # Directly attached
                    features[13] = 1.0 # Carboxyl attached to ring
                    break
        
        # 14. Number of carboxyls normalized
        features[14] = min(len(carboxyl_carbons), 4) / 4.0
        
        # ========================================
        # PART 4: ALIPHATIC CHARACTER FEATURES (5D)
        # ========================================
        
        # 15. sp3 carbon (aliphatic character)
        is_sp3_carbon = (atom.GetSymbol() == 'C' and 
                         atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3)
        features[15] = 1.0 if is_sp3_carbon else 0.0
        
        # 16. Saturated carbon (no double/triple bonds)
        is_saturated = True
        for bond in atom.GetBonds():
            if bond.GetBondType() in [Chem.rdchem.BondType.DOUBLE, 
                                       Chem.rdchem.BondType.TRIPLE]:
                is_saturated = False
                break
        features[16] = 1.0 if (atom.GetSymbol() == 'C' and is_saturated) else 0.0
        
        # 17. Oxygen neighbor count (for oxidation state)
        o_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == 'O')
        features[17] = min(o_neighbors, 3) / 3.0
        
        # 18. Hydrogen count (saturation indicator)
        h_count = atom.GetTotalNumHs()
        features[18] = min(h_count, 4) / 4.0
        
        # 19. Local oxidation indicator (C bonded to O)
        if atom.GetSymbol() == 'C':
            double_bonded_o = sum(1 for bond in atom.GetBonds() 
                                  if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
                                  and bond.GetOtherAtom(atom).GetSymbol() == 'O')
            features[19] = min(double_bonded_o, 2) / 2.0
        
    except Exception as e:
        pass # Return zeros on error
    
    return np.array(features, dtype=np.float32)


def get_cram_molecule_features(mol):
    """
    Extract molecule-level CRAM features for guidance.
    
    These are used to condition the generation process.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        dict with molecule-level CRAM features
    """
    features = {}
    
    try:
        # Count carboxyl groups
        carboxyl_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
        carboxyl_matches = mol.GetSubstructMatches(carboxyl_pattern)
        features['num_carboxyl'] = len(carboxyl_matches)
        features['is_cram_carboxyl'] = len(carboxyl_matches) >= 2
        
        # Count alicyclic rings
        ring_info = mol.GetRingInfo()
        alicyclic_count = 0
        aromatic_count = 0
        for ring in ring_info.AtomRings():
            ring_aromatic = any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
            if ring_aromatic:
                aromatic_count += 1
            else:
                alicyclic_count += 1
        
        features['num_alicyclic'] = alicyclic_count
        features['num_aromatic'] = aromatic_count
        features['is_cram_alicyclic'] = alicyclic_count > 0 and aromatic_count == 0
        
        # Count junction atoms (ring-to-chain connections)
        junction_count = 0
        for atom in mol.GetAtoms():
            if atom.IsInRing():
                has_chain_neighbor = any(not n.IsInRing() for n in atom.GetNeighbors())
                if has_chain_neighbor:
                    junction_count += 1
        features['num_junctions'] = junction_count
        features['has_ring_chain_hybrid'] = alicyclic_count > 0 and junction_count > 0
        
        # Carbon type analysis
        total_c = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C')
        aromatic_c = sum(1 for a in mol.GetAtoms() 
                        if a.GetSymbol() == 'C' and a.GetIsAromatic())
        aliphatic_c = total_c - aromatic_c
        
        features['total_carbons'] = total_c
        features['aromatic_carbons'] = aromatic_c
        features['aliphatic_carbons'] = aliphatic_c
        features['aliphatic_ratio'] = aliphatic_c / max(total_c, 1)
        features['is_cram_aliphatic'] = aliphatic_c / max(total_c, 1) >= 0.7
        
        # Elemental ratios
        num_o = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'O')
        num_h = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'H') + \
                sum(a.GetTotalNumHs() for a in mol.GetAtoms())
        
        features['o_c_ratio'] = num_o / max(total_c, 1)
        features['h_c_ratio'] = num_h / max(total_c, 1)
        features['is_cram_marine_dom'] = (0.4 <= features['o_c_ratio'] <= 0.8 and 
                                           1.2 <= features['h_c_ratio'] <= 2.0)
        
        # Overall CRAM score
        cram_score = 0
        if features['is_cram_carboxyl']:
            cram_score += 0.3
        if features['is_cram_alicyclic']:
            cram_score += 0.2
        if features['is_cram_aliphatic']:
            cram_score += 0.2
        if features['has_ring_chain_hybrid']:
            cram_score += 0.15
        if features['is_cram_marine_dom']:
            cram_score += 0.15
        
        features['cram_score'] = cram_score
        features['is_true_cram'] = cram_score >= 0.7
        
    except Exception as e:
        features['error'] = str(e)
    
    return features


# Test the new features


# ============================================================================
# ENHANCED NODE FEATURES WITH CRAM RING-CHAIN (108D)
# ============================================================================
# 
# This extends the node feature extraction to include CRAM-specific features,
# making the model explicitly aware of cyclic+chain hybrid structures.
#
# Feature breakdown:
# - Base features (45D): Atom type, degree, hybridization, etc.
# - Synthetic features (28D): C3/C4 analysis, stereochemistry
# - Marine features (15D): Oxidation, degradation markers
# - CRAM Ring-Chain features (20D): Alicyclic rings, junctions, carboxyl proximity
#
# Total: 108D per atom (vs. 88D before)
# ============================================================================

_original_get_enhanced_node_features = get_enhanced_node_features


def get_enhanced_node_features_with_cram(mol, mode='unified'):
    """
    Extract 108D enhanced node features with CRAM ring-chain awareness.
    
    This is the upgraded version that includes explicit CRAM structure features
    for better capture of cyclic+chain hybrid molecules.
    
    Feature dimensions:
    - synthetic mode: 93D (45 base + 28 synthetic + 20 CRAM)
    - marine/unified mode: 108D (45 base + 28 synthetic + 15 marine + 20 CRAM)
    
    Args:
        mol: RDKit molecule object
        mode: 'synthetic', 'marine', or 'unified'
        
    Returns:
        List of feature arrays (one per atom)
    """
    if mol is None:
        return None
    
    try:
        # Get base features (45D)
        base_features = get_base_node_features_45D(mol)
        if not base_features:
            return None
        
        enhanced_features = []
        
        for atom_idx in range(mol.GetNumAtoms()):
            # Base features
            base = base_features[atom_idx]
            
            # Synthetic features (28D)
            synthetic = get_synthetic_features_28D(mol, atom_idx)
            
            # CRAM Ring-Chain features (20D) - NEW!
            cram_rc = get_cram_ring_chain_features_20D(mol, atom_idx)
            
            if mode == 'synthetic':
                # 45 + 28 + 20 = 93D
                combined = np.concatenate([base, synthetic, cram_rc])
            else: # marine or unified
                # Marine features (15D)
                marine = get_marine_features_15D(mol, atom_idx)
                # 45 + 28 + 15 + 20 = 108D
                combined = np.concatenate([base, synthetic, marine, cram_rc])
            
            enhanced_features.append(combined)
        
        return enhanced_features
        
    except Exception as e:
        print(f"Warning: Error in get_enhanced_node_features_with_cram: {e}")
        # Fallback to original
        return _original_get_enhanced_node_features(mol, mode)


def get_node_feature_dim(mode='unified'):
    """
    Get the node feature dimension for a given mode.
    
    Args:
        mode: 'synthetic', 'marine', or 'unified'
        
    Returns:
        int: Feature dimension
    """
    if mode == 'synthetic':
        return 93 # 45 + 28 + 20
    else: # marine or unified
        return 108 # 45 + 28 + 15 + 20


# Override the default function
get_enhanced_node_features = get_enhanced_node_features_with_cram

