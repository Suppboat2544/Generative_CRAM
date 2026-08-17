"""
=============================================================================
CRITICAL IMPROVEMENT 1: STEREOCHEMISTRY-AWARE NODE FEATURES (12D)
=============================================================================

This module adds stereochemistry awareness that is FUNDAMENTAL to CRAM chemistry:
- Chiral centers affect molecular recognition and stability
- Ring conformations determine COOH orientation and reactivity
- cis/trans arrangements affect packing and aggregation behavior
- These features help explain why some CRAM structures persist for millennia
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def get_stereochemistry_features_12D(mol: Chem.Mol) -> np.ndarray:
    """
    Extract 12D stereochemistry features per atom.
    
    Chemistry rationale:
    - Marine CRAM often has specific stereochemistry from biosynthesis
    - Terpenoid-derived CRAM preserves stereochemistry from isoprene units
    - Ring fusion stereochemistry (cis/trans) affects stability
    
    Features (12D per atom):
    [0-2]   Chiral center features (is_chiral, is_R, is_S)
    [3-5]   Ring position features (axial_like, equatorial_like, ring_junction)
    [6-8]   Geometric features (in_cis_ring, in_trans_ring, ring_strain_proxy)
    [9-11]  Conformational capacity (can_rotate, sp3_character, h_bond_geometry)
    """
    num_atoms = mol.GetNumAtoms()
    features = np.zeros((num_atoms, 12), dtype=np.float32)
    
    # Try to assign stereochemistry from SMILES
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except:
        pass
    
    # Get ring information
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    
    # Build ring membership map
    atom_to_rings = {i: [] for i in range(num_atoms)}
    for ring_idx, ring in enumerate(atom_rings):
        for atom_idx in ring:
            atom_to_rings[atom_idx].append(ring_idx)
    
    # Get chiral centers
    chiral_centers = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))
    
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        # ===== Chiral center features (0-2) =====
        if atom_idx in chiral_centers:
            features[atom_idx, 0] = 1.0  # is_chiral
            chirality = chiral_centers[atom_idx]
            if chirality == 'R':
                features[atom_idx, 1] = 1.0  # is_R
            elif chirality == 'S':
                features[atom_idx, 2] = 1.0  # is_S
            # For '?' (unassigned), both stay 0
        
        # ===== Ring position features (3-5) =====
        rings = atom_to_rings[atom_idx]
        if rings:
            # Heuristic for axial/equatorial based on connectivity pattern
            # In real 3D, this would use conformer analysis
            neighbors_in_ring = sum(1 for n in atom.GetNeighbors() 
                                   if any(n.GetIdx() in atom_rings[r] for r in rings))
            neighbors_total = atom.GetDegree()
            
            # Atoms with substituents pointing "up/down" vs "sideways" 
            if neighbors_total > 2 and atom.GetHybridization() == Chem.HybridizationType.SP3:
                # Approximate: more in-ring neighbors = more equatorial-like substituents
                features[atom_idx, 3] = 1.0 - (neighbors_in_ring / neighbors_total)  # axial_like
                features[atom_idx, 4] = neighbors_in_ring / neighbors_total  # equatorial_like
            
            # Ring junction detection
            if len(rings) > 1:
                features[atom_idx, 5] = 1.0  # ring_junction
        
        # ===== Geometric/Ring strain features (6-8) =====
        for ring_idx in rings:
            ring = atom_rings[ring_idx]
            ring_size = len(ring)
            
            # Small rings (3-4) have high strain
            if ring_size <= 4:
                features[atom_idx, 8] = max(features[atom_idx, 8], 1.0)  # high strain
            elif ring_size == 5:
                features[atom_idx, 8] = max(features[atom_idx, 8], 0.3)  # moderate strain
            # 6-membered rings: check if chair-like (low strain) or boat-like (higher)
            elif ring_size == 6:
                # Heuristic: all sp3 in 6-ring suggests chair conformation
                ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
                sp3_count = sum(1 for a in ring_atoms if a.GetHybridization() == Chem.HybridizationType.SP3)
                if sp3_count == 6:  # Cyclohexane-like
                    features[atom_idx, 6] = 0.5  # Assume trans (chair-like) by default
                    features[atom_idx, 8] = max(features[atom_idx, 8], 0.05)  # Low strain
            elif ring_size >= 7:
                features[atom_idx, 8] = max(features[atom_idx, 8], 0.2)  # Medium rings
        
        # ===== Conformational capacity features (9-11) =====
        # Rotatable bonds attached to this atom
        rotatable = 0
        for bond in atom.GetBonds():
            if not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE:
                other = bond.GetOtherAtom(atom)
                if other.GetAtomicNum() > 1:  # Not H
                    rotatable += 1
        features[atom_idx, 9] = min(rotatable / 3.0, 1.0)  # Normalized rotation capacity
        
        # sp3 character (important for conformational flexibility)
        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            features[atom_idx, 10] = 1.0
        elif atom.GetHybridization() == Chem.HybridizationType.SP2:
            features[atom_idx, 10] = 0.5
        
        # H-bond geometry potential
        # Atoms that can form H-bonds with specific geometry requirements
        if atom.GetAtomicNum() in [7, 8] or (atom.GetAtomicNum() == 6 and 
                                              any(n.GetAtomicNum() in [7, 8] for n in atom.GetNeighbors())):
            features[atom_idx, 11] = 0.5
            # Enhance if proper geometry for COOH
            if atom.GetAtomicNum() == 8:
                c_neighbor = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 6]
                if c_neighbor:
                    for c in c_neighbor:
                        o_neighbors = sum(1 for n in c.GetNeighbors() if n.GetAtomicNum() == 8)
                        if o_neighbors >= 2:  # Part of COOH/COO-
                            features[atom_idx, 11] = 1.0
    
    return features


def get_conformational_stability_score(mol: Chem.Mol) -> float:
    """
    Estimate conformational stability relevant to CRAM persistence.
    
    High scores indicate:
    - Chair-preferred cyclohexanes
    - trans-fused ring systems
    - Equatorially-positioned carboxyls
    - Minimal ring strain
    
    Returns: score in [0, 1] where higher = more stable
    """
    score = 0.5  # Baseline
    
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()
    
    # Count 6-membered saturated rings (cyclohexane-like) - very stable
    for ring in rings:
        if len(ring) == 6:
            ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
            if all(a.GetHybridization() == Chem.HybridizationType.SP3 for a in ring_atoms):
                score += 0.1  # Boost for chair-preferred rings
    
    # Penalize small rings (high strain)
    small_rings = sum(1 for ring in rings if len(ring) <= 4)
    score -= small_rings * 0.15
    
    # Check for trans-fused rings (more stable than cis)
    # Heuristic: ring junctions with sp3 centers and even number of junction atoms
    junction_atoms = set()
    for i, ring1 in enumerate(rings):
        for j, ring2 in enumerate(rings[i+1:], i+1):
            common = set(ring1) & set(ring2)
            junction_atoms.update(common)
    
    if len(junction_atoms) >= 2:
        # Even number of junction atoms often indicates trans fusion
        if len(junction_atoms) % 2 == 0:
            score += 0.1
    
    return np.clip(score, 0.0, 1.0)


def get_3d_conformer_features_8D(mol: Chem.Mol, atom_idx: int, 
                                  conformer: Optional[Chem.Conformer] = None) -> np.ndarray:
    """
    Extract 3D-derived features if conformer is available.
    
    Features (8D):
    [0-2]   Local geometry (planarity, tetrahedrality, linearity)
    [3-5]   Spatial exposure (accessible, buried, edge)
    [6-7]   H-bond geometry (donor_angle_ok, acceptor_angle_ok)
    """
    features = np.zeros(8, dtype=np.float32)
    
    if conformer is None:
        # Try to generate 3D conformer
        mol_3d = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol_3d, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol_3d)
            conformer = mol_3d.GetConformer()
        except:
            return features  # Return zeros if 3D generation fails
    
    try:
        pos = conformer.GetAtomPosition(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        neighbors = atom.GetNeighbors()
        
        if len(neighbors) >= 3:
            # Calculate local geometry
            neighbor_positions = [conformer.GetAtomPosition(n.GetIdx()) for n in neighbors[:3]]
            
            # Vector from atom to neighbors
            v1 = np.array([neighbor_positions[0].x - pos.x, 
                          neighbor_positions[0].y - pos.y,
                          neighbor_positions[0].z - pos.z])
            v2 = np.array([neighbor_positions[1].x - pos.x,
                          neighbor_positions[1].y - pos.y,
                          neighbor_positions[1].z - pos.z])
            v3 = np.array([neighbor_positions[2].x - pos.x,
                          neighbor_positions[2].y - pos.y,
                          neighbor_positions[2].z - pos.z])
            
            # Planarity: how coplanar are the neighbors?
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-6:
                normal = normal / normal_norm
                planarity = abs(np.dot(normal, v3 / np.linalg.norm(v3)))
                features[0] = 1.0 - planarity  # Higher for more planar
            
            # Tetrahedrality: sp3-like geometry
            if len(neighbors) == 4:
                features[1] = 1.0  # Perfect tetrahedral opportunity
        
        # Spatial exposure (approximate by neighbor count and distance)
        num_neighbors = len(neighbors)
        features[3] = 1.0 - min(num_neighbors / 4.0, 1.0)  # accessible if fewer neighbors
        features[4] = min(num_neighbors / 4.0, 1.0)  # buried if more neighbors
        
    except:
        pass
    
    return features
