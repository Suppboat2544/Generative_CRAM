"""
163-Dimensional Feature Extractor for CRAM Molecules

Feature Groups (163 total):
- Base features (1-45): Atom types, hybridization, basic properties
- Ring-Chain features (46-65): Ring system analysis  
- Stereochemistry (66-77): Chiral centers, E/Z configuration
- Marine DOM (78-120): Ocean-specific molecular patterns
- Terpenoid (121-130): Terpene-like structural features
- Functional groups (131-143): Key functional group counts
- Oxidation state (144-163): Carbon oxidation patterns
"""

using MolecularGraph
using LinearAlgebra
using Statistics

# Feature dimension constants
const FEATURE_DIMS = (
    base = 1:45,
    ring_chain = 46:65,
    stereo = 66:77,
    marine_dom = 78:120,
    terpenoid = 121:130,
    functional = 131:143,
    oxidation = 144:163
)

const TOTAL_FEATURES = 163

# Atom type one-hot encoding order
const ATOM_TYPES = [:C, :H, :O, :N, :S, :P, :F, :Cl, :Br, :I]

"""
    extract_features(smiles::String) -> FeatureVector

Extract 163-dimensional molecular features from SMILES.

# Arguments
- `smiles`: SMILES string

# Returns
- `FeatureVector` with 163-dimensional feature vector

# Example
```julia
fv = extract_features("CC1CCC(CC1)(C(=O)O)C(=O)O")
println(length(fv.features))  # 163
```
"""
function extract_features(smiles::String)::FeatureVector
    mol = try
        smilestomol(smiles)
    catch
        return FeatureVector(zeros(TOTAL_FEATURES))
    end
    
    return extract_features(mol)
end

"""
    extract_features(mol) -> FeatureVector

Extract features from MolecularGraph molecule object.
"""
function extract_features(mol)::FeatureVector
    n_atoms = nodecount(mol)
    
    if n_atoms == 0
        return FeatureVector(zeros(TOTAL_FEATURES))
    end
    
    # Extract per-atom features
    atom_features = zeros(n_atoms, TOTAL_FEATURES)
    
    for i in 1:n_atoms
        atom_features[i, :] = extract_atom_features(mol, i)
    end
    
    # Aggregate to molecule-level features (mean pooling)
    mol_features = vec(mean(atom_features, dims=1))
    
    # Add molecule-level features
    add_molecule_features!(mol_features, mol)
    
    return FeatureVector(mol_features, atom_features)
end

"""
    extract_atom_features(mol, atom_idx::Int) -> Vector{Float64}

Extract 163-dimensional features for a single atom.
"""
function extract_atom_features(mol, atom_idx::Int)::Vector{Float64}
    features = zeros(TOTAL_FEATURES)
    atom = getatom(mol, atom_idx)
    
    # === BASE FEATURES (1-45) ===
    # Atom type one-hot (1-10)
    sym = Symbol(atomsymbol(atom))
    for (j, atype) in enumerate(ATOM_TYPES)
        if sym == atype
            features[j] = 1.0
            break
        end
    end
    
    # Degree (11-15) - one-hot up to 4+
    degree = nodedegree(mol, atom_idx)
    deg_idx = min(degree, 4) + 1
    features[10 + deg_idx] = 1.0
    
    # Implicit hydrogens (16-20) - one-hot up to 4+
    n_h = implicithcount(mol, atom_idx)
    h_idx = min(n_h, 4) + 1
    features[15 + h_idx] = 1.0
    
    # Formal charge (21-25) - encoded as [-2, -1, 0, +1, +2]
    charge = try
        getcharge(atom)
    catch
        0
    end
    charge_idx = clamp(charge + 3, 1, 5)
    features[20 + charge_idx] = 1.0
    
    # Aromaticity (26)
    is_arom = try
        isaromatic(mol)[atom_idx]
    catch
        false
    end
    features[26] = is_arom ? 1.0 : 0.0
    
    # In ring (27)
    in_ring = try
        isringatom(mol)[atom_idx]
    catch
        false
    end
    features[27] = in_ring ? 1.0 : 0.0
    
    # Hybridization approximation (28-31): sp, sp2, sp3, other
    features[28:31] = estimate_hybridization(mol, atom_idx, degree, n_h, is_arom)
    
    # Electronegativity (32) - normalized
    en = get(Dict(:C => 2.55, :H => 2.20, :O => 3.44, :N => 3.04, :S => 2.58), sym, 2.5)
    features[32] = (en - 2.0) / 2.0  # Normalize to ~[0, 1]
    
    # Atomic mass (33) - normalized
    mass = get(ATOMIC_MASSES, sym, 12.0)
    features[33] = mass / 100.0
    
    # Valence electrons (34-38)
    valence = get(Dict(:C => 4, :H => 1, :O => 6, :N => 5, :S => 6), sym, 4)
    features[33 + valence] = 1.0
    
    # Is hydrogen bond donor/acceptor (39-40)
    features[39] = (sym == :O || sym == :N) && n_h > 0 ? 1.0 : 0.0  # Donor
    features[40] = (sym == :O || sym == :N) ? 1.0 : 0.0  # Acceptor
    
    # Additional base (41-45) - reserved
    
    # === RING-CHAIN FEATURES (46-65) ===
    features[46:65] = extract_ring_features(mol, atom_idx, in_ring)
    
    # === STEREOCHEMISTRY (66-77) ===
    features[66:77] = extract_stereo_features(mol, atom_idx)
    
    # === MARINE DOM FEATURES (78-120) ===
    features[78:120] = extract_marine_features(mol, atom_idx, sym)
    
    # === TERPENOID FEATURES (121-130) ===
    features[121:130] = extract_terpenoid_features(mol, atom_idx)
    
    # === FUNCTIONAL GROUP FEATURES (131-143) ===
    features[131:143] = extract_functional_features(mol, atom_idx, sym)
    
    # === OXIDATION STATE FEATURES (144-163) ===
    features[144:163] = extract_oxidation_features(mol, atom_idx, sym, degree, n_h)
    
    return features
end

"""
Estimate hybridization state from connectivity.
"""
function estimate_hybridization(mol, idx, degree, n_h, is_aromatic)::Vector{Float64}
    hyb = zeros(4)
    
    total_bonds = degree + n_h
    
    if is_aromatic
        hyb[2] = 1.0  # sp2
    elseif total_bonds == 4
        hyb[3] = 1.0  # sp3
    elseif total_bonds == 3
        hyb[2] = 1.0  # sp2
    elseif total_bonds == 2
        hyb[1] = 1.0  # sp
    else
        hyb[4] = 1.0  # other
    end
    
    return hyb
end

"""
Extract ring-related features.
"""
function extract_ring_features(mol, atom_idx, in_ring)::Vector{Float64}
    features = zeros(20)
    
    if !in_ring
        return features
    end
    
    # Ring membership (1-6): 3, 4, 5, 6, 7, 8+ membered
    rings = try
        sssr(mol)
    catch
        return features
    end
    
    for ring in rings
        ring_atoms = collect(ring)
        if atom_idx in ring_atoms
            ring_size = length(ring_atoms)
            size_idx = clamp(ring_size - 2, 1, 6)
            features[size_idx] = 1.0
        end
    end
    
    # Number of rings containing this atom (7)
    ring_count = sum(atom_idx in collect(r) for r in rings)
    features[7] = min(ring_count, 4) / 4.0
    
    # Is bridgehead (8) - simplified detection
    features[8] = ring_count > 1 ? 1.0 : 0.0
    
    # Ring is alicyclic (9)
    aromatic = try
        isaromatic(mol)
    catch
        return features
    end
    
    for ring in rings
        ring_atoms = collect(ring)
        if atom_idx in ring_atoms
            is_alicyclic = !all(aromatic[a] for a in ring_atoms)
            if is_alicyclic
                features[9] = 1.0
                break
            end
        end
    end
    
    # Fused ring system (10)
    features[10] = ring_count >= 2 ? 1.0 : 0.0
    
    # Reserved (11-20)
    
    return features
end

"""
Extract stereochemistry features.
"""
function extract_stereo_features(mol, atom_idx)::Vector{Float64}
    features = zeros(12)
    
    # Potential chiral center (has 4 different substituents)
    degree = nodedegree(mol, atom_idx)
    if degree >= 3
        features[1] = 1.0  # Potential stereocenter
    end
    
    # Reserved for actual stereochemistry parsing (2-12)
    
    return features
end

"""
Extract marine DOM-specific features.
"""
function extract_marine_features(mol, atom_idx, sym)::Vector{Float64}
    features = zeros(43)
    
    # Carboxyl carbon (1)
    if sym == :C
        neighbors = neighbors_atoms(mol, atom_idx)
        has_double_o = false
        has_single_o = false
        
        for n in neighbors
            n_atom = getatom(mol, n)
            n_sym = Symbol(atomsymbol(n_atom))
            if n_sym == :O
                # Simplified: count oxygen neighbors
                bond = try
                    getbond(mol, atom_idx, n)
                catch
                    continue
                end
                if bondorder(bond) == 2
                    has_double_o = true
                else
                    has_single_o = true
                end
            end
        end
        
        if has_double_o && has_single_o
            features[1] = 1.0  # Carboxyl carbon
        end
    end
    
    # Carboxyl oxygen (2-3)
    if sym == :O
        features[2] = 1.0  # Oxygen (potential carboxyl)
    end
    
    # Aliphatic carbon chain (4-10)
    if sym == :C
        degree = nodedegree(mol, atom_idx)
        n_h = implicithcount(mol, atom_idx)
        is_arom = try
            isaromatic(mol)[atom_idx]
        catch
            false
        end
        
        if !is_arom && (degree + n_h == 4)
            features[4] = 1.0  # Aliphatic carbon
            
            # Chain length estimation based on neighbors
            features[5] = min(degree, 4) / 4.0
        end
    end
    
    # Methyl group (11)
    if sym == :C
        n_h = implicithcount(mol, atom_idx)
        degree = nodedegree(mol, atom_idx)
        if n_h == 3 && degree == 1
            features[11] = 1.0  # Methyl
        end
    end
    
    # Hydroxyl oxygen (12)
    if sym == :O
        n_h = implicithcount(mol, atom_idx)
        if n_h >= 1
            features[12] = 1.0  # Hydroxyl
        end
    end
    
    # Reserved (13-43) for additional marine-specific patterns
    
    return features
end

"""
Extract terpenoid-like features.
"""
function extract_terpenoid_features(mol, atom_idx)::Vector{Float64}
    features = zeros(10)
    
    # Isoprene unit detection (C5 pattern)
    # Simplified: look for branched carbon patterns
    
    degree = nodedegree(mol, atom_idx)
    if degree >= 3
        features[1] = 1.0  # Branching point
    end
    
    # Methyl branch (2)
    neighbors = neighbors_atoms(mol, atom_idx)
    for n in neighbors
        n_atom = getatom(mol, n)
        if atomsymbol(n_atom) == "C"
            n_degree = nodedegree(mol, n)
            n_h = implicithcount(mol, n)
            if n_degree == 1 && n_h == 3
                features[2] = 1.0  # Has methyl branch
                break
            end
        end
    end
    
    # Reserved (3-10)
    
    return features
end

"""
Extract functional group features.
"""
function extract_functional_features(mol, atom_idx, sym)::Vector{Float64}
    features = zeros(13)
    
    # Carbonyl carbon (1)
    if sym == :C
        for n in neighbors_atoms(mol, atom_idx)
            n_atom = getatom(mol, n)
            if atomsymbol(n_atom) == "O"
                bond = try
                    getbond(mol, atom_idx, n)
                catch
                    continue
                end
                if bondorder(bond) == 2
                    features[1] = 1.0  # Carbonyl
                    break
                end
            end
        end
    end
    
    # Ether oxygen (2)
    if sym == :O
        degree = nodedegree(mol, atom_idx)
        n_h = implicithcount(mol, atom_idx)
        if degree == 2 && n_h == 0
            features[2] = 1.0  # Ether
        end
    end
    
    # Ester linkage (3)
    if sym == :O
        neighbors = neighbors_atoms(mol, atom_idx)
        for n in neighbors
            n_atom = getatom(mol, n)
            if atomsymbol(n_atom) == "C"
                # Check if this C has =O
                for nn in neighbors_atoms(mol, n)
                    if nn != atom_idx
                        nn_atom = getatom(mol, nn)
                        if atomsymbol(nn_atom) == "O"
                            bond = try
                                getbond(mol, n, nn)
                            catch
                                continue
                            end
                            if bondorder(bond) == 2
                                features[3] = 1.0  # Ester
                                break
                            end
                        end
                    end
                end
            end
        end
    end
    
    # Reserved (4-13)
    
    return features
end

"""
Extract oxidation state features.
"""
function extract_oxidation_features(mol, atom_idx, sym, degree, n_h)::Vector{Float64}
    features = zeros(20)
    
    if sym != :C
        return features
    end
    
    # Estimate carbon oxidation state based on attached atoms
    o_count = 0
    n_count = 0
    
    for n in neighbors_atoms(mol, atom_idx)
        n_atom = getatom(mol, n)
        n_sym = atomsymbol(n_atom)
        if n_sym == "O"
            o_count += 1
        elseif n_sym == "N"
            n_count += 1
        end
    end
    
    # Oxidation level encoding (1-6)
    # 0 = no heteroatoms, up to 5 = highly oxidized
    ox_level = min(o_count + n_count, 5)
    features[ox_level + 1] = 1.0
    
    # Hydroxyl-bearing carbon (7)
    for n in neighbors_atoms(mol, atom_idx)
        n_atom = getatom(mol, n)
        if atomsymbol(n_atom) == "O"
            n_h_o = implicithcount(mol, n)
            if n_h_o >= 1
                features[7] = 1.0
                break
            end
        end
    end
    
    # Carbonyl carbon (8)
    for n in neighbors_atoms(mol, atom_idx)
        n_atom = getatom(mol, n)
        if atomsymbol(n_atom) == "O"
            bond = try
                getbond(mol, atom_idx, n)
            catch
                continue
            end
            if bondorder(bond) == 2
                features[8] = 1.0
                break
            end
        end
    end
    
    # Reserved (9-20)
    
    return features
end

"""
Get neighbor atom indices.
"""
function neighbors_atoms(mol, atom_idx)::Vector{Int}
    return collect(adjacencies(mol, atom_idx))
end

"""
Add molecule-level aggregate features.
"""
function add_molecule_features!(features::Vector{Float64}, mol)
    # These are added to/modify the aggregated features
    
    n_atoms = nodecount(mol)
    
    # Normalize some features by molecule size
    if n_atoms > 0
        # Feature 33 (mass) - update to total mass
        features[33] = calculate_mw(mol) / 750.0  # Normalize by max CRAM MW
    end
    
    # Add ring count information
    n_rings = circuitrank(mol)
    features[46] = min(n_rings, 5) / 5.0
end
