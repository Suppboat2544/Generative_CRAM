"""
CRAM Molecule Validator

Validates SMILES strings against CRAM (Carboxyl-Rich Alicyclic Molecules) criteria:
- ≥2 carboxyl groups (-COOH)
- Alicyclic ring structure
- ≥60% aliphatic carbon
- Molecular weight: 100-750 Da
"""

using MolecularGraph
using Graphs

"""
    validate_cram(smiles::String) -> ValidationResult

Validate a SMILES string against CRAM criteria.

# Arguments
- `smiles`: SMILES string to validate

# Returns
- `ValidationResult` containing validation details

# Example
```julia
result = validate_cram("CC1CCC(CC1)(C(=O)O)C(=O)O")
println(result.is_cram)  # true
println(result.n_cooh)   # 2
```
"""
function validate_cram(smiles::String)::ValidationResult
    # Parse SMILES
    mol = try
        smilestomol(smiles)
    catch
        return ValidationResult(smiles)
    end
    
    if isnothing(mol)
        return ValidationResult(smiles)
    end
    
    # Get canonical SMILES
    canonical = try
        canonicalize(mol) |> tosmiles
    catch
        smiles
    end
    
    # Count atoms
    n_atoms = nodecount(mol)
    
    # Calculate molecular weight
    mw = calculate_mw(mol)
    
    # Count carboxyl groups
    n_cooh = count_carboxyl_groups(mol)
    
    # Check for alicyclic rings
    has_alicyclic = has_alicyclic_ring(mol)
    
    # Calculate aliphatic percentage
    aliphatic_pct = calculate_aliphatic_percentage(mol)
    
    # Count rings
    n_rings = circuitrank(mol)
    
    # Count aromatic atoms
    aromatic_atoms = count_aromatic_atoms(mol)
    
    # Check CRAM criteria
    is_cram = (
        n_cooh >= CRAM_CRITERIA.min_cooh &&
        has_alicyclic &&
        aliphatic_pct >= CRAM_CRITERIA.min_aliphatic_pct &&
        mw >= CRAM_CRITERIA.min_mw &&
        mw <= CRAM_CRITERIA.max_mw
    )
    
    return ValidationResult(
        smiles,
        canonical,
        true,  # is_valid
        is_cram,
        n_cooh,
        has_alicyclic,
        aliphatic_pct,
        mw,
        n_atoms,
        n_rings,
        aromatic_atoms
    )
end

"""
    calculate_mw(mol) -> Float64

Calculate molecular weight from MolecularGraph molecule.
"""
function calculate_mw(mol)::Float64
    mw = 0.0
    for i in 1:nodecount(mol)
        atom = getatom(mol, i)
        sym = Symbol(atomsymbol(atom))
        mw += get(ATOMIC_MASSES, sym, 0.0)
    end
    
    # Add implicit hydrogens
    for i in 1:nodecount(mol)
        atom = getatom(mol, i)
        n_h = implicithcount(mol, i)
        mw += n_h * ATOMIC_MASSES[:H]
    end
    
    return mw
end

"""
    count_carboxyl_groups(mol) -> Int

Count the number of carboxyl groups (-COOH) in a molecule.
Uses SMARTS pattern matching.
"""
function count_carboxyl_groups(mol)::Int
    # SMARTS pattern for carboxyl group: C(=O)O or C(=O)[OH]
    carboxyl_pattern = try
        smartstomol("[CX3](=O)[OX2H1]")
    catch
        return 0
    end
    
    matches = try
        substructmatches(mol, carboxyl_pattern)
    catch
        return 0
    end
    
    return length(matches)
end

"""
    has_alicyclic_ring(mol) -> Bool

Check if molecule contains an alicyclic (non-aromatic) ring.
"""
function has_alicyclic_ring(mol)::Bool
    # Get ring information
    rings = try
        sssr(mol)
    catch
        return false
    end
    
    if isempty(rings)
        return false
    end
    
    # Check each ring for non-aromaticity
    aromatic_atoms = try
        isaromatic(mol)
    catch
        return true  # If we can't determine, assume alicyclic exists
    end
    
    for ring in rings
        # Ring is alicyclic if not all atoms are aromatic
        ring_atoms = collect(ring)
        is_aromatic_ring = all(aromatic_atoms[a] for a in ring_atoms)
        if !is_aromatic_ring
            return true
        end
    end
    
    return false
end

"""
    calculate_aliphatic_percentage(mol) -> Float64

Calculate the percentage of carbons that are aliphatic (sp3).
"""
function calculate_aliphatic_percentage(mol)::Float64
    total_carbons = 0
    aliphatic_carbons = 0
    
    for i in 1:nodecount(mol)
        atom = getatom(mol, i)
        if atomsymbol(atom) == "C"
            total_carbons += 1
            
            # Check if sp3 hybridized (aliphatic)
            # sp3 carbons have 4 single bonds (counting implicit H)
            degree = nodedegree(mol, i)
            n_h = implicithcount(mol, i)
            
            # Check aromaticity
            is_arom = try
                isaromatic(mol)[i]
            catch
                false
            end
            
            if !is_arom && (degree + n_h == 4)
                aliphatic_carbons += 1
            end
        end
    end
    
    return total_carbons > 0 ? aliphatic_carbons / total_carbons : 0.0
end

"""
    count_aromatic_atoms(mol) -> Int

Count the number of aromatic atoms in the molecule.
"""
function count_aromatic_atoms(mol)::Int
    aromatic = try
        isaromatic(mol)
    catch
        return 0
    end
    
    return sum(aromatic)
end

"""
    batch_validate(smiles_list::Vector{String}) -> Vector{ValidationResult}

Validate multiple SMILES strings.

# Arguments
- `smiles_list`: Vector of SMILES strings

# Returns
- Vector of ValidationResult objects
"""
function batch_validate(smiles_list::Vector{String})::Vector{ValidationResult}
    return [validate_cram(s) for s in smiles_list]
end

"""
    batch_validate(input_file::String; output_file::Union{String,Nothing}=nothing) -> DataFrame

Validate SMILES from a file and optionally save results.

# Arguments
- `input_file`: Path to file containing SMILES (one per line)
- `output_file`: Optional path to save CSV results

# Returns
- DataFrame with validation results
"""
function batch_validate(input_file::String; output_file::Union{String,Nothing}=nothing)::DataFrame
    # Read SMILES from file
    smiles_list = String[]
    open(input_file, "r") do f
        for line in eachline(f)
            s = strip(line)
            if !isempty(s) && !startswith(s, "#")
                push!(smiles_list, s)
            end
        end
    end
    
    # Validate all
    results = batch_validate(smiles_list)
    
    # Convert to DataFrame
    df = DataFrame(
        smiles = [r.smiles for r in results],
        canonical = [r.canonical for r in results],
        is_valid = [r.is_valid for r in results],
        is_cram = [r.is_cram for r in results],
        n_cooh = [r.n_cooh for r in results],
        has_alicyclic = [r.has_alicyclic for r in results],
        aliphatic_pct = [r.aliphatic_pct for r in results],
        mw = [r.mw for r in results],
        n_atoms = [r.n_atoms for r in results],
        n_rings = [r.n_rings for r in results]
    )
    
    # Save if output file specified
    if !isnothing(output_file)
        CSV.write(output_file, df)
    end
    
    return df
end
