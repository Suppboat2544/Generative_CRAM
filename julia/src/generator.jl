"""
CRAM Molecule Generator

Mutation-based generation of CRAM molecules using SMILES manipulation.
"""

using Random
using ProgressMeter

# Default seed molecules (known CRAM structures)
const DEFAULT_SEEDS = [
    "CC1CCC(CC1)(C(=O)O)C(=O)O",           # 1,1-cyclohexanedicarboxylic acid derivative
    "OC(=O)C1CCCCC1C(=O)O",                 # cyclohexane-1,2-dicarboxylic acid
    "CC(C)C1CCC(CC1)(C(=O)O)C(=O)O",        # isopropyl cyclohexane dicarboxylic acid
    "OC(=O)C1CCC(CC1)C(=O)O",               # cyclohexane-1,4-dicarboxylic acid
    "CC1(C)CCC(CC1)(C(=O)O)C(=O)O",         # dimethyl cyclohexane dicarboxylic acid
    "OC(=O)CC1CCCCC1C(=O)O",                # cyclohexane with acetic acid
    "CC(C)(C(=O)O)C1CCCCC1C(=O)O",          # branched cyclohexane dicarboxylic
]

# SMILES tokens for mutation
const SMILES_TOKENS = [
    # Atoms
    "C", "O", "N", "S", "c", "n", "o", "s",
    # Bonds  
    "=", "#", "-",
    # Branches
    "(", ")",
    # Rings
    "1", "2", "3", "4", "5", "6",
    # Groups
    "CC", "CCC", "CCCC",
    "C(=O)O",   # Carboxyl
    "C(=O)",    # Carbonyl
    "CO",       # Methoxy
    "O",        # Hydroxyl
]

# Insertable fragments for CRAM generation
const CRAM_FRAGMENTS = [
    "C(=O)O",       # Carboxyl group
    "CC",           # Ethyl
    "CCC",          # Propyl
    "C(C)C",        # Isopropyl
    "C",            # Methyl
    "O",            # Hydroxyl when attached
    "CC(C)",        # Isobutyl
]

"""
    generate_cram(; n::Int=100, config::GenerationConfig=GenerationConfig()) -> Vector{String}

Generate CRAM molecules using mutation-based approach.

# Arguments
- `n`: Number of molecules to generate (overrides config)
- `config`: Generation configuration

# Returns
- Vector of valid CRAM SMILES strings

# Example
```julia
molecules = generate_cram(n=50)
println(length(molecules))  # 50
```
"""
function generate_cram(; n::Int=100, config::GenerationConfig=GenerationConfig())::Vector{String}
    # Apply seed if provided
    if !isnothing(config.seed)
        Random.seed!(config.seed)
    end
    
    actual_n = n > 0 ? n : config.n_molecules
    
    molecules = String[]
    seen = Set{String}()
    attempts = 0
    
    prog = Progress(actual_n, desc="Generating CRAM molecules: ")
    
    while length(molecules) < actual_n && attempts < config.max_attempts
        attempts += 1
        
        # Pick random seed
        seed = rand(DEFAULT_SEEDS)
        
        # Apply mutations
        mutated = mutate_smiles(seed, config.mutation_rate)
        
        # Validate
        if config.validate
            result = validate_cram(mutated)
            
            if result.is_cram && !(result.canonical in seen)
                push!(molecules, result.canonical)
                push!(seen, result.canonical)
                next!(prog)
            end
        else
            if !(mutated in seen)
                push!(molecules, mutated)
                push!(seen, mutated)
                next!(prog)
            end
        end
    end
    
    finish!(prog)
    
    return molecules
end

"""
    mutate_smiles(smiles::String, mutation_rate::Float64=0.3) -> String

Apply random mutations to a SMILES string.

# Mutation Types
1. Point mutation: Replace a token
2. Insertion: Add a fragment
3. Deletion: Remove a token
4. Swap: Exchange two positions

# Arguments
- `smiles`: Input SMILES string
- `mutation_rate`: Probability of mutation at each position

# Returns
- Mutated SMILES string
"""
function mutate_smiles(smiles::String, mutation_rate::Float64=0.3)::String
    chars = collect(smiles)
    n = length(chars)
    
    if n == 0
        return smiles
    end
    
    # Decide mutation type
    mutation_type = rand(1:4)
    
    if mutation_type == 1
        # Point mutation
        return point_mutate(smiles, mutation_rate)
    elseif mutation_type == 2
        # Insertion
        return insert_fragment(smiles)
    elseif mutation_type == 3
        # Deletion (careful not to break structure)
        return safe_delete(smiles)
    else
        # Swap
        return swap_positions(smiles)
    end
end

"""
Apply point mutations to SMILES.
"""
function point_mutate(smiles::String, rate::Float64)::String
    chars = collect(smiles)
    result = Char[]
    
    i = 1
    while i <= length(chars)
        c = chars[i]
        
        if rand() < rate && c ∉ ['(', ')', '1', '2', '3', '4', '5', '6', '[', ']', '+', '-']
            # Mutate this position
            if c == 'C' || c == 'c'
                push!(result, rand(['C', 'C', 'C', 'O', 'N']))  # Bias toward C
            elseif c == 'O' || c == 'o'
                push!(result, rand(['O', 'O', 'C', 'N']))
            elseif c == '='
                push!(result, rand(['=', '-']))
            else
                push!(result, c)
            end
        else
            push!(result, c)
        end
        
        i += 1
    end
    
    return String(result)
end

"""
Insert a CRAM-relevant fragment.
"""
function insert_fragment(smiles::String)::String
    # Find safe insertion points (after C atoms not in special groups)
    insertion_points = Int[]
    
    for i in 1:length(smiles)
        if smiles[i] == 'C' && (i == length(smiles) || smiles[i+1] ∉ ['l', 'a'])
            # Not Cl or Ca
            push!(insertion_points, i)
        end
    end
    
    if isempty(insertion_points)
        return smiles
    end
    
    # Pick insertion point and fragment
    pos = rand(insertion_points)
    fragment = rand(CRAM_FRAGMENTS)
    
    # Insert as branch
    if pos < length(smiles)
        return smiles[1:pos] * "(" * fragment * ")" * smiles[pos+1:end]
    else
        return smiles * "(" * fragment * ")"
    end
end

"""
Safely delete a token without breaking structure.
"""
function safe_delete(smiles::String)::String
    # Find deletable segments (single C atoms not critical to structure)
    chars = collect(smiles)
    n = length(chars)
    
    if n <= 10
        return smiles  # Too short to delete
    end
    
    # Find isolated C not in rings
    for _ in 1:10  # Try up to 10 times
        i = rand(1:n)
        c = chars[i]
        
        if c == 'C'
            # Check if safe to delete
            prev = i > 1 ? chars[i-1] : ' '
            next = i < n ? chars[i+1] : ' '
            
            # Don't delete if in ring number or special context
            if prev ∉ ['1', '2', '3', '4', '5', '6', 'C'] && 
               next ∉ ['1', '2', '3', '4', '5', '6', '(', '=', 'l', 'a']
                return smiles[1:i-1] * smiles[i+1:end]
            end
        end
    end
    
    return smiles
end

"""
Swap two positions in SMILES.
"""
function swap_positions(smiles::String)::String
    chars = collect(smiles)
    n = length(chars)
    
    if n < 4
        return smiles
    end
    
    # Find swappable atoms
    swappable = Int[]
    for i in 1:n
        if chars[i] in ['C', 'O', 'N', 'S']
            push!(swappable, i)
        end
    end
    
    if length(swappable) < 2
        return smiles
    end
    
    # Pick two positions
    i, j = rand(swappable, 2)
    if i != j
        chars[i], chars[j] = chars[j], chars[i]
    end
    
    return String(chars)
end

"""
    crossover(smiles1::String, smiles2::String) -> String

Perform crossover between two SMILES strings.
"""
function crossover(smiles1::String, smiles2::String)::String
    n1, n2 = length(smiles1), length(smiles2)
    
    if n1 < 4 || n2 < 4
        return smiles1
    end
    
    # Single point crossover
    point1 = rand(2:n1-2)
    point2 = rand(2:n2-2)
    
    # Combine
    return smiles1[1:point1] * smiles2[point2:end]
end

"""
    batch_generate(n::Int; output_file::Union{String,Nothing}=nothing, config::GenerationConfig=GenerationConfig()) -> DataFrame

Generate multiple CRAM molecules and optionally save to file.

# Arguments
- `n`: Number of molecules to generate
- `output_file`: Optional CSV output path
- `config`: Generation configuration

# Returns
- DataFrame with generated molecules and their properties
"""
function batch_generate(n::Int; 
                        output_file::Union{String,Nothing}=nothing, 
                        config::GenerationConfig=GenerationConfig())::DataFrame
    
    # Generate molecules
    molecules = generate_cram(n=n, config=config)
    
    # Validate and collect properties
    results = batch_validate(molecules)
    
    # Create DataFrame
    df = DataFrame(
        smiles = [r.smiles for r in results],
        canonical = [r.canonical for r in results],
        is_cram = [r.is_cram for r in results],
        n_cooh = [r.n_cooh for r in results],
        aliphatic_pct = [r.aliphatic_pct for r in results],
        mw = [r.mw for r in results],
        n_atoms = [r.n_atoms for r in results],
        n_rings = [r.n_rings for r in results]
    )
    
    # Save if output specified
    if !isnothing(output_file)
        CSV.write(output_file, df)
        println("📁 Saved $(nrow(df)) molecules to: $output_file")
    end
    
    return df
end
