"""
Type definitions for CRAM Generator.
"""

"""
    ValidationResult

Result of CRAM validation containing molecular properties and validity status.

# Fields
- `smiles::String`: Input SMILES string
- `canonical::String`: Canonical SMILES (empty if invalid)
- `is_valid::Bool`: Whether SMILES is chemically valid
- `is_cram::Bool`: Whether molecule meets CRAM criteria
- `n_cooh::Int`: Number of carboxyl groups
- `has_alicyclic::Bool`: Presence of alicyclic ring
- `aliphatic_pct::Float64`: Percentage of aliphatic carbons
- `mw::Float64`: Molecular weight in Daltons
- `n_atoms::Int`: Total number of atoms
- `n_rings::Int`: Number of rings
- `aromatic_atoms::Int`: Number of aromatic atoms
"""
struct ValidationResult
    smiles::String
    canonical::String
    is_valid::Bool
    is_cram::Bool
    n_cooh::Int
    has_alicyclic::Bool
    aliphatic_pct::Float64
    mw::Float64
    n_atoms::Int
    n_rings::Int
    aromatic_atoms::Int
end

# Constructor for invalid molecules
function ValidationResult(smiles::String)
    ValidationResult(smiles, "", false, false, 0, false, 0.0, 0.0, 0, 0, 0)
end

"""
    FeatureVector

163-dimensional molecular feature vector for CRAM analysis.

# Feature Groups
- Base features (1-45): Atom types, hybridization, basic properties
- Ring-Chain features (46-65): Ring system analysis
- Stereochemistry (66-77): Chiral centers, E/Z configuration
- Marine DOM (78-120): Ocean-specific molecular patterns
- Terpenoid (121-130): Terpene-like structural features
- Functional groups (131-143): Key functional group counts
- Oxidation state (144-163): Carbon oxidation patterns
"""
struct FeatureVector
    features::Vector{Float64}
    atom_features::Matrix{Float64}
    n_atoms::Int
    
    function FeatureVector(features::Vector{Float64}, atom_features::Matrix{Float64})
        @assert length(features) == 163 "Feature vector must have 163 dimensions"
        new(features, atom_features, size(atom_features, 1))
    end
end

# Convenience constructor for molecule-level features only
function FeatureVector(features::Vector{Float64})
    FeatureVector(features, reshape(features, 1, :))
end

"""
    GenerationConfig

Configuration for CRAM molecule generation.

# Fields
- `n_molecules::Int`: Number of molecules to generate
- `mutation_rate::Float64`: Probability of mutation per position
- `crossover_rate::Float64`: Probability of crossover
- `max_attempts::Int`: Maximum generation attempts
- `validate::Bool`: Whether to validate generated molecules
- `seed::Union{Int, Nothing}`: Random seed for reproducibility
"""
Base.@kwdef struct GenerationConfig
    n_molecules::Int = 100
    mutation_rate::Float64 = 0.3
    crossover_rate::Float64 = 0.2
    max_attempts::Int = 10000
    validate::Bool = true
    seed::Union{Int, Nothing} = nothing
end

"""
    CRAMCriteria

CRAM molecule classification criteria.
"""
const CRAM_CRITERIA = (
    min_cooh = 2,
    max_cooh = 6,
    min_aliphatic_pct = 0.40,
    max_aliphatic_pct = 0.90,
    min_mw = 100.0,
    max_mw = 800.0,
    require_alicyclic = true
)

"""
    RING_CLASSES

Ring class encoding (aligned with notebook):
- FUSED_ARO_ALICYCLIC (0): Fused aromatic-alicyclic rings
- MIXED_ARO_ALICYCLIC (1): Mixed aromatic-alicyclic rings
- BENZENE_ALIPHATIC (2): Benzene with aliphatic chains
- PURE_ALICYCLIC (3): Only alicyclic rings
- PURE_AROMATIC (4): Only aromatic rings
- ACYCLIC (5): No rings
"""
const RING_CLASSES = Dict{String, Int}(
    "fused" => 0,
    "fused_aro_alicyclic" => 0,
    "mixed" => 1,
    "mixed_aro_alicyclic" => 1,
    "benzene_aliphatic" => 2,
    "pure_alicyclic" => 3,
    "pure_aromatic" => 4,
    "acyclic" => 5
)

const NUM_RING_CLASSES = 6

"""
    RING_SIZE_CLASSES

Ring size classification:
- STRAINED (0): 3-4 membered rings
- CRAM_PREFERRED (1): 5-7 membered rings (typical for CRAM)
- LARGE (2): 8+ membered rings
"""
const RING_SIZE_CLASSES = Dict{String, Int}(
    "strained" => 0,
    "cram_preferred" => 1,
    "large" => 2
)

const NUM_RING_SIZE_CLASSES = 3

"""
    normalize_mw(mw::Float64) -> Float64

Normalize molecular weight to [0, 1] range.
"""
function normalize_mw(mw::Float64)::Float64
    return (mw - CRAM_CRITERIA.min_mw) / (CRAM_CRITERIA.max_mw - CRAM_CRITERIA.min_mw)
end

"""
    normalize_dbe_c(dbe_c::Float64) -> Float64

Normalize DBE/C ratio to [0, 1] range (typical range 0.3-0.7).
"""
function normalize_dbe_c(dbe_c::Float64)::Float64
    return (dbe_c - 0.3) / (0.7 - 0.3)
end

"""
    normalize_dbe_o(dbe_o::Float64) -> Float64

Normalize DBE/O ratio to [0, 1] range (typical range 0.5-2.5).
"""
function normalize_dbe_o(dbe_o::Float64)::Float64
    return (dbe_o - 0.5) / (2.5 - 0.5)
end

"""
    normalize_dbe_h(dbe_h::Float64) -> Float64

Normalize DBE/H ratio to [0, 1] range (typical range 0.1-0.4).
"""
function normalize_dbe_h(dbe_h::Float64)::Float64
    return (dbe_h - 0.1) / (0.4 - 0.1)
end

"""
    ATOMIC_MASSES

Standard atomic masses for molecular weight calculation.
"""
const ATOMIC_MASSES = Dict{Symbol, Float64}(
    :C => 12.011,
    :H => 1.008,
    :O => 15.999,
    :N => 14.007,
    :S => 32.065,
    :P => 30.974,
    :F => 18.998,
    :Cl => 35.453,
    :Br => 79.904,
    :I => 126.904
)
