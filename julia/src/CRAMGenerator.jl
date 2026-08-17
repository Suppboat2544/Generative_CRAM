"""
    CRAMGenerator

Julia package for generating and validating Carboxyl-Rich Alicyclic Molecules (CRAM).

CRAM molecules are key components of marine dissolved organic matter (DOM) characterized by:
- ≥2 carboxyl groups (-COOH)
- Alicyclic ring structure (non-aromatic ring)
- ≥60% aliphatic carbon
- Molecular weight: 100-750 Da

# Features
- SMILES validation for CRAM criteria
- 163-dimensional molecular feature extraction
- Mutation-based molecule generation
- Batch processing capabilities

# Example
```julia
using CRAMGenerator

# Validate a molecule
result = validate_cram("CC1CCC(CC1)(C(=O)O)C(=O)O")
println(result.is_cram)  # true

# Generate molecules
molecules = generate_cram(n=100)

# Extract features
features = extract_features("CC1CCC(CC1)(C(=O)O)C(=O)O")
```
"""
module CRAMGenerator

using MolecularGraph
using Graphs
using DataFrames
using CSV
using JSON3
using Random
using Statistics
using LinearAlgebra
using Printf

# Export public API
export validate_cram, ValidationResult
export extract_features, FeatureVector
export generate_cram, mutate_smiles
export batch_validate, batch_generate
export cram_info
export is_environmentally_plausible, elemental_ratios, in_env_window, summarize_env
export ENV_OC_MIN, ENV_OC_MAX, ENV_HC_MIN, ENV_HC_MAX, EXPECTED_ENV_PERCENT, ENV_NOTE
export CRAM_CRITERIA, RING_CLASSES, RING_SIZE_CLASSES
export NUM_RING_CLASSES, NUM_RING_SIZE_CLASSES
export normalize_mw, normalize_dbe_c, normalize_dbe_o, normalize_dbe_h

# Include submodules
include("types.jl")
include("validator.jl")
include("features.jl")
include("generator.jl")
include("utils.jl")
include("environmental.jl")

end # module
