# Julia edition

Validate CRAM molecules, extract 163D features, and **mutation-generate** SMILES
(MolecularGraph.jl). This is not the Python CVAE.

Sibling editions: [Python CVAE](../python/README.md) · [Java REST/CLI](../java/README.md) · [repo root](../README.md)

## Features

- **SMILES validation** against CRAM structural criteria
- **163D feature extraction**
- **Mutation generation** (SMILES edits, not SELFIES and not CVAE)
- **Batch processing**

Mutation is CRAM-valid by construction but typically only **~30% environmentally plausible**
(Hertkorn O/C 0.40–0.60, H/C 1.20–1.70). Java `--seed 42` recorded **50%** (CDK) / **56%** (RDKit).
Compare with Java:

```bash
cd julia
julia --project=. cli.jl smoke -n 40 --seed 42
julia --project=. test/smoke.jl
```

Training SMILES: download **`cram_data`** from [Google Drive](https://drive.google.com/drive/folders/1mA7STHTioD52j54_GYeP_1OnNFslBNc4?usp=sharing). Full mapping: [DATA.md](../DATA.md).

## Installation

### Prerequisites

- Julia 1.9 or higher
- Package manager (Pkg)

### Setup

```julia
# From the repository: Generative_CRAM/julia
cd("path/to/Generative_CRAM/julia")

# Activate and instantiate the project
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Install Dependencies

```julia
using Pkg
Pkg.add([
    "MolecularGraph",
    "Graphs", 
    "DataFrames",
    "CSV",
    "JSON3",
    "ArgParse",
    "ProgressMeter"
])
```

## Quick Start

### As a Package

```julia
using CRAMGenerator

# Validate a molecule
result = validate_cram("CC1CCC(CC1)(C(=O)O)C(=O)O")
println(result.is_cram)      # true
println(result.n_cooh)       # 2
println(result.aliphatic_pct) # ~0.857

# Generate CRAM molecules
molecules = generate_cram(n=50)

# Extract 163D features
fv = extract_features("CC1CCC(CC1)(C(=O)O)C(=O)O")
println(length(fv.features))  # 163

# Batch validation
df = batch_validate("molecules.txt", output_file="results.csv")

# Display criteria info
cram_info()
```

### Command Line Interface

```bash
# Validate molecules
julia cli.jl validate "CC1CCC(CC1)(C(=O)O)C(=O)O"

# Batch validate from file
julia cli.jl batch input.txt -o results.csv

# Generate 100 CRAM molecules
julia cli.jl generate -n 100 -o generated.csv

# Generate with custom parameters
julia cli.jl generate -n 50 -m 0.4 --max-attempts 5000

# Smoke: ~30% environmentally plausible (Hertkorn window)
julia cli.jl smoke -n 40 --seed 42

# Extract features
julia cli.jl features "CC1CCC(CC1)(C(=O)O)C(=O)O" -v

# Show CRAM criteria
julia cli.jl info
```

## CRAM Criteria

CRAM molecules are characterized by:

| Criterion | Requirement | Description |
|-----------|-------------|-------------|
| **Carboxyl Groups** | ≥2 | -COOH functional groups |
| **Alicyclic Ring** | Required | Non-aromatic ring structure |
| **Aliphatic Carbon** | ≥60% | sp³ hybridized carbons |
| **Molecular Weight** | 100-750 Da | Typical marine DOM range |

## 163D Feature Vector

The feature extractor produces 163-dimensional vectors:

| Range | Category | Description |
|-------|----------|-------------|
| 1-45 | Base | Atom types, hybridization, electronegativity |
| 46-65 | Ring-Chain | Ring size, fusion, bridgehead atoms |
| 66-77 | Stereochemistry | Chiral centers, E/Z configuration |
| 78-120 | Marine DOM | Ocean-specific molecular patterns |
| 121-130 | Terpenoid | Terpene-like structural features |
| 131-143 | Functional | Carbonyl, ether, ester groups |
| 144-163 | Oxidation | Carbon oxidation state patterns |

## API Reference

### Validation

```julia
# Single molecule validation
result = validate_cram(smiles::String) -> ValidationResult

# Batch validation
results = batch_validate(smiles_list::Vector{String}) -> Vector{ValidationResult}
df = batch_validate(input_file::String; output_file=nothing) -> DataFrame
```

### Feature Extraction

```julia
# Extract features from SMILES
fv = extract_features(smiles::String) -> FeatureVector

# Access features
fv.features      # 163-element Vector{Float64}
fv.atom_features # Matrix{Float64} (n_atoms × 163)
fv.n_atoms       # Number of atoms
```

### Generation

```julia
# Generate CRAM molecules
molecules = generate_cram(; n=100, config=GenerationConfig()) -> Vector{String}

# Batch generate with properties
df = batch_generate(n; output_file=nothing, config=config) -> DataFrame

# Configuration
config = GenerationConfig(
    n_molecules = 100,
    mutation_rate = 0.3,
    crossover_rate = 0.2,
    max_attempts = 10000,
    validate = true,
    seed = 42  # For reproducibility
)
```

### Types

```julia
# Validation result
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

# Feature vector
struct FeatureVector
    features::Vector{Float64}      # 163D molecule features
    atom_features::Matrix{Float64} # Per-atom features
    n_atoms::Int
end
```

## Performance Tips

1. **Multi-threading**: Julia supports multi-threading
   ```bash
   julia -t auto cli.jl generate -n 1000
   ```

2. **Batch Processing**: Process molecules in batches for efficiency
   ```julia
   results = batch_validate(large_smiles_list)
   ```

3. **Precompilation**: First run compiles; subsequent runs are faster

## Project Structure

```
julia/
├── Project.toml           # Package manifest
├── cli.jl                 # Command-line interface
├── README.md              # This file
└── src/
    ├── CRAMGenerator.jl   # Main module
    ├── types.jl           # Type definitions
    ├── validator.jl       # CRAM validation
    ├── features.jl        # 163D feature extraction
    ├── generator.jl       # Molecule generation
    └── utils.jl           # Utility functions
```

## Dependencies

- **MolecularGraph.jl**: Molecular graph operations and SMILES parsing
- **Graphs.jl**: Graph algorithms
- **DataFrames.jl**: Tabular data handling
- **CSV.jl**: CSV file I/O
- **ArgParse.jl**: Command-line argument parsing
- **ProgressMeter.jl**: Progress bars

## License

MIT License - See main project LICENSE file.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{cram_generator_julia,
  title = {CRAM Generator - Julia Edition},
  author = {Klabklaydee Supaporn},
  year = {2026},
  url = {https://github.com/Suppboat2544/Generative_CRAM}
}
```

## See Also

Sibling editions: [Python CVAE](../python/README.md) · [Java REST/CLI](../java/README.md) · [Repository root](../README.md)

SMILES lists for batch jobs: **`cram_data`** on [Google Drive](https://drive.google.com/drive/folders/1mA7STHTioD52j54_GYeP_1OnNFslBNc4?usp=sharing) ([DATA.md](../DATA.md)).
