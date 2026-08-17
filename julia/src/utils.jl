"""
Utility functions for CRAM Generator.
"""

using Printf

"""
    cram_info() -> Nothing

Display CRAM molecule criteria and system information.
"""
function cram_info()
    println("""
╔═══════════════════════════════════════════════════════════╗
║              🧬 CRAM Molecule Criteria                    ║
╚═══════════════════════════════════════════════════════════╝

CRAM (Carboxyl-Rich Alicyclic Molecules) are key components
of marine dissolved organic matter (DOM).

┌───────────────────────────────────────────────────────────┐
│ Structural Requirements:                                  │
├───────────────────────────────────────────────────────────┤
│ 1. ≥2 Carboxyl Groups (-COOH)                            │
│    • Primary acidic functionality                         │
│    • Responsible for metal binding                        │
│                                                           │
│ 2. Alicyclic Ring Structure                              │
│    • Non-aromatic ring system                             │
│    • Typically 5-7 membered rings                         │
│    • May include fused ring systems                       │
│                                                           │
│ 3. ≥60% Aliphatic Carbon                                 │
│    • sp³ hybridized carbons                               │
│    • Excludes aromatic carbons                            │
│                                                           │
│ 4. Molecular Weight: 100-750 Da                          │
│    • Typical marine DOM range                             │
│    • Excludes very small/large molecules                  │
└───────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│ 163D Feature Categories:                                  │
├───────────────────────────────────────────────────────────┤
│ • Base Features (1-45)     : Atom types, hybridization   │
│ • Ring-Chain (46-65)       : Ring system analysis        │
│ • Stereochemistry (66-77)  : Chiral centers              │
│ • Marine DOM (78-120)      : Ocean-specific patterns     │
│ • Terpenoid (121-130)      : Terpene-like features       │
│ • Functional (131-143)     : Functional group counts     │
│ • Oxidation (144-163)      : Carbon oxidation states     │
└───────────────────────────────────────────────────────────┘
    """)
    
    # Print Julia environment info
    println("┌───────────────────────────────────────────────────────────┐")
    println("│ Julia Environment:                                        │")
    println("├───────────────────────────────────────────────────────────┤")
    @printf("│ Julia Version: %-42s │\n", string(VERSION))
    @printf("│ Threads:       %-42d │\n", Threads.nthreads())
    println("└───────────────────────────────────────────────────────────┘")
    
    return nothing
end

"""
    print_validation_result(result::ValidationResult; verbose::Bool=false)

Pretty-print a validation result.
"""
function print_validation_result(result::ValidationResult; verbose::Bool=false)
    println()
    println("═" ^ 60)
    println("🧬 CRAM Validation Report")
    println("═" ^ 60)
    println()
    println("📋 SMILES: $(result.smiles)")
    
    if !isempty(result.canonical)
        println("📋 Canonical: $(result.canonical)")
    end
    
    println()
    println("─" ^ 60)
    println("📊 CRAM Criteria:")
    println("─" ^ 60)
    
    # Carboxyl groups
    cooh_ok = result.n_cooh >= 2
    println("  $(cooh_ok ? "✅" : "❌") Carboxyl groups: $(result.n_cooh) (≥2 required)")
    
    # Alicyclic ring
    println("  $(result.has_alicyclic ? "✅" : "❌") Alicyclic ring: $(result.has_alicyclic ? "Present" : "Absent")")
    
    # Aliphatic carbon
    aliphatic = result.aliphatic_pct * 100
    aliphatic_ok = aliphatic >= 60
    @printf("  %s Aliphatic carbon: %.1f%% (≥60%% required)\n", aliphatic_ok ? "✅" : "❌", aliphatic)
    
    # Molecular weight
    mw_ok = 100 <= result.mw <= 750
    @printf("  %s Molecular weight: %.1f Da (100-750 range)\n", mw_ok ? "✅" : "❌", result.mw)
    
    println()
    println("─" ^ 60)
    println("🎯 Result: $(result.is_cram ? "✅ VALID CRAM MOLECULE" : "❌ NOT A CRAM MOLECULE")")
    println("═" ^ 60)
    
    if verbose
        println()
        println("📊 Additional Properties:")
        println("   Atoms: $(result.n_atoms)")
        println("   Rings: $(result.n_rings)")
        println("   Aromatic atoms: $(result.aromatic_atoms)")
    end
end

"""
    smiles_to_mol(smiles::String)

Parse SMILES to MolecularGraph molecule (with error handling).
"""
function smiles_to_mol(smiles::String)
    try
        return smilestomol(smiles)
    catch
        return nothing
    end
end

"""
    is_valid_smiles(smiles::String) -> Bool

Check if a SMILES string is valid.
"""
function is_valid_smiles(smiles::String)::Bool
    mol = smiles_to_mol(smiles)
    return !isnothing(mol)
end

"""
    canonicalize_smiles(smiles::String) -> Union{String, Nothing}

Get canonical SMILES representation.
"""
function canonicalize_smiles(smiles::String)::Union{String, Nothing}
    mol = smiles_to_mol(smiles)
    if isnothing(mol)
        return nothing
    end
    
    try
        return canonicalize(mol) |> tosmiles
    catch
        return smiles
    end
end
