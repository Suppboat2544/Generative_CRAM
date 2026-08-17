#!/usr/bin/env julia
"""
Julia mutation smoke test.

Validates a seed, extracts 163D features, generates CRAM-valid mutants,
and reports the Hertkorn environmental pass rate (expected ~30%).
"""

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "CRAMGenerator.jl"))
using .CRAMGenerator
using Random

const SEED = "CC1CCC(CC1)(C(=O)O)C(=O)O"

function main()
    println("=== Julia CRAM mutation smoke ===")

    result = validate_cram(SEED)
    @assert result.is_cram "reference seed should be CRAM-valid"
    @assert is_environmentally_plausible(SEED) "reference seed should be in the Hertkorn window"
    println("validate: PASS")

    fv = extract_features(SEED)
    @assert length(fv.features) == 163
    @assert fv.n_atoms > 0
    println("features: $(fv.n_atoms) x 163 PASS")

    config = GenerationConfig(n_molecules=40, max_attempts=8000, seed=42, validate=true)
    elapsed = @elapsed (mols = generate_cram(n=40, config=config))
    @assert length(mols) >= 10 "should produce CRAM-valid mutants"

    env = count(is_environmentally_plausible, mols)
    pct = 100.0 * env / length(mols)
    println("generate: $(length(mols)) CRAM-valid")
    println(summarize_env(env, length(mols); seconds=elapsed))

    @assert pct >= 5.0 "environmental rate should not be near zero"
    @assert pct <= 70.0 "mutation is not property-conditioned; the trained CVAE is better"
    println("smoke: PASS (expected ~$(EXPECTED_ENV_PERCENT)% environmentally plausible)")
    println("The trained Python CVAE (64D) is better: property-conditioned, ~100× faster.")
    return 0
end

exit(main())
