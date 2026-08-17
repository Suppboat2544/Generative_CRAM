#!/usr/bin/env julia
"""
CRAM Generator CLI
==================
Command-line interface for CRAM molecule operations.

Usage:
    julia cli.jl validate <smiles>...
    julia cli.jl batch <input_file> [-o <output>]
    julia cli.jl generate [-n <count>] [-o <output>]
    julia cli.jl features <smiles>... [-v]
    julia cli.jl info
"""

using ArgParse

# Add package to load path
push!(LOAD_PATH, @__DIR__)

# Import the package
include("src/CRAMGenerator.jl")
using .CRAMGenerator

function parse_commandline()
    s = ArgParseSettings(
        prog = "CRAM Generator",
        description = "🧬 Deep Learning Toolkit for Marine DOM Molecules",
        version = "1.0.0",
        add_version = true
    )

    @add_arg_table! s begin
        "command"
            help = "Command: validate, batch, generate, features, smoke, info"
            required = true
            
        "args"
            help = "Command arguments (SMILES strings or file paths)"
            nargs = '*'
            
        "--output", "-o"
            help = "Output file path"
            default = ""
            
        "--count", "-n"
            help = "Number of molecules to generate"
            arg_type = Int
            default = 100
            
        "--verbose", "-v"
            help = "Verbose output"
            action = :store_true
            
        "--mutation-rate", "-m"
            help = "Mutation rate for generation"
            arg_type = Float64
            default = 0.3
            
        "--max-attempts"
            help = "Maximum generation attempts"
            arg_type = Int
            default = 10000
            
        "--seed"
            help = "Random seed for reproducibility"
            arg_type = Int
            default = nothing
    end

    return parse_args(s)
end

function print_banner()
    println("""
╔═══════════════════════════════════════════════════════════╗
║           🧬 CRAM Generator - Julia Edition               ║
║     Deep Learning Toolkit for Marine DOM Molecules        ║
╚═══════════════════════════════════════════════════════════╝
    """)
end

function cmd_validate(args, parsed)
    if isempty(args)
        println("❌ Error: No SMILES provided")
        println("Usage: julia cli.jl validate <smiles>...")
        return 1
    end
    
    all_valid = true
    for smiles in args
        result = validate_cram(smiles)
        print_validation_result(result, verbose=parsed["verbose"])
        if !result.is_cram
            all_valid = false
        end
    end
    
    return all_valid ? 0 : 1
end

function cmd_batch(args, parsed)
    if isempty(args)
        println("❌ Error: No input file provided")
        println("Usage: julia cli.jl batch <input_file> [-o <output>]")
        return 1
    end
    
    input_file = args[1]
    
    if !isfile(input_file)
        println("❌ Error: File not found: $input_file")
        return 1
    end
    
    output_file = isempty(parsed["output"]) ? nothing : parsed["output"]
    
    println("\n📂 Processing molecules from: $input_file")
    
    df = batch_validate(input_file, output_file=output_file)
    
    valid_count = sum(df.is_cram)
    total = nrow(df)
    
    println()
    println("📊 Batch Validation Complete")
    println("   Total: $total")
    println("   Valid CRAM: $valid_count ($(round(100*valid_count/total, digits=1))%)")
    println("   Invalid: $(total - valid_count)")
    
    if !isnothing(output_file)
        println("📁 Results saved to: $output_file")
    end
    
    return 0
end

function cmd_generate(args, parsed)
    n = parsed["count"]
    output = isempty(parsed["output"]) ? "generated_cram.csv" : parsed["output"]
    
    println()
    println("═" ^ 60)
    println("🧬 CRAM Generation")
    println("═" ^ 60)
    println("Target: $n molecules")
    println("Output: $output")
    println("Mutation rate: $(parsed["mutation-rate"])")
    println("═" ^ 60)
    
    config = GenerationConfig(
        n_molecules = n,
        mutation_rate = parsed["mutation-rate"],
        max_attempts = parsed["max-attempts"],
        seed = parsed["seed"]
    )
    
    df = batch_generate(n, output_file=output, config=config)
    
    println()
    println("═" ^ 60)
    println("✅ Generated $(nrow(df)) CRAM molecules")
    println("📁 Saved to: $output")
    println("═" ^ 60)
    
    return 0
end

function cmd_features(args, parsed)
    if isempty(args)
        println("❌ Error: No SMILES provided")
        println("Usage: julia cli.jl features <smiles>... [-v]")
        return 1
    end
    
    println()
    println("═" ^ 60)
    println("🔬 163D Feature Extraction")
    println("═" ^ 60)
    
    for smiles in args
        fv = extract_features(smiles)
        
        println("\n📋 SMILES: $smiles")
        println("   Feature dimensions: $(length(fv.features))")
        println("   Atoms: $(fv.n_atoms)")
        println("   Non-zero features: $(sum(fv.features .!= 0))")
        
        if parsed["verbose"]
            println("\n   Feature breakdown:")
            println("   ├── Base features (1-45): $(round(sum(abs.(fv.features[1:45])), digits=2))")
            println("   ├── Ring-Chain (46-65): $(round(sum(abs.(fv.features[46:65])), digits=2))")
            println("   ├── Stereo (66-77): $(round(sum(abs.(fv.features[66:77])), digits=2))")
            println("   ├── Marine DOM (78-120): $(round(sum(abs.(fv.features[78:120])), digits=2))")
            println("   ├── Terpenoid (121-130): $(round(sum(abs.(fv.features[121:130])), digits=2))")
            println("   ├── Functional (131-143): $(round(sum(abs.(fv.features[131:143])), digits=2))")
            println("   └── Oxidation (144-163): $(round(sum(abs.(fv.features[144:163])), digits=2))")
        end
    end
    
    println()
    return 0
end

function cmd_smoke(args, parsed)
    seed = "CC1CCC(CC1)(C(=O)O)C(=O)O"
    n = parsed["count"]
    rng_seed = parsed["seed"] === nothing ? 42 : parsed["seed"]

    println("=== Julia CRAM mutation smoke ===")
    result = validate_cram(seed)
    if !result.is_cram || !is_environmentally_plausible(seed)
        println("smoke FAIL: reference seed is not CRAM / not in Hertkorn window")
        return 1
    end
    println("validate: PASS")

    fv = extract_features(seed)
    if length(fv.features) != 163
        println("smoke FAIL: 163D features")
        return 1
    end
    println("features: $(fv.n_atoms) x 163 PASS")

    config = GenerationConfig(
        n_molecules = n,
        mutation_rate = parsed["mutation-rate"],
        max_attempts = parsed["max-attempts"],
        seed = rng_seed,
        validate = true,
    )
    mols = generate_cram(n=n, config=config)
    env = count(is_environmentally_plausible, mols)
    println("generate: $(length(mols)) CRAM-valid")
    println(summarize_env(env, length(mols)))
    if length(mols) < 10
        println("smoke FAIL: too few CRAM-valid mutants")
        return 1
    end
    println("smoke: PASS (expected ~$(EXPECTED_ENV_PERCENT)% environmentally plausible)")
    return 0
end

function cmd_info(args, parsed)
    cram_info()
    return 0
end

function main()
    print_banner()
    
    parsed = parse_commandline()
    command = lowercase(parsed["command"])
    args = parsed["args"]
    
    exit_code = if command == "validate"
        cmd_validate(args, parsed)
    elseif command == "batch"
        cmd_batch(args, parsed)
    elseif command == "generate"
        cmd_generate(args, parsed)
    elseif command == "features"
        cmd_features(args, parsed)
    elseif command == "smoke"
        cmd_smoke(args, parsed)
    elseif command == "info"
        cmd_info(args, parsed)
    else
        println("❌ Unknown command: $command")
        println("Available commands: validate, batch, generate, features, smoke, info")
        1
    end
    
    exit(exit_code)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
