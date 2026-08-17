"""
CRAM Generator CLI Module
=========================
Rich command-line interface for CRAM molecule operations.
"""

import argparse
import sys
import csv
import random
from pathlib import Path
from typing import List, Optional


def print_banner():
    """Print application banner."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║           🧬 CRAM Generator - Python Edition              ║
║     Deep Learning Toolkit for Marine DOM Molecules        ║
╚═══════════════════════════════════════════════════════════╝
    """)


def print_validation_result(result: dict, verbose: bool = False):
    """Print formatted validation result."""
    print()
    print("═" * 60)
    print("🧬 CRAM Validation Report")
    print("═" * 60)
    print()
    print(f"📋 SMILES: {result.get('smiles', 'N/A')}")
    
    if result.get('canonical'):
        print(f"📋 Canonical: {result['canonical']}")
    
    print()
    print("─" * 60)
    print("📊 CRAM Criteria:")
    print("─" * 60)
    
    # Carboxyl groups
    cooh = result.get('n_cooh', 0)
    cooh_ok = cooh >= 2
    print(f"  {'✅' if cooh_ok else '❌'} Carboxyl groups: {cooh} (≥2 required)")
    
    # Alicyclic ring
    alicyclic = result.get('has_alicyclic', False)
    print(f"  {'✅' if alicyclic else '❌'} Alicyclic ring: {'Present' if alicyclic else 'Absent'}")
    
    # Aliphatic carbon
    aliphatic = result.get('aliphatic_pct', 0) * 100
    aliphatic_ok = aliphatic >= 60
    print(f"  {'✅' if aliphatic_ok else '❌'} Aliphatic carbon: {aliphatic:.1f}% (≥60% required)")
    
    # Molecular weight
    mw = result.get('mw', 0)
    mw_ok = 100 <= mw <= 750
    print(f"  {'✅' if mw_ok else '❌'} Molecular weight: {mw:.1f} Da (100-750 range)")
    
    print()
    print("─" * 60)
    is_valid = result.get('is_cram', False)
    print(f"🎯 Result: {'✅ VALID CRAM MOLECULE' if is_valid else '❌ NOT A CRAM MOLECULE'}")
    print("═" * 60)
    
    if verbose and result.get('properties'):
        print()
        print("📊 Additional Properties:")
        for key, value in result['properties'].items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")


def validate_command(args):
    """Validate SMILES for CRAM criteria."""
    from src.validator import CRAMValidator
    
    validator = CRAMValidator()
    
    for smiles in args.smiles:
        result = validator.validate_smiles(smiles)
        result['smiles'] = smiles
        print_validation_result(result, args.verbose)
    
    return 0 if all(validator.validate_smiles(s).get('is_cram', False) for s in args.smiles) else 1


def batch_command(args):
    """Batch validate SMILES from file."""
    from src.validator import CRAMValidator
    
    validator = CRAMValidator()
    
    # Read input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}")
        return 1
    
    smiles_list = []
    with open(input_path, 'r') as f:
        for line in f:
            smiles = line.strip()
            if smiles and not smiles.startswith('#'):
                smiles_list.append(smiles)
    
    if not smiles_list:
        print("❌ Error: No SMILES found in file")
        return 1
    
    print(f"\n📂 Processing {len(smiles_list)} molecules from {input_path}")
    
    # Validate all
    results = []
    valid_count = 0
    
    for smiles in smiles_list:
        result = validator.validate_smiles(smiles)
        is_valid = result.get('is_cram', False)
        if is_valid:
            valid_count += 1
        
        results.append({
            'smiles': smiles,
            'valid': is_valid,
            'carboxyl_count': result.get('n_cooh', 0),
            'aliphatic_percent': result.get('aliphatic_pct', 0) * 100,
            'molecular_weight': result.get('mw', 0),
            'has_alicyclic_ring': result.get('has_alicyclic', False)
        })
    
    # Write output if specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"📁 Results saved to: {output_path}")
    
    # Print summary
    print()
    print("📊 Batch Validation Complete")
    print(f"   Total: {len(smiles_list)}")
    print(f"   Valid CRAM: {valid_count} ({100*valid_count/len(smiles_list):.1f}%)")
    print(f"   Invalid: {len(smiles_list) - valid_count}")
    
    return 0


def generate_command(args):
    """Generate CRAM molecules."""
    from src.validator import CRAMValidator
    from src.generators import SELFIESMutationGenerator
    import pandas as pd
    import selfies as sf
    
    print()
    print("═" * 60)
    print("🧬 CRAM Generation")
    print("═" * 60)
    print(f"Target: {args.count:,} molecules")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    print("═" * 60)
    
    # Default seeds
    seeds = [
        "CC1CCC(CC1)(C(=O)O)C(=O)O",
        "OC(=O)C1CCCCC1C(=O)O",
        "CC(C)C1CCC(CC1)(C(=O)O)C(=O)O",
    ]
    
    if args.mode == 'cvae':
        # CVAE generation requires trained model
        if not args.checkpoint:
            print("❌ CVAE mode requires --checkpoint path to trained model")
            print("   Use --mode mutation for model-free generation")
            return 1
        
        try:
            import torch
            from src.models import PropertyConditionedCRAMVAE
            from src.tokenizer import SELFIESTokenizer
            from src.generators import FastPropertyConditionedGenerator
            
            print("\n🔧 Loading CVAE model...")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load checkpoint
            checkpoint = torch.load(args.checkpoint, map_location=device)
            
            # Initialize tokenizer
            tokenizer = SELFIESTokenizer()
            tokenizer.build_vocab(seeds)
            
            # Initialize model with proper config dicts
            encoder_config = {
                'in_dim': 163,
                'hidden_dim': 256,
                'latent_dim': 64,
                'num_heads': 4,
                'num_layers': 3,
                'dropout': 0.1
            }
            decoder_config = {
                'hidden_dim': 256,
                'num_layers': 2,
                'max_len': 100,
                'dropout': 0.1
            }
            
            model = PropertyConditionedCRAMVAE(
                vocab_size=tokenizer.vocab_size,
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                condition_dim=32
            )
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            model = model.to(device)
            model.eval()
            
            # Initialize generator
            generator = FastPropertyConditionedGenerator(
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            
            # Generate with conditions
            print(f"   Target COOH: {args.target_cooh}")
            print(f"   Target MW: {args.target_mw}")
            print(f"   Target Aliphatic: {args.target_aliphatic}")
            
            molecules = generator.generate_batch(
                properties={
                    'n_cooh': args.target_cooh,
                    'mw': args.target_mw,
                    'aliphatic_pct': args.target_aliphatic,
                    'ring_class': 1,
                    'has_benzene': 0
                },
                num_samples=args.count,
                temperature=args.temperature
            )
            
            df = pd.DataFrame({'smiles': molecules})
            
        except Exception as e:
            print(f"❌ Error loading CVAE model: {e}")
            print("   Falling back to mutation mode...")
            args.mode = 'mutation'
    
    if args.mode == 'mutation':
        # Mutation-based generation (no model needed)
        from tqdm import tqdm
        
        validator = CRAMValidator()
        generator = SELFIESMutationGenerator(
            seed_smiles=seeds,
            mutation_rate=args.mutation_rate
        )
        
        molecules = []
        attempts = 0
        
        with tqdm(total=args.count, desc="Generating") as pbar:
            while len(molecules) < args.count and attempts < args.max_attempts:
                attempts += 1
                
                # Pick random seed and convert to SELFIES
                seed = random.choice(seeds)
                try:
                    seed_selfies = sf.encoder(seed)
                    if not seed_selfies:
                        continue
                    
                    # Mutate
                    mutated_selfies = generator.mutate(seed_selfies)
                    
                    # Convert back to SMILES
                    new_smiles = sf.decoder(mutated_selfies)
                    if not new_smiles:
                        continue
                    
                    # Validate
                    result = validator.validate_smiles(new_smiles)
                    if result.get('is_cram', False):
                        if new_smiles not in molecules:
                            molecules.append(new_smiles)
                            pbar.update(1)
                except:
                    pass
        
        df = pd.DataFrame({'smiles': molecules})
    
    # Save results
    df.to_csv(args.output, index=False)
    
    print()
    print("═" * 60)
    print(f"✅ Generated {len(df):,} CRAM molecules")
    print(f"📁 Saved to: {args.output}")
    print("═" * 60)
    
    return 0


def features_command(args):
    """Extract 163D features from SMILES."""
    from rdkit import Chem
    from src.features import DeepCRAMFeatureExtractor
    import numpy as np
    
    extractor = DeepCRAMFeatureExtractor()
    
    print()
    print("═" * 60)
    print("🔬 163D Feature Extraction")
    print("═" * 60)
    
    for smiles in args.smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"\n❌ Invalid SMILES: {smiles}")
            continue
        
        features = extractor.extract_features(mol)
        
        print(f"\n📋 SMILES: {smiles}")
        print(f"   Atoms: {mol.GetNumAtoms()}")
        print(f"   Feature shape: {features.shape}")
        print(f"   Non-zero: {np.count_nonzero(features):,}")
        
        if args.verbose:
            print("\n   Feature breakdown:")
            print(f"   ├── Base features (0-44): {np.sum(np.abs(features[:, :45])):.2f}")
            print(f"   ├── Ring-Chain (45-64): {np.sum(np.abs(features[:, 45:65])):.2f}")
            print(f"   ├── Stereo (65-76): {np.sum(np.abs(features[:, 65:77])):.2f}")
            print(f"   ├── Marine DOM (77-119): {np.sum(np.abs(features[:, 77:120])):.2f}")
            print(f"   ├── Terpenoid (120-129): {np.sum(np.abs(features[:, 120:130])):.2f}")
            print(f"   ├── Functional (130-142): {np.sum(np.abs(features[:, 130:143])):.2f}")
            print(f"   └── Oxidation (143-162): {np.sum(np.abs(features[:, 143:163])):.2f}")
        
        if args.save:
            output = Path(args.save)
            np.save(output, features)
            print(f"   💾 Saved to: {output}")
    
    print()
    return 0


def info_command(args):
    """Show CRAM criteria and system information."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║              🧬 CRAM Molecule Criteria                    ║
╚═══════════════════════════════════════════════════════════╝

CRAM (Carboxyl-Rich Alicyclic Molecules) are key components
of marine dissolved organic matter (DOM).

┌───────────────────────────────────────────────────────────┐
│ Criterion            │ Requirement                       │
├───────────────────────────────────────────────────────────┤
│ Carboxyl groups      │ ≥2 (COOH or COO⁻)                 │
│ Ring structure       │ Alicyclic (non-aromatic)          │
│ Aliphatic carbon     │ >60%                              │
│ Molecular weight     │ 100-750 Da                        │
└───────────────────────────────────────────────────────────┘

Example valid CRAM: CC1CCC(CC1)(C(=O)O)C(=O)O

""")
    
    # System info
    import platform
    print("📊 System Information:")
    print(f"   Python: {platform.python_version()}")
    print(f"   Platform: {platform.system()} {platform.machine()}")
    
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA: {torch.version.cuda} ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("   MPS: Available (Apple Silicon)")
    except ImportError:
        pass
    
    try:
        from rdkit import __version__ as rdkit_version
        print(f"   RDKit: {rdkit_version}")
    except ImportError:
        pass
    
    print()
    return 0


def ui_command(args):
    """Launch Gradio web interface."""
    from ui.app import launch_app
    
    print_banner()
    print(f"🌐 Starting server on port {args.port}...")
    if args.share:
        print("📤 Creating public link...")
    print()
    
    launch_app(
        share=args.share,
        server_port=args.port,
        debug=args.debug
    )
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="cram",
        description="🧬 CRAM Generator - Deep Learning Toolkit for Marine DOM Molecules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s validate "CC1CCC(CC1)(C(=O)O)C(=O)O"
  %(prog)s batch molecules.txt -o results.csv
  %(prog)s generate --count 100 --output cram.csv
  %(prog)s generate --mode cvae --checkpoint model.pt --count 1000
  %(prog)s features "CC1CCC(CC1)(C(=O)O)C(=O)O" -v
  %(prog)s ui --port 7860
  %(prog)s info
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate command
    validate_p = subparsers.add_parser("validate", help="Validate SMILES for CRAM criteria")
    validate_p.add_argument("smiles", nargs="+", help="SMILES string(s) to validate")
    validate_p.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")
    
    # Batch command
    batch_p = subparsers.add_parser("batch", help="Batch validate from file")
    batch_p.add_argument("input_file", help="Input file with SMILES (one per line)")
    batch_p.add_argument("-o", "--output", help="Output CSV file")
    
    # Generate command
    gen_p = subparsers.add_parser("generate", help="Generate CRAM molecules")
    gen_p.add_argument("--count", "-n", type=int, default=100, help="Number of molecules")
    gen_p.add_argument("--output", "-o", default="cram_generated.csv", help="Output file")
    gen_p.add_argument("--mode", choices=["mutation", "cvae"], default="mutation",
                       help="Generation mode")
    gen_p.add_argument("--checkpoint", help="CVAE checkpoint path (required for cvae mode)")
    gen_p.add_argument("--batch-size", type=int, default=256, help="Batch size for CVAE")
    gen_p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    gen_p.add_argument("--mutation-rate", type=float, default=0.3, help="Mutation rate")
    gen_p.add_argument("--max-attempts", type=int, default=10000, help="Max generation attempts")
    gen_p.add_argument("--target-cooh", type=float, default=2.5, help="Target COOH count (CVAE)")
    gen_p.add_argument("--target-mw", type=float, default=300, help="Target MW (CVAE)")
    gen_p.add_argument("--target-aliphatic", type=float, default=0.8, help="Target aliphatic % (CVAE)")
    
    # Features command
    feat_p = subparsers.add_parser("features", help="Extract 163D features")
    feat_p.add_argument("smiles", nargs="+", help="SMILES string(s)")
    feat_p.add_argument("-v", "--verbose", action="store_true", help="Show feature breakdown")
    feat_p.add_argument("--save", help="Save features to .npy file")
    
    # UI command
    ui_p = subparsers.add_parser("ui", help="Launch Gradio interface")
    ui_p.add_argument("--port", type=int, default=7860, help="Server port")
    ui_p.add_argument("--share", action="store_true", help="Create public link")
    ui_p.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Info command
    subparsers.add_parser("info", help="Show CRAM criteria and system info")
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        print_banner()
        parser.print_help()
        return 0
    
    commands = {
        "validate": validate_command,
        "batch": batch_command,
        "generate": generate_command,
        "features": features_command,
        "ui": ui_command,
        "info": info_command,
    }
    
    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
