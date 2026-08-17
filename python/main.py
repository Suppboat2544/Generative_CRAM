"""
CRAM Generator - Main Entry Point
==================================
Command-line interface for CRAM molecule generation.

Usage:
    python main.py validate <smiles>
    python main.py generate --count 1000 --output results.csv
    python main.py ui --port 7860
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from tqdm import tqdm


def validate_command(args):
    """Validate SMILES for CRAM criteria."""
    from src.validator import CRAMValidator
    
    validator = CRAMValidator()
    
    print("=" * 60)
    print("🧬 CRAM Validation")
    print("=" * 60)
    
    for smiles in args.smiles:
        print(f"\nSMILES: {smiles}")
        result = validator.validate_smiles(smiles)
        
        print(f"  Is CRAM: {'✅ Yes' if result['is_cram'] else '❌ No'}")
        print(f"  COOH count: {result['n_cooh']}")
        print(f"  Has alicyclic: {result['has_alicyclic']}")
        print(f"  Aliphatic %: {result['aliphatic_pct']:.1%}")
        print(f"  MW: {result['mw']:.1f} Da")
        
        if result['failures']:
            print("  Failures:")
            for failure in result['failures']:
                print(f"    • {failure}")
    
    print("\n" + "=" * 60)


def generate_command(args):
    """Generate CRAM molecules using CVAE or SELFIES mutation."""
    from src.generators import FastPropertyConditionedGenerator, SELFIESMutationGenerator
    from src.models import create_cvae_model
    import torch

    print("=" * 60)
    print("CRAM Generation")
    print("=" * 60)
    print(f"Target: {args.count:,} molecules")
    print(f"Output: {args.output}")
    print(f"Mode: {args.mode}")
    print("=" * 60)

    if args.seeds:
        seed_df = pd.read_csv(args.seeds)
        col = "smiles" if "smiles" in seed_df.columns else seed_df.columns[0]
        seeds = seed_df[col].dropna().astype(str).tolist()
        print(f"Loaded {len(seeds)} seed molecules")
    else:
        seeds = [
            "CC1CCC(CC1)(C(=O)O)C(=O)O",
            "OC(=O)C1CCCCC1C(=O)O",
            "CC(C)C1CCC(CC1)(C(=O)O)C(=O)O",
            "OC(=O)C1CCC(C(=O)O)CC1",
            "CC1CC(C(=O)O)CC(C(=O)O)C1",
        ]
        print("Using default CRAM seed molecules")

    if args.mode == "vae":
        if not args.checkpoint:
            print("VAE mode requires --checkpoint path to a trained model")
            return

        print(f"\nLoading CVAE checkpoint: {args.checkpoint}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        vocab = ckpt.get("selfies_vocab") or ckpt.get("vocab")
        idx_to_token = ckpt.get("idx_to_token")
        if idx_to_token is None and vocab:
            idx_to_token = {int(v): k for k, v in vocab.items()}
        elif idx_to_token:
            idx_to_token = {int(k): v for k, v in idx_to_token.items()}
        vocab_size = ckpt.get("vocab_size", len(vocab) if vocab else None)
        latent_dim = ckpt.get("latent_dim", 64)
        hidden_dim = ckpt.get("hidden_dim", 256)
        if vocab_size is None and "output_proj.weight" in state:
            vocab_size = int(state["output_proj.weight"].shape[0])
        if vocab_size is None:
            print("Checkpoint is missing vocab_size / selfies_vocab")
            return

        if "decoder_embedding.weight" in state:
            from src.models.deep_cram_vae import DeepCRAMDecoderVAE, training_style_vocab

            if vocab:
                aligned = training_style_vocab(vocab.keys(), vocab_size)
            else:
                aligned = training_style_vocab([], vocab_size)
            vocab = aligned
            idx_to_token = {i: t for t, i in vocab.items()}
            model = DeepCRAMDecoderVAE(
                vocab_size=vocab_size,
                latent_dim=int(state["latent_proj.weight"].shape[1]) if "latent_proj.weight" in state else latent_dim,
                hidden_dim=hidden_dim,
                max_len=int(state["decoder_pos_encoding"].shape[0]) if "decoder_pos_encoding" in state else 150,
            )
            model.load_state_dict(state, strict=False)
            model = model.to(device)
            model.eval()
            print(f"Loaded Transformer CVAE (decoder-only) latent_dim={model.latent_dim} vocab={vocab_size}")
            smiles = model.sample_smiles(
                idx_to_token=idx_to_token,
                n=args.count,
                temperature=args.temperature,
                max_length=min(ckpt.get("max_len", 100), model.max_len),
            )
        else:
            model = create_cvae_model(
                vocab_size=vocab_size,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                condition_dim=ckpt.get("condition_dim", 32),
                max_len=ckpt.get("max_len", 150),
            )
            model.load_state_dict(state)
            model = model.to(device)
            model.eval()
            print(f"Loaded model with latent_dim={latent_dim}")

            generator = FastPropertyConditionedGenerator(
                model=model,
                selfies_vocab=vocab,
                idx_to_token=idx_to_token,
                device=device,
            )
            smiles = generator.generate_fast(
                n=args.count,
                n_cooh=args.n_cooh,
                ring_class=args.ring_class,
                temperature=args.temperature,
            )
        results = [{"smiles": s, "method": "cvae"} for s in smiles]
    else:
        generator = SELFIESMutationGenerator(seed_smiles=seeds)
        smiles = generator.generate(
            target_count=args.count,
            max_attempts=args.max_attempts,
        )
        results = [{"smiles": s, "method": "mutation"} for s in smiles]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)

    print("\n" + "=" * 60)
    print(f"Generated {len(results):,} molecules")
    print(f"Saved to: {args.output}")
    print("=" * 60)


def ui_command(args):
    """Launch Gradio UI."""
    from ui.app import launch_app
    
    print("=" * 60)
    print("🧬 CRAM Generator - User Interface")
    print("=" * 60)
    print(f"Starting server on port {args.port}...")
    print("=" * 60)
    
    launch_app(share=args.share, port=args.port)


def features_command(args):
    """Extract features from SMILES."""
    from rdkit import Chem
    from src.features import DeepCRAMFeatureExtractor
    import numpy as np
    
    extractor = DeepCRAMFeatureExtractor()
    
    print("=" * 60)
    print("🔬 Feature Extraction")
    print("=" * 60)
    
    for smiles in args.smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"\n❌ Invalid SMILES: {smiles}")
            continue
        
        packed = extractor.extract(mol)
        if packed is None:
            print(f"\nCould not extract features: {smiles}")
            continue
        atom_features = packed["atom_features"]

        print(f"\nSMILES: {smiles}")
        print(f"  Atoms: {mol.GetNumAtoms()}")
        print(f"  Atom features shape: {atom_features.shape}")
        print(f"  Non-zero features: {np.count_nonzero(atom_features):,}")

        if args.verbose:
            print(f"  Feature dim per atom: {atom_features.shape[1]} (expected 163)")
    
    print("\n" + "=" * 60)


def info_command(args):
    """Display system information."""
    import platform
    
    print("=" * 60)
    print("🧬 CRAM Generator - System Information")
    print("=" * 60)
    
    print(f"\nPython: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    
    try:
        import torch
        print(f"\nPyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Check for Apple Silicon MPS support
        if hasattr(torch.backends, 'mps'):
            print(f"MPS available: {torch.backends.mps.is_available()}")
            if torch.backends.mps.is_available():
                print("  ✅ Apple Silicon GPU acceleration enabled")
    except ImportError:
        print("\nPyTorch: Not installed")
    
    try:
        import dgl
        print(f"DGL: {dgl.__version__}")
    except ImportError:
        print("DGL: Not installed")
    
    try:
        import selfies as sf
        print(f"SELFIES: {sf.__version__}")
    except ImportError:
        print("SELFIES: Not installed")
    
    try:
        from rdkit import Chem
        from rdkit import __version__ as rdkit_version
        print(f"RDKit: {rdkit_version}")
    except ImportError:
        print("RDKit: Not installed")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="🧬 CRAM Generator - Deep Learning for Marine DOM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py validate "CC1CCC(CC1)(C(=O)O)C(=O)O"
  python main.py generate --count 1000 --output cram_molecules.csv
  python main.py ui --port 7860
  python main.py features "CC1CCC(CC1)(C(=O)O)C(=O)O" --verbose
  python main.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate SMILES for CRAM")
    validate_parser.add_argument("smiles", nargs="+", help="SMILES string(s) to validate")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate CRAM molecules")
    generate_parser.add_argument("--count", type=int, default=1000, help="Number of molecules")
    generate_parser.add_argument("--output", default="cram_generated.csv", help="Output file")
    generate_parser.add_argument("--seeds", help="CSV file with seed molecules")
    generate_parser.add_argument("--max-attempts", type=int, default=100000, help="Max generation attempts")
    generate_parser.add_argument("--save-interval", type=int, default=100, help="Save checkpoint interval")
    generate_parser.add_argument("--mode", choices=['vae', 'mutation'], default='mutation',
                                  help="Generation mode: 'vae' (requires trained model) or 'mutation'")
    generate_parser.add_argument("--checkpoint", help="Path to trained VAE model checkpoint (required for vae mode)")
    generate_parser.add_argument("--vae-only", action="store_true", help="Use pure VAE generation without mutation fallback")
    generate_parser.add_argument("--vae-ratio", type=float, default=0.6, help="Ratio of VAE vs mutation when not --vae-only")
    generate_parser.add_argument("--batch-size", type=int, default=256, help="Batch size for VAE generation")
    generate_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for VAE")
    generate_parser.add_argument("--n-cooh", type=int, default=3, help="Target carboxyl count for CVAE generation")
    generate_parser.add_argument("--ring-class", default="pure_alicyclic",
                                 help="CVAE ring class: fused, mixed, benzene_aliphatic, pure_alicyclic")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch Gradio interface")
    ui_parser.add_argument("--port", type=int, default=7860, help="Server port")
    ui_parser.add_argument("--share", action="store_true", help="Create public link")
    
    # Features command
    features_parser = subparsers.add_parser("features", help="Extract features from SMILES")
    features_parser.add_argument("smiles", nargs="+", help="SMILES string(s)")
    features_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Info command
    subparsers.add_parser("info", help="Display system information")
    
    args = parser.parse_args()
    
    if args.command == "validate":
        validate_command(args)
    elif args.command == "generate":
        generate_command(args)
    elif args.command == "ui":
        ui_command(args)
    elif args.command == "features":
        features_command(args)
    elif args.command == "info":
        info_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
