"""
About Tab
=========
Tab for application information and credits.
"""

import gradio as gr


def create_about_tab():
    """Create the about/credits tab."""
    with gr.Tab("About"):
        gr.Markdown("""
        # CRAM Generator
        
        ## Overview
        
        **CRAM (Carboxyl-Rich Alicyclic Molecules) Generator** is a deep learning toolkit 
        for generating and validating novel CRAM-like molecules found in marine dissolved 
        organic matter (DOM).
        
        ---
        
        ## Features
        
        ### 🔬 Property-Conditioned CVAE
        - Conditional Variational Autoencoder for targeted molecule generation
        - Control molecular properties: carboxyl groups, molecular weight, aliphatic content
        - Fast batch generation with 100K+ molecules capability
        
        ### 🧬 SELFIES Mutation Generator
        - Chemically valid mutations using SELFIES representation
        - Maintains structural validity during generation
        - Configurable mutation rates
        
        ### ✅ CRAM Validation
        - Multi-criteria validation for CRAM characteristics
        - Marine DOM property assessment
        - Synthetic accessibility scoring
        
        ### 📊 163D Feature Extraction
        - **Base Features (45D):** Atom types, bonds, molecular descriptors
        - **Ring-Chain Features (20D):** Ring systems and chain analysis
        - **Stereochemistry (12D):** Chiral centers and isomerism
        - **Marine DOM (43D):** Ocean chemistry indicators
        - **Terpenoid (10D):** Natural product features
        - **Functional Groups (13D):** Chemical functionality
        - **Oxidation (20D):** Redox state indicators
        
        ---
        
        ## Architecture
        
        ```
        PropertyConditionedCRAMVAE
        ├── PropertyConditionEncoder (3 → 32D)
        ├── DeepCRAMEncoder (163D → 256D → 64D latent)
        │   ├── GraphConv layers
        │   ├── GAT attention
        │   └── Global pooling
        └── SELFIESDecoder (64D + 32D → SELFIES tokens)
            ├── GRU layers
            └── Token projection
        ```
        
        ---
        
        ## Technical Details
        
        - **Token IDs:** PAD=0, SOS=1, EOS=2, UNK=3
        - **Latent Dimension:** 64
        - **Hidden Dimension:** 256
        - **Condition Dimension:** 32
        - **Allowed Atoms:** C, H, O, N, S, F, Cl, Br, I
        
        ---
        
        ## Usage
        
        ### 1. Validation Tab
        Enter a single SMILES to validate and analyze its properties.
        
        ### 2. Batch Validation Tab
        Validate multiple SMILES at once for high-throughput screening.
        
        ### 3. Generate (Mutation) Tab
        Create novel molecules by mutating a seed structure.
        
        ### 4. Generate (CVAE) Tab
        Use the trained CVAE model to generate molecules with specific properties.
        
        ---
        
        ## Requirements
        
        - Python 3.8+
        - PyTorch 1.9+
        - RDKit 2021.03+
        - PyTorch Geometric
        - Gradio
        - selfies
        
        ---
        
        ## Citation
        
        If you use this tool in your research, please cite:
        
        ```
        @software{cram_generator,
            title = {CRAM Generator: Deep Learning for Marine DOM Molecules},
            year = {2024},
            url = {https://github.com/your-repo/cram-generator}
        }
        ```
        
        ---
        
        ## License
        
        MIT License
        
        ---
        
        ## Acknowledgments
        
        - RDKit for cheminformatics
        - PyTorch and PyTorch Geometric for deep learning
        - SELFIES for molecular representation
        - Gradio for the user interface
        """)
    
    return None
