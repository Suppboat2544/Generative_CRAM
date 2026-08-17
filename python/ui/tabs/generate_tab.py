"""
Generation Tab
==============
Tab for mutation-based SELFIES generation.
"""

import gradio as gr
from typing import Tuple
import time


def generate_mutations(
    seed_smiles: str,
    num_molecules: int,
    mutation_rate: float,
    max_attempts: int
) -> Tuple[str, str]:
    """Generate molecules via SELFIES mutation."""
    from src.generators import SELFIESMutationGenerator
    from src.validator import CRAMValidator
    
    if not seed_smiles.strip():
        return "Please provide a seed SMILES", ""
    
    # Initialize
    generator = SELFIESMutationGenerator(mutation_rate=mutation_rate)
    validator = CRAMValidator()
    
    start_time = time.time()
    
    # Generate
    molecules = []
    attempts = 0
    
    while len(molecules) < num_molecules and attempts < max_attempts:
        try:
            new_smiles = generator.mutate(seed_smiles)
            if new_smiles:
                result = validator.validate(new_smiles)
                if result.get('valid', False):
                    if new_smiles not in molecules:
                        molecules.append(new_smiles)
        except Exception:
            pass
        attempts += 1
    
    elapsed = time.time() - start_time
    
    # Format results
    summary = f"""
## Generation Results

- **Seed SMILES:** `{seed_smiles[:50]}{'...' if len(seed_smiles) > 50 else ''}`
- **Requested:** {num_molecules}
- **Generated:** {len(molecules)}
- **Attempts:** {attempts}
- **Success Rate:** {100 * len(molecules) / max(attempts, 1):.2f}%
- **Time:** {elapsed:.2f}s
- **Rate:** {len(molecules) / max(elapsed, 0.001):.1f} mol/s
"""
    
    # Molecule list
    mol_list = "\n".join([f"{i+1}. `{smi}`" for i, smi in enumerate(molecules[:50])])
    if len(molecules) > 50:
        mol_list += f"\n\n... and {len(molecules) - 50} more"
    
    return summary, mol_list


def create_generate_tab():
    """Create the mutation generation tab."""
    with gr.Tab("Generate (Mutation)"):
        gr.Markdown("""
        ## SELFIES Mutation Generator
        Generate novel CRAM-like molecules by mutating a seed structure.
        Uses SELFIES for chemically valid mutations.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                seed_input = gr.Textbox(
                    label="Seed SMILES",
                    placeholder="Enter seed molecule SMILES",
                    value="CC(=O)O"  # Acetic acid
                )
                
                num_molecules = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=10,
                    step=1,
                    label="Number of Molecules"
                )
                
                mutation_rate = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.3,
                    step=0.1,
                    label="Mutation Rate"
                )
                
                max_attempts = gr.Slider(
                    minimum=100,
                    maximum=10000,
                    value=1000,
                    step=100,
                    label="Max Attempts"
                )
                
                generate_btn = gr.Button("Generate", variant="primary")
                
                # Example seeds
                gr.Markdown("### Example Seeds")
                examples = [
                    ("Acetic Acid", "CC(=O)O"),
                    ("Benzoic Acid", "c1ccccc1C(=O)O"),
                    ("Succinic Acid", "OC(=O)CCC(=O)O"),
                    ("Citric Acid", "OC(=O)CC(O)(CC(=O)O)C(=O)O"),
                ]
                
                for name, smi in examples:
                    btn = gr.Button(f"{name}", size="sm")
                    btn.click(fn=lambda s=smi: s, outputs=[seed_input])
            
            with gr.Column(scale=2):
                gen_summary = gr.Markdown(label="Summary")
                gen_molecules = gr.Markdown(label="Generated Molecules")
        
        # Event handler
        generate_btn.click(
            fn=generate_mutations,
            inputs=[seed_input, num_molecules, mutation_rate, max_attempts],
            outputs=[gen_summary, gen_molecules]
        )
    
    return seed_input, generate_btn
